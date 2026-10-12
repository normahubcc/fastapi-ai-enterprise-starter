"""Enterprise AI Gateway with Circuit Breaker, Exponential Retry, and TTFT tracking."""

import asyncio
import enum
import logging
import random
import time
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.core.config import Settings, get_settings

logger = logging.getLogger("normahub.gateway")


class CircuitState(enum.StrEnum):
    """Circuit Breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is OPEN and fast-fails requests."""

    def __init__(self, message: str = "Circuit breaker is OPEN. NormaHub service is degraded."):
        super().__init__(message)


class LatencyTracker:
    """Sliding-window latency and TTFT metrics collector for rolling percentiles."""

    def __init__(self, max_samples: int = 2000):
        self._samples: deque[float] = deque(maxlen=max_samples)
        self._lock = asyncio.Lock()

    async def record_ttft(self, ttft_ms: float) -> None:
        async with self._lock:
            self._samples.append(ttft_ms)

    def get_stats(self) -> dict[str, float | int]:
        if not self._samples:
            return {
                "ttft_p50_ms": 0.0,
                "ttft_p95_ms": 0.0,
                "ttft_p99_ms": 0.0,
                "avg_ttft_ms": 0.0,
                "sample_count": 0,
            }
        sorted_samples = sorted(self._samples)
        n = len(sorted_samples)

        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            d0 = sorted_samples[f] * (c - k)
            d1 = sorted_samples[c] * (k - f)
            return round(d0 + d1, 2)

        return {
            "ttft_p50_ms": percentile(0.50),
            "ttft_p95_ms": percentile(0.95),
            "ttft_p99_ms": percentile(0.99),
            "avg_ttft_ms": round(sum(sorted_samples) / n, 2),
            "sample_count": n,
        }


class CircuitBreaker:
    """Stateful circuit breaker protecting against upstream API cascading failures."""

    def __init__(self, failure_threshold: int = 5, recovery_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        async with self._lock:
            now = time.time()
            if self.state == CircuitState.OPEN:
                if now - self.last_failure_time >= self.recovery_seconds:
                    logger.info("Circuit breaker transitioning from OPEN to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            return True

    async def record_success(self) -> None:
        async with self._lock:
            if self.state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                logger.info("Circuit breaker recovered! Transitioning to CLOSED.")
            self.state = CircuitState.CLOSED
            self.failure_count = 0

    async def record_failure(self) -> None:
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker tripped to OPEN! Consecutive failures: %d", self.failure_count
                )


class NormaHubGateway:
    """Resilient async gateway for NormaHub / OpenAI-compatible API."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.settings.circuit_breaker_failure_threshold,
            recovery_seconds=self.settings.circuit_breaker_recovery_seconds,
        )
        self.latency_tracker = LatencyTracker()
        self._client: AsyncOpenAI | None = None

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.normahub_api_key,
                base_url=self.settings.normahub_base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.settings.gateway_timeout_seconds,
                    write=15.0,
                    pool=10.0,
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def ping(self) -> dict[str, Any]:
        """Check gateway health and measure upstream ping latency."""
        start = time.perf_counter()
        client = self.get_client()
        try:
            # Lightweight models list check
            await client.models.list()
            latency = (time.perf_counter() - start) * 1000.0
            return {
                "status": "connected",
                "base_url": self.settings.normahub_base_url,
                "ping_ms": round(latency, 2),
            }
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000.0
            return {
                "status": "unreachable",
                "base_url": self.settings.normahub_base_url,
                "ping_ms": round(latency, 2),
                "error": str(e),
            }

    async def create_embedding(self, text: str, model: str | None = None) -> list[float]:
        """Generate normalized vector embedding with retry."""
        target_model = model or self.settings.embedding_model
        client = self.get_client()

        async def _call() -> list[float]:
            response = await client.embeddings.create(
                model=target_model,
                input=text,
            )
            return response.data[0].embedding

        return await self._execute_with_retry(_call)

    async def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], float, float]:
        """Execute chat completion with circuit breaker, retry, and fallback model."""
        target_model = model or self.settings.default_model

        async def _invoke(m: str) -> dict[str, Any]:
            client = self.get_client()
            params: dict[str, Any] = {
                "model": m,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }
            if tools:
                params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice
            if response_format:
                params["response_format"] = response_format

            resp = await client.chat.completions.create(**params)
            return resp.model_dump()

        start = time.perf_counter()
        try:
            result = await self._execute_with_retry(lambda: _invoke(target_model))
            total_latency = (time.perf_counter() - start) * 1000.0
            ttft = total_latency * 0.4  # estimated TTFT for non-streaming
            await self.latency_tracker.record_ttft(ttft)
            return result, round(ttft, 2), round(total_latency, 2)
        except Exception as primary_error:
            # If primary model fails or circuit is open, attempt fallback model
            if self.settings.fallback_model and self.settings.fallback_model != target_model:
                logger.warning(
                    "Primary model %s failed (%s). Attempting fallback model %s.",
                    target_model,
                    primary_error,
                    self.settings.fallback_model,
                )
                try:
                    result = await self._execute_with_retry(
                        lambda: _invoke(self.settings.fallback_model)
                    )
                    total_latency = (time.perf_counter() - start) * 1000.0
                    ttft = total_latency * 0.4
                    await self.latency_tracker.record_ttft(ttft)
                    return result, round(ttft, 2), round(total_latency, 2)
                except Exception as fallback_error:
                    logger.error(
                        "Fallback model %s also failed: %s",
                        self.settings.fallback_model,
                        fallback_error,
                    )
            raise primary_error

    async def stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream SSE completion with precise Time-To-First-Token (TTFT) tracking."""
        if not await self.circuit_breaker.can_execute():
            raise CircuitBreakerOpenException()

        target_model = model or self.settings.default_model
        client = self.get_client()
        params: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        start_time = time.perf_counter()
        first_token_received = False
        ttft_recorded: float | None = None

        try:
            response_stream = await client.chat.completions.create(**params)
            async for chunk in response_stream:
                now = time.perf_counter()
                chunk_dict = chunk.model_dump()
                choices = chunk_dict.get("choices", [])
                has_content = False
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content") or delta.get("tool_calls"):
                        has_content = True

                if not first_token_received and has_content:
                    first_token_received = True
                    ttft_ms = (now - start_time) * 1000.0
                    ttft_recorded = round(ttft_ms, 2)
                    await self.latency_tracker.record_ttft(ttft_recorded)
                    chunk_dict["ttft_ms"] = ttft_recorded

                yield chunk_dict

            await self.circuit_breaker.record_success()
        except Exception as e:
            await self.circuit_breaker.record_failure()
            logger.error("Error during streaming generation: %s", e)
            raise

    async def _execute_with_retry(self, coro_func: Any) -> Any:
        """Execute async coroutine with circuit breaker check and exponential backoff retry."""
        if not await self.circuit_breaker.can_execute():
            raise CircuitBreakerOpenException()

        retries = self.settings.gateway_max_retries
        backoff = self.settings.gateway_backoff_factor

        for attempt in range(retries + 1):
            try:
                res = await coro_func()
                await self.circuit_breaker.record_success()
                return res
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                logger.warning(
                    "Transient upstream error on attempt %d/%d: %s",
                    attempt + 1,
                    retries + 1,
                    e,
                )
                if attempt == retries:
                    await self.circuit_breaker.record_failure()
                    raise
                # Exponential backoff with jitter
                sleep_duration = (backoff**attempt) + random.uniform(0.1, 0.5)
                await asyncio.sleep(sleep_duration)
            except APIError as e:
                # 5xx server errors from upstream are retryable
                if hasattr(e, "status_code") and e.status_code and e.status_code >= 500:
                    logger.warning(
                        "5xx upstream error on attempt %d/%d: %s",
                        attempt + 1,
                        retries + 1,
                        e,
                    )
                    if attempt == retries:
                        await self.circuit_breaker.record_failure()
                        raise
                    sleep_duration = (backoff**attempt) + random.uniform(0.1, 0.5)
                    await asyncio.sleep(sleep_duration)
                else:
                    await self.circuit_breaker.record_failure()
                    raise
            except Exception:
                await self.circuit_breaker.record_failure()
                raise


_gateway_singleton: NormaHubGateway | None = None


def get_gateway() -> NormaHubGateway:
    """Return shared gateway singleton."""
    global _gateway_singleton
    if _gateway_singleton is None:
        _gateway_singleton = NormaHubGateway()
    return _gateway_singleton
