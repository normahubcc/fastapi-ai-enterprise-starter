"""Tests for NormaHub Gateway, Circuit Breaker, Retries, and TTFT Tracking."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from openai import RateLimitError

from app.core.config import Settings
from app.core.gateway import (
    CircuitBreaker,
    CircuitBreakerOpenException,
    CircuitState,
    LatencyTracker,
    NormaHubGateway,
)


@pytest.mark.asyncio
async def test_circuit_breaker_transitions() -> None:
    """Verify state transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""
    cb = CircuitBreaker(failure_threshold=3, recovery_seconds=0.1)
    assert cb.state == CircuitState.CLOSED
    assert await cb.can_execute() is True

    # Record 2 failures (under threshold)
    await cb.record_failure()
    await cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    assert await cb.can_execute() is True

    # 3rd failure trips the breaker to OPEN
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert await cb.can_execute() is False

    # Wait for recovery cooldown
    await asyncio.sleep(0.15)

    # Breaker should allow probe request and enter HALF_OPEN
    assert await cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # Success closes the circuit
    await cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_fast_fail_exception() -> None:
    """Verify that an OPEN circuit raises CircuitBreakerOpenException."""
    cb = CircuitBreaker(failure_threshold=1, recovery_seconds=10.0)
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN

    settings = Settings(circuit_breaker_failure_threshold=1, circuit_breaker_recovery_seconds=10.0)
    gw = NormaHubGateway(settings=settings)
    gw.circuit_breaker = cb

    with pytest.raises(CircuitBreakerOpenException):
        await gw._execute_with_retry(AsyncMock())


@pytest.mark.asyncio
async def test_latency_tracker_percentiles() -> None:
    """Verify TTFT percentile calculations."""
    tracker = LatencyTracker(max_samples=100)

    # Empty stats
    empty_stats = tracker.get_stats()
    assert empty_stats["sample_count"] == 0
    assert empty_stats["ttft_p50_ms"] == 0.0

    # Feed 100 samples from 1ms to 100ms
    for ms in range(1, 101):
        await tracker.record_ttft(float(ms))

    stats = tracker.get_stats()
    assert stats["sample_count"] == 100
    assert 49.0 <= stats["ttft_p50_ms"] <= 51.0
    assert 94.0 <= stats["ttft_p95_ms"] <= 96.0
    assert 98.0 <= stats["ttft_p99_ms"] <= 100.0


@pytest.mark.asyncio
async def test_gateway_retry_on_transient_error() -> None:
    """Verify retry handler recovers after transient RateLimitError."""
    settings = Settings(
        gateway_max_retries=2,
        gateway_backoff_factor=0.01,  # ultra-fast for testing
    )
    gw = NormaHubGateway(settings=settings)

    mock_func = AsyncMock()
    # Fail first time with RateLimitError, succeed on second attempt
    mock_func.side_effect = [
        RateLimitError("Rate limit exceeded", response=AsyncMock(status_code=429), body=None),
        {"result": "success"},
    ]

    result = await gw._execute_with_retry(mock_func)
    assert result == {"result": "success"}
    assert mock_func.call_count == 2
    assert gw.circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_gateway_fallback_model_routing() -> None:
    """Verify gateway falls back to secondary model when primary fails."""
    settings = Settings(
        default_model="gpt-6-astra",
        fallback_model="claude-sonnet-5-5",
        gateway_max_retries=0,
    )
    gw = NormaHubGateway(settings=settings)

    call_models: list[str] = []

    async def mock_invoke(m: str) -> dict[str, object]:
        call_models.append(m)
        if m == "gpt-6-astra":
            raise RuntimeError("Upstream GPT-6 failure")
        return {
            "id": "chatcmpl-fallback-test",
            "choices": [{"message": {"role": "assistant", "content": "Fallback response"}}],
        }

    async def mock_retry(fn: object) -> object:
        return await fn()

    with patch.object(gw, "_execute_with_retry", new=mock_retry):
        with patch.object(gw, "get_client") as mock_client:
            mock_completions = AsyncMock()

            async def side_effect(**kwargs: object) -> object:
                from unittest.mock import MagicMock

                model = kwargs.get("model")
                if model == "gpt-6-astra":
                    raise RuntimeError("Primary model failure")
                res = MagicMock()
                res.model_dump.return_value = {
                    "id": "fallback-id",
                    "model": "claude-sonnet-5-5",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                }
                return res

            mock_completions.create = AsyncMock(side_effect=side_effect)
            mock_client.return_value.chat.completions = mock_completions

            result, ttft, latency = await gw.create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-6-astra",
            )

            assert result["model"] == "claude-sonnet-5-5"
            assert ttft > 0
            assert latency > 0
