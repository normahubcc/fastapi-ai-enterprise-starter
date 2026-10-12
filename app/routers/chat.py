"""Router for Chat Completions, SSE Streaming, Function Calling, and Structured Outputs."""

import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.core.gateway import CircuitBreakerOpenException, NormaHubGateway, get_gateway
from app.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
)
from app.services.semantic_cache import SemanticCacheService, get_cache_service

logger = logging.getLogger("normahub.chat")
router = APIRouter(prefix="/v1/chat", tags=["Chat & Streaming"])


def _extract_last_user_query(messages: list[ChatMessage]) -> str:
    """Extract text content from the last user message in history."""
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return ""


@router.post(
    "/completions",
    response_model=ChatCompletionResponse,
    summary="OpenAI-compatible Chat Completion with Semantic Caching",
)
async def chat_completions(
    request: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
    gateway: NormaHubGateway = Depends(get_gateway),
    cache: SemanticCacheService = Depends(get_cache_service),
) -> ChatCompletionResponse:
    """Generate completion with semantic cache lookup and automatic resilience failover."""
    model = request.model or settings.default_model
    user_query = _extract_last_user_query(request.messages)
    query_embedding: list[float] | None = None

    # 1. Semantic Cache check (only for standard conversational queries without custom tools)
    if request.use_cache and user_query and not request.tools:
        try:
            query_embedding = await gateway.create_embedding(user_query)
            cached_resp, sim_score = await cache.get(
                query=user_query,
                query_embedding=query_embedding,
                model=model,
            )
            if cached_resp is not None:
                logger.info(
                    "Semantic Cache HIT (score=%.4f) for query: '%s'",
                    sim_score,
                    user_query[:50],
                )
                choices = [
                    ChatCompletionChoice(
                        index=c.get("index", 0),
                        message=ChatMessage(**c.get("message", {})),
                        finish_reason=c.get("finish_reason", "stop"),
                    )
                    for c in cached_resp.get("choices", [])
                ]
                usage_data = cached_resp.get("usage")
                usage = ChatUsage(**usage_data) if usage_data else None

                return ChatCompletionResponse(
                    id=cached_resp.get("id", f"chatcmpl-cache-{int(time.time()*1000)}"),
                    model=model,
                    choices=choices,
                    usage=usage,
                    cached=True,
                    ttft_ms=5.0,
                    latency_ms=12.0,
                )
        except Exception as e:
            logger.warning("Error during cache lookup: %s", e)

    # 2. Forward request to NormaHub Gateway
    try:
        messages_dicts = [m.model_dump(exclude_none=True) for m in request.messages]
        completion_data, ttft_ms, latency_ms = await gateway.create_chat_completion(
            messages=messages_dicts,
            model=model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 1024,
            tools=request.tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
        )

        choices = [
            ChatCompletionChoice(
                index=c.get("index", 0),
                message=ChatMessage(**c.get("message", {})),
                finish_reason=c.get("finish_reason", "stop"),
            )
            for c in completion_data.get("choices", [])
        ]
        usage_data = completion_data.get("usage")
        usage = ChatUsage(**usage_data) if usage_data else None

        # 3. Store result in Semantic Cache if eligible
        if request.use_cache and user_query and query_embedding and not request.tools:
            try:
                await cache.set(
                    query=user_query,
                    query_embedding=query_embedding,
                    response=completion_data,
                    model=model,
                )
            except Exception as e:
                logger.warning("Failed to store query in cache: %s", e)

        return ChatCompletionResponse(
            id=completion_data.get("id", f"chatcmpl-{int(time.time()*1000)}"),
            model=model,
            choices=choices,
            usage=usage,
            cached=False,
            ttft_ms=ttft_ms,
            latency_ms=latency_ms,
        )

    except CircuitBreakerOpenException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Failed to complete chat request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"NormaHub Upstream Gateway Error: {e}",
        ) from e


@router.post(
    "/stream",
    summary="Server-Sent Events (SSE) Stream with TTFT telemetry",
    response_class=StreamingResponse,
)
async def chat_stream(
    request: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
    gateway: NormaHubGateway = Depends(get_gateway),
) -> StreamingResponse:
    """Stream response tokens sequentially over SSE protocol with TTFT calculation."""
    model = request.model or settings.default_model
    messages_dicts = [m.model_dump(exclude_none=True) for m in request.messages]

    async def event_generator() -> AsyncIterator[str]:
        try:
            stream_iter = gateway.stream_chat_completion(
                messages=messages_dicts,
                model=model,
                temperature=request.temperature or 0.7,
                max_tokens=request.max_tokens or 1024,
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
            async for chunk in stream_iter:
                yield f"data: {json.dumps(chunk)}\n\n"

            yield "data: [DONE]\n\n"
        except CircuitBreakerOpenException as e:
            error_payload = {
                "error": {
                    "message": str(e),
                    "type": "circuit_breaker_open",
                    "code": 503,
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
        except Exception as e:
            logger.error("SSE stream error: %s", e)
            error_payload = {
                "error": {
                    "message": str(e),
                    "type": "gateway_stream_error",
                    "code": 502,
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
