"""Tests for Chat completions, streaming, and Health check endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from tests.conftest import generate_deterministic_embedding


@pytest.mark.asyncio
async def test_chat_completions_endpoint(async_client: AsyncClient) -> None:
    """Verify /v1/chat/completions with mock gateway and semantic cache."""
    req_payload = {
        "model": "gpt-6-astra",
        "messages": [{"role": "user", "content": "How do I optimize FastAPI?"}],
        "temperature": 0.5,
        "use_cache": True,
    }

    mock_resp = {
        "id": "chatcmpl-test-1",
        "model": "gpt-6-astra",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Use async handlers and connection pools."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }

    with patch(
        "app.core.gateway.NormaHubGateway.create_embedding",
        new=AsyncMock(side_effect=lambda text, **k: generate_deterministic_embedding(text)),
    ):
        with patch(
            "app.core.gateway.NormaHubGateway.create_chat_completion",
            new=AsyncMock(return_value=(mock_resp, 120.0, 310.0)),
        ):
            # First call: cache miss
            resp1 = await async_client.post("/v1/chat/completions", json=req_payload)
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["cached"] is False
            assert "optimize FastAPI" in req_payload["messages"][0]["content"]

            # Second call with identical query: cache HIT
            resp2 = await async_client.post("/v1/chat/completions", json=req_payload)
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["cached"] is True
            assert data2["choices"][0]["message"]["content"] == "Use async handlers and connection pools."


@pytest.mark.asyncio
async def test_chat_stream_endpoint(async_client: AsyncClient) -> None:
    """Verify /v1/chat/stream yields SSE chunks."""
    req_payload = {
        "model": "gpt-6-astra",
        "messages": [{"role": "user", "content": "Stream me a response"}],
    }

    chunks_data = [
        {"id": "chunk-1", "choices": [{"delta": {"content": "Hello"}}]},
        {"id": "chunk-2", "choices": [{"delta": {"content": " World!"}}]},
        {"id": "chunk-3", "choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]

    async def mock_stream(*args: object, **kwargs: object):
        for c in chunks_data:
            yield c

    with patch("app.core.gateway.NormaHubGateway.stream_chat_completion", side_effect=mock_stream):
        resp = await async_client.post("/v1/chat/stream", json=req_payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        text = resp.text
        assert "data: " in text
        assert "[DONE]" in text
        assert "Hello" in text


@pytest.mark.asyncio
async def test_health_and_metrics_endpoints(async_client: AsyncClient) -> None:
    """Verify /health and /metrics endpoints return expected operational data."""
    mock_ping = {
        "status": "connected",
        "base_url": "https://api.normahub.cc/v1",
        "ping_ms": 25.0,
    }
    with patch("app.core.gateway.NormaHubGateway.ping", new=AsyncMock(return_value=mock_ping)):
        # Test /health
        resp_health = await async_client.get("/health")
        assert resp_health.status_code == 200
        data_health = resp_health.json()
        assert "status" in data_health
        assert "circuit_breaker" in data_health
        assert "cache" in data_health
        assert "latency_telemetry" in data_health

        # Test /v1/health
        resp_v1_health = await async_client.get("/v1/health")
        assert resp_v1_health.status_code == 200

    # Test /metrics
    resp_metrics = await async_client.get("/metrics")
    assert resp_metrics.status_code == 200
    metrics_text = resp_metrics.text
    assert "normahub_cache_hits_total" in metrics_text
    assert "normahub_ttft_p50_ms" in metrics_text
    assert "normahub_circuit_breaker_state" in metrics_text
