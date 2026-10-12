"""Health inspection and observability endpoints."""

import time
from typing import Any

from fastapi import APIRouter, Depends, Response

from app.core.config import Settings, get_settings
from app.core.gateway import NormaHubGateway, get_gateway
from app.services.semantic_cache import SemanticCacheService, get_cache_service

router = APIRouter(tags=["Health & Observability"])
_service_start_time = time.time()


@router.get(
    "/health",
    summary="Enterprise health check with gateway ping and TTFT percentiles",
)
@router.get(
    "/v1/health",
    summary="Enterprise health check (v1 prefix)",
)
async def health_check(
    settings: Settings = Depends(get_settings),
    gateway: NormaHubGateway = Depends(get_gateway),
    cache: SemanticCacheService = Depends(get_cache_service),
) -> dict[str, Any]:
    """Perform diagnostics across upstream NormaHub gateway, cache backend, and latency stats."""
    uptime = time.time() - _service_start_time
    gateway_ping = await gateway.ping()
    cache_ping = await cache.ping()
    cache_stats = cache.get_stats()
    ttft_stats = gateway.latency_tracker.get_stats()

    is_healthy = (
        gateway.circuit_breaker.state.value != "OPEN"
        and gateway_ping.get("status") != "unreachable"
    )

    return {
        "status": "healthy" if is_healthy else "degraded",
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.app_env,
        "uptime_seconds": round(uptime, 1),
        "normahub_gateway": gateway_ping,
        "cache": {
            **cache_ping,
            **cache_stats,
        },
        "circuit_breaker": {
            "state": gateway.circuit_breaker.state.value,
            "failure_count": gateway.circuit_breaker.failure_count,
            "failure_threshold": gateway.circuit_breaker.failure_threshold,
        },
        "latency_telemetry": ttft_stats,
    }


@router.get(
    "/metrics",
    summary="Prometheus-compatible metrics exposition",
)
async def prometheus_metrics(
    gateway: NormaHubGateway = Depends(get_gateway),
    cache: SemanticCacheService = Depends(get_cache_service),
) -> Response:
    """Export standard Prometheus metrics for Grafana / Datadog agents."""
    cache_stats = cache.get_stats()
    ttft_stats = gateway.latency_tracker.get_stats()

    lines = [
        "# HELP normahub_cache_hits_total Total semantic cache hits",
        "# TYPE normahub_cache_hits_total counter",
        f"normahub_cache_hits_total {cache_stats['hit_count']}",
        "# HELP normahub_cache_misses_total Total semantic cache misses",
        "# TYPE normahub_cache_misses_total counter",
        f"normahub_cache_misses_total {cache_stats['miss_count']}",
        "# HELP normahub_cache_saved_tokens_total Estimated LLM tokens saved via cache",
        "# TYPE normahub_cache_saved_tokens_total counter",
        f"normahub_cache_saved_tokens_total {cache_stats['saved_tokens_approx']}",
        "# HELP normahub_ttft_p50_ms TTFT 50th percentile in milliseconds",
        "# TYPE normahub_ttft_p50_ms gauge",
        f"normahub_ttft_p50_ms {ttft_stats['ttft_p50_ms']}",
        "# HELP normahub_ttft_p95_ms TTFT 95th percentile in milliseconds",
        "# TYPE normahub_ttft_p95_ms gauge",
        f"normahub_ttft_p95_ms {ttft_stats['ttft_p95_ms']}",
        "# HELP normahub_ttft_p99_ms TTFT 99th percentile in milliseconds",
        "# TYPE normahub_ttft_p99_ms gauge",
        f"normahub_ttft_p99_ms {ttft_stats['ttft_p99_ms']}",
        "# HELP normahub_circuit_breaker_state 0: CLOSED, 1: HALF_OPEN, 2: OPEN",
        "# TYPE normahub_circuit_breaker_state gauge",
        f"normahub_circuit_breaker_state {0 if gateway.circuit_breaker.state.value == 'CLOSED' else (1 if gateway.circuit_breaker.state.value == 'HALF_OPEN' else 2)}",
    ]

    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
