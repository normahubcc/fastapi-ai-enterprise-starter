"""FastAPI AI Enterprise Starter Application Entrypoint."""

import logging
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.gateway import get_gateway
from app.routers.chat import router as chat_router
from app.routers.health import router as health_router
from app.routers.rag import router as rag_router
from app.services.semantic_cache import get_cache_service


class RequestIdFilter(logging.Filter):
    """Filter that ensures every LogRecord has a request_id attribute."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


# Configure structured enterprise logging
handler = logging.StreamHandler(sys.stdout)
handler.addFilter(RequestIdFilter())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s (req_id=%(request_id)s): %(message)s",
    handlers=[handler],
)
logger = logging.getLogger("normahub.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle management for connection pools and caching warm-up."""
    settings = get_settings()
    logger.info("Initializing %s v%s...", settings.app_name, settings.app_version)

    # Initialize Redis connection for semantic vector cache
    cache = get_cache_service()
    await cache.init_redis()

    yield

    # Graceful shutdown
    logger.info("Shutting down gateway and connection pools...")
    await cache.close()
    gateway = get_gateway()
    await gateway.close()


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "High-Load Enterprise AI Gateway on FastAPI with Hybrid RAG, "
        "Semantic Vector Caching, SSE Streaming, and TTFT Latency Telemetry via NormaHub."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enterprise CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID", "X-TTFT-Ms"],
)


@app.middleware("http")
async def latency_and_tracing_middleware(request: Request, call_next: object) -> Response:
    """Inject unique request tracking ID and compute total processing latency."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    # Pass request to downstream handlers
    response: Response = await call_next(request)  # type: ignore[misc]

    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"

    if not request.url.path.startswith(("/health", "/metrics")):
        logger.info(
            "HTTP %s %s [%d] in %.2fms",
            request.method,
            request.url.path,
            response.status_code,
            process_time_ms,
            extra={"request_id": request_id},
        )

    return response


# Register Routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(rag_router)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint providing gateway status and documentation links."""
    return {
        "status": "online",
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "gateway_provider": "NormaHub API (https://normahub.cc)",
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Capture uncaught exceptions and return uniform JSON error."""
    logger.exception("Unhandled error processing request %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal Gateway Server Error",
                "detail": str(exc) if settings.debug else None,
            }
        },
    )
