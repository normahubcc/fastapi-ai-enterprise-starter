"""Pytest fixtures and test environment setup."""

import hashlib
import math
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from app.core.config import Settings
from app.core.gateway import NormaHubGateway
from app.main import app
from app.services.rag_engine import HybridRAGEngine
from app.services.semantic_cache import SemanticCacheService


def generate_deterministic_embedding(text: str, dim: int = 128) -> list[float]:
    """Generate normalized pseudo-vector based on sha256 hash of text."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    vec: list[float] = []
    for i in range(dim):
        # Deterministic trigonometric signal
        val = math.sin((seed % 10000) * 0.1 + i * 0.3)
        vec.append(val)
    # Normalize vector to unit length
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture providing isolated test settings."""
    return Settings(
        normahub_api_key="test-key-normahub-enterprise",
        normahub_base_url="https://api.normahub.cc/v1",
        default_model="gpt-6-astra",
        fallback_model="claude-sonnet-5-5",
        embedding_model="text-embedding-3-small",
        redis_url="",  # force in-memory cache for deterministic unit tests
        circuit_breaker_failure_threshold=3,
        circuit_breaker_recovery_seconds=0.2,
        cache_similarity_threshold=0.90,
    )


@pytest.fixture
def test_gateway(mock_settings: Settings) -> NormaHubGateway:
    """Gateway instance with mockable OpenAI client."""
    gateway = NormaHubGateway(settings=mock_settings)

    # Patch create_embedding to return deterministic vector
    async def _mock_embedding(text: str, model: str | None = None) -> list[float]:
        return generate_deterministic_embedding(text)

    gateway.create_embedding = _mock_embedding  # type: ignore[assignment]
    return gateway


@pytest.fixture
def test_cache(mock_settings: Settings) -> SemanticCacheService:
    """Isolated Semantic Cache instance using in-memory store."""
    return SemanticCacheService(settings=mock_settings)


@pytest.fixture
def test_rag(test_gateway: NormaHubGateway, mock_settings: Settings) -> HybridRAGEngine:
    """Isolated Hybrid RAG engine instance."""
    return HybridRAGEngine(gateway=test_gateway, settings=mock_settings)


@pytest_asyncio.fixture
async def async_client() -> AsyncIterator[httpx.AsyncClient]:
    """Async HTTP test client bound to FastAPI application."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
