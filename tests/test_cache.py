"""Tests for Semantic Vector Cache and Cosine Similarity."""

import asyncio

import numpy as np
import pytest

from app.core.config import Settings
from app.services.semantic_cache import (
    InMemoryVectorCache,
    SemanticCacheService,
    cosine_similarity,
)


def test_cosine_similarity_math() -> None:
    """Test geometric accuracy of cosine similarity calculation."""
    # Identical vectors -> 1.0
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert np.isclose(cosine_similarity(v1, v2), 1.0)

    # Orthogonal vectors -> 0.0
    v3 = [0.0, 1.0, 0.0]
    assert np.isclose(cosine_similarity(v1, v3), 0.0)

    # Opposite vectors -> -1.0
    v4 = [-1.0, 0.0, 0.0]
    assert np.isclose(cosine_similarity(v1, v4), -1.0)

    # Scaled vectors maintain similarity of 1.0
    v5 = [3.0, 0.0, 0.0]
    assert np.isclose(cosine_similarity(v1, v5), 1.0)


@pytest.mark.asyncio
async def test_in_memory_vector_cache_hit_and_miss() -> None:
    """Verify semantic cache hit above threshold and miss below threshold."""
    cache = InMemoryVectorCache(max_size=10)

    vec_base = [1.0, 0.0, 0.0]
    vec_near = [0.99, 0.1, 0.0]  # Very high similarity (> 0.99)
    vec_far = [0.0, 1.0, 0.0]   # Orthogonal (0.0 similarity)

    sample_resp = {
        "id": "chatcmpl-cached-test",
        "choices": [{"message": {"role": "assistant", "content": "FastAPI is great!"}}],
        "usage": {"total_tokens": 120},
    }

    # Store base query
    await cache.put(
        key="key1",
        query="What is FastAPI?",
        embedding=vec_base,
        response=sample_resp,
        model="gpt-6-astra",
        ttl_seconds=3600,
    )

    # Query with near vector (threshold 0.90) -> HIT
    hit_resp, score = await cache.get_similar(
        query_embedding=vec_near,
        model="gpt-6-astra",
        threshold=0.90,
    )
    assert hit_resp is not None
    assert hit_resp["choices"][0]["message"]["content"] == "FastAPI is great!"
    assert score > 0.90

    # Query with far vector -> MISS
    miss_resp, miss_score = await cache.get_similar(
        query_embedding=vec_far,
        model="gpt-6-astra",
        threshold=0.90,
    )
    assert miss_resp is None
    assert miss_score < 0.90


@pytest.mark.asyncio
async def test_in_memory_cache_model_isolation() -> None:
    """Ensure cache entries are isolated per model."""
    cache = InMemoryVectorCache(max_size=10)
    vec = [1.0, 0.0]
    sample_resp = {"content": "OK"}

    await cache.put(
        key="k1",
        query="test",
        embedding=vec,
        response=sample_resp,
        model="gpt-6-astra",
        ttl_seconds=3600,
    )

    # Searching with a different model must return None
    resp, _ = await cache.get_similar(
        query_embedding=vec,
        model="claude-sonnet-5-5",
        threshold=0.90,
    )
    assert resp is None


@pytest.mark.asyncio
async def test_cache_ttl_expiration() -> None:
    """Verify expired cache items are evicted upon access."""
    cache = InMemoryVectorCache(max_size=10)
    vec = [1.0, 0.0]

    await cache.put(
        key="k1",
        query="expiring query",
        embedding=vec,
        response={"val": 1},
        model="gpt-6-astra",
        ttl_seconds=1,  # 1 second TTL
    )

    assert await cache.size() == 1

    # Wait for expiry
    await asyncio.sleep(1.05)

    hit_resp, _ = await cache.get_similar(vec, "gpt-6-astra", threshold=0.90)
    assert hit_resp is None
    assert await cache.size() == 0


@pytest.mark.asyncio
async def test_semantic_cache_service_stats() -> None:
    """Verify token savings and hit ratio reporting."""
    settings = Settings(cache_similarity_threshold=0.85, redis_url="")
    service = SemanticCacheService(settings=settings)

    vec = [1.0, 0.0]
    resp = {"id": "test", "usage": {"total_tokens": 250}}

    # Initial stats
    s0 = service.get_stats()
    assert s0["hit_count"] == 0
    assert s0["miss_count"] == 0
    assert s0["saved_tokens_approx"] == 0

    # Miss
    await service.get("query1", vec, "gpt-6-astra")
    assert service.get_stats()["miss_count"] == 1

    # Store
    await service.set("query1", vec, resp, "gpt-6-astra")

    # Hit
    cached, _ = await service.get("query1", vec, "gpt-6-astra")
    assert cached is not None
    s1 = service.get_stats()
    assert s1["hit_count"] == 1
    assert s1["saved_tokens_approx"] == 250
    assert s1["hit_ratio"] == 0.5  # 1 hit, 1 miss -> 50%
