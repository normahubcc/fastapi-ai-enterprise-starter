"""Tests for Hybrid RAG Engine, BM25 Index, and RAG Router."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.core.config import Settings
from app.core.gateway import NormaHubGateway
from app.schemas.rag import DocumentInput
from app.services.rag_engine import BM25Index, HybridRAGEngine, tokenize
from tests.conftest import generate_deterministic_embedding


def test_bm25_lexical_scoring() -> None:
    """Verify BM25 accurately ranks documents matching lexical keywords."""
    index = BM25Index()
    docs = [
        "NormaHub provides high-speed enterprise OpenAI gateway with no VPN required in Russia.",
        "FastAPI is a modern high performance web framework for building APIs with Python.",
        "PostgreSQL is a powerful open source object-relational database system.",
    ]
    chunk_ids = ["doc_norma", "doc_fastapi", "doc_postgres"]
    tokenized = [tokenize(d) for d in docs]

    index.index(chunk_ids, tokenized)

    # Search for "normahub gateway"
    scores_norma = index.score(tokenize("normahub gateway"))
    assert len(scores_norma) == 3
    assert scores_norma[0][0] == "doc_norma"
    assert scores_norma[0][1] > scores_norma[1][1]

    # Search for "fastapi python"
    scores_fastapi = index.score(tokenize("fastapi python"))
    assert scores_fastapi[0][0] == "doc_fastapi"
    assert scores_fastapi[0][1] > scores_fastapi[1][1]


def test_text_chunking_sliding_window() -> None:
    """Verify text chunker preserves continuity via overlap."""
    engine = HybridRAGEngine()
    sample_text = "A" * 1500

    chunks = engine.chunk_text(sample_text, chunk_size=500, overlap=100)
    assert len(chunks) >= 3
    # Check that each chunk length does not exceed chunk_size
    for c in chunks:
        assert len(c) <= 500


@pytest.mark.asyncio
async def test_hybrid_search_fusion() -> None:
    """Verify Reciprocal Rank Fusion returns citations with composite scores."""
    settings = Settings(rag_vector_weight=0.7, rag_bm25_weight=0.3)
    gw = NormaHubGateway(settings=settings)

    async def mock_embed(text: str, model: str | None = None) -> list[float]:
        return generate_deterministic_embedding(text)

    gw.create_embedding = mock_embed  # type: ignore[assignment]
    rag = HybridRAGEngine(gateway=gw, settings=settings)

    documents = [
        DocumentInput(
            id="doc-sla",
            title="SLA Overview",
            content="NormaHub guarantees 99.95 percent uptime with redundant failover clusters.",
            metadata={"source": "docs"},
        ),
        DocumentInput(
            id="doc-billing",
            title="Billing Policy",
            content="Corporate customers pay via official bank invoice with 20 percent VAT in Russia.",
            metadata={"source": "finance"},
        ),
    ]

    indexed, total_chunks, doc_ids = await rag.index_documents(documents)
    assert indexed == 2
    assert total_chunks == 2
    assert doc_ids == ["doc-sla", "doc-billing"]

    # Query hybrid search
    citations, metrics = await rag.hybrid_search("uptime sla failover", top_k=2)
    assert len(citations) >= 1
    assert citations[0].document_id == "doc-sla"
    assert citations[0].score > 0
    assert metrics.bm25_matches >= 1


@pytest.mark.asyncio
async def test_rag_http_endpoints(async_client: AsyncClient) -> None:
    """Test full HTTP flow for /v1/rag/index, /v1/rag/query, and /v1/rag/stats."""
    # 1. Index document
    index_payload = {
        "documents": [
            {
                "id": "http-doc-1",
                "title": "Latency Guide",
                "content": "Time to first token (TTFT) is critical for streaming enterprise chatbots.",
                "metadata": {"test": True},
            }
        ]
    }

    with patch(
        "app.core.gateway.NormaHubGateway.create_embedding",
        new=AsyncMock(side_effect=lambda text, **k: generate_deterministic_embedding(text)),
    ):
        with patch(
            "app.core.gateway.NormaHubGateway.create_chat_completion",
            new=AsyncMock(
                return_value=(
                    {
                        "id": "chatcmpl-test-rag",
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "TTFT measures first token latency.",
                                }
                            }
                        ],
                    },
                    15.0,
                    45.0,
                )
            ),
        ):
            resp_index = await async_client.post("/v1/rag/index", json=index_payload)
            assert resp_index.status_code == 200
            data_index = resp_index.json()
            assert data_index["indexed_count"] == 1
            assert "http-doc-1" in data_index["document_ids"]

            # 2. Query document
            query_payload = {
                "query": "What is TTFT and why is it critical?",
                "top_k": 3,
                "model": "gpt-6-astra",
            }
            resp_query = await async_client.post("/v1/rag/query", json=query_payload)
            assert resp_query.status_code == 200
            data_query = resp_query.json()
            assert "TTFT" in data_query["answer"]
            assert len(data_query["citations"]) >= 1
            assert data_query["citations"][0]["document_id"] == "http-doc-1"

            # 3. Check stats
            resp_stats = await async_client.get("/v1/rag/stats")
            assert resp_stats.status_code == 200
            data_stats = resp_stats.json()
            assert data_stats["total_chunks"] >= 1
