"""Router for Hybrid RAG document indexing and intelligent query synthesis."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.gateway import CircuitBreakerOpenException
from app.schemas.rag import (
    DocumentIndexRequest,
    DocumentIndexResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from app.services.rag_engine import HybridRAGEngine, get_rag_engine
from app.services.semantic_cache import SemanticCacheService, get_cache_service

logger = logging.getLogger("normahub.rag_router")
router = APIRouter(prefix="/v1/rag", tags=["Hybrid RAG Engine"])


@router.post(
    "/index",
    response_model=DocumentIndexResponse,
    summary="Index batch of documents with BM25 and Dense Embeddings",
)
async def index_documents(
    request: DocumentIndexRequest,
    rag_engine: HybridRAGEngine = Depends(get_rag_engine),
) -> DocumentIndexResponse:
    """Chunk and index documents into both inverted BM25 and dense vector memory."""
    try:
        indexed_count, chunk_count, doc_ids = await rag_engine.index_documents(request.documents)
        return DocumentIndexResponse(
            indexed_count=indexed_count,
            chunk_count=chunk_count,
            document_ids=doc_ids,
        )
    except Exception as e:
        logger.error("Failed to index documents: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document indexing failed: {e}",
        ) from e


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Query Knowledge Base with BM25 + Vector Fusion and LLM Synthesis",
)
async def query_rag(
    request: RAGQueryRequest,
    rag_engine: HybridRAGEngine = Depends(get_rag_engine),
    cache: SemanticCacheService = Depends(get_cache_service),
) -> RAGQueryResponse:
    """Execute hybrid search over knowledge base and generate grounded answer with citations."""
    try:
        (
            answer,
            citations,
            metrics,
            ttft_ms,
            latency_ms,
        ) = await rag_engine.query_and_synthesize(
            query=request.query,
            top_k=request.top_k,
            model=request.model,
            temperature=request.temperature,
        )

        return RAGQueryResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            retrieval_metrics=metrics,
            cached=False,
            ttft_ms=ttft_ms,
            total_latency_ms=latency_ms,
        )
    except CircuitBreakerOpenException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Error executing RAG query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"RAG Synthesis Error: {e}",
        ) from e


@router.get(
    "/stats",
    summary="Get RAG knowledge base statistics",
)
async def get_rag_stats(
    rag_engine: HybridRAGEngine = Depends(get_rag_engine),
) -> dict[str, Any]:
    """Return total indexed documents, chunks, and BM25 corpus stats."""
    return {
        "total_chunks": len(rag_engine._chunks),
        "bm25_corpus_size": rag_engine.bm25_index.corpus_size,
        "bm25_avg_doc_len": round(rag_engine.bm25_index.avg_doc_len, 2),
    }
