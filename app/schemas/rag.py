"""Pydantic schemas for Hybrid RAG indexing and querying."""

from typing import Any

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    """Raw document for indexing into RAG knowledge base."""

    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full text document content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata attributes")


class DocumentIndexRequest(BaseModel):
    """Request payload for indexing batch of documents."""

    documents: list[DocumentInput] = Field(
        ..., min_length=1, description="List of documents to index"
    )


class DocumentIndexResponse(BaseModel):
    """Result of batch document indexing."""

    indexed_count: int
    chunk_count: int
    document_ids: list[str]


class RAGQueryRequest(BaseModel):
    """Query request for hybrid search and context synthesis."""

    query: str = Field(..., min_length=1, description="User question or search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Top relevant chunks to retrieve")
    model: str | None = Field(default=None, description="Model to synthesize answer")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    use_cache: bool = Field(default=True, description="Check semantic cache before synthesis")


class Citation(BaseModel):
    """Source reference chunk with score attribution."""

    document_id: str
    title: str
    score: float
    snippet: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalMetrics(BaseModel):
    """Diagnostic details of hybrid BM25 + dense retrieval fusion."""

    bm25_matches: int
    vector_matches: int
    fusion_method: str = "reciprocal_rank_fusion"


class RAGQueryResponse(BaseModel):
    """Synthesized response with citations and latency metrics."""

    query: str
    answer: str
    citations: list[Citation]
    retrieval_metrics: RetrievalMetrics
    cached: bool = False
    ttft_ms: float | None = None
    total_latency_ms: float
