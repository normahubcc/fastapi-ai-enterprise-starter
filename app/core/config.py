"""Application configuration using Pydantic Settings v2."""

from functools import lru_cache
from typing import Annotated

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_origins(v: str | list[str]) -> list[str]:
    if isinstance(v, str):
        if v.startswith("[") and v.endswith("]"):
            import json

            try:
                return json.loads(v)
            except Exception:
                pass
        return [item.strip() for item in v.split(",") if item.strip()]
    return v


class Settings(BaseSettings):
    """Enterprise application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application Metadata
    app_name: str = Field(
        default="NormaHub FastAPI Enterprise AI Gateway",
        description="Application title displayed in OpenAPI and logs",
    )
    app_version: str = Field(default="1.0.0", description="SemVer version")
    app_env: str = Field(default="production", description="Environment: production, staging, dev")
    debug: bool = Field(default=False, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, description="Bind port")
    workers: int = Field(default=4, description="Worker process count")

    # NormaHub Gateway API Settings
    normahub_api_key: str = Field(
        default="norma-mock-key",
        description="NormaHub API Key obtained from https://normahub.cc",
    )
    normahub_base_url: str = Field(
        default="https://api.normahub.cc/v1",
        description="OpenAI-compatible base URL for NormaHub",
    )
    default_model: str = Field(
        default="gpt-6-astra",
        description="Default production LLM model",
    )
    fallback_model: str = Field(
        default="claude-sonnet-5-5",
        description="Fallback model when circuit breaker is tripped",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model for RAG and semantic cache",
    )

    # Resilience & Circuit Breaker Settings
    gateway_timeout_seconds: float = Field(
        default=60.0,
        description="Gateway client timeout in seconds",
    )
    gateway_max_retries: int = Field(
        default=3,
        description="Max retry attempts for transient errors",
    )
    gateway_backoff_factor: float = Field(
        default=1.5,
        description="Exponential backoff factor",
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        description="Number of consecutive failures to open circuit",
    )
    circuit_breaker_recovery_seconds: float = Field(
        default=30.0,
        description="Recovery duration before half-open probe in seconds",
    )

    # Semantic Cache & Redis Settings
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for vector cache",
    )
    cache_similarity_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for semantic cache hit",
    )
    cache_ttl_seconds: int = Field(
        default=86400,
        description="Time-to-live for cached queries in seconds",
    )
    memory_cache_max_size: int = Field(
        default=5000,
        description="Maximum entries for in-memory LRU vector cache",
    )

    # Hybrid RAG Engine Settings
    rag_vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for dense vector similarity in hybrid search",
    )
    rag_bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 lexical score in hybrid search",
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        description="Number of top retrieved chunks passed to context",
    )
    rag_chunk_size: int = Field(
        default=800,
        ge=100,
        description="Document chunk size in characters",
    )
    rag_chunk_overlap: int = Field(
        default=150,
        ge=0,
        description="Overlap between consecutive chunks in characters",
    )
    rag_storage_path: str = Field(
        default="data/rag_index.db",
        description="Path to persistent SQLite vector and metadata store",
    )

    # CORS
    allowed_origins: Annotated[list[str], BeforeValidator(_parse_origins)] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origin domains",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
