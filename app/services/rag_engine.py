"""Hybrid RAG Engine combining BM25 lexical retrieval and dense vector embeddings."""

import asyncio
import json
import logging
import math
import os
import re
import sqlite3
import time
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.core.gateway import NormaHubGateway, get_gateway
from app.schemas.rag import Citation, DocumentInput, RetrievalMetrics

logger = logging.getLogger("normahub.rag")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Production BM25Okapi inverted index for lexical keyword ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: list[int] = []
        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []
        self.doc_ids: list[str] = []

    def index(self, chunk_ids: list[str], tokenized_chunks: list[list[str]]) -> None:
        self.doc_ids = chunk_ids
        self.corpus_size = len(tokenized_chunks)
        self.doc_lengths = [len(doc) for doc in tokenized_chunks]
        self.avg_doc_len = (
            sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0.0
        )
        self.doc_freqs.clear()
        self.term_freqs = []

        for doc in tokenized_chunks:
            tf: dict[str, int] = {}
            seen: set[str] = set()
            for token in doc:
                tf[token] = tf.get(token, 0) + 1
                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)
            self.term_freqs.append(tf)

    def score(self, query_tokens: list[str]) -> list[tuple[str, float]]:
        """Calculate BM25 scores for all indexed chunks."""
        scores: list[tuple[str, float]] = []
        if self.corpus_size == 0 or not query_tokens:
            return scores

        for chunk_id, doc_len, tf_dict in zip(
            self.doc_ids, self.doc_lengths, self.term_freqs, strict=False
        ):
            score = 0.0
            for token in query_tokens:
                if token not in tf_dict:
                    continue
                # Term Frequency component
                f = tf_dict[token]
                # Inverse Document Frequency component
                n = self.doc_freqs.get(token, 0)
                idf = math.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1.0)
                # BM25 scoring formula
                denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                tf_part = (f * (self.k1 + 1.0)) / denom if denom > 0 else 0.0
                score += idf * tf_part
            scores.append((chunk_id, score))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class ChunkRecord:
    """Indexed chunk containing content, dense vector embedding, and metadata."""

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        title: str,
        chunk_index: int,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.title = title
        self.chunk_index = chunk_index
        self.content = content
        self.embedding = embedding
        self.metadata = metadata


class HybridRAGEngine:
    """Enterprise RAG engine integrating BM25 lexical search and dense embeddings with SQLite persistence."""

    def __init__(
        self,
        gateway: NormaHubGateway | None = None,
        settings: Settings | None = None,
        db_path: str | None = None,
    ):
        self.settings = settings or get_settings()
        self.gateway = gateway or get_gateway()
        self.bm25_index = BM25Index()
        self._chunks: dict[str, ChunkRecord] = {}
        self._lock = asyncio.Lock()
        self.db_path = db_path or getattr(self.settings, "rag_storage_path", "data/rag_index.db")
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite vector store and restore indexed chunks."""
        if self.db_path != ":memory:":
            parent_dir = os.path.dirname(self.db_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            conn.commit()

            cursor = conn.cursor()
            cursor.execute(
                "SELECT chunk_id, doc_id, title, chunk_index, content, embedding, metadata FROM rag_chunks;"
            )
            rows = cursor.fetchall()
            for row in rows:
                c_id, d_id, title, c_idx, content, emb_str, meta_str = row
                record = ChunkRecord(
                    chunk_id=c_id,
                    doc_id=d_id,
                    title=title,
                    chunk_index=c_idx,
                    content=content,
                    embedding=json.loads(emb_str),
                    metadata=json.loads(meta_str),
                )
                self._chunks[c_id] = record

            if self._chunks:
                all_chunk_ids = list(self._chunks.keys())
                tokenized_corpus = [
                    tokenize(self._chunks[cid].content) for cid in all_chunk_ids
                ]
                self.bm25_index.index(all_chunk_ids, tokenized_corpus)
                logger.info(
                    "Restored %d RAG chunks from SQLite storage (%s)",
                    len(self._chunks),
                    self.db_path,
                )

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
        """Split document text using sliding window with character overlap."""
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= len(text):
                break
            start += chunk_size - overlap
        return chunks

    async def index_documents(
        self, documents: list[DocumentInput]
    ) -> tuple[int, int, list[str]]:
        """Chunk, vectorize, and index batch of documents."""
        indexed_doc_ids: list[str] = []
        new_chunks: list[ChunkRecord] = []

        chunk_size = self.settings.rag_chunk_size
        overlap = self.settings.rag_chunk_overlap

        for doc in documents:
            text_chunks = self.chunk_text(doc.content, chunk_size=chunk_size, overlap=overlap)
            for idx, text_chunk in enumerate(text_chunks):
                chunk_id = f"{doc.id}_chk_{idx}"
                embedding = await self.gateway.create_embedding(text_chunk)
                record = ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc.id,
                    title=doc.title,
                    chunk_index=idx,
                    content=text_chunk,
                    embedding=embedding,
                    metadata=doc.metadata,
                )
                new_chunks.append(record)
            indexed_doc_ids.append(doc.id)

        async with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for chk in new_chunks:
                    self._chunks[chk.chunk_id] = chk
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO rag_chunks
                        (chunk_id, doc_id, title, chunk_index, content, embedding, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            chk.chunk_id,
                            chk.doc_id,
                            chk.title,
                            chk.chunk_index,
                            chk.content,
                            json.dumps(chk.embedding),
                            json.dumps(chk.metadata),
                            time.time(),
                        ),
                    )
                conn.commit()

            # Rebuild BM25 index across all current chunks
            all_chunk_ids = list(self._chunks.keys())
            tokenized_corpus = [
                tokenize(self._chunks[cid].content) for cid in all_chunk_ids
            ]
            self.bm25_index.index(all_chunk_ids, tokenized_corpus)

        return len(indexed_doc_ids), len(new_chunks), indexed_doc_ids

    async def clear_index(self) -> None:
        """Purge all chunks from memory and persistent SQLite storage."""
        async with self._lock:
            self._chunks.clear()
            self.bm25_index = BM25Index()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM rag_chunks;")
                conn.commit()

    def _dense_search(self, query_embedding: list[float]) -> list[tuple[str, float]]:
        """Compute cosine similarity across all dense chunk embeddings."""
        if not self._chunks:
            return []

        q_vec = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0.0:
            return []

        results: list[tuple[str, float]] = []
        for chunk_id, record in self._chunks.items():
            doc_vec = np.asarray(record.embedding, dtype=np.float32)
            d_norm = np.linalg.norm(doc_vec)
            if d_norm == 0.0:
                continue
            sim = float(np.dot(q_vec, doc_vec) / (q_norm * d_norm))
            results.append((chunk_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    async def hybrid_search(
        self, query: str, top_k: int = 5
    ) -> tuple[list[Citation], RetrievalMetrics]:
        """Execute hybrid search using Reciprocal Rank Fusion (RRF)."""
        async with self._lock:
            if not self._chunks:
                return [], RetrievalMetrics(bm25_matches=0, vector_matches=0)

            # 1. Lexical BM25 search
            query_tokens = tokenize(query)
            bm25_results = self.bm25_index.score(query_tokens)
            bm25_matches = sum(1 for _, score in bm25_results if score > 0.0)

            # 2. Dense Vector search
            query_embedding = await self.gateway.create_embedding(query)
            dense_results = self._dense_search(query_embedding)
            vector_matches = len(dense_results)

            # 3. Reciprocal Rank Fusion (RRF)
            # RRF Score = sum(weight / (60 + rank))
            rrf_scores: dict[str, float] = {}
            k_rrf = 60.0
            vector_weight = self.settings.rag_vector_weight
            bm25_weight = self.settings.rag_bm25_weight

            for rank, (chunk_id, _) in enumerate(dense_results):
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                    vector_weight / (k_rrf + rank + 1)
                )

            for rank, (chunk_id, score) in enumerate(bm25_results):
                if score > 0:
                    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (
                        bm25_weight / (k_rrf + rank + 1)
                    )

            # Sort fused candidates
            sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            top_selected = sorted_chunks[:top_k]

            citations: list[Citation] = []
            for chunk_id, fused_score in top_selected:
                record = self._chunks[chunk_id]
                citations.append(
                    Citation(
                        document_id=record.doc_id,
                        title=record.title,
                        score=round(fused_score, 4),
                        snippet=record.content[:300] + ("..." if len(record.content) > 300 else ""),
                        chunk_index=record.chunk_index,
                        metadata=record.metadata,
                    )
                )

            metrics = RetrievalMetrics(
                bm25_matches=bm25_matches,
                vector_matches=vector_matches,
                fusion_method="reciprocal_rank_fusion",
            )
            return citations, metrics

    async def query_and_synthesize(
        self,
        query: str,
        top_k: int = 5,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> tuple[str, list[Citation], RetrievalMetrics, float, float]:
        """Retrieve hybrid context and synthesize authoritative response via LLM."""
        citations, metrics = await self.hybrid_search(query=query, top_k=top_k)

        # Build augmented prompt
        context_blocks: list[str] = []
        for i, cit in enumerate(citations, 1):
            context_blocks.append(
                f"[{i}] Document: '{cit.title}' (ID: {cit.document_id}):\n{cit.snippet}"
            )
        context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant documents found."

        system_prompt = (
            "You are NormaHub Enterprise AI Assistant. Answer the user question accurately and "
            "strictly based on the retrieved context below. Cite document numbers [1], [2] when appropriate."
        )
        user_prompt = f"Context Information:\n{context_str}\n\nUser Question:\n{query}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion_data, ttft_ms, latency_ms = await self.gateway.create_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
        )

        choices = completion_data.get("choices", [])
        answer = choices[0]["message"]["content"] if choices else "Unable to generate response."

        return answer, citations, metrics, ttft_ms, latency_ms


_rag_singleton: HybridRAGEngine | None = None


def get_rag_engine() -> HybridRAGEngine:
    """Return shared Hybrid RAG engine singleton."""
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = HybridRAGEngine()
    return _rag_singleton
