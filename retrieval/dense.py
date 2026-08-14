"""Qdrant dense vector similarity retriever."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from config.settings import Settings, get_settings
from ingestion.embedder import BGEEmbedder
from retrieval.models import RetrievedChunk
from utils.logging import get_logger
from vectorstore.qdrant_client import QdrantStore, QdrantStoreError

logger = get_logger(__name__)


class DenseRetrievalError(Exception):
    """Raised when dense retrieval cannot be completed."""


def _as_vector(raw: Any) -> Optional[list[float]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        if not raw:
            return None
        raw = next(iter(raw.values()))
    try:
        return [float(value) for value in raw]
    except (TypeError, ValueError):
        return None


def point_to_chunk(point: Any) -> RetrievedChunk:
    payload: dict[str, Any] = dict(getattr(point, "payload", None) or {})
    metadata = dict(payload)
    timestamp = str(payload.get("timestamp") or metadata.get("timestamp") or "")
    return RetrievedChunk(
        content=str(payload.get("content") or ""),
        title=str(payload.get("title") or ""),
        url=str(payload.get("url") or ""),
        source=str(payload.get("source") or ""),
        timestamp=timestamp,
        chunk_id=str(payload.get("chunk_id") or getattr(point, "id", "") or ""),
        document_id=str(payload.get("document_id") or ""),
        metadata=metadata,
        embedding=_as_vector(getattr(point, "vector", None)),
        dense_score=float(getattr(point, "score", 0.0) or 0.0),
    )


class DenseRetriever:
    """Retrieve chunks by cosine similarity in Qdrant."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        store: Optional[QdrantStore] = None,
        embedder: Optional[BGEEmbedder] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or QdrantStore(self.settings)
        self.embedder = embedder or BGEEmbedder(self.settings)

    def search(
        self,
        query: Optional[str] = None,
        vector: Optional[Sequence[float]] = None,
        *,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        limit = top_k or self.settings.dense_top_k
        query_vector = self._resolve_vector(query=query, vector=vector)
        try:
            points = self.store.search(
                query_vector,
                limit=limit,
                source_filter=source_filter,
                with_vectors=True,
            )
        except QdrantStoreError as exc:
            logger.error("dense_search_failed", error=str(exc))
            raise DenseRetrievalError("Dense vector search failed") from exc
        chunks = [point_to_chunk(point) for point in points]
        chunks = [chunk for chunk in chunks if chunk.content.strip()]
        logger.info(
            "dense_search_complete",
            results=len(chunks),
            top_k=limit,
            top_score=chunks[0].dense_score if chunks else None,
        )
        return chunks

    def _resolve_vector(
        self,
        query: Optional[str],
        vector: Optional[Sequence[float]],
    ) -> list[float]:
        if vector is not None:
            values = [float(item) for item in vector]
            if not values:
                raise DenseRetrievalError("Query vector is empty")
            return values
        if query is None or not query.strip():
            raise DenseRetrievalError("Provide a query string or an embedding vector")
        encoded = self.embedder.encode([query.strip()])
        if not encoded:
            raise DenseRetrievalError("Embedder returned no query vector")
        return encoded[0]
