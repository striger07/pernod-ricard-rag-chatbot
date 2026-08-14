"""Qdrant collection management, payload mapping, and health checks."""

from __future__ import annotations

from typing import Any, Optional, Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import Settings, get_settings
from ingestion.models import TextChunk
from utils.logging import get_logger

logger = get_logger(__name__)


class QdrantStoreError(Exception):
    """Raised when Qdrant operations fail."""


def chunk_point_id(chunk: TextChunk) -> str:
    """Deterministic UUID so re-indexing the same chunk is idempotent."""
    key = f"{chunk.url}::{chunk.chunk_index}::{chunk.content_hash}"
    return str(uuid5(NAMESPACE_URL, key))


class QdrantStore:
    """Production wrapper around QdrantClient for the RAG collection."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[QdrantClient] = None) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def collection_name(self) -> str:
        return self.settings.qdrant_collection

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            kwargs: dict[str, Any] = {
                "host": self.settings.qdrant_host,
                "port": self.settings.qdrant_port,
                "https": self.settings.qdrant_https,
                "timeout": self.settings.qdrant_timeout_seconds,
            }
            if self.settings.qdrant_api_key:
                kwargs["api_key"] = self.settings.qdrant_api_key
            try:
                self._client = QdrantClient(**kwargs)
            except Exception as exc:
                logger.error("qdrant_connect_failed", host=self.settings.qdrant_host, error=str(exc))
                raise QdrantStoreError("Unable to connect to Qdrant") from exc
        return self._client

    def health_check(self) -> dict[str, Any]:
        """Return health metadata for localhost:6333 or a configured remote instance."""
        try:
            collections = self.client.get_collections()
            names = [item.name for item in collections.collections]
            exists = self.collection_name in names
            points = 0
            if exists:
                info = self.client.get_collection(self.collection_name)
                points = int(info.points_count or 0)
            logger.info(
                "qdrant_health_ok",
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                collection=self.collection_name,
                exists=exists,
                points=points,
            )
            return {
                "status": "ok",
                "host": self.settings.qdrant_host,
                "port": self.settings.qdrant_port,
                "collection": self.collection_name,
                "collection_exists": exists,
                "points_count": points,
            }
        except Exception as exc:
            logger.error("qdrant_health_failed", error=str(exc))
            return {
                "status": "error",
                "host": self.settings.qdrant_host,
                "port": self.settings.qdrant_port,
                "error": "qdrant_unavailable",
            }

    def ensure_collection(self, vector_size: Optional[int] = None) -> None:
        size = vector_size or self.settings.embedding_dimension
        try:
            existing = {item.name for item in self.client.get_collections().collections}
            if self.collection_name in existing:
                logger.info("qdrant_collection_exists", collection=self.collection_name)
                self._ensure_payload_indexes()
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=size,
                    distance=qmodels.Distance.COSINE,
                    on_disk=False,
                ),
                hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=100),
                optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=20000),
                on_disk_payload=True,
            )
            self._ensure_payload_indexes()
            logger.info(
                "qdrant_collection_created",
                collection=self.collection_name,
                vector_size=size,
            )
        except UnexpectedResponse as exc:
            logger.error("qdrant_collection_init_failed", error=str(exc))
            raise QdrantStoreError("Failed to initialize Qdrant collection") from exc

    def recreate_collection(self, vector_size: Optional[int] = None) -> None:
        size = vector_size or self.settings.embedding_dimension
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=size,
                    distance=qmodels.Distance.COSINE,
                ),
                hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=100),
                on_disk_payload=True,
            )
            self._ensure_payload_indexes()
            logger.info("qdrant_collection_recreated", collection=self.collection_name)
        except Exception as exc:
            logger.error("qdrant_recreate_failed", error=str(exc))
            raise QdrantStoreError("Failed to recreate Qdrant collection") from exc

    def upsert_chunks(self, chunks: Sequence[TextChunk], *, wait: bool = True) -> int:
        if not chunks:
            return 0
        self.ensure_collection(vector_size=self.settings.embedding_dimension)
        points: list[qmodels.PointStruct] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise QdrantStoreError(f"Chunk {chunk.chunk_id} is missing an embedding")
            if len(chunk.embedding) != self.settings.embedding_dimension:
                raise QdrantStoreError(
                    f"Embedding dimension {len(chunk.embedding)} does not match "
                    f"{self.settings.embedding_dimension}"
                )
            points.append(
                qmodels.PointStruct(
                    id=chunk_point_id(chunk),
                    vector=chunk.embedding,
                    payload=chunk.to_qdrant_payload(),
                )
            )
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=wait,
            )
        except Exception as exc:
            logger.error("qdrant_upsert_failed", count=len(points), error=str(exc))
            raise QdrantStoreError("Failed to upsert points into Qdrant") from exc
        logger.info("qdrant_upsert_complete", count=len(points), collection=self.collection_name)
        return len(points)

    def delete_by_url(self, url: str) -> None:
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="url",
                                match=qmodels.MatchValue(value=url),
                            )
                        ]
                    )
                ),
            )
            logger.info("qdrant_deleted_by_url", url=url)
        except Exception as exc:
            logger.error("qdrant_delete_failed", url=url, error=str(exc))
            raise QdrantStoreError("Failed to delete points by URL") from exc

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        source_filter: Optional[str] = None,
        *,
        with_vectors: bool = False,
    ) -> list[qmodels.ScoredPoint]:
        query_filter = None
        if source_filter:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source",
                        match=qmodels.MatchValue(value=source_filter),
                    )
                ]
            )
        try:
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=list(vector),
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=with_vectors,
            )
        except Exception as exc:
            logger.error("qdrant_search_failed", error=str(exc))
            raise QdrantStoreError("Qdrant similarity search failed") from exc

    def _ensure_payload_indexes(self) -> None:
        for field_name, schema in (
            ("url", qmodels.PayloadSchemaType.KEYWORD),
            ("source", qmodels.PayloadSchemaType.KEYWORD),
            ("document_id", qmodels.PayloadSchemaType.KEYWORD),
            ("title", qmodels.PayloadSchemaType.KEYWORD),
            ("timestamp", qmodels.PayloadSchemaType.KEYWORD),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:
                logger.info(
                    "qdrant_payload_index_skip",
                    field=field_name,
                    error=str(exc),
                )


def get_qdrant_store(settings: Optional[Settings] = None) -> QdrantStore:
    return QdrantStore(settings=settings)
