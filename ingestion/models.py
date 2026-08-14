"""Shared ingestion data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CrawledDocument:
    """A cleaned document produced by the crawler."""

    content: str
    title: str
    url: str
    source: str
    timestamp: datetime
    document_id: str = field(default_factory=lambda: str(uuid4()))
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload_base(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "document_id": self.document_id,
            "content_hash": self.content_hash,
        }


@dataclass
class TextChunk:
    """A semantically chunked fragment ready for embedding and indexing."""

    content: str
    title: str
    url: str
    source: str
    timestamp: datetime
    document_id: str
    chunk_index: int
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def to_qdrant_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "content": self.content,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "content_hash": self.content_hash,
        }
        payload.update(self.metadata)
        return payload
