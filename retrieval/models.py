"""Structured retrieval result models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def chunk_identity(chunk: "RetrievedChunk") -> str:
    """Stable key used by RRF fusion across retrievers."""
    if chunk.chunk_id:
        return chunk.chunk_id
    if chunk.url and chunk.metadata.get("chunk_index") is not None:
        return f"{chunk.url}::{chunk.metadata.get('chunk_index')}"
    if chunk.url:
        return f"{chunk.url}::{chunk.content[:48]}"
    return chunk.content[:96]


@dataclass
class RetrievedChunk:
    """One retrieved passage with scores from every hybrid stage that produced it."""

    content: str
    title: str
    url: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    chunk_id: str = ""
    document_id: str = ""
    embedding: Optional[list[float]] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    mmr_score: Optional[float] = None
    confidence_score: Optional[float] = None

    def identity(self) -> str:
        return chunk_identity(self)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("embedding", None)
        return payload


@dataclass
class RetrievalResult:
    """Hybrid retrieval output including a unified confidence in [0.0, 1.0]."""

    query: str
    chunks: list[RetrievedChunk]
    confidence: float
    dense_count: int = 0
    bm25_count: int = 0
    fused_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "confidence": self.confidence,
            "dense_count": self.dense_count,
            "bm25_count": self.bm25_count,
            "fused_count": self.fused_count,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
