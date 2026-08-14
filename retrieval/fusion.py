"""Reciprocal Rank Fusion of dense and BM25 ranked lists."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from config.settings import Settings, get_settings
from retrieval.models import RetrievedChunk, chunk_identity
from utils.logging import get_logger

logger = get_logger(__name__)


def rrf_score(rank: int, k: int) -> float:
    """Standard RRF contribution for a 1-based rank."""
    if rank < 1:
        raise ValueError("RRF rank must be 1-based and >= 1")
    if k < 1:
        raise ValueError("RRF k must be >= 1")
    return 1.0 / (float(k) + float(rank))


class ReciprocalRankFusion:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion (not concatenation)."""

    def __init__(self, settings: Optional[Settings] = None, k: Optional[int] = None) -> None:
        self.settings = settings or get_settings()
        self.k = int(k if k is not None else self.settings.rrf_k)

    def fuse(self, *ranked_lists: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        if self.k < 1:
            raise ValueError("RRF k must be >= 1")
        contributions: dict[str, RetrievedChunk] = {}
        totals: dict[str, float] = {}

        for ranked in ranked_lists:
            for position, chunk in enumerate(ranked, start=1):
                key = chunk_identity(chunk)
                totals[key] = totals.get(key, 0.0) + rrf_score(position, self.k)
                existing = contributions.get(key)
                if existing is None:
                    contributions[key] = self._copy(chunk)
                    continue
                self._merge_into(existing, chunk)

        fused: list[RetrievedChunk] = []
        for key, chunk in contributions.items():
            chunk.rrf_score = totals[key]
            fused.append(chunk)
        fused.sort(key=lambda item: item.rrf_score or 0.0, reverse=True)
        logger.info(
            "rrf_complete",
            lists=len(ranked_lists),
            k=self.k,
            fused=len(fused),
            top_rrf=fused[0].rrf_score if fused else None,
        )
        return fused

    def _copy(self, chunk: RetrievedChunk) -> RetrievedChunk:
        return RetrievedChunk(
            content=chunk.content,
            title=chunk.title,
            url=chunk.url,
            source=chunk.source,
            metadata=dict(chunk.metadata),
            timestamp=chunk.timestamp,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            embedding=list(chunk.embedding) if chunk.embedding is not None else None,
            dense_score=chunk.dense_score,
            bm25_score=chunk.bm25_score,
            rrf_score=chunk.rrf_score,
            mmr_score=chunk.mmr_score,
            confidence_score=chunk.confidence_score,
        )

    def _merge_into(self, target: RetrievedChunk, incoming: RetrievedChunk) -> None:
        if incoming.dense_score is not None:
            target.dense_score = incoming.dense_score
        if incoming.bm25_score is not None:
            target.bm25_score = incoming.bm25_score
        if incoming.embedding is not None and target.embedding is None:
            target.embedding = list(incoming.embedding)
        if incoming.title and not target.title:
            target.title = incoming.title
        if incoming.url and not target.url:
            target.url = incoming.url
        if incoming.source and not target.source:
            target.source = incoming.source
        if incoming.timestamp and not target.timestamp:
            target.timestamp = incoming.timestamp
        if incoming.content and len(incoming.content) > len(target.content):
            target.content = incoming.content
        target.metadata.update(incoming.metadata)


def fuse_rankings(
    rankings: Iterable[Sequence[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    fusion = ReciprocalRankFusion(k=k)
    return fusion.fuse(*tuple(rankings))
