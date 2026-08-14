"""Hybrid retrieval orchestrator: dense + BM25 → RRF → MMR → confidence."""

from __future__ import annotations

from typing import Optional, Sequence

from config.settings import Settings, get_settings
from ingestion.embedder import BGEEmbedder
from retrieval.bm25 import BM25Index
from retrieval.dense import DenseRetrievalError, DenseRetriever
from retrieval.fusion import ReciprocalRankFusion
from retrieval.mmr import MMRReranker
from retrieval.models import RetrievalResult, RetrievedChunk
from utils.logging import get_logger
from vectorstore.qdrant_client import QdrantStore

logger = get_logger(__name__)


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def _clip_01(value: float) -> float:
    return max(0.0, min(1.0, value))


class HybridRetrievalPipeline:
    """Run the full hybrid retrieval stack and score confidence in [0.0, 1.0]."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        dense: Optional[DenseRetriever] = None,
        bm25: Optional[BM25Index] = None,
        fusion: Optional[ReciprocalRankFusion] = None,
        mmr: Optional[MMRReranker] = None,
        embedder: Optional[BGEEmbedder] = None,
        store: Optional[QdrantStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or BGEEmbedder(self.settings)
        self.store = store or QdrantStore(self.settings)
        self.dense = dense or DenseRetriever(
            settings=self.settings, store=self.store, embedder=self.embedder
        )
        self.bm25 = bm25 or BM25Index(self.settings)
        self.fusion = fusion or ReciprocalRankFusion(self.settings)
        self.mmr = mmr or MMRReranker(settings=self.settings, embedder=self.embedder)

    def retrieve(
        self,
        query: str,
        *,
        query_vector: Optional[Sequence[float]] = None,
        source_filter: Optional[str] = None,
        final_top_k: Optional[int] = None,
    ) -> RetrievalResult:
        prepared = normalize_query(query)
        if not prepared:
            logger.info("retrieval_empty_query")
            return RetrievalResult(query=query, chunks=[], confidence=0.0)

        vector = list(query_vector) if query_vector is not None else None
        if vector is None:
            try:
                encoded = self.embedder.encode([prepared])
                vector = encoded[0] if encoded else None
            except Exception as exc:
                logger.warning("query_embedding_failed", error=str(exc))
                vector = None

        dense_hits = self._dense_search(prepared, vector, source_filter)
        bm25_hits = self._bm25_search(prepared)
        fused = self.fusion.fuse(dense_hits, bm25_hits)
        selected = self.mmr.rerank(
            prepared,
            fused,
            query_embedding=vector,
            top_k=final_top_k or self.settings.final_top_k,
        )
        scored = [self._score_chunk(chunk) for chunk in selected]
        confidence = self._overall_confidence(scored, dense_hits, bm25_hits)
        result = RetrievalResult(
            query=prepared,
            chunks=scored,
            confidence=confidence,
            dense_count=len(dense_hits),
            bm25_count=len(bm25_hits),
            fused_count=len(fused),
        )
        logger.info(
            "hybrid_retrieval_complete",
            query_chars=len(prepared),
            dense=result.dense_count,
            bm25=result.bm25_count,
            fused=result.fused_count,
            final=len(result.chunks),
            confidence=result.confidence,
            top_scores=[
                {
                    "title": chunk.title,
                    "dense": chunk.dense_score,
                    "bm25": chunk.bm25_score,
                    "rrf": chunk.rrf_score,
                    "mmr": chunk.mmr_score,
                    "confidence": chunk.confidence_score,
                }
                for chunk in result.chunks[:5]
            ],
        )
        return result

    def _dense_search(
        self,
        query: str,
        vector: Optional[list[float]],
        source_filter: Optional[str],
    ) -> list[RetrievedChunk]:
        try:
            if vector is not None:
                return self.dense.search(vector=vector, source_filter=source_filter)
            return self.dense.search(query=query, source_filter=source_filter)
        except DenseRetrievalError as exc:
            logger.error("dense_branch_failed", error=str(exc))
            return []

    def _bm25_search(self, query: str) -> list[RetrievedChunk]:
        try:
            return self.bm25.search(query, top_k=self.settings.bm25_top_k)
        except Exception as exc:
            logger.error("bm25_branch_failed", error=str(exc))
            return []

    def _score_chunk(self, chunk: RetrievedChunk) -> RetrievedChunk:
        dense = _clip_01(float(chunk.dense_score or 0.0))
        rrf_ceiling = 2.0 / (float(self.fusion.k) + 1.0)
        rrf = _clip_01(float(chunk.rrf_score or 0.0) / rrf_ceiling) if rrf_ceiling else 0.0
        mmr_raw = float(chunk.mmr_score or 0.0)
        mmr = _clip_01((mmr_raw + 1.0) / 2.0)
        bm25_present = 1.0 if chunk.bm25_score not in (None, 0.0) else 0.0
        chunk.confidence_score = round(
            _clip_01(0.40 * dense + 0.30 * rrf + 0.20 * mmr + 0.10 * bm25_present),
            6,
        )
        return chunk

    def _overall_confidence(
        self,
        chunks: Sequence[RetrievedChunk],
        dense_hits: Sequence[RetrievedChunk],
        bm25_hits: Sequence[RetrievedChunk],
    ) -> float:
        if not chunks:
            return 0.0
        top = [float(chunk.confidence_score or 0.0) for chunk in chunks[:3]]
        mean_top = sum(top) / len(top)
        both_retrievers = 1.0 if dense_hits and bm25_hits else 0.6 if (dense_hits or bm25_hits) else 0.0
        agreement = 0.0
        if chunks:
            lead = chunks[0]
            if lead.dense_score not in (None, 0.0) and lead.bm25_score not in (None, 0.0):
                agreement = 0.15
        return round(_clip_01(0.75 * mean_top + 0.10 * both_retrievers + agreement), 6)
