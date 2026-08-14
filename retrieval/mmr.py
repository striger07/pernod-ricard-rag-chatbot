"""Maximal Marginal Relevance re-ranking for diverse context."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from config.settings import Settings, get_settings
from ingestion.embedder import BGEEmbedder
from retrieval.models import RetrievedChunk
from utils.logging import get_logger

logger = get_logger(__name__)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class MMRReranker:
    """Select a diverse subset of fused candidates against the query embedding."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedder: Optional[BGEEmbedder] = None,
        lambda_mult: Optional[float] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or BGEEmbedder(self.settings)
        self.lambda_mult = float(lambda_mult if lambda_mult is not None else self.settings.mmr_lambda)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: Optional[int] = None,
        candidate_pool: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        limit = top_k or self.settings.final_top_k
        pool_size = candidate_pool or self.settings.mmr_candidates
        pool = list(candidates[: max(limit, pool_size)])
        query_vector = list(query_embedding) if query_embedding is not None else self._embed_query(query)
        document_vectors = self._document_vectors(pool)
        if query_vector is None or document_vectors is None:
            logger.warning("mmr_embeddings_unavailable_falling_back_to_rrf")
            selected = pool[:limit]
            for chunk in selected:
                chunk.mmr_score = chunk.rrf_score
            return selected

        relevance = [cosine_similarity(query_vector, vector) for vector in document_vectors]
        selected_indices: list[int] = []
        remaining = set(range(len(pool)))
        mmr_values: dict[int, float] = {}

        while remaining and len(selected_indices) < limit:
            best_index = None
            best_score = float("-inf")
            for index in remaining:
                if not selected_indices:
                    score = self.lambda_mult * relevance[index]
                else:
                    redundancy = max(
                        cosine_similarity(document_vectors[index], document_vectors[chosen])
                        for chosen in selected_indices
                    )
                    score = self.lambda_mult * relevance[index] - (1.0 - self.lambda_mult) * redundancy
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is None:
                break
            selected_indices.append(best_index)
            remaining.remove(best_index)
            mmr_values[best_index] = best_score

        ranked = [pool[index] for index in selected_indices]
        for index, chunk in zip(selected_indices, ranked, strict=True):
            chunk.mmr_score = float(mmr_values[index])
        logger.info(
            "mmr_complete",
            pool=len(pool),
            selected=len(ranked),
            lambda_mult=self.lambda_mult,
        )
        return ranked

    def _embed_query(self, query: str) -> Optional[list[float]]:
        try:
            encoded = self.embedder.encode([query])
        except Exception as exc:
            logger.warning("mmr_query_embed_failed", error=str(exc))
            return None
        return encoded[0] if encoded else None

    def _document_vectors(self, pool: Sequence[RetrievedChunk]) -> Optional[list[list[float]]]:
        vectors: list[Optional[list[float]]] = [
            list(chunk.embedding) if chunk.embedding is not None else None for chunk in pool
        ]
        missing_indices = [index for index, vector in enumerate(vectors) if vector is None]
        if missing_indices:
            texts = [pool[index].content for index in missing_indices]
            try:
                encoded = self.embedder.encode(texts)
            except Exception as exc:
                logger.warning("mmr_document_embed_failed", error=str(exc))
                if all(vector is None for vector in vectors):
                    return None
                encoded = []
            for offset, index in enumerate(missing_indices):
                if offset < len(encoded):
                    vectors[index] = encoded[offset]
                    pool[index].embedding = encoded[offset]
        if any(vector is None for vector in vectors):
            return None
        return [vector for vector in vectors if vector is not None]
