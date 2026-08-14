"""Persistent BM25 lexical retriever over the chunked corpus."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from rank_bm25 import BM25Okapi

from config.settings import Settings, get_settings
from ingestion.models import TextChunk
from retrieval.models import RetrievedChunk
from utils.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class BM25IndexError(Exception):
    """Raised when the BM25 index cannot be built, loaded, or queried."""


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens suitable for BM25Okapi."""
    return _TOKEN_RE.findall(text.lower())


def _record_from_text_chunk(chunk: TextChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "title": chunk.title,
        "url": chunk.url,
        "source": chunk.source,
        "timestamp": chunk.timestamp.isoformat() if chunk.timestamp else "",
        "metadata": dict(chunk.metadata),
        "tokens": tokenize(f"{chunk.title} {chunk.content}"),
    }


def _record_from_retrieved(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "title": chunk.title,
        "url": chunk.url,
        "source": chunk.source,
        "timestamp": chunk.timestamp,
        "metadata": dict(chunk.metadata),
        "tokens": tokenize(f"{chunk.title} {chunk.content}"),
    }


def _record_to_chunk(record: dict[str, Any], score: float) -> RetrievedChunk:
    metadata = dict(record.get("metadata") or {})
    return RetrievedChunk(
        content=str(record.get("content") or ""),
        title=str(record.get("title") or ""),
        url=str(record.get("url") or ""),
        source=str(record.get("source") or ""),
        timestamp=str(record.get("timestamp") or ""),
        chunk_id=str(record.get("chunk_id") or ""),
        document_id=str(record.get("document_id") or ""),
        metadata=metadata,
        bm25_score=float(score),
    )


class BM25Index:
    """Disk-backed BM25 index synchronized with ingested chunks."""

    def __init__(self, settings: Optional[Settings] = None, index_path: Optional[str] = None) -> None:
        self.settings = settings or get_settings()
        self.index_path = Path(index_path or self.settings.bm25_index_path)
        self._records: list[dict[str, Any]] = []
        self._engine: Optional[BM25Okapi] = None
        self._ids: dict[str, int] = {}

    @property
    def size(self) -> int:
        return len(self._records)

    def build(self, chunks: Sequence[TextChunk] | Sequence[RetrievedChunk]) -> None:
        self._records = []
        self._ids = {}
        self._engine = None
        self.update(chunks, persist=True)

    def update(self, chunks: Sequence[TextChunk] | Sequence[RetrievedChunk], *, persist: bool = True) -> int:
        if self._engine is None and self.index_path.exists() and not self._records:
            self.load()
        added = 0
        for chunk in chunks:
            record = (
                _record_from_text_chunk(chunk)
                if isinstance(chunk, TextChunk)
                else _record_from_retrieved(chunk)
            )
            chunk_id = str(record["chunk_id"] or record["url"] or record["content"][:48])
            record["chunk_id"] = chunk_id
            existing = self._ids.get(chunk_id)
            if existing is not None:
                self._records[existing] = record
            else:
                self._ids[chunk_id] = len(self._records)
                self._records.append(record)
                added += 1
        self._rebuild_engine()
        if persist:
            self.save()
        logger.info("bm25_index_updated", size=self.size, added=added)
        return added

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.settings.bm25_k1,
            "b": self.settings.bm25_b,
            "records": self._records,
        }
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        try:
            with tmp_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(self.index_path)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            logger.error("bm25_save_failed", path=str(self.index_path), error=str(exc))
            raise BM25IndexError("Failed to persist BM25 index") from exc
        logger.info("bm25_index_saved", path=str(self.index_path), size=self.size)

    def load(self) -> None:
        if not self.index_path.exists():
            raise BM25IndexError(f"BM25 index not found at {self.index_path}")
        try:
            with self.index_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            logger.error("bm25_load_failed", path=str(self.index_path), error=str(exc))
            raise BM25IndexError("Failed to load BM25 index") from exc
        self._records = list(payload.get("records") or [])
        self._ids = {
            str(record.get("chunk_id") or index): index for index, record in enumerate(self._records)
        }
        self._rebuild_engine()
        logger.info("bm25_index_loaded", path=str(self.index_path), size=self.size)

    def search(self, query: str, *, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        if self._engine is None:
            if self.index_path.exists():
                self.load()
            else:
                logger.warning("bm25_search_empty_index")
                return []
        if not self._records or self._engine is None:
            return []
        limit = top_k or self.settings.bm25_top_k
        tokens = tokenize(query)
        if not tokens:
            return []
        try:
            scores = self._engine.get_scores(tokens)
        except Exception as exc:
            logger.error("bm25_search_failed", error=str(exc))
            raise BM25IndexError("BM25 scoring failed") from exc
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: list[RetrievedChunk] = []
        for index, score in ranked[:limit]:
            if float(score) <= 0.0:
                continue
            results.append(_record_to_chunk(self._records[index], float(score)))
        logger.info("bm25_search_complete", results=len(results), top_k=limit)
        return results

    def _rebuild_engine(self) -> None:
        if not self._records:
            self._engine = None
            self._ids = {}
            return
        corpus = [list(record.get("tokens") or tokenize(str(record.get("content") or ""))) for record in self._records]
        self._engine = BM25Okapi(corpus, k1=self.settings.bm25_k1, b=self.settings.bm25_b)
        self._ids = {
            str(record.get("chunk_id") or index): index for index, record in enumerate(self._records)
        }
