"""Semantic chunker that preserves paragraph/section boundaries."""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np

from config.settings import Settings, get_settings
from ingestion.models import CrawledDocument, TextChunk
from utils.logging import get_logger

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


class EmbeddingFn(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


def _split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in _PARAGRAPH_SPLIT_RE.split(text) if part.strip()]
    return parts


def _split_oversized(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = _SENTENCE_RE.split(paragraph)
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        extra = len(sentence) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            pieces.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += extra
    if current:
        pieces.append(" ".join(current).strip())

    final: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            final.append(piece)
            continue
        for start in range(0, len(piece), max_chars):
            final.append(piece[start : start + max_chars].strip())
    return [item for item in final if item]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _overlap_prefix(previous: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or not previous:
        return ""
    if len(previous) <= overlap_chars:
        return previous.strip()
    window = previous[-overlap_chars:]
    boundary = window.find(" ")
    if boundary == -1:
        return window.strip()
    return window[boundary + 1 :].strip()


class SemanticChunker:
    """Merge consecutive paragraphs until a semantic or size boundary is hit."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def chunk_documents(
        self,
        documents: Iterable[CrawledDocument],
        embedder: Optional[EmbeddingFn] = None,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        document_count = 0
        for document in documents:
            document_count += 1
            chunks.extend(self.chunk_document(document, embedder=embedder))
        logger.info("chunking_complete", documents=document_count, chunks=len(chunks))
        return chunks

    def chunk_document(
        self,
        document: CrawledDocument,
        embedder: Optional[EmbeddingFn] = None,
    ) -> list[TextChunk]:
        units = self._semantic_units(document.content)
        if not units:
            return []

        groups = self._group_units(units, embedder=embedder)
        chunks: list[TextChunk] = []
        previous_text = ""
        for index, group in enumerate(groups):
            overlap = _overlap_prefix(previous_text, self.settings.chunk_overlap_chars)
            body = "\n\n".join(group).strip()
            content = f"{overlap}\n\n{body}".strip() if overlap else body
            if len(content) < self.settings.chunk_min_chars and chunks:
                previous = chunks[-1]
                previous.content = f"{previous.content}\n\n{body}".strip()
                previous_text = previous.content
                continue
            chunk = TextChunk(
                content=content,
                title=document.title,
                url=document.url,
                source=document.source,
                timestamp=document.timestamp,
                document_id=document.document_id,
                chunk_index=index,
                content_hash=document.content_hash,
                metadata={
                    **document.metadata,
                    "chunk_index": index,
                    "title": document.title,
                    "url": document.url,
                    "source": document.source,
                    "timestamp": document.timestamp.isoformat(),
                },
            )
            chunks.append(chunk)
            previous_text = body

        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_count"] = len(chunks)

        logger.info(
            "document_chunked",
            url=document.url,
            chunks=len(chunks),
            title=document.title,
        )
        return chunks

    def _semantic_units(self, text: str) -> list[str]:
        sections: list[str] = []
        current: list[str] = []
        for line in text.split("\n"):
            if _HEADING_RE.match(line.strip()):
                if current:
                    sections.append("\n".join(current).strip())
                    current = []
                current.append(line.strip())
            else:
                current.append(line)
        if current:
            sections.append("\n".join(current).strip())

        units: list[str] = []
        for section in sections:
            for paragraph in _split_paragraphs(section):
                units.extend(_split_oversized(paragraph, self.settings.chunk_max_chars))
        return units

    def _group_units(self, units: list[str], embedder: Optional[EmbeddingFn]) -> list[list[str]]:
        if len(units) == 1:
            return [units]

        similarities: list[float] = []
        if embedder is not None and len(units) > 1:
            try:
                vectors = embedder.encode(units)
                for index in range(len(units) - 1):
                    similarities.append(_cosine(vectors[index], vectors[index + 1]))
            except Exception as exc:
                logger.warning("semantic_similarity_unavailable", error=str(exc))
                similarities = []

        groups: list[list[str]] = []
        current: list[str] = [units[0]]
        current_len = len(units[0])
        threshold = self.settings.semantic_similarity_threshold
        max_chars = self.settings.chunk_max_chars

        for index in range(1, len(units)):
            unit = units[index]
            candidate_len = current_len + 2 + len(unit)
            similar = True
            if similarities:
                similar = similarities[index - 1] >= threshold
            if candidate_len > max_chars or not similar:
                groups.append(current)
                current = [unit]
                current_len = len(unit)
            else:
                current.append(unit)
                current_len = candidate_len
        if current:
            groups.append(current)
        return groups


def estimate_token_count(text: str) -> int:
    """Approximate token count without requiring a tokenizer dependency."""
    return max(1, math.ceil(len(text.split()) * 1.3))
