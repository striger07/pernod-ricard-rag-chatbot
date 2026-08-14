"""BAAI/bge-m3 embedding service with batching and exponential backoff."""

from __future__ import annotations

import time
from typing import Optional, Sequence

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import Settings, get_settings
from ingestion.models import TextChunk
from utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when embeddings cannot be produced."""


class BGEEmbedder:
    """Local sentence-transformers embedder using BAAI/bge-m3 (1024-d dense)."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._model = None

    @property
    def dimension(self) -> int:
        return self.settings.embedding_dimension

    @property
    def model_name(self) -> str:
        return self.settings.embedding_model

    def _load_model(self):
        if self._model is not None:
            return self._model
        started = time.perf_counter()
        logger.info(
            "embedding_model_load_start",
            model=self.settings.embedding_model,
            device=self.settings.embedding_device,
        )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError("sentence-transformers is not installed") from exc

        try:
            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.settings.embedding_device,
            )
        except Exception as exc:
            raise EmbeddingError(f"Failed to load embedding model {self.settings.embedding_model}") from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("embedding_model_load_complete", elapsed_ms=elapsed_ms)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches with retry/backoff and OOM batch-size fallback."""
        if not texts:
            return []
        normalized = [self._prepare_text(text) for text in texts]
        batch_size = self.settings.embedding_batch_size
        vectors: list[list[float]] = []
        index = 0
        while index < len(normalized):
            batch = normalized[index : index + batch_size]
            try:
                encoded = self._encode_batch(batch)
            except Exception as exc:
                message = str(exc).lower()
                if batch_size > 1 and ("out of memory" in message or "cuda" in message):
                    batch_size = max(1, batch_size // 2)
                    logger.warning("embedding_batch_reduced", batch_size=batch_size, error=str(exc))
                    continue
                raise
            vectors.extend(encoded)
            index += len(batch)
        if len(vectors) != len(texts):
            raise EmbeddingError("Embedding count does not match input count")
        return vectors

    def embed_chunks(self, chunks: Sequence[TextChunk]) -> list[TextChunk]:
        vectors = self.encode([chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
            chunk.metadata["embedding_model"] = self.model_name
            chunk.metadata["embedding_dimension"] = self.dimension
        logger.info("chunks_embedded", count=len(chunks), model=self.model_name)
        return list(chunks)

    def _prepare_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            return " "
        return cleaned

    def _encode_batch(self, batch: Sequence[str]) -> list[list[float]]:
        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(self.settings.embedding_max_retries),
            wait=wait_exponential(multiplier=0.4, min=0.2, max=30),
            retry=retry_if_exception_type((OSError, RuntimeError, TimeoutError, EmbeddingError)),
        )
        for attempt in retrying:
            with attempt:
                model = self._load_model()
                started = time.perf_counter()
                try:
                    output = model.encode(
                        list(batch),
                        batch_size=len(batch),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                except Exception as exc:
                    logger.error("embedding_batch_failed", size=len(batch), error=str(exc))
                    raise EmbeddingError("Embedding batch failed") from exc
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                logger.info("embedding_batch_complete", size=len(batch), elapsed_ms=elapsed_ms)
                return [row.astype(float).tolist() for row in output]
        raise EmbeddingError("Embedding batch exhausted retries")


def get_embedder(settings: Optional[Settings] = None) -> BGEEmbedder:
    return BGEEmbedder(settings=settings)
