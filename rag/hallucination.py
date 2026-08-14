"""Retrieval confidence boundary that blocks hallucinated answers."""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from config.settings import Settings, get_settings
from rag.llm_service import GrokLLMService, LLMServiceError
from retrieval.models import RetrievalResult
from utils.logging import get_logger

logger = get_logger(__name__)

INSUFFICIENT_INFORMATION = "I don't have that information."


class HallucinationBoundary:
    """Refuse generation when hybrid retrieval confidence is below threshold."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    @property
    def threshold(self) -> float:
        return float(self.settings.retrieval_confidence_threshold)

    def is_sufficient(self, confidence: float) -> bool:
        return float(confidence) >= self.threshold

    def refusal_text(self) -> str:
        return INSUFFICIENT_INFORMATION

    def evaluate(self, retrieval: RetrievalResult) -> bool:
        allowed = self.is_sufficient(retrieval.confidence)
        logger.info(
            "hallucination_boundary",
            confidence=retrieval.confidence,
            threshold=self.threshold,
            allowed=allowed,
            evidence_chunks=len(retrieval.chunks),
        )
        return allowed

    async def generate_or_refuse(
        self,
        retrieval: RetrievalResult,
        generate: Callable[[], Awaitable[str]],
    ) -> str:
        """Call `generate` only when evidence is sufficient; otherwise return the fixed refusal."""
        if not self.evaluate(retrieval):
            logger.info("llm_bypassed_insufficient_evidence", confidence=retrieval.confidence)
            return INSUFFICIENT_INFORMATION
        try:
            return await generate()
        except LLMServiceError:
            logger.error("llm_failed_after_boundary_passed")
            raise

    async def complete_with_grok(
        self,
        retrieval: RetrievalResult,
        messages: list[dict[str, str]],
        llm: GrokLLMService,
    ) -> str:
        async def _generate() -> str:
            return await llm.generate(messages)  # type: ignore[arg-type]

        return await self.generate_or_refuse(retrieval, _generate)
