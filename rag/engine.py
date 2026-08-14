"""RAG chat engine: guardrails → retrieval → confidence boundary → Grok."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from api.schemas import ChatRequest, ChatResponse, Citation
from api.sessions import SessionState, SessionStore
from config.settings import Settings, get_settings
from guardrails.orchestrator import GuardrailOrchestrator
from rag.hallucination import INSUFFICIENT_INFORMATION, HallucinationBoundary
from rag.llm_service import GrokLLMService, LLMServiceError
from rag.prompts import build_messages
from retrieval.models import RetrievalResult, RetrievedChunk
from retrieval.pipeline import HybridRetrievalPipeline
from utils.logging import get_logger

logger = get_logger(__name__)


def citations_from_chunks(chunks: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for chunk in chunks:
        url = (chunk.url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        score = chunk.confidence_score
        if score is None:
            score = chunk.rrf_score if chunk.rrf_score is not None else chunk.dense_score or 0.0
        citations.append(
            Citation(title=chunk.title or "Source", url=url, score=float(score or 0.0))
        )
    return citations


@dataclass
class StreamEvent:
    payload: dict[str, Any]


class RagEngine:
    """Production chat/retrieve orchestration used by the FastAPI routes."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        guardrails: Optional[GuardrailOrchestrator] = None,
        retriever: Optional[HybridRetrievalPipeline] = None,
        llm: Optional[GrokLLMService] = None,
        boundary: Optional[HallucinationBoundary] = None,
        sessions: Optional[SessionStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.guardrails = guardrails or GuardrailOrchestrator(self.settings)
        self.retriever = retriever or HybridRetrievalPipeline(self.settings)
        self.llm = llm or GrokLLMService(self.settings)
        self.boundary = boundary or HallucinationBoundary(self.settings)
        self.sessions = sessions or SessionStore()

    def _session(self, session_id: Optional[str]) -> SessionState:
        return self.sessions.get_or_create(session_id)

    def _header_verified(self, header_value: Optional[str]) -> bool:
        if not header_value:
            return False
        return header_value.strip().lower() in {"1", "true", "yes", "verified"}

    def inspect(
        self,
        query: str,
        request: ChatRequest,
        session: SessionState,
        header_verified: bool,
    ):
        return self.guardrails.inspect_query(
            query,
            age_verified=request.age_verified,
            declared_age=request.declared_age,
            session_verified=session.age_verified,
            header_verified=header_verified,
        )

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> RetrievalResult:
        return self.retriever.retrieve(query, final_top_k=top_k)

    async def chat(self, request: ChatRequest, *, header_verified: bool = False) -> ChatResponse:
        session = self._session(request.session_id)
        decision = self.inspect(request.message, request, session, header_verified)
        if decision.blocked:
            logger.info("chat_blocked", policy=decision.policy, session_id=session.session_id)
            return ChatResponse(
                answer=decision.message,
                blocked=True,
                policy=decision.policy,
                confidence=None,
                citations=[],
                session_id=session.session_id,
                redirect_url=decision.redirect_url,
            )
        session.age_verified = True

        retrieval = self.retrieve(request.message)
        if not self.boundary.is_sufficient(retrieval.confidence):
            logger.info("chat_insufficient_evidence", confidence=retrieval.confidence)
            return ChatResponse(
                answer=INSUFFICIENT_INFORMATION,
                blocked=False,
                policy="hallucination_boundary",
                confidence=retrieval.confidence,
                citations=[],
                session_id=session.session_id,
            )

        history = [turn.model_dump() for turn in request.history] or session.history
        messages = build_messages(request.message, retrieval.chunks, history)
        try:
            raw = await self.llm.generate(messages)  # type: ignore[arg-type]
        except LLMServiceError:
            logger.error("chat_llm_failed", session_id=session.session_id)
            raise
        answer = self.guardrails.polish_answer(
            raw, needs_responsible_drinking=decision.needs_responsible_drinking
        )
        session.append("user", request.message)
        session.append("assistant", answer)
        return ChatResponse(
            answer=answer,
            blocked=False,
            policy="allow",
            confidence=retrieval.confidence,
            citations=citations_from_chunks(retrieval.chunks),
            session_id=session.session_id,
        )

    async def chat_stream(
        self,
        request: ChatRequest,
        *,
        header_verified: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        session = self._session(request.session_id)
        decision = self.inspect(request.message, request, session, header_verified)
        if decision.blocked:
            yield {
                "type": "blocked",
                "policy": decision.policy,
                "answer": decision.message,
                "redirect_url": decision.redirect_url,
                "session_id": session.session_id,
            }
            return
        session.age_verified = True
        retrieval = self.retrieve(request.message)
        citations = [item.model_dump() for item in citations_from_chunks(retrieval.chunks)]
        yield {
            "type": "meta",
            "confidence": retrieval.confidence,
            "citations": citations,
            "session_id": session.session_id,
            "policy": "allow",
        }
        if not self.boundary.is_sufficient(retrieval.confidence):
            yield {"type": "token", "text": INSUFFICIENT_INFORMATION}
            yield {
                "type": "done",
                "answer": INSUFFICIENT_INFORMATION,
                "policy": "hallucination_boundary",
                "session_id": session.session_id,
            }
            return
        history = [turn.model_dump() for turn in request.history] or session.history
        messages = build_messages(request.message, retrieval.chunks, history)
        collected: list[str] = []
        async for token in self.llm.generate_stream(messages):  # type: ignore[arg-type]
            collected.append(token)
            yield {"type": "token", "text": token}
        raw = "".join(collected).strip()
        polished = self.guardrails.polish_answer(
            raw, needs_responsible_drinking=decision.needs_responsible_drinking
        )
        suffix = polished[len(raw) :] if polished.startswith(raw) else ""
        if suffix:
            yield {"type": "token", "text": suffix}
        session.append("user", request.message)
        session.append("assistant", polished)
        yield {
            "type": "done",
            "answer": polished,
            "policy": "allow",
            "session_id": session.session_id,
        }
