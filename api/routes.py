"""HTTP routes for health, chat, retrieval, and protected ingestion."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from api.dependencies import age_header, get_engine, require_admin
from api.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from config.settings import Settings, get_settings
from ingestion.pipeline import IngestionPipeline
from rag.engine import RagEngine
from rag.llm_service import LLMServiceError
from utils.logging import get_logger
from vectorstore.qdrant_client import QdrantStore

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    store = QdrantStore(settings)
    qdrant = store.health_check()
    status_value = "ok" if qdrant.get("status") == "ok" else "degraded"
    return {
        "status": status_value,
        "app": settings.app_name,
        "environment": settings.app_env,
        "qdrant": qdrant,
        "age_gate_enabled": settings.age_gate_enabled,
        "minimum_legal_drinking_age": settings.minimum_legal_drinking_age,
        "drinkaware_url": settings.drinkaware_url,
        "niaaa_url": settings.niaaa_url,
    }


@router.get("/ready")
def ready(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    store = QdrantStore(settings)
    qdrant = store.health_check()
    return {"ready": qdrant.get("status") == "ok", "qdrant": qdrant}


def _chat_payload(result: ChatResponse) -> dict[str, Any]:
    return result.model_dump()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    engine: RagEngine = Depends(get_engine),
    header_verified: bool = Depends(age_header),
) -> JSONResponse:
    try:
        result = await engine.chat(body, header_verified=header_verified)
    except LLMServiceError as exc:
        logger.error("route_chat_llm_error")
        raise HTTPException(status_code=502, detail="The language model is unavailable") from exc
    except Exception as exc:
        logger.error("route_chat_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail="Chat request failed") from exc
    status_code = status.HTTP_403_FORBIDDEN if result.policy == "age_gate" else status.HTTP_200_OK
    return JSONResponse(status_code=status_code, content=_chat_payload(result))


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    engine: RagEngine = Depends(get_engine),
    header_verified: bool = Depends(age_header),
) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        try:
            async for payload in engine.chat_stream(body, header_verified=header_verified):
                yield f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"
        except LLMServiceError:
            logger.error("route_stream_llm_error")
            error = {"type": "error", "message": "The language model is unavailable"}
            yield f"data: {json.dumps(error)}\n\n"
        except Exception:
            logger.error("route_stream_failed")
            error = {"type": "error", "message": "Streaming chat failed"}
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    body: RetrieveRequest,
    engine: RagEngine = Depends(get_engine),
    header_verified: bool = Depends(age_header),
) -> JSONResponse:
    session = engine.sessions.get_or_create(body.session_id)
    decision = engine.guardrails.inspect_query(
        body.query,
        age_verified=body.age_verified,
        declared_age=body.declared_age,
        session_verified=session.age_verified,
        header_verified=header_verified,
    )
    if decision.blocked:
        status_code = status.HTTP_403_FORBIDDEN if decision.policy == "age_gate" else status.HTTP_200_OK
        payload = RetrieveResponse(
            query=body.query,
            blocked=True,
            policy=decision.policy,
            confidence=0.0,
            chunks=[],
            session_id=session.session_id,
            answer=decision.message,
        )
        return JSONResponse(status_code=status_code, content=payload.model_dump())
    session.age_verified = True
    try:
        result = engine.retrieve(body.query, top_k=body.top_k)
    except Exception as exc:
        logger.error("route_retrieve_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Retrieval failed") from exc
    payload = RetrieveResponse(
        query=result.query,
        blocked=False,
        policy="allow",
        confidence=result.confidence,
        chunks=[chunk.to_dict() for chunk in result.chunks],
        session_id=session.session_id,
    )
    return JSONResponse(content=payload.model_dump())


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin)])
async def ingest(
    body: IngestRequest,
    request: Request,
) -> IngestResponse:
    settings = get_settings()
    pipeline: IngestionPipeline | None = getattr(request.app.state, "ingestion", None)
    if pipeline is None:
        pipeline = IngestionPipeline(settings)
        request.app.state.ingestion = pipeline
    logger.info("ingest_started", recreate=body.recreate, crawl_only=body.crawl_only)
    try:
        if body.crawl_only:
            documents = await pipeline.crawl(body.urls)
            return IngestResponse(documents=len(documents), chunks=0, upserted=0)
        summary = await pipeline.run(body.urls, recreate=body.recreate)
    except Exception as exc:
        logger.error("ingest_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Ingestion failed") from exc
    return IngestResponse(
        documents=int(summary.get("documents", 0)),
        chunks=int(summary.get("chunks", 0)),
        upserted=int(summary.get("upserted", 0)),
    )
