"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routes import router
from api.sessions import SessionStore
from config.settings import get_settings
from guardrails.orchestrator import GuardrailOrchestrator
from rag.engine import RagEngine
from rag.hallucination import HallucinationBoundary
from rag.llm_service import GrokLLMService
from retrieval.pipeline import HybridRetrievalPipeline
from utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

_MAX_BODY_BYTES = 262144


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    if getattr(application.state, "engine", None) is None:
        sessions = SessionStore()
        application.state.engine = RagEngine(
            settings=settings,
            guardrails=GuardrailOrchestrator(settings),
            retriever=HybridRetrievalPipeline(settings),
            llm=GrokLLMService(settings),
            boundary=HallucinationBoundary(settings),
            sessions=sessions,
        )
    logger.info("application_startup", app=settings.app_name, env=settings.app_env)
    yield
    logger.info("application_shutdown")


def create_app(*, engine: RagEngine | None = None) -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Pernod Ricard RAG Chatbot",
        version="0.3.0",
        lifespan=lifespan,
    )
    if engine is not None:
        application.state.engine = engine
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def limit_request_body(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > _MAX_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"error": {"code": "payload_too_large", "message": "Request body is too large"}},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"code": "invalid_content_length", "message": "Invalid Content-Length"}},
                )
        return await call_next(request)

    @application.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "http_error", "message": message}},
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "Invalid request payload"}},
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, _exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "An unexpected error occurred"}},
        )

    application.include_router(router)
    return application


app = create_app()
