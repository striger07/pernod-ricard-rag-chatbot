from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from api.main import create_app
from api.sessions import SessionStore
from config.settings import Settings, get_settings
from guardrails.orchestrator import GuardrailOrchestrator
from rag.engine import RagEngine
from rag.hallucination import INSUFFICIENT_INFORMATION, HallucinationBoundary
from retrieval.models import RetrievalResult, RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        content="Absolut is a Swedish vodka in the Pernod Ricard portfolio.",
        title="Absolut",
        url="https://www.absolut.com/",
        source="Absolut",
        timestamp=datetime.now(timezone.utc).isoformat(),
        chunk_id="abs-1",
        metadata={"title": "Absolut", "url": "https://www.absolut.com/"},
        dense_score=0.9,
        rrf_score=0.03,
        mmr_score=0.5,
        confidence_score=0.8,
    )


def _settings() -> Settings:
    return Settings(
        age_gate_enabled=True,
        admin_api_token="test-admin",
        retrieval_confidence_threshold=0.35,
        groq_api_key="test",
        app_env="development",
        _env_file=None,
    )


def _engine(confidence: float = 0.8) -> RagEngine:
    settings = _settings()
    retriever = MagicMock()
    retriever.retrieve.return_value = RetrievalResult(
        query="absolut",
        chunks=[_chunk()],
        confidence=confidence,
    )
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="Absolut is a Swedish vodka produced in Ahus.")
    return RagEngine(
        settings=settings,
        guardrails=GuardrailOrchestrator(settings),
        retriever=retriever,
        llm=llm,
        boundary=HallucinationBoundary(settings),
        sessions=SessionStore(),
    )


def _client(engine: RagEngine | None = None) -> TestClient:
    engine = engine or _engine()
    app = create_app(engine=engine)
    app.dependency_overrides[get_settings] = _settings
    return TestClient(app)


def test_health_endpoint():
    with _client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "age_gate_enabled" in body
        assert "drinkaware_url" in body


def test_chat_requires_age_gate():
    with _client() as client:
        response = client.post("/chat", json={"message": "Tell me about Absolut vodka"})
        assert response.status_code == 403
        assert response.json()["blocked"] is True
        assert response.json()["policy"] == "age_gate"


def test_chat_pricing_refusal():
    with _client() as client:
        response = client.post(
            "/chat",
            json={"message": "What is the price of Absolut?", "age_verified": True, "declared_age": 30},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["blocked"] is True
        assert body["policy"] == "pricing"
        assert body["redirect_url"]


def test_chat_success_with_citations():
    with _client() as client:
        response = client.post(
            "/chat",
            json={"message": "Tell me about Absolut vodka", "age_verified": True, "declared_age": 28},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["blocked"] is False
        assert "Absolut" in body["answer"]
        assert body["citations"][0]["url"] == "https://www.absolut.com/"
        assert body["confidence"] == 0.8


def test_chat_hallucination_boundary_exact_text():
    with _client(_engine(confidence=0.1)) as client:
        response = client.post(
            "/chat",
            json={"message": "Tell me about Absolut vodka", "age_verified": True},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == INSUFFICIENT_INFORMATION


def test_ingest_requires_admin_token():
    engine = _engine()
    app = create_app(engine=engine)
    app.dependency_overrides[get_settings] = _settings
    pipeline = MagicMock()
    pipeline.crawl = AsyncMock(return_value=[object(), object()])
    app.state.ingestion = pipeline
    with TestClient(app) as client:
        denied = client.post("/ingest", json={"crawl_only": True})
        assert denied.status_code == 401
        response = client.post(
            "/ingest",
            json={"crawl_only": True},
            headers={"X-Admin-Token": "test-admin"},
        )
        assert response.status_code == 200
        assert response.json()["documents"] == 2
