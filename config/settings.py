"""Application configuration via environment variables."""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Pernod Ricard RAG chatbot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = Field(default="pernod-ricard-rag")
    app_env: str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = Field(default="INFO")
    cors_allowed_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501"
    )
    admin_api_token: str = Field(default="")

    # --- Groq (OpenAI-compatible) ---
    groq_api_key: str = Field(default="")
    groq_api_base: str = Field(default="https://api.groq.com/openai/v1")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_eval_model: str = Field(default="llama-3.1-8b-instant")
    groq_eval_requests_per_second: float = Field(default=0.4, ge=0.01, le=5.0)
    groq_timeout_seconds: float = Field(default=60.0, ge=1.0)
    groq_max_tokens: int = Field(default=1024, ge=16)
    groq_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- OpenAI (optional embedding fallback) ---
    openai_api_key: str = Field(default="")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_embedding_dimension: int = Field(default=1536, ge=1)

    # --- Embeddings (primary: BAAI/bge-m3) ---
    embedding_provider: str = Field(default="baai")
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_dimension: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    embedding_max_retries: int = Field(default=5, ge=1, le=20)
    embedding_device: str = Field(default="cpu")

    # --- Qdrant ---
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_grpc_port: int = Field(default=6334, ge=1, le=65535)
    qdrant_api_key: Optional[str] = Field(default=None)
    qdrant_https: bool = Field(default=False)
    qdrant_collection: str = Field(default="pernod_ricard_chunks")
    qdrant_timeout_seconds: int = Field(default=30, ge=1)

    # --- Retrieval ---
    retrieval_confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    dense_top_k: int = Field(default=20, ge=1)
    bm25_top_k: int = Field(default=20, ge=1)
    final_top_k: int = Field(default=8, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    mmr_candidates: int = Field(default=30, ge=1)

    # --- BM25 ---
    bm25_index_path: str = Field(default="data/indexes/bm25.pkl")
    bm25_k1: float = Field(default=1.5, ge=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)

    # --- Chunking ---
    chunk_max_chars: int = Field(default=1800, ge=50)
    chunk_min_chars: int = Field(default=200, ge=1)
    chunk_overlap_chars: int = Field(default=220, ge=0)
    semantic_similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    # --- Crawler ---
    crawl_max_pages: int = Field(default=40, ge=1)
    crawl_max_depth: int = Field(default=2, ge=0)
    crawl_request_timeout_seconds: int = Field(default=30, ge=1)
    crawl_max_retries: int = Field(default=3, ge=1)
    crawl_retry_backoff_seconds: float = Field(default=1.5, ge=0.1)
    crawl_user_agent: str = Field(
        default="PernodRicardRAGBot/1.0 (+https://www.pernod-ricard.com)"
    )
    crawl_rate_limit_seconds: float = Field(default=1.0, ge=0.0)
    crawl_use_synthetic_fallback: bool = Field(default=True)
    synthetic_data_dir: str = Field(default="data/synthetic")

    # --- Age gate ---
    age_gate_enabled: bool = Field(default=True)
    minimum_legal_drinking_age: int = Field(default=18, ge=18, le=21)
    age_gate_strict_server_enforcement: bool = Field(default=True)
    age_verification_header: str = Field(default="X-Age-Verified")

    # --- Responsible drinking ---
    drinkaware_url: str = Field(default="https://www.drinkaware.co.uk")
    niaaa_url: str = Field(default="https://www.niaaa.nih.gov")

    # --- Frontend ---
    backend_url: str = Field(default="http://127.0.0.1:8000")

    # --- RAGAS ---
    ragas_faithfulness_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    ragas_context_precision_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    ragas_answer_relevancy_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    ragas_output_dir: str = Field(default="data/indexes")
    ragas_use_library_metrics: bool = Field(default=False)

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"baai", "openai"}
        if normalized not in allowed:
            raise ValueError(f"embedding_provider must be one of {sorted(allowed)}")
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalized

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def normalize_groq_api_key(cls, value: Optional[str]) -> str:
        if value is None:
            return ""
        key = str(value).strip().strip('"').strip("'")
        second = key.find("gsk_", 1)
        if second != -1:
            key = key[second:]
        return key

    @field_validator("qdrant_api_key", mode="before")
    @classmethod
    def empty_api_key_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() in {"prod", "production"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests)."""
    get_settings.cache_clear()
