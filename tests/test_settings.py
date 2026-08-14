from config.settings import Settings, reset_settings_cache
from config.sources import allowed_hostnames, official_url_for_source_name


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    reset_settings_cache()
    settings = Settings(_env_file=None)
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_dimension == 1024
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333
    assert settings.retrieval_confidence_threshold == 0.35
    assert settings.drinkaware_url.startswith("https://")
    assert settings.niaaa_url.startswith("https://")
    assert settings.age_gate_enabled is True
    assert settings.minimum_legal_drinking_age >= 18


def test_embedding_provider_validation():
    settings = Settings(embedding_provider="BAAI", _env_file=None)
    assert settings.embedding_provider == "baai"


def test_cors_origins_list():
    settings = Settings(
        cors_allowed_origins="http://localhost:8501, http://127.0.0.1:8501",
        _env_file=None,
    )
    assert "http://localhost:8501" in settings.cors_origins_list


def test_official_brand_url():
    assert "absolut" in official_url_for_source_name("Absolut").lower()


def test_allowlist_contains_pernod_and_brands():
    hosts = allowed_hostnames()
    assert "pernod-ricard.com" in hosts
    assert "jamesonwhiskey.com" in hosts
    assert "theglenlivet.com" in hosts
