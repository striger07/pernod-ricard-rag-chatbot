from datetime import datetime, timezone
from unittest.mock import MagicMock

from config.settings import Settings
from ingestion.models import TextChunk
from vectorstore.qdrant_client import QdrantStore, chunk_point_id


def _chunk() -> TextChunk:
    return TextChunk(
        content="Chivas Regal is a blended Scotch whisky.",
        title="Chivas Regal",
        url="https://www.chivas.com/",
        source="Chivas Regal",
        timestamp=datetime.now(timezone.utc),
        document_id="doc-chivas",
        chunk_index=0,
        content_hash="hash-1",
        embedding=[0.1] * 4,
        metadata={"title": "Chivas Regal", "url": "https://www.chivas.com/"},
    )


def test_payload_mapping_contains_required_fields():
    payload = _chunk().to_qdrant_payload()
    for key in ("content", "title", "url", "source", "timestamp", "document_id", "chunk_index"):
        assert key in payload
        assert payload[key] not in (None, "")


def test_point_id_is_stable():
    chunk = _chunk()
    assert chunk_point_id(chunk) == chunk_point_id(chunk)


def test_health_check_ok():
    settings = Settings(qdrant_host="localhost", qdrant_port=6333, qdrant_collection="demo", _env_file=None)
    client = MagicMock()
    collection = MagicMock()
    collection.name = "demo"
    client.get_collections.return_value.collections = [collection]
    client.get_collection.return_value.points_count = 12
    store = QdrantStore(settings=settings, client=client)
    health = store.health_check()
    assert health["status"] == "ok"
    assert health["host"] == "localhost"
    assert health["port"] == 6333
    assert health["collection_exists"] is True
    assert health["points_count"] == 12


def test_health_check_error():
    settings = Settings(_env_file=None)
    client = MagicMock()
    client.get_collections.side_effect = OSError("connection refused")
    store = QdrantStore(settings=settings, client=client)
    health = store.health_check()
    assert health["status"] == "error"


def test_upsert_requires_embeddings():
    settings = Settings(embedding_dimension=4, _env_file=None)
    client = MagicMock()
    client.get_collections.return_value.collections = []
    store = QdrantStore(settings=settings, client=client)
    chunk = _chunk()
    chunk.embedding = None
    try:
        store.upsert_chunks([chunk])
        raised = False
    except Exception:
        raised = True
    assert raised
