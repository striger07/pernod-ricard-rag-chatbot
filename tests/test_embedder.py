from datetime import datetime, timezone

from config.settings import Settings
from ingestion.embedder import BGEEmbedder
from ingestion.models import TextChunk


def test_embedder_batches_and_retries(monkeypatch):
    settings = Settings(embedding_batch_size=2, embedding_max_retries=3, embedding_dimension=3, _env_file=None)
    embedder = BGEEmbedder(settings)
    calls = {"n": 0}

    class FakeModel:
        def encode(self, batch, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("temporary failure")
            import numpy as np

            return np.ones((len(batch), 3), dtype=float)

    monkeypatch.setattr(embedder, "_load_model", lambda: FakeModel())
    vectors = embedder.encode(["one", "two", "three"])
    assert len(vectors) == 3
    assert len(vectors[0]) == 3
    assert calls["n"] >= 2


def test_embed_chunks_attaches_vectors(monkeypatch):
    settings = Settings(embedding_batch_size=8, embedding_dimension=2, _env_file=None)
    embedder = BGEEmbedder(settings)

    class FakeModel:
        def encode(self, batch, **_kwargs):
            import numpy as np

            return np.array([[0.1, 0.2] for _ in batch], dtype=float)

    monkeypatch.setattr(embedder, "_load_model", lambda: FakeModel())
    chunk = TextChunk(
        content="Absolut vodka is produced in Sweden.",
        title="Absolut",
        url="https://www.absolut.com/en/",
        source="Absolut",
        timestamp=datetime.now(timezone.utc),
        document_id="doc-1",
        chunk_index=0,
    )
    embedded = embedder.embed_chunks([chunk])
    assert embedded[0].embedding == [0.1, 0.2]
    assert embedded[0].metadata["embedding_model"] == settings.embedding_model


def test_encode_empty():
    embedder = BGEEmbedder(Settings(_env_file=None))
    assert embedder.encode([]) == []
