from unittest.mock import MagicMock

from config.settings import Settings
from retrieval.models import RetrievedChunk
from retrieval.pipeline import HybridRetrievalPipeline, normalize_query


def test_normalize_query():
    assert normalize_query("  Absolut   vodka\n") == "Absolut vodka"


def test_empty_query_zero_confidence():
    pipeline = HybridRetrievalPipeline(settings=Settings(_env_file=None))
    result = pipeline.retrieve("   ")
    assert result.chunks == []
    assert result.confidence == 0.0


def test_hybrid_pipeline_scores_and_metadata():
    settings = Settings(
        final_top_k=2,
        mmr_candidates=5,
        rrf_k=60,
        retrieval_confidence_threshold=0.35,
        _env_file=None,
    )
    dense_hit = RetrievedChunk(
        content="Absolut is a Swedish vodka brand in the Pernod Ricard portfolio.",
        title="Absolut",
        url="https://www.absolut.com/",
        source="Absolut",
        timestamp="2026-01-01T00:00:00+00:00",
        chunk_id="abs-1",
        metadata={"title": "Absolut", "url": "https://www.absolut.com/", "source": "Absolut"},
        embedding=[1.0, 0.0],
        dense_score=0.88,
    )
    bm25_hit = RetrievedChunk(
        content="Absolut is a Swedish vodka brand in the Pernod Ricard portfolio.",
        title="Absolut",
        url="https://www.absolut.com/",
        source="Absolut",
        timestamp="2026-01-01T00:00:00+00:00",
        chunk_id="abs-1",
        metadata={"title": "Absolut", "url": "https://www.absolut.com/"},
        bm25_score=4.2,
    )
    extra = RetrievedChunk(
        content="Beefeater is a London Dry gin distilled in London.",
        title="Beefeater",
        url="https://www.beefeatergin.com/",
        source="Beefeater",
        timestamp="2026-01-01T00:00:00+00:00",
        chunk_id="beef-1",
        metadata={"title": "Beefeater"},
        embedding=[0.0, 1.0],
        dense_score=0.41,
    )

    embedder = MagicMock()
    embedder.encode.return_value = [[1.0, 0.0]]
    dense = MagicMock()
    dense.search.return_value = [dense_hit, extra]
    bm25 = MagicMock()
    bm25.search.return_value = [bm25_hit]

    pipeline = HybridRetrievalPipeline(
        settings=settings,
        dense=dense,
        bm25=bm25,
        embedder=embedder,
    )
    result = pipeline.retrieve("Where is Absolut vodka from?", query_vector=[1.0, 0.0])
    assert result.dense_count == 2
    assert result.bm25_count == 1
    assert result.fused_count >= 2
    assert 0.0 <= result.confidence <= 1.0
    lead = result.chunks[0]
    assert lead.content
    assert lead.title
    assert lead.url
    assert lead.source
    assert lead.metadata
    assert lead.rrf_score is not None
    assert lead.mmr_score is not None
    assert lead.confidence_score is not None
    assert lead.chunk_id == "abs-1"
    assert lead.dense_score == 0.88
    assert lead.bm25_score == 4.2
