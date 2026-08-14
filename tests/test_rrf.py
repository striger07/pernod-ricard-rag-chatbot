from retrieval.fusion import ReciprocalRankFusion, rrf_score
from retrieval.models import RetrievedChunk


def _chunk(chunk_id: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        title=chunk_id,
        url=f"https://www.pernod-ricard.com/{chunk_id}",
        source="Pernod Ricard",
        chunk_id=chunk_id,
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={"title": chunk_id, "url": f"https://www.pernod-ricard.com/{chunk_id}"},
    )


def test_rrf_formula_rank_one_with_k_60():
    assert rrf_score(1, 60) == 1.0 / 61.0


def test_rrf_prefers_items_in_both_lists():
    dense = [
        _chunk("a", "Absolut vodka is produced in Sweden."),
        _chunk("b", "Chivas Regal is a blended Scotch whisky."),
    ]
    bm25 = [
        _chunk("b", "Chivas Regal is a blended Scotch whisky."),
        _chunk("c", "Jameson is a triple-distilled Irish whiskey."),
    ]
    fused = ReciprocalRankFusion(k=60).fuse(dense, bm25)
    assert fused[0].chunk_id == "b"
    assert fused[0].rrf_score == rrf_score(2, 60) + rrf_score(1, 60)
    b = next(item for item in fused if item.chunk_id == "b")
    a = next(item for item in fused if item.chunk_id == "a")
    c = next(item for item in fused if item.chunk_id == "c")
    assert b.rrf_score > a.rrf_score
    assert b.rrf_score > c.rrf_score


def test_rrf_is_not_concatenation():
    dense = [_chunk("a", "one"), _chunk("b", "two")]
    bm25 = [_chunk("b", "two"), _chunk("a", "one")]
    fused = ReciprocalRankFusion(k=60).fuse(dense, bm25)
    ids = [item.chunk_id for item in fused]
    assert len(ids) == 2
    assert set(ids) == {"a", "b"}
