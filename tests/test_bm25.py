from datetime import datetime, timezone

from config.settings import Settings
from ingestion.models import TextChunk
from retrieval.bm25 import BM25Index, tokenize


def _chunk(chunk_id: str, content: str, title: str) -> TextChunk:
    return TextChunk(
        content=content,
        title=title,
        url=f"https://www.pernod-ricard.com/{chunk_id}",
        source="Pernod Ricard",
        timestamp=datetime.now(timezone.utc),
        document_id="doc-1",
        chunk_index=0,
        chunk_id=chunk_id,
        metadata={"title": title, "url": f"https://www.pernod-ricard.com/{chunk_id}"},
    )


def test_tokenize_lowercases():
    assert tokenize("Absolut Vodka 2024") == ["absolut", "vodka", "2024"]


def test_bm25_persists_and_ranks(tmp_path):
    settings = Settings(bm25_index_path=str(tmp_path / "bm25.pkl"), bm25_top_k=5, _env_file=None)
    index = BM25Index(settings)
    index.build(
        [
            _chunk("abs", "Absolut is a Swedish vodka brand distilled in Ahus.", "Absolut"),
            _chunk("chv", "Chivas Regal is a blended Scotch whisky from Aberdeen.", "Chivas"),
            _chunk("gln", "The Glenlivet is a Speyside single malt Scotch whisky.", "Glenlivet"),
            _chunk("bee", "Beefeater is a London Dry gin distilled in London.", "Beefeater"),
        ]
    )
    assert (tmp_path / "bm25.pkl").exists()
    hits = index.search("Swedish vodka Absolut")
    assert hits
    assert hits[0].chunk_id == "abs"
    assert hits[0].bm25_score is not None and hits[0].bm25_score > 0
    assert hits[0].title == "Absolut"
    assert hits[0].url.startswith("https://")

    loaded = BM25Index(settings)
    loaded.load()
    again = loaded.search("Swedish vodka")
    assert again[0].chunk_id == "abs"


def test_bm25_update_adds_documents(tmp_path):
    settings = Settings(bm25_index_path=str(tmp_path / "bm25.pkl"), _env_file=None)
    index = BM25Index(settings)
    index.build(
        [
            _chunk("abs", "Absolut is a Swedish vodka brand.", "Absolut"),
            _chunk("chv", "Chivas Regal is a blended Scotch whisky.", "Chivas"),
            _chunk("gln", "The Glenlivet is a Speyside single malt.", "Glenlivet"),
        ]
    )
    index.update([_chunk("jam", "Jameson is a triple-distilled Irish whiskey.", "Jameson")])
    hits = index.search("Irish whiskey Jameson")
    assert hits[0].chunk_id == "jam"
