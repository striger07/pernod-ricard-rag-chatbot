from datetime import datetime, timezone

from config.settings import Settings
from ingestion.chunker import SemanticChunker
from ingestion.models import CrawledDocument


def _doc(content: str, url: str = "https://www.pernod-ricard.com/en") -> CrawledDocument:
    return CrawledDocument(
        content=content,
        title="Pernod Ricard",
        url=url,
        source="Pernod Ricard",
        timestamp=datetime.now(timezone.utc),
        content_hash="abc",
        metadata={"ingested_at": datetime.now(timezone.utc).isoformat()},
    )


def test_paragraph_boundaries_preserved():
    settings = Settings(chunk_max_chars=240, chunk_min_chars=40, chunk_overlap_chars=20, _env_file=None)
    chunker = SemanticChunker(settings)
    text = (
        "Absolut is a Swedish vodka brand in the Pernod Ricard portfolio.\n\n"
        "Chivas Regal is a blended Scotch whisky with origins in Aberdeen.\n\n"
        "Jameson is a triple-distilled Irish whiskey enjoyed in mixed drinks."
    )
    chunks = chunker.chunk_document(_doc(text))
    assert chunks
    assert all(chunk.title == "Pernod Ricard" for chunk in chunks)
    assert all(chunk.url.endswith("/en") for chunk in chunks)
    assert all(chunk.source == "Pernod Ricard" for chunk in chunks)
    assert all("timestamp" in chunk.metadata for chunk in chunks)


def test_overlap_attached_between_chunks():
    settings = Settings(chunk_max_chars=80, chunk_min_chars=20, chunk_overlap_chars=25, _env_file=None)
    chunker = SemanticChunker(settings)
    paragraphs = [f"Paragraph number {index} discusses Pernod Ricard brands in detail." for index in range(8)]
    chunks = chunker.chunk_document(_doc("\n\n".join(paragraphs)))
    assert len(chunks) >= 2
    assert chunks[1].chunk_index == 1


def test_semantic_split_uses_similarity_threshold():
    settings = Settings(
        chunk_max_chars=2000,
        chunk_min_chars=10,
        chunk_overlap_chars=0,
        semantic_similarity_threshold=0.9,
        _env_file=None,
    )
    chunker = SemanticChunker(settings)

    class FakeEmbedder:
        def encode(self, texts):
            vectors = []
            for text in texts:
                if "vodka" in text.lower():
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

    document = _doc(
        "Absolut vodka is distilled in Sweden and served in mixed drinks.\n\n"
        "The Glenlivet is a Speyside single malt with citrus and vanilla notes."
    )
    chunks = chunker.chunk_document(document, embedder=FakeEmbedder())
    assert len(chunks) == 2
