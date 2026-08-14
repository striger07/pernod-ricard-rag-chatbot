from pathlib import Path

import pytest

from config.settings import Settings
from config.sources import allowed_hostnames
from ingestion.cleaner import html_to_text, markdown_to_text, normalize_text
from ingestion.crawler import CrawlError, KnowledgeCrawler, load_synthetic_documents, validate_public_url


def test_html_strips_nav_and_scripts():
    html = """
    <html>
      <head><title>Jameson Irish Whiskey</title></head>
      <body>
        <nav>Home About</nav>
        <script>alert('x')</script>
        <main>
          <h1>Jameson</h1>
          <p>Jameson is a triple-distilled Irish whiskey in the Pernod Ricard portfolio.</p>
        </main>
        <footer>cookies</footer>
      </body>
    </html>
    """
    title, text = html_to_text(html)
    assert title == "Jameson Irish Whiskey"
    assert "Jameson is a triple-distilled" in text
    assert "alert" not in text
    assert "Home About" not in text


def test_normalize_text_preserves_paragraphs():
    text = normalize_text("Hello   world.\n\n\n\nNext   paragraph.")
    assert "Hello world." in text
    assert "\n\n" in text


def test_markdown_title_extraction():
    title, body = markdown_to_text("# Absolut Vodka\n\nAbsolut is a Swedish vodka.")
    assert title == "Absolut Vodka"
    assert "Swedish vodka" in body


def test_load_synthetic_documents():
    documents = load_synthetic_documents("data/synthetic")
    assert len(documents) >= 8
    absolut = next(doc for doc in documents if "Absolut" in doc.title)
    assert absolut.url.startswith("synthetic://")
    assert absolut.source
    assert absolut.timestamp.tzinfo is not None
    assert absolut.content_hash
    assert "ingested_at" in absolut.metadata


def test_validate_url_rejects_unknown_host():
    with pytest.raises(CrawlError):
        validate_public_url("https://evil.example/steal", allowed_hostnames())


def test_validate_url_rejects_non_http():
    with pytest.raises(CrawlError):
        validate_public_url("file:///etc/passwd", allowed_hostnames())


@pytest.mark.asyncio
async def test_crawler_falls_back_to_synthetic(tmp_path: Path):
    markdown = tmp_path / "sample.md"
    markdown.write_text("# Beefeater Gin\n\nBeefeater is a London Dry gin distilled in London.", encoding="utf-8")
    settings = Settings(
        synthetic_data_dir=str(tmp_path),
        crawl_use_synthetic_fallback=True,
        crawl_max_pages=2,
        crawl_rate_limit_seconds=0.0,
        _env_file=None,
    )
    crawler = KnowledgeCrawler(settings)
    documents = await crawler.crawl(urls=["https://blocked.example/page"])
    assert documents
    assert documents[0].metadata["crawl_method"] == "synthetic_fallback"
    assert "Beefeater" in documents[0].title
