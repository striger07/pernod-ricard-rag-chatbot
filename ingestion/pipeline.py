"""End-to-end crawl → chunk → embed → Qdrant indexing pipeline."""

from __future__ import annotations

import argparse
import asyncio
from typing import Optional, Sequence

from config.settings import Settings, get_settings
from ingestion.chunker import SemanticChunker
from ingestion.crawler import KnowledgeCrawler
from ingestion.embedder import BGEEmbedder
from ingestion.models import CrawledDocument, TextChunk
from retrieval.bm25 import BM25Index
from utils.logging import configure_logging, get_logger
from vectorstore.qdrant_client import QdrantStore

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates ingestion while preserving metadata through every stage."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        crawler: Optional[KnowledgeCrawler] = None,
        chunker: Optional[SemanticChunker] = None,
        embedder: Optional[BGEEmbedder] = None,
        store: Optional[QdrantStore] = None,
        bm25: Optional[BM25Index] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.crawler = crawler or KnowledgeCrawler(self.settings)
        self.chunker = chunker or SemanticChunker(self.settings)
        self.embedder = embedder or BGEEmbedder(self.settings)
        self.store = store or QdrantStore(self.settings)
        self.bm25 = bm25 or BM25Index(self.settings)

    async def crawl(self, urls: Optional[Sequence[str]] = None) -> list[CrawledDocument]:
        documents = await self.crawler.crawl(urls)
        logger.info("pipeline_crawl_complete", documents=len(documents))
        return documents

    def chunk(self, documents: Sequence[CrawledDocument], *, semantic: bool = True) -> list[TextChunk]:
        embedder = self.embedder if semantic else None
        return self.chunker.chunk_documents(documents, embedder=embedder)

    def embed(self, chunks: Sequence[TextChunk]) -> list[TextChunk]:
        return self.embedder.embed_chunks(chunks)

    def index(self, chunks: Sequence[TextChunk], *, recreate: bool = False) -> int:
        if recreate:
            self.store.recreate_collection(vector_size=self.settings.embedding_dimension)
        else:
            self.store.ensure_collection(vector_size=self.settings.embedding_dimension)
        upserted = self.store.upsert_chunks(chunks)
        if recreate:
            self.bm25.build(chunks)
        else:
            self.bm25.update(chunks, persist=True)
        logger.info("bm25_synchronized", size=self.bm25.size, upserted=upserted)
        return upserted

    async def run(
        self,
        urls: Optional[Sequence[str]] = None,
        *,
        recreate: bool = False,
        semantic_chunking: bool = True,
    ) -> dict[str, int]:
        documents = await self.crawl(urls)
        chunks = self.chunk(documents, semantic=semantic_chunking)
        embedded = self.embed(chunks)
        upserted = self.index(embedded, recreate=recreate)
        summary = {
            "documents": len(documents),
            "chunks": len(chunks),
            "upserted": upserted,
        }
        logger.info("pipeline_complete", **summary)
        return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pernod Ricard RAG ingestion pipeline")
    parser.add_argument("--crawl-only", action="store_true", help="Crawl without indexing")
    parser.add_argument("--index-only", action="store_true", help="Index previously crawled synthetic/local data")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the Qdrant collection")
    parser.add_argument("--no-semantic", action="store_true", help="Chunk without embedding similarity splits")
    parser.add_argument("--url", action="append", dest="urls", default=None, help="Seed URL (repeatable)")
    return parser.parse_args(argv)


async def async_main(argv: Optional[Sequence[str]] = None) -> dict[str, int]:
    settings = get_settings()
    configure_logging(settings)
    args = parse_args(argv)
    pipeline = IngestionPipeline(settings=settings)
    semantic = not args.no_semantic
    if args.crawl_only:
        documents = await pipeline.crawl(args.urls)
        return {"documents": len(documents), "chunks": 0, "upserted": 0}
    if args.index_only:
        from ingestion.crawler import load_synthetic_documents

        documents = load_synthetic_documents(settings.synthetic_data_dir)
        chunks = pipeline.chunk(documents, semantic=semantic)
        embedded = pipeline.embed(chunks)
        upserted = pipeline.index(embedded, recreate=args.recreate)
        summary = {"documents": len(documents), "chunks": len(chunks), "upserted": upserted}
        logger.info("pipeline_index_only_complete", **summary)
        return summary
    return await pipeline.run(args.urls, recreate=args.recreate, semantic_chunking=semantic)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, int]:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
