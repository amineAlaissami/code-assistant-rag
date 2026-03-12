"""
IngestionPipeline — orchestrates scrape → chunk → return.
The pipeline does not write to the vector store; that belongs to the
vectorstore module. It returns a list of Chunks ready to be embedded.
"""

import logging
from pathlib import Path

from config.settings import ChunkStrategy, settings
from ingestion.chunker import Chunker
from ingestion.models import Chunk, RawDocument
from ingestion.scraper import DocScraper

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        seed_urls: list[str] | None = None,
        strategy: ChunkStrategy | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        scraper_delay: float = 0.5,
        max_pages: int | None = None,
    ):
        self.scraper = DocScraper(
            seed_urls=seed_urls,
            delay=scraper_delay,
            max_pages=max_pages,
        )
        self.chunker = Chunker(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[Chunk]:
        """Scrape all seed URLs, chunk every document, return all chunks."""
        logger.info("Starting ingestion pipeline (strategy=%s)", self.chunker.strategy.value)

        docs = self._scrape()
        if not docs:
            logger.warning("No documents scraped — pipeline produced 0 chunks")
            return []

        chunks = self._chunk(docs)
        logger.info(
            "Pipeline complete: %d docs → %d chunks",
            len(docs),
            len(chunks),
        )
        return chunks

    def run_from_files(self, directory: str | Path) -> list[Chunk]:
        """
        Load pre-downloaded HTML files from a local directory instead of
        hitting the network.  Useful for offline development and testing.
        """
        directory = Path(directory)
        html_files = list(directory.glob("*.html"))
        if not html_files:
            raise FileNotFoundError(f"No .html files found in {directory}")

        docs: list[RawDocument] = []
        for path in html_files:
            html = path.read_text(encoding="utf-8", errors="replace")
            doc = self.scraper._parse(url=path.name, html=html)
            if doc:
                docs.append(doc)

        logger.info("Loaded %d docs from %s", len(docs), directory)
        return self._chunk(docs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scrape(self) -> list[RawDocument]:
        docs: list[RawDocument] = []
        for doc in self.scraper.iter_documents():
            docs.append(doc)
            logger.debug("Scraped: %s (%d chars)", doc.url, len(doc.content))
        return docs

    def _chunk(self, docs: list[RawDocument]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in docs:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
            logger.debug(
                "  %s → %d chunks (strategy=%s)",
                doc.title,
                len(chunks),
                self.chunker.strategy.value,
            )
        return all_chunks
