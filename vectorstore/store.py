"""
ChromaDB vector store with local sentence-transformer embeddings.
No API key required — embeddings run entirely on device.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import settings
from ingestion.models import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._collection_name = collection_name or settings.chroma_collection_name
        self._model_name = embedding_model or settings.local_embedding_model

        logger.info("Loading embedding model: %s", self._model_name)
        self._encoder = SentenceTransformer(self._model_name)

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' ready — %d docs stored",
            self._collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed chunks and upsert into ChromaDB. Safe to call multiple times."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        # Stable ID: re-running the pipeline won't create duplicates
        ids = [f"{c.source_url}::{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "source_url": c.source_url,
                "title": c.title,
                "section": c.section,
                "chunk_index": c.chunk_index,
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]

        logger.info("Embedding %d chunks with %s…", len(chunks), self._model_name)
        embeddings = self._encoder.encode(texts, show_progress_bar=True).tolist()

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(
            "Upserted %d chunks — collection total: %d",
            len(chunks),
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search_similar(self, query: str, top_k: Optional[int] = None) -> list[Chunk]:
        """Return the top-k most similar chunks for a query string."""
        k = top_k or settings.top_k_results
        total = self._collection.count()
        if total == 0:
            logger.warning("Collection is empty — run add_chunks() first")
            return []

        query_embedding = self._encoder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                Chunk(
                    text=text,
                    source_url=meta.get("source_url", ""),
                    title=meta.get("title", ""),
                    section=meta.get("section", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    metadata={**meta, "distance": distance},
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of chunks currently stored in the collection."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete and recreate the collection (irreversible)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' cleared", self._collection_name)
