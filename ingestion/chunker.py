"""
Splits RawDocuments into Chunks using four strategies:
  fixed       — token-count windows with overlap
  sentence    — split on sentence boundaries, respect max size
  semantic    — paragraph-level grouping by topic proximity
  code_aware  — never breaks inside a fenced code block
"""

import re
from typing import Callable

from config.settings import ChunkStrategy, settings
from ingestion.models import Chunk, RawDocument


class Chunker:
    def __init__(
        self,
        strategy: ChunkStrategy | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.strategy = strategy or settings.chunk_strategy
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self._dispatch: dict[ChunkStrategy, Callable] = {
            ChunkStrategy.FIXED: self._fixed,
            ChunkStrategy.SENTENCE: self._sentence,
            ChunkStrategy.SEMANTIC: self._semantic,
            ChunkStrategy.CODE_AWARE: self._code_aware,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        fn = self._dispatch[self.strategy]
        raw_chunks = fn(doc.content)
        return [
            Chunk(
                text=text,
                source_url=doc.url,
                title=doc.title,
                section=doc.section,
                chunk_index=i,
                metadata={
                    "strategy": self.strategy.value,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            )
            for i, text in enumerate(raw_chunks)
            if text.strip()
        ]

    def chunk_many(self, docs: list[RawDocument]) -> list[Chunk]:
        result: list[Chunk] = []
        for doc in docs:
            result.extend(self.chunk(doc))
        return result

    # ------------------------------------------------------------------
    # Strategy: FIXED (word-level windows)
    # ------------------------------------------------------------------

    def _fixed(self, text: str) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ------------------------------------------------------------------
    # Strategy: SENTENCE
    # ------------------------------------------------------------------

    def _sentence(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            words = len(sent.split())
            if current_len + words > self.chunk_size and current:
                chunks.append(" ".join(current))
                # keep overlap: last few sentences
                overlap_words = 0
                overlap_sents: list[str] = []
                for s in reversed(current):
                    overlap_words += len(s.split())
                    overlap_sents.insert(0, s)
                    if overlap_words >= self.chunk_overlap:
                        break
                current = overlap_sents
                current_len = overlap_words

            current.append(sent)
            current_len += words

        if current:
            chunks.append(" ".join(current))

        return chunks

    # ------------------------------------------------------------------
    # Strategy: SEMANTIC (paragraph-based grouping)
    # ------------------------------------------------------------------

    def _semantic(self, text: str) -> list[str]:
        # Split on blank lines or section headers
        paragraphs = re.split(r"\n{2,}|(?=###\s)", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for para in paragraphs:
            wlen = len(para.split())
            if current_len + wlen > self.chunk_size and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            current_parts.append(para)
            current_len += wlen

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    # ------------------------------------------------------------------
    # Strategy: CODE_AWARE
    # ------------------------------------------------------------------

    def _code_aware(self, text: str) -> list[str]:
        """
        Split text while treating fenced code blocks as atomic units —
        a chunk boundary is never inserted inside ``` ... ```.
        """
        segments = self._split_preserve_code(text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for seg in segments:
            seg_len = len(seg.split())
            # If a single code block is larger than chunk_size, emit it alone
            if seg_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

            if current_len + seg_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

            current.append(seg)
            current_len += seg_len

        if current:
            chunks.append(" ".join(current))

        return chunks

    @staticmethod
    def _split_preserve_code(text: str) -> list[str]:
        """Return list of alternating prose/code segments."""
        pattern = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
        parts = pattern.split(text)
        return [p for p in parts if p.strip()]
