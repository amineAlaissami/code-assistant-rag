"""
Scrapes Python documentation pages from docs.python.org.
Extracts clean text, preserving code blocks and section structure.
"""

import time
import logging
from typing import Iterator
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from ingestion.models import RawDocument

logger = logging.getLogger(__name__)

# Top-level Python doc sections to index
DEFAULT_SEED_URLS = [
    "https://docs.python.org/3/library/functions.html",
    "https://docs.python.org/3/library/stdtypes.html",
    "https://docs.python.org/3/library/exceptions.html",
    "https://docs.python.org/3/library/string.html",
    "https://docs.python.org/3/library/re.html",
    "https://docs.python.org/3/library/collections.html",
    "https://docs.python.org/3/library/itertools.html",
    "https://docs.python.org/3/library/functools.html",
    "https://docs.python.org/3/library/pathlib.html",
    "https://docs.python.org/3/library/os.html",
    "https://docs.python.org/3/library/os.path.html",
    "https://docs.python.org/3/library/io.html",
    "https://docs.python.org/3/library/json.html",
    "https://docs.python.org/3/library/datetime.html",
    "https://docs.python.org/3/library/typing.html",
    "https://docs.python.org/3/library/dataclasses.html",
    "https://docs.python.org/3/library/abc.html",
    "https://docs.python.org/3/library/contextlib.html",
    "https://docs.python.org/3/library/asyncio.html",
    "https://docs.python.org/3/tutorial/index.html",
]

_BASE = "https://docs.python.org/3/"


class DocScraper:
    def __init__(
        self,
        seed_urls: list[str] | None = None,
        delay: float = 0.5,
        timeout: int = 10,
        max_pages: int | None = None,
    ):
        self.seed_urls = seed_urls or DEFAULT_SEED_URLS
        self.delay = delay
        self.timeout = timeout
        self.max_pages = max_pages
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "code-assistant-rag/1.0 (educational)"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self) -> list[RawDocument]:
        docs = []
        for doc in self.iter_documents():
            docs.append(doc)
        return docs

    def iter_documents(self) -> Iterator[RawDocument]:
        seen: set[str] = set()
        queue = list(self.seed_urls)
        count = 0

        while queue:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)

            try:
                doc = self._fetch(url)
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s", url, exc)
                continue

            if doc is None:
                continue

            yield doc
            count += 1
            logger.info("[%d] scraped %s", count, url)

            if self.max_pages and count >= self.max_pages:
                break

            time.sleep(self.delay)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> RawDocument | None:
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return self._parse(url, resp.text)

    def _parse(self, url: str, html: str) -> RawDocument | None:
        soup = BeautifulSoup(html, "lxml")

        # Remove nav/footer noise
        for tag in soup.select("nav, footer, .headerlink, #indices-and-tables"):
            tag.decompose()

        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else urlparse(url).path

        body = soup.find("div", {"role": "main"}) or soup.find("div", class_="body") or soup.body
        if body is None:
            return None

        content = self._extract_text(body)
        section = self._infer_section(url)

        return RawDocument(
            url=url,
            title=title,
            content=content,
            section=section,
            html=str(body),
        )

    def _extract_text(self, tag) -> str:
        """Walk the DOM and produce readable text, keeping code blocks fenced."""
        parts: list[str] = []
        for node in tag.descendants:
            if not hasattr(node, "name"):
                # NavigableString
                text = str(node).strip()
                if text:
                    parts.append(text)
            elif node.name in ("pre", "code"):
                code = node.get_text()
                if "\n" in code:
                    parts.append(f"\n```\n{code.strip()}\n```\n")
            elif node.name in ("h1", "h2", "h3", "h4"):
                parts.append(f"\n\n### {node.get_text(strip=True)}\n")
            elif node.name == "p":
                parts.append("\n")
        return " ".join(parts)

    @staticmethod
    def _infer_section(url: str) -> str:
        path = urlparse(url).path  # e.g. /3/library/json.html
        parts = [p for p in path.split("/") if p and p != "3"]
        return parts[0] if parts else "misc"
