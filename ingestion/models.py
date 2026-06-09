from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RawDocument:
    url: str
    title: str
    content: str
    section: str = ""
    html: str = ""


@dataclass
class Chunk:
    text: str
    source_url: str
    title: str
    section: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)
