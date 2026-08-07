"""Core data types shared across extraction, classification, and chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class BlockType(Enum):
    BODY = "body"
    HEADING = "heading"
    CAPTION = "caption"
    EQUATION = "equation"
    HEADER_FOOTER = "header_footer"


@dataclass
class RawBlock:
    """A text block as extracted from the PDF, with layout metadata."""

    page: int  # 0-indexed
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    text: str
    font_sizes: list[float] = field(default_factory=list)
    font_names: list[str] = field(default_factory=list)
    bold_ratio: float = 0.0  # fraction of chars in bold spans
    page_width: float = 612.0
    page_height: float = 792.0

    # Filled in by classification
    block_type: BlockType = BlockType.BODY
    caption_key: tuple[str, str] | None = None  # ("figure", "3")
    equation_number: str | None = None

    @property
    def median_size(self) -> float:
        if not self.font_sizes:
            return 0.0
        s = sorted(self.font_sizes)
        return s[len(s) // 2]


# A "unit" is the atomic object the chunker packs. Units are never split
# across chunks (except oversized pure-prose units, which split at sentence
# boundaries — never at an equation).
@dataclass
class Unit:
    blocks: list[RawBlock]
    section: str  # heading path at this point in the document

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)

    @property
    def pages(self) -> list[int]:
        return sorted({b.page for b in self.blocks})

    @property
    def has_equation(self) -> bool:
        return any(b.block_type == BlockType.EQUATION for b in self.blocks)

    def references(self) -> set[tuple[str, str]]:
        """Figure/table keys referenced in the prose of this unit."""
        keys: set[tuple[str, str]] = set()
        for b in self.blocks:
            if b.block_type != BlockType.BODY:
                continue
            for m in _REF_RE.finditer(b.text):
                kind = "figure" if m.group(1).lower().startswith("fig") else "table"
                keys.add((kind, m.group(2)))
        return keys


_REF_RE = re.compile(r"\b(Fig(?:ure)?s?\.?|Table)\s*(\d+[a-zA-Z]?)", re.IGNORECASE)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    section: str
    pages: list[int]  # 1-indexed for humans
    text: str
    captions: list[str] = field(default_factory=list)  # attached caption texts
    caption_keys: list[str] = field(default_factory=list)  # e.g. "figure 3"
    equation_numbers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "pages": self.pages,
            "text": self.text,
            "captions": self.captions,
            "caption_keys": self.caption_keys,
            "equation_numbers": self.equation_numbers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)

    @property
    def full_text(self) -> str:
        """Chunk text plus attached captions — what gets indexed and shown."""
        parts = [self.text]
        parts.extend(self.captions)
        return "\n\n".join(parts)

    def citation(self) -> str:
        pages = f"p.{self.pages[0]}" if len(self.pages) == 1 else f"pp.{self.pages[0]}-{self.pages[-1]}"
        sec = f" §{self.section}" if self.section else ""
        return f"[{self.doc_id}{sec}, {pages}]"
