"""Corpus retrieval with cross-paper attribution.

Retrieval returns hits across every paper in the index; each hit carries the
paper, section, and page it came from. `build_context` renders the hits as a
numbered source list ([S1], [S2], ...) so a downstream model can cite them,
and `resolve_citations` maps those tags back to human-readable attributions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .index import CorpusIndex, Hit


@dataclass
class Source:
    tag: str  # "S1"
    hit: Hit

    @property
    def attribution(self) -> str:
        return self.hit.chunk.citation()


def retrieve(index: CorpusIndex, query: str, k: int = 8) -> list[Source]:
    hits = index.search(query, k=k)
    return [Source(tag=f"S{i + 1}", hit=h) for i, h in enumerate(hits)]


def build_context(sources: list[Source]) -> str:
    parts = []
    for s in sources:
        c = s.hit.chunk
        header = f"[{s.tag}] {c.citation()}"
        parts.append(f"{header}\n{c.full_text}")
    return "\n\n---\n\n".join(parts)


_TAG_RE = re.compile(r"\[(S\d+)\]")


def resolve_citations(text: str, sources: list[Source]) -> tuple[str, list[str]]:
    """Replace [S1]-style tags with attributions; return (text, sources used)."""
    by_tag = {s.tag: s for s in sources}
    used: list[str] = []

    def sub(m: re.Match) -> str:
        s = by_tag.get(m.group(1))
        if s is None:
            return m.group(0)
        if s.attribution not in used:
            used.append(s.attribution)
        return s.attribution

    return _TAG_RE.sub(sub, text), used
