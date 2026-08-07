"""BM25 corpus index over chunks, with JSON persistence.

BM25 is implemented directly (~50 lines) rather than pulled in as a
dependency: retrieval quality here is dominated by chunk quality, and a
transparent scorer makes eval failures debuggable. Tokenization keeps Greek
letters and digit-bearing tokens ("1300nm", "μt") because those carry most of
the signal in optics papers.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .blocks import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9α-ωΑ-Ω]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float


class CorpusIndex:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._lengths: list[int] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            tokens = tokenize(c.full_text)
            tf = Counter(tokens)
            self.chunks.append(c)
            self._tf.append(tf)
            self._lengths.append(len(tokens))
            for term in tf:
                self._df[term] += 1

    @property
    def _avg_len(self) -> float:
        return sum(self._lengths) / len(self._lengths) if self._lengths else 1.0

    def search(self, query: str, k: int = 8, exclude_doc: str | None = None) -> list[Hit]:
        q_terms = tokenize(query)
        n = len(self.chunks)
        if n == 0:
            return []
        scores = [0.0] * n
        for term in q_terms:
            df = self._df.get(term)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._tf):
                f = tf.get(term)
                if not f:
                    continue
                norm = f * (K1 + 1) / (f + K1 * (1 - B + B * self._lengths[i] / self._avg_len))
                scores[i] += idf * norm
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if exclude_doc and self.chunks[i].doc_id == exclude_doc:
                continue
            hits.append(Hit(chunk=self.chunks[i], score=scores[i]))
            if len(hits) >= k:
                break
        return hits

    # -- persistence ---------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"chunks": [c.to_dict() for c in self.chunks]}, ensure_ascii=False)
        )

    @classmethod
    def load(cls, path: str | Path) -> "CorpusIndex":
        data = json.loads(Path(path).read_text())
        idx = cls()
        idx.add_chunks([Chunk.from_dict(d) for d in data["chunks"]])
        return idx

    @property
    def doc_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.chunks:
            seen.setdefault(c.doc_id, None)
        return list(seen)
