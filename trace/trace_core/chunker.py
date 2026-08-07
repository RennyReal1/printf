"""Layout-aware chunking for technical PDFs.

Naive fixed-window chunking fails on papers in two specific ways:

1. **Captions drift.** "Figure 3. Depth-resolved attenuation of Chardonnay
   berries..." floats wherever the figure landed in the layout — often a page
   away from the paragraph that says "as shown in Fig. 3". A window chunker
   puts the caption in one chunk and the discussion in another, so retrieval
   finds one without the other and the explanation loses either the data
   description or its interpretation.

2. **Equations split.** A fixed window happily cuts between "the attenuation
   coefficient is obtained from" and the Beer-Lambert expression that follows,
   or between the equation and the "where μt is..." line that defines its
   symbols. The resulting chunks are individually meaningless.

This chunker fixes both with three invariants:

- **Units are atomic.** A unit is a paragraph fused with any display equation
  it introduces and any trailing "where ..." symbol-definition paragraph.
  Units never split across chunks, so an equation always travels with the
  prose that introduces and defines it.
- **Captions attach to their first referencing paragraph.** Each caption is
  removed from stream position and attached to the chunk containing the first
  unit whose prose references its key ("Fig. 3" -> ("figure", "3")).
  Unreferenced captions fall back to the nearest unit on their own page.
- **Chunks respect section boundaries.** A chunk never spans a heading, so
  the section path in the attribution is always exact.

Oversized pure-prose units split at sentence boundaries; units containing an
equation are never split at all.
"""

from __future__ import annotations

import re

from .blocks import BlockType, Chunk, RawBlock, Unit

# Continuation words that glue a symbol-definition paragraph to the equation
# above it: "where μt is the attenuation coefficient..."
_CONTINUATION_RE = re.compile(r"^(where|with|here|in which|and)\b", re.IGNORECASE)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.35) + 1


def build_units(blocks: list[RawBlock]) -> tuple[list[Unit], list[RawBlock]]:
    """Group classified blocks into atomic units; return (units, captions).

    Captions are pulled out of stream order — attachment happens in
    `chunk_units` once references are known.
    """
    units: list[Unit] = []
    captions: list[RawBlock] = []
    section_parts: list[str] = []
    current: list[RawBlock] = []

    def flush() -> None:
        nonlocal current
        if current:
            units.append(Unit(blocks=current, section=" > ".join(section_parts)))
            current = []

    for b in blocks:
        if b.block_type == BlockType.HEADER_FOOTER:
            continue
        if b.block_type == BlockType.CAPTION:
            captions.append(b)
            continue
        if b.block_type == BlockType.HEADING:
            flush()
            section_parts = [b.text.strip()]
            continue
        if b.block_type == BlockType.EQUATION:
            # Equation glues to the unit being built (its introducing
            # paragraph). An equation with no open unit starts one.
            current.append(b)
            continue
        # BODY
        if current and current[-1].block_type == BlockType.EQUATION:
            # Paragraph directly after an equation: glue if it defines the
            # equation's symbols, otherwise it starts a fresh unit.
            if _CONTINUATION_RE.match(b.text.strip()):
                current.append(b)
                flush()
                continue
            flush()
            current.append(b)
        elif current:
            flush()
            current.append(b)
        else:
            current.append(b)
    flush()
    return units, captions


def chunk_units(
    units: list[Unit],
    captions: list[RawBlock],
    doc_id: str,
    max_tokens: int = 450,
) -> list[Chunk]:
    # Map each caption to the index of the unit it belongs with.
    owner: dict[int, list[RawBlock]] = {}
    for cap in captions:
        idx = _find_referencing_unit(cap, units)
        owner.setdefault(idx, []).append(cap)

    chunks: list[Chunk] = []
    buf: list[Unit] = []
    buf_caps: list[RawBlock] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_caps, buf_tokens
        if not buf:
            return
        chunks.append(_make_chunk(doc_id, len(chunks), buf, buf_caps))
        buf, buf_caps, buf_tokens = [], [], 0

    for i, u in enumerate(units):
        u_caps = owner.get(i, [])
        cap_tokens = sum(approx_tokens(c.text) for c in u_caps)
        u_tokens = approx_tokens(u.text) + cap_tokens

        if buf and buf[-1].section != u.section:
            flush()  # never cross a section boundary

        if buf_tokens + u_tokens > max_tokens and buf:
            flush()

        if u_tokens > max_tokens and not u.has_equation and not u_caps:
            # Oversized prose: split at sentence boundaries.
            for piece in _split_prose_unit(u, max_tokens):
                chunks.append(_make_chunk(doc_id, len(chunks), [piece], []))
            continue

        # Oversized units with an equation or caption stay whole — the
        # invariants outrank the budget.
        buf.append(u)
        buf_caps.extend(u_caps)
        buf_tokens += u_tokens
    flush()
    return chunks


def chunk_document(blocks: list[RawBlock], doc_id: str, max_tokens: int = 450) -> list[Chunk]:
    units, captions = build_units(blocks)
    return chunk_units(units, captions, doc_id, max_tokens=max_tokens)


def _find_referencing_unit(cap: RawBlock, units: list[Unit]) -> int:
    key = cap.caption_key
    for i, u in enumerate(units):
        if key in u.references():
            return i
    # Fallback: nearest unit on the caption's own page (prefer preceding).
    best, best_dist = 0, float("inf")
    for i, u in enumerate(units):
        if cap.page in {b.page for b in u.blocks}:
            dist = min(abs(b.bbox[1] - cap.bbox[1]) for b in u.blocks if b.page == cap.page)
            if dist < best_dist:
                best, best_dist = i, dist
    return best


def _make_chunk(doc_id: str, n: int, units: list[Unit], caps: list[RawBlock]) -> Chunk:
    pages = sorted({p for u in units for p in u.pages} | {c.page for c in caps})
    eq_numbers = [
        b.equation_number
        for u in units
        for b in u.blocks
        if b.block_type == BlockType.EQUATION and b.equation_number
    ]
    return Chunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}#{n}",
        section=units[0].section if units else "",
        pages=[p + 1 for p in pages],
        text="\n\n".join(u.text for u in units),
        captions=[c.text for c in caps],
        caption_keys=[f"{c.caption_key[0]} {c.caption_key[1]}" for c in caps if c.caption_key],
        equation_numbers=eq_numbers,
    )


def _split_prose_unit(u: Unit, max_tokens: int) -> list[Unit]:
    sentences = _SENTENCE_SPLIT_RE.split(u.text)
    pieces: list[Unit] = []
    buf: list[str] = []
    buf_tokens = 0
    page = u.blocks[0].page
    for s in sentences:
        t = approx_tokens(s)
        if buf and buf_tokens + t > max_tokens:
            pieces.append(_prose_unit(" ".join(buf), u.section, page))
            buf, buf_tokens = [], 0
        buf.append(s)
        buf_tokens += t
    if buf:
        pieces.append(_prose_unit(" ".join(buf), u.section, page))
    return pieces


def _prose_unit(text: str, section: str, page: int) -> Unit:
    return Unit(blocks=[RawBlock(page=page, bbox=(0, 0, 0, 0), text=text)], section=section)
