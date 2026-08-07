"""Block classification for technical PDFs.

The order of operations matters:

1. Header/footer removal is corpus-of-pages evidence (repetition + position),
   so it runs over the whole document first.
2. Captions are recognized lexically ("Figure 3." / "Table 2:") — cheap and
   high precision.
3. Display equations are recognized by a combination of layout (centered,
   short) and content (math-symbol density, math fonts, trailing "(N)"
   equation number). Any one signal alone misfires on technical prose —
   "μs = 2.4 mm−1 at 1300 nm" inside a paragraph is mathy but is not a
   display equation — so we require either strong content evidence or
   moderate content evidence plus a layout cue.
4. Headings are short lines set larger/bolder than the body median, or
   numbered-section patterns.
"""

from __future__ import annotations

import re
from collections import Counter

from .blocks import BlockType, RawBlock

CAPTION_RE = re.compile(
    r"^(Fig(?:ure)?|Table|Scheme|Algorithm|Box)\.?\s+(\d+[a-zA-Z]?)\s*[.:|—–-]",
)

NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")

EQ_NUMBER_RE = re.compile(r"\((\d+[a-z]?)\)\s*$")

_MATH_CHARS = set("=+−±×÷∫∑∏√∂∇"
                  "≈≠≤≥∝∞→←↔"
                  "αβγδεζηθικ"
                  "λμνξπρστυφ"
                  "χψωΓΔΘΛΞΠΣ"
                  "ΦΨΩ^_/\\{}[]()|<>~")

_MATH_FONT_HINTS = ("math", "cmmi", "cmsy", "cmex", "symbol", "italic")


def classify_document(blocks: list[RawBlock]) -> list[RawBlock]:
    """Classify blocks in place and return them, headers/footers included
    (marked, not dropped — the chunker decides what to keep)."""
    n_pages = max((b.page for b in blocks), default=0) + 1
    body_size = _body_font_size(blocks)
    repeated = _repeated_edge_lines(blocks, n_pages)

    for b in blocks:
        if _is_header_footer(b, repeated, body_size):
            b.block_type = BlockType.HEADER_FOOTER
            continue
        m = CAPTION_RE.match(b.text.strip())
        if m:
            kind = "figure" if m.group(1).lower().startswith("fig") else m.group(1).lower()
            b.block_type = BlockType.CAPTION
            b.caption_key = (kind, m.group(2))
            continue
        if _is_display_equation(b):
            b.block_type = BlockType.EQUATION
            em = EQ_NUMBER_RE.search(b.text.strip())
            b.equation_number = em.group(1) if em else None
            continue
        if _is_heading(b, body_size):
            b.block_type = BlockType.HEADING
            continue
        b.block_type = BlockType.BODY
    return blocks


def _body_font_size(blocks: list[RawBlock]) -> float:
    """The dominant font size, weighted by text volume — the body size."""
    counter: Counter[float] = Counter()
    for b in blocks:
        for s in b.font_sizes:
            counter[round(s, 1)] += 1
    if not counter:
        return 10.0
    return counter.most_common(1)[0][0]


def _repeated_edge_lines(blocks: list[RawBlock], n_pages: int) -> set[str]:
    """Normalized texts that recur near page edges on multiple pages.

    Digits are collapsed so 'OPTICS EXPRESS  Vol. 29, p. 4401' and '...p. 4402'
    count as the same running header.
    """
    if n_pages < 2:
        return set()
    seen: Counter[str] = Counter()
    for b in blocks:
        if not _near_edge(b):
            continue
        key = re.sub(r"\d+", "#", b.text.strip().lower())
        seen[key] += 1
    threshold = max(2, n_pages // 2)
    return {k for k, c in seen.items() if c >= threshold}


def _near_edge(b: RawBlock) -> bool:
    top = b.bbox[3] < 0.10 * b.page_height
    bottom = b.bbox[1] > 0.90 * b.page_height
    return top or bottom


def _is_header_footer(b: RawBlock, repeated: set[str], body_size: float) -> bool:
    if not _near_edge(b):
        return False
    stripped = b.text.strip()
    if re.fullmatch(r"\d{1,4}", stripped):  # bare page number
        return True
    key = re.sub(r"\d+", "#", stripped.lower())
    if key in repeated:
        return True
    # No repetition evidence (e.g. single-page extract): a near-edge line
    # that is short and set no larger than the body is almost certainly a
    # running header/footer, not a heading (headings are set larger) or
    # body prose (body is never flush against the page edge).
    return len(stripped.split()) <= 8 and "\n" not in stripped and b.median_size <= body_size


def math_density(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    mathy = sum(1 for c in chars if c in _MATH_CHARS or c.isdigit())
    return mathy / len(chars)


def _long_word_count(text: str) -> int:
    return sum(1 for w in re.findall(r"[A-Za-z]{4,}", text))


def _is_display_equation(b: RawBlock) -> bool:
    text = b.text.strip()
    if len(text) > 300 or not text:
        return False
    density = math_density(text)
    long_words = _long_word_count(text)
    math_fonts = sum(
        1 for f in b.font_names if any(h in f.lower() for h in _MATH_FONT_HINTS)
    )
    font_ratio = math_fonts / len(b.font_names) if b.font_names else 0.0
    numbered = bool(EQ_NUMBER_RE.search(text))
    centered = _is_centered(b)

    # Strong content evidence stands alone; moderate evidence needs a layout
    # or numbering cue. Prose guards: real sentences carry many long words.
    if density >= 0.55 and long_words <= 3:
        return True
    if density >= 0.35 and long_words <= 5 and (numbered or centered or font_ratio > 0.4):
        return True
    return False


def _is_heading(b: RawBlock, body_size: float) -> bool:
    text = b.text.strip()
    if "\n" in text or len(text.split()) > 12 or not text:
        return False
    if text.endswith("."):
        # "2. Methods" keeps its trailing structure via the numbered pattern;
        # a sentence ending in a period is prose.
        if not NUMBERED_HEADING_RE.match(text):
            return False
    larger = b.median_size > 1.12 * body_size
    boldish = b.bold_ratio > 0.6
    numbered = bool(NUMBERED_HEADING_RE.match(text))
    return numbered or larger or boldish


def _is_centered(b: RawBlock) -> bool:
    x0, _, x1, _ = b.bbox
    width = b.page_width
    left_margin = x0
    right_margin = width - x1
    if left_margin <= 0.12 * width:  # flush with body text margin
        return False
    return abs(left_margin - right_margin) < 0.25 * width
