"""PDF -> RawBlock extraction via PyMuPDF, preserving layout metadata.

We keep bounding boxes, per-span font sizes/names, and bold flags because the
classifier needs them: display equations are recognized partly by centering
and math fonts, headings by relative font size, headers/footers by vertical
position and cross-page repetition.
"""

from __future__ import annotations

import fitz  # PyMuPDF

from .blocks import RawBlock

_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold


def extract_blocks(path: str) -> list[RawBlock]:
    doc = fitz.open(path)
    blocks: list[RawBlock] = []
    for pno, page in enumerate(doc):
        d = page.get_text("dict", sort=True)
        for b in d.get("blocks", []):
            if b.get("type") != 0:  # text blocks only; images have no text
                continue
            lines = []
            sizes: list[float] = []
            fonts: list[str] = []
            bold_chars = 0
            total_chars = 0
            for line in b.get("lines", []):
                line_text = "".join(span["text"] for span in line.get("spans", []))
                if line_text.strip():
                    lines.append(line_text.strip())
                for span in line.get("spans", []):
                    n = len(span["text"])
                    if n == 0:
                        continue
                    sizes.append(span["size"])
                    fonts.append(span["font"])
                    total_chars += n
                    if span["flags"] & _BOLD_FLAG:
                        bold_chars += n
            text = _dehyphenate("\n".join(lines))
            if not text.strip():
                continue
            blocks.append(
                RawBlock(
                    page=pno,
                    bbox=tuple(b["bbox"]),
                    text=text,
                    font_sizes=sizes,
                    font_names=fonts,
                    bold_ratio=bold_chars / total_chars if total_chars else 0.0,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                )
            )
    doc.close()
    return blocks


def _dehyphenate(text: str) -> str:
    """Join words hyphenated across line breaks: 'attenua-\\ntion' -> 'attenuation'."""
    import re

    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)
