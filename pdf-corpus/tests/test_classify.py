"""Classification tests — the signals that separate equations/captions/headings
from body prose, including the false-positive cases that break naive rules."""

from pdfcorpus.blocks import BlockType, RawBlock
from pdfcorpus.classify import classify_document, math_density


def body(text, **kw):
    kw.setdefault("font_sizes", [10.0] * max(len(text), 1))
    return RawBlock(page=0, bbox=(72, 300, 540, 320), text=text, **kw)


def test_caption_detected_with_key():
    blocks = classify_document([body("Figure 3. Depth-resolved attenuation of Chardonnay berries.")])
    assert blocks[0].block_type == BlockType.CAPTION
    assert blocks[0].caption_key == ("figure", "3")


def test_table_caption_key():
    blocks = classify_document([body("Table 2: Fitted optical properties at 1300 nm.")])
    assert blocks[0].block_type == BlockType.CAPTION
    assert blocks[0].caption_key == ("table", "2")


def test_display_equation_centered_numbered():
    eq = RawBlock(
        page=0,
        bbox=(230, 300, 380, 318),  # centered
        text="I(z) = I_0 exp(-2 μ_t z)   (1)",
        font_sizes=[11.0] * 20,
        page_width=612.0,
    )
    blocks = classify_document([eq])
    assert blocks[0].block_type == BlockType.EQUATION
    assert blocks[0].equation_number == "1"


def test_mathy_prose_not_equation():
    # A real sentence with numbers and a Greek symbol must stay BODY —
    # this is the classic false positive naive density thresholds hit.
    text = (
        "The attenuation coefficient μt was approximately 2.4 mm-1 at 1300 nm "
        "for ripe berries, consistent with values reported for other soft fruit."
    )
    blocks = classify_document([body(text)])
    assert blocks[0].block_type == BlockType.BODY


def test_numbered_heading():
    blocks = classify_document([body("2. Materials and Methods", font_sizes=[10.0] * 5)])
    assert blocks[0].block_type == BlockType.HEADING


def test_larger_font_heading():
    b = RawBlock(page=0, bbox=(72, 100, 300, 118), text="Results", font_sizes=[14.0] * 7)
    body_block = RawBlock(page=0, bbox=(72, 300, 540, 320), text="x " * 200, font_sizes=[10.0] * 400)
    blocks = classify_document([b, body_block])
    assert blocks[0].block_type == BlockType.HEADING


def test_running_header_removed():
    pages = []
    for pno in range(4):
        pages.append(
            RawBlock(
                page=pno,
                bbox=(72, 10, 540, 24),  # top edge
                text=f"OPTICS EXPRESS  Vol. 29  {4400 + pno}",
                font_sizes=[8.0] * 20,
                page_height=792.0,
            )
        )
    out = classify_document(pages)
    assert all(b.block_type == BlockType.HEADER_FOOTER for b in out)


def test_bare_page_number_removed():
    b = RawBlock(page=2, bbox=(300, 770, 320, 784), text="4402", font_sizes=[8.0] * 4, page_height=792.0)
    out = classify_document([b])
    assert out[0].block_type == BlockType.HEADER_FOOTER


def test_single_page_header_removed_without_repetition():
    # No cross-page repetition to lean on, but a short small-font line flush
    # against the top edge is still a running header, not body text.
    header = RawBlock(page=0, bbox=(72, 28, 300, 40), text="J. Grape Optics Vol. 12",
                      font_sizes=[8.0] * 20, page_height=792.0)
    body_block = RawBlock(page=0, bbox=(72, 300, 540, 320), text="x " * 200,
                          font_sizes=[10.0] * 400, page_height=792.0)
    out = classify_document([header, body_block])
    assert out[0].block_type == BlockType.HEADER_FOOTER
    assert out[1].block_type == BlockType.BODY


def test_math_density_monotonic():
    assert math_density("μ = ∫ f dx") > math_density("the coefficient was measured")
