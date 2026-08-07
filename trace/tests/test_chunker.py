"""Chunking invariant tests — the three properties that make this chunker
work where fixed-window chunking fails."""

from trace_core.blocks import BlockType, RawBlock
from trace_core.chunker import build_units, chunk_document, chunk_units


def blk(text, btype=BlockType.BODY, page=0, y=300, **kw):
    b = RawBlock(page=page, bbox=(72, y, 540, y + 20), text=text, **kw)
    b.block_type = btype
    return b


def caption(text, key, page=0, y=600):
    b = blk(text, BlockType.CAPTION, page=page, y=y)
    b.caption_key = key
    return b


def equation(text, num=None, page=0, y=300):
    b = blk(text, BlockType.EQUATION, page=page, y=y)
    b.equation_number = num
    return b


# --- Invariant 1: equations never split, and stay with their prose ---------

def test_equation_glued_to_intro_and_definition():
    blocks = [
        blk("The intensity decays with depth according to"),
        equation("I(z) = I_0 exp(-2 μ_t z)   (1)", num="1"),
        blk("where μ_t is the total attenuation coefficient and z is depth."),
    ]
    units, _ = build_units(blocks)
    assert len(units) == 1
    assert units[0].has_equation
    # All three blocks are in the single unit.
    assert len(units[0].blocks) == 3


def test_non_definition_paragraph_after_equation_starts_new_unit():
    blocks = [
        blk("The model is given by"),
        equation("y = a x + b   (2)", num="2"),
        blk("We fit this to the measured profiles using least squares."),
    ]
    units, _ = build_units(blocks)
    assert len(units) == 2
    assert units[0].has_equation
    assert not units[1].has_equation


def test_equation_never_split_even_when_oversized():
    long_intro = "word " * 500
    blocks = [
        blk(long_intro),
        equation("μ_t = μ_a + μ_s   (3)", num="3"),
        blk("where μ_a is absorption and μ_s is scattering."),
    ]
    chunks = chunk_document(blocks, "doc", max_tokens=100)
    # The equation-bearing unit is oversized but must remain in exactly one chunk.
    eq_chunks = [c for c in chunks if "3" in c.equation_numbers]
    assert len(eq_chunks) == 1
    assert "μ_a + μ_s" in eq_chunks[0].text
    assert "where μ_a is absorption" in eq_chunks[0].text


# --- Invariant 2: captions attach to their first referencing paragraph -----

def test_caption_attaches_to_referencing_paragraph():
    blocks = [
        blk("Intro paragraph with no figure reference."),
        blk("As shown in Fig. 3, the attenuation increases with ripeness."),
        caption("Figure 3. Depth-resolved attenuation of Chardonnay berries.", ("figure", "3"), page=1, y=650),
        blk("Later unrelated discussion paragraph."),
    ]
    chunks = chunk_document(blocks, "doc", max_tokens=1000)
    # Find the chunk that references Fig. 3; the caption must live there.
    ref_chunks = [c for c in chunks if "Fig. 3" in c.text]
    assert ref_chunks
    assert any("Depth-resolved attenuation" in cap for c in ref_chunks for cap in c.captions)


def test_caption_key_recorded():
    blocks = [
        blk("We plot the spectra in Fig. 1."),
        caption("Figure 1. Reflectance spectra.", ("figure", "1")),
    ]
    chunks = chunk_document(blocks, "doc", max_tokens=1000)
    assert any("figure 1" in c.caption_keys for c in chunks)


def test_unreferenced_caption_falls_back_to_nearest_on_page():
    blocks = [
        blk("Paragraph on page one.", page=0, y=200),
        caption("Table 5. Fit parameters.", ("table", "5"), page=0, y=400),
    ]
    chunks = chunk_document(blocks, "doc", max_tokens=1000)
    assert any("Fit parameters" in cap for c in chunks for cap in c.captions)


# --- Invariant 3: chunks never cross a section boundary --------------------

def test_chunks_respect_section_boundaries():
    blocks = [
        blk("1. Introduction", BlockType.HEADING),
        blk("Intro body text."),
        blk("2. Methods", BlockType.HEADING),
        blk("Methods body text."),
    ]
    chunks = chunk_document(blocks, "doc", max_tokens=1000)
    sections = {c.section for c in chunks}
    assert "1. Introduction" in sections
    assert "2. Methods" in sections
    # No chunk mixes the two sections.
    for c in chunks:
        assert not ("Intro body" in c.text and "Methods body" in c.text)


# --- Packing behaviour -----------------------------------------------------

def test_prose_packs_up_to_budget():
    blocks = [blk(f"Sentence number {i} about optics.") for i in range(20)]
    chunks = chunk_document(blocks, "doc", max_tokens=40)
    assert len(chunks) > 1  # must split
    assert all(c.section == "" for c in chunks)


def test_oversized_prose_splits_at_sentences():
    text = ". ".join(f"This is sentence {i} of a very long paragraph" for i in range(60)) + "."
    blocks = [blk(text)]
    chunks = chunk_document(blocks, "doc", max_tokens=60)
    assert len(chunks) > 1


def test_pages_are_one_indexed():
    blocks = [blk("Body on the first page.", page=0)]
    chunks = chunk_document(blocks, "doc", max_tokens=1000)
    assert chunks[0].pages == [1]
