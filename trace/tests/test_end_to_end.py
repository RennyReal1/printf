"""End-to-end test over a synthetic PDF built with PyMuPDF (see conftest).

Exercises the real extract -> classify -> chunk -> index path, so a regression
in the PyMuPDF layer (font flags, bbox handling, dehyphenation) is caught here
even though the unit tests use synthetic RawBlocks.
"""

from trace_core.chunker import chunk_document
from trace_core.classify import classify_document
from trace_core.extract import extract_blocks
from trace_core.index import CorpusIndex
from trace_core.retrieve import retrieve


def ingest(path, doc_id="grape_oct"):
    blocks = classify_document(extract_blocks(path))
    return chunk_document(blocks, doc_id, max_tokens=400)


def test_running_header_stripped_from_chunks(oct_pdf):
    chunks = ingest(oct_pdf)
    assert all("J. Grape Optics" not in c.text for c in chunks)


def test_equation_travels_with_its_definition(oct_pdf):
    chunks = ingest(oct_pdf)
    eq_chunks = [c for c in chunks if "exp(-2 mu z)" in c.text]
    assert len(eq_chunks) == 1
    # The "where mu is..." definition must be in the same chunk as the equation.
    assert "total attenuation coefficient" in eq_chunks[0].text


def test_caption_attached_to_referencing_paragraph(oct_pdf):
    chunks = ingest(oct_pdf)
    ref_chunks = [c for c in chunks if "Fig. 2" in c.text]
    assert ref_chunks
    joined_caps = " ".join(cap for c in ref_chunks for cap in c.captions)
    assert "Depth-resolved attenuation" in joined_caps


def test_sections_are_exact(oct_pdf):
    chunks = ingest(oct_pdf)
    sections = {c.section for c in chunks}
    assert any("Introduction" in s for s in sections)
    assert any("Results" in s for s in sections)


def test_retrieval_finds_grounded_chunk(oct_pdf):
    idx = CorpusIndex()
    idx.add_chunks(ingest(oct_pdf))
    sources = retrieve(idx, "attenuation coefficient of Chardonnay berries", k=3)
    assert sources
    top = sources[0].hit.chunk
    assert top.doc_id == "grape_oct"
    assert top.pages  # attribution carries page numbers
