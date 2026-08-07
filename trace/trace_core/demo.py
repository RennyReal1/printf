"""`trace demo` — build a tiny synthetic OCT corpus and serve it, so you can
load the extension and try highlighting in under a minute (no real PDFs or API
key needed to see retrieval; explanations still need ANTHROPIC_API_KEY).

The synthetic papers deliberately reproduce the layout patterns Trace is built
for, so highlighting actually exercises the chunker:

  * a display equation with a "(3)" number and a following "where mu is..."
    definition line (must stay in one chunk),
  * a caption placed a page away from the paragraph that references Fig. 2
    (must re-attach to that paragraph's chunk),
  * a running header on every page (must be stripped).
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from .chunker import chunk_document
from .classify import classify_document
from .extract import extract_blocks
from .index import CorpusIndex

_HEADER = {
    "grape_oct_2021": "J. Grape Optics  Vol. 12",
    "oct_review": "Optical Coherence Tomography: A Review",
    "skin_optics_2019": "Biomed. Opt. Express  Vol. 10",
}


def _textbox(page, rect, text, fontsize=10):
    page.insert_textbox(pymupdf.Rect(*rect), text, fontsize=fontsize)


def _paper_grape(path: str) -> None:
    doc = pymupdf.open()
    # Page 1: intro + theory with the numbered equation and its definition.
    p = doc.new_page()
    p.insert_text((72, 30), _HEADER["grape_oct_2021"], fontsize=8)
    p.insert_text((72, 90), "1. Introduction", fontsize=14)
    _textbox(p, (72, 110, 540, 190),
             "Optical coherence tomography (OCT) provides depth-resolved imaging "
             "of grape berry tissue. The measured signal decays with depth "
             "according to the attenuation coefficient of the tissue, which "
             "carries information about ripeness.")
    p.insert_text((72, 220), "2. Theory", fontsize=14)
    _textbox(p, (72, 240, 540, 300),
             "Assuming single scattering, the detected OCT signal follows a "
             "Beer-Lambert decay and the attenuation coefficient is obtained from")
    p.insert_text((250, 330), "I(z) = I0 exp(-2 mu z)   (3)", fontsize=11)
    _textbox(p, (72, 360, 540, 420),
             "where mu is the total attenuation coefficient and z is the depth "
             "into the berry tissue. The coefficient is estimated by least-squares "
             "regression of log-intensity against depth.")
    p.insert_text((300, 770), "1", fontsize=8)

    # Page 2: results paragraph that references Fig. 2, caption lower down.
    p = doc.new_page()
    p.insert_text((72, 30), _HEADER["grape_oct_2021"], fontsize=8)
    p.insert_text((72, 90), "3. Results", fontsize=14)
    _textbox(p, (72, 110, 540, 200),
             "The fitted attenuation profiles are shown in Fig. 2 for Chardonnay "
             "berries at three ripening stages. Attenuation decreased with sugar "
             "accumulation across all samples, reaching approximately 2.4 mm-1 at "
             "1300 nm for ripe berries. Scattering dominated absorption at this "
             "wavelength.")
    _textbox(p, (72, 620, 540, 710),
             "Figure 2. Depth-resolved attenuation coefficient of Chardonnay "
             "berries measured at 1300 nm across three ripening stages.", fontsize=9)
    p.insert_text((300, 770), "2", fontsize=8)
    doc.save(path)
    doc.close()


def _paper_review(path: str) -> None:
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 30), _HEADER["oct_review"], fontsize=8)
    p.insert_text((72, 90), "1. Principles", fontsize=14)
    _textbox(p, (72, 110, 540, 220),
             "Optical coherence tomography resolves depth via low-coherence "
             "interferometry. Axial resolution is inversely proportional to the "
             "source bandwidth, so broadband sources give finer depth sectioning. "
             "Penetration depth in turbid media is limited by total attenuation, "
             "which is why longer wavelengths such as 1300 nm are chosen to "
             "balance resolution against depth. Multiple scattering degrades the "
             "exponential-decay model at large depths.")
    p.insert_text((300, 770), "1", fontsize=8)
    doc.save(path)
    doc.close()


def _paper_skin(path: str) -> None:
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 30), _HEADER["skin_optics_2019"], fontsize=8)
    p.insert_text((72, 90), "3. Results", fontsize=14)
    _textbox(p, (72, 110, 540, 220),
             "Melanin absorption changes the attenuation profile of human skin. "
             "Absorption dominates the near-infrared attenuation of pigmented "
             "tissue, in contrast to fruit where scattering dominates. The "
             "attenuation coefficient is extracted from the slope of the "
             "log-intensity depth profile, the same method used for berries, but "
             "wavelength selection trades off absorption contrast against the "
             "scattering background.")
    p.insert_text((300, 770), "1", fontsize=8)
    doc.save(path)
    doc.close()


_BUILDERS = {
    "grape_oct_2021": _paper_grape,
    "oct_review": _paper_review,
    "skin_optics_2019": _paper_skin,
}


def build_demo_corpus(out_dir: str) -> str:
    """Write the synthetic PDFs, ingest them, and return the index path."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    idx = CorpusIndex()
    for doc_id, builder in _BUILDERS.items():
        pdf_path = str(d / f"{doc_id}.pdf")
        builder(pdf_path)
        blocks = classify_document(extract_blocks(pdf_path))
        chunks = chunk_document(blocks, doc_id)
        idx.add_chunks(chunks)
    index_path = str(d / "corpus.json")
    idx.save(index_path)
    return index_path


def run_demo(out_dir: str = "trace-demo", port: int = 8765) -> None:
    from .server import serve

    index_path = build_demo_corpus(out_dir)
    print(
        f"Built a demo corpus in {out_dir}/ (3 synthetic OCT papers).\n"
        "Try highlighting one of these once the extension is loaded:\n"
        '  - "Attenuation decreased with sugar accumulation across all samples."\n'
        '  - "the attenuation coefficient is obtained from"\n'
        '  - "Penetration depth in turbid media is limited by total attenuation."\n'
    )
    serve(index_path, port)
