"""Synthetic OCT-paper fixtures built with PyMuPDF.

Real papers can't ship in the repo, so the tests build small PDFs that
reproduce the layout patterns the chunker is designed for: a caption placed a
page away from the paragraph that references it, a centered numbered display
equation with a "where ..." definition line, and a running header on every
page.
"""

from __future__ import annotations

import fitz
import pytest

HEADER = "J. Grape Optics Vol. 12"

INTRO = (
    "Optical coherence tomography provides depth-resolved imaging of grape "
    "berry tissue. The measured signal decays with depth according to the "
    "attenuation coefficient of the tissue."
)

EQ_INTRO = (
    "Assuming single scattering, the detected OCT signal follows a "
    "Beer-Lambert decay and the attenuation coefficient is obtained from"
)

EQUATION = "I(z) = I0 exp(-2 mu z)   (3)"

WHERE_LINE = (
    "where mu is the total attenuation coefficient and z is the depth into "
    "the berry tissue."
)

FIG_REF_PARA = (
    "The fitted attenuation profiles are shown in Fig. 2 for Chardonnay "
    "berries at three ripening stages. Attenuation decreased with sugar "
    "accumulation across all samples."
)

CAPTION = (
    "Figure 2. Depth-resolved attenuation coefficient of Chardonnay berries "
    "measured at 1300 nm across three ripening stages."
)


def _make_pdf(path: str) -> None:
    doc = fitz.open()
    # Page 1: header, heading, intro, equation intro, equation, where-line.
    p1 = doc.new_page()
    p1.insert_text((72, 30), HEADER, fontsize=8)
    p1.insert_text((72, 90), "1. Introduction", fontsize=14)
    p1.insert_textbox(fitz.Rect(72, 110, 540, 200), INTRO, fontsize=10)
    p1.insert_text((72, 230), "2. Theory", fontsize=14)
    p1.insert_textbox(fitz.Rect(72, 250, 540, 300), EQ_INTRO, fontsize=10)
    # Centered display equation.
    p1.insert_text((250, 330), EQUATION, fontsize=11)
    p1.insert_textbox(fitz.Rect(72, 360, 540, 410), WHERE_LINE, fontsize=10)
    p1.insert_text((300, 770), "1", fontsize=8)  # page number

    # Page 2: header, results heading, the paragraph that references Fig. 2,
    # then the caption itself lower on the page.
    p2 = doc.new_page()
    p2.insert_text((72, 30), HEADER, fontsize=8)
    p2.insert_text((72, 90), "3. Results", fontsize=14)
    p2.insert_textbox(fitz.Rect(72, 110, 540, 200), FIG_REF_PARA, fontsize=10)
    p2.insert_textbox(fitz.Rect(72, 620, 540, 700), CAPTION, fontsize=9)
    p2.insert_text((300, 770), "2", fontsize=8)

    doc.save(path)
    doc.close()


@pytest.fixture
def oct_pdf(tmp_path):
    path = str(tmp_path / "grape_oct.pdf")
    _make_pdf(path)
    return path
