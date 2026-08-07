"""The demo corpus must exercise the chunker's invariants — otherwise the
one-command demo wouldn't actually show what Trace does."""

from trace_core.demo import build_demo_corpus
from trace_core.index import CorpusIndex
from trace_core.retrieve import retrieve


def test_demo_corpus_builds_and_loads(tmp_path):
    index_path = build_demo_corpus(str(tmp_path))
    idx = CorpusIndex.load(index_path)
    assert set(idx.doc_ids) == {"grape_oct_2021", "oct_review", "skin_optics_2019"}
    assert len(idx.chunks) >= 3


def test_demo_equation_stays_with_definition(tmp_path):
    idx = CorpusIndex.load(build_demo_corpus(str(tmp_path)))
    eq = [c for c in idx.chunks if "exp(-2 mu z)" in c.text]
    assert len(eq) == 1
    assert "total attenuation coefficient" in eq[0].text  # the "where..." line


def test_demo_caption_attached(tmp_path):
    idx = CorpusIndex.load(build_demo_corpus(str(tmp_path)))
    ref = [c for c in idx.chunks if "Fig. 2" in c.text]
    assert ref
    assert any("Depth-resolved attenuation" in cap for c in ref for cap in c.captions)


def test_demo_header_stripped(tmp_path):
    idx = CorpusIndex.load(build_demo_corpus(str(tmp_path)))
    assert all("J. Grape Optics" not in c.text for c in idx.chunks)


def test_demo_retrieval_is_cross_paper(tmp_path):
    idx = CorpusIndex.load(build_demo_corpus(str(tmp_path)))
    sources = retrieve(idx, "attenuation coefficient extracted from depth", k=5)
    docs = {s.hit.chunk.doc_id for s in sources}
    assert len(docs) >= 2
