"""BM25 index + cross-paper retrieval and attribution tests."""

from trace_core.blocks import Chunk
from trace_core.index import CorpusIndex, tokenize
from trace_core.retrieve import build_context, resolve_citations, retrieve


def make_chunk(doc, n, text, section="", pages=(1,), captions=()):
    return Chunk(
        doc_id=doc,
        chunk_id=f"{doc}#{n}",
        section=section,
        pages=list(pages),
        text=text,
        captions=list(captions),
    )


def build_index():
    idx = CorpusIndex()
    idx.add_chunks(
        [
            make_chunk("grape2021", 0, "Optical attenuation coefficient of grape berries measured by OCT.", section="Results", pages=[4]),
            make_chunk("grape2021", 1, "Scattering dominates absorption at 1300 nm in ripe fruit."),
            make_chunk("oct_review", 0, "Optical coherence tomography resolves depth via low-coherence interferometry.", section="Intro", pages=[2]),
            make_chunk("skin2019", 0, "Melanin absorption changes the attenuation profile of human skin.", pages=[7]),
        ]
    )
    return idx


def test_tokenize_keeps_units_and_greek():
    toks = tokenize("μt was 2.4 mm-1 at 1300nm")
    assert "μt" in toks
    assert "1300nm" in toks


def test_retrieval_spans_multiple_papers():
    idx = build_index()
    sources = retrieve(idx, "attenuation coefficient of fruit", k=4)
    docs = {s.hit.chunk.doc_id for s in sources}
    assert "grape2021" in docs
    assert len(sources) >= 2


def test_attribution_includes_section_and_page():
    idx = build_index()
    sources = retrieve(idx, "attenuation coefficient grape berries", k=1)
    attr = sources[0].attribution
    assert "grape2021" in attr
    assert "Results" in attr
    assert "p.4" in attr


def test_context_tags_are_numbered():
    idx = build_index()
    sources = retrieve(idx, "optical coherence tomography", k=3)
    ctx = build_context(sources)
    assert "[S1]" in ctx


def test_resolve_citations_maps_tags_to_attribution():
    idx = build_index()
    sources = retrieve(idx, "attenuation coefficient grape", k=2)
    text = "The coefficient is measured in [S1] and confirmed elsewhere."
    resolved, used = resolve_citations(text, sources)
    assert "[S1]" not in resolved
    assert used
    assert sources[0].attribution in resolved


def test_exclude_doc_filters_source_paper():
    idx = build_index()
    hits = idx.search("attenuation coefficient", k=5, exclude_doc="grape2021")
    assert all(h.chunk.doc_id != "grape2021" for h in hits)


def test_persistence_roundtrip(tmp_path):
    idx = build_index()
    p = tmp_path / "corpus.json"
    idx.save(p)
    loaded = CorpusIndex.load(p)
    assert len(loaded.chunks) == len(idx.chunks)
    a = retrieve(idx, "melanin skin", k=1)[0].attribution
    b = retrieve(loaded, "melanin skin", k=1)[0].attribution
    assert a == b
