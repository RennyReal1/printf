# pdfcorpus — layout-aware retrieval across technical PDFs

Corpus retrieval, cross-paper attribution, and highlight explanation over a
library of technical papers (built and tuned on an OCT / grape-attenuation
corpus). The part worth owning is the **chunking strategy** — the rest is a
thin retrieval + explanation layer on top of it.

## Why chunking is the hard part

Naive fixed-window chunking fails on technical PDFs in two specific,
demonstrable ways. Both are the reason the tool works on OCT papers where
generic RAG pipelines don't.

1. **Figure and table captions drift.** A caption ("Figure 3. Depth-resolved
   attenuation of Chardonnay berries…") floats wherever the figure landed in
   the two-column layout — often a page away from the paragraph that says "as
   shown in Fig. 3." A window chunker files the caption and its discussion in
   different chunks, so retrieval surfaces one without the other and the
   explanation loses either the data description or its interpretation.

2. **Equations split mid-chunk.** A fixed window happily cuts between "the
   attenuation coefficient is obtained from" and the Beer-Lambert expression,
   or between the equation and the "where μt is…" line that defines its
   symbols. The pieces are individually meaningless.

`pdfcorpus` enforces three invariants instead (`pdfcorpus/chunker.py`):

- **Units are atomic.** A *unit* is a paragraph fused with any display
  equation it introduces and any trailing "where …" symbol-definition
  paragraph. Units never split across chunks, so an equation always travels
  with the prose that introduces and defines it.
- **Captions attach to their first referencing paragraph.** Each caption is
  pulled out of stream position and re-attached to the chunk whose prose first
  references its key (`Fig. 3` → `("figure", "3")`). Unreferenced captions
  fall back to the nearest unit on their own page.
- **Chunks never cross a section heading**, so the section path in each
  attribution is exact.

Oversized *prose* units split at sentence boundaries; a unit containing an
equation is never split at all — the invariant outranks the token budget.

## Pipeline

```
extract.py    PDF → RawBlock[]      (PyMuPDF; keeps bbox, font size, bold flags)
classify.py   RawBlock → typed      (body / heading / caption / equation / header-footer)
chunker.py    typed blocks → Chunk[] (the three invariants above)
index.py      Chunk[] → BM25 index  (transparent ~50-line scorer, JSON-persisted)
retrieve.py   query → Source[]      (cross-paper hits + [doc §section, p.N] attribution)
explain.py    highlight → explanation (grounded via retrieval, or generic baseline)
```

### Classification notes

The classifier is where most of the subtlety lives. Display equations are
recognized by combining layout (centered, short) with content (math-symbol
density, math fonts, trailing `(N)` equation number) — any single signal alone
misfires on technical prose (`μt = 2.4 mm⁻¹ at 1300 nm` inside a sentence is
mathy but is *not* a display equation), so the rule requires strong content
evidence or moderate evidence plus a layout cue. Running headers/footers are
removed by cross-page repetition (digits normalized) plus a position+size
fallback for single-page extracts. See `tests/test_classify.py` for the
false-positive cases these rules are built to survive.

## Usage

```bash
pip install -r requirements.txt

# 1. Ingest a folder of papers into a corpus index
python -m pdfcorpus ingest papers/*.pdf --index corpus.json

# 2. Search across the corpus (BM25, cross-paper)
python -m pdfcorpus query "attenuation coefficient grape ripeness" --index corpus.json

# 3. Explain a highlighted passage, grounded in the corpus
python -m pdfcorpus explain "Attenuation decreased with sugar accumulation." \
    --index corpus.json --paper grape_oct_2021

# ...or the generic baseline (no retrieval), for comparison
python -m pdfcorpus explain "Attenuation decreased with sugar accumulation." --generic
```

Explanation and the eval use the Anthropic API (`claude-opus-5`); set
`ANTHROPIC_API_KEY` first. Ingestion, chunking, and search need no API key.

## Eval: does grounding beat generic?

`eval/highlights.json` is a 20-highlight set over the OCT corpus. Each highlight
lists corpus facts a context-grounded explanation should surface that a generic
one structurally can't (a specific figure, a measured value, a cross-paper
contrast). `eval/run_eval.py` generates both explanations per highlight and has
an LLM judge score them blind (A/B order randomized) on *grounding*,
*cross_paper*, and *correctness*, then reports a grounded win-rate.

```bash
# Retrieval-only sanity check, no API calls:
python -m eval.run_eval --index corpus.json --dry-run

# Full judged run:
python -m eval.run_eval --index corpus.json --out eval/results.json
```

## Tests

```bash
python -m pytest
```

32 tests: classification signals and their false positives, the three chunking
invariants, BM25 retrieval + attribution + persistence, and an end-to-end pass
over a synthetic PDF built with PyMuPDF (`tests/conftest.py`) that exercises the
real extract→classify→chunk→index path. No network or API key required.
