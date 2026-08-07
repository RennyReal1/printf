# Trace — highlight a passage, get an explanation grounded in your own corpus

Trace turns a library of technical papers into a reading companion: highlight a
passage in your browser and get an explanation that's grounded in *your* corpus
— connected to the related methods, figures, equations, and results across your
other papers, each cited — instead of the generic textbook answer a stranger's
model gives you. Built and tuned on an OCT / grape-attenuation corpus.

It ships as two pieces:

- **A browser extension** (`extension/`) — the front-end. You highlight a
  passage and it shows the explanation in a side panel.
- **A local backend** (`trace serve`) — the brains. It holds the corpus index
  and calls the model, running on your machine and your API key. Nothing leaves
  your laptop except the model request.

The extension is intentionally thin; all the real work — chunking, retrieval,
explanation — lives in the Python core (`trace_core/`).

## Why chunking is the hard part

The part worth owning is the **chunking strategy for technical PDFs**
(`trace_core/chunker.py`). Naive fixed-window chunking fails on papers in two
specific, demonstrable ways — both are why Trace works on OCT papers where
generic RAG pipelines don't.

1. **Figure and table captions drift.** A caption ("Figure 3. Depth-resolved
   attenuation of Chardonnay berries…") floats wherever the figure landed in
   the two-column layout — often a page away from the paragraph that says "as
   shown in Fig. 3." A window chunker files the caption and its discussion in
   different chunks, so retrieval surfaces one without the other.

2. **Equations split mid-chunk.** A fixed window happily cuts between "the
   attenuation coefficient is obtained from" and the Beer-Lambert expression,
   or between the equation and the "where μt is…" line that defines its
   symbols. The pieces are individually meaningless.

Trace enforces three invariants instead:

- **Units are atomic.** A *unit* is a paragraph fused with any display equation
  it introduces and any trailing "where …" symbol-definition paragraph. Units
  never split across chunks, so an equation always travels with the prose that
  introduces and defines it.
- **Captions attach to their first referencing paragraph.** Each caption is
  pulled out of stream position and re-attached to the chunk whose prose first
  references its key (`Fig. 3` → `("figure", "3")`), with a nearest-on-page
  fallback for unreferenced captions.
- **Chunks never cross a section heading**, so the section path in each
  attribution is exact.

Oversized *prose* units split at sentence boundaries; a unit containing an
equation is never split at all — the invariant outranks the token budget.

## Try it in one minute

```bash
pip install -e .
trace demo                # builds a 3-paper synthetic OCT corpus and serves it
```

Then load the extension (step 3 below) and highlight one of the passages
`trace demo` prints. No real PDFs needed; explanations still need an API key,
but retrieval and the highlight UX work without one.

## What retrieval looks like

`trace query` over the demo corpus — note `[S2]`: the display equation stays
fused with the `where mu is…` line that defines its symbols, and hits span
three different papers:

```
$ trace query "how is the attenuation coefficient extracted from depth" --index corpus.json -k 3

[S1] [skin_optics_2019 §3. Results, p.1]  (score 3.41)
  ... The attenuation coefficient is extracted from the slope of the
  log-intensity depth profile, the same method used for berries ...

[S2] [grape_oct_2021 §2. Theory, p.1]  (score 2.71)
  Assuming single scattering, the detected OCT signal follows a Beer-Lambert
  decay and the attenuation coefficient is obtained from I(z) = I0 exp(-2 mu z)
  (3) where mu is the total attenuation coefficient and z is the depth ...

[S3] [oct_review §1. Principles, p.1]  (score 1.22)
  ... Penetration depth in turbid media is limited by total attenuation, which
  is why longer wavelengths such as 1300 nm are chosen ...
```

## Setup

```bash
pip install -e .          # installs the `trace` command + deps (PyMuPDF, anthropic)
export ANTHROPIC_API_KEY=sk-ant-...   # or: ant auth login
```

(If you'd rather not install, every `trace <cmd>` below also works as
`python -m trace_core <cmd>`.)

### 1. Build your corpus index

```bash
trace ingest ~/papers/*.pdf --index corpus.json
```

Prints, per paper, how many captions were attached and equations detected — a
quick sanity check that the chunker parsed the layout.

### 2. Start the backend

```bash
trace serve --index corpus.json      # serves http://127.0.0.1:8765
```

Leave this running while you read.

### 3. Load the extension (Chrome / Edge / any Chromium browser)

1. Go to `chrome://extensions`, turn on **Developer mode** (top-right).
2. **Load unpacked** → select this repo's `extension/` folder.
3. Click the Trace icon: the popup shows **connected — N chunks across M
   papers** when it can see the backend. (Set a different port there if you
   ran `trace serve --port …`.)

Now **highlight any passage** on a web page (arXiv HTML, a journal page, a
PDF.js viewer). An **Explain with Trace** chip appears; click it and the
grounded explanation, with sources, opens in a panel. Right-click → *Explain
selection with Trace* works too.

> **Chrome's built-in PDF viewer** renders PDFs in a way extensions can't read
> the selection from. For those, use the popup's **paste box** — paste the
> passage and hit Explain. (Reading papers as HTML, or through a PDF.js-based
> viewer, gives the full in-page highlight experience.)

## Command line

The same core is usable without the browser:

```bash
trace query "attenuation coefficient grape ripeness" --index corpus.json
trace explain "Attenuation decreased with sugar accumulation." --index corpus.json --paper grape_oct_2021
trace explain "Attenuation decreased with sugar accumulation." --generic   # baseline, no retrieval
```

## Eval: does grounding beat generic?

`eval/highlights.json` is a 20-highlight set over the OCT corpus. Each highlight
lists corpus facts a context-grounded explanation should surface that a generic
one structurally can't (a specific figure, a measured value, a cross-paper
contrast). `eval/run_eval.py` generates both explanations per highlight and has
an LLM judge score them blind (A/B order randomized) on *grounding*,
*cross_paper*, and *correctness*, then reports a grounded win-rate.

```bash
python -m eval.run_eval --index corpus.json --dry-run     # retrieval only, no API calls
python -m eval.run_eval --index corpus.json               # full judged run
```

## Tests

```bash
python -m pytest
```

41 tests, no network or API key required: classification signals and their
false positives, the three chunking invariants, BM25 retrieval + attribution +
persistence, an end-to-end pass over a synthetic PDF built with PyMuPDF, the
backend server (health, CORS, request validation), and the demo corpus.

## Layout

```
trace_core/     Python core — extract, classify, chunk, index, retrieve, explain, server
extension/      Chromium MV3 extension (content script, background worker, popup)
eval/           20-highlight grounded-vs-generic eval
tests/          36 tests
```
