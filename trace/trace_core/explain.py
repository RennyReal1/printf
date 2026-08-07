"""Highlight explanation: context-grounded (corpus retrieval) vs generic.

Both paths exist on purpose — the eval harness compares them head to head.
"""

from __future__ import annotations

import anthropic

from .index import CorpusIndex
from .retrieve import build_context, resolve_citations, retrieve

MODEL = "claude-opus-5"

_GENERIC_SYSTEM = (
    "You are helping a researcher understand a passage they highlighted in a "
    "scientific paper. Explain what the highlighted passage means and why it "
    "matters, using only your general knowledge. Be concrete and concise; "
    "aim for one or two short paragraphs."
)

_GROUNDED_SYSTEM = (
    "You are helping a researcher understand a passage they highlighted in a "
    "scientific paper. You are given retrieved excerpts from the researcher's "
    "own paper corpus, each tagged [S1], [S2], etc.\n\n"
    "Explain what the highlighted passage means and why it matters, grounding "
    "the explanation in the retrieved excerpts wherever they are relevant: "
    "connect the highlight to related methods, results, figures, and equations "
    "across the corpus. Cite every claim that comes from an excerpt with its "
    "tag, e.g. [S2]. If the excerpts contradict or refine the highlight, say "
    "so. Do not cite excerpts that are irrelevant. Be concrete and concise; "
    "aim for one or two short paragraphs."
)


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def explain_generic(highlight: str, paper_hint: str = "") -> str:
    prompt = f"Highlighted passage{f' (from {paper_hint})' if paper_hint else ''}:\n\n{highlight}"
    resp = _client().messages.create(
        model=MODEL,
        max_tokens=2048,
        system=_GENERIC_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return _text_of(resp)


def explain_grounded(
    highlight: str,
    index: CorpusIndex,
    paper_hint: str = "",
    k: int = 8,
) -> tuple[str, list[str]]:
    """Returns (explanation with resolved attributions, list of sources cited)."""
    sources = retrieve(index, highlight, k=k)
    context = build_context(sources)
    prompt = (
        f"Retrieved corpus excerpts:\n\n{context}\n\n"
        f"Highlighted passage{f' (from {paper_hint})' if paper_hint else ''}:\n\n{highlight}"
    )
    resp = _client().messages.create(
        model=MODEL,
        max_tokens=2048,
        system=_GROUNDED_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return resolve_citations(_text_of(resp), sources)


def _text_of(resp: anthropic.types.Message) -> str:
    if resp.stop_reason == "refusal":
        return "(model declined to answer this request)"
    return "".join(b.text for b in resp.content if b.type == "text").strip()
