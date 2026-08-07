"""Eval: does context-grounded explanation beat generic on the highlight set?

For each of the 20 highlights we generate two explanations — one grounded in
corpus retrieval, one from general knowledge only — and have an LLM judge
score both (blind to which is which, order randomized) on:

  grounding:    are claims tied to specific corpus facts (figures, equations,
                measured values) rather than generic background?
  cross_paper:  does it connect the highlight to *other* papers in the corpus?
  correctness:  is it faithful to the retrieved material / not fabricated?

The judge also reports a head-to-head `winner`. The headline metric is the
grounded win-rate on `cross_paper` and `grounding` — the two axes a stranger's
generic explanation structurally cannot win.

Usage:
  python -m eval.run_eval --index corpus.json --highlights eval/highlights.json
  python -m eval.run_eval ... --dry-run     # no API calls; checks retrieval only
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import anthropic

from pdfcorpus.explain import explain_generic, explain_grounded
from pdfcorpus.index import CorpusIndex
from pdfcorpus.retrieve import retrieve

JUDGE_MODEL = "claude-opus-5"

_JUDGE_SYSTEM = (
    "You are a rigorous evaluator comparing two explanations of a passage a "
    "researcher highlighted in a scientific paper. Score each explanation on "
    "three axes from 1-5:\n"
    "  grounding: claims tied to specific, concrete facts (named figures, "
    "equations, measured values, methods) rather than generic textbook "
    "background.\n"
    "  cross_paper: connects the highlight to other papers/results beyond the "
    "one it came from.\n"
    "  correctness: faithful and non-fabricated.\n\n"
    "Then pick an overall winner. Judge only on explanatory quality for a "
    "researcher who knows the field; do not reward length or hedging."
)

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation_a": {"$ref": "#/$defs/scores"},
        "explanation_b": {"$ref": "#/$defs/scores"},
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["explanation_a", "explanation_b", "winner", "reason"],
    "additionalProperties": False,
    "$defs": {
        "scores": {
            "type": "object",
            "properties": {
                "grounding": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "cross_paper": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "correctness": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            },
            "required": ["grounding", "cross_paper", "correctness"],
            "additionalProperties": False,
        }
    },
}


def judge(highlight: str, expects: list[str], a: str, b: str) -> dict:
    client = anthropic.Anthropic()
    prompt = (
        f"Highlighted passage:\n{highlight}\n\n"
        f"A good explanation would surface facts like: {'; '.join(expects)}\n\n"
        f"--- Explanation A ---\n{a}\n\n"
        f"--- Explanation B ---\n{b}\n\n"
        "Score both and pick a winner."
    )
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        system=_JUDGE_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def run(index_path: str, highlights_path: str, dry_run: bool, k: int, seed: int) -> dict:
    idx = CorpusIndex.load(index_path)
    data = json.loads(Path(highlights_path).read_text())
    rng = random.Random(seed)

    results = []
    grounded_wins = generic_wins = ties = 0
    sum_grounded = {"grounding": 0, "cross_paper": 0, "correctness": 0}
    sum_generic = {"grounding": 0, "cross_paper": 0, "correctness": 0}

    for h in data["highlights"]:
        sources = retrieve(idx, h["text"], k=k)
        retrieved_docs = sorted({s.hit.chunk.doc_id for s in sources})
        if dry_run:
            results.append({"id": h["id"], "retrieved_docs": retrieved_docs, "n_sources": len(sources)})
            continue

        grounded, used = explain_grounded(h["text"], idx, paper_hint=h["paper"], k=k)
        generic = explain_generic(h["text"], paper_hint=h["paper"])

        # Randomize A/B so the judge can't learn a position bias.
        grounded_is_a = rng.random() < 0.5
        a, b = (grounded, generic) if grounded_is_a else (generic, grounded)
        verdict = judge(h["text"], h["expects"], a, b)

        g_scores = verdict["explanation_a"] if grounded_is_a else verdict["explanation_b"]
        n_scores = verdict["explanation_b"] if grounded_is_a else verdict["explanation_a"]
        for ax in sum_grounded:
            sum_grounded[ax] += g_scores[ax]
            sum_generic[ax] += n_scores[ax]

        w = verdict["winner"]
        grounded_won = (w == "A" and grounded_is_a) or (w == "B" and not grounded_is_a)
        generic_won = (w == "A" and not grounded_is_a) or (w == "B" and grounded_is_a)
        grounded_wins += grounded_won
        generic_wins += generic_won
        ties += w == "tie"

        results.append({
            "id": h["id"],
            "paper": h["paper"],
            "retrieved_docs": retrieved_docs,
            "grounded_scores": g_scores,
            "generic_scores": n_scores,
            "winner": "grounded" if grounded_won else "generic" if generic_won else "tie",
            "sources_cited": used,
        })

    n = len(data["highlights"])
    summary = {
        "n": n,
        "dry_run": dry_run,
        "results": results,
    }
    if not dry_run:
        summary["headline"] = {
            "grounded_win_rate": grounded_wins / n,
            "generic_win_rate": generic_wins / n,
            "tie_rate": ties / n,
            "avg_grounded": {ax: sum_grounded[ax] / n for ax in sum_grounded},
            "avg_generic": {ax: sum_generic[ax] / n for ax in sum_generic},
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="corpus.json")
    ap.add_argument("--highlights", default="eval/highlights.json")
    ap.add_argument("--out", default="eval/results.json")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="retrieval only, no API calls")
    args = ap.parse_args(argv)

    summary = run(args.index, args.highlights, args.dry_run, args.k, args.seed)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        cross = sum(1 for r in summary["results"] if len(r["retrieved_docs"]) > 1)
        print(f"dry run: {summary['n']} highlights, "
              f"{cross} retrieved from >1 paper (cross-paper potential)")
    else:
        h = summary["headline"]
        print(f"grounded win-rate: {h['grounded_win_rate']:.0%}  "
              f"generic: {h['generic_win_rate']:.0%}  tie: {h['tie_rate']:.0%}")
        print(f"avg grounding  grounded={h['avg_grounded']['grounding']:.2f} "
              f"generic={h['avg_generic']['grounding']:.2f}")
        print(f"avg cross_paper grounded={h['avg_grounded']['cross_paper']:.2f} "
              f"generic={h['avg_generic']['cross_paper']:.2f}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
