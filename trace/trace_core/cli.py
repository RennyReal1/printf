"""Trace CLI: ingest PDFs, query the corpus, explain highlights, serve the backend.

    trace ingest papers/*.pdf --index corpus.json
    trace query "attenuation coefficient grape" --index corpus.json
    trace explain "..." --index corpus.json [--generic]
    trace serve --index corpus.json          # local backend for the extension

(Equivalently `python -m trace_core <cmd>` if not pip-installed.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chunker import chunk_document
from .classify import classify_document
from .extract import extract_blocks
from .index import CorpusIndex
from .retrieve import retrieve


def ingest(paths: list[str], index_path: str, max_tokens: int) -> None:
    idx = CorpusIndex()
    for p in paths:
        doc_id = Path(p).stem
        blocks = classify_document(extract_blocks(p))
        chunks = chunk_document(blocks, doc_id, max_tokens=max_tokens)
        idx.add_chunks(chunks)
        n_caps = sum(len(c.captions) for c in chunks)
        n_eqs = sum(len(c.equation_numbers) for c in chunks)
        print(f"{doc_id}: {len(chunks)} chunks, {n_caps} captions attached, {n_eqs} numbered equations")
    idx.save(index_path)
    print(f"index -> {index_path} ({len(idx.chunks)} chunks, {len(idx.doc_ids)} papers)")


def query(q: str, index_path: str, k: int) -> None:
    idx = CorpusIndex.load(index_path)
    for s in retrieve(idx, q, k=k):
        c = s.hit.chunk
        print(f"\n[{s.tag}] {c.citation()}  (score {s.hit.score:.2f})")
        preview = c.full_text[:400].replace("\n", " ")
        print(f"  {preview}{'...' if len(c.full_text) > 400 else ''}")


def explain(highlight: str, index_path: str, k: int, generic: bool, paper: str) -> None:
    from .explain import explain_generic, explain_grounded

    if generic:
        print(explain_generic(highlight, paper_hint=paper))
        return
    idx = CorpusIndex.load(index_path)
    text, used = explain_grounded(highlight, idx, paper_hint=paper, k=k)
    print(text)
    if used:
        print("\nSources:")
        for u in used:
            print(f"  {u}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trace")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="chunk PDFs and build the corpus index")
    p_ing.add_argument("pdfs", nargs="+")
    p_ing.add_argument("--index", default="corpus.json")
    p_ing.add_argument("--max-tokens", type=int, default=450)

    p_q = sub.add_parser("query", help="BM25 search across the corpus")
    p_q.add_argument("query")
    p_q.add_argument("--index", default="corpus.json")
    p_q.add_argument("-k", type=int, default=8)

    p_e = sub.add_parser("explain", help="explain a highlighted passage")
    p_e.add_argument("highlight")
    p_e.add_argument("--index", default="corpus.json")
    p_e.add_argument("-k", type=int, default=8)
    p_e.add_argument("--generic", action="store_true", help="skip retrieval (baseline)")
    p_e.add_argument("--paper", default="", help="paper the highlight came from")

    p_s = sub.add_parser("serve", help="run the local backend for the Trace extension")
    p_s.add_argument("--index", default="corpus.json")
    p_s.add_argument("--port", type=int, default=8765)

    args = ap.parse_args(argv)
    if args.cmd == "ingest":
        ingest(args.pdfs, args.index, args.max_tokens)
    elif args.cmd == "query":
        query(args.query, args.index, args.k)
    elif args.cmd == "explain":
        explain(args.highlight, args.index, args.k, args.generic, args.paper)
    elif args.cmd == "serve":
        from .server import serve as run_serve

        run_serve(args.index, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
