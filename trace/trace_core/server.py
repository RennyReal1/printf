"""Local Trace backend for the browser extension.

The extension is a thin front-end: it captures the passage you highlight and
POSTs it here. This server holds the corpus index and calls the model, then
returns the grounded explanation and its sources. It runs on your machine, on
your API key — nothing leaves your laptop except the model request.

Stdlib only (http.server) so there is no extra dependency to install beyond
what the corpus already needs.

    trace serve --index corpus.json [--port 8765]

Endpoints (all responses JSON, permissive CORS for the extension):

    GET  /health   -> {status, papers, chunks}   liveness + corpus summary
    POST /explain  -> {highlight, paper?, k?}     grounded explanation + sources
                      also accepts {"generic": true} for the baseline
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .index import CorpusIndex

DEFAULT_PORT = 8765


def make_handler(index: CorpusIndex, index_path: str):
    class TraceHandler(BaseHTTPRequestHandler):
        # Quieter logging: one line per request without the default noise.
        def log_message(self, fmt, *args):  # noqa: A003
            print(f"trace: {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _cors(self) -> None:
            # The extension fetches from arbitrary page origins, so allow any.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self) -> None:  # CORS preflight
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/health":
                self._send(200, {
                    "status": "ok",
                    "index": index_path,
                    "papers": index.doc_ids,
                    "chunks": len(index.chunks),
                })
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.split("?")[0] != "/explain":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send(400, {"error": "invalid JSON body"})
                return

            highlight = (data.get("highlight") or "").strip()
            if not highlight:
                self._send(400, {"error": "missing 'highlight'"})
                return
            paper = data.get("paper", "")
            k = int(data.get("k", 8))

            # Imported lazily so /health and startup don't require the SDK or a key.
            from .explain import explain_generic, explain_grounded

            try:
                if data.get("generic"):
                    text = explain_generic(highlight, paper_hint=paper)
                    self._send(200, {"explanation": text, "sources": [], "grounded": False})
                else:
                    text, used = explain_grounded(highlight, index, paper_hint=paper, k=k)
                    self._send(200, {"explanation": text, "sources": used, "grounded": True})
            except Exception as e:  # surface a clean error to the extension UI
                self._send(502, {"error": f"{type(e).__name__}: {e}"})

    return TraceHandler


def serve(index_path: str, port: int = DEFAULT_PORT) -> None:
    index = CorpusIndex.load(index_path)
    handler = make_handler(index, index_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(
        f"Trace backend on http://127.0.0.1:{port}  "
        f"({len(index.chunks)} chunks across {len(index.doc_ids)} papers)\n"
        f"Point the Trace extension at this port, then highlight a passage.\n"
        f"Ctrl-C to stop."
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping Trace backend")
        httpd.shutdown()
