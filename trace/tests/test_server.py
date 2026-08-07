"""Backend smoke tests that need no API key: liveness, corpus summary, CORS,
and request validation. The /explain success path calls the model, so it's
covered by the CLI/extension manually rather than here."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from trace_core.blocks import Chunk
from trace_core.index import CorpusIndex
from trace_core.server import make_handler


@pytest.fixture
def backend():
    idx = CorpusIndex()
    idx.add_chunks([
        Chunk(doc_id="grape", chunk_id="grape#0", section="Results", pages=[4],
              text="Attenuation of grape berries at 1300 nm."),
    ])
    handler = make_handler(idx, "mem")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, json.loads(r.read())


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_reports_corpus(backend):
    status, data = _get(backend + "/health")
    assert status == 200
    assert data["status"] == "ok"
    assert data["papers"] == ["grape"]
    assert data["chunks"] == 1


def test_health_has_cors(backend):
    with urllib.request.urlopen(backend + "/health") as r:
        assert r.headers["Access-Control-Allow-Origin"] == "*"


def test_explain_requires_highlight(backend):
    status, data = _post(backend + "/explain", {})
    assert status == 400
    assert "highlight" in data["error"]


def test_unknown_path_404(backend):
    status, data = _post(backend + "/nope", {"highlight": "x"})
    assert status == 404
