"""Prove the whole grounded-explanation pipeline wires together end to end
with the model call mocked.

This closes the gap between "the model returns text" (the SDK's job) and
"the passage flows through retrieval, the model, and citation resolution back
out as JSON" (our job). The only thing NOT exercised here is the literal
network request to Anthropic — everything else on the path is.
"""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from trace_core.blocks import Chunk
from trace_core.index import CorpusIndex
from trace_core.server import make_handler


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeMessage:
    stop_reason = "end_turn"

    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def create(self, **kwargs):
        # Echo the retrieved context back with an [S1] citation so we can
        # verify resolve_citations maps it to the real chunk attribution.
        return _FakeMessage("The attenuation coefficient is defined in [S1].")


class _FakeClient:
    def __init__(self, *a, **k):
        self.messages = _FakeMessages()


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    # explain._client() calls anthropic.Anthropic(); replace it.
    monkeypatch.setattr("trace_core.explain.anthropic.Anthropic", _FakeClient)


@pytest.fixture
def index():
    idx = CorpusIndex()
    idx.add_chunks([
        Chunk(doc_id="grape_oct_2021", chunk_id="grape_oct_2021#0", section="Theory",
              pages=[1], text="The attenuation coefficient is obtained from a Beer-Lambert fit.",
              equation_numbers=["3"]),
    ])
    return idx


def test_grounded_explanation_resolves_citations(index):
    from trace_core.explain import explain_grounded

    text, used = explain_grounded("How is the attenuation coefficient found?", index)
    # The [S1] tag from the (mocked) model output was resolved to the chunk's
    # human-readable attribution, and that attribution is reported as a source.
    assert "[S1]" not in text
    assert used
    assert any("grape_oct_2021" in u and "Theory" in u for u in used)


def test_server_explain_returns_grounded_json(index):
    handler = make_handler(index, "mem")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/explain",
            data=json.dumps({"highlight": "How is the attenuation coefficient found?"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            assert r.status == 200
            data = json.loads(r.read())
        assert data["grounded"] is True
        assert "[S1]" not in data["explanation"]  # citation resolved
        assert data["sources"]
        assert any("grape_oct_2021" in s for s in data["sources"])
    finally:
        httpd.shutdown()


def test_server_generic_path(index):
    handler = make_handler(index, "mem")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/explain",
            data=json.dumps({"highlight": "x y z passage", "generic": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())
        assert data["grounded"] is False
        assert data["sources"] == []
        assert data["explanation"]
    finally:
        httpd.shutdown()
