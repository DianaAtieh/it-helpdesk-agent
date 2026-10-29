"""Local-only performance fixes (offline, no network): the shared HTTP client in the MCP service, the 127.0.0.1
defaults, and the query-embedding timing marks.

    python -m pytest tests/test_local_perf.py -v
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx

from app import timing
from app.agent import retrieval
from services import helpdesk_mcp as server
from tests.test_ws_chat import env  # noqa: F401  (fixture)

KIT = Path(__file__).resolve().parent.parent


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test", timeout=5)


def install(client: httpx.AsyncClient) -> None:
    server._client, server._client_loop = client, asyncio.get_running_loop()


# ------------------------------------------------------------------ the MCP service's shared client

def test_calls_reuse_one_client_and_keep_the_api_key_and_error_mapping(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("x-api-key")))
        if request.url.path.endswith("/missing"):
            return httpx.Response(404, json={"error": {"code": "employee_not_found", "message": "no"}})
        return httpx.Response(200, json={"ok": True})

    async def run():
        install(mock_client(handler))
        client = server._shared_client()
        results = [await server._call("GET", "/api/x"), await server._call("GET", "/api/missing"),
                   await server._call("GET", "/api/y")]
        assert server._shared_client() is client                       # not rebuilt per call
        return results

    monkeypatch.setattr(server, "API_KEY", "k-1")
    results = asyncio.run(run())
    assert results[0] == (True, {"ok": True})
    assert results[1][0] is False and results[1][1]["code"] == "employee_not_found" and results[1][1]["http_status"] == 404
    assert seen == [("/api/x", "k-1"), ("/api/missing", "k-1"), ("/api/y", "k-1")]


def test_timeouts_and_unreachable_are_still_reported_as_before(monkeypatch):
    def timing_out(request):
        raise httpx.ReadTimeout("slow", request=request)

    def unreachable(request):
        raise httpx.ConnectError("refused", request=request)

    async def run(handler):
        install(mock_client(handler))
        return await server._call("GET", "/api/x")

    assert asyncio.run(run(timing_out))[1]["code"] == "timeout"
    assert asyncio.run(run(unreachable))[1]["code"] == "unreachable"


def test_the_client_is_per_event_loop_and_closed_by_the_server_lifespan():
    clients = []

    async def one():
        clients.append(server._shared_client())
        assert server._shared_client() is clients[-1]

    asyncio.run(one())
    asyncio.run(one())
    assert clients[0] is not clients[1]                               # never reused on another loop

    async def lifespan():
        client = server._shared_client()
        async with server._lifespan(server.mcp):
            assert not client.is_closed
        assert client.is_closed and server._client is None
        assert server._shared_client() is not client                  # a fresh one on next use

    asyncio.run(lifespan())


def test_api_url_defaults_to_ipv4_loopback_not_localhost():
    # An empty value stops the kit's .env from supplying one, so this shows the built-in default.
    out = subprocess.run([sys.executable, "-c", "from services import helpdesk_mcp as s; print(s.API_URL)"], cwd=KIT,
                         env={**os.environ, "HELPDESK_API_URL": ""}, capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "http://127.0.0.1:8102", out.stderr[-300:]


def test_the_lifespan_builds_the_client_at_start_up_so_the_first_call_does_not():
    async def run():
        assert server._client is None or server._client_loop is not asyncio.get_running_loop()
        async with server._lifespan(server.mcp):
            return server._client is not None and server._client_loop is asyncio.get_running_loop()

    assert asyncio.run(run())


def test_triage_marks_each_mcp_call_separately(env, caplog):
    from tests.test_timing import first_turn, marks
    caplog.set_level(logging.INFO, logger="helpdesk-app.timing")
    from tests.fakes import FakeStreamChat
    first_turn(env, FakeStreamChat(["MODE: ANSWER\n", "Update NakhlaConnect to 5.4 from Software Center.\n",
                                    "CITATIONS: KB-002"]))
    got = marks(caplog)
    order = ["triage_start", "triage_lookup_start", "triage_lookup_end", "triage_status_start", "triage_status_end",
             "triage_end"]
    assert [m for m in order if m not in got] == []
    assert [got[m] for m in order] == sorted(got[m] for m in order)


# ------------------------------------------------------------------ the Gradio default

def test_ui_default_app_url_is_ipv4_loopback_and_still_overridable():
    code = "import ui.app_ui as u; print(u.APP_URL, u.ws_url())"
    base = {**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"}       # no network call on import
    base.pop("APP_URL", None)
    default = subprocess.run([sys.executable, "-c", code], cwd=KIT, env=base, capture_output=True, text=True, timeout=120)
    assert default.stdout.split() == ["http://127.0.0.1:8000", "ws://127.0.0.1:8000/ws/chat"], default.stderr[-300:]
    override = subprocess.run([sys.executable, "-c", code], cwd=KIT, env={**base, "APP_URL": "http://localhost:9000"},
                              capture_output=True, text=True, timeout=120)
    assert override.stdout.split() == ["http://localhost:9000", "ws://localhost:9000/ws/chat"]


# ------------------------------------------------------------------ query-embedding timing

def test_the_query_embedding_is_timed_without_logging_its_text(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    provider = retrieval.live_embeddings("openai/text-embedding-3-small")
    dims = provider.expected_dimensions
    item = SimpleNamespace(index=0, embedding=[0.1] * dims)
    provider._client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(data=[item])))
    caplog.set_level(logging.INFO, logger="helpdesk-app.timing")

    secret_query = "my-very-private-query-text"
    timer = timing.start_turn("t-embed")
    try:
        vectors = provider.embed([secret_query])
    finally:
        timing._current.set(None)                                      # do not leak the timer into other tests
    assert len(vectors) == 1 and len(vectors[0]) == dims
    assert {"embed_start", "embed_end"} <= set(timer.marks) and timer.marks["embed_end"] >= timer.marks["embed_start"]
    assert "embed_ms" in timer.phases()
    assert secret_query not in caplog.text and "sk-test" not in caplog.text


def test_embedding_failure_still_marks_the_end_and_propagates(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    provider = retrieval.live_embeddings("openai/text-embedding-3-small")

    def boom(**kw):
        raise ValueError("bad")

    provider._client = SimpleNamespace(embeddings=SimpleNamespace(create=boom))
    timer = timing.start_turn("t-fail")
    try:
        provider.embed(["q"])
        raised = False
    except Exception:
        raised = True
    finally:
        timing._current.set(None)
    assert raised and "embed_end" in timer.marks
