"""The Gradio UI's streaming chat handler against a FAKE /ws/chat server (offline, no app, no LLM).

The fake is a real WebSocket server on an ephemeral localhost port, so the UI's real client code runs: the
handshake header, the one JSON frame, event parsing, cancellation. The fake speaks the protocol documented
in app/main.py.

    python -m pytest tests/test_ui_streaming.py -v
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
import websockets.asyncio.server

KIT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("app_ui", KIT / "ui" / "app_ui.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

SID = "ui-" + "a" * 32
TOKEN = "tok-" + "t" * 40
REPLY = "Update NakhlaConnect to 5.4. Then restart it."
PENDING = {"action_id": "act-1", "type": "create_ticket", "summary": "Open a ticket",
           "payload": {"subject": "VPN", "requester_email": "sara.mikhail@nakhla.example"}}


def final(reply=REPLY, seq=3, turn="t1", **extra):
    return {"type": "final", "turn_id": turn, "seq": seq, "provisional": False, "display": "keep", "reply": reply,
            "citations": [{"id": "KB-002", "title": "VPN client versions", "url": None}], "state": "answered",
            "session_id": SID, "tool_calls": [], **extra}


def chunk(text, seq, turn="t1"):
    return {"type": "chunk", "turn_id": turn, "seq": seq, "provisional": True, "text": text}


class FakeServer:
    """Plays `script` (a list of events, or callables taking the connection) for each connection."""

    def __init__(self, script):
        self.script, self.requests, self.headers, self.server = script, [], [], None
        self.closed = asyncio.Event()

    async def handler(self, ws):
        self.headers.append(ws.request.headers.get("X-Session-Token"))
        self.requests.append(json.loads(await ws.recv()))
        try:
            for item in self.script:
                if callable(item):
                    await item(ws)
                else:
                    await ws.send(json.dumps(item))
            await ws.close()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.closed.set()

    async def __aenter__(self):
        self.server = await websockets.asyncio.server.serve(self.handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()


@pytest.fixture
def point_at(monkeypatch):
    def apply(port):
        monkeypatch.setattr(ui, "APP_URL", f"http://127.0.0.1:{port}")
    return apply


async def run(message="My VPN drops", history=None, auth=None, email="sara.mikhail@nakhla.example"):
    out = []
    async for step in ui.send_stream(message, history or [], SID, email, auth):
        out.append(step)
    return out


def chat_of(step):
    return step[1]


def test_chunks_are_shown_in_order_then_replaced_by_the_verified_final(point_at):
    async def go():
        async with FakeServer([chunk("Update NakhlaConnect to 5.4. ", 1), chunk("Then restart it.", 2),
                               final(session_token=TOKEN)]) as srv:
            point_at(srv.port)
            return srv, await run()
    srv, steps = asyncio.run(go())
    chats = [s[1] for s in steps if isinstance(s[1], list)]
    partials = [c[-1]["content"] for c in chats if c and c[-1]["role"] == "assistant"]
    assert partials[0] == "Update NakhlaConnect to 5.4. " + ui.STREAMING_CURSOR
    assert partials[1] == "Update NakhlaConnect to 5.4. Then restart it." + ui.STREAMING_CURSOR
    last = chats[-1]
    assert [m["role"] for m in last] == ["user", "assistant"] and last[-1]["content"] == REPLY   # no duplicate bubble
    assert srv.requests == [{"type": "chat", "session_id": SID, "message": "My VPN drops",
                             "user": {"email": "sara.mikhail@nakhla.example"}}]
    assert steps[-1][-1] == {"session_id": SID, "token": TOKEN}                                   # token persisted
    assert "KB-002" in steps[-1][6]                                                               # citations panel


def test_chunks_reach_the_ui_before_the_server_has_finished(point_at):
    """The server holds the final back until the UI has displayed the first chunk: no replay of a finished answer."""
    async def go():
        gate = asyncio.Event()

        async def wait_for_ui(ws):
            await asyncio.wait_for(gate.wait(), 5)

        async with FakeServer([chunk("Update NakhlaConnect to 5.4. ", 1), wait_for_ui, final(seq=2)]) as srv:
            point_at(srv.port)
            seen = []
            async for step in ui.send_stream("My VPN drops", [], SID, "sara.mikhail@nakhla.example", None):
                if isinstance(step[1], list) and step[1] and step[1][-1]["role"] == "assistant":
                    seen.append(step[1][-1]["content"])
                    if len(seen) == 1:
                        assert not srv.closed.is_set()       # the server is still "generating"
                        gate.set()
            return seen
    seen = asyncio.run(go())
    assert seen[0].startswith("Update NakhlaConnect") and seen[-1] == REPLY


def status(label, seq, turn="t1", stage="knowledge_base"):
    return {"type": "status", "turn_id": turn, "seq": seq, "stage": stage, "label": label}


def assistant_texts(steps):
    return [s[1][-1]["content"] for s in steps
            if isinstance(s[1], list) and s[1] and s[1][-1]["role"] == "assistant"]


def test_status_is_shown_then_replaced_by_the_first_chunk_and_never_persists(point_at):
    async def go():
        async with FakeServer([status("Checking employee information", 1), status("Searching the knowledge base", 2),
                               chunk("Update NakhlaConnect to 5.4. ", 3), chunk("Then restart it.", 4),
                               final(seq=5)]) as srv:
            point_at(srv.port)
            return await run()
    texts = assistant_texts(asyncio.run(go()))
    assert ui.STATUS_PREFIX in texts[0] and "Checking employee information" in texts[0]
    assert "Searching the knowledge base" in texts[1]
    assert texts[2] == "Update NakhlaConnect to 5.4. " + ui.STREAMING_CURSOR          # status gone at first chunk
    assert all(ui.STATUS_PREFIX not in t for t in texts[2:])
    assert texts[-1] == REPLY and ui.STATUS_PREFIX not in texts[-1]


def test_a_status_arriving_after_a_chunk_is_ignored(point_at):
    async def go():
        async with FakeServer([chunk("Update NakhlaConnect to 5.4. ", 1), status("Searching the knowledge base", 2),
                               final(seq=3)]) as srv:
            point_at(srv.port)
            return await run()
    texts = assistant_texts(asyncio.run(go()))
    assert all(ui.STATUS_PREFIX not in t for t in texts) and texts[-1] == REPLY


def test_a_status_then_final_without_chunks_leaves_only_the_verified_reply(point_at):
    async def go():
        async with FakeServer([status("Preparing the answer", 1, stage="preparing"), final(seq=2)]) as srv:
            point_at(srv.port)
            return await run()
    steps = asyncio.run(go())
    texts = assistant_texts(steps)
    assert ui.STATUS_PREFIX in texts[0] and texts[-1] == REPLY
    assert [m["role"] for m in steps[-1][1]] == ["user", "assistant"]


def test_malformed_or_oversized_status_labels_are_bounded_or_ignored(point_at):
    async def go():
        async with FakeServer([status(None, 1), status("x" * 500, 2), final(seq=3)]) as srv:
            point_at(srv.port)
            return await run()
    texts = assistant_texts(asyncio.run(go()))
    assert all(len(t) < 200 for t in texts) and texts[-1] == REPLY


def test_retract_removes_the_provisional_text_and_final_replaces_it(point_at):
    retract = {"type": "retract", "turn_id": "t1", "seq": 2, "reason": "policy"}

    async def go():
        async with FakeServer([chunk("Skip the posture check. ", 1), retract,
                               final("I can't help with that.", seq=3, state="policy_refusal")]) as srv:
            point_at(srv.port)
            return await run()
    steps = asyncio.run(go())
    chats = [s[1] for s in steps if isinstance(s[1], list)]
    assert chats[-2] == chats[0] and [m["role"] for m in chats[-2]] == ["user"]          # bubble removed
    assert not any("posture" in m["content"] for m in chats[-1])
    assert chats[-1][-1]["content"] == "I can't help with that."


def test_the_existing_sessions_token_goes_in_a_header_never_the_url(point_at):
    async def go():
        async with FakeServer([final(seq=1)]) as srv:
            point_at(srv.port)
            await run(auth={"session_id": SID, "token": TOKEN})
            await run(auth={"session_id": "ui-someone-else", "token": "other-token"})
            return srv
    srv = asyncio.run(go())
    assert srv.headers == [TOKEN, None]                       # another session's token is never sent
    assert TOKEN not in ui.ws_url() and "?" not in ui.ws_url()
    assert all("token" not in r for r in srv.requests)


def test_a_draft_final_enables_approval_and_keeps_the_card(point_at):
    async def go():
        async with FakeServer([final("Draft ready.", seq=1, state="await_approval", pending_action=PENDING)]) as srv:
            point_at(srv.port)
            return await run()
    last = asyncio.run(go())[-1]
    assert last[2] == PENDING and "Ticket draft" in last[3]
    assert last[4]["interactive"] is True and last[5]["interactive"] is True


def test_decide_still_posts_to_api_approve_with_the_stored_token(monkeypatch):
    calls = []

    def fake_call(path, body, token=None):
        calls.append((path, body, token))
        return {"reply": "Ticket IT-1 created.", "state": "ticket_created"}, None

    monkeypatch.setattr(ui, "call_app", fake_call)
    out = ui.decide(True, [], SID, PENDING, {"session_id": SID, "token": TOKEN})
    assert calls == [("/api/approve", {"session_id": SID, "action_id": "act-1", "approved": True}, TOKEN)]
    assert out[0][-1]["content"] == "Ticket IT-1 created."


def test_an_app_error_is_shown_safely_and_keeps_the_pending_draft(point_at):
    err = {"type": "error", "turn_id": "t1", "seq": 1, "code": "busy", "message": "Please send it again."}

    async def go():
        async with FakeServer([err]) as srv:
            point_at(srv.port)
            return await run()
    last = asyncio.run(go())[-1]
    assert last[1][-1]["content"] == "Please send it again."
    assert last[2] == ui.gr.skip()                          # pending_state untouched, not cleared


def test_a_connection_lost_mid_answer_is_reported_and_not_retried(point_at):
    async def drop(ws):
        await ws.close()

    async def go():
        async with FakeServer([chunk("Update ", 1), drop]) as srv:
            point_at(srv.port)
            return srv, await run()
    srv, steps = asyncio.run(go())
    assert "connection" in steps[-1][1][-1]["content"].lower() and len(srv.requests) == 1


def test_an_idle_server_times_out(point_at, monkeypatch):
    monkeypatch.setattr(ui, "WS_IDLE_SECONDS", 0.2)

    async def hang(ws):
        await asyncio.sleep(2)

    async def go():
        async with FakeServer([hang]) as srv:
            point_at(srv.port)
            return await run()
    assert "too long" in asyncio.run(go())[-1][1][-1]["content"]


def test_garbage_and_replayed_events_are_ignored_or_reported(point_at):
    async def go():
        async with FakeServer([chunk("a", 1, turn="t1"), chunk("EVIL", 2, turn="other"), chunk("dup", 1),
                               final(seq=3)]) as srv:
            point_at(srv.port)
            return await run()
    steps = asyncio.run(go())
    assert not any("EVIL" in m["content"] or "dup" in m["content"]
                   for s in steps if isinstance(s[1], list) for m in s[1])

    async def garbage(ws):
        await ws.send("not json")

    async def go2():
        async with FakeServer([garbage]) as srv:
            point_at(srv.port)
            return await run()
    assert "unreadable" in asyncio.run(go2())[-1][1][-1]["content"]


def test_ui_cancellation_closes_the_connection(point_at):
    async def go():
        async def linger(ws):
            await ws.wait_closed()

        async with FakeServer([chunk("a", 1), linger, final(seq=2)]) as srv:
            point_at(srv.port)
            agen = ui.send_stream("hi", [], SID, "", None)
            await agen.__anext__()
            await agen.__anext__()                          # first chunk shown
            await agen.aclose()                             # the user left / cancelled
            await asyncio.wait_for(srv.closed.wait(), 3)
    asyncio.run(go())


def test_websocket_unavailable_falls_back_to_http_and_says_so(point_at, monkeypatch):
    calls = []

    def fake_call(path, body, token=None):
        calls.append((path, token))
        return {"reply": REPLY, "state": "answered"}, TOKEN

    monkeypatch.setattr(ui, "call_app", fake_call)
    monkeypatch.setattr(ui, "APP_URL", "http://127.0.0.1:9")          # nothing listens there
    steps = asyncio.run(run())
    assert calls == [("/api/chat", None)]
    assert steps[-1][1][-1]["content"].startswith(ui.FALLBACK_NOTE) and REPLY in steps[-1][1][-1]["content"]
    assert steps[-1][-1] == {"session_id": SID, "token": TOKEN}


def test_the_http_fallback_sends_this_tabs_token_on_later_turns(point_at, monkeypatch):
    calls = []

    def fake_call(path, body, token=None):
        calls.append((path, token))
        return {"reply": REPLY, "state": "answered"}, None

    monkeypatch.setattr(ui, "call_app", fake_call)
    monkeypatch.setattr(ui, "APP_URL", "http://127.0.0.1:9")
    asyncio.run(run(auth={"session_id": SID, "token": TOKEN}))
    asyncio.run(run(auth={"session_id": "ui-" + "b" * 32, "token": TOKEN}))      # another tab's token is never sent
    assert calls == [("/api/chat", TOKEN), ("/api/chat", None)]


def test_the_sync_send_path_sends_this_tabs_token(monkeypatch):
    calls = []

    def fake_call(path, body, token=None):
        calls.append((path, token))
        return {"reply": REPLY, "state": "answered"}, None

    monkeypatch.setattr(ui, "call_app", fake_call)
    ui.send("hello", [], SID, "sara.mikhail@nakhla.example", {"session_id": SID, "token": TOKEN})
    ui.send("hello", [], SID, "sara.mikhail@nakhla.example", None)
    assert calls == [("/api/chat", TOKEN), ("/api/chat", None)]


def test_an_unauthorized_websocket_is_not_retried_over_http(point_at, monkeypatch):
    monkeypatch.setattr(ui, "call_app", lambda *a, **k: pytest.fail("fallback must not run"))
    err = {"type": "error", "turn_id": "t1", "seq": 1, "code": "unauthorized", "message": "Session token invalid."}

    async def go():
        async with FakeServer([err]) as srv:
            point_at(srv.port)
            return await run()
    assert asyncio.run(go())[-1][1][-1]["content"] == "Session token invalid."


def test_empty_messages_do_nothing(point_at):
    steps = asyncio.run(run("   "))
    assert len(steps) == 1 and steps[0][0] == ""


def test_ws_url_follows_the_app_url(monkeypatch):
    monkeypatch.setattr(ui, "APP_URL", "http://localhost:8000")
    assert ui.ws_url() == "ws://localhost:8000/ws/chat"
    monkeypatch.setattr(ui, "APP_URL", "https://helpdesk.example")
    assert ui.ws_url() == "wss://helpdesk.example/ws/chat"
