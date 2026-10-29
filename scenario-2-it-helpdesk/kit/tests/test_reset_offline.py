"""POST /reset, offline: no helpdesk simulator, no LLM, no embeddings (fake chat/KB/MCP, temporary checkpoint DB).

Complements the simulator-backed tests/test_reset.py. It never calls the simulator's /admin/reset, so it cannot
clear tickets created by hand.

    python -m pytest tests/test_reset_offline.py -v
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import app.main as app_main
from app.main import app, thread_config
from tests.fakes import FakeKnowledgeBase, FakeStreamChat, kb_result
from tests.test_escalate_graph import FIELDS
from tests.test_ws_chat import (GOOD, KB_NX112, SARA, TICKET_TEXT, body, converse, env, final_of, kinds,  # noqa: F401
                                next_event, sid, wait_until_idle)

OTHER = "samir.nassar@nakhla.example"


def saved(client, session_id):
    return client.portal.call(app.state.graph.aget_state, thread_config(session_id)).values


def start(env, llm=None, kb=None):
    """A temporary session with one answered turn; returns (client, session_id, token)."""
    client = env.use(llm or FakeStreamChat(GOOD), kb)
    session_id = sid()
    token = final_of(converse(client, body(session_id)))["session_token"]
    return client, session_id, token


def reset(client, session_id, token=None, email=None, **extra):
    payload = {"session_id": session_id, **({"user": {"email": email}} if email else {})}
    return client.post("/reset", json=payload, headers={"X-Session-Token": token} if token else {}, **extra)


# ------------------------------------------------------------------ session-scoped reset

def test_reset_with_the_token_deletes_the_checkpoint_and_the_session(env):
    client, session_id, token = start(env)
    assert len(saved(client, session_id)["messages"]) == 2
    r = reset(client, session_id, token, SARA)
    assert r.status_code == 200 and r.json() == {"status": "ok", "reset": session_id}
    assert saved(client, session_id) == {} and session_id not in app_main.sessions


def test_reset_drops_a_pending_draft_and_the_old_approval_no_longer_works(env):
    kb = FakeKnowledgeBase(kb_result(KB_NX112, codes=("NX-112",)))
    client = env.use(FakeStreamChat(GOOD, json_replies=(FIELDS,)), kb)
    session_id = sid()
    final = final_of(converse(client, body(session_id, TICKET_TEXT)))
    token, action = final["session_token"], final["pending_action"]["action_id"]
    assert final["state"] == "await_approval" and saved(client, session_id)["pending_action"]

    assert reset(client, session_id, token, SARA).status_code == 200
    assert saved(client, session_id) == {}
    approve = client.post("/api/approve", headers={"X-Session-Token": token},
                          json={"session_id": session_id, "action_id": action, "approved": True})
    assert approve.status_code in (403, 404, 409) and env.mcp.created == []      # nothing was filed


def test_reset_leaves_other_sessions_untouched(env):
    client, session_a, token_a = start(env)
    session_b = sid()
    token_b = final_of(converse(client, body(session_b, email=OTHER)))["session_token"]
    reset(client, session_a, token_a, SARA)
    state_b = saved(client, session_b)
    assert state_b["user_email"] == OTHER and len(state_b["messages"]) == 2 and session_b in app_main.sessions
    assert final_of(converse(client, body(session_b, email=OTHER, token=token_b)))["state"] == "answered"


def test_the_session_id_can_be_reused_after_reset_by_another_employee(env):
    client, session_id, token = start(env)
    assert reset(client, session_id, token, SARA).status_code == 200
    final = final_of(converse(client, body(session_id, email=OTHER)))
    assert final["state"] == "answered" and saved(client, session_id)["user_email"] == OTHER
    assert [m.content for m in saved(client, session_id)["messages"]][::2] == ["My VPN keeps disconnecting"]


def test_a_stale_token_after_reset_does_not_get_access_to_the_new_conversation(env):
    """Observation: what happens to the OLD token once the session id is reused."""
    client, session_id, old_token = start(env)
    reset(client, session_id, old_token, SARA)
    new = final_of(converse(client, body(session_id, email=OTHER)))
    new_token = new["session_token"]
    assert new_token and new_token != old_token                       # a fresh conversation gets a fresh token
    stale = converse(client, body(session_id, email=OTHER, token=old_token))
    assert kinds(stale) == ["error"] and stale[0]["code"] == "unauthorized"


def test_reset_of_an_unknown_session_is_harmless(env):
    client = env.use(FakeStreamChat(GOOD))
    assert reset(client, sid()).status_code == 200


# ------------------------------------------------------------------ unauthorized resets

def test_a_reset_without_or_with_the_wrong_token_is_refused_and_deletes_nothing(env, monkeypatch):
    monkeypatch.delenv("APP_ADMIN_TOKEN", raising=False)
    client, session_id, token = start(env)
    other_token = final_of(converse(client, body(sid())))["session_token"]
    client.cookies.clear()
    for presented in (None, "x" * 43, other_token):
        r = reset(client, session_id, presented, SARA)                 # the email alone is not enough
        assert r.status_code == 403, presented
        assert token not in r.text
    assert len(saved(client, session_id)["messages"]) == 2 and session_id in app_main.sessions
    assert final_of(converse(client, body(session_id, token=token)))["state"] == "answered"


def test_reset_all_needs_the_admin_token_and_deletes_nothing_without_it(env, monkeypatch):
    client, session_id, token = start(env)
    monkeypatch.delenv("APP_ADMIN_TOKEN", raising=False)
    assert client.post("/reset", json={}).status_code == 403            # disabled when no admin token is set
    assert client.post("/reset").status_code == 403
    monkeypatch.setenv("APP_ADMIN_TOKEN", "the-admin-token")
    assert client.post("/reset", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.post("/reset", headers={"X-Session-Token": token}).status_code == 403   # a session token is not admin
    assert len(saved(client, session_id)["messages"]) == 2


def test_the_reset_request_is_validated(env):
    client = env.use(FakeStreamChat(GOOD))
    assert client.post("/reset", json={"session_id": "bad id with spaces!"}).status_code == 422
    assert client.post("/reset", json={"session_id": sid(), "user": {"email": "x"}}).status_code == 422


# ------------------------------------------------------------------ reset during an active turn

def test_a_reset_during_a_websocket_turn_waits_for_it_and_the_conversation_does_not_come_back(env):
    release = threading.Event()

    async def gate():
        for _ in range(500):
            if release.is_set():
                return
            await asyncio.sleep(0.01)

    second = ["MODE: ANSWER\n", "Update NakhlaConnect to 5.4 from Software Center. ", gate,
              "Then restart it.\n", "CITATIONS: KB-002"]
    client, session_id, token = start(env, FakeStreamChat(GOOD, second))
    out = {}

    def do_reset():
        out["response"] = reset(client, session_id, token, SARA)
        out["done_at"] = time.monotonic()

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text(json.dumps(body(session_id, token=token)))
        assert next_event(ws)["type"] == "chunk"                         # the turn is running and holds the lock
        worker = threading.Thread(target=do_reset)
        worker.start()
        time.sleep(0.4)
        assert "response" not in out                                     # reset is waiting, not racing the turn
        released_at = time.monotonic()
        release.set()
        while next_event(ws)["type"] != "final":
            pass
    worker.join(10)
    assert out["response"].status_code == 200 and out["done_at"] >= released_at
    assert saved(client, session_id) == {} and session_id not in app_main.sessions   # the finished turn did not resurrect it
    wait_until_idle_or_gone(session_id)


def wait_until_idle_or_gone(session_id):
    session = app_main.sessions.get(session_id)
    if session:
        wait_until_idle(session_id)


def test_a_reset_that_waits_too_long_gets_409_and_deletes_nothing(env, monkeypatch):
    monkeypatch.setattr(app_main, "lock_wait_seconds", lambda: 0.1)
    client, session_id, token = start(env)
    session = app_main.sessions[session_id]
    client.portal.call(session["lock"].acquire)                          # a turn "still running"
    try:
        r = reset(client, session_id, token, SARA)
    finally:
        session["lock"].release()
    assert r.status_code == 409
    assert len(saved(client, session_id)["messages"]) == 2 and session_id in app_main.sessions
    assert not session["lock"].locked() and session["waiting"] == 0       # the refused reset leaked nothing


def test_a_failed_deletion_is_a_500_and_keeps_the_session_and_frees_the_lock(env, monkeypatch):
    client, session_id, token = start(env)

    async def broken(thread_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(app.state.checkpointer, "adelete_thread", broken)
    r = reset(client, session_id, token, SARA)
    assert r.status_code == 500 and "locked" not in r.text                # no internal detail
    monkeypatch.undo()
    assert len(saved(client, session_id)["messages"]) == 2 and not app_main.sessions[session_id]["lock"].locked()
