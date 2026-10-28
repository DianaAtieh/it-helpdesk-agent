"""Who may approve: the per-session token. Run from the kit folder:

    python -m pytest tests/test_approve_auth.py -v

The first /api/chat of a new conversation issues a random token (X-Session-Token header + HttpOnly cookie);
the server keeps only its hash, in the checkpoint. /api/approve (approve AND reject) needs it back: without
a valid one the answer is 403, the draft is untouched and create_ticket is never called.
sessions.clear() simulates a restart or an eviction.

Setup and fixtures: tests/conftest.py (needs the helpdesk API on port 8102).
"""
from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

import app.main as main
from app.main import SESSION_TOKEN_HEADER, app, session_cookie_name, sessions, thread_config
from tests.helpers import TOKENS, created_tickets, ensure_token, make_draft, propose, requires_api

pytestmark = [requires_api, pytest.mark.usefixtures("clean")]

SARA = {"email": "sara.mikhail@nakhla.example"}


def post_chat(client, session_id: str, message: str = "Is the VPN down?", user: dict | None = SARA):
    return client.post("/api/chat", json={"session_id": session_id, "message": message,
                                          **({"user": user} if user else {})})


def post_approve(client, session_id: str, action_id: str, approved: bool = True, token: str | None = None):
    headers = {SESSION_TOKEN_HEADER: token} if token else {}
    return client.post("/api/approve", json={"session_id": session_id, "action_id": action_id,
                                             "approved": approved}, headers=headers)


def saved(client, session_id: str) -> dict:
    return client.portal.call(app.state.graph.aget_state, thread_config(session_id)).values


@pytest.fixture
def tool_calls(monkeypatch) -> list[str]:
    """Every MCP tool the app calls during the test, in order."""
    seen: list[str] = []
    real = main.mcp.call

    async def spy(tool, args=None, **kwargs):
        seen.append(tool)
        return await real(tool, args, **kwargs)

    monkeypatch.setattr(main.mcp, "call", spy)
    return seen


def assert_nothing_changed(client, session_id: str, action, tool_calls: list[str]) -> None:
    assert saved(client, session_id)["pending_action"]["action_id"] == action.action_id     # draft untouched
    assert sessions[session_id]["pending_action"].action_id == action.action_id
    assert "create_ticket" not in tool_calls
    assert created_tickets() == []


# ------------------------------------------------------------- issuing

def test_the_first_turn_issues_a_token_only_in_a_header_and_an_httponly_cookie(client, caplog):
    with caplog.at_level(logging.INFO):
        r = post_chat(client, "s-1")
    token = r.headers[SESSION_TOKEN_HEADER]
    assert len(token) >= 40 and r.headers["Cache-Control"] == "no-store"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{session_cookie_name('s-1')}={token}") and "HttpOnly" in cookie
    assert "samesite=strict" in cookie.lower() and "Secure" not in cookie                    # plain-http default
    assert token not in r.text                                                               # not in the body
    assert all(token not in record.getMessage() for record in caplog.records)                # not in the logs
    state = saved(client, "s-1")
    assert state["token_hash"] == main.hash_token(token) and token not in str(state)          # only the hash


def test_a_token_is_never_issued_twice_for_one_conversation(client):
    assert SESSION_TOKEN_HEADER in post_chat(client, "s-1").headers
    assert SESSION_TOKEN_HEADER not in post_chat(client, "s-1").headers
    sessions.clear()                                                           # after a restart
    client.cookies.clear()                                                     # someone who only knows the id
    r = post_chat(client, "s-1")
    assert SESSION_TOKEN_HEADER not in r.headers and "set-cookie" not in r.headers


def test_a_failed_first_turn_issues_no_token_and_the_next_turn_does(client, monkeypatch):
    async def broken(payload, config):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(app.state.graph, "ainvoke", broken)
    r = post_chat(client, "s-1")
    assert r.json()["state"] == "error" and SESSION_TOKEN_HEADER not in r.headers
    monkeypatch.undo()
    assert SESSION_TOKEN_HEADER in post_chat(client, "s-1").headers


def test_the_secure_flag_is_set_when_configured(client, monkeypatch):
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "true")
    assert "Secure" in post_chat(client, "s-1").headers["set-cookie"]


# ------------------------------------------------------------- the owner

@pytest.mark.parametrize("approved", [True, False], ids=["approve", "reject"])
def test_the_owner_can_approve_and_reject(client, approved):
    action = propose(client, "s-1", make_draft())
    r = post_approve(client, "s-1", action.action_id, approved, token=TOKENS["s-1"])
    assert r.status_code == 200
    assert len(created_tickets()) == (1 if approved else 0)
    assert saved(client, "s-1")["pending_action"] is None


def test_a_cookie_keeping_client_works_unchanged_like_the_eval_runner(client):
    # evals/run_evals.py: one httpx.Client for every request, no headers of its own. The TestClient is an
    # httpx.Client too: the cookie from the first chat turn goes back with /api/approve by itself.
    post_chat(client, "s-1", "my VPN fails with NX-112")
    action = propose(client, "s-1", make_draft())                              # no new token: the chat gave it
    assert "s-1" not in TOKENS
    r = post_approve(client, "s-1", action.action_id)                          # no header: the cookie only
    assert r.status_code == 200 and r.json()["result"]["ticket_id"].startswith("IT-")


def test_two_sessions_in_one_cookie_jar_each_use_their_own_cookie(client):
    post_chat(client, "s-1")
    post_chat(client, "s-2")
    assert {session_cookie_name("s-1"), session_cookie_name("s-2")} <= set(client.cookies)
    action = propose(client, "s-1", make_draft())
    assert post_approve(client, "s-1", action.action_id).status_code == 200


def test_authorization_survives_a_restart(client):
    action = propose(client, "s-1", make_draft())
    sessions.clear()                                                           # the hash is read back from disk
    r = post_approve(client, "s-1", action.action_id, token=TOKENS["s-1"])
    assert r.status_code == 200 and len(created_tickets()) == 1


# ------------------------------------------------------------- everyone else: 403, nothing changes

@pytest.mark.parametrize("approved", [True, False], ids=["approve", "reject"])
def test_another_sessions_token_cannot_approve_or_reject(client, tool_calls, approved):
    action = propose(client, "s-1", make_draft())
    other = ensure_token(client, "s-2")
    r = post_approve(client, "s-1", action.action_id, approved, token=other)
    assert r.status_code == 403 and r.json()["state"] == "unauthorized" and r.json()["result"] is None
    assert_nothing_changed(client, "s-1", action, tool_calls)


@pytest.mark.parametrize("approved", [True, False], ids=["approve", "reject"])
def test_a_missing_token_is_refused(client, tool_calls, approved):
    action = propose(client, "s-1", make_draft())
    r = post_approve(client, "s-1", action.action_id, approved)                # no header, no cookie
    assert r.status_code == 403 and "not sent" in r.json()["reply"]
    assert_nothing_changed(client, "s-1", action, tool_calls)


@pytest.mark.parametrize("approved", [True, False], ids=["approve", "reject"])
def test_an_invalid_token_is_refused(client, tool_calls, approved):
    action = propose(client, "s-1", make_draft())
    r = post_approve(client, "s-1", action.action_id, approved, token=TOKENS["s-1"] + "x")
    assert r.status_code == 403 and "does not belong" in r.json()["reply"]
    assert_nothing_changed(client, "s-1", action, tool_calls)


@pytest.mark.parametrize("approved", [True, False], ids=["approve", "reject"])
def test_an_expired_token_is_refused_and_says_what_to_do(client, tool_calls, monkeypatch, approved):
    action = propose(client, "s-1", make_draft())
    monkeypatch.setenv("SESSION_TOKEN_TTL_SECONDS", "0")
    r = post_approve(client, "s-1", action.action_id, approved, token=TOKENS["s-1"])
    assert r.status_code == 403 and "expired" in r.json()["reply"]
    assert "start a new conversation" in r.json()["reply"]
    assert_nothing_changed(client, "s-1", action, tool_calls)


def test_a_wrong_header_is_not_rescued_by_a_valid_cookie(client, tool_calls):
    post_chat(client, "s-1")                                                   # the jar holds the valid cookie
    action = propose(client, "s-1", make_draft())
    r = post_approve(client, "s-1", action.action_id, token="not-the-token")
    assert r.status_code == 403
    assert_nothing_changed(client, "s-1", action, tool_calls)


def test_a_conversation_without_a_token_can_never_approve(client, tool_calls):
    # Saved before tokens existed (or a draft created without any chat turn): fail closed.
    action, _ = client.portal.call(main.propose_draft, "s-1", make_draft())
    assert ensure_token(client, "s-1") is None                                 # it already has a checkpoint
    r = post_approve(client, "s-1", action.action_id, token="any-token")
    assert r.status_code == 403
    assert_nothing_changed(client, "s-1", action, tool_calls)


def test_a_replay_without_the_token_does_not_reveal_the_ticket_id(client):
    action = propose(client, "s-1", make_draft())
    ticket_id = post_approve(client, "s-1", action.action_id, token=TOKENS["s-1"]).json()["result"]["ticket_id"]
    r = post_approve(client, "s-1", action.action_id)
    assert r.status_code == 403 and ticket_id not in r.text


# ------------------------------------------------------------- idempotency is unchanged

def test_repeated_approvals_with_the_token_still_file_once(client):
    action = propose(client, "s-1", make_draft())
    first = post_approve(client, "s-1", action.action_id, token=TOKENS["s-1"]).json()
    second = post_approve(client, "s-1", action.action_id, token=TOKENS["s-1"]).json()
    assert second["result"] == {"ticket_id": first["result"]["ticket_id"]} and "already filed" in second["reply"]
    assert len(created_tickets()) == 1


def test_concurrent_approvals_with_the_token_still_file_once(client):
    action = propose(client, "s-1", make_draft())
    body = {"session_id": "s-1", "action_id": action.action_id, "approved": True}

    async def both():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                                     headers={SESSION_TOKEN_HEADER: TOKENS["s-1"]}) as api:
            return [r.json() for r in await asyncio.gather(api.post("/api/approve", json=body),
                                                           api.post("/api/approve", json=body))]

    outs = client.portal.call(both)
    ids = {out["result"]["ticket_id"] for out in outs}
    assert len(ids) == 1 and len(created_tickets()) == 1


# ------------------------------------------------------------- /reset follows the same model

def test_reset_needs_the_token_and_an_expired_token_still_proves_ownership(client, monkeypatch):
    post_chat(client, "s-1")
    jar = dict(client.cookies)
    client.cookies.clear()
    assert client.post("/reset", json={"session_id": "s-1", "user": SARA}).status_code == 403   # email: not enough
    client.cookies.update(jar)
    monkeypatch.setenv("SESSION_TOKEN_TTL_SECONDS", "0")                       # expired, but it is yours
    r = client.post("/reset", json={"session_id": "s-1"})
    assert r.status_code == 200 and saved(client, "s-1") == {}
    assert session_cookie_name("s-1") in r.headers["set-cookie"] and "Max-Age=0" in r.headers["set-cookie"]


# ------------------------------------------------------------- the Gradio UI sends the right token

def test_the_ui_keeps_the_token_per_session_and_never_sends_it_for_another(monkeypatch):
    import ui.app_ui as ui

    sent: list[dict] = []

    class FakeResponse:
        status_code = 200

        def __init__(self, headers):
            self.headers = headers

        def json(self):
            return {"reply": "ok", "pending_action": None}

    def fake_post(url, json, headers, timeout):
        sent.append({"url": url, "headers": dict(headers)})
        return FakeResponse({SESSION_TOKEN_HEADER: "tok-A"} if url.endswith("/api/chat") else {})

    monkeypatch.setattr(ui.httpx, "post", fake_post)
    *_, auth = ui.send("hi", [], "ui-A", "sara.mikhail@nakhla.example", None)
    assert auth == {"session_id": "ui-A", "token": "tok-A"}
    assert sent[-1]["headers"] == {}                                           # chat sends no token

    pending = {"action_id": "act-1"}
    ui.decide(True, [], "ui-A", pending, auth)
    assert sent[-1]["headers"] == {SESSION_TOKEN_HEADER: "tok-A"}              # its own session: sent
    ui.decide(True, [], "ui-B", pending, auth)
    assert sent[-1]["headers"] == {}                                           # another session: never sent
