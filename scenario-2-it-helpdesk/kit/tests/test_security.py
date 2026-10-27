"""Security tests for our API: one per vulnerability found in the review. Run from the kit folder:

    python -m pytest tests -v

Setup and fixtures: tests/conftest.py (needs the helpdesk API on port 8102).
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

import app.main as main
from app.main import app, get_session, sessions
from app.mcp_client import server_env
from app.schemas import MAX_MESSAGE_CHARS
from tests.helpers import arm_fault, created_tickets, make_draft, propose, requires_api, token_headers

pytestmark = [requires_api, pytest.mark.usefixtures("clean")]

SARA = {"email": "sara.mikhail@nakhla.example"}


def chat(client, session_id="s-1", message="Is the VPN down?", user=None):
    body = {"session_id": session_id, "message": message}
    if user:
        body["user"] = user
    return client.post("/api/chat", json=body)


# ------------------------------------------------------------- 1. Approve and Reject cannot race

def test_reject_during_a_slow_approval_cannot_hide_a_filed_ticket(client):
    action = propose(client, "s-1", make_draft())
    arm_fault("create_ticket", "slow")              # create_ticket takes 3 s

    async def approve_then_reject():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", headers=token_headers("s-1")) as a:
            body = {"session_id": "s-1", "action_id": action.action_id}
            approving = asyncio.create_task(a.post("/api/approve", json={**body, "approved": True}))
            await asyncio.sleep(0.5)                # Reject arrives while create_ticket is in flight
            rejecting = asyncio.create_task(a.post("/api/approve", json={**body, "approved": False}))
            return (await approving).json(), (await rejecting).json()

    approved, rejected = client.portal.call(approve_then_reject)

    ticket_id = approved["result"]["ticket_id"]
    assert len(created_tickets()) == 1
    assert "did not file" not in rejected["reply"]          # the old bug: "not filed" for a filed ticket
    assert ticket_id in rejected["reply"]
    assert rejected["result"] == {"ticket_id": ticket_id}


# ------------------------------------------------------------- 2. input size limits

@pytest.mark.parametrize("body", [
    {"session_id": "s-1", "message": "x" * (MAX_MESSAGE_CHARS + 1)},
    {"session_id": "s-1", "message": ""},
    {"session_id": "s" * 129, "message": "hi"},
    {"session_id": "s-1; DROP TABLE", "message": "hi"},
    {"session_id": "s-1", "message": "hi", "user": {"email": "a" * 250 + "@nakhla.example"}},
])
def test_oversized_or_malformed_chat_requests_are_rejected(client, body):
    assert client.post("/api/chat", json=body).status_code == 422
    assert len(sessions) == 0


def test_long_action_id_is_rejected(client):
    r = client.post("/api/approve", json={"session_id": "s-1", "action_id": "a" * 65, "approved": True})
    assert r.status_code == 422


def test_message_at_the_limit_and_eval_style_session_ids_are_accepted(client):
    assert chat(client, session_id="eval-S2-01-ab12cd", message="x" * MAX_MESSAGE_CHARS).status_code == 200
    assert chat(client, session_id="ui-0123456789").status_code == 200


# ------------------------------------------------------------- 3. resetting every session needs the token

def test_reset_all_without_the_token_is_refused(client, monkeypatch):
    monkeypatch.delenv("APP_ADMIN_TOKEN", raising=False)
    chat(client, "s-1")
    assert client.post("/reset").status_code == 403
    assert client.post("/reset", headers={"X-Admin-Token": "anything"}).status_code == 403
    assert "s-1" in sessions


def test_reset_all_with_a_wrong_token_is_refused(client, monkeypatch):
    monkeypatch.setenv("APP_ADMIN_TOKEN", "the-right-token")
    chat(client, "s-1")
    assert client.post("/reset", headers={"X-Admin-Token": "a-wrong-token"}).status_code == 403
    assert "s-1" in sessions


def test_reset_all_with_the_token_works(client, monkeypatch):
    monkeypatch.setenv("APP_ADMIN_TOKEN", "the-right-token")
    chat(client, "s-1")
    chat(client, "s-2")
    r = client.post("/reset", headers={"X-Admin-Token": "the-right-token"})
    assert r.status_code == 200 and r.json()["sessions_cleared"] == 2
    assert len(sessions) == 0


def test_reset_of_one_session_needs_no_token_and_leaves_the_others(client, monkeypatch):
    monkeypatch.delenv("APP_ADMIN_TOKEN", raising=False)
    chat(client, "s-1")
    chat(client, "s-2")
    assert client.post("/reset", json={"session_id": "s-1"}).status_code == 200
    assert "s-1" not in sessions and "s-2" in sessions


# ------------------------------------------------------------- 4. a session cannot switch employees

def test_a_session_cannot_switch_to_another_employee(client):
    assert chat(client, user=SARA).status_code == 200
    r = chat(client, user={"email": "samir.nassar@nakhla.example"}).json()       # the CEO, a VIP

    assert r["state"] == "identity_mismatch"
    assert r["tool_calls"] == []                                                # nothing was looked up
    assert get_session("s-1")["user"]["email"] == SARA["email"]


def test_the_same_employee_can_keep_talking(client):
    chat(client, user=SARA)
    assert chat(client, user={"email": "  Sara.Mikhail@nakhla.example "}).json()["state"] != "identity_mismatch"


def test_a_linked_session_cannot_be_continued_without_the_email(client):
    # V5: someone who only knows the session id (it is in the logs) must not act as its owner.
    chat(client, user=SARA)
    r = chat(client).json()                                                     # no user at all
    assert r["state"] == "identity_mismatch" and r["tool_calls"] == []
    assert "send your work email" in r["reply"]
    assert get_session("s-1")["user"] == SARA


def test_an_anonymous_session_can_keep_talking_without_an_email(client):
    # Sessions that never named an employee (sample case S2-02) are not affected.
    assert chat(client).json()["state"] == "skeleton"
    assert chat(client).json()["state"] == "skeleton"


# ------------------------------------------------------------- 5. the session store is bounded

def test_the_session_store_never_grows_past_its_limit(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_SESSIONS", 5)
    for i in range(12):
        get_session(f"s-{i}")
    assert len(sessions) == 5
    assert list(sessions) == [f"s-{i}" for i in range(7, 12)]          # the most recently used survive


# ------------------------------------------------------------- 6. the MCP server gets no extra secrets

def test_the_mcp_server_environment_holds_no_other_secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-secret")
    monkeypatch.setenv("APP_ADMIN_TOKEN", "admin-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("HELPDESK_API_URL", "http://127.0.0.1:8102")

    env = server_env()
    assert env["HELPDESK_API_URL"] == "http://127.0.0.1:8102"
    assert not {"OPENROUTER_API_KEY", "APP_ADMIN_TOKEN", "AWS_SECRET_ACCESS_KEY"} & set(env)
    assert "sk-or-test-secret" not in env.values()
