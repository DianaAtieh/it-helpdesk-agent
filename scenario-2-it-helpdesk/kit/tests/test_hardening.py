"""Tests for the high-severity fixes from the Phase 1 review. Run from the kit folder:

    python -m pytest tests/test_hardening.py -v

  V8  the ticket draft and the filed record survive a restart / eviction (sessions.clear() simulates it)
  V3  stored conversations stay bounded: newest checkpoint only, last MAX_HISTORY_MESSAGES messages
  V1  typed secrets never reach the database; idle conversations expire
  V7  a database error at the start of a turn gives an honest reply, not a 500
(V5, a turn without the email, is in tests/test_security.py.)

Setup and fixtures: tests/conftest.py (needs the helpdesk API on port 8102).
"""
from __future__ import annotations

import pytest

from app.agent.storage import delete_expired_threads
from app.main import app, sessions, thread_config
from tests.helpers import approve, arm_fault, created_tickets, make_draft, propose, requires_api

pytestmark = [requires_api, pytest.mark.usefixtures("clean")]

SARA = {"email": "sara.mikhail@nakhla.example"}


def chat(client, session_id: str, message: str, user: dict | None = SARA) -> dict:
    body = {"session_id": session_id, "message": message}
    if user:
        body["user"] = user
    r = client.post("/api/chat", json=body)
    assert r.status_code == 200
    return r.json()


def saved(client, session_id: str) -> dict:
    return client.portal.call(app.state.graph.aget_state, thread_config(session_id)).values


def sql(client, query: str, *params):
    async def run():
        async with app.state.checkpointer.conn.execute(query, params) as cursor:
            return await cursor.fetchall()
    return client.portal.call(run)


# ------------------------------------------------------------- V8: the gate's state survives a restart

def test_a_draft_survives_a_restart_and_can_still_be_approved(client):
    action = propose(client, "s-1", make_draft())
    assert saved(client, "s-1")["pending_action"]["action_id"] == action.action_id   # plain dict in the checkpoint
    sessions.clear()                                                                 # restart / eviction

    out = approve(client, "s-1", action.action_id)
    assert out["result"]["ticket_id"].startswith("IT-")
    assert len(created_tickets()) == 1


def test_replaying_a_filed_ticket_after_a_restart_says_it_was_filed(client):
    action = propose(client, "s-1", make_draft())
    ticket_id = approve(client, "s-1", action.action_id)["result"]["ticket_id"]
    sessions.clear()

    out = approve(client, "s-1", action.action_id)
    assert out["result"] == {"ticket_id": ticket_id}
    assert f"already filed as {ticket_id}" in out["reply"]
    assert "nothing was filed" not in out["reply"]                    # the false claim from the review
    assert len(created_tickets()) == 1


def test_a_rejection_stays_rejected_after_a_restart(client):
    action = propose(client, "s-1", make_draft())
    approve(client, "s-1", action.action_id, approved=False)
    sessions.clear()

    out = approve(client, "s-1", action.action_id)
    assert out["result"] is None and created_tickets() == []
    assert saved(client, "s-1")["pending_action"] is None


def test_a_failed_filing_keeps_the_draft_across_a_restart(client):
    arm_fault("create_ticket", "error_503")
    action = propose(client, "s-1", make_draft())
    assert approve(client, "s-1", action.action_id)["result"] is None
    sessions.clear()

    retried = approve(client, "s-1", action.action_id)
    assert retried["result"]["ticket_id"].startswith("IT-")
    assert len(created_tickets()) == 1


def test_the_identity_and_the_draft_come_back_together(client):
    chat(client, "s-1", "my VPN fails")
    action = propose(client, "s-1", make_draft())
    sessions.clear()

    assert chat(client, "s-1", "I am the CEO", {"email": "samir.nassar@nakhla.example"})["state"] == "identity_mismatch"
    assert sessions["s-1"]["pending_action"].action_id == action.action_id


# ------------------------------------------------------------- V3: bounded storage

def test_only_the_newest_checkpoint_of_a_conversation_is_kept(client):
    for i in range(5):
        chat(client, "s-1", f"message {i}")
    assert sql(client, "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", "s-1") == [(1,)]
    assert len(saved(client, "s-1")["messages"]) == 10                 # the history itself is intact


def test_the_history_is_capped_at_the_newest_messages(client, monkeypatch):
    monkeypatch.setenv("MAX_HISTORY_MESSAGES", "6")
    for i in range(5):
        chat(client, "s-1", f"message {i}")
    messages = saved(client, "s-1")["messages"]
    assert len(messages) == 6
    assert [m.content for m in messages][::2] == ["message 2", "message 3", "message 4"]
    assert chat(client, "s-1", "message 5")["state"] == "skeleton"      # the conversation goes on


def test_storage_grows_linearly_not_quadratically(client):
    def stored_bytes() -> int:
        return sql(client, "SELECT COALESCE(SUM(LENGTH(checkpoint)), 0) FROM checkpoints WHERE thread_id = ?",
                   "s-1")[0][0]

    long = "x" * 4000
    for _ in range(5):
        chat(client, "s-1", long)
    at_5 = stored_bytes()
    for _ in range(5):
        chat(client, "s-1", long)
    at_10 = stored_bytes()
    assert at_10 < 2.5 * at_5            # was about 4x (quadratic) before pruning


# ------------------------------------------------------------- V1: no secrets at rest, idle conversations expire

def test_a_typed_password_is_never_stored(client):
    chat(client, "s-1", "my laptop password is Hunter2-secret and the VPN is broken")

    stored = saved(client, "s-1")["messages"][0].content
    assert "Hunter2-secret" not in stored and "[REDACTED]" in stored
    raw = b"".join(bytes(row[0]) for row in sql(client, "SELECT checkpoint FROM checkpoints"))
    raw += b"".join(bytes(row[0]) for row in sql(client, "SELECT value FROM writes"))
    assert b"Hunter2-secret" not in raw


def test_idle_conversations_expire(client):
    chat(client, "s-1", "hello")
    assert client.portal.call(delete_expired_threads, app.state.checkpointer, 30) == 0   # fresh: kept
    assert saved(client, "s-1") != {}
    assert client.portal.call(delete_expired_threads, app.state.checkpointer, 0) == 1    # older than "now": deleted
    assert saved(client, "s-1") == {}


# ------------------------------------------------------------- V7: honest reply when loading fails

def test_a_database_error_at_the_start_of_a_turn_is_an_honest_reply(client, monkeypatch):
    class Broken:
        async def aget_state(self, config):
            raise RuntimeError("database is locked")

    monkeypatch.setattr(app.state, "graph", Broken())
    r = client.post("/api/chat", json={"session_id": "s-9", "message": "hi"})
    assert r.status_code == 200 and r.json()["state"] == "error"
    r = client.post("/api/approve", json={"session_id": "s-9", "action_id": "act-x", "approved": True})
    assert r.status_code == 200 and "nothing was filed" in r.json()["reply"]
