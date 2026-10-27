"""Tests for POST /reset with the SQLite checkpointer: saved conversations are deleted too. Run from the kit folder:

    python -m pytest tests/test_reset.py -v

Setup and fixtures: tests/conftest.py (needs the helpdesk API on port 8102).
"""
from __future__ import annotations

import pytest

from app.main import app, sessions, thread_config
from tests.helpers import approve, make_draft, propose, requires_api

pytestmark = [requires_api, pytest.mark.usefixtures("clean")]

SARA = {"email": "sara.mikhail@nakhla.example"}
SAMIR = {"email": "samir.nassar@nakhla.example"}
TOKEN = "the-admin-token"


def chat(client, session_id: str, message: str = "Is the VPN down?", user: dict | None = None) -> dict:
    body = {"session_id": session_id, "message": message}
    if user:
        body["user"] = user
    r = client.post("/api/chat", json=body)
    assert r.status_code == 200
    return r.json()


def saved(client, session_id: str) -> dict:
    return client.portal.call(app.state.graph.aget_state, thread_config(session_id)).values


def reset_one(client, session_id: str, user: dict | None = None):
    body = {"session_id": session_id}
    if user:
        body["user"] = user                                   # a linked conversation needs its employee's email
    return client.post("/reset", json=body)


# ------------------------------------------------------------- one session

def test_reset_removes_the_saved_history_and_identity(client):
    chat(client, "s-1", "first", SARA)
    assert len(saved(client, "s-1")["messages"]) == 2

    r = reset_one(client, "s-1", SARA)
    assert r.status_code == 200 and r.json() == {"status": "ok", "reset": "s-1"}
    assert saved(client, "s-1") == {}                         # no messages, no user_email
    assert "s-1" not in sessions


def test_after_reset_the_same_session_id_can_be_used_by_another_employee(client):
    chat(client, "s-1", "first", SARA)
    assert chat(client, "s-1", "I am the CEO", SAMIR)["state"] == "identity_mismatch"

    reset_one(client, "s-1", SARA)
    out = chat(client, "s-1", "I am the CEO", SAMIR)
    assert out["state"] == "skeleton"
    state = saved(client, "s-1")
    assert state["user_email"] == SAMIR["email"]
    assert [m.content for m in state["messages"]][::2] == ["I am the CEO"]   # nothing left from Sara


def test_resetting_one_session_leaves_the_others_untouched(client):
    chat(client, "s-1", "from one", SARA)
    chat(client, "s-2", "from two", SAMIR)

    reset_one(client, "s-1", SARA)
    other = saved(client, "s-2")
    assert [m.content for m in other["messages"]][::2] == ["from two"]
    assert other["user_email"] == SAMIR["email"]
    assert "s-2" in sessions and sessions["s-2"]["user"] == SAMIR


def test_reset_deletes_a_conversation_that_is_only_on_disk(client):
    chat(client, "s-1", "first", SARA)
    sessions.clear()                                          # evicted, or the app restarted

    reset_one(client, "s-1", SARA)
    assert saved(client, "s-1") == {}
    assert chat(client, "s-1", "hello", SAMIR)["state"] == "skeleton"        # the old identity is gone


def test_reset_of_an_unknown_session_is_harmless(client):
    r = reset_one(client, "never-used")
    assert r.status_code == 200 and r.json()["reset"] == "never-used"


def test_reset_drops_a_pending_ticket_draft(client):
    action = propose(client, "s-1", make_draft())
    reset_one(client, "s-1")
    out = approve(client, "s-1", action.action_id)
    assert out["result"] is None and "nothing was filed" in out["reply"]


def test_a_failed_deletion_is_reported_and_keeps_the_session(client, monkeypatch):
    chat(client, "s-1", "first", SARA)

    async def broken(thread_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(app.state.checkpointer, "adelete_thread", broken)
    r = reset_one(client, "s-1", SARA)
    assert r.status_code == 500
    assert "s-1" in sessions and sessions["s-1"]["user"] == SARA            # nothing half-reset
    monkeypatch.undo()
    assert len(saved(client, "s-1")["messages"]) == 2


# ------------------------------------------------------------- every session (admin)

def test_reset_all_without_a_valid_token_deletes_nothing(client, monkeypatch):
    monkeypatch.setenv("APP_ADMIN_TOKEN", TOKEN)
    chat(client, "s-1", "first", SARA)

    assert client.post("/reset").status_code == 403
    assert client.post("/reset", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert len(saved(client, "s-1")["messages"]) == 2 and "s-1" in sessions


def test_reset_all_is_disabled_when_no_token_is_configured(client, monkeypatch):
    monkeypatch.delenv("APP_ADMIN_TOKEN", raising=False)
    chat(client, "s-1", "first", SARA)
    assert client.post("/reset", headers={"X-Admin-Token": ""}).status_code == 403
    assert len(saved(client, "s-1")["messages"]) == 2


def test_reset_all_deletes_every_saved_conversation_even_ones_not_in_memory(client, monkeypatch):
    monkeypatch.setenv("APP_ADMIN_TOKEN", TOKEN)
    chat(client, "s-1", "one", SARA)
    chat(client, "s-2", "two", SAMIR)
    sessions.pop("s-1")                                       # s-1 is now only on disk

    r = client.post("/reset", headers={"X-Admin-Token": TOKEN})
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "reset": "all", "sessions_cleared": 1, "threads_deleted": 2}
    assert saved(client, "s-1") == {} and saved(client, "s-2") == {}
    assert len(sessions) == 0
    assert chat(client, "s-1", "fresh", SAMIR)["state"] == "skeleton"        # the ids are free again
