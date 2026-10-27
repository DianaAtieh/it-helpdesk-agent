"""Concurrency tests (V9, and the gaps listed in V14). Run from the kit folder:

    python -m pytest tests/test_concurrency.py -v

Requests run truly at the same time on the app's event loop (httpx.ASGITransport inside client.portal).
A slow turn is made with the helpdesk API's "slow" fault (the call waits 3 s, then succeeds).
  * requests of one session run one after another, and nothing they save is lost;
  * a request waits at most SESSION_LOCK_WAIT_SECONDS, and at most SESSION_MAX_WAITING may wait (V9);
  * approve, chat and reset of one session cannot interleave.

Setup and fixtures: tests/conftest.py (needs the helpdesk API on port 8102).
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from langchain_core.messages import HumanMessage

import app.main as main
from app.main import app, get_session, sessions, thread_config
from tests.helpers import (TURN_RAN, approve, arm_fault, created_tickets, ensure_token, make_draft, propose,
                           requires_api, token_headers)

pytestmark = [requires_api, pytest.mark.usefixtures("clean")]

SARA = {"email": "sara.mikhail@nakhla.example"}


def chat_body(session_id: str, message: str, user: dict | None = SARA) -> dict:
    return {"session_id": session_id, "message": message, **({"user": user} if user else {})}


def together(client, *requests: tuple[float, str, dict]) -> list[tuple[int, dict]]:
    """Send (delay in seconds, path, body) requests concurrently, each with its session's token if the test
    issued one (ensure_token / propose). Returns (status, json) in the same order."""
    async def send(api: httpx.AsyncClient, delay: float, path: str, body: dict):
        await asyncio.sleep(delay)
        r = await api.post(path, json=body, headers=token_headers(body["session_id"]))
        return r.status_code, r.json()

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                                     timeout=60) as api:
            return await asyncio.gather(*(send(api, *request) for request in requests))
    return client.portal.call(run)


def human_messages(client, session_id: str) -> list[str]:
    values = client.portal.call(app.state.graph.aget_state, thread_config(session_id)).values
    return [m.content for m in values.get("messages", []) if isinstance(m, HumanMessage)]


# ------------------------------------------------------------- one session: one request at a time

def test_two_chats_on_one_session_at_once_both_run_and_both_are_saved(client):
    arm_fault("status", "slow")
    (s1, first), (s2, second) = together(client, (0, "/api/chat", chat_body("s-1", "first")),
                                         (0.3, "/api/chat", chat_body("s-1", "second")))
    assert (s1, s2) == (200, 200)
    assert {first["state"], second["state"]} <= TURN_RAN
    assert human_messages(client, "s-1") == ["first", "second"]             # in order, no lost update


def test_two_sessions_at_once_do_not_block_each_other(client):
    arm_fault("status", "slow")
    (_, a), (_, b) = together(client, (0, "/api/chat", chat_body("s-a", "from A")),
                              (0.3, "/api/chat", chat_body("s-b", "from B")))
    assert {a["state"], b["state"]} <= TURN_RAN                           # neither was refused as busy
    assert human_messages(client, "s-a") == ["from A"] and human_messages(client, "s-b") == ["from B"]


# ------------------------------------------------------------- V9: no unbounded waiting

def test_a_request_that_waits_too_long_is_told_to_retry_and_changes_nothing(client, monkeypatch):
    monkeypatch.setenv("SESSION_LOCK_WAIT_SECONDS", "0.5")
    arm_fault("status", "slow")
    (_, first), (_, second) = together(client, (0, "/api/chat", chat_body("s-1", "first")),
                                       (0.3, "/api/chat", chat_body("s-1", "second")))
    assert first["state"] in TURN_RAN
    assert second["state"] == "busy" and second["tool_calls"] == [] and "again" in second["reply"]
    assert human_messages(client, "s-1") == ["first"]                       # the refused message was not run


def test_only_a_limited_number_of_requests_may_wait_for_one_session(client, monkeypatch):
    monkeypatch.setenv("SESSION_MAX_WAITING", "1")
    arm_fault("status", "slow")
    results = together(client, (0, "/api/chat", chat_body("s-1", "first")),
                       (0.3, "/api/chat", chat_body("s-1", "second")),      # waits
                       (0.6, "/api/chat", chat_body("s-1", "third")))       # one too many: refused at once
    states = [out["state"] for _, out in results]
    assert set(states[:2]) <= TURN_RAN and states[2] == "busy"
    assert human_messages(client, "s-1") == ["first", "second"]
    assert sessions["s-1"]["waiting"] == 0 and not sessions["s-1"]["lock"].locked()


def test_a_second_approval_that_waits_too_long_files_nothing_and_a_retry_tells_the_truth(client, monkeypatch):
    monkeypatch.setenv("SESSION_LOCK_WAIT_SECONDS", "0.5")
    action = propose(client, "s-1", make_draft())
    arm_fault("create_ticket", "slow")
    body = {"session_id": "s-1", "action_id": action.action_id, "approved": True}
    (_, first), (_, second) = together(client, (0, "/api/approve", body), (0.3, "/api/approve", body))

    ticket_id = first["result"]["ticket_id"]
    assert second["state"] == "busy" and second["result"] is None and "IT-" not in second["reply"]
    assert len(created_tickets()) == 1
    assert approve(client, "s-1", action.action_id)["result"] == {"ticket_id": ticket_id}   # the retry


def test_a_session_with_a_waiting_request_is_never_evicted(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_SESSIONS", 1)
    waiting = get_session("s-waiting")
    waiting["waiting"] = 1                                                  # as if a request were queued
    get_session("s-new")
    assert "s-waiting" in sessions
    waiting["waiting"] = 0


# ------------------------------------------------------------- approve, chat and reset cannot interleave

def test_a_double_approval_at_the_same_moment_files_one_ticket(client):
    action = propose(client, "s-1", make_draft())
    arm_fault("create_ticket", "slow")
    body = {"session_id": "s-1", "action_id": action.action_id, "approved": True}
    (_, first), (_, second) = together(client, (0, "/api/approve", body), (0.1, "/api/approve", body))

    ticket_id = first["result"]["ticket_id"]
    assert second["result"] == {"ticket_id": ticket_id} and "already filed" in second["reply"]
    assert len(created_tickets()) == 1


def test_a_chat_during_an_approval_waits_and_both_are_saved(client):
    human = "my VPN still fails"
    action = propose(client, "s-1", make_draft())
    arm_fault("create_ticket", "slow")
    (_, approved), (_, chat_out) = together(
        client, (0, "/api/approve", {"session_id": "s-1", "action_id": action.action_id, "approved": True}),
        (0.3, "/api/chat", chat_body("s-1", human)))

    ticket_id = approved["result"]["ticket_id"]
    assert chat_out["state"] in TURN_RAN
    assert human_messages(client, "s-1") == [human]
    values = client.portal.call(app.state.graph.aget_state, thread_config("s-1")).values
    assert values["filed"] == {action.action_id: ticket_id} and values["pending_action"] is None
    assert len(created_tickets()) == 1


def test_a_reset_during_a_chat_waits_for_it_and_the_conversation_does_not_come_back(client):
    ensure_token(client, "s-1")                     # the owner holds the token before the race starts
    arm_fault("status", "slow")
    (_, out), (status, reset) = together(client, (0, "/api/chat", chat_body("s-1", "first")),
                                         (0.3, "/reset", {"session_id": "s-1", "user": SARA}))
    assert out["state"] in TURN_RAN
    assert status == 200 and reset["reset"] == "s-1"
    assert human_messages(client, "s-1") == []                              # deleted after the turn saved
    assert "s-1" not in sessions


def test_a_reset_that_waits_too_long_is_refused_and_deletes_nothing(client, monkeypatch):
    monkeypatch.setenv("SESSION_LOCK_WAIT_SECONDS", "0.5")
    ensure_token(client, "s-1")
    arm_fault("status", "slow")
    (_, out), (status, _) = together(client, (0, "/api/chat", chat_body("s-1", "first")),
                                     (0.3, "/reset", {"session_id": "s-1", "user": SARA}))
    assert out["state"] in TURN_RAN and status == 409
    assert human_messages(client, "s-1") == ["first"]
