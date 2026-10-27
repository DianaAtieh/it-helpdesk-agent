"""V6: an email arriving on an anonymous conversation never unlocks what was said before it. Run from the kit folder:

    python -m pytest tests/test_session_claim.py -v

An email is a claim, not proof: anyone who knows an anonymous session id could send one. So when the first
email arrives, the conversation's earlier messages, replies and ticket draft are deleted (also from older
checkpoints) before the email is linked. The filed record is kept, so replays are still answered truthfully.
sessions.clear() simulates a restart or an eviction.

Setup and fixtures: tests/conftest.py (needs the helpdesk API on port 8102).
"""
from __future__ import annotations

import logging

import pytest
from langchain_core.messages import HumanMessage

from app import main
from app.main import CLAIM_NOTE, app, sessions, thread_config
from tests.helpers import TURN_RAN, approve, created_tickets, make_draft, propose, requires_api

pytestmark = [requires_api, pytest.mark.usefixtures("clean")]

SARA = {"email": "sara.mikhail@nakhla.example"}
SAMIR = {"email": "samir.nassar@nakhla.example"}          # the CEO (VIP)
SECRET_TEXT = "my home router is ZX-PRIVATE-7781 and my manager is on leave"


def chat(client, session_id: str, message: str, user: dict | None = None) -> dict:
    body = {"session_id": session_id, "message": message}
    if user:
        body["user"] = user
    r = client.post("/api/chat", json=body)
    assert r.status_code == 200
    return r.json()


def saved(client, session_id: str) -> dict:
    return client.portal.call(app.state.graph.aget_state, thread_config(session_id)).values


def human_messages(client, session_id: str) -> list[str]:
    return [m.content for m in saved(client, session_id).get("messages", []) if isinstance(m, HumanMessage)]


def raw_database(client) -> bytes:
    async def run():
        rows = []
        for query in ("SELECT checkpoint FROM checkpoints", "SELECT value FROM writes"):
            async with app.state.checkpointer.conn.execute(query) as cursor:
                rows += await cursor.fetchall()
        return b"".join(bytes(row[0]) for row in rows if row[0] is not None)
    return client.portal.call(run)


# ------------------------------------------------------------- the legitimate employee

def test_1_an_email_after_anonymous_turns_links_the_session_and_drops_the_earlier_history(client):
    chat(client, "s-1", "My VPN is not working. Can you fix it?")
    out = chat(client, "s-1", "It shows NX-112", SARA)

    assert out["state"] in TURN_RAN and out["reply"].startswith(CLAIM_NOTE)
    assert human_messages(client, "s-1") == ["It shows NX-112"]             # only the turn after the email
    assert saved(client, "s-1")["user_email"] == SARA["email"]


def test_2_an_email_on_the_first_turn_clears_nothing_and_adds_no_note(client):
    out = chat(client, "s-1", "Is the VPN down?", SARA)
    assert CLAIM_NOTE not in out["reply"]
    assert human_messages(client, "s-1") == ["Is the VPN down?"]


def test_3_a_session_that_stays_anonymous_keeps_its_history(client):
    chat(client, "s-1", "first")
    out = chat(client, "s-1", "second")
    assert CLAIM_NOTE not in out["reply"]
    assert human_messages(client, "s-1") == ["first", "second"]


def test_4_after_the_claim_the_same_employee_continues_normally(client):
    chat(client, "s-1", "anonymous")
    chat(client, "s-1", "now with my email", SARA)
    out = chat(client, "s-1", "and again", SARA)

    assert out["state"] in TURN_RAN and CLAIM_NOTE not in out["reply"]
    assert human_messages(client, "s-1") == ["now with my email", "and again"]


# ------------------------------------------------------------- the attacker

def test_5_a_claimer_gets_none_of_the_anonymous_text_not_even_in_the_raw_database(client):
    chat(client, "s-1", SECRET_TEXT)
    chat(client, "s-1", "what did I tell you before?", SAMIR)

    state = saved(client, "s-1")
    assert all("ZX-PRIVATE-7781" not in str(m.content) for m in state["messages"])
    assert "ZX-PRIVATE-7781" not in str(state.get("reply"))
    assert b"ZX-PRIVATE-7781" not in raw_database(client)                   # older checkpoints were pruned


def test_6_after_an_attacker_claims_first_the_real_employee_is_refused_and_sees_nothing(client):
    chat(client, "s-1", SECRET_TEXT)                                         # the real employee, anonymous
    chat(client, "s-1", "hello", SAMIR)                                      # the attacker claims

    out = chat(client, "s-1", "it is me, Sara", SARA)
    assert out["state"] == "identity_mismatch" and out["tool_calls"] == []
    assert "ZX-PRIVATE-7781" not in out["reply"]
    assert saved(client, "s-1")["user_email"] == SAMIR["email"]


def test_7_an_anonymous_draft_is_dropped_by_the_claim_and_cannot_be_approved(client):
    chat(client, "s-1", "my VPN fails with NX-112")
    action = propose(client, "s-1", make_draft())
    chat(client, "s-1", "hello", SAMIR)

    assert saved(client, "s-1")["pending_action"] is None
    out = approve(client, "s-1", action.action_id)
    assert out["result"] is None and "nothing was filed" in out["reply"]
    assert created_tickets() == []


def test_8_a_ticket_filed_before_the_claim_is_still_reported_truthfully_and_never_twice(client):
    chat(client, "s-1", "my VPN fails with NX-112")
    action = propose(client, "s-1", make_draft())
    ticket_id = approve(client, "s-1", action.action_id)["result"]["ticket_id"]
    chat(client, "s-1", "another anonymous message")
    chat(client, "s-1", "hello", SARA)                                       # the claim clears the history

    assert saved(client, "s-1")["filed"] == {action.action_id: ticket_id}
    out = approve(client, "s-1", action.action_id)
    assert out["result"] == {"ticket_id": ticket_id} and f"already filed as {ticket_id}" in out["reply"]
    assert len(created_tickets()) == 1


# ------------------------------------------------------------- restarts and failures

def test_9_the_claim_still_clears_the_history_after_a_restart(client):
    chat(client, "s-1", SECRET_TEXT)
    sessions.clear()

    out = chat(client, "s-1", "hello", SAMIR)
    assert out["reply"].startswith(CLAIM_NOTE)
    assert human_messages(client, "s-1") == ["hello"]
    assert b"ZX-PRIVATE-7781" not in raw_database(client)


def test_10_the_link_made_by_a_claim_survives_a_restart(client):
    chat(client, "s-1", "anonymous")
    chat(client, "s-1", "hello", SAMIR)
    sessions.clear()

    assert chat(client, "s-1", "it is me, Sara", SARA)["state"] == "identity_mismatch"
    out = chat(client, "s-1", "still me", SAMIR)
    assert out["state"] in TURN_RAN and CLAIM_NOTE not in out["reply"]      # nothing left to clear
    assert human_messages(client, "s-1") == ["hello", "still me"]


@pytest.mark.parametrize("failing_step", ["clear", "prune"])
def test_11_if_clearing_fails_the_email_is_not_linked_and_nothing_runs(client, monkeypatch, failing_step):
    chat(client, "s-1", SECRET_TEXT)

    async def broken(*args, **kwargs):
        raise RuntimeError("database is locked")

    if failing_step == "clear":
        monkeypatch.setattr(app.state.graph, "aupdate_state", broken)
    else:
        monkeypatch.setattr(main, "prune_thread", broken)
    out = chat(client, "s-1", "hello", SAMIR)

    assert out["state"] == "error" and out["tool_calls"] == []
    assert "ZX-PRIVATE-7781" not in out["reply"]
    assert sessions["s-1"]["user"] is None                                   # not linked
    assert saved(client, "s-1").get("user_email") is None
    assert "hello" not in human_messages(client, "s-1")                      # the graph did not run

    monkeypatch.undo()                                                       # the database works again
    retried = chat(client, "s-1", "hello again", SAMIR)
    assert retried["state"] in TURN_RAN
    assert human_messages(client, "s-1") == ["hello again"]
    assert b"ZX-PRIVATE-7781" not in raw_database(client)


# ------------------------------------------------------------- logs

def test_the_logs_never_show_a_raw_session_id(client, caplog):
    sid = "secret-session-0123456789abcdef"
    with caplog.at_level(logging.INFO, logger="helpdesk-app"):
        chat(client, sid, "anonymous")
        chat(client, sid, "hello", SARA)
        approve(client, sid, "act-unknown")
        client.post("/reset", json={"session_id": sid})

    assert caplog.records                                                    # the app did log these requests
    assert all(sid not in record.getMessage() for record in caplog.records)
    assert any(main.session_ref(sid) in record.getMessage() for record in caplog.records)
