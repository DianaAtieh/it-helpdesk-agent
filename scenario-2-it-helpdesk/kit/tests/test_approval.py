"""Tests for the approval gate (app/tickets.py + POST /api/approve). Run from the kit folder:

    python -m pytest tests -v

Setup and fixtures: tests/conftest.py (needs the helpdesk API on port 8102).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.main import get_session
from app.tickets import TicketDraftError
from tests.helpers import approve, arm_fault, created_tickets, ensure_token, make_draft, propose, requires_api

pytestmark = [requires_api, pytest.mark.usefixtures("clean")]


# ------------------------------------------------------------- happy path

def test_approve_files_exactly_the_approved_draft(client):
    action = propose(client, "s-1", make_draft())
    out = approve(client, "s-1", action.action_id)

    ticket_id = out["result"]["ticket_id"]
    assert ticket_id.startswith("IT-") and ticket_id in out["reply"]
    assert out["pending_action"] is None
    assert [c["tool"] for c in out["tool_calls"]] == ["create_ticket"]

    tickets = created_tickets()
    assert len(tickets) == 1
    for field in ("subject", "category", "priority", "requester_email", "department", "attempted_steps", "kb_refs"):
        assert tickets[0][field] == action.payload[field]
    assert tickets[0]["assigned_team"] == "Network Operations"


# ------------------------------------------------------------- nothing written

def test_reject_writes_nothing(client):
    action = propose(client, "s-1", make_draft())
    out = approve(client, "s-1", action.action_id, approved=False)

    assert out["result"] is None and out["pending_action"] is None
    assert "IT-2" not in out["reply"]
    assert created_tickets() == []


def test_unknown_or_foreign_action_writes_nothing(client):
    action = propose(client, "s-1", make_draft())

    ensure_token(client, "s-2")                                             # s-2's own, valid token
    assert approve(client, "s-2", action.action_id)["result"] is None      # another session
    assert approve(client, "s-1", "act-made-up")["result"] is None          # invented id
    assert created_tickets() == []


def test_stale_draft_cannot_be_approved(client):
    old = propose(client, "s-1", make_draft())
    propose(client, "s-1", make_draft(subject="NX-112 still failing after a restart"))

    assert approve(client, "s-1", old.action_id)["result"] is None
    assert created_tickets() == []


# ------------------------------------------------------------- duplicates and failures

def test_double_approval_files_once(client):
    action = propose(client, "s-1", make_draft())
    first = approve(client, "s-1", action.action_id)
    second = approve(client, "s-1", action.action_id)

    assert second["result"]["ticket_id"] == first["result"]["ticket_id"]
    assert len(created_tickets()) == 1


def test_503_claims_nothing_keeps_the_draft_and_retry_works(client):
    arm_fault("create_ticket", "error_503")
    action = propose(client, "s-1", make_draft())

    failed = approve(client, "s-1", action.action_id)
    assert failed["result"] is None
    assert "IT-2" not in failed["reply"] and "not created" in failed["reply"]
    assert failed["pending_action"]["action_id"] == action.action_id
    assert failed["tool_calls"][0] == {**failed["tool_calls"][0], "tool": "create_ticket", "ok": False}
    assert created_tickets() == []

    retried = approve(client, "s-1", action.action_id)
    assert retried["result"]["ticket_id"].startswith("IT-")
    assert len(created_tickets()) == 1


# ------------------------------------------------------------- draft checks (before anything is shown)

def test_department_comes_from_the_directory_and_vip_is_p1(client):
    action = propose(client, "s-1", make_draft(requester_email="samir.nassar@nakhla.example",
                                               department="Sales", priority="P3"))
    assert action.payload["department"] == "Executive Office"
    assert action.payload["priority"] == "P1"


def test_more_than_three_users_is_p1(client):
    action = propose(client, "s-1", make_draft(affected_users=4, priority="P3"))
    assert action.payload["priority"] == "P1"


def test_invalid_drafts_are_never_shown(client):
    with pytest.raises(TicketDraftError):
        propose(client, "s-1", make_draft(kb_refs=["KB-999"]))
    with pytest.raises(TicketDraftError):
        propose(client, "s-1", make_draft(requester_email="nobody@nakhla.example"))
    assert get_session("s-1")["pending_action"] is None

    with pytest.raises(ValidationError):
        make_draft(category="vpn")
    with pytest.raises(ValidationError):
        make_draft(attempted_steps=[])
