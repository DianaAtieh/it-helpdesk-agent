"""Tests for the approval gate (app/tickets.py + POST /api/approve). Run from the kit folder:

    python -m pytest tests -v

They need the helpdesk API running (uvicorn services.helpdesk_api:app --port 8102) and are
skipped otherwise. Each test calls the API's test-only /admin/reset, so tickets you created
by hand on that API are cleared.
"""
from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app, get_session, mcp, sessions
from app.tickets import TicketDraft, TicketDraftError, propose_ticket

API = (os.getenv("HELPDESK_API_URL") or "http://localhost:8102").rstrip("/")


def _api_up() -> bool:
    try:
        return httpx.get(f"{API}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not _api_up(), reason=f"helpdesk API not running at {API}")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # starts the app's lifespan, so the MCP server runs
        yield c


@pytest.fixture(autouse=True)
def clean():
    httpx.post(f"{API}/admin/reset", timeout=5).raise_for_status()
    sessions.clear()
    yield
    httpx.delete(f"{API}/admin/faults", timeout=5)


def make_draft(**overrides) -> TicketDraft:
    fields = dict(
        subject="NX-112 on the VPN since this morning",
        description="NakhlaConnect shows NX-112 when I connect from home.",
        category="network_access",
        priority="P2",
        requester_email="sara.mikhail@nakhla.example",
        attempted_steps=["NakhlaManage Sync", "Restarted the laptop"],
        kb_refs=["KB-002"],
    )
    fields.update(overrides)
    return TicketDraft(**fields)


def propose(client: TestClient, session_id: str, draft: TicketDraft):
    # propose_ticket uses the app's MCP session, so it must run on the app's event loop.
    action, _ = client.portal.call(propose_ticket, get_session(session_id), draft, mcp)
    return action


def approve(client: TestClient, session_id: str, action_id: str, approved: bool = True) -> dict:
    r = client.post("/api/approve", json={"session_id": session_id, "action_id": action_id, "approved": approved})
    assert r.status_code == 200
    return r.json()


def created_tickets() -> list[dict]:
    return httpx.get(f"{API}/admin/tickets", timeout=5).json()["tickets"]


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
    httpx.post(f"{API}/admin/faults", json={"target": "create_ticket", "mode": "error_503", "count": 1},
               timeout=5).raise_for_status()
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
