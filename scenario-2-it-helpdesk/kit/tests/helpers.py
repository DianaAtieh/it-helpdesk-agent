"""Shared helpers for the app tests. The fixtures are in conftest.py."""
from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import get_session, mcp
from app.tickets import TicketDraft, propose_ticket

API = (os.getenv("HELPDESK_API_URL") or "http://localhost:8102").rstrip("/")


def _api_up() -> bool:
    try:
        return httpx.get(f"{API}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


requires_api = pytest.mark.skipif(not _api_up(), reason=f"helpdesk API not running at {API}")


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


def arm_fault(target: str, mode: str, count: int = 1) -> None:
    httpx.post(f"{API}/admin/faults", json={"target": target, "mode": mode, "count": count},
               timeout=5).raise_for_status()
