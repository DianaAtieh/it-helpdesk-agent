"""Shared helpers for the app tests. The fixtures are in conftest.py."""
from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import SESSION_TOKEN_HEADER, get_session, issue_session_token, propose_draft, session_turn
from app.tickets import TicketDraft

API = (os.getenv("HELPDESK_API_URL") or "http://localhost:8102").rstrip("/")

# The states of a chat turn that ran normally (not refused, busy or failed). With the tests' fake knowledge
# base an ordinary turn is "clarify", an NX-503 turn "incident" (the real status page has INC-2207 active).
TURN_RAN = frozenset({"answered", "clarify", "not_documented", "incident"})


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


# Session tokens issued by ensure_token(), by session id. The clean fixture empties it. A session started by a
# chat turn through the TestClient has its token in the client's cookie jar instead (sent automatically).
TOKENS: dict[str, str] = {}


async def _issue_token(session_id: str) -> str | None:
    async with session_turn(get_session(session_id)):
        return await issue_session_token(session_id)


def ensure_token(client: TestClient, session_id: str) -> str | None:
    """Start a NEW conversation's token without a chat turn (as /api/chat would) and remember it."""
    token = client.portal.call(_issue_token, session_id)
    if token:
        TOKENS[session_id] = token
    return token


def token_headers(session_id: str) -> dict:
    """The X-Session-Token header for a session whose token ensure_token() issued, else {} (cookie or none)."""
    return {SESSION_TOKEN_HEADER: TOKENS[session_id]} if session_id in TOKENS else {}


def propose(client: TestClient, session_id: str, draft: TicketDraft):
    # The same path the agent will use (load, check, save). It uses the app's MCP session and
    # checkpointer, so it must run on the app's event loop. A conversation started here (no chat turn
    # before) gets its session token first, like the first chat turn would give it.
    ensure_token(client, session_id)
    action, _ = client.portal.call(propose_draft, session_id, draft)
    return action


def approve(client: TestClient, session_id: str, action_id: str, approved: bool = True,
            expect: int = 200) -> dict:
    r = client.post("/api/approve", json={"session_id": session_id, "action_id": action_id, "approved": approved},
                    headers=token_headers(session_id))
    assert r.status_code == expect
    return r.json()


def created_tickets() -> list[dict]:
    return httpx.get(f"{API}/admin/tickets", timeout=5).json()["tickets"]


def arm_fault(target: str, mode: str, count: int = 1) -> None:
    httpx.post(f"{API}/admin/faults", json={"target": target, "mode": mode, "count": count},
               timeout=5).raise_for_status()
