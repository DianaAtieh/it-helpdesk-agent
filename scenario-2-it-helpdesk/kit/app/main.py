"""Our helpdesk agent API (walking skeleton). Run from the kit folder:

    uvicorn app.main:app --port 8000 --reload

Endpoints follow common/CONTRACT.md. The agent is not connected yet: /api/chat
checks the real status page through MCP and returns it in the ChatResponse shape
(walking skeleton: UI -> our API -> MCP -> helpdesk API -> reply).

The helpdesk API must be running on port 8102 (uvicorn services.helpdesk_api:app --port 8102).
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException

# The kit's .env, without overriding variables that are already set (e.g. by Docker).
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from app.mcp_client import HelpdeskMCP  # noqa: E402  (after load_dotenv on purpose)
from app.schemas import ApproveRequest, ChatRequest, ChatResponse, ResetRequest  # noqa: E402
from app.tickets import ROUTING, TICKET_ID  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("helpdesk-app")

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS") or 1000)

mcp = HelpdeskMCP()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Start the MCP server once with the app, and stop it when the app stops.
    await mcp.start()
    try:
        yield
    finally:
        await mcp.stop()


app = FastAPI(title="Nakhla IT Helpdesk Agent", version="0.1.0", lifespan=lifespan)

# Session state, keyed by session_id:
#   user            {"email": ...} or None
#   pending_action  the one PendingAction waiting for approval, or None
#   filed           {action_id: ticket_id} for approvals that already created a ticket (replay protection)
#   lock            one request per session at a time, so Approve and Reject cannot race
# Temporary: the LangGraph SQLite checkpointer replaces this dict later.
# Least recently used first; at most MAX_SESSIONS are kept, so new session ids cannot fill the memory.
sessions: OrderedDict[str, dict] = OrderedDict()


def get_session(session_id: str) -> dict:
    session = sessions.get(session_id)
    if session is None:
        session = {"user": None, "pending_action": None, "filed": {}, "lock": asyncio.Lock()}
        sessions[session_id] = session
        _evict_idle_sessions()
    sessions.move_to_end(session_id)
    return session


def _evict_idle_sessions() -> None:
    """Drop the least recently used sessions above MAX_SESSIONS, never one with a request in progress."""
    for session_id in list(sessions):
        if len(sessions) <= MAX_SESSIONS:
            break
        if not sessions[session_id]["lock"].locked():
            del sessions[session_id]


def ticket_failure_reply(error: dict) -> tuple[str, bool]:
    """An honest reply for a failed create_ticket, and whether the same draft may be retried."""
    code = error.get("code")
    if code == "timeout":
        # A timeout confirms nothing either way; retrying is safe thanks to the idempotency key.
        return ("I couldn't confirm that the ticket was created: the ticket system did not answer in time. "
                "You can approve again. It is safe and will not create a duplicate."), True
    if code == "validation_error":
        return ("The ticket was not created: the ticket system rejected the draft. Nothing was filed. "
                "Tell me more about the problem and I will prepare a corrected draft."), False
    return ("The ticket was not created: the ticket system is unavailable right now. Nothing was filed. "
            "You can approve again in a moment to retry."), True


def describe_status(data: dict) -> str:
    """Turn a get_service_status result into a short reply."""
    if not data.get("ok"):
        code = (data.get("error") or {}).get("code", "unknown")
        return f"I could not check the status page right now ({code}). Please try again in a moment."
    status = data["status"]
    active = [i for i in status.get("incidents", []) if i.get("status") == "active"]
    if not active:
        return "The status page shows no active incidents right now."
    lines = ["The status page shows these active incidents:"]
    for inc in active:
        line = f"- {inc['id']} ({inc.get('priority', '?')}): {inc.get('title', '')}"
        if inc.get("workaround"):
            line += f". Workaround: {inc['workaround']}"
        lines.append(line)
    return "\n".join(lines)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session = get_session(req.session_id)
    async with session["lock"]:
        return await _chat(req, session)


async def _chat(req: ChatRequest, session: dict) -> ChatResponse:
    # Without a login the email is a claim, but a conversation cannot switch to another employee
    # (e.g. to a VIP for a P1 ticket) once it started.
    current = (session["user"] or {}).get("email")
    if req.user and current and req.user.email != current:
        log.warning("chat session=%s refused: user changed from %s to %s", req.session_id, current, req.user.email)
        return ChatResponse(
            session_id=req.session_id,
            reply="This conversation is already linked to another employee. Please start a new conversation.",
            state="identity_mismatch",
        )
    if req.user and not current:
        session["user"] = req.user.model_dump()
    log.info("chat session=%s user=%s", req.session_id, (session["user"] or {}).get("email"))

    # Walking skeleton: one real MCP read. The LangGraph agent replaces this later.
    status, call = await mcp.call("get_service_status")
    return ChatResponse(
        session_id=req.session_id,
        reply=describe_status(status),
        state="skeleton",
        tool_calls=[call],
    )


@app.post("/api/approve", response_model=ChatResponse)
async def approve(req: ApproveRequest) -> ChatResponse:
    """The approval gate: the only place in the app that writes a ticket.

    The session lock makes Approve, Reject and chat turns of one session run one after another:
    without it, a Reject arriving while create_ticket is in flight said "not filed" for a filed ticket.
    """
    session = get_session(req.session_id)
    async with session["lock"]:
        return await _approve(req, session)


async def _approve(req: ApproveRequest, session: dict) -> ChatResponse:
    # Replay of an approval that already created a ticket (double click): never write twice.
    already = session["filed"].get(req.action_id)
    if already:
        log.info("approve replay session=%s action=%s ticket=%s", req.session_id, req.action_id, already)
        reply = (f"This ticket was already filed as {already}. I did not file it again." if req.approved else
                 f"This ticket was already filed as {already}, so it can no longer be rejected.")
        return ChatResponse(
            session_id=req.session_id,
            reply=reply,
            state="ticket_created",
            result={"ticket_id": already},
        )

    # Unknown, stale (replaced by a newer draft) or from another session: write nothing.
    pending = session["pending_action"]
    if pending is None or pending.action_id != req.action_id:
        log.info("approve ignored session=%s action=%s (no matching pending action)",
                 req.session_id, req.action_id)
        return ChatResponse(
            session_id=req.session_id,
            reply="There is no ticket draft waiting for approval, so nothing was filed.",
        )

    if not req.approved:
        session["pending_action"] = None
        log.info("approve rejected session=%s action=%s", req.session_id, req.action_id)
        return ChatResponse(
            session_id=req.session_id,
            reply="Okay, I did not file the ticket. Tell me if you want to change anything.",
        )

    # Exactly the approved payload, with the action id as the idempotency key.
    data, call = await mcp.call("create_ticket", {**pending.payload, "idempotency_key": pending.action_id})
    ticket_id = str((data.get("ticket") or {}).get("ticket_id", "")) if data.get("ok") else ""

    if data.get("ok") and TICKET_ID.fullmatch(ticket_id):
        session["pending_action"] = None
        session["filed"][pending.action_id] = ticket_id
        team = data["ticket"].get("assigned_team") or ROUTING.get(pending.payload["category"], "IT")
        log.info("ticket created session=%s action=%s ticket=%s", req.session_id, pending.action_id, ticket_id)
        return ChatResponse(
            session_id=req.session_id,
            reply=f"Done. Your ticket is {ticket_id}. {team} will contact you.",
            state="ticket_created",
            result={"ticket_id": ticket_id, "assigned_team": team},
            tool_calls=[call],
        )

    # Failed, or an answer without a valid ticket id: claim nothing.
    error = data.get("error") or {"code": "bad_tool_output"}
    reply, can_retry = ticket_failure_reply(error)
    if not can_retry:
        session["pending_action"] = None
    log.warning("ticket not created session=%s action=%s error=%s", req.session_id, pending.action_id,
                error.get("code"))
    return ChatResponse(
        session_id=req.session_id,
        reply=reply,
        state="await_approval" if can_retry else "ticket_failed",
        pending_action=pending if can_retry else None,
        tool_calls=[call],
    )


@app.post("/reset")
async def reset(req: ResetRequest | None = None, x_admin_token: str | None = Header(default=None)) -> dict:
    """Clear our app's session state. Never touches the knowledge base or the helpdesk API.

    One session: open to anyone who knows its id. Every session: needs the X-Admin-Token header
    to match APP_ADMIN_TOKEN, and is disabled when APP_ADMIN_TOKEN is not set.
    """
    if req and req.session_id:
        session = sessions.get(req.session_id)
        if session is not None:
            async with session["lock"]:          # wait for a request of this session to finish
                sessions.pop(req.session_id, None)
        return {"status": "ok", "reset": req.session_id}

    expected = os.getenv("APP_ADMIN_TOKEN") or ""
    if not expected or not secrets.compare_digest(x_admin_token or "", expected):
        log.warning("reset of all sessions refused (missing or wrong admin token)")
        raise HTTPException(status_code=403, detail="Resetting every session needs a valid X-Admin-Token.")
    count = len(sessions)
    sessions.clear()
    log.info("reset all sessions: %d cleared", count)
    return {"status": "ok", "reset": "all", "sessions_cleared": count}
