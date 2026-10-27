"""Our helpdesk agent API (walking skeleton). Run from the kit folder:

    uvicorn app.main:app --port 8000 --reload

Endpoints follow common/CONTRACT.md. The agent is not connected yet: /api/chat
checks the real status page through MCP and returns it in the ChatResponse shape
(walking skeleton: UI -> our API -> MCP -> helpdesk API -> reply).

The helpdesk API must be running on port 8102 (uvicorn services.helpdesk_api:app --port 8102).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.mcp_client import HelpdeskMCP
from app.schemas import ApproveRequest, ChatRequest, ChatResponse, ResetRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("helpdesk-app")

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

# Session state, keyed by session_id: {"user": {...} | None, "pending_action": PendingAction | None}.
# Temporary: the LangGraph SQLite checkpointer replaces this dict later.
sessions: dict[str, dict] = {}


def get_session(session_id: str) -> dict:
    return sessions.setdefault(session_id, {"user": None, "pending_action": None})


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
    if req.user:
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
def approve(req: ApproveRequest) -> ChatResponse:
    session = get_session(req.session_id)
    pending = session["pending_action"]

    # Unknown, already used, or belongs to another draft: write nothing.
    if pending is None or pending.action_id != req.action_id:
        log.info("approve ignored session=%s action=%s (no matching pending action)",
                 req.session_id, req.action_id)
        return ChatResponse(
            session_id=req.session_id,
            reply="There is no ticket draft waiting for approval, so nothing was filed.",
        )

    # The action is used up either way, so a replayed approval cannot write twice.
    session["pending_action"] = None

    if not req.approved:
        log.info("approve rejected session=%s action=%s", req.session_id, req.action_id)
        return ChatResponse(
            session_id=req.session_id,
            reply="Okay, I did not file the ticket. Tell me if you want to change anything.",
        )

    # Step 6 (approval gate) calls create_ticket over MCP here, with idempotency_key = action_id.
    log.info("approve accepted session=%s action=%s (ticket creation not connected yet)",
             req.session_id, req.action_id)
    return ChatResponse(
        session_id=req.session_id,
        reply="Approval received, but ticket creation is not connected yet. No ticket was filed.",
    )


@app.post("/reset")
def reset(req: ResetRequest | None = None) -> dict:
    """Clear our app's session state. Never touches the knowledge base or the helpdesk API."""
    if req and req.session_id:
        sessions.pop(req.session_id, None)
        return {"status": "ok", "reset": req.session_id}
    count = len(sessions)
    sessions.clear()
    return {"status": "ok", "reset": "all", "sessions_cleared": count}
