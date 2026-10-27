"""Our helpdesk agent API (walking skeleton). Run from the kit folder:

    uvicorn app.main:app --port 8000 --reload

Endpoints follow common/CONTRACT.md. /api/chat runs the LangGraph agent (app/agent/), whose
conversation memory is saved in SQLite per session_id. Phase 1: the graph has one node that checks
the real status page through MCP (UI -> our API -> graph -> MCP -> helpdesk API -> reply).

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
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

KIT = Path(__file__).resolve().parent.parent

# The kit's .env, without overriding variables that are already set (e.g. by Docker).
load_dotenv(KIT / ".env", override=False)

from app.agent.graph import build_graph  # noqa: E402  (after load_dotenv on purpose)
from app.agent.nodes import describe_status  # noqa: E402,F401  (re-exported: it used to live here)
from app.mcp_client import HelpdeskMCP  # noqa: E402
from app.schemas import ApproveRequest, ChatRequest, ChatResponse, ResetRequest, ToolCall  # noqa: E402
from app.tickets import ROUTING, TICKET_ID  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("helpdesk-app")

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS") or 1000)
DEFAULT_CHECKPOINT_DB = KIT / "app" / "data" / "checkpoints.db"

mcp = HelpdeskMCP()


def checkpoint_db_path() -> Path:
    """CHECKPOINT_DB, read when the app starts (so tests can point it elsewhere). Relative paths are under the kit."""
    path = Path(os.getenv("CHECKPOINT_DB") or DEFAULT_CHECKPOINT_DB)
    return path if path.is_absolute() else KIT / path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The SQLite checkpointer (conversation memory for the LangGraph agent) and the MCP server both live
    # exactly as long as the app. The saver binds to this event loop, so it must be created here.
    db_path = checkpoint_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        app.state.graph = build_graph(checkpointer, mcp)
        log.info("checkpointer ready: %s", db_path)
        await mcp.start()
        try:
            yield
        finally:
            await mcp.stop()
            app.state.graph = None
            app.state.checkpointer = None


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


def thread_config(session_id: str) -> dict:
    """The LangGraph config for one conversation: its checkpoints are stored under this thread id."""
    return {"configurable": {"thread_id": session_id}}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session = get_session(req.session_id)
    async with session["lock"]:
        return await _chat(req, session)


async def _chat(req: ChatRequest, session: dict) -> ChatResponse:
    graph = app.state.graph
    config = thread_config(req.session_id)

    # The saved conversation outlives the in-memory session (evicted, or the app restarted): take the
    # email back from the checkpoint, so the identity lock below still holds for this conversation.
    if session["user"] is None:
        saved_email = (await graph.aget_state(config)).values.get("user_email")
        if saved_email:
            session["user"] = {"email": saved_email}

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
    email = (session["user"] or {}).get("email")
    log.info("chat session=%s user=%s", req.session_id, email)

    # The session's (locked) email is the source of truth for user_email; active_incident is set by
    # the graph on every turn. Only plain strings and LangChain messages go into the checkpoint.
    turn: dict = {"messages": [HumanMessage(content=req.message)]}
    if email:
        turn["user_email"] = email
    try:
        out = await graph.ainvoke(turn, config)
    except Exception:  # the checkpoint database or a node failed: reply honestly, never a 500
        log.exception("chat session=%s: the agent failed", req.session_id)
        return ChatResponse(
            session_id=req.session_id,
            reply="Sorry, something went wrong on our side and I could not answer. Please try again in a moment.",
            state="error",
        )
    return ChatResponse(
        session_id=req.session_id,
        reply=out["reply"],
        state="skeleton",
        tool_calls=[ToolCall(**call) for call in out.get("tool_calls", [])],
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


async def delete_all_threads(checkpointer: AsyncSqliteSaver) -> int:
    """Delete every saved conversation in the checkpoint database, not only the ones still in memory.

    alist(None) lists the checkpoints of every thread (public API). Returns how many threads were deleted.
    """
    thread_ids = {c.config["configurable"]["thread_id"] async for c in checkpointer.alist(None)}
    for thread_id in thread_ids:
        await checkpointer.adelete_thread(thread_id)
    return len(thread_ids)


@app.post("/reset")
async def reset(req: ResetRequest | None = None, x_admin_token: str | None = Header(default=None)) -> dict:
    """Clear our app's session state and saved conversations. Never touches the knowledge base or the helpdesk API.

    One session: open to anyone who knows its id. Its in-memory state and its checkpoint thread (messages,
    user_email) are deleted, so the same session id starts fresh, even for another employee.
    Every session: needs the X-Admin-Token header to match APP_ADMIN_TOKEN, and is disabled when
    APP_ADMIN_TOKEN is not set. It deletes every thread in the checkpoint database, including ones no longer
    in memory (after an eviction or a restart). Limitation: it does not wait for requests already in progress;
    a turn that finishes after the reset saves its checkpoint again.
    """
    checkpointer = app.state.checkpointer
    if req and req.session_id:
        # get_session also covers a session that is only on disk: its lock then guards the deletion too.
        session = get_session(req.session_id)
        async with session["lock"]:              # wait for a request of this session to finish
            try:
                await checkpointer.adelete_thread(req.session_id)
            except Exception:
                log.exception("reset session=%s: could not delete the saved conversation", req.session_id)
                raise HTTPException(status_code=500, detail="The saved conversation could not be deleted.")
            sessions.pop(req.session_id, None)
        log.info("reset session=%s (in-memory state and saved conversation deleted)", req.session_id)
        return {"status": "ok", "reset": req.session_id}

    expected = os.getenv("APP_ADMIN_TOKEN") or ""
    if not expected or not secrets.compare_digest(x_admin_token or "", expected):
        log.warning("reset of all sessions refused (missing or wrong admin token)")
        raise HTTPException(status_code=403, detail="Resetting every session needs a valid X-Admin-Token.")
    count = len(sessions)
    try:
        threads = await delete_all_threads(checkpointer)
    except Exception:
        log.exception("reset all: could not delete the saved conversations")
        raise HTTPException(status_code=500, detail="The saved conversations could not be deleted.")
    sessions.clear()
    log.info("reset all sessions: %d in memory cleared, %d saved conversations deleted", count, threads)
    return {"status": "ok", "reset": "all", "sessions_cleared": count, "threads_deleted": threads}
