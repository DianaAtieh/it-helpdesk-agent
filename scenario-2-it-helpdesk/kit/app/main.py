"""Our helpdesk agent API (walking skeleton). Run from the kit folder:

    uvicorn app.main:app --port 8000 --reload

Endpoints follow common/CONTRACT.md. /api/chat runs the LangGraph agent (app/agent/), whose
conversation memory is saved in SQLite per session_id. Phase 2: triage (directory + status page over MCP),
then either a verified incident or a grounded answer from the knowledge base (app/agent/graph.py).

The helpdesk API must be running on port 8102 (uvicorn services.helpdesk_api:app --port 8102).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import secrets
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

KIT = Path(__file__).resolve().parent.parent

# The kit's .env, without overriding variables that are already set (e.g. by Docker).
load_dotenv(KIT / ".env", override=False)

from app.agent.graph import FINAL_NODE, build_graph  # noqa: E402  (after load_dotenv on purpose)
from app.agent.nodes import describe_status  # noqa: E402,F401  (re-exported: it used to live here)
from app.agent.retrieval import KnowledgeBaseError, load_knowledge_base  # noqa: E402
from app.agent.storage import delete_expired_threads, prune_thread, retention_days, trim_history  # noqa: E402
from app.guardrails import WITHHELD, looks_like_instructions, redact_secrets  # noqa: E402
from app.llm import LLMError, OpenRouterChat  # noqa: E402
from app.mcp_client import HelpdeskMCP  # noqa: E402
from app.schemas import (ApproveRequest, ChatRequest, ChatResponse, Citation, PendingAction,  # noqa: E402
                         ResetRequest, ToolCall)
from app.tickets import ROUTING, TICKET_ID, TicketDraft, TicketDraftError, propose_ticket  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("helpdesk-app")

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS") or 1000)
DEFAULT_CHECKPOINT_DB = KIT / "app" / "data" / "checkpoints.db"

mcp = HelpdeskMCP()

# A session id is all it takes to continue a conversation, so the logs never show it: they show a keyed
# hash instead. The key is new on every start, so a reference is stable only while the app runs.
_LOG_KEY = secrets.token_bytes(16)


def session_ref(session_id: str) -> str:
    """The session id as it appears in the logs: s#<10 hex characters>, not reversible."""
    return "s#" + hashlib.blake2b(session_id.encode(), key=_LOG_KEY, digest_size=5).hexdigest()


def checkpoint_db_path() -> Path:
    """CHECKPOINT_DB, read when the app starts (so tests can point it elsewhere). Relative paths are under the kit."""
    path = Path(os.getenv("CHECKPOINT_DB") or DEFAULT_CHECKPOINT_DB)
    return path if path.is_absolute() else KIT / path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The SQLite checkpointer (conversation memory for the LangGraph agent) and the MCP server both live
    # exactly as long as the app. The saver binds to this event loop, so it must be created here.
    # The knowledge base and the chat model are built once here and injected into the graph. If either
    # cannot be built (no index, no key, no MODEL), the app still starts and answers honestly that it
    # cannot search or answer right now.
    db_path = checkpoint_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        app.state.kb = load_knowledge_base()
    except KnowledgeBaseError as exc:
        log.warning("knowledge base unavailable (%s): chat answers will say so", exc.code)
        app.state.kb = None
    try:
        app.state.llm = OpenRouterChat.from_env()
    except LLMError as exc:
        log.warning("chat model unavailable (%s): chat answers will say so", exc.code)
        app.state.llm = None
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        app.state.graph = build_graph(checkpointer, mcp, app.state.kb, app.state.llm,
                                      evaluate_evidence=os.getenv("EVIDENCE_EVALUATION", "").strip().lower() in ("1", "true", "yes", "on"))
        expired = await delete_expired_threads(checkpointer)
        log.info("checkpointer ready: %s (%d conversations idle for more than %d days deleted)",
                 db_path, expired, retention_days())
        await mcp.start()
        try:
            yield
        finally:
            await mcp.stop()
            app.state.graph = None
            app.state.checkpointer = None


app = FastAPI(title="Nakhla IT Helpdesk Agent", version="0.1.0", lifespan=lifespan)

# Session state, keyed by session_id: an in-memory cache of what the checkpoint saves, plus the lock.
#   user            {"email": ...} or None                                (saved as user_email)
#   pending_action  the one PendingAction waiting for approval, or None   (saved as pending_action)
#   filed           {action_id: ticket_id} for approvals that already created a ticket (saved as filed)
#   loaded          True once the saved values were read back from the checkpoint
#   lock            one request per session at a time, so Approve and Reject cannot race
#   waiting         how many requests are waiting for the lock (see session_turn)
#   stored          True if the conversation had a checkpoint when it was loaded
#   token_hash      sha256 of the session token, or None                 (saved as token_hash)
#   token_issued_at ISO time the token was issued                        (saved as token_issued_at)
# Least recently used first; at most MAX_SESSIONS are kept, so new session ids cannot fill the memory.
# An evicted session loses nothing: load_session() reads it back from the checkpoint.
sessions: OrderedDict[str, dict] = OrderedDict()


def lock_wait_seconds() -> float:
    return float(os.getenv("SESSION_LOCK_WAIT_SECONDS") or 30)


def max_waiting_requests() -> int:
    return int(os.getenv("SESSION_MAX_WAITING") or 2)


class SessionBusy(Exception):
    """The session is busy with another request, and this one may not (or no longer) wait for it."""


@asynccontextmanager
async def session_turn(session: dict):
    """Hold the session's lock for one request, but never queue without limit (V9).

    One request of a session can hold the lock for a long time (one MCP call may take up to 25 s), so a
    request waits at most SESSION_LOCK_WAIT_SECONDS, and at most SESSION_MAX_WAITING requests may wait at
    once. Otherwise SessionBusy is raised and the request did nothing.
    """
    if session["lock"].locked() and session["waiting"] >= max_waiting_requests():
        raise SessionBusy
    session["waiting"] += 1
    try:
        await asyncio.wait_for(session["lock"].acquire(), lock_wait_seconds())
    except TimeoutError:
        raise SessionBusy from None
    finally:
        session["waiting"] -= 1
    try:
        yield
    finally:
        session["lock"].release()


def get_session(session_id: str) -> dict:
    session = sessions.get(session_id)
    if session is None:
        session = {"user": None, "pending_action": None, "filed": {}, "loaded": False, "lock": asyncio.Lock(),
                   "waiting": 0, "stored": False, "token_hash": None, "token_issued_at": None}
        sessions[session_id] = session
        _evict_idle_sessions(keep=session_id)
    sessions.move_to_end(session_id)
    return session


async def load_session(session_id: str, session: dict) -> None:
    """Fill a fresh in-memory session from its checkpoint (after a restart or an eviction). Call under the lock."""
    if session["loaded"]:
        return
    saved = (await app.state.graph.aget_state(thread_config(session_id))).values
    if saved.get("user_email"):
        session["user"] = {"email": saved["user_email"]}
    if saved.get("pending_action"):
        session["pending_action"] = PendingAction(**saved["pending_action"])
    session["filed"] = dict(saved.get("filed") or {})
    session["token_hash"] = saved.get("token_hash")
    session["token_issued_at"] = saved.get("token_issued_at")
    session["stored"] = bool(saved)
    session["loaded"] = True


async def save_approval_state(session_id: str, session: dict) -> None:
    """Save the draft and the filed record in the checkpoint, so a restart cannot lose or misreport them."""
    pending = session["pending_action"]
    await app.state.graph.aupdate_state(thread_config(session_id), {
        "pending_action": pending.model_dump() if pending else None,   # plain dict: checkpoint-safe
        "filed": dict(session["filed"]),
    }, as_node=FINAL_NODE)
    session["stored"] = True          # it has a checkpoint now: never issue it a token later
    await prune_thread(app.state.checkpointer, session_id)


async def propose_draft(session_id: str, draft: TicketDraft) -> tuple[PendingAction, list[ToolCall]]:
    """The one way to create a ticket draft: load the session, check the draft (propose_ticket), save it.

    Call it under the session's lock (the agent will, from inside /api/chat). Writes no ticket.
    Raises TicketDraftError for an invalid draft; nothing is stored then.
    """
    session = get_session(session_id)
    await load_session(session_id, session)
    action, calls = await propose_ticket(session, draft, mcp)
    await save_approval_state(session_id, session)
    return action, calls


CLAIM_NOTE = ("For your privacy, I don't carry over anything said in this conversation before you gave your "
              "email. Please describe your problem again if I need it.")


async def clear_anonymous_history(session_id: str, session: dict) -> bool:
    """Before an anonymous conversation is linked to an email: forget everything said in it. Call under the lock.

    An email is a claim, not proof (anyone who knows the session id can send one), so the employee who links
    the conversation gets none of its earlier messages, replies or ticket draft. The filed record stays, so a
    replayed approval is still answered truthfully. Returns True if something was cleared.
    Raises if the database fails; the caller must then not link the email.
    """
    config = thread_config(session_id)
    saved = (await app.state.graph.aget_state(config)).values
    cleared = bool(saved.get("messages") or saved.get("pending_action") or session["pending_action"])
    if cleared:
        await app.state.graph.aupdate_state(config, {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            "pending_action": None,
            "reply": None,
            "active_incident": None,
            "tool_calls": [],
            "filed": dict(session["filed"]),
        }, as_node=FINAL_NODE)
    # Always: the older checkpoints still hold the history. This also finishes an earlier claim whose
    # pruning failed after the history was already cleared from the newest checkpoint.
    await prune_thread(app.state.checkpointer, session_id)
    session["pending_action"] = None
    return cleared


def _evict_idle_sessions(keep: str) -> None:
    """Drop the least recently used sessions above MAX_SESSIONS, never one with a request in progress or
    waiting, and never `keep` (the session just created: evicting it made the request fail with a KeyError)."""
    for session_id in list(sessions):
        if len(sessions) <= MAX_SESSIONS:
            break
        if session_id != keep and not sessions[session_id]["lock"].locked() and not sessions[session_id]["waiting"]:
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


# ---------------------------------------------------------------- session token (who may approve)
# A session id names a conversation; it does not prove who is asking (and an email is only a claim). So the
# first /api/chat of a NEW conversation issues a random token to that client, and /api/approve (approve AND
# reject) and /reset of that conversation require it. The server keeps only its sha256 in the checkpoint.
# The client gets it two ways, and sends it back either way (the header wins when both are present):
#   * an HttpOnly cookie, so clients that keep cookies (httpx.Client, as in evals/run_evals.py) need no change;
#   * the X-Session-Token response header, for clients that keep it themselves (the Gradio UI).
# Never in a URL, a response body or a log. The course's JSON contract is unchanged.
SESSION_TOKEN_HEADER = "X-Session-Token"


def session_token_ttl_seconds() -> int:
    """How long a token may approve, from when it was issued. After that: start a new conversation."""
    return int(os.getenv("SESSION_TOKEN_TTL_SECONDS") or 24 * 3600)


def session_cookie_secure() -> bool:
    """Set SESSION_COOKIE_SECURE=true behind HTTPS. Off by default: a Secure cookie is never sent over plain
    http, which the local app, the tests and the eval runner use."""
    return (os.getenv("SESSION_COOKIE_SECURE") or "").strip().lower() in {"1", "true", "yes"}


def session_cookie_name(session_id: str) -> str:
    """One cookie per conversation, so several sessions in one cookie jar never overwrite each other."""
    return "hd_session_" + hashlib.sha256(session_id.encode()).hexdigest()[:16]


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def presented_token(request: Request, session_id: str) -> str | None:
    """The token the client sent: the X-Session-Token header if present, otherwise this session's cookie."""
    return request.headers.get(SESSION_TOKEN_HEADER) or request.cookies.get(session_cookie_name(session_id))


def token_problem(session: dict, presented: str | None) -> str | None:
    """None if `presented` is this session's token and still valid; otherwise "missing", "invalid" or "expired".

    Expiry is only reported to a client that holds the right token."""
    if not presented:
        return "missing"
    stored = session["token_hash"]
    if not stored or not hmac.compare_digest(hash_token(presented), stored):
        return "invalid"
    issued = datetime.fromisoformat(session["token_issued_at"])
    if datetime.now(timezone.utc) - issued > timedelta(seconds=session_token_ttl_seconds()):
        return "expired"
    return None


def new_session_token() -> tuple[str, dict]:
    """A fresh token and the state fields that store it (its hash and the time it was issued)."""
    token = secrets.token_urlsafe(32)
    return token, {"token_hash": hash_token(token), "token_issued_at": datetime.now(timezone.utc).isoformat()}


def send_session_token(response: Response, session_id: str, token: str) -> None:
    response.headers[SESSION_TOKEN_HEADER] = token
    response.headers["Cache-Control"] = "no-store"
    response.set_cookie(session_cookie_name(session_id), token, max_age=session_token_ttl_seconds(),
                        httponly=True, samesite="strict", secure=session_cookie_secure(), path="/")


async def issue_session_token(session_id: str) -> str | None:
    """Give a NEW conversation (no checkpoint, no token) its token and save the hash. None if it already has
    a checkpoint: a token is never issued twice, or anyone who knows the session id could get one.

    /api/chat issues tokens inside the turn; this is for paths that start a conversation without a chat turn
    (the tests' propose helper). Call under the session's lock.
    """
    session = get_session(session_id)
    await load_session(session_id, session)
    if session["token_hash"] or session["stored"]:
        return None
    token, fields = new_session_token()
    await app.state.graph.aupdate_state(thread_config(session_id), fields, as_node=FINAL_NODE)
    await prune_thread(app.state.checkpointer, session_id)
    session.update(fields, stored=True)
    return token


UNAUTHORIZED_REPLY = {
    "missing": "This conversation's approval key was not sent, so I did not act on this request. Nothing was changed.",
    "invalid": "This approval key does not belong to this conversation, so I did not act on it. Nothing was changed.",
    "expired": ("The approval key of this conversation has expired, so I did not act on it. Nothing was changed. "
                "Please start a new conversation and ask for the ticket again."),
}


class Unauthorized(Exception):
    """The request did not carry this conversation's valid session token."""

    def __init__(self, problem: str):
        super().__init__(problem)
        self.problem = problem


def thread_config(session_id: str) -> dict:
    """The LangGraph config for one conversation: its checkpoints are stored under this thread id."""
    return {"configurable": {"thread_id": session_id}}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, response: Response) -> ChatResponse:
    session = get_session(req.session_id)
    try:
        async with session_turn(session):
            return await _chat(req, session, response)
    except SessionBusy:
        log.info("chat session=%s busy: an earlier request is still running", session_ref(req.session_id))
        return ChatResponse(
            session_id=req.session_id,
            reply=("I'm still working on an earlier message in this conversation, so I did not read this one. "
                   "Please send it again in a moment."),
            state="busy",
        )


def agent_error_reply(session_id: str) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        reply="Sorry, something went wrong on our side and I could not answer. Please try again in a moment.",
        state="error",
    )


async def _chat(req: ChatRequest, session: dict, response: Response) -> ChatResponse:
    graph = app.state.graph
    config = thread_config(req.session_id)

    # The saved conversation outlives the in-memory session (evicted, or the app restarted): read the
    # email, the draft and the filed record back, so the identity lock and the gate still hold.
    try:
        await load_session(req.session_id, session)
    except Exception:  # the checkpoint database failed: reply honestly, never a 500
        log.exception("chat session=%s: could not load the saved conversation", session_ref(req.session_id))
        return agent_error_reply(req.session_id)

    # Without a login the email is a claim, but once a conversation is linked to an employee, every turn
    # must name that same employee: a different email (e.g. a VIP for a P1 ticket) or NO email (someone who
    # only knows the session id) cannot continue it.
    current = (session["user"] or {}).get("email")
    if current and (req.user is None or req.user.email != current):
        log.warning("chat session=%s refused: linked to %s, request names %s", session_ref(req.session_id), current,
                    req.user.email if req.user else "no user")
        return ChatResponse(
            session_id=req.session_id,
            reply=("This conversation is already linked to another employee. Please start a new conversation."
                   if req.user else
                   "This conversation is linked to an employee. Please send your work email with every message."),
            state="identity_mismatch",
        )
    claimed = False
    if req.user and not current:
        # First email on this conversation: link it only after its anonymous past is gone (V6).
        try:
            claimed = await clear_anonymous_history(req.session_id, session)
        except Exception:  # could not forget the history: do not link the email, do not run the graph
            log.exception("chat session=%s: could not clear the anonymous history", session_ref(req.session_id))
            return agent_error_reply(req.session_id)
        if claimed:
            log.info("chat session=%s: anonymous history cleared before linking an email",
                     session_ref(req.session_id))
        session["user"] = req.user.model_dump()
    email = (session["user"] or {}).get("email")
    log.info("chat session=%s user=%s", session_ref(req.session_id), email)

    # Secrets never reach the agent, the checkpoint database or an external API (SP-001).
    message = redact_secrets(req.message)
    if message != req.message:
        log.info("chat session=%s: secrets redacted from the message", session_ref(req.session_id))

    # The session's (locked) email is the source of truth for user_email; active_incident is set by
    # the graph on every turn. Only plain strings and LangChain messages go into the checkpoint.
    # The message has its own id, so it can be replaced after the turn (see below).
    human = HumanMessage(content=message, id=str(uuid.uuid4()))
    turn: dict = {"messages": [human]}
    if email:
        turn["user_email"] = email
    # A new conversation gets its token in this turn: saved with the turn, sent only if the turn succeeds.
    new_token, token_fields = None, {}
    if not session["token_hash"] and not session["stored"]:
        new_token, token_fields = new_session_token()
        turn.update(token_fields)
    try:
        out = await graph.ainvoke(turn, config)
        # A ticket draft proposed this turn (Phase 4) becomes a pending action only through propose_draft, which
        # checks it against the directory and the ticket schema. Nothing is filed here: only /api/approve files.
        pending_action, draft_calls = None, []
        if out.get("outcome") == "draft_proposed" and isinstance(out.get("draft"), dict):
            try:
                pending_action, draft_calls, failure = await _propose_turn_draft(req.session_id, session,
                                                                                 out["draft"])
            except Exception:  # e.g. the database failed while saving the draft (the old draft is restored)
                log.exception("chat session=%s: the ticket draft could not be saved", session_ref(req.session_id))
                pending_action, draft_calls, failure = None, [], DRAFT_SAVE_FAILED_REPLY
            if failure:
                # The saved reply said a draft was prepared; replace it, so the history never says that falsely.
                await graph.aupdate_state(config, {
                    "messages": [AIMessage(content=failure, id=out["messages"][-1].id)],
                    "reply": failure, "outcome": "error", "draft": None,
                }, as_node=FINAL_NODE)
                out = {**out, "reply": failure, "outcome": "error", "draft": None}
        # Instructions aimed at the assistant are answered this turn only, never replayed from the history (V4).
        if looks_like_instructions(message):
            await graph.aupdate_state(config, {"messages": [HumanMessage(content=WITHHELD, id=human.id)]},
                                      as_node=FINAL_NODE)
            log.info("chat session=%s: instruction-like message withheld from later turns",
                     session_ref(req.session_id))
        # Keep the stored conversation bounded: the last messages only, and only the newest checkpoint.
        await trim_history(graph, config)
        await prune_thread(app.state.checkpointer, req.session_id)
    except Exception:  # the checkpoint database or a node failed: reply honestly, never a 500
        log.exception("chat session=%s: the agent failed", session_ref(req.session_id))
        return agent_error_reply(req.session_id)
    session["stored"] = True
    if new_token:
        session.update(token_fields)
        send_session_token(response, req.session_id, new_token)
    # The graph's outcome maps onto the contract's existing fields: state (answered | clarify | incident |
    # not_documented | blocked | error, or await_approval with a pending_action when a draft was stored),
    # citations (only documents in this turn's evidence, checked by respond) and usage.
    return ChatResponse(
        session_id=req.session_id,
        reply=f"{CLAIM_NOTE} {out['reply']}" if claimed else out["reply"],
        state="await_approval" if pending_action else (out.get("outcome") or "error"),
        pending_action=pending_action,
        citations=[Citation(id=c["id"], title=c.get("title") or "") for c in out.get("citations") or []],
        tool_calls=[ToolCall(**call) for call in out.get("tool_calls", [])] + list(draft_calls),
        usage=out.get("usage") or None,
    )


DRAFT_IDENTITY_REPLY = ("I couldn't prepare the ticket draft, because this conversation isn't linked to the work "
                        "email it is for. Nothing was filed and there is nothing to approve.")
DRAFT_INVALID_REPLY = ("I couldn't prepare a valid ticket draft: the details didn't pass the ticket checks. Nothing "
                       "was filed and there is nothing to approve. Please try again, or describe the problem in a "
                       "bit more detail.")
DRAFT_SAVE_FAILED_REPLY = ("Sorry, I couldn't save the ticket draft right now. Nothing was filed and there is "
                           "nothing new to approve. Please try again in a moment.")


async def _propose_turn_draft(session_id: str, session: dict,
                              draft: dict) -> tuple[PendingAction | None, list[ToolCall], str | None]:
    """Store a draft proposed by the graph as the pending action, or say why not. Call under the session lock.

    Returns (pending_action, tool_calls, failure_reply). Fails closed: the requester must be this conversation's
    locked email (the graph sets it from there; this checks again, so no model output can ever choose it).
    """
    locked = str((session["user"] or {}).get("email") or "").strip().lower()
    requester = str(draft.get("requester_email") or "").strip().lower()
    if not locked or requester != locked:
        log.warning("chat session=%s: ticket draft refused: the requester is not the conversation's email",
                    session_ref(session_id))
        return None, [], DRAFT_IDENTITY_REPLY
    previous = session["pending_action"]
    try:
        action, calls = await propose_draft(session_id, TicketDraft(**draft))
    except (TicketDraftError, ValidationError) as exc:
        problems = getattr(exc, "problems", None) or ["invalid draft fields"]
        log.info("chat session=%s: ticket draft not valid (%d problem(s))", session_ref(session_id), len(problems))
        return None, list(getattr(exc, "tool_calls", [])), DRAFT_INVALID_REPLY
    except Exception:
        session["pending_action"] = previous          # never leave an unsaved draft approvable
        raise
    log.info("chat session=%s: ticket draft %s waiting for approval", session_ref(session_id), action.action_id)
    return action, calls, None


@app.post("/api/approve", response_model=ChatResponse)
async def approve(req: ApproveRequest, request: Request) -> ChatResponse:
    """The approval gate: the only place in the app that writes a ticket.

    The session lock makes Approve, Reject and chat turns of one session run one after another:
    without it, a Reject arriving while create_ticket is in flight said "not filed" for a filed ticket.
    Approve and reject both need the conversation's session token (header or cookie); without a valid one
    the answer is HTTP 403 and nothing is changed or written.
    """
    session = get_session(req.session_id)
    try:
        async with session_turn(session):
            return await _approve(req, session, presented_token(request, req.session_id))
    except Unauthorized as refused:
        log.warning("approve session=%s refused: %s session token", session_ref(req.session_id), refused.problem)
        body = ChatResponse(session_id=req.session_id, reply=UNAUTHORIZED_REPLY[refused.problem],
                            state="unauthorized")
        return JSONResponse(status_code=403, content=body.model_dump())
    except SessionBusy:
        log.info("approve session=%s busy: an earlier request is still running", session_ref(req.session_id))
        return ChatResponse(
            session_id=req.session_id,
            reply=("I'm still working on an earlier request in this conversation, so I did not act on this one. "
                   "Please try again in a moment."),
            state="busy",
        )


async def _save_or_log(session_id: str, session: dict) -> None:
    """Persist the gate's state. A failure is logged, never turned into a wrong answer about the ticket."""
    try:
        await save_approval_state(session_id, session)
    except Exception:
        log.exception("approve session=%s: could not save the approval state", session_ref(session_id))


async def _approve(req: ApproveRequest, session: dict, token: str | None) -> ChatResponse:
    # After a restart or an eviction, the draft and the filed record come back from the checkpoint.
    try:
        await load_session(req.session_id, session)
    except Exception:  # cannot know the draft's state: write nothing, and say so
        log.exception("approve session=%s: could not load the saved approval state", session_ref(req.session_id))
        return ChatResponse(
            session_id=req.session_id,
            reply="I could not check the ticket draft right now, so nothing was filed. Please try again in a moment.",
            state="error",
        )

    # Only the client that holds this conversation's token may approve or reject, or learn a ticket id.
    # Checked before anything else: a refused request changes no state and calls no tool.
    problem = token_problem(session, token)
    if problem:
        raise Unauthorized(problem)

    # Replay of an approval that already created a ticket (double click): never write twice.
    already = session["filed"].get(req.action_id)
    if already:
        log.info("approve replay session=%s action=%s ticket=%s", session_ref(req.session_id), req.action_id, already)
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
                 session_ref(req.session_id), req.action_id)
        return ChatResponse(
            session_id=req.session_id,
            reply="There is no ticket draft waiting for approval, so nothing was filed.",
        )

    if not req.approved:
        session["pending_action"] = None
        await _save_or_log(req.session_id, session)
        log.info("approve rejected session=%s action=%s", session_ref(req.session_id), req.action_id)
        return ChatResponse(
            session_id=req.session_id,
            reply="Okay, I did not file the ticket. Tell me if you want to change anything.",
        )

    # Exactly the approved payload, with the action id as the idempotency key.
    data, call = await mcp.call("create_ticket", {**pending.payload, "idempotency_key": pending.action_id},
                                approved=True)
    ticket_id = str((data.get("ticket") or {}).get("ticket_id", "")) if data.get("ok") else ""

    if data.get("ok") and TICKET_ID.fullmatch(ticket_id):
        session["pending_action"] = None
        session["filed"][pending.action_id] = ticket_id
        await _save_or_log(req.session_id, session)
        team = data["ticket"].get("assigned_team") or ROUTING.get(pending.payload["category"], "IT")
        log.info("ticket created session=%s action=%s ticket=%s", session_ref(req.session_id), pending.action_id,
                 ticket_id)
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
        await _save_or_log(req.session_id, session)
    log.warning("ticket not created session=%s action=%s error=%s", session_ref(req.session_id), pending.action_id,
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


def admin_token_ok(token: str | None) -> bool:
    """True if the X-Admin-Token header matches APP_ADMIN_TOKEN. Always False when APP_ADMIN_TOKEN is not set."""
    expected = os.getenv("APP_ADMIN_TOKEN") or ""
    return bool(expected) and secrets.compare_digest(token or "", expected)


@app.post("/reset")
async def reset(request: Request, response: Response, req: ResetRequest | None = None,
                x_admin_token: str | None = Header(default=None)) -> dict:
    """Clear our app's session state and saved conversations. Never touches the knowledge base or the helpdesk API.

    One session: its in-memory state and its checkpoint thread (messages, user_email, token) are deleted, so the
    same session id starts fresh, even for another employee. A conversation with a session token can only be
    reset with that token (header or cookie; an expired one still proves it is yours) or with the admin token.
    Older conversations without a token keep the V11 rule: the linked employee's email in "user", or the admin
    token; an anonymous one by anyone who knows its id.
    Every session: needs the X-Admin-Token header to match APP_ADMIN_TOKEN, and is disabled when
    APP_ADMIN_TOKEN is not set. It deletes every thread in the checkpoint database, including ones no longer
    in memory (after an eviction or a restart). Limitation: it does not wait for requests already in progress;
    a turn that finishes after the reset saves its checkpoint again.
    """
    checkpointer = app.state.checkpointer
    if req and req.session_id:
        # get_session also covers a session that is only on disk: its lock then guards the deletion too.
        session = get_session(req.session_id)
        try:
            async with session_turn(session):        # wait (a limited time) for a request of this session
                await _reset_one(req, session, x_admin_token, presented_token(request, req.session_id))
        except SessionBusy:
            raise HTTPException(status_code=409, detail="This conversation is busy. Try the reset again in a moment.")
        log.info("reset session=%s (in-memory state and saved conversation deleted)", session_ref(req.session_id))
        response.delete_cookie(session_cookie_name(req.session_id), path="/")
        return {"status": "ok", "reset": req.session_id}

    if not admin_token_ok(x_admin_token):
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


async def _reset_one(req: ResetRequest, session: dict, x_admin_token: str | None, token: str | None) -> None:
    """Delete one conversation, if the caller may. Call under the session's lock. Raises HTTPException otherwise."""
    try:
        await load_session(req.session_id, session)  # the linked email may be only on disk
    except Exception:
        log.exception("reset session=%s: could not load the saved conversation", session_ref(req.session_id))
        raise HTTPException(status_code=500,
                            detail="The saved conversation could not be checked, so nothing was deleted.")
    if session["token_hash"]:
        if not admin_token_ok(x_admin_token) and token_problem(session, token) not in (None, "expired"):
            log.warning("reset session=%s refused: no valid session token", session_ref(req.session_id))
            raise HTTPException(status_code=403,
                                detail="Resetting this conversation needs its session token (or the admin token).")
    linked = (session["user"] or {}).get("email")
    if (not session["token_hash"] and linked and not (req.user and req.user.email == linked)
            and not admin_token_ok(x_admin_token)):
        log.warning("reset session=%s refused: the request does not name the linked employee",
                    session_ref(req.session_id))
        raise HTTPException(status_code=403,
                            detail="This conversation is linked to an employee. Send the same work email to reset it.")
    try:
        await app.state.checkpointer.adelete_thread(req.session_id)
    except Exception:
        log.exception("reset session=%s: could not delete the saved conversation", session_ref(req.session_id))
        raise HTTPException(status_code=500, detail="The saved conversation could not be deleted.")
    sessions.pop(req.session_id, None)
