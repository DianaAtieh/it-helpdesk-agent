"""'Show me my tickets': recognise the request, read its filters, fetch the employee's own tickets over MCP.

Deterministic, no model. The filters come only from words in the message, and only values that exist:
  status    open | resolved | closed       (the ticket schema's ticket_record.status)
  approval  pending | approved | rejected  (the employee's decision on the draft, app/ticket_store.py)
They are independent ("my approved open tickets" uses both). "In progress" is not a status this helpdesk has, so
it is never invented: the reply says so and shows every ticket instead.

The employee is the conversation's directory-checked record (triage), never a name or email in the message.
"""
from __future__ import annotations

import re
from typing import Awaitable, Callable

from app.agent import escalation

OUTCOME = "tickets"

_TICKETS = re.compile(r"(?i)\btickets?\b")
_MINE = re.compile(r"(?i)\b(?:my|mine|i\s+(?:approved|rejected|declined|created|opened|filed|submitted|made|"
                   r"raised|have)|previous|past|earlier|old)\b")
_ASKS = re.compile(r"(?i)\b(?:show|list|see|view|display|check|what|which|where|any|status|happened|give|get|find|"
                   r"have|bring|load)\b")
_STATUS = (("open", re.compile(r"(?i)\bopen\b")),
           ("resolved", re.compile(r"(?i)\b(?:resolved|solved|fixed)\b")),
           ("closed", re.compile(r"(?i)\bclosed\b")))
_APPROVAL = (("approved", re.compile(r"(?i)\bapproved\b")),
             ("rejected", re.compile(r"(?i)\b(?:rejected|declined|refused)\b")),
             ("pending", re.compile(r"(?i)\b(?:pending|waiting|awaiting|undecided)\b")))
_UNKNOWN_STATUS = re.compile(r"(?i)\bin[\s-]progress\b")
_ALL = re.compile(r"(?i)\ball\b")


def is_request(text: str) -> bool:
    """True for a request to SEE the employee's tickets; never for a request to open one ("open a ticket")."""
    text = text or ""
    return (bool(_TICKETS.search(text) and _MINE.search(text) and _ASKS.search(text))
            and not escalation.explicit_ticket_request(text))


def read_filter(text: str) -> dict:
    """{"status", "approval"} from the message (None = any), plus "unsupported" for a status that does not exist."""
    text = text or ""
    if _ALL.search(text) and not any(p.search(text) for _, p in (*_STATUS, *_APPROVAL)):
        return {"status": None, "approval": None, "unsupported": None}
    status = next((value for value, pattern in _STATUS if pattern.search(text)), None)
    approval = next((value for value, pattern in _APPROVAL if pattern.search(text)), None)
    unsupported = "in progress" if _UNKNOWN_STATUS.search(text) else None
    return {"status": status, "approval": approval, "unsupported": unsupported}


def describe(selected: dict) -> str:
    words = [w for w in (selected.get("approval"), selected.get("status")) if w]
    return " ".join([*words, "tickets"])


def reply_for(tickets: list[dict], selected: dict) -> str:
    """A short, code-written line above the cards (the records themselves are rendered by the UI)."""
    note = ""
    if selected.get("unsupported"):
        note = (f"\"{selected['unsupported'].capitalize()}\" is not a ticket status here (tickets are open, resolved "
                f"or closed), so I'm showing all of them. ")
    what = describe(selected)
    if not tickets:
        return f"{note}You don't have any {what}."
    return f"{note}Here are your {what} ({len(tickets)}). They are shown in the ticket list."


NOT_IDENTIFIED_REPLY = ("I can only show tickets for a signed-in employee from the company directory, so I can't "
                        "list any here.")
UNAVAILABLE_REPLY = "Sorry, I couldn't load your tickets right now. Please try again in a moment."


def make_my_tickets(tickets_mcp) -> Callable[[dict], Awaitable[dict]]:
    """The graph node for route "tickets": one get_my_tickets call for the conversation's own employee."""

    async def my_tickets(state: dict) -> dict:
        text, _ = escalation.last_messages(state.get("messages") or [])
        selected = read_filter(text)
        query = {"status": selected["status"], "approval": selected["approval"]}
        base = {"outcome": OUTCOME, "ticket_filter": query, "tickets": [], "citations": [], "usage": {}}
        employee = state.get("employee")
        email = str(state.get("user_email") or "").strip().lower()
        if not email or not isinstance(employee, dict) or str(employee.get("email", "")).lower() != email:
            return {**base, "reply": NOT_IDENTIFIED_REPLY}
        if tickets_mcp is None:
            return {**base, "reply": UNAVAILABLE_REPLY}
        args = {"requester_email": email, **{k: v for k, v in query.items() if v}}
        data, call = await tickets_mcp.call("get_my_tickets", args)
        calls = [*(state.get("tool_calls") or []), call.model_dump()]
        if not data.get("ok"):
            return {**base, "reply": UNAVAILABLE_REPLY, "tool_calls": calls}
        tickets = [t for t in data.get("tickets") or [] if isinstance(t, dict)]   # ownership: the SQL WHERE
        return {**base, "tickets": tickets, "reply": reply_for(tickets, selected), "tool_calls": calls}

    return my_tickets
