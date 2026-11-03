"""'Show me my tickets': recognise the request, read its filters, fetch the employee's own tickets over MCP.

Deterministic, no model. The filters come only from words in the message, and only values that exist:
  status    open | resolved | closed       (the ticket schema's ticket_record.status)
  approval  pending | approved | rejected  (the employee's decision on the draft, app/ticket_store.py)
They are independent ("my approved open tickets" uses both). "In progress" is not a status this helpdesk has, so
it is never invented: the reply says so and shows every ticket instead.

The employee is the conversation's directory-checked record (triage), never a name or email in the message.

"What is the status of IT-1021?": one ticket, by its ID. get_ticket can return any employee's ticket, so ownership is
checked here, in code, against the conversation's locked email. A ticket of someone else gets the same reply as one
that does not exist, so a reply never tells whether an ID is in use. Only the status, team and subject are shown: a
ticket's notes and resolution are untrusted text.

The IT number is the helpdesk's and may repeat: the simulated API starts again at IT-2001 after a restart. The
Application Reference (from the unique action_id) is what stays the same for a ticket. It is also the last line of the
filed description, so a helpdesk record is matched to a saved ticket by that exact reference; a different reference is
never the same ticket. Tickets from before the reference have none: they match only when everything else agrees. In
every case exactly one saved ticket must fit (same_ticket); otherwise the last saved status is shown, never a guess.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Awaitable, Callable

from app.agent import escalation
from app.tickets import REFERENCE_LABEL, reference

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
_REAL_NUMBER = re.compile(r"IT-\d{4}")             # the ticket schema's form: any other number cannot be a ticket
_TICKET_ID = re.compile(r"(?i)\bIT-\d+\b|\bREF-[0-9A-Z]+(?:\.[0-9A-Z]+)?\b")
MATCH_SECONDS = 300          # how far apart the helpdesk's creation time and the saved approval may be
_ASKS_STATUS = re.compile(r"(?i)\b(?:status|state|update|progress|happened|happening|open|closed|resolved|solved|fixed|"
                          r"done|finished|yet|check|where|about)\b")


def asked_ticket(text: str) -> str | None:
    """The one IT number or Application Reference the message asks about, or None (none, several, or a request to
    open a ticket)."""
    text = text or ""
    ids = {found.upper() for found in _TICKET_ID.findall(text)}
    if len(ids) != 1 or not _ASKS_STATUS.search(text) or escalation.explicit_ticket_request(text):
        return None
    return ids.pop()


def is_request(text: str) -> bool:
    """True for a request to SEE the employee's tickets; never for a request to open one ("open a ticket")."""
    text = text or ""
    if asked_ticket(text):
        return True
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
NOT_FOUND_REPLY = "I couldn't find that ticket on your account. Please check the ticket number."


LAST_SAVED_NOTE = "This is the last saved status. The helpdesk may no longer have this ticket after a restart."


def _reference_note(ticket: dict) -> str:
    return f" Application Reference: {reference(ticket['action_id'])}." if ticket.get("action_id") else ""


def status_reply(ticket: dict) -> str:
    team = f" Assigned team: {ticket['assigned_team']}." if ticket.get("assigned_team") else ""
    return (f"Your ticket {ticket['ticket_id']} ({ticket.get('subject', '')}) is {ticket.get('status')}."
            f"{team}{_reference_note(ticket)}")


def last_saved_reply(ticket: dict) -> str:
    return (f"Your ticket {ticket['ticket_id']} ({ticket.get('subject', '')}) has the saved status "
            f"{ticket.get('status')}.{_reference_note(ticket)} {LAST_SAVED_NOTE}")


def not_filed_reply(ticket: dict) -> str:
    return (f"Application Reference {reference(ticket.get('action_id'))} ({ticket.get('subject', '')}) has no IT ticket "
            f"number: it was not filed (approval: {ticket.get('approval')}).")


def several_reply(ticket_id: str, tickets: list[dict], current: dict | None) -> str:
    lines = [f"- Created {str(t.get('created_at') or '')[:10]}, {t.get('subject', '')}: "
             f"{t.get('status')} ({'confirmed' if t is current else 'last saved'}). "
             f"Application Reference: {reference(t.get('action_id'))}" for t in tickets]
    note = "" if current is not None and len(tickets) == 1 else f"\n\n{LAST_SAVED_NOTE}"
    return f"You have {len(tickets)} tickets numbered {ticket_id}:\n" + "\n".join(lines) + note


def card(ticket: dict) -> dict:
    """The fields of one owned helpdesk ticket that may be shown. Never its notes or resolution."""
    return {key: ticket.get(key) for key in ("ticket_id", "subject", "category", "priority", "department",
                                             "assigned_team", "status", "created_at")} | {"approval": "approved"}


def _when(value) -> datetime:
    return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ")


def filed_reference(description) -> str | None:
    """The Application Reference on the last line of a ticket description, or None (a ticket from before it)."""
    last = str(description or "").rstrip().rsplit("\n", 1)[-1].strip()
    return last.removeprefix(REFERENCE_LABEL) if last.startswith(REFERENCE_LABEL + "REF-") else None


def same_ticket(record: dict, saved: dict) -> bool:
    """True only if a helpdesk record is this saved ticket: by the exact reference, or, when neither side has one (a
    ticket from before the reference), if the subject, the whole description and the time all agree."""
    filed, kept = filed_reference(record.get("description")), filed_reference(saved.get("description"))
    if filed or kept:
        return filed == kept == reference(saved.get("action_id"))
    if record.get("subject") != saved.get("subject") or record.get("description") != saved.get("description"):
        return False
    try:
        return abs((_when(record.get("created_at")) - _when(saved.get("updated_at"))).total_seconds()) <= MATCH_SECONDS
    except ValueError:
        return False


def make_my_tickets(tickets_mcp, helpdesk=None) -> Callable[[dict], Awaitable[dict]]:
    """The graph node for route "tickets": one get_my_tickets call for the conversation's own employee, or, for one
    ticket, get_ticket with the ownership check next to the employee's own saved tickets."""

    async def one_ticket(asked: str, email: str, base: dict) -> dict:
        calls = list(base["tool_calls"])
        saved: list[dict] = []
        unavailable = tickets_mcp is None
        if tickets_mcp is not None:          # ownership: the query filters by the locked email
            data, call = await tickets_mcp.call("get_my_tickets", {"requester_email": email})
            calls.append(call.model_dump())
            unavailable = not data.get("ok")
            saved = [t for t in data.get("tickets") or [] if isinstance(t, dict)]
        by_reference = asked.startswith("REF-")
        if not by_reference and not _REAL_NUMBER.fullmatch(asked):
            return {**base, "reply": NOT_FOUND_REPLY, "tool_calls": calls}
        matches = [t for t in saved if (reference(t.get("action_id")) == asked if by_reference
                                        else t.get("ticket_id") == asked)]
        if by_reference and not matches:
            return {**base, "reply": UNAVAILABLE_REPLY if unavailable else NOT_FOUND_REPLY, "tool_calls": calls}
        number = matches[0].get("ticket_id") if by_reference else asked
        if not number:                       # a pending or rejected draft: known here only
            return {**base, "tickets": matches, "reply": not_filed_reply(matches[0]), "tool_calls": calls}

        record = None
        if helpdesk is None:
            unavailable = True
        else:
            data, call = await helpdesk.call("get_ticket", {"ticket_id": number})
            calls.append(call.model_dump())
            found = data.get("ticket") if data.get("ok") else None
            if isinstance(found, dict) and str(found.get("requester_email", "")).strip().lower() == email:
                record = found               # ownership: the locked email, in code
            unavailable = unavailable or (not data.get("ok")
                                          and (data.get("error") or {}).get("code") != "ticket_not_found")
        if not matches:                      # not saved here, for example a historical ticket
            if record:
                return {**base, "tickets": [card(record)], "reply": status_reply(record), "tool_calls": calls}
            return {**base, "reply": UNAVAILABLE_REPLY if unavailable else NOT_FOUND_REPLY, "tool_calls": calls}

        confirmed = [t for t in matches if record and same_ticket(record, t)]
        current = confirmed[0] if len(confirmed) == 1 else None
        if current is not None:
            shown = {**current, "status": record.get("status"),
                     "assigned_team": record.get("assigned_team") or current.get("assigned_team")}
            matches = [shown if t is current else t for t in matches]
            current = shown
        if len(matches) > 1:
            return {**base, "tickets": matches, "reply": several_reply(number, matches, current), "tool_calls": calls}
        reply = status_reply(current) if current else last_saved_reply(matches[0])
        return {**base, "tickets": matches, "reply": reply, "tool_calls": calls}

    async def my_tickets(state: dict) -> dict:
        text, _ = escalation.last_messages(state.get("messages") or [])
        selected = read_filter(text)
        query = {"status": selected["status"], "approval": selected["approval"]}
        base = {"outcome": OUTCOME, "ticket_filter": query, "tickets": [], "citations": [], "usage": {}}
        employee = state.get("employee")
        email = str(state.get("user_email") or "").strip().lower()
        if not email or not isinstance(employee, dict) or str(employee.get("email", "")).lower() != email:
            return {**base, "reply": NOT_IDENTIFIED_REPLY}
        ticket_id = asked_ticket(text)
        if ticket_id:
            return await one_ticket(ticket_id, email, {**base, "tool_calls": list(state.get("tool_calls") or [])})
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
