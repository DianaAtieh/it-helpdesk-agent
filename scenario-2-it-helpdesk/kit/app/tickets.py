"""Ticket drafts: the first half of the approval gate.

The LLM never files a ticket. The agent builds a TicketDraft and calls
propose_ticket(), which checks it in code and stores it as the session's
pending_action. Only POST /api/approve sends exactly that payload to
create_ticket over MCP (see app/main.py).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Literal

from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field

from app.mcp_client import HelpdeskMCP
from app.schemas import PendingAction, ToolCall
from app.ticket_store import DEFAULT_TICKETS_DB

KIT = Path(__file__).resolve().parent.parent
DATASET = KIT / "dataset"

# The schema the helpdesk API enforces, and the manifest's document ids and routing table.
_SCHEMA = json.loads((DATASET / "schemas" / "ticket.schema.json").read_text(encoding="utf-8"))
_VALIDATOR = Draft202012Validator(_SCHEMA)
_MANIFEST = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
DOC_IDS: set[str] = {d["id"] for d in _MANIFEST["documents"]}
ROUTING: dict[str, str] = _MANIFEST["routing"]          # category -> team (KB-015 section 3)

TICKET_ID = re.compile(r"IT-\d+")

# The IT number belongs to the helpdesk and may repeat: the simulated API starts again after a restart. The Application
# Reference comes from the action_id, so it never changes. It is the last line of the filed description, which is how
# a helpdesk record is matched exactly to a saved ticket (app/agent/my_tickets.py).
REFERENCE_LABEL = "Application Reference: "
SHORTENED_NOTE = " [description shortened to fit the ticket limit]"
DESCRIPTION_MAX = _SCHEMA["properties"]["description"]["maxLength"]


def reference(action_id: str | None) -> str:
    """The permanent Application Reference of a saved ticket. It is not an IT number. The dot in a full UUID keeps it
    below the 32 characters the secret filter treats as a token (app/guardrails.py), so it survives chat and replies."""
    value = str(action_id or "").removeprefix("act-").upper()
    return f"REF-{value[:16]}.{value[16:]}" if len(value) == 32 else f"REF-{value}"


def with_reference(description: str, action_id: str) -> str:
    """The description unchanged, plus the reference as its own last line. If they do not fit the helpdesk's limit, the
    description is cut and says so, so nothing is lost silently."""
    line = f"\n\n{REFERENCE_LABEL}{reference(action_id)}"
    if len(description) + len(line) > DESCRIPTION_MAX:
        description = description[:DESCRIPTION_MAX - len(line) - len(SHORTENED_NOTE)].rstrip() + SHORTENED_NOTE
    return description + line


def _action_id_taken(action_id: str) -> bool:
    """True if a saved ticket already has this action_id. An unreadable store is not a reason to refuse a draft: its
    primary key still keeps every action_id once."""
    path = Path(os.getenv("TICKETS_DB") or DEFAULT_TICKETS_DB)
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5)
        try:
            return conn.execute("SELECT 1 FROM tickets WHERE action_id = ?", (action_id,)).fetchone() is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def new_action_id() -> str:
    """A new action_id from a full UUID, checked against the saved tickets. Existing action_ids are never changed."""
    action_id = f"act-{uuid.uuid4().hex}"
    while _action_id_taken(action_id):
        action_id = f"act-{uuid.uuid4().hex}"
    return action_id

Category = Literal["network_access", "identity_access", "hardware", "software",
                   "email_collab", "security", "onboarding", "other"]
Priority = Literal["P1", "P2", "P3", "P4"]


class TicketDraft(BaseModel):
    """What the agent proposes. Mirrors dataset/schemas/ticket.schema.json."""
    subject: str = Field(min_length=5, max_length=120)
    description: str = Field(min_length=10, max_length=4000)
    category: Category
    priority: Priority
    requester_email: str
    department: str | None = None     # ignored: always taken from the directory
    attempted_steps: list[str] = Field(min_length=1, max_length=20)
    kb_refs: list[str] = Field(default_factory=list, max_length=10)
    affected_users: int = Field(default=1, ge=1, le=1000)


class TicketDraftError(ValueError):
    """The draft cannot be shown to the employee. `problems` says why."""

    def __init__(self, problems: list[str], tool_calls: list[ToolCall]):
        super().__init__("; ".join(problems))
        self.problems = problems
        self.tool_calls = tool_calls


def apply_priority_rules(priority: str, department: str, affected_users: int) -> str:
    """KB-015 section 2 and KB-016: rules can raise a priority, never lower it."""
    if department == "Executive Office":        # VIP: every ticket is P1
        return "P1"
    if affected_users > 3:                      # more than 3 users (KB-015 v3.0, beats KB-017)
        return "P1"
    return priority


# Executive Office tickets: the requester's email was given in chat and is not verified (no login). The flag is
# a review marker for the Service Desk, not a hold; app/agent/escalation.py adds the full notice to the description.
IDENTITY_SUMMARY_FLAG = "identity not verified"


def summarize(payload: dict) -> str:
    """The one line the employee approves: it must say exactly what will be filed."""
    team = ROUTING.get(payload["category"], "IT Service Desk")
    flag = f" · {IDENTITY_SUMMARY_FLAG}" if payload["department"] == "Executive Office" else ""
    return (f"Ticket draft · {payload['category']} · {payload['priority']} · "
            f"{payload['department']} → {team}{flag}: {payload['subject']}")


async def propose_ticket(session: dict, draft: TicketDraft,
                         mcp: HelpdeskMCP) -> tuple[PendingAction, list[ToolCall]]:
    """Check a draft and store it as the session's pending_action. Nothing is written here.

    Raises TicketDraftError if the draft is not valid; the session is not changed then.
    """
    email = draft.requester_email.strip().lower()
    found, call = await mcp.call("lookup_employee", {"email": email})
    if not found.get("ok"):
        code = (found.get("error") or {}).get("code", "unknown")
        raise TicketDraftError([f"requester {email} could not be looked up ({code})"], [call])

    employee = found["employee"]
    payload = draft.model_dump()
    payload["requester_email"] = email
    payload["department"] = employee["department"]              # the directory is the source of truth
    payload["priority"] = apply_priority_rules(draft.priority, employee["department"], draft.affected_users)
    payload["kb_refs"] = list(dict.fromkeys(draft.kb_refs))     # drop duplicates, keep order

    action_id = new_action_id()
    payload["description"] = with_reference(payload["description"], action_id)

    problems = [f"kb_refs: unknown document {ref}" for ref in payload["kb_refs"] if ref not in DOC_IDS]
    problems += [f"{'/'.join(map(str, e.path)) or 'draft'}: {e.message}" for e in _VALIDATOR.iter_errors(payload)]
    if problems:
        raise TicketDraftError(problems, [call])

    action = PendingAction(action_id=action_id, type="create_ticket",
                           summary=summarize(payload), payload=payload)
    session["pending_action"] = action                          # replaces any older draft
    return action, [call]
