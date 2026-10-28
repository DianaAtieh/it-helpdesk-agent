"""Ticket escalation policy (Phase 4): deterministic, no model, no network, no ticket writes.

This module decides WHETHER a ticket draft may be prepared and builds its payload. The chat model only
describes the problem (subject, description, a category suggestion, a base priority suggestion, the steps the
employee says they tried, how many people are affected). Everything that matters is set here, in code:

  * the requester is the conversation's locked email; a draft fails closed without one;
  * the department and the VIP and contractor flags come from the directory record, never from chat;
  * category: Finance VPN or remote-access problems go to Security Operations (`security`, KB-016);
  * priority: the model may only suggest P2-P4; code raises it (Executive Office or more than 3 users: P1,
    via app.tickets.apply_priority_rules; Finance VPN: at least P2; an employee who says they are completely
    blocked from working, with no workaround: at least P2, KB-015). Asking for a higher priority never helps;
  * kb_refs: only documents in this turn's evidence;
  * Executive Office tickets carry an identity notice in the description: the email was given in chat and is
    NOT verified. It is a REVIEW FLAG for the Service Desk, not an enforced hold: the ticket is still created
    when the employee approves (KB-015/KB-016: every Executive Office ticket is P1).

The draft built here is not a ticket and not yet a pending action: the caller passes it to main.propose_draft,
which validates it against the ticket schema and stores it for approval. Only POST /api/approve writes a ticket.
Out of scope for now: the security-incident P1 trigger of KB-015 (lost device, phishing) is not applied.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage

from app.tickets import IDENTITY_SUMMARY_FLAG, apply_priority_rules  # noqa: F401  (the flag is re-exported)

CATEGORIES = ("network_access", "identity_access", "hardware", "software", "email_collab", "security",
              "onboarding", "other")
MODEL_PRIORITIES = ("P2", "P3", "P4")          # P1 only ever comes from a rule, never from the model
DEFAULT_PRIORITY = "P3"
EXECUTIVE_OFFICE = "Executive Office"
FINANCE = "Finance"

SUBJECT_MAX, DESCRIPTION_MAX, STEP_MAX, STEPS_MAX, KB_REFS_MAX = 120, 4000, 300, 20, 10

IDENTITY_NOTICE = ("IDENTITY NOT VERIFIED: the requester's email was given in chat and has not been verified. "
                   "Service Desk: verify the requester's identity before treating this request as authorized "
                   "(Executive Office, KB-016).")

# ---------------------------------------------------------------- replies when no draft is made

ASK_EMAIL_REPLY = ("I can prepare a ticket, but I need your work email first so I can look you up in the "
                   "directory. What is your work email?")
NO_EMPLOYEE_REPLY = ("I couldn't find your work email in the company directory, so I can't prepare a ticket. "
                     "Please check the email address, or contact the IT Service Desk directly.")
ASK_STEPS_REPLY = ("Before I prepare the ticket: what have you already tried to fix this? "
                   "The ticket needs at least one step you tried.")
CONTRACTOR_VPN_REPLY = ("Contractors don't use the VPN: remote access is through the Nakhla Workspace VDI portal "
                        "at https://vdi.nakhla.example (KB-016, SP-002), so I can't open a ticket for VPN access. "
                        "If the VDI portal itself doesn't work, tell me what you see and I can prepare a ticket.")
INCIDENT_REPLY = "This problem is already covered by an active incident, so I won't open a separate ticket."
FINANCE_VPN_STEP = ("Reported the problem to the IT assistant; Finance VPN problems go straight to Security "
                    "Operations without client troubleshooting (KB-016).")

# ---------------------------------------------------------------- is this a ticket request?

# An explicit request: "open a ticket", "please file a request", "escalate this", "if it needs a ticket, prepare
# it". Detected in the employee's latest message only; the model never decides to escalate on its own.
_TICKET_REQUEST = re.compile(
    r"(?i)\b(?:open|create|file|raise|log|submit|make|start|prepare|draft|put\s+in|get)\s+"
    r"(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:new\s+)?(?:support\s+|it\s+|helpdesk\s+|service\s+desk\s+)?"
    r"(?:ticket|request)\b"
    r"|\bescalat(?:e|ed|ing|ion)\b"
    r"|\bneeds?\s+a\s+ticket\b"
    r"|\bprepare\s+it\b")
# "Don't open a ticket", "I do not want to escalate", "no need to file a ticket": the negation must sit right
# before the request verb, so "it's not working, please open a ticket" is still a request.
_NEGATED_REQUEST = re.compile(
    r"(?i)(?:\bdon't|\bdont|\bdo\s+not|\bno\s+need\s+to|\bnever)\s+(?:want\s+to\s+|need\s+to\s+|have\s+to\s+)?"
    r"(?:open|create|file|raise|log|submit|make|start|prepare|draft|escalat\w*)\b")
# The agent's previous reply offered escalation ("I can escalate it to IT as a ticket", "Would you like me to
# prepare a ticket?"), and the employee's answer is a clear, short yes.
_ESCALATION_OFFER = re.compile(
    r"(?i)\b(?:i\s+can|i\s+could|shall\s+i|should\s+i|would\s+you\s+like\s+me\s+to|do\s+you\s+want\s+me\s+to)\b"
    r"[^.?!]{0,80}\b(?:escalat\w*|ticket|request)\b")
_CLEAR_YES = re.compile(
    r"(?i)^\s*(?:yes|yeah|yep|sure|ok|okay|please|please\s+do|go\s+ahead|do\s+it|yes\s+please)"
    r"(?:[\s,.!]+(?:please|do\s+it|go\s+ahead|thanks|thank\s+you|thx))*[\s.!]*$")


def last_messages(messages: list) -> tuple[str, str | None]:
    """The employee's latest message and the assistant reply just before it (None if there was none)."""
    latest, previous = "", None
    items = list(messages or [])
    for index in range(len(items) - 1, -1, -1):
        if isinstance(items[index], HumanMessage):
            latest = str(items[index].content)
            earlier = [m for m in items[:index] if isinstance(m, (HumanMessage, AIMessage))]
            if earlier and isinstance(earlier[-1], AIMessage):
                previous = str(earlier[-1].content)
            break
    return latest, previous


def is_ticket_request(latest: str, previous_assistant: str | None = None) -> bool:
    """True only for an explicit ticket request, or a clear yes to the agent's own offer to escalate."""
    latest = latest or ""
    if _TICKET_REQUEST.search(latest) and not _NEGATED_REQUEST.search(latest):
        return True
    return bool(previous_assistant and _ESCALATION_OFFER.search(previous_assistant) and _CLEAR_YES.match(latest))


def continues_ticket_request(previous_assistant: str | None) -> bool:
    """True if the agent's previous reply was its own question for the attempted steps of a ticket: the answer
    to it ("I restarted and synced") carries on the ticket request even without the word "ticket"."""
    return bool(previous_assistant) and previous_assistant.strip() == ASK_STEPS_REPLY


def wants_ticket(messages: list) -> bool:
    """The routing decision after retrieval: an explicit request, a clear yes to an offer, or an answer to the
    agent's question for attempted steps. Decided in code from the conversation; never by the model."""
    latest, previous = last_messages(messages)
    return is_ticket_request(latest, previous) or continues_ticket_request(previous)


# ---------------------------------------------------------------- the reply when a draft is prepared

DRAFT_READY_REPLY = ("I've prepared a ticket draft for you to check. Nothing is filed until you approve it, "
                     "and you can reject it.")
IDENTITY_REVIEW_SENTENCE = ("Because this is an Executive Office request, the ticket will be marked for the "
                            "Service Desk to verify your identity before acting on it.")
ESCALATION_FAILED_REPLY = ("Sorry, I couldn't prepare the ticket draft right now. Nothing was filed. "
                           "Please try again in a moment.")


def draft_reply(draft: dict, employee: dict | None) -> str:
    """The fixed reply for a prepared draft (never the model's text). It claims nothing about identity."""
    if requires_identity_review(employee):
        return f"{DRAFT_READY_REPLY} {IDENTITY_REVIEW_SENTENCE}"
    return DRAFT_READY_REPLY


# ---------------------------------------------------------------- topic helpers

_CURLY_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# NX- codes are the VPN client's (KB-002); NXM- codes are mail, so they do not count.
_VPN_CODE = re.compile(r"^NX-\d+$")
_VPN_WORDS = re.compile(r"(?i)\b(?:vpn|nakhlaconnect|remote\s+access|gateway|gw-?\d|fin-secure)\b")
_VDI_WORDS = re.compile(r"(?i)\b(?:vdi|workspace\s+portal|nakhla\s+workspace)\b")


def is_vpn_topic(text: str, codes: list[str] | tuple[str, ...] = ()) -> bool:
    return any(_VPN_CODE.match(str(code)) for code in codes) or bool(_VPN_WORDS.search(text or ""))


def is_vdi_topic(text: str) -> bool:
    return bool(_VDI_WORDS.search(text or ""))


# KB-015 P2: "one to three users completely blocked from working, no workaround". Recognised only when the
# employee STATES it about their situation, in a sentence of its own, not as a hypothetical ("if it fails again I
# cannot work"), and not when the message mentions a workaround or says they can still work. A request for a
# higher priority ("make it urgent", "P1 please") is not a statement of being blocked and changes nothing.
_BLOCKED = re.compile(
    r"(?i)\b(?:can't|cannot|can\s+not|unable\s+to|not\s+able\s+to)\s+(?:do\s+(?:my|any)\s+)?work(?:ing)?\b"
    r"(?!\s+out\b)"
    r"|\b(?:can't|cannot|unable\s+to)\s+do\s+my\s+job\b"
    r"|\b(?:completely|totally|fully)\s+blocked\b"
    r"|\bblocked\s+from\s+working\b")
_HYPOTHETICAL = re.compile(r"(?i)\b(?:if|unless|otherwise|would|could|might|may|when|whenever)\b")
_WORKAROUND = re.compile(
    r"(?i)\bwork\s*-?\s*around\b|\bi\s+can\s+still\b|\bstill\s+able\s+to\b|\bfor\s+now\s+i\b"
    r"|\bin\s+the\s+meantime\s+i\b|\binstead\b|\bnot\s+(?:completely|totally|fully)\s+blocked\b")


def reports_blocked(text: str) -> bool:
    """True if the employee says they are completely blocked from working, with no workaround (KB-015 P2)."""
    text = (text or "").translate(_CURLY_APOSTROPHES)
    if _WORKAROUND.search(text):
        return False
    return any(_BLOCKED.search(sentence) and not _HYPOTHETICAL.search(sentence)
               for sentence in _SENTENCE_SPLIT.split(text) if sentence.strip())


# ---------------------------------------------------------------- may a draft be made at all?

def ticket_blocker(state: dict, text: str) -> tuple[str, str] | None:
    """Why no draft may be made, as (reason, reply), or None if a draft may be prepared. Fails closed.

    The identity rules: a locked conversation email is required, and the directory record must belong to it.
    """
    if state.get("route") == "incident":
        return "incident", INCIDENT_REPLY
    email = (state.get("user_email") or "").strip().lower()
    if not email:
        return "no_identity", ASK_EMAIL_REPLY
    employee = state.get("employee")
    if not isinstance(employee, dict) or not employee.get("department"):
        return "no_employee", NO_EMPLOYEE_REPLY
    if str(employee.get("email", "")).strip().lower() != email:
        return "employee_mismatch", NO_EMPLOYEE_REPLY
    if employee.get("contractor") is True and is_vpn_topic(text, state.get("codes") or []) and not is_vdi_topic(text):
        return "contractor_vpn", CONTRACTOR_VPN_REPLY
    return None


# ---------------------------------------------------------------- building the draft

@dataclass(frozen=True)
class EscalationOutcome:
    kind: str                     # draft | ask_steps | blocked
    draft: dict | None = None     # TicketDraft-compatible fields (kind == "draft")
    reply: str | None = None      # what to tell the employee when no draft is made
    reason: str | None = None     # blocked: incident | no_identity | no_employee | employee_mismatch | contractor_vpn
    notes: list[str] = field(default_factory=list)   # what code changed in the model's suggestion (for tests/logs)


def _text(value, limit: int) -> str:
    return " ".join(value.split())[:limit].strip() if isinstance(value, str) else ""


def _steps(value) -> list[str]:
    if not isinstance(value, list):
        return []
    steps = [_text(step, STEP_MAX) for step in value]
    return list(dict.fromkeys(step for step in steps if len(step) >= 3))[:STEPS_MAX]


def _affected_users(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 1
    return min(max(value, 1), 1000)


def requires_identity_review(employee: dict | None) -> bool:
    """Executive Office (VIP) requesters, as the directory says: their tickets carry the identity notice."""
    return isinstance(employee, dict) and (employee.get("department") == EXECUTIVE_OFFICE
                                           or employee.get("vip") is True)


def build_draft(fields: dict, state: dict, text: str) -> EscalationOutcome:
    """Turn the model's description of the problem into a ticket draft, applying every rule in code.

    fields: the model's JSON (subject, description, category, priority, attempted_steps, affected_users,
            kb_refs). Anything missing or invalid is replaced or dropped, never trusted.
    state:  the graph state (user_email, employee, codes, evidence, route).
    text:   the employee's latest message.
    """
    blocked = ticket_blocker(state, text)
    if blocked:
        return EscalationOutcome("blocked", reply=blocked[1], reason=blocked[0])

    fields = fields if isinstance(fields, dict) else {}
    employee = state["employee"]
    department = employee["department"]
    codes = list(state.get("codes") or [])
    notes: list[str] = []

    finance_vpn = department == FINANCE and is_vpn_topic(text, codes)

    steps = _steps(fields.get("attempted_steps"))
    if not steps:
        if not finance_vpn:
            return EscalationOutcome("ask_steps", reply=ASK_STEPS_REPLY)
        steps = [FINANCE_VPN_STEP]
        notes.append("finance_vpn_step")

    category = fields.get("category") if fields.get("category") in CATEGORIES else "other"
    if finance_vpn and category != "security":
        category = "security"
        notes.append("finance_vpn_security")

    priority = fields.get("priority")
    if priority == "P1":
        notes.append("model_p1_capped")
        priority = "P2"
    elif priority not in MODEL_PRIORITIES:
        priority = DEFAULT_PRIORITY
    if priority in ("P3", "P4") and reports_blocked(text):     # KB-015 P2; raises to P2 at most, never P1
        priority = "P2"
        notes.append("blocked_p2")
    if finance_vpn and priority in ("P3", "P4"):
        priority = "P2"
        notes.append("finance_vpn_p2")
    affected = _affected_users(fields.get("affected_users"))
    raised = apply_priority_rules(priority, department, affected)
    if raised != priority:
        notes.append("rule_p1")
    priority = raised

    evidence_ids = {record.get("doc_id") for record in state.get("evidence") or [] if isinstance(record, dict)}
    raw_refs = fields.get("kb_refs") if isinstance(fields.get("kb_refs"), list) else []
    kb_refs = list(dict.fromkeys(ref for ref in raw_refs if isinstance(ref, str) and ref in evidence_ids))
    kb_refs = kb_refs[:KB_REFS_MAX]

    subject = _text(fields.get("subject"), SUBJECT_MAX)
    if len(subject) < 5:
        subject = _text(f"IT problem: {', '.join(codes) or category}", SUBJECT_MAX)
    description = (fields.get("description") or "").strip() if isinstance(fields.get("description"), str) else ""
    if len(description) < 10:
        description = text.strip() if len(text.strip()) >= 10 else f"{subject}. Reported through the IT assistant."
    if affected > 1:
        description += f"\n\nAffected users (reported by the requester): {affected}."
    if requires_identity_review(employee):
        description = f"{IDENTITY_NOTICE}\n\n{description}"
        notes.append("identity_notice")
    description = description[:DESCRIPTION_MAX]

    draft = {
        "subject": subject,
        "description": description,
        "category": category,
        "priority": priority,
        "requester_email": state["user_email"].strip().lower(),      # the locked email, never the model's
        "department": department,                                     # the directory's, never the model's
        "attempted_steps": steps,
        "kb_refs": kb_refs,
        "affected_users": affected,
    }
    return EscalationOutcome("draft", draft=draft, notes=notes)
