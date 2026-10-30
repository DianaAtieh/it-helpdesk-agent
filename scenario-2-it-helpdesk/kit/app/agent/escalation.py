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
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

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
# The answer node's fixed reply to a Finance employee's VPN problem (KB-002 "Exceptions", KB-016): no client
# troubleshooting, an offer to prepare the security ticket. A ticket request that answers it ("please open a ticket")
# is a Finance VPN request even without VPN words (answers_finance_vpn_reply).
FINANCE_VPN_REPLY = ("For Finance laptops, VPN and remote-access problems go straight to Security Operations. Your "
                     "NakhlaConnect uses the FIN-SECURE profile on GW-3, which only Security Operations may change, so "
                     "please don't update, reinstall or reconfigure the client yourself. I can prepare a security "
                     "ticket for Security Operations: just say \"please open a ticket\". Nothing is filed until you "
                     "approve the draft.")

# ---------------------------------------------------------------- is this a ticket request?

# An explicit request: "open a ticket", "please file a request", "escalate this", "if it needs a ticket, prepare
# it". Detected in the employee's latest message only; the model never decides to escalate on its own.
_TICKET_REQUEST = re.compile(
    r"(?i)\b(?:open|create|file|raise|log|submit|make|start|prepare|draft|put\s+in|get)\s+"
    r"(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:(?:new|support|it|helpdesk|service\s+desk)\s+){0,3}"
    r"(?:ticket|request)\b"
    r"|\bescalat(?:e|ed|ing|ion)\b"
    r"|\bneeds?\s+a\s+ticket\b"
    r"|\bprepare\s+it\b")
# "Don't open a ticket", "I do not want to escalate", "no need to file a ticket": the negation must sit right
# before the request verb, so "it's not working, please open a ticket" is still a request.
_NEGATED_REQUEST = re.compile(
    r"(?i)(?:\bdon't|\bdont|\bdo\s+not|\bno\s+need\s+to|\bnever)\s+(?:want\s+to\s+|need\s+to\s+|have\s+to\s+)?"
    r"(?:open|create|file|raise|log|submit|make|start|prepare|draft|escalat\w*)\b")
# A question ABOUT the process ("How do I create a ticket?", "where can I open a ticket") is not a request to
# make one. Only the words before the request verb, in the same sentence, count: "How do I fix this? Please open
# a ticket" is still a request, and "Can you create a ticket for me?" is an action request.
_INFORMATIONAL = re.compile(
    r"(?i)\b(?:how\s+(?:do|does|can|could|should|would|to)|where\s+(?:do|does|can|could|should)"
    r"|what(?:'s|\s+is)\s+the\s+(?:process|procedure|way))\b[^.?!]*$")
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


def explicit_ticket_request(latest: str) -> bool:
    latest = latest or ""
    return (any(not _INFORMATIONAL.search(latest[:m.start()]) for m in _TICKET_REQUEST.finditer(latest))
            and not _NEGATED_REQUEST.search(latest))


def is_ticket_request(latest: str, previous_assistant: str | None = None) -> bool:
    """True only for an explicit ticket request, or a clear yes to the agent's own offer to escalate. A bare yes to
    the resolution question ("Did this resolve your issue, or ...?") answers its first half: resolved, no ticket.
    An offer the answer itself made ("I can escalate it to IT as a ticket") still turns a yes into a request."""
    if explicit_ticket_request(latest):
        return True
    return bool(previous_assistant and _ESCALATION_OFFER.search(answer_body(previous_assistant))
                and _CLEAR_YES.match(latest or ""))


# ---------------------------------------------------------------- after an answer: resolved, or a ticket?

# Added by respond under every verified answer; the employee's reply to it decides what happens next.
RESOLUTION_QUESTION = "Did this resolve your issue, or would you like me to open a support ticket?"
RESOLVED_REPLY = ("I'm glad that's resolved. I haven't opened a ticket. If the problem comes back, just tell me, "
                  "or say \"open a ticket\".")


def answer_body(previous_assistant: str) -> str:
    """The previous reply without the resolution question respond appended to it."""
    return (previous_assistant or "").replace(RESOLUTION_QUESTION, "")


_RESOLVED = re.compile(r"(?i)\b(?:resolved|fixed|solved|sorted|works?|worked|working)\b")
_NOT_RESOLVED = re.compile(r"(?i)\b(?:not|never|still|again)\b|n't\b|^\s*no\b(?!\s+need\b)")


def reports_resolved(messages: list) -> bool:
    """True if the employee answers the resolution question by saying the issue is resolved (or a bare yes).
    An explicit ticket request, or any negation ("still not working"), is never "resolved"."""
    latest, previous = last_messages(messages)
    if not previous or RESOLUTION_QUESTION not in previous or explicit_ticket_request(latest):
        return False
    if _CLEAR_YES.match(latest or ""):          # a yes to the answer's own escalation offer is a ticket request
        return not _ESCALATION_OFFER.search(answer_body(previous))
    return bool(_RESOLVED.search(latest or "")) and not _NOT_RESOLVED.search(latest or "")


def answers_finance_vpn_reply(messages: list) -> bool:
    """True if the agent's previous reply was its fixed Finance VPN reply: the employee's answer to it is about
    that VPN problem, even when it only says "please open a ticket"."""
    _, previous_assistant = last_messages(messages)
    return bool(previous_assistant) and previous_assistant.strip() == FINANCE_VPN_REPLY


def continues_ticket_request(previous_assistant: str | None) -> bool:
    """True if the agent's previous reply was its own question for the attempted steps of a ticket: the answer
    to it ("I restarted and synced") carries on the ticket request even without the word "ticket"."""
    return bool(previous_assistant) and previous_assistant.strip() == ASK_STEPS_REPLY


def wants_ticket(messages: list, state: dict | None = None) -> bool:
    """The routing decision after retrieval: an explicit request, a clear yes to an offer, or an answer to the
    agent's question for missing ticket fields. Decided in code from the conversation; never by the model."""
    latest, previous = last_messages(messages)
    return (is_ticket_request(latest, previous) or continues_ticket_request(previous)
            or (state is not None and collecting(state)))


# ---------------------------------------------------------------- mandatory ticket fields, collected over turns
#
# TicketIntake is the structured, strict schema of what a ticket needs to know about the PROBLEM. It reuses the ticket
# contract's names (dataset/schemas/ticket.schema.json: subject, description, priority, affected_users,
# attempted_steps) and adds what that contract has no field for (device, operating system, location, error code,
# device name). The helpdesk API accepts no extra properties, so those go into the ticket's description as a
# "Details" block (intake_to_fields). Who the employee is (email, name, employee ID, department) is never part of
# it: build_draft takes that from the directory record.
#
# The collected values live in state["ticket_fields"] = {"values": {...}, "missing": [...], "asked": <question>,
# "text": <the employee's words for this ticket>}, saved by the checkpointer, so a later answer adds to them.
# Whether a field is missing is decided HERE, by Pydantic (TicketIntake.model_validate), never by the model.

DEVICE_TYPES = ("laptop", "desktop", "phone", "tablet", "printer", "monitor", "dock", "other")
URGENCIES = MODEL_PRIORITIES                  # P2 | P3 | P4 from the employee's answer; P1 only ever from a rule


class TicketIntake(BaseModel):
    """What the ticket must know about the problem. Required fields are mandatory; None means not given yet."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    subject: str = Field(min_length=5, max_length=SUBJECT_MAX)              # the issue title
    description: str = Field(min_length=10, max_length=DESCRIPTION_MAX)     # the problem, in the employee's words
    device_type: Literal[DEVICE_TYPES]
    operating_system: str = Field(min_length=2, max_length=60)
    location: str = Field(min_length=2, max_length=60)                      # an office, or remote
    affected_users: int = Field(default=1, ge=1, le=1000)   # optional: only when the employee says so (never asked)
    priority: Literal[URGENCIES]                                            # the urgency
    attempted_steps: list[Annotated[str, Field(min_length=3, max_length=STEP_MAX)]] = Field(
        min_length=1, max_length=STEPS_MAX)
    error_code: str | None = Field(default=None, max_length=20)             # optional
    device_name: str | None = Field(default=None, max_length=80)            # optional (hostname)

    @field_validator("affected_users", mode="before")
    @classmethod
    def _no_bool(cls, value):
        if isinstance(value, bool):
            raise ValueError("a number of people")
        return value


MANDATORY_FIELDS = tuple(name for name, f in TicketIntake.model_fields.items() if f.is_required())
_ADAPTERS = {name: TypeAdapter(Annotated[f.annotation, *f.metadata] if f.metadata else f.annotation)
             for name, f in TicketIntake.model_fields.items()}
# What to ask for each field the employee must give. The subject is written from the description, never asked.
ASK_FOR = {
    "description": "what exactly is happening (the problem and any error message you see)",
    "device_type": "which device it is on (laptop, desktop, phone, tablet, printer, ...)",
    "operating_system": "which operating system the device runs (Windows, macOS, Linux, ...)",
    "location": "where you are working from (which office, or remote)",
    "priority": ("how urgent it is: completely blocked with no workaround, a workaround exists, or a request or "
                 "question"),
    "attempted_steps": "what you have already tried to fix it",
}
EMPLOYEE_FIELDS = tuple(ASK_FOR)
ASK_DESCRIPTION_REPLY = ("Before I prepare the ticket: what exactly is happening? Please describe the problem and "
                         "any error message you see.")
_FIELD_WORDS = {
    "description": re.compile(r"(?i)\b(?:happen\w*|describe|problem|issue|error|symptoms?|going\s+on|wrong)\b"),
    "device_type": re.compile(r"(?i)\b(?:device|laptop|desktop|computer|phone|tablet|printer)\b"),
    "operating_system": re.compile(r"(?i)\b(?:operating\s+system|os|windows|macos|linux)\b"),
    "location": re.compile(r"(?i)\b(?:where|location|office|remote|home)\b"),
    "affected_users": re.compile(r"(?i)\b(?:how\s+many|people|colleagues|anyone\s+else|others|affected)\b"),
    "priority": re.compile(r"(?i)\b(?:urgent|urgency|blocked|workaround|priority)\b"),
    "attempted_steps": re.compile(r"(?i)\b(?:tried|try|steps?|attempted|done\s+so\s+far)\b"),
}
# Words that carry no description of a problem: a bare "please open a ticket" describes nothing.
_FILLER = frozenset("""
    a an the i me my we you your it this that is are am be can could would will please pls hi hello hey thanks
    thank ok okay yes yeah sure need want help for to of and with just now so do something issue problem ticket
    support it's i'm can't
""".split())


def ask_reply(missing: list[str]) -> str:
    """The fixed question for exactly the missing fields."""
    if missing == ["attempted_steps"]:
        return ASK_STEPS_REPLY
    if missing == ["description"]:
        return ASK_DESCRIPTION_REPLY
    asks = [ASK_FOR[f] for f in missing if f in ASK_FOR]
    if len(asks) == 1:
        return f"Before I prepare the ticket, please tell me {asks[0]}."
    return "Before I prepare the ticket, please tell me:\n" + "\n".join(f"- {a}" for a in asks)


# Kept for callers and tests that ask for the two original fields together.
ASK_BOTH_REPLY = ask_reply(["description", "attempted_steps"])


def collecting(state: dict) -> bool:
    """True if the agent's previous reply was its own question for missing ticket fields: this message answers it."""
    fields = state.get("ticket_fields")
    if not isinstance(fields, dict) or not fields.get("missing") or not fields.get("asked"):
        return False
    _, previous = last_messages(state.get("messages") or [])
    return bool(previous) and previous.strip() == str(fields["asked"]).strip()


def collection_text(state: dict) -> str:
    """The employee's words about the problem this ticket is for. Continuing a collection: everything they said for
    it so far plus this answer. Otherwise the request plus the employee's previous message, because the problem is
    usually described one turn before the request ("NX-112 ..." -> answer -> "open a ticket")."""
    messages = state.get("messages") or []
    latest, previous = last_messages(messages)
    if collecting(state):
        return f"{state['ticket_fields'].get('text') or ''}\n{latest}".strip()
    if previous:
        employee = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
        if len(employee) >= 2:
            return f"{employee[-2]}\n{latest}".strip()
    return latest


_ANY_CODE = re.compile(r"(?i)(?<![a-z0-9])[a-z]{2,5}[-\s]?\d{2,5}(?![a-z0-9])")


def describes_problem(text: str) -> bool:
    """An error code ("nx-781" is a description on its own), or at least two meaningful words once the ticket
    request itself is removed (never a bare "open a ticket")."""
    rest = _TICKET_REQUEST.sub(" ", text or "")
    if _ANY_CODE.search(rest):
        return True
    words = re.findall(r"[a-z0-9][a-z0-9'-]*", rest.lower())
    return len([w for w in words if w not in _FILLER]) >= 2


def _valid(name: str, value):
    """The value if it is valid for that one field, else None (so a bad value is never stored)."""
    if value in (None, "", []) or isinstance(value, bool):
        return None
    try:
        return _ADAPTERS[name].validate_python(value)
    except ValidationError:
        return None


def merge_fields(previous: dict | None, proposed) -> dict:
    """The collected values after this turn: earlier values are kept, valid new ones are added, steps accumulate.
    Nothing is invented here; the model saw the earlier values, so it may extend them (never blank them)."""
    values = dict(previous or {})
    proposed = dict(proposed) if isinstance(proposed, dict) else {}
    if proposed.get("priority") == "P1":            # P1 only ever comes from a rule: a model's P1 is capped at P2
        proposed["priority"] = "P2"
    for name in TicketIntake.model_fields:
        if name == "attempted_steps":
            continue
        value = _valid(name, proposed.get(name))
        if value is not None:
            values[name] = value
    values["attempted_steps"] = list(dict.fromkeys([*_steps(values.get("attempted_steps")),
                                                    *_steps(proposed.get("attempted_steps"))]))[:STEPS_MAX]
    for key in ("category", "kb_refs"):                # code still checks both in build_draft
        if proposed.get(key) not in (None, "", []):
            values[key] = proposed[key]
    return values


def is_finance_vpn(state: dict, text: str) -> bool:
    employee = state.get("employee") if isinstance(state.get("employee"), dict) else {}
    return employee.get("department") == FINANCE and (is_vpn_topic(text, list(state.get("codes") or []))
                                                      or answers_finance_vpn_reply(state.get("messages") or []))


def prefill(values: dict, state: dict, request_text: str) -> dict:
    """What code already knows, so the employee is never asked for it: the location of the directory record (the
    employee may still give another), the error code from the message, the Finance VPN step, and a subject written
    from the description. The operating system is always asked: the device with the problem may not be the one the
    directory lists."""
    values = dict(values)
    employee = state.get("employee") if isinstance(state.get("employee"), dict) else {}
    if not values.get("location") and _valid("location", employee.get("location")):
        values["location"] = _valid("location", employee.get("location"))
    codes = list(state.get("codes") or [])
    if not values.get("error_code") and codes:
        values["error_code"] = codes[0]
    if not values.get("attempted_steps") and is_finance_vpn(state, request_text):
        values["attempted_steps"] = [FINANCE_VPN_STEP]
    if not values.get("subject") and values.get("description"):
        values["subject"] = _text(f"{values['description']}", SUBJECT_MAX)
    return values


def missing_fields(values: dict, state: dict, text: str, request_text: str | None = None) -> list[str]:
    """The mandatory fields still missing or invalid, by Pydantic validation of the whole intake. A description
    counts only when the employee's own words (text) describe a problem."""
    missing: set[str] = set()
    try:
        TicketIntake.model_validate(values)
    except ValidationError as exc:
        missing = {str(error["loc"][0]) for error in exc.errors() if error.get("loc")}
    if not describes_problem(text):
        missing.add("description")
    if "subject" in missing and "description" not in missing:
        missing.discard("subject")                     # written from the description by prefill, never asked
    return [name for name in MANDATORY_FIELDS if name in missing]


def validated_intake(values: dict, state: dict, text: str, request_text: str | None = None) -> TicketIntake | None:
    """The complete intake, or None while anything mandatory is missing. The only way to a draft."""
    if missing_fields(values, state, text, request_text):
        return None
    return TicketIntake.model_validate(values)


def missing_question(proposed: str | None, missing: list[str]) -> str:
    """The model's question if it asks for exactly the missing fields (and nothing already given), else a fixed one."""
    question = " ".join(proposed.split())[:400] if isinstance(proposed, str) else ""
    if (question.endswith("?") and not re.search(r"https?://|\d", question)
            and all(_FIELD_WORDS[f].search(question) for f in missing if f in _FIELD_WORDS)
            and not any(_FIELD_WORDS[f].search(question) for f in EMPLOYEE_FIELDS if f not in missing)
            and not _FIELD_WORDS["affected_users"].search(question)):        # never asked
        return question
    return ask_reply(missing)


def intake_to_fields(intake: TicketIntake, proposed: dict) -> dict:
    """The ticket contract's fields from a complete intake: the details the contract has no field for go into the
    description, in a fixed "Details" block. category and kb_refs are the model's suggestions (build_draft checks)."""
    device = intake.device_type + (f" ({intake.device_name})" if intake.device_name else "")
    details = [f"- Device: {device}", f"- Operating system: {intake.operating_system}",
               f"- Location: {intake.location}", f"- Urgency given: {intake.priority}"]
    if intake.error_code:
        details.insert(0, f"- Error code: {intake.error_code}")
    proposed = proposed if isinstance(proposed, dict) else {}
    return {"subject": intake.subject, "description": f"{intake.description}\n\nDetails:\n" + "\n".join(details),
            "category": proposed.get("category"), "priority": intake.priority,
            "attempted_steps": list(intake.attempted_steps), "affected_users": intake.affected_users,
            "kb_refs": proposed.get("kb_refs") or []}


# ---------------------------------------------------------------- the reply when a draft is prepared

DRAFT_READY_REPLY = ("I've prepared a ticket draft for you to check. Nothing is filed until you approve it, "
                     "and you can reject it.")
IDENTITY_REVIEW_SENTENCE = ("Because this is an Executive Office request, the ticket will be marked for the "
                            "Service Desk to verify your identity before acting on it.")
ESCALATION_FAILED_REPLY = ("Sorry, I couldn't prepare the ticket draft right now. Nothing was filed. "
                           "Please try again in a moment.")


APPROVAL_REQUEST = "Do you approve this ticket? Click Approve to file it, or Reject to cancel."


def draft_summary(draft: dict, employee: dict | None = None, intake=None) -> str:
    """The draft's fields as the employee will see them in the approval card (code-built, from the validated draft
    and intake). The employee lines come from the directory record, never from the model."""
    description = str(draft.get("description") or "")
    if description.startswith(IDENTITY_NOTICE):
        description = description[len(IDENTITY_NOTICE):].strip()
    if intake is not None:
        description = intake.description
    employee = employee if isinstance(employee, dict) else {}
    who = " · ".join(str(v) for v in (employee.get("name"), employee.get("employee_id"), draft.get("requester_email"))
                     if v)
    lines = [f"- Employee: {who}"] if who else []
    lines += [f"- Subject: {draft.get('subject', '')}",
              f"- Description: {' '.join(description.split())[:300]}"]
    if intake is not None:
        device = intake.device_type + (f" ({intake.device_name})" if intake.device_name else "")
        lines += [f"- Error code: {intake.error_code or 'none given'}", f"- Device: {device}",
                  f"- Operating system: {intake.operating_system}", f"- Location: {intake.location}",
                  f"- People affected: {intake.affected_users}"]
    lines += [f"- Category: {draft.get('category', '')}",
              f"- Priority: {draft.get('priority', '')}",
              f"- Department: {draft.get('department', '')}",
              f"- Steps already tried: {'; '.join(draft.get('attempted_steps') or [])}"]
    if draft.get("kb_refs"):
        lines.append(f"- Related articles: {', '.join(draft['kb_refs'])}")
    return "Ticket summary:\n" + "\n".join(lines)


def draft_reply(draft: dict, employee: dict | None, intake=None) -> str:
    """The fixed reply for a prepared draft (never the model's text): the summary and the request for explicit
    approval. It claims nothing about identity."""
    head = DRAFT_READY_REPLY
    if requires_identity_review(employee):
        head = f"{DRAFT_READY_REPLY} {IDENTITY_REVIEW_SENTENCE}"
    return f"{head}\n\n{draft_summary(draft, employee, intake)}\n\n{APPROVAL_REQUEST}"


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

    finance_vpn = department == FINANCE and (is_vpn_topic(text, codes)
                                             or answers_finance_vpn_reply(state.get("messages") or []))

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


# ---------------------------------------------------------------- the requester's open tickets (memory)
#
# search_tickets gives the employee's ticket history from the helpdesk API, the system of record, so it is the
# same in every session and after a restart. It is used ONLY here, in code, to avoid a second ticket for a problem
# that already has an open one. Privacy: the email given in chat is not proof of identity, so nothing from the
# history is ever shown to the employee or given to the model: no ticket ID, subject, description, note or
# resolution. Only two derived signals are kept per OPEN ticket: its error codes and normalised subject keywords.
# Ticket text is untrusted (a past ticket may carry planted instructions); keywords cannot carry an instruction.

DUPLICATE_REPLY = ("It looks like a request about this problem is already open with the IT Service Desk, so I "
                   "haven't prepared another ticket. If something has changed, tell me what's new, or contact the "
                   "Service Desk directly.")
MAX_OPEN_TICKETS = 10
_HISTORY_CODE = re.compile(r"\b[A-Z]{2,5}-\d{2,5}\b")
_HISTORY_WORD = re.compile(r"[a-z0-9][a-z0-9-]*")
_HISTORY_STOPWORDS = frozenset("""
    a an and the of on in at to for from with without by my our your his her their me we i you it its is are was
    were be been am not no nor or but so cannot can can't does doesn't do don't did didn't this that these those
    after before since when while still any all every new old very please help issue issues problem problems error
    errors broken working work works fails failed failing failure ticket tickets open opened request show shows
    nakhla today yesterday morning again also just get got some anything nothing
""".split())


def _history_keywords(text: str) -> set[str]:
    """Lower-case words without stopwords, codes or ticket IDs; a trailing plural "s" removed (monitors -> monitor)."""
    words = set()
    for word in _HISTORY_WORD.findall(_HISTORY_CODE.sub(" ", text or "").lower()):
        if len(word) < 2 or word in _HISTORY_STOPWORDS or re.fullmatch(r"it-\d+", word):
            continue
        words.add(word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word)
    return words


def history_signals(tickets) -> list[dict]:
    """The minimum kept from search_tickets: per OPEN ticket, {"codes": [...], "keywords": [...]} from its subject.
    Nothing else (no ID, subject text, description, notes, resolution or requester) leaves this function."""
    out = []
    for ticket in tickets if isinstance(tickets, list) else []:
        if not isinstance(ticket, dict) or ticket.get("status") != "open":
            continue
        subject = str(ticket.get("subject") or "")[:200]
        codes = sorted({c for c in _HISTORY_CODE.findall(subject) if not c.startswith("IT-")})
        out.append({"codes": codes, "keywords": sorted(_history_keywords(subject))})
        if len(out) == MAX_OPEN_TICKETS:
            break
    return out


def duplicate_text(messages: list) -> str:
    """What the duplicate check compares with the open tickets: the employee's latest message, plus the previous
    employee message ONLY when the latest one continues the ticket request (a clear yes to the agent's offer to
    escalate, or the answer to its question for the attempted steps), because then the problem was described one
    message earlier. A new request is judged on its own words, so an earlier, unrelated topic in the same
    conversation never turns it into a duplicate."""
    latest, previous_assistant = last_messages(messages)
    continues = bool(previous_assistant) and (
        continues_ticket_request(previous_assistant)
        or bool(_ESCALATION_OFFER.search(previous_assistant) and _CLEAR_YES.match(latest or "")))
    if not continues:
        return latest
    employee = [str(m.content) for m in messages or [] if isinstance(m, HumanMessage)]
    earlier = employee[-2] if len(employee) >= 2 else ""
    return f"{earlier}\n{latest}".strip()


def open_duplicate(signals: list[dict], text: str, codes: list[str] | tuple[str, ...] = ()) -> bool:
    """True if the request matches one of the requester's open tickets: a shared error code, or at least two shared
    meaningful subject keywords (e.g. "dock" and "d3")."""
    asked_codes = {c for c in (*codes, *_HISTORY_CODE.findall(text or "")) if not c.startswith("IT-")}
    asked_words = _history_keywords(text)
    for signal in signals or []:
        if set(signal.get("codes") or ()) & asked_codes:
            return True
        if len(set(signal.get("keywords") or ()) & asked_words) >= 2:
            return True
    return False
