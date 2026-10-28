"""Graph nodes. Each node takes the state and returns only the fields it changes.

Nodes get their dependencies (the MCP client, the knowledge base, the chat model) from a factory, so tests
can pass fakes. The flow (app/agent/graph.py): triage -> (incident | retrieve -> answer) -> respond.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable, Protocol

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import escalation, policy
from app.agent.prompts import (ESCALATION_SYSTEM_PROMPT, SYSTEM_PROMPT, build_answer_input,
                               build_escalation_input)
from app.agent.retrieval import KnowledgeBase, KnowledgeBaseError, strict_codes
from app.agent.state import AgentState
from app.agent.verifier import verify_answer
from app.guardrails import redact_secrets
from app.llm import ChatModel, LLMError

log = logging.getLogger("helpdesk-app.agent")


class MCPClient(Protocol):
    """What the nodes need from app.mcp_client.HelpdeskMCP: call(tool, args) -> (data, ToolCall)."""

    async def call(self, tool: str, args: dict | None = None) -> tuple[dict, Any]: ...


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


# ---------------------------------------------------------------- triage (Phase 2): no LLM

EMPLOYEE_FIELDS = ("email", "department", "vip", "contractor")
INCIDENT_FIELDS = ("id", "title", "service", "priority", "error_codes", "workaround", "instructions")

# Loose forms ("nx 503", "NX503", "nx-503") are accepted ONLY to match a code an active incident lists, so a
# phrase like "Win 11" can never match anything unless an incident really lists WIN-11.
_LOOSE_CODE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,5})[-\s]?(\d{2,5})(?![A-Za-z0-9])", re.ASCII)


def latest_user_text(state: AgentState) -> str:
    """The newest employee message (already redacted by /api/chat; instruction-like ones are withheld)."""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def loose_codes(text: str) -> set[str]:
    return {f"{letters.upper()}-{digits}" for letters, digits in _LOOSE_CODE.findall(text or "")}


def employee_record(data: dict, email: str) -> dict | None:
    """The directory fields the agent needs, or None. The record must be for the email that was looked up."""
    employee = data.get("employee") if data.get("ok") else None
    if not isinstance(employee, dict) or str(employee.get("email", "")).strip().lower() != email:
        return None
    return {key: employee.get(key) for key in EMPLOYEE_FIELDS}


def active_incidents(data: dict) -> list[dict]:
    """The status page's ACTIVE incidents, trimmed to the fields the agent uses (resolved ones are dropped)."""
    incidents = (data.get("status") or {}).get("incidents") or []
    return [{key: incident.get(key) for key in INCIDENT_FIELDS}
            | {"error_codes": [str(code).upper() for code in incident.get("error_codes") or []]}
            for incident in incidents if isinstance(incident, dict) and incident.get("status") == "active"]


def match_incident(incidents: list[dict], text: str, codes: list[str]) -> dict | None:
    """The first active incident that lists one of the user's error codes (strict or loose form), or None."""
    candidates = set(codes) | loose_codes(text)
    for incident in incidents:
        if candidates & set(incident["error_codes"]):
            return incident
    return None


def make_triage(mcp: MCPClient) -> Callable[[AgentState], Awaitable[dict]]:
    """Who is asking, what the status page says, which error codes were given, and where to go next.

    Deterministic: no LLM. The department comes from the directory (lookup_employee) for the conversation's
    locked email, never from the message. A failed lookup or status call is recorded in tool_calls and never
    replaced by a guess: employee stays None, and status_checked=False means "no incident" is unknown.
    """

    async def triage(state: AgentState) -> dict:
        text = latest_user_text(state)
        email = (state.get("user_email") or "").strip().lower() or None
        calls: list[dict] = []

        # Reuse the directory record only if it was looked up for this very email; otherwise look it up again.
        cached = state.get("employee")
        if email is None:
            employee = None
        elif isinstance(cached, dict) and cached.get("email") == email:
            employee = cached
        else:
            data, call = await mcp.call("lookup_employee", {"email": email})
            calls.append(call.model_dump())
            employee = employee_record(data, email)

        data, call = await mcp.call("get_service_status")
        calls.append(call.model_dump())
        status_checked = bool(data.get("ok"))
        incidents = active_incidents(data) if status_checked else []

        codes = list(strict_codes(text))
        matched = match_incident(incidents, text, codes)
        return {
            "employee": employee,
            "codes": codes,
            "incidents": incidents,
            "status_checked": status_checked,
            "active_incident": matched,
            "route": "incident" if matched else "retrieve",
            "tool_calls": calls,
        }

    return triage


def route_after_triage(state: AgentState) -> str:
    """The conditional edge after triage (wired in a later step): "incident" or "retrieve"."""
    return "incident" if state.get("route") == "incident" else "retrieve"


# ---------------------------------------------------------------- retrieve (Phase 2): no LLM

KB_TOOL = "search_knowledge_base"     # how a knowledge-base search appears in the response's tool_calls


def previous_user_text(state: AgentState) -> str:
    """The employee message before the latest one, or ""."""
    seen = 0
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            seen += 1
            if seen == 2:
                return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def search_query(state: AgentState) -> str:
    """What to search for: the latest message, plus the one before it when the last turn asked a question
    (the employee's answer, e.g. "it shows NX-112", only makes sense together with the original problem)."""
    latest = latest_user_text(state)
    if state.get("outcome") == "clarify":
        earlier = previous_user_text(state)
        if earlier:
            return f"{earlier}\n{latest}"
    return latest


def make_retrieve(kb: KnowledgeBase | None) -> Callable[[AgentState], Awaitable[dict]]:
    """Search the knowledge base, scoped to the department from the directory (triage), never a claimed one.

    kb is None when the knowledge base could not be loaded at startup: then kb_status is "unavailable" and
    the answer node says so honestly instead of answering without evidence.
    """

    async def retrieve(state: AgentState) -> dict:
        employee = state.get("employee")
        department = employee.get("department") if isinstance(employee, dict) else None
        calls = list(state.get("tool_calls") or [])          # triage's calls of this turn come first
        empty = {"evidence": [], "undocumented_codes": [], "sufficiency": "none",
                 "requires_department_verification": False, "rule_warnings": []}
        if kb is None:
            return {**empty, "kb_status": "unavailable", "tool_calls": calls}
        try:
            result = await kb.search(search_query(state), department)
        except KnowledgeBaseError as exc:
            calls.append({"tool": KB_TOOL, "ok": False, "ms": 0})
            return {**empty, "kb_status": exc.code, "tool_calls": calls}
        calls.append({"tool": KB_TOOL, "ok": True, "ms": result.ms})
        found = result.to_state()
        codes = list(dict.fromkeys([*(state.get("codes") or []), *found["codes"]]))
        # Security policies win (manifest rule 1): a non-SP chunk recommending a forbidden action never reaches
        # the model, the escalation's kb_refs or the verifier. SP chunks and prohibitions are kept.
        evidence, _dropped = policy.filter_evidence(found["evidence"])
        return {"evidence": evidence, "undocumented_codes": found["undocumented_codes"],
                "sufficiency": found["sufficiency"],
                "requires_department_verification": found["requires_department_verification"],
                "rule_warnings": found["rule_warnings"], "codes": codes, "kb_status": "ok", "tool_calls": calls}

    return retrieve


# ---------------------------------------------------------------- answer (Phase 2): the one LLM call

MAX_REPLY_CHARS = 2000

KB_UNAVAILABLE_REPLY = ("I couldn't search the IT knowledge base right now, so I can't give you documented steps. "
                        "Please try again in a moment.")
LLM_FAILED_REPLY = "Sorry, I couldn't prepare an answer right now. Please try again in a moment."
UNGROUNDED_REPLY = ("I couldn't find documented steps for this in the IT knowledge base. If you tell me the exact "
                    "error message or code, I can look again, or I can escalate it to IT as a ticket.")


def not_documented_reply(codes: list[str]) -> str:
    listed = ", ".join(codes)
    return (f"Our IT knowledge base has no documented fix for {listed}, so I won't guess at steps. "
            f"I can escalate it to IT as a ticket so a technician can look into it.")


def clarify_reply(state: AgentState) -> str:
    if not isinstance(state.get("employee"), dict):
        return ("I need a bit more detail to help. What exact error message or code do you see, and what is your "
                "work email (so I can check which rules apply to you)?")
    return "I need a bit more detail to help. What exact error message or code do you see, and in which app?"


# A model "clarification" carries no citation, so it is shown only if it is clearly a request for one of a few
# pieces of information (an ALLOWLIST), never advice dressed as a question ("Is it fixed when you use GW-1?",
# "Should you just ignore the version check?"). Every sentence must:
#   * be a question that opens like a request for information (What / Which / Could you tell me ...), and
#   * ask for an allowed topic: error code or message, work email, department, app, operating system, version.
# On top of that, nothing that hints at a step, a setting, a credential or a bypass, and no digits (no echoed
# numbers, passwords or versions). Anything else, or anything unclear, gets the deterministic clarify_reply:
# rejecting a harmless question only costs a more generic one.
MAX_CLARIFICATION_CHARS = 250
_QUESTION_SPLIT = re.compile(r"(?<=[.!?])\s+")
_INFO_REQUEST = re.compile(
    r"(?i)^(?:what|which|(?:could|can|would)\s+you\s+(?:please\s+)?(?:tell|share|confirm|send|give)|"
    r"may\s+i\s+(?:have|ask)|please\s+(?:tell|share|confirm|send))\b")
_ALLOWED_TOPIC = re.compile(
    r"(?i)\b(?:error|code|message|e-?mail|department|team|app|apps|application|program|software|"
    r"operating\s+system|os|windows|macos|mac|linux|version|release)\b")
_STEP_HINT = re.compile(
    r"(?i)\b(?:open|set|chang|edit|disabl|enabl|turn|switch|install|uninstall|updat|upgrad|restart|reboot|reset|"
    r"run|click|tap|go\s+to|select|delet|remov|reconnect|tr(?:y|ied|ies|ying)|bypass|skip|download|configur|"
    r"modif|press|clear|flush|renam|disconnect|uncheck|toggle|ignor|overrid|instead|trick|shortcut|workaround|"
    r"fix|password|passcode|passphrase|pin\b|otp|token|secret|credential|admin|root\b|sudo)"
    r"|[=`\\/<>{}\[\]]|https?:|www\.|\d")


def is_safe_clarification(text: str) -> bool:
    """True only if every sentence is a question asking for an allowed piece of information (see above)."""
    text = (text or "").strip()
    if not text or len(text) > MAX_CLARIFICATION_CHARS or "\n" in text or not text.endswith("?"):
        return False
    if _STEP_HINT.search(text):
        return False
    sentences = [sentence.strip() for sentence in _QUESTION_SPLIT.split(text) if sentence.strip()]
    return all(sentence.endswith("?") and _INFO_REQUEST.match(sentence) and _ALLOWED_TOPIC.search(sentence)
               for sentence in sentences)


def cited(data: dict, evidence: list[dict]) -> list[dict]:
    """The model's citations that are documents in this turn's evidence, as [{"id", "title"}], in order."""
    titles = {}
    for record in evidence:
        titles.setdefault(record["doc_id"], record.get("title") or "")
    raw = data.get("citations")
    ids = [c for c in raw if isinstance(c, str)] if isinstance(raw, list) else []
    return [{"id": doc_id, "title": titles[doc_id]} for doc_id in dict.fromkeys(ids) if doc_id in titles]


def make_answer(llm: ChatModel | None) -> Callable[[AgentState], Awaitable[dict]]:
    """Answer from the evidence only, with citations; or ask one question; or say honestly what is missing.

    The model gets text only (no tools): it can never create or change a ticket. Its citations are kept only
    when they name documents in this turn's evidence; an answer left with no valid citation is not shown.
    Deterministic replies, without a model call, when there is nothing to ground an answer on.
    """

    def result(reply: str, outcome: str, citations: list[dict] | None = None, usage: dict | None = None) -> dict:
        return {"reply": reply, "outcome": outcome, "citations": citations or [], "usage": usage or {}}

    async def answer(state: AgentState) -> dict:
        evidence = list(state.get("evidence") or [])
        codes = list(state.get("codes") or [])
        undocumented = list(state.get("undocumented_codes") or [])

        # A clear request to break a security policy gets the fixed refusal, before anything else, with no model
        # call. Its one citation is the rule's SP document, issued by code (kept apart in policy_citations).
        refused = policy.check_request(latest_user_text(state), state.get("employee"), codes)
        if refused:
            return {**result(refused.reply, "policy_refusal"), "policy_citations": [refused.citation]}
        if state.get("kb_status") != "ok":
            return result(KB_UNAVAILABLE_REPLY, "error")
        # Nobody known and no error code: ask who they are and what they see before any answer (S2-02).
        # With a recognised code, an anonymous employee still gets the normal flow below.
        if not isinstance(state.get("employee"), dict) and not codes:
            return result(clarify_reply(state), "clarify")
        if codes and set(codes) <= set(undocumented):
            return result(not_documented_reply(undocumented), "not_documented")     # nothing documents them
        if not evidence:
            return result(not_documented_reply(undocumented) if undocumented else clarify_reply(state),
                          "not_documented" if undocumented else "clarify")
        if llm is None:
            return result(LLM_FAILED_REPLY, "error")

        try:
            out = await llm.complete_json(SYSTEM_PROMPT, build_answer_input(state, latest_user_text(state)))
        except LLMError:
            return result(LLM_FAILED_REPLY, "error")
        reply = out.data.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            return result(LLM_FAILED_REPLY, "error", usage=out.usage)
        reply = reply.strip()[:MAX_REPLY_CHARS]
        citations = cited(out.data, evidence)
        if out.data.get("needs_clarification") is True:
            # A question needs no citation, so it must really be only a question; otherwise ask our own.
            question = reply if is_safe_clarification(reply) else clarify_reply(state)
            return result(question, "clarify", [], out.usage)
        if not citations:                                   # an answer must rest on the evidence it was given
            return result(UNGROUNDED_REPLY, "not_documented", usage=out.usage)
        return result(reply, "answered", citations, out.usage)

    return answer


# ---------------------------------------------------------------- escalate (Phase 4): a draft proposal only

def route_after_retrieve(state: AgentState) -> str:
    """The conditional edge after retrieve: "escalate" for a ticket request (decided in code from the
    conversation, see escalation.wants_ticket), otherwise "answer"."""
    return "escalate" if escalation.wants_ticket(state.get("messages") or []) else "answer"


def make_escalate(llm: ChatModel | None, mcp: MCPClient | None = None) -> Callable[[AgentState], Awaitable[dict]]:
    """Prepare a ticket draft PROPOSAL in the state. It never files, approves or sends anything.

    Order: the deterministic blockers first (no identity, no directory record, a Contractor VPN request), with
    no model call; then one tool-free model call for the descriptive fields only; then escalation.build_draft,
    which sets the requester, department, category, priority and kb_refs in code. Every reply is a fixed text.
    """

    def result(reply: str, outcome: str, draft: dict | None = None, usage: dict | None = None) -> dict:
        return {"reply": reply, "outcome": outcome, "draft": draft, "citations": [], "usage": usage or {}}

    async def escalate(state: AgentState) -> dict:
        text, _ = escalation.last_messages(state.get("messages") or [])
        blocked = escalation.ticket_blocker(state, text)
        if blocked:
            reason, reply = blocked
            return result(reply, "clarify" if reason == "no_identity" else "blocked")
        # No draft for a ticket that asks for something a security policy forbids (an admin password, an SMS
        # code, a security bypass). PHONE_RESET is the exception: SP-001 section 2 makes a ticket the allowed path
        # for an assisted reset (with manager approval and in-person or video identity verification).
        refused = policy.check_request(text, state.get("employee"), state.get("codes") or [])
        if refused and refused.rule != "PHONE_RESET":
            return {**result(refused.reply, "policy_refusal"), "policy_citations": [refused.citation]}

        # Memory: the requester's OPEN tickets (search_tickets), only for the directory-confirmed email, only to
        # avoid a duplicate. Fail-open on purpose: if the lookup fails, the draft is still proposed, and nothing is
        # ever filed without the employee's approval in /api/approve. Nothing from the history reaches the model or
        # the reply; the log has counts and booleans only.
        calls = list(state.get("tool_calls") or [])
        signals: list[dict] = []
        checked = False
        email = (state.get("employee") or {}).get("email")
        if mcp is not None and email:
            data, call = await mcp.call("search_tickets", {"requester_email": email, "status": "open",
                                                           "limit": escalation.MAX_OPEN_TICKETS})
            calls.append(call.model_dump())
            checked = bool(data.get("ok"))
            signals = escalation.history_signals(data.get("tickets")) if checked else []
        duplicate = checked and escalation.open_duplicate(
            signals, escalation.duplicate_text(state.get("messages") or []), state.get("codes") or [])
        log.info("memory: open_tickets=%d history_checked=%s duplicate=%s", len(signals), checked, duplicate)
        memory = {"open_ticket_signals": signals, "history_checked": checked, "tool_calls": calls}
        if duplicate:
            return {**result(escalation.DUPLICATE_REPLY, "blocked"), **memory}

        if llm is None:
            return {**result(escalation.ESCALATION_FAILED_REPLY, "error"), **memory}
        try:
            out = await llm.complete_json(ESCALATION_SYSTEM_PROMPT, build_escalation_input(state, text))
        except LLMError:
            return {**result(escalation.ESCALATION_FAILED_REPLY, "error"), **memory}
        outcome = escalation.build_draft(out.data, state, text)
        if outcome.kind == "draft":
            return {**result(escalation.draft_reply(outcome.draft, state.get("employee")), "draft_proposed",
                             outcome.draft, out.usage), **memory}
        if outcome.kind == "ask_steps":
            return {**result(outcome.reply, "clarify", usage=out.usage), **memory}
        return {**result(outcome.reply, "blocked", usage=out.usage), **memory}

    return escalate


# ---------------------------------------------------------------- respond (Phase 2): the last check, no LLM

# A ticket ID may only ever come from the ticket system, in /api/approve's `result`. Never in a chat reply.
_TICKET_ID = re.compile(r"\bIT-\d{2,7}\b")
# Sentences that claim a ticket exists: the agent cannot create tickets, so such a claim is always false here.
# Passive form ("a ticket was created"): false unless the sentence is a genuine conditional about the process
# ("Once a ticket is filed, ..."). First-person form ("I've opened a ticket"): always false.
_TICKET_CLAIM_PASSIVE = re.compile(
    r"(?i)\b(?:ticket|request|case)\b[^.!?\n]{0,40}?\b(?:has been|have been|was|were|is now|is)\s+"
    r"(?:created|filed|opened|submitted|raised|logged|registered)\b")
_TICKET_CLAIM_FIRST_PERSON = re.compile(
    r"(?i)\b(?:i|we|i've|we've)\s+(?:have\s+|just\s+|already\s+)*(?:created|filed|opened|submitted|raised|logged)\s+"
    r"(?:a|an|the|your|this)\s+(?:new\s+)?(?:ticket|request|case)\b")
# Sentences that claim the agent already acted on the issue. It can only OFFER escalation, so these are false.
# Offers ("I can escalate it", "Would you like me to help prepare a request?") and instructions to the employee
# ("report it to Security Operations") have no such subject + past tense, and are kept.
_ACTION_CLAIM = re.compile(
    r"(?i)\b(?:i|we)(?:'ve|\s+have)?\s+(?:just\s+|already\s+)*"
    r"(?:escalated|forwarded|reported|notified|assigned|passed\s+(?:it|this|your))\b"
    r"|\b(?:issue|request|problem|case|query|ticket)\s+(?:has|have)\s+(?:already\s+)?been\s+"
    r"(?:escalated|forwarded|reported|assigned|passed\s+on)\b"
    r"|\b(?:issue|request|problem|case|query|ticket)\s+(?:was|were)\s+(?:escalated|forwarded|reported|assigned)\b"
    r"|\b(?:it|security\s+operations|the\s+service\s+desk|a\s+technician|the\s+team|support|someone)\s+"
    r"(?:(?:has|have)\s+(?:already\s+)?been|was|were)\s+(?:notified|informed|alerted)\b")
# "A technician will contact you" is a promise only an existing ticket could make.
_CONTACT_PROMISE = re.compile(
    r"(?i)\b(?:a\s+technician|an\s+engineer|someone|it|the\s+it\s+team|security\s+operations|the\s+service\s+desk|"
    r"support)\s+(?:will|is\s+going\s+to)\s+(?:contact|call|reach\s+out\s+to|email|get\s+back\s+to|visit)\s+you\b")
# A genuine conditional about the ticket process ("Once a ticket is filed, Security Operations will contact you",
# "If you approve the draft, a technician will call you") describes what WOULD happen, so it is kept. Narrow on
# purpose: the sentence must START with the condition and be about a ticket, request, approval or escalation.
# "A technician will contact you when they are free" or "After lunch, someone will call you" are still promises.
# Never an exemption for a claim that something already happened (_ACTION_CLAIM, first-person ticket claims).
_CONDITIONAL_LEAD = re.compile(r"(?i)^\s*(?:if|once|after|when|whenever|as\s+soon\s+as|unless)\b")
_PROCESS_EVENT = re.compile(r"(?i)\b(?:ticket|request|draft|approv\w*|escalat\w*)\b")
_CURLY_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
TICKET_ID_REMOVED = "[ticket ID removed]"


def is_genuine_conditional(sentence: str) -> bool:
    return bool(_CONDITIONAL_LEAD.match(sentence)) and bool(_PROCESS_EVENT.search(sentence))


def is_false_claim(sentence: str) -> bool:
    """True if the sentence claims a ticket exists or an action already happened (see the patterns above)."""
    sentence = sentence.translate(_CURLY_APOSTROPHES)        # "I’ve escalated" is "I've escalated"
    if _TICKET_CLAIM_FIRST_PERSON.search(sentence) or _ACTION_CLAIM.search(sentence):
        return True
    if _TICKET_CLAIM_PASSIVE.search(sentence) or _CONTACT_PROMISE.search(sentence):
        return not is_genuine_conditional(sentence)
    return False


def incident_reply(incident: dict) -> str:
    """The reply for a verified active incident, from the status page only (no model involved)."""
    parts = [f"This matches the active incident {incident['id']}: {incident.get('title') or 'service problem'}."]
    if incident.get("workaround"):
        parts.append(f"Workaround: {incident['workaround']}")
    parts.append(incident.get("instructions")
                 or "IT is already working on it, so there is no need to open a separate ticket.")
    return " ".join(parts)


def without_ticket_claims(text: str) -> str:
    """Drop every sentence that claims a ticket was created or an action was already taken (is_false_claim);
    keep the line structure of the rest."""
    lines = []
    for line in text.split("\n"):
        sentences = [s for s in _SENTENCE_END.split(line) if not is_false_claim(s)]
        kept = " ".join(sentences).strip()
        if kept or not line.strip():
            lines.append(kept)
    return "\n".join(lines).strip()


def safe_reply(text: str) -> str:
    """The outgoing reply: secrets redacted, false ticket claims and ticket IDs removed, bounded, never empty."""
    text = redact_secrets(text or "")
    text = without_ticket_claims(text)
    text = _TICKET_ID.sub(TICKET_ID_REMOVED, text).strip()[:MAX_REPLY_CHARS].strip()
    return text or LLM_FAILED_REPLY


def employee_messages(state: AgentState) -> str:
    """Everything the employee wrote in this conversation (as stored: redacted, capped at the last messages). The
    verifier accepts facts the employee gave, also in an earlier turn (e.g. before a clarifying question)."""
    return "\n".join(str(m.content) for m in state.get("messages") or [] if isinstance(m, HumanMessage))


def valid_policy_citations(raw) -> list[dict]:
    """Code-issued citations from this turn, kept only if they name an SP document of a policy rule."""
    allowed = set(policy.RULE_DOC.values())
    out = []
    for citation in raw or []:
        if (isinstance(citation, dict) and citation.get("id") in allowed
                and citation.get("source") == policy.POLICY_SOURCE):
            out.append({"id": citation["id"], "title": policy.TITLES.get(citation["id"], ""),
                        "source": policy.POLICY_SOURCE})
    return list({c["id"]: c for c in out}.values())


def make_respond() -> Callable[[AgentState], Awaitable[dict]]:
    """The node that ends every turn (FINAL_NODE): one place that checks what leaves the agent.

      * a verified active incident gets the status-page template (no citations, no model);
      * otherwise the answer node's reply: citations kept only if they are documents in THIS turn's evidence,
        and an "answered" reply left with no valid citation is replaced by the honest fallback;
      * always: secrets redacted, invented ticket IDs and "ticket created" claims removed, non-empty, bounded;
      * a ticket draft proposal survives only on the turn escalate made it (outcome "draft_proposed"); on every
        other turn it is cleared, so an old draft can never be picked up later.
    """

    async def respond(state: AgentState) -> dict:
        incident = state.get("active_incident")
        draft = None
        policy_citations: list[dict] = []
        if state.get("route") == "incident" and isinstance(incident, dict) and incident.get("id"):
            reply, outcome, citations, usage = incident_reply(incident), "incident", [], {}
        else:
            titles: dict[str, str] = {}
            for record in state.get("evidence") or []:
                titles.setdefault(record["doc_id"], record.get("title") or "")
            ids = [c.get("id") for c in state.get("citations") or [] if isinstance(c, dict)]
            citations = [{"id": doc_id, "title": titles[doc_id]} for doc_id in dict.fromkeys(ids) if doc_id in titles]
            reply = state.get("reply") or ""
            outcome = state.get("outcome") or "error"
            usage = dict(state.get("usage") or {})
            if outcome == "answered" and not citations:
                reply, outcome = UNGROUNDED_REPLY, "not_documented"
            if outcome == "answered":                       # a model-written answer: Phase 3 checks
                advice = policy.find_forbidden_advice(reply)
                if advice:                                  # the whole reply goes, never a patched version
                    log.info("policy: model answer replaced by the %s refusal", advice.rule)
                    reply, outcome, citations = advice.reply, "policy_refusal", []
                    policy_citations = [advice.citation]
                else:
                    checked = verify_answer(reply, citations, list(state.get("evidence") or []),
                                            employee_messages(state), fallback_reply=UNGROUNDED_REPLY)
                    # Counts only: never the reply, the removed sentences, the facts or the employee's words.
                    log.info("verifier: checked=%d unsupported=%d removed_sentences=%d citations_dropped=%d "
                             "fallback=%s", checked.checked, len(checked.unsupported), len(checked.removed),
                             len(citations) - len(checked.citations), checked.fallback)
                    reply, citations = checked.reply, checked.citations
                    if checked.fallback:
                        outcome = "not_documented"
            if outcome == "policy_refusal" and not policy_citations:
                policy_citations = valid_policy_citations(state.get("policy_citations"))
            if outcome == "draft_proposed" and isinstance(state.get("draft"), dict):
                draft = dict(state["draft"])
        reply = safe_reply(reply)
        # The response shows model/evidence citations and, on a refusal, the code-issued SP citation; the state
        # keeps the code-issued ones apart (policy_citations), so they are never mistaken for the model's.
        shown = citations + [{"id": c["id"], "title": c["title"]} for c in policy_citations]
        return {"reply": reply, "messages": [AIMessage(content=reply)], "citations": shown,
                "policy_citations": policy_citations, "outcome": outcome, "usage": usage, "draft": draft}

    return respond
