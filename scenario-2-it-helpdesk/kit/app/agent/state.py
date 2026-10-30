"""The agent's state: the fields every node reads and writes, saved by the checkpointer after each step.

Only checkpoint-safe values go in here: LangChain messages, strings, numbers, lists and plain dicts.
No Pydantic objects (the checkpoint serializer warns about, or in strict mode blocks, custom types).

Phase 1 fields, plus the triage, retrieve and answer fields (Phase 2). Escalation fields are added with the
nodes that fill them.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # The conversation. add_messages APPENDS each turn's new messages to the saved history.
    messages: Annotated[list[AnyMessage], add_messages]
    # Who is talking (as claimed by the client; /api/chat already locks it per session).
    user_email: str | None
    # The ACTIVE incident whose error codes include a code in the user's message (set by triage), or None.
    active_incident: dict | None
    # Triage (Phase 2). No reducer: triage writes every one of them on every turn, so none carries over.
    #   employee        {"email", "department", "vip", "contractor"} from the directory (lookup_employee) for
    #                   user_email, or None (no email, unknown employee, or the lookup failed). The department
    #                   comes from the directory only, never from what the user writes.
    #   codes           error codes in the latest message (strict form, e.g. "NX-781", "CRM-1150")
    #   incidents       the active incidents, trimmed to id, title, service, priority, error_codes,
    #                   workaround and instructions ([] when the status page could not be read)
    #   status_checked  False when the status page could not be read: then "no incident" is NOT known
    #   route           "incident" (a matching active incident), "status" (a service status question, answered
    #                   from the status page), "preference" (the message ONLY states a persistent preference: a
    #                   fixed reply, no knowledge base, embedding or model) or "retrieve"
    #   services        per-service status for a status question only, else [] (see nodes.status_services)
    #   status_topics   the services a status question names ([] otherwise)
    employee: dict | None
    codes: list[str]
    incidents: list[dict]
    status_checked: bool
    route: str
    services: list[dict]
    status_topics: list[str]
    # Long-term presentation preferences of the session's locked employee, loaded by /api/chat and /ws/chat on every
    # turn from app/agent/preferences.py ({} when none or no email). Only {language, verbosity, format} from the
    # whitelist; read ONLY by prompts.build_answer_input, never by routing, policy, tickets or approvals.
    preferences: dict[str, str]
    # Retrieve (Phase 2), written on every turn that searches the knowledge base:
    #   evidence        citeable chunks only (app.agent.retrieval.evidence_record dicts); their text is DATA
    #   undocumented_codes  codes in the question that no evidence mentions (e.g. CRM-1150)
    #   sufficiency     the retriever's level: sufficient | conditional_only | context_only | none
    #   requires_department_verification, rule_warnings: from the retriever, passed on to the answer
    #   rule_coverage   {lane: status} from the retriever's rule lanes (complete | partial | unavailable |
    #                   department_required | no_inventory_rule | topic_not_established). The answer node's
    #                   safety gate reads this, never the rule_warnings text.
    #   kb_status       ok | unavailable | search_failed
    evidence: list[dict]
    undocumented_codes: list[str]
    sufficiency: str
    requires_department_verification: bool
    rule_warnings: list[str]
    rule_coverage: dict[str, str]
    kb_status: str
    # Evidence evaluation (optional step after retrieve, app/agent/evidence_eval.py; written every turn it runs):
    #   evaluation        {"decision", "evaluations", "retried", ...}: a trace of the proposal and what code did with it
    #   evaluation_hold   {"outcome", "reply"} when the evaluator held the answer back (fixed code-written reply)
    #   evaluation_usage  the evaluator's token counts, added to the answer's usage
    evaluation: dict
    evaluation_hold: dict | None
    evaluation_usage: dict
    # Answer (Phase 2):
    #   citations       [{"id", "title"}], only documents that are in this turn's evidence
    #   outcome         answered | clarify | not_documented | policy_refusal | error
    #   usage           the chat model's token counts for this turn ({} when it was not called)
    citations: list[dict]
    outcome: str
    usage: dict
    # Phase 3: citations issued by CODE for a security-policy refusal ([{"id", "title", "source": "policy"}],
    # only the matched rule's SP document), kept apart from the model's citations. Set on every turn by respond.
    policy_citations: list[dict]
    # Memory (set by escalate on ticket-request turns): the requester's OPEN tickets from search_tickets, reduced to
    # [{"codes": [...], "keywords": [...]}] (escalation.history_signals): never a ticket ID, subject, description,
    # note or email. history_checked is False when the lookup failed or was not made (then no duplicate check).
    open_ticket_signals: list[dict]
    # AI verifier (build_graph(verify_answers=True)), set by respond on every turn: None when it did not run, else
    # counts only ({applied, fallback, checked, removed, groundedness, reason, verdicts}), never answer text.
    answer_verification: dict | None
    history_checked: bool
    # Escalate (Phase 4): a ticket draft PROPOSAL built by app.agent.escalation (TicketDraft-compatible plain
    # dict), only on the turn it was made (respond clears it on every other turn). It is not a ticket and not
    # yet a pending action: /api/chat validates and stores it with main.propose_draft; only /api/approve writes.
    # outcome is then "draft_proposed"; a blocked request is "blocked", a missing step or email "clarify".
    draft: dict | None
    # This turn's reply text, and this turn's tool calls as {"tool", "ok", "ms"} dicts.
    # No reducer: each turn overwrites them, so they never carry over from the previous turn.
    reply: str
    tool_calls: list[dict]
    # The approval gate's state, saved so it survives a restart or an eviction from memory
    # (written by app/main.py, never by a node): the ticket draft waiting for approval as a plain
    # PendingAction dict, and {action_id: ticket_id} for drafts that already became tickets.
    pending_action: dict | None
    filed: dict[str, str]
    # Who may approve (also written by app/main.py, never by a node): the sha256 of the session token
    # issued on the conversation's first turn, and when it was issued (ISO time). Never the token itself.
    token_hash: str
    token_issued_at: str
