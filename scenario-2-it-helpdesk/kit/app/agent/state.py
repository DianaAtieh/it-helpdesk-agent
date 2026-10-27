"""The agent's state: the fields every node reads and writes, saved by the checkpointer after each step.

Only checkpoint-safe values go in here: LangChain messages, strings, numbers, lists and plain dicts.
No Pydantic objects (the checkpoint serializer warns about, or in strict mode blocks, custom types).

Phase 1 fields only. Later phases add triage, retrieval and escalation fields when a node fills them.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # The conversation. add_messages APPENDS each turn's new messages to the saved history.
    messages: Annotated[list[AnyMessage], add_messages]
    # Who is talking (as claimed by the client; /api/chat already locks it per session).
    user_email: str | None
    # The first active incident on the status page (the API's plain dict), or None.
    active_incident: dict | None
    # This turn's reply text, and this turn's tool calls as {"tool", "ok", "ms"} dicts.
    # No reducer: each turn overwrites them, so they never carry over from the previous turn.
    reply: str
    tool_calls: list[dict]
    # The approval gate's state, saved so it survives a restart or an eviction from memory
    # (written by app/main.py, never by a node): the ticket draft waiting for approval as a plain
    # PendingAction dict, and {action_id: ticket_id} for drafts that already became tickets.
    pending_action: dict | None
    filed: dict[str, str]
