"""Request and response shapes of our API, as defined in common/CONTRACT.md.

The field names are the contract: the Gradio UI, the eval runner and the hidden
tests read them. Do not rename them.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- requests

class User(BaseModel):
    email: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    user: User | None = None          # optional: sample case S2-02 sends no user


class ApproveRequest(BaseModel):
    session_id: str
    action_id: str
    approved: bool


class ResetRequest(BaseModel):
    session_id: str | None = None     # None = reset every session


# ---------------------------------------------------------------- response

class Citation(BaseModel):
    id: str                           # e.g. "KB-002"
    title: str = ""
    url: str | None = None


class ToolCall(BaseModel):
    tool: str                         # e.g. "get_service_status"
    ok: bool
    ms: int


class PendingAction(BaseModel):
    action_id: str                    # we generate it; it is also the idempotency key
    type: str = "create_ticket"
    summary: str                      # exactly what the human approves
    payload: dict[str, Any]           # the ticket fields


class ChatResponse(BaseModel):
    session_id: str
    reply: str                        # always present
    state: str | None = None
    pending_action: PendingAction | None = None
    result: dict[str, Any] | None = None       # only after the API really created a ticket
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
