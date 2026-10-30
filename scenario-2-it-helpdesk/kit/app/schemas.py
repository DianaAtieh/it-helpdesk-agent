"""Request and response shapes of our API, as defined in common/CONTRACT.md.

The field names are the contract: the Gradio UI, the eval runner and the hidden
tests read them. Do not rename them.
"""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_serializer

# Size limits keep one request from costing real money (LLM tokens) or memory.
SessionId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]
MAX_MESSAGE_CHARS = 4000


# ---------------------------------------------------------------- requests

class User(BaseModel):
    email: str = Field(min_length=3, max_length=254)

    @field_validator("email")
    @classmethod
    def normalise(cls, value: str) -> str:
        return value.strip().lower()


class ChatRequest(BaseModel):
    session_id: SessionId
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    user: User | None = None          # optional: sample case S2-02 sends no user


class SessionStartRequest(BaseModel):
    """POST /api/session/start: link a brand-new conversation to a signed-in employee before any message."""
    session_id: SessionId
    user: User


class ApproveRequest(BaseModel):
    session_id: SessionId
    action_id: str = Field(min_length=1, max_length=64)
    approved: bool


class ResetRequest(BaseModel):
    session_id: SessionId | None = None     # None = reset every session (needs the admin token)
    user: User | None = None                # a linked conversation needs its employee's email (or the admin token)


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
    # Only on a "show me my tickets" turn (state "tickets"): the employee's own ticket records from get_my_tickets,
    # for the UI to render as cards, and the filters that were applied ({"status", "approval"}).
    tickets: list[dict[str, Any]] | None = None
    ticket_filter: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    def _contract_shape(self, handler):
        """Every other turn keeps the contract's exact field set: the two ticket-list fields appear only when set."""
        data = handler(self)
        if self.tickets is None:
            data.pop("tickets", None)
            data.pop("ticket_filter", None)
        return data
