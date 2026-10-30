"""MCP server for the employee's own ticket history (stdio), started by the app next to the kit's helpdesk MCP.

Same infrastructure as the kit's server (FastMCP from the `mcp` package, the app's HelpdeskMCP client class), but a
separate process because the data is the app's own: the permanent ticket record in app/ticket_store.py (tickets.db),
which includes drafts the employee rejected or has not decided yet. The kit's services/ stay unchanged.

Identity: requester_email is passed ONLY by app code (app/agent/nodes.py and app/main.py), always the
conversation's locked, directory-checked email; no model can call MCP tools in this app, and the employee's text
never chooses it. The query filters by it in SQL, so a call can never return another employee's ticket.

    python -m app.tickets_mcp        (normally started by the app over stdio)
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.ticket_store import APPROVALS, STATUSES, TicketStore

mcp = FastMCP("nakhla-my-tickets", log_level="WARNING")
_store: TicketStore | None = None


def store() -> TicketStore:
    global _store
    if _store is None:
        _store = TicketStore()
    return _store


def _invalid(field: str, expected: str) -> dict:
    return {"ok": False, "error": {"code": "invalid_argument", "message": f"{field} must be {expected}."}}


@mcp.tool()
def get_my_tickets(requester_email: str, status: str | None = None, approval: str | None = None) -> dict:
    """The tickets of ONE employee (requester_email, set by the app from the authenticated session), newest first.

    status: open | resolved | closed (the helpdesk status of an approved ticket), or omit for any.
    approval: pending | approved | rejected (the employee's decision on the draft), or omit for any.
    The two filters are independent and combine. Returns {"ok": true, "tickets": [...], "count": n}.
    """
    if not isinstance(requester_email, str) or "@" not in requester_email:
        return _invalid("requester_email", "the employee's work email")
    if status is not None and status not in STATUSES:
        return _invalid("status", " | ".join(STATUSES))
    if approval is not None and approval not in APPROVALS:
        return _invalid("approval", " | ".join(APPROVALS))
    tickets = store().list_for(requester_email, status=status, approval=approval)
    return {"ok": True, "tickets": tickets, "count": len(tickets)}


if __name__ == "__main__":
    mcp.run()
