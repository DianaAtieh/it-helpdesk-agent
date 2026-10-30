"""Permanent record of each employee's tickets: every draft they were asked to approve, and the decision.

The helpdesk API (services/, a simulation) keeps tickets in memory only and never sees a rejected draft, so the
app keeps its own record here, in its own SQLite file (app/data/tickets.db, or TICKETS_DB), like preferences.db.
It survives a restart, a logout, a new conversation and a new login. Nothing here ever deletes a decided ticket.

One row per approval request, keyed by its action_id (the approval gate's idempotency key), so a replayed approval
or a resumed graph can never make a second row:

  approval  pending   a draft is waiting for the employee's Approve or Reject (not filed)
            approved  the employee approved it and the helpdesk created it: ticket_id and status are set
            rejected  the employee rejected it: nothing was filed (ticket_id and status stay NULL)
  status    the helpdesk ticket status (ticket.schema.json: open | resolved | closed), only for approved tickets

Ownership: every read is filtered by requester_email in the SQL query itself. The email always comes from the
conversation's locked, directory-checked identity (app code), never from a model or from the employee's text.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
DEFAULT_TICKETS_DB = KIT / "app" / "data" / "tickets.db"
APPROVALS = ("pending", "approved", "rejected")
STATUSES = ("open", "resolved", "closed")           # the ticket schema's ticket_record.status enum
MAX_LIST = 50
FIELDS = ("action_id", "ticket_id", "subject", "description", "category", "priority", "department",
          "assigned_team", "status", "approval", "created_at", "updated_at")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _email(value) -> str:
    return str(value or "").strip().lower()


class TicketStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.getenv("TICKETS_DB") or DEFAULT_TICKETS_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS tickets (
                action_id TEXT PRIMARY KEY, requester_email TEXT NOT NULL, ticket_id TEXT, subject TEXT NOT NULL,
                description TEXT NOT NULL, category TEXT, priority TEXT, department TEXT, assigned_team TEXT,
                status TEXT, approval TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS tickets_owner ON tickets (requester_email, created_at)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def record_pending(self, action_id: str, payload: dict, assigned_team: str | None = None) -> None:
        """A draft now waits for approval. Re-recording the same action_id changes nothing."""
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tickets (action_id, requester_email, subject, description, category, priority, "
                "department, assigned_team, approval, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                "'pending', ?, ?) ON CONFLICT (action_id) DO NOTHING",
                (action_id, _email(payload.get("requester_email")), str(payload.get("subject") or ""),
                 str(payload.get("description") or ""), payload.get("category"), payload.get("priority"),
                 payload.get("department"), assigned_team, now, now))

    def discard_pending(self, action_id: str) -> None:
        """A pending draft that was replaced by a newer one was never decided: it is not a ticket."""
        with self._connect() as conn:
            conn.execute("DELETE FROM tickets WHERE action_id = ? AND approval = 'pending'", (action_id,))

    def decide(self, action_id: str, approval: str, ticket_id: str | None = None, status: str | None = None,
               assigned_team: str | None = None) -> bool:
        """Record the employee's decision on a PENDING draft, once. Returns False if there was none to decide."""
        if approval not in ("approved", "rejected"):
            raise ValueError(f"unknown approval decision: {approval}")
        status = status if approval == "approved" and status in STATUSES else None
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE tickets SET approval = ?, ticket_id = ?, status = ?, "
                "assigned_team = COALESCE(?, assigned_team), updated_at = ? "
                "WHERE action_id = ? AND approval = 'pending'",
                (approval, ticket_id if approval == "approved" else None, status, assigned_team, _now(), action_id))
            return cur.rowcount == 1

    def list_for(self, requester_email: str, status: str | None = None, approval: str | None = None) -> list[dict]:
        """The employee's own tickets, newest first; the filters are independent and combine with AND."""
        email = _email(requester_email)
        if not email:
            return []
        if status is not None and status not in STATUSES:
            raise ValueError(f"unknown status: {status}")
        if approval is not None and approval not in APPROVALS:
            raise ValueError(f"unknown approval decision: {approval}")
        sql, args = "SELECT * FROM tickets WHERE requester_email = ?", [email]
        if status is not None:
            sql, args = sql + " AND status = ?", [*args, status]
        if approval is not None:
            sql, args = sql + " AND approval = ?", [*args, approval]
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY created_at DESC, rowid DESC LIMIT ?", [*args, MAX_LIST]).fetchall()
        return [{key: row[key] for key in FIELDS} for row in rows]
