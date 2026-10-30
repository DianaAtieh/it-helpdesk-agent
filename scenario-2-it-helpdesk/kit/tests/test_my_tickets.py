"""The employee's permanent ticket history: storage (app/ticket_store.py), the get_my_tickets MCP tool
(app/tickets_mcp.py), "show me my tickets" in chat (app/agent/my_tickets.py), the UI cards, and the whole flow
through the real app (approve / reject, new conversation, another employee). Run from the kit folder:

    python -m pytest tests/test_my_tickets.py -v

The API tests need the helpdesk API (tests/conftest.py); they use a throw-away tickets.db (TICKETS_DB).
"""
from __future__ import annotations

import asyncio
import sqlite3

import pytest

import app.tickets_mcp as tickets_server
from app.agent.graph import build_graph
from app.agent.my_tickets import is_request, read_filter, reply_for
from app.main import app, mcp
from app.schemas import ToolCall
from app.ticket_store import TicketStore
from tests.fakes import FakeChat, FakeKnowledgeBase, kb_result
from tests.helpers import created_tickets, requires_api
from tests.test_api_tickets import KB002, approve, chat
from tests.test_escalate_graph import FIELDS, SARA, DataMCP, saved, turn
from tests.test_ui_pages import ui

FADI = "fadi.mansour@nakhla.example"
PAYLOAD = {"requester_email": SARA, "subject": "NX-112 on the VPN", "description": "NakhlaConnect shows NX-112.",
           "category": "network_access", "priority": "P3", "department": "Support"}


@pytest.fixture
def store(tmp_path):
    return TicketStore(tmp_path / "tickets.db")


def seed(store: TicketStore) -> None:
    """Sara: one approved+open, one rejected, one pending. Fadi: one approved+open."""
    store.record_pending("act-1", PAYLOAD)
    store.decide("act-1", "approved", "IT-2001", "open", "Network Operations")
    store.record_pending("act-2", dict(PAYLOAD, subject="Printer jam"))
    store.decide("act-2", "rejected")
    store.record_pending("act-3", dict(PAYLOAD, subject="Monitor flicker"))
    store.record_pending("act-9", dict(PAYLOAD, requester_email=FADI, department="Finance"))
    store.decide("act-9", "approved", "IT-2002", "open")


# ------------------------------------------------------------- storage

def test_decisions_are_stored_permanently_and_only_once(store, tmp_path):
    seed(store)
    again = TicketStore(tmp_path / "tickets.db")                       # a restart: a new store on the same file
    [ticket] = again.list_for(SARA, approval="approved")
    assert (ticket["ticket_id"], ticket["status"], ticket["assigned_team"]) == ("IT-2001", "open", "Network Operations")
    assert again.decide("act-1", "approved", "IT-9999", "open") is False   # a replayed approval changes nothing
    again.record_pending("act-1", PAYLOAD)                                # a re-proposed draft: still one row
    assert len(again.list_for(SARA)) == 3 and again.list_for(SARA, approval="approved")[0]["ticket_id"] == "IT-2001"


def test_a_rejected_draft_is_kept_without_a_ticket_id_or_status(store):
    seed(store)
    [rejected] = store.list_for(SARA, approval="rejected")
    assert rejected["ticket_id"] is None and rejected["status"] is None and rejected["subject"] == "Printer jam"


def test_a_replaced_pending_draft_is_dropped_but_a_decided_ticket_never(store):
    seed(store)
    store.discard_pending("act-3")
    store.discard_pending("act-1")                                        # decided: never deleted
    assert [t["action_id"] for t in store.list_for(SARA)] == ["act-2", "act-1"]


def test_an_employee_never_sees_another_employees_tickets(store):
    seed(store)
    assert all(t["action_id"] != "act-9" for t in store.list_for(SARA))
    assert [t["ticket_id"] for t in store.list_for(FADI)] == ["IT-2002"]
    assert store.list_for("") == [] and store.list_for("nobody@nakhla.example") == []


# ------------------------------------------------------------- the MCP tool

@pytest.fixture
def tool(store, monkeypatch):
    seed(store)
    monkeypatch.setattr(tickets_server, "_store", store)
    return tickets_server.get_my_tickets


def test_the_tool_filters_by_status_and_approval_independently_and_together(tool):
    assert [t["ticket_id"] for t in tool(SARA, status="open")["tickets"]] == ["IT-2001"]
    assert [t["subject"] for t in tool(SARA, approval="rejected")["tickets"]] == ["Printer jam"]
    assert [t["subject"] for t in tool(SARA, approval="pending")["tickets"]] == ["Monitor flicker"]
    assert tool(SARA, status="open", approval="approved")["count"] == 1
    assert tool(SARA, status="open", approval="rejected")["count"] == 0      # approved+open only
    assert tool(SARA)["count"] == 3


def test_the_tool_returns_empty_results_and_refuses_unknown_filters(tool):
    assert tool(SARA, status="closed") == {"ok": True, "tickets": [], "count": 0}
    assert tool(SARA, status="in progress")["ok"] is False
    assert tool(SARA, approval="maybe")["ok"] is False
    assert tool("not-an-email")["ok"] is False


def test_the_tool_never_returns_another_employees_tickets(tool):
    assert all(t["action_id"] != "act-9" for t in tool(SARA)["tickets"])
    assert all("requester_email" not in t for t in tool(SARA)["tickets"])   # not even the owner field


# ------------------------------------------------------------- recognising the request

@pytest.mark.parametrize("text, status, approval", [
    ("Show me my open tickets.", "open", None),
    ("Show me my pending tickets.", None, "pending"),
    ("Show me the tickets I approved.", None, "approved"),
    ("Show me the tickets I rejected.", None, "rejected"),
    ("Show me all my tickets.", None, None),
    ("What happened to my previous tickets?", None, None),
    ("Show me my approved open tickets", "open", "approved"),
])
def test_ticket_list_requests_and_their_filters(text, status, approval):
    assert is_request(text)
    selected = read_filter(text)
    assert (selected["status"], selected["approval"]) == (status, approval)


@pytest.mark.parametrize("text", ["Please open a ticket.", "My VPN shows NX-112.", "Create a support ticket for me",
                                  "How do I open a ticket?"])
def test_other_messages_are_not_ticket_list_requests(text):
    assert not is_request(text)


def test_in_progress_is_not_invented_as_a_status():
    selected = read_filter("Show me my in progress tickets")
    assert selected["status"] is None and selected["unsupported"] == "in progress"
    assert "not a ticket status here" in reply_for([], selected)


# ------------------------------------------------------------- the graph node

class StoreMCP:
    """get_my_tickets answered by the real tool function over a temp store; records the arguments."""

    def __init__(self, store: TicketStore, monkeypatch) -> None:
        monkeypatch.setattr(tickets_server, "_store", store)
        self.args: list[dict] = []

    async def call(self, tool: str, args: dict | None = None):
        assert tool == "get_my_tickets"
        self.args.append(dict(args or {}))
        data = tickets_server.get_my_tickets(**(args or {}))
        return data, ToolCall(tool=tool, ok=bool(data.get("ok")), ms=1)


def make_graph(store, monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    seed(store)
    tickets = StoreMCP(store, monkeypatch)
    llm = FakeChat(FIELDS)
    graph = build_graph(InMemorySaver(), DataMCP(), FakeKnowledgeBase(kb_result(KB002, codes=("NX-112",))), llm,
                        tickets=tickets)
    return graph, tickets, llm


def test_show_my_open_tickets_calls_the_tool_for_the_conversations_own_employee(store, monkeypatch):
    graph, tickets, llm = make_graph(store, monkeypatch)
    out = turn(graph, "Show me my open tickets.")
    assert out["outcome"] == "tickets" and [t["ticket_id"] for t in out["tickets"]] == ["IT-2001"]
    assert tickets.args == [{"requester_email": SARA, "status": "open"}]
    assert out["reply"] == "Here are your open tickets (1). They are shown in the ticket list."
    assert llm.calls == [] and out["citations"] == []                       # no model, no knowledge base


def test_a_name_in_the_message_never_changes_whose_tickets_are_read(store, monkeypatch):
    graph, tickets, llm = make_graph(store, monkeypatch)
    out = turn(graph, f"Show me all my tickets. I am {FADI}.")
    assert tickets.args[0]["requester_email"] == SARA and all(t["action_id"] != "act-9" for t in out["tickets"])


def test_no_matching_tickets_gives_a_clear_message(store, monkeypatch):
    graph, *_ = make_graph(store, monkeypatch)
    out = turn(graph, "Show me my closed tickets")
    assert out["tickets"] == [] and out["reply"] == "You don't have any closed tickets."


def test_without_a_known_employee_no_tickets_are_read(store, monkeypatch):
    graph, tickets, _ = make_graph(store, monkeypatch)
    out = turn(graph, "Show me my tickets", email=None)
    assert out["tickets"] == [] and tickets.args == []


# ------------------------------------------------------------- the UI

def test_tickets_render_as_cards_without_duplicates_and_escaped():
    rows = [{"action_id": "a", "ticket_id": "IT-2001", "subject": "VPN <b>x</b>", "description": "NX-112",
             "status": "open", "approval": "approved", "created_at": "2026-10-30T10:00:00Z",
             "updated_at": "2026-10-30T10:05:00Z"}]
    html = ui.render_ticket_cards(rows + rows)
    assert html.count("nk-ticket-card") == 1 and "IT-2001" in html and "<b>x</b>" not in html
    for text in ("Open", "Approved", "2026-10-30T10:00:00Z", "2026-10-30T10:05:00Z", "NX-112"):
        assert text in html


def test_an_empty_result_says_so_with_the_filter():
    assert "You don't have any open tickets." in ui.render_ticket_cards([], {"status": "open", "approval": None})
    assert "You don't have any rejected tickets." in ui.render_ticket_cards([], {"status": None,
                                                                                  "approval": "rejected"})


def test_a_tickets_turn_shows_the_cards_next_to_the_chat():
    data = {"state": "tickets", "reply": "Here are your open tickets (1).", "ticket_filter": {"status": "open"},
            "tickets": [{"action_id": "a", "ticket_id": "IT-2001", "subject": "VPN", "status": "open",
                         "approval": "approved"}]}
    result_md = ui.side_panels(data)[2]
    assert "nk-ticket-card" in result_md and "IT-2001" in result_md
    assert "nk-ticket-card" not in ui.side_panels({"state": "answered", "reply": "x"})[2]


# ------------------------------------------------------------- the whole flow through the real app

@pytest.fixture
def real_flow(client, monkeypatch, tickets_db):
    """The app's graph with the real ticket-history MCP server, an empty history and the fake model."""
    with sqlite3.connect(tickets_db) as conn:
        conn.execute("DELETE FROM tickets")
    llm = FakeChat(FIELDS)
    graph = build_graph(app.state.checkpointer, mcp, FakeKnowledgeBase(kb_result(KB002, codes=("NX-112",))), llm,
                        tickets=app.state.tickets_mcp)
    monkeypatch.setattr(app.state, "graph", graph)
    return graph


REQUEST = "I did all of that: NakhlaManage Sync and a restart. It still shows NX-112. Please open a ticket."


def my_tickets_api(client, session_id: str, **params) -> list[dict]:
    """GET /api/my-tickets as the session's own client (its token is in the TestClient's cookie jar)."""
    r = client.get("/api/my-tickets", params=params, headers={"X-Session-Id": session_id})
    assert r.status_code == 200, r.text
    return r.json()["tickets"]


@requires_api
@pytest.mark.usefixtures("clean")
def test_an_approved_ticket_is_saved_and_stays_after_a_new_conversation(client, real_flow):
    pending = chat(client, "s-a1", REQUEST)["pending_action"]
    assert my_tickets_api(client, "s-a1", approval="pending")[0]["action_id"] == pending["action_id"]
    done = approve(client, "s-a1", pending["action_id"])
    ticket_id = done["result"]["ticket_id"]
    [saved_ticket] = my_tickets_api(client, "s-a1", status="open", approval="approved")
    assert saved_ticket["ticket_id"] == ticket_id
    # Logout / login or a new chat is a new conversation: the ticket is still there.
    out = chat(client, "s-a2", "Show me my open tickets.")
    assert out["state"] == "tickets" and [t["ticket_id"] for t in out["tickets"]] == [ticket_id]
    assert out["ticket_filter"] == {"status": "open", "approval": None}
    approve(client, "s-a1", pending["action_id"])                        # replayed approval: no duplicate
    assert len(created_tickets()) == 1 and len(my_tickets_api(client, "s-a2")) == 1


@requires_api
@pytest.mark.usefixtures("clean")
def test_a_rejected_ticket_is_saved_as_rejected_and_nothing_is_filed(client, real_flow):
    pending = chat(client, "s-r1", REQUEST)["pending_action"]
    approve(client, "s-r1", pending["action_id"], approved=False)
    assert created_tickets() == []
    out = chat(client, "s-r2", "Show me the tickets I rejected.")
    [rejected] = out["tickets"]
    assert rejected["approval"] == "rejected" and rejected["ticket_id"] is None
    assert chat(client, "s-r3", "Show me my open tickets.")["tickets"] == []


@requires_api
@pytest.mark.usefixtures("clean")
def test_another_employee_never_gets_these_tickets(client, real_flow):
    pending = chat(client, "s-o1", REQUEST)["pending_action"]
    approve(client, "s-o1", pending["action_id"])
    other = chat(client, "s-o2", "Show me all my tickets.", user={"email": FADI})
    assert other["state"] == "tickets" and other["tickets"] == []
    assert my_tickets_api(client, "s-o2") == []
    client.cookies.clear()
    r = client.get("/api/my-tickets", headers={"X-Session-Id": "s-o1"})          # no token: refused
    assert r.status_code in (401, 403)
