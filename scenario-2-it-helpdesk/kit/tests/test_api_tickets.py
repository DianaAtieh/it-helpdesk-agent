"""API tests for ticket drafts from chat (Phase 4 Step 3): /api/chat proposes, /api/approve files. Run from the
kit folder:

    python -m pytest tests/test_api_tickets.py -v

The real app, its real MCP subprocess and the (local, simulated) helpdesk API; only the chat model and the
knowledge base are fakes (tests/fakes.py), so no external calls. A ticket request in chat must end in a
pending_action (state "await_approval") and NO ticket; only an approved /api/approve call files one.
Setup and fixtures: tests/conftest.py (needs the helpdesk API on port 8102, or HELPDESK_API_URL).
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.agent.escalation import DRAFT_READY_REPLY, IDENTITY_NOTICE
from app.agent.graph import build_graph
from app.main import DRAFT_IDENTITY_REPLY, DRAFT_INVALID_REPLY, app, mcp, sessions, thread_config
from app.tickets import IDENTITY_SUMMARY_FLAG
from tests.fakes import FakeChat, FakeKnowledgeBase, evidence, kb_result
from tests.helpers import arm_fault, created_tickets, requires_api

pytestmark = [requires_api, pytest.mark.usefixtures("clean")]

SARA = {"email": "sara.mikhail@nakhla.example"}          # Support
SAMIR = {"email": "samir.nassar@nakhla.example"}         # Executive Office (VIP)
KB002 = evidence("KB-002", "NX-112: device certificate missing. Sync NakhlaManage and restart.",
                 title="VPN error codes and fixes")
FIELDS = {"subject": "NX-112 on the VPN since this morning",
          "description": "NakhlaConnect shows NX-112 since this morning. I work from home and cannot work.",
          "category": "network_access", "priority": "P2",
          "attempted_steps": ["NakhlaManage Sync", "Restarted the laptop"], "kb_refs": ["KB-002"], "affected_users": 1}
ANSWER = {"reply": "Sync NakhlaManage, then restart the laptop.", "citations": ["KB-002"]}
REQUEST = "I did all of that: NakhlaManage Sync and a restart. It still shows NX-112. Please open a ticket."


class Tampered:
    """The app's real graph, with its output changed before /api/chat sees it: to prove that main.py itself
    fails closed, whatever the graph (or a model) returns."""

    def __init__(self, graph, change):
        self._graph, self._change = graph, change

    def __getattr__(self, name):
        return getattr(self._graph, name)

    async def ainvoke(self, payload, config):
        return self._change(await self._graph.ainvoke(payload, config))


@pytest.fixture
def agent(monkeypatch):
    """Install a graph with the fake knowledge base and the given scripted model answers; returns the model."""
    def install(*replies, change=None):
        llm = FakeChat(*replies)
        graph = build_graph(app.state.checkpointer, mcp, FakeKnowledgeBase(kb_result(KB002, codes=("NX-112",))), llm)
        monkeypatch.setattr(app.state, "graph", Tampered(graph, change) if change else graph)
        return llm
    return install


def chat(client, session_id: str, message: str, user: dict | None = SARA) -> dict:
    body = {"session_id": session_id, "message": message, **({"user": user} if user else {})}
    r = client.post("/api/chat", json=body)
    assert r.status_code == 200
    return r.json()


def approve(client, session_id: str, action_id: str, approved: bool = True, expect: int = 200) -> dict:
    r = client.post("/api/approve", json={"session_id": session_id, "action_id": action_id, "approved": approved})
    assert r.status_code == expect
    return r.json()


def saved(client, session_id: str) -> dict:
    return client.portal.call(app.state.graph.aget_state, thread_config(session_id)).values


# ------------------------------------------------------------- a draft from chat, then approval

def test_a_ticket_request_returns_a_pending_draft_and_files_nothing(client, agent):
    agent(FIELDS)
    out = chat(client, "s-1", REQUEST)
    assert out["state"] == "await_approval" and out["reply"] == DRAFT_READY_REPLY and out["result"] is None
    pending = out["pending_action"]
    assert pending["type"] == "create_ticket" and pending["action_id"].startswith("act-")
    assert pending["payload"]["requester_email"] == SARA["email"] and pending["payload"]["department"] == "Support"
    assert pending["payload"]["category"] == "network_access" and pending["payload"]["priority"] == "P2"
    assert pending["payload"]["kb_refs"] == ["KB-002"]
    assert pending["summary"].startswith("Ticket draft · network_access · P2 · Support → Network Operations: ")
    assert created_tickets() == []                                               # nothing filed by chat
    assert saved(client, "s-1")["pending_action"]["action_id"] == pending["action_id"]
    assert "create_ticket" not in [c["tool"] for c in out["tool_calls"]]


def test_the_s2_05_flow_answer_then_draft_then_approve_files_exactly_the_draft(client, agent):
    agent(ANSWER, FIELDS)
    first = chat(client, "s-1", "My NakhlaConnect VPN shows NX-112 since this morning. I work from home.")
    assert first["state"] == "answered" and first["pending_action"] is None
    pending = chat(client, "s-1", REQUEST)["pending_action"]
    done = approve(client, "s-1", pending["action_id"])                          # the cookie from the first turn
    ticket_id = done["result"]["ticket_id"]
    assert done["state"] == "ticket_created" and ticket_id in done["reply"]
    [ticket] = created_tickets()
    assert ticket["ticket_id"] == ticket_id and ticket["category"] == "network_access" and ticket["priority"] == "P2"
    assert ticket["requester_email"] == SARA["email"] and ticket["kb_refs"] == ["KB-002"]
    assert ticket["assigned_team"] == "Network Operations"


def test_replaying_the_approval_files_once(client, agent):
    agent(FIELDS)
    pending = chat(client, "s-1", REQUEST)["pending_action"]
    first = approve(client, "s-1", pending["action_id"])
    again = approve(client, "s-1", pending["action_id"])
    assert again["result"] == {"ticket_id": first["result"]["ticket_id"]} and "already filed" in again["reply"]
    assert len(created_tickets()) == 1


def test_rejecting_the_draft_files_nothing(client, agent):
    agent(FIELDS)
    pending = chat(client, "s-1", REQUEST)["pending_action"]
    out = approve(client, "s-1", pending["action_id"], approved=False)
    assert out["result"] is None and "IT-" not in out["reply"] and created_tickets() == []
    assert saved(client, "s-1").get("pending_action") is None
    assert "nothing was filed" in approve(client, "s-1", pending["action_id"])["reply"]


def test_a_helpdesk_failure_claims_nothing_and_a_retry_files_once(client, agent):
    agent(FIELDS)
    pending = chat(client, "s-1", REQUEST)["pending_action"]
    arm_fault("create_ticket", "error_503")
    failed = approve(client, "s-1", pending["action_id"])
    assert failed["result"] is None and "IT-" not in failed["reply"] and failed["state"] == "await_approval"
    assert created_tickets() == []
    retried = approve(client, "s-1", pending["action_id"])
    assert retried["result"]["ticket_id"].startswith("IT-") and len(created_tickets()) == 1


def test_a_newer_draft_makes_the_older_one_stale(client, agent):
    agent(FIELDS, dict(FIELDS, subject="NX-112 again, second draft"))
    old = chat(client, "s-1", REQUEST)["pending_action"]
    new = chat(client, "s-1", "Still NX-112 after another restart. Please open a ticket.")["pending_action"]
    assert "nothing was filed" in approve(client, "s-1", old["action_id"])["reply"]
    assert approve(client, "s-1", new["action_id"])["result"]["ticket_id"].startswith("IT-")
    assert len(created_tickets()) == 1


def test_only_the_conversations_client_can_approve_a_chat_made_draft(client, agent):
    agent(FIELDS)
    pending = chat(client, "s-1", REQUEST)["pending_action"]
    client.cookies.clear()                                                       # another client, no token
    out = approve(client, "s-1", pending["action_id"], expect=403)
    assert out["state"] == "unauthorized" and created_tickets() == []
    assert saved(client, "s-1")["pending_action"]["action_id"] == pending["action_id"]


# ------------------------------------------------------------- Executive Office: the identity review flag

def test_an_executive_office_draft_is_p1_and_flagged_in_the_summary_and_the_ticket(client, agent):
    agent(dict(FIELDS, priority="P4", subject="Monitor flickers on my laptop"))
    out = chat(client, "s-1", "My monitor flickers. I restarted it. Please open a ticket.", SAMIR)
    pending = out["pending_action"]
    assert pending["payload"]["priority"] == "P1" and pending["payload"]["department"] == "Executive Office"
    assert IDENTITY_SUMMARY_FLAG in pending["summary"] and pending["payload"]["description"].startswith(IDENTITY_NOTICE)
    assert "verify your identity" in out["reply"] and "verified your identity" not in out["reply"]
    approve(client, "s-1", pending["action_id"])
    [ticket] = created_tickets()
    assert ticket["priority"] == "P1" and ticket["description"].startswith(IDENTITY_NOTICE)


def test_other_departments_have_no_identity_flag(client, agent):
    agent(FIELDS)
    pending = chat(client, "s-1", REQUEST)["pending_action"]
    assert IDENTITY_SUMMARY_FLAG not in pending["summary"]
    assert IDENTITY_NOTICE not in pending["payload"]["description"]


# ------------------------------------------------------------- main.py fails closed, whatever the graph returns

def _draft_with(**changes):
    def change(out):
        if out.get("outcome") == "draft_proposed":
            out = {**out, "draft": {**out["draft"], **changes}}
        return out
    return change


def test_a_draft_for_another_requester_is_refused(client, agent):
    agent(FIELDS, change=_draft_with(requester_email=SAMIR["email"]))
    out = chat(client, "s-1", REQUEST)
    assert (out["state"], out["reply"], out["pending_action"]) == ("error", DRAFT_IDENTITY_REPLY, None)
    assert saved(client, "s-1").get("pending_action") is None and created_tickets() == []
    last = saved(client, "s-1")["messages"][-1]
    assert isinstance(last, AIMessage) and last.content == DRAFT_IDENTITY_REPLY       # the history is not left lying


def test_a_draft_without_a_requester_is_refused(client, agent):
    agent(FIELDS, change=_draft_with(requester_email=""))
    out = chat(client, "s-1", REQUEST)
    assert (out["state"], out["pending_action"]) == ("error", None) and out["reply"] == DRAFT_IDENTITY_REPLY


def test_a_conversation_without_an_email_never_gets_a_draft_stored(client, agent):
    def forge(out):                                     # pretend the graph proposed a draft anyway
        return {**out, "outcome": "draft_proposed", "draft": {**FIELDS, "requester_email": SARA["email"],
                                                              "department": "Support"}}
    agent(FIELDS, change=forge)
    out = chat(client, "s-1", REQUEST, user=None)
    assert (out["state"], out["reply"], out["pending_action"]) == ("error", DRAFT_IDENTITY_REPLY, None)
    assert sessions["s-1"]["pending_action"] is None


@pytest.mark.parametrize("changes", [{"kb_refs": ["KB-999"]}, {"subject": "x"}, {"category": "printers"}],
                         ids=["unknown kb ref", "subject too short", "unknown category"])
def test_an_invalid_draft_is_an_honest_reply_with_no_card(client, agent, changes):
    agent(FIELDS, change=_draft_with(**changes))
    out = chat(client, "s-1", REQUEST)
    assert (out["state"], out["reply"], out["pending_action"]) == ("error", DRAFT_INVALID_REPLY, None)
    assert "IT-" not in out["reply"] and saved(client, "s-1").get("pending_action") is None
    assert saved(client, "s-1")["messages"][-1].content == DRAFT_INVALID_REPLY
    assert created_tickets() == []


# ------------------------------------------------------------- ordinary turns are unchanged

def test_an_ordinary_question_has_no_pending_action(client, agent):
    agent(ANSWER)
    out = chat(client, "s-1", "NakhlaConnect shows NX-112 since this morning.")
    assert out["state"] == "answered" and out["pending_action"] is None


def test_an_incident_turn_never_drafts_even_when_asked(client, agent):
    llm = agent(FIELDS)
    out = chat(client, "s-1", "NakhlaConnect keeps failing with NX-503 since this morning. Please open a ticket.")
    assert out["state"] == "incident" and out["pending_action"] is None and llm.calls == []


# ------------------------------------------------------------- saving the draft fails: no misleading history

def test_a_draft_that_cannot_be_saved_leaves_no_misleading_message_in_the_history(client, agent, monkeypatch):
    import app.main as main
    from app.main import DRAFT_SAVE_FAILED_REPLY

    agent(FIELDS)

    async def broken(session_id, session):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(main, "save_approval_state", broken)
    out = chat(client, "s-1", REQUEST)
    assert (out["state"], out["reply"], out["pending_action"]) == ("error", DRAFT_SAVE_FAILED_REPLY, None)
    assert sessions["s-1"]["pending_action"] is None                             # nothing approvable in memory
    state = saved(client, "s-1")
    assert state.get("pending_action") is None and state.get("draft") is None
    assert isinstance(state["messages"][-1], AIMessage)
    assert state["messages"][-1].content == DRAFT_SAVE_FAILED_REPLY               # not "I've prepared a draft"
    assert all(DRAFT_READY_REPLY not in str(m.content) for m in state["messages"])
    assert created_tickets() == []


def test_when_a_new_draft_cannot_be_saved_the_previous_draft_stays_the_approvable_one(client, agent, monkeypatch):
    import app.main as main

    agent(FIELDS, dict(FIELDS, subject="NX-112 again, second draft"))
    first = chat(client, "s-1", REQUEST)["pending_action"]
    real = main.save_approval_state

    async def broken(session_id, session):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(main, "save_approval_state", broken)
    assert chat(client, "s-1", "Still NX-112. Please open a ticket.")["pending_action"] is None
    assert sessions["s-1"]["pending_action"].action_id == first["action_id"]      # the old draft, restored
    monkeypatch.setattr(main, "save_approval_state", real)
    done = approve(client, "s-1", first["action_id"])
    assert done["result"]["ticket_id"].startswith("IT-") and len(created_tickets()) == 1
    assert created_tickets()[0]["subject"] == FIELDS["subject"]                   # the first draft, as approved
