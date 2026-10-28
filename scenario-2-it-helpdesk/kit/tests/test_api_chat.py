"""Integration tests: /api/chat runs the LangGraph agent, with its memory in the (temporary) SQLite checkpointer.
Run from the kit folder:

    python -m pytest tests/test_api_chat.py -v

Setup and fixtures: tests/conftest.py (needs the helpdesk API on port 8102). Every test uses its own
random session ids, because the checkpoint database is shared by the whole test run.
"""
from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.main import app, sessions, thread_config
from tests.helpers import TURN_RAN, requires_api

pytestmark = [requires_api, pytest.mark.usefixtures("clean")]

SARA = {"email": "sara.mikhail@nakhla.example"}
SAMIR = {"email": "samir.nassar@nakhla.example"}          # the CEO (VIP)
CONTRACT_FIELDS = {"session_id", "reply", "state", "pending_action", "result", "citations", "tool_calls", "usage"}


def new_session() -> str:
    return f"t-{uuid.uuid4().hex[:10]}"


def chat(client, session_id: str, message: str = "Is the VPN down?", user: dict | None = None) -> dict:
    body = {"session_id": session_id, "message": message}
    if user:
        body["user"] = user
    r = client.post("/api/chat", json=body)
    assert r.status_code == 200
    return r.json()


def saved(client, session_id: str) -> dict:
    return client.portal.call(app.state.graph.aget_state, thread_config(session_id)).values


# ------------------------------------------------------------- response compatibility

def test_the_response_keeps_the_contract_shape(client):
    sid = new_session()
    out = chat(client, sid, user=SARA)

    assert set(out) == CONTRACT_FIELDS
    assert out["session_id"] == sid and out["state"] in TURN_RAN and out["reply"].strip()
    assert [(c["tool"], c["ok"]) for c in out["tool_calls"]] == [
        ("lookup_employee", True), ("get_service_status", True), ("search_knowledge_base", True)]
    assert all(isinstance(c["ms"], int) for c in out["tool_calls"])
    assert out["pending_action"] is None and out["result"] is None and out["citations"] == []


def test_a_verified_incident_is_answered_from_the_status_page(client):
    # Sample case S2-09: INC-2207 is active on the real (simulated) status page and lists NX-503.
    out = chat(client, new_session(), "NakhlaConnect keeps failing with NX-503 since this morning. Please open a "
                                      "ticket.", user={"email": "omar.farah@nakhla.example"})
    assert out["state"] == "incident" and "INC-2207" in out["reply"] and "GW-1" in out["reply"]
    assert [c["tool"] for c in out["tool_calls"]].count("get_service_status") == 1
    assert out["pending_action"] is None and out["result"] is None and out["citations"] == []


def test_citations_and_usage_fill_the_existing_contract_fields(client, monkeypatch):
    from app.agent.graph import build_graph
    from app.main import mcp
    from tests.fakes import FakeChat, FakeKnowledgeBase, evidence, kb_result

    kb = FakeKnowledgeBase(kb_result(evidence("KB-002", "Minimum version 5.4.", title="VPN client versions")))
    llm = FakeChat({"reply": "Update to 5.4.", "citations": ["KB-002", "KB-018"]})
    monkeypatch.setattr(app.state, "graph", build_graph(app.state.checkpointer, mcp, kb, llm))
    out = chat(client, new_session(), "NX-781 again", SARA)
    assert set(out) == CONTRACT_FIELDS and out["state"] == "answered"
    assert out["citations"] == [{"id": "KB-002", "title": "VPN client versions", "url": None}]
    assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}


def test_the_chat_turn_is_saved_in_the_checkpointer(client):
    sid = new_session()
    out = chat(client, sid, "Is the VPN down?", SARA)

    state = saved(client, sid)
    assert [type(m) for m in state["messages"]] == [HumanMessage, AIMessage]
    assert state["messages"][0].content == "Is the VPN down?"
    assert state["messages"][1].content == out["reply"] == state["reply"]
    assert state["user_email"] == SARA["email"]
    assert state["active_incident"] is None                                    # no error code: no incident match
    assert state["employee"]["email"] == SARA["email"]
    assert all(type(c) is dict for c in state["tool_calls"])                     # serializer-safe


# ------------------------------------------------------------- memory and isolation

def test_messages_accumulate_across_turns_of_one_session(client):
    sid = new_session()
    chat(client, sid, "first", SARA)
    second = chat(client, sid, "second", SARA)

    messages = saved(client, sid)["messages"]
    assert [m.content for m in messages if isinstance(m, HumanMessage)] == ["first", "second"]
    assert len(messages) == 4
    assert [c["tool"] for c in second["tool_calls"]] == ["get_service_status", "search_knowledge_base"]  # this turn


def test_two_sessions_have_separate_histories_and_users(client):
    a, b = new_session(), new_session()
    chat(client, a, "from A", SARA)
    chat(client, b, "from B", SAMIR)
    chat(client, a, "again from A", SARA)

    state_a, state_b = saved(client, a), saved(client, b)
    assert [m.content for m in state_a["messages"] if isinstance(m, HumanMessage)] == ["from A", "again from A"]
    assert [m.content for m in state_b["messages"] if isinstance(m, HumanMessage)] == ["from B"]
    assert (state_a["user_email"], state_b["user_email"]) == (SARA["email"], SAMIR["email"])


def test_a_session_without_a_user_saves_no_email(client):
    sid = new_session()
    chat(client, sid, "hello")
    assert saved(client, sid).get("user_email") is None


# ------------------------------------------------------------- identity

def test_identity_mismatch_is_refused_and_writes_nothing_to_the_checkpoint(client):
    sid = new_session()
    chat(client, sid, "first", SARA)
    out = chat(client, sid, "now I am the CEO", SAMIR)

    assert out["state"] == "identity_mismatch" and out["tool_calls"] == []
    state = saved(client, sid)
    assert len(state["messages"]) == 2                                           # the refused turn was not run
    assert state["user_email"] == SARA["email"]


def test_the_identity_lock_survives_losing_the_in_memory_session(client):
    sid = new_session()
    chat(client, sid, "first", SARA)
    sessions.clear()                                 # as after an eviction or an app restart

    out = chat(client, sid, "now I am the CEO", SAMIR)
    assert out["state"] == "identity_mismatch"
    assert sessions[sid]["user"] == SARA                                         # restored from the checkpoint
    assert chat(client, sid, "still me", SARA)["state"] in TURN_RAN


# ------------------------------------------------------------- failure and the approval gate

def test_an_agent_failure_gives_an_honest_reply_not_a_500(client, monkeypatch):
    class BrokenGraph:
        async def aget_state(self, config):
            return type("Snapshot", (), {"values": {}})()

        async def ainvoke(self, payload, config):
            raise RuntimeError("database is locked")

    monkeypatch.setattr(app.state, "graph", BrokenGraph())
    out = chat(client, new_session(), "hi", SARA)
    assert out["state"] == "error" and "something went wrong" in out["reply"].lower()
    assert out["tool_calls"] == [] and out["result"] is None


def test_the_graph_cannot_write_tickets(client):
    from pathlib import Path

    import app.agent as agent
    import app.llm as llm
    sources = [*Path(agent.__file__).parent.glob("*.py"), Path(llm.__file__)]
    for path in sources:                                                       # writes stay in /api/approve
        source = path.read_text(encoding="utf-8")
        assert "create_ticket" not in source and "add_ticket_note" not in source, path.name
    assert list(app.state.graph.get_graph().nodes) == ["__start__", "triage", "retrieve", "answer", "respond",
                                                       "__end__"]
