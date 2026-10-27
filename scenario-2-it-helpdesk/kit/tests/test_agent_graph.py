"""Unit tests for the agent graph (app/agent/), isolated from FastAPI, MCP and the network. Run from the kit folder:

    python -m pytest tests/test_agent_graph.py -v

A fake MCP client returns canned status-page data; the checkpointer is LangGraph's InMemorySaver
(it serializes state like the SQLite saver does). No helpdesk API needed.
"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent.graph import build_graph
from app.agent.nodes import describe_status
from app.schemas import ToolCall

INC_2207 = {"id": "INC-2207", "status": "active", "priority": "P1", "service": "vpn",
            "title": "VPN gateway GW-2 degraded", "workaround": "Select GW-1 (Jerusalem), then reconnect."}
STATUS_OK = {"ok": True, "status": {"services": [{"id": "vpn", "status": "degraded"}], "incidents": [
    {"id": "INC-2198", "status": "resolved", "priority": "P3", "title": "Delayed inbound external email"},
    INC_2207,
]}}
STATUS_DOWN = {"ok": False, "error": {"code": "service_unavailable", "message": "Simulated outage."}}


class FakeMCP:
    """Same interface as app.mcp_client.HelpdeskMCP.call; records every call."""

    def __init__(self, data: dict):
        self.data = data
        self.calls: list[tuple[str, dict | None]] = []

    async def call(self, tool: str, args: dict | None = None) -> tuple[dict, ToolCall]:
        self.calls.append((tool, args))
        return self.data, ToolCall(tool=tool, ok=bool(self.data.get("ok")), ms=7)


def config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def turn(graph, thread_id: str, text: str, email: str | None = None) -> dict:
    payload = {"messages": [HumanMessage(content=text)], "user_email": email}
    return asyncio.run(graph.ainvoke(payload, config(thread_id)))


def saved(graph, thread_id: str) -> dict:
    return asyncio.run(graph.aget_state(config(thread_id))).values


# ------------------------------------------------------------- reply and tool calls

def test_check_status_replies_with_the_status_page_and_records_the_tool_call():
    mcp = FakeMCP(STATUS_OK)
    out = turn(build_graph(InMemorySaver(), mcp), "t-1", "Is the VPN down?", "sara.mikhail@nakhla.example")

    assert mcp.calls == [("get_service_status", None)]
    assert out["reply"] == describe_status(STATUS_OK)
    assert "INC-2207" in out["reply"] and "INC-2198" not in out["reply"]       # resolved incidents are left out
    assert out["tool_calls"] == [{"tool": "get_service_status", "ok": True, "ms": 7}]
    assert out["active_incident"] == INC_2207
    assert out["user_email"] == "sara.mikhail@nakhla.example"


def test_a_failed_status_call_is_reported_honestly():
    out = turn(build_graph(InMemorySaver(), FakeMCP(STATUS_DOWN)), "t-1", "Is the VPN down?")

    assert out["reply"] == describe_status(STATUS_DOWN)
    assert "could not check" in out["reply"]
    assert out["tool_calls"] == [{"tool": "get_service_status", "ok": False, "ms": 7}]
    assert out["active_incident"] is None


def test_state_holds_only_checkpoint_safe_values():
    out = turn(build_graph(InMemorySaver(), FakeMCP(STATUS_OK)), "t-1", "hi")
    assert all(type(call) is dict for call in out["tool_calls"])                  # not ToolCall objects
    assert isinstance(out["reply"], str) and isinstance(out["active_incident"], dict)


# ------------------------------------------------------------- memory across turns

def test_messages_accumulate_across_two_turns_of_the_same_thread():
    graph = build_graph(InMemorySaver(), FakeMCP(STATUS_OK))
    turn(graph, "t-1", "Is the VPN down?")
    second = turn(graph, "t-1", "And now?")

    messages = saved(graph, "t-1")["messages"]
    assert [type(m) for m in messages] == [HumanMessage, AIMessage, HumanMessage, AIMessage]
    assert [m.content for m in messages[::2]] == ["Is the VPN down?", "And now?"]
    assert second["tool_calls"] == [{"tool": "get_service_status", "ok": True, "ms": 7}]   # this turn only


def test_two_threads_never_see_each_others_messages():
    graph = build_graph(InMemorySaver(), FakeMCP(STATUS_OK))
    turn(graph, "thread-a", "message from A", "a@nakhla.example")
    turn(graph, "thread-b", "message from B", "b@nakhla.example")
    turn(graph, "thread-a", "second message from A")

    a, b = saved(graph, "thread-a"), saved(graph, "thread-b")
    assert [m.content for m in a["messages"] if isinstance(m, HumanMessage)] == ["message from A",
                                                                                "second message from A"]
    assert [m.content for m in b["messages"] if isinstance(m, HumanMessage)] == ["message from B"]
    assert b["user_email"] == "b@nakhla.example"


def test_an_unknown_thread_has_no_saved_state():
    graph = build_graph(InMemorySaver(), FakeMCP(STATUS_OK))
    turn(graph, "t-1", "hi")
    assert saved(graph, "never-used") == {}


# ------------------------------------------------------------- the real SQLite saver

def test_the_state_round_trips_through_the_sqlite_saver_without_serializer_warnings(caplog):
    async def run() -> dict:
        async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
            await checkpointer.setup()
            graph = build_graph(checkpointer, FakeMCP(STATUS_OK))
            await graph.ainvoke({"messages": [HumanMessage(content="Is the VPN down?")]}, config("t-1"))
            await graph.ainvoke({"messages": [HumanMessage(content="And now?")]}, config("t-1"))
            return (await graph.aget_state(config("t-1"))).values

    with caplog.at_level(logging.WARNING, logger="langgraph"):
        values = asyncio.run(run())

    assert len(values["messages"]) == 4
    assert values["tool_calls"] == [{"tool": "get_service_status", "ok": True, "ms": 7}]
    assert values["active_incident"] == INC_2207
    assert not [r.getMessage() for r in caplog.records if r.name.startswith("langgraph")]
