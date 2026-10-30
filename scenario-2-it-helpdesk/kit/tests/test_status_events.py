"""Real-time progress `status` events: emitted by the actual graph stages through the existing custom stream.

Offline: fake LLM / MCP / KB. Statuses are generic, fixed labels; they never carry user, tool or policy data and
never enter the final reply.

    python -m pytest tests/test_status_events.py -v
"""
from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_graph
from app.agent.nodes import STREAM_ANSWER_KEY
from app.agent.streaming import NODE_STATUS, STATUS_LABELS, status_event
from tests.fakes import FakeKnowledgeBase, FakeStreamChat, answer_body, kb_result
from tests.test_agent_graph import FakeMCP, SARA
from tests.test_answer_streaming import GOOD, GOOD_REPLY, KB002, TEXT, run_turn


def statuses(events):
    return [e for e in events if e["type"] == "status"]


def test_status_precedes_the_first_chunk_and_follows_the_real_stages():
    events, final = run_turn(FakeStreamChat(GOOD), with_status=True)
    stages = [e["stage"] for e in statuses(events)]
    assert stages == ["employee", "knowledge_base", "preparing", "generating"]
    first_chunk = next(i for i, e in enumerate(events) if e["type"] == "chunk")
    assert all(e["type"] == "status" for e in events[:first_chunk]) and first_chunk == 4
    assert not statuses(events[first_chunk:])                       # none once the answer is flowing


def test_labels_are_fixed_generic_text_only():
    events, _ = run_turn(FakeStreamChat(GOOD), with_status=True)
    for e in statuses(events):
        assert set(e) == {"type", "stage", "label"} and e == status_event(e["stage"])
        assert e["label"] == STATUS_LABELS[e["stage"]]
        for private in ("sara", "nakhla", "VPN", "KB-002", "mcp", "lookup"):
            assert private.lower() not in e["label"].lower()


def test_status_text_never_reaches_the_final_state():
    events, final = run_turn(FakeStreamChat(GOOD), with_status=True)
    reply = final["messages"][-1].content
    assert answer_body(reply) == GOOD_REPLY
    for label in STATUS_LABELS.values():
        assert label not in reply


def test_no_status_when_streaming_is_off():
    events, final = run_turn(FakeStreamChat(GOOD, json_replies=({"reply": GOOD_REPLY, "citations": ["KB-002"]},)),
                             streaming=False, with_status=True)
    assert events == [] and final["messages"][-1].content


def test_status_is_emitted_before_retrieval_starts():
    """Retrieval blocks until the consumer has received the knowledge-base status: if the status were sent
    after (or at) the embedding call this would time out."""
    seen_status = asyncio.Event()
    order = []

    class GatedKB(FakeKnowledgeBase):
        async def search(self, query, department=None):
            order.append("search_start")
            await asyncio.wait_for(seen_status.wait(), 5)
            order.append("search_released")
            return await super().search(query, department)

    graph = build_graph(InMemorySaver(), FakeMCP(), GatedKB(kb_result(KB002)), FakeStreamChat(GOOD))

    async def go():
        async for mode, payload in graph.astream(
                {"messages": [HumanMessage(content=TEXT)], "user_email": SARA},
                {"configurable": {"thread_id": "t-order", STREAM_ANSWER_KEY: True}},
                stream_mode=["custom", "values"]):
            if mode == "custom" and payload.get("stage") == "knowledge_base":
                order.append("status_received")
                seen_status.set()

    asyncio.run(go())
    assert order.index("status_received") < order.index("search_released")


def test_blocked_policy_turn_gets_only_safe_statuses_and_no_chunks():
    kb = FakeKnowledgeBase(kb_result(KB002, rule_coverage={"department_rules": "complete",
                                                           "governing_policy": "unavailable"}))
    llm = FakeStreamChat(GOOD)
    events, _ = run_turn(llm, kb=kb, with_status=True)
    assert "chunk" not in [e["type"] for e in events] and llm.stream_calls == []
    assert {e["stage"] for e in statuses(events)} <= set(NODE_STATUS.values())
    assert "generating" not in {e["stage"] for e in statuses(events)}


def test_node_status_stages_are_all_defined_labels():
    assert set(NODE_STATUS.values()) <= set(STATUS_LABELS)
