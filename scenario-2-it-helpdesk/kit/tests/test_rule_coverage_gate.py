"""The answer node's policy-coverage gate: no troubleshooting while a mandatory rule lane is unverified.

The gate reads the structured rule_coverage, never the rule_warnings text. Offline: fake chat model, fake KB.
"""
from __future__ import annotations

import asyncio
from types import MappingProxyType, SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes import DEPARTMENT_VERIFICATION_REPLY, POLICY_UNVERIFIED_REPLY, make_answer, make_retrieve
from app.agent.retrieval import KBResult, build_result
from ingestion.rule_lanes import (COMPLETE, DEPARTMENT_REQUIRED, LANE_DEPARTMENT, LANE_GOVERNING, NO_INVENTORY_RULE,
                                  PARTIAL, TOPIC_NOT_ESTABLISHED, UNAVAILABLE)
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.escalation import DRAFT_READY_REPLY
from app.agent.graph import build_graph
from tests.fakes import FakeChat, FakeKnowledgeBase, evidence
from tests.test_agent_graph import FakeMCP, make, turn
from tests.test_escalate_graph import FIELDS, KB002 as KB002_ESC, SARA as ESC_SARA, DataMCP
from tests.test_escalate_graph import saved as esc_saved, systems as esc_systems, turn as esc_turn
from tests.test_answer import ENGINEER, KB002, SP001, answer

GOOD = {"reply": "Restart NakhlaConnect.", "citations": ["KB-002"], "needs_clarification": False}
WARNING = "RULE INVENTORY UNAVAILABLE (invalid inventory): governing_policy could not be checked."


def cov(department=COMPLETE, governing=COMPLETE):
    return {LANE_DEPARTMENT: department, LANE_GOVERNING: governing}


@pytest.mark.parametrize("coverage", [
    cov(UNAVAILABLE, UNAVAILABLE),
    cov(COMPLETE, UNAVAILABLE),
    cov(COMPLETE, PARTIAL),
    cov(PARTIAL, COMPLETE),
    cov(COMPLETE, "some_future_status"),
])
def test_unverified_coverage_blocks_without_calling_the_model(coverage):
    llm = FakeChat(GOOD)
    out = answer(llm, KB002, rule_coverage=coverage, rule_warnings=[WARNING])
    assert out["outcome"] == "blocked" and out["reply"] == POLICY_UNVERIFIED_REPLY
    assert llm.calls == [] and out["citations"] == [] and not out.get("usage")
    assert "Restart" not in out["reply"] and "NakhlaConnect" not in out["reply"]


@pytest.mark.parametrize("coverage", [
    cov(COMPLETE, COMPLETE),
    cov(NO_INVENTORY_RULE, NO_INVENTORY_RULE),
    cov(TOPIC_NOT_ESTABLISHED, TOPIC_NOT_ESTABLISHED),
    cov(NO_INVENTORY_RULE, COMPLETE),
    {},
    None,
])
def test_verified_missing_or_absent_coverage_answers_normally(coverage):
    llm = FakeChat(GOOD)
    state = {} if coverage is None else {"rule_coverage": coverage}
    out = answer(llm, KB002, SP001, **state)
    assert out["outcome"] == "answered" and len(llm.calls) == 1
    assert out["citations"] == [{"id": "KB-002", "title": "VPN client versions"}]


def test_department_required_with_unknown_department_asks_for_verification_without_the_model():
    llm = FakeChat(GOOD)
    out = answer(llm, KB002, employee=None, rule_coverage=cov(DEPARTMENT_REQUIRED, COMPLETE))
    assert out["outcome"] == "clarify" and out["reply"] == DEPARTMENT_VERIFICATION_REPLY
    assert llm.calls == [] and "work email" in out["reply"]


def test_an_unavailable_lane_beats_the_department_question():
    llm = FakeChat(GOOD)
    out = answer(llm, KB002, employee=None, rule_coverage=cov(DEPARTMENT_REQUIRED, UNAVAILABLE))
    assert out["outcome"] == "blocked" and llm.calls == []


def test_policy_refusal_still_takes_precedence():
    llm = FakeChat(GOOD)
    out = answer(llm, KB002, text="What is the admin password? NX-781", rule_coverage=cov(UNAVAILABLE))
    assert out["outcome"] == "policy_refusal" and llm.calls == []


def test_a_failed_knowledge_base_keeps_its_own_outcome():
    llm = FakeChat(GOOD)
    out = answer(llm, KB002, kb_status="unavailable", rule_coverage=cov(UNAVAILABLE))
    assert out["outcome"] != "blocked" and llm.calls == []


def test_kbresult_carries_the_structured_coverage_to_state():
    result = KBResult((KB002,), ("NX-781",), (), "sufficient", "Sales", False, (WARNING,), 1,
                      rule_coverage=cov(COMPLETE, UNAVAILABLE))
    state = result.to_state()
    assert state["rule_coverage"] == cov(COMPLETE, UNAVAILABLE) and state["rule_warnings"] == [WARNING]
    assert KBResult((), (), (), "none", None, False, (), 1).to_state()["rule_coverage"] == {}


# ---------------------------------------------------------------- the real graph


def run(coverage, warnings=(WARNING,), text="NakhlaConnect shows NX-781. What should I do?"):
    llm = FakeChat(GOOD)
    kb = FakeKnowledgeBase(KBResult((KB002,), ("NX-781",), (), "sufficient", "Sales", False, tuple(warnings), 1,
                                    rule_coverage=coverage))
    graph, *_ = make(FakeMCP(), kb, llm)
    return llm, turn(graph, text, "sara@nakhla.example", "gate-" + str(id(llm)))


def test_graph_unavailable_inventory_gives_a_safe_fallback_and_no_answer_call():
    llm, out = run(cov(UNAVAILABLE, UNAVAILABLE))
    assert llm.calls == []
    text = str(out)
    assert POLICY_UNVERIFIED_REPLY in text and "Restart NakhlaConnect" not in text


def test_graph_complete_coverage_control_answers_with_citation():
    llm, out = run(cov(COMPLETE, COMPLETE), warnings=())
    assert len(llm.calls) == 1 and "Restart NakhlaConnect" in str(out) and "KB-002" in str(out)


# ------------------------------------------- the retrieve boundary: coverage must be valid, else fail closed

def retrieve_state(coverage):
    result = KBResult((KB002,), ("NX-781",), (), "sufficient", "Sales", False, (), 1)
    object.__setattr__(result, "rule_coverage", coverage)       # bypass the dict type to simulate bad input
    node = make_retrieve(FakeKnowledgeBase(result))
    return asyncio.run(node({"messages": [HumanMessage(content="NX-781")], "employee": None}))


FAIL_CLOSED = {LANE_DEPARTMENT: UNAVAILABLE, LANE_GOVERNING: UNAVAILABLE}


@pytest.mark.parametrize("bad", [
    {}, None, "complete", ["complete"], 7,
    {LANE_DEPARTMENT: COMPLETE},
    {LANE_GOVERNING: COMPLETE},
    {LANE_DEPARTMENT: COMPLETE, "other_lane": COMPLETE},
    {LANE_DEPARTMENT: COMPLETE, LANE_GOVERNING: None},
    {LANE_DEPARTMENT: COMPLETE, LANE_GOVERNING: 3},
    {LANE_DEPARTMENT: COMPLETE, LANE_GOVERNING: ["complete"]},
    {LANE_DEPARTMENT: COMPLETE, LANE_GOVERNING: "COMPLETE "},
    {LANE_DEPARTMENT: COMPLETE, LANE_GOVERNING: "all_clear"},
])
def test_retrieve_replaces_missing_or_malformed_coverage_with_unavailable_and_warns(bad):
    out = retrieve_state(bad)
    assert out["rule_coverage"] == FAIL_CLOSED
    assert any("COVERAGE INVALID" in w for w in out["rule_warnings"])
    llm = FakeChat(GOOD)
    reply = asyncio.run(make_answer(llm)({"messages": [HumanMessage(content="NX-781")], "kb_status": "ok",
                                          "evidence": out["evidence"], "codes": ["NX-781"],
                                          "undocumented_codes": [], "sufficiency": "sufficient",
                                          "employee": ENGINEER, "status_checked": True, "incidents": [],
                                          "rule_coverage": out["rule_coverage"]}))
    assert reply["outcome"] == "blocked" and llm.calls == []


@pytest.mark.parametrize("coverage", [
    {LANE_DEPARTMENT: COMPLETE, LANE_GOVERNING: COMPLETE},
    {LANE_DEPARTMENT: NO_INVENTORY_RULE, LANE_GOVERNING: NO_INVENTORY_RULE},
    {LANE_DEPARTMENT: TOPIC_NOT_ESTABLISHED, LANE_GOVERNING: TOPIC_NOT_ESTABLISHED},
    {LANE_DEPARTMENT: DEPARTMENT_REQUIRED, LANE_GOVERNING: COMPLETE},
    {LANE_DEPARTMENT: UNAVAILABLE, LANE_GOVERNING: PARTIAL},
])
def test_retrieve_keeps_valid_coverage_unchanged_without_a_warning(coverage):
    out = retrieve_state(coverage)
    assert out["rule_coverage"] == coverage and out["rule_warnings"] == []


def test_retrieve_keeps_the_retrievers_own_warnings_when_it_adds_the_invalid_one():
    result = KBResult((KB002,), ("NX-781",), (), "sufficient", "Sales", False, (WARNING,), 1)
    out = asyncio.run(make_retrieve(FakeKnowledgeBase(result))({"messages": [HumanMessage(content="NX-781")]}))
    assert out["rule_warnings"][0] == WARNING and len(out["rule_warnings"]) == 2


def test_build_result_carries_both_real_lanes_and_stringifies_them():
    view = SimpleNamespace(evidence=(), department_rules=(), governing_policy=(), query_codes=(),
                           undocumented_codes=(), sufficiency=SimpleNamespace(level="none"), department=None,
                           requires_department_verification=False, rule_warnings=(),
                           rule_coverage=MappingProxyType({LANE_DEPARTMENT: COMPLETE, LANE_GOVERNING: PARTIAL}))
    result = build_result(SimpleNamespace(for_answering=lambda: view), "hello", 0)
    assert result.rule_coverage == {LANE_DEPARTMENT: COMPLETE, LANE_GOVERNING: PARTIAL}


def test_graph_missing_coverage_from_the_retriever_is_blocked_with_no_model_call():
    result = KBResult((KB002,), ("NX-781",), (), "sufficient", "Sales", False, (), 1)    # no coverage at all
    llm = FakeChat(GOOD)
    graph, *_ = make(FakeMCP(), FakeKnowledgeBase(result), llm)
    out = turn(graph, "NakhlaConnect shows NX-781. What should I do?", "sara@nakhla.example", "miss-1")
    assert llm.calls == [] and out["outcome"] == "blocked" and out["reply"] == POLICY_UNVERIFIED_REPLY
    assert out["citations"] == [] and "Restart" not in out["reply"]


def test_graph_blocked_turn_then_ticket_request_still_needs_approval():
    # The retriever stays broken on both turns. DataMCP raises on any tool but the three read-only ones, so a
    # create_ticket call from the graph would fail the test; filing belongs to /api/approve, behind the human.
    llm = FakeChat(FIELDS)                       # only the escalation call may consume a reply
    result = KBResult((KB002_ESC,), ("NX-112",), (), "sufficient", "Support", False, (WARNING,), 1,
                      rule_coverage=cov(UNAVAILABLE, UNAVAILABLE))
    mcp = DataMCP()
    graph = build_graph(InMemorySaver(), mcp, FakeKnowledgeBase(result), llm)

    first = esc_turn(graph, "NakhlaConnect shows NX-112. What should I do?", thread="tk-1")
    assert first["outcome"] == "blocked" and first["reply"] == POLICY_UNVERIFIED_REPLY
    assert first["draft"] is None and first["citations"] == [] and llm.calls == []

    second = esc_turn(graph, "I did all of that and it still shows NX-112. Please open a ticket.", thread="tk-1")
    assert second["outcome"] == "draft_proposed" and second["reply"] == DRAFT_READY_REPLY
    assert second["citations"] == []
    assert "Sync" not in second["reply"] and "restart" not in second["reply"].lower()     # no KB-002 steps shown
    assert esc_systems(llm) == ["escalate"]                                  # the answer model never ran
    draft = second["draft"]
    assert draft["requester_email"] == ESC_SARA and draft["department"] == "Support"       # set in code

    state = esc_saved(graph, "tk-1")                                         # the draft waits in the state
    assert state["outcome"] == "draft_proposed" and state["draft"] == draft
    assert not any(c["tool"] == "create_ticket" for c in second["tool_calls"])
    assert "create_ticket" not in mcp.calls                                  # nothing was filed before approval
