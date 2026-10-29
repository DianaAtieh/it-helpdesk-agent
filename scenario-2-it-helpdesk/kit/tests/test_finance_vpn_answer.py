"""Fix B: a Finance employee's VPN problem gets the fixed Security Operations reply (KB-002 "Exceptions", KB-016).

Offline graph tests with the simulated directory and status page (tests/test_escalate_graph.py) and a fake model.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.agent import nodes
from app.agent.escalation import ASK_STEPS_REPLY, FINANCE_VPN_REPLY, FINANCE_VPN_STEP
from app.agent.nodes import FINANCE_VPN_DOCS, FINANCE_VPN_OUTCOME, valid_finance_vpn_citations
from tests.fakes import FakeChat, FakeKnowledgeBase, evidence, kb_result
from tests.test_escalate_graph import FADI, FIELDS, SARA, ZIAD, make, systems, turn

KB = Path(__file__).resolve().parents[1] / "dataset" / "knowledge_base"
KARIM = "karim.saba@nakhla.example"          # Engineering
KB002_781 = evidence("KB-002", "NX-781: client below the minimum 5.4. Update to 5.4.2 from Software Center.",
                     title="VPN error codes and fixes")
ANSWER_781 = {"reply": "Update NakhlaConnect to 5.4.2 from Software Center.", "citations": ["KB-002"],
              "needs_clarification": False}
SECURITY_FIELDS = dict(FIELDS, subject="NX-781 on the Finance VPN", category="network_access", priority="P4",
                       attempted_steps=[])


def finance_graph(*replies):
    return make(*(replies or (ANSWER_781,)), kb=FakeKnowledgeBase(kb_result(KB002_781, codes=("NX-781",))))


def assert_fixed_reply(out, llm):
    assert out["outcome"] == FINANCE_VPN_OUTCOME and out["reply"] == FINANCE_VPN_REPLY
    assert [c["id"] for c in out["citations"]] == ["KB-002", "KB-016"]
    assert out.get("draft") is None and llm.calls == []                    # no model call, no ticket


# ------------------------------------------------------------------ Finance VPN problems: the fixed reply

@pytest.mark.parametrize("text", [
    "NakhlaConnect shows NX-781. How do I update the client?",
    "NakhlaConnect shows NX-403 when I connect from home.",
    "My VPN shows NX-112.",
    "My VPN keeps disconnecting on the FIN-SECURE profile.",
])
def test_a_finance_vpn_problem_gets_the_security_operations_reply_without_the_model(text):
    graph, _, _, llm = finance_graph()
    assert_fixed_reply(turn(graph, text, email=FADI), llm)


def test_the_reply_routes_to_security_operations_and_gives_no_client_steps():
    for phrase in ("Security Operations", "FIN-SECURE", "GW-3"):
        assert phrase in FINANCE_VPN_REPLY
    assert "don't update, reinstall or reconfigure" in FINANCE_VPN_REPLY and "please open a ticket" in FINANCE_VPN_REPLY
    assert "Nothing is filed until you approve" in FINANCE_VPN_REPLY
    assert "5.4" not in FINANCE_VPN_REPLY and "Software Center" not in FINANCE_VPN_REPLY


# ------------------------------------------------------------------ the ticket flow stays as it is

@pytest.mark.parametrize("follow_up", ["please open a ticket", "Yes please"])
def test_answering_the_reply_gives_the_security_draft_and_files_nothing(follow_up):
    graph, _, _, llm = finance_graph(SECURITY_FIELDS)
    turn(graph, "NakhlaConnect shows NX-781. How do I update the client?", email=FADI, thread="f-1")
    out = turn(graph, follow_up, email=FADI, thread="f-1")
    assert out["outcome"] == "draft_proposed" and out["reply"] != ASK_STEPS_REPLY
    assert out["draft"]["category"] == "security" and out["draft"]["priority"] == "P2"
    assert FINANCE_VPN_STEP in out["draft"]["attempted_steps"] and systems(llm) == ["escalate"]


def test_an_explicit_finance_ticket_request_is_unchanged():
    graph, _, _, llm = finance_graph(SECURITY_FIELDS)
    out = turn(graph, "NakhlaConnect shows NX-403 when I connect from home. If it needs a ticket, prepare it for me.",
               email=FADI)
    assert out["outcome"] == "draft_proposed" and out["draft"]["category"] == "security"
    assert out["draft"]["priority"] == "P2" and systems(llm) == ["escalate"]


def test_a_non_finance_ticket_request_after_any_reply_still_needs_steps():
    graph, *_ = finance_graph(SECURITY_FIELDS)
    out = turn(graph, "NakhlaConnect shows NX-781. Please open a ticket.", email=KARIM)
    assert out["reply"] == ASK_STEPS_REPLY and out.get("draft") is None


# ------------------------------------------------------------------ everything else is unchanged

def test_other_departments_still_get_the_model_answer():
    graph, _, _, llm = finance_graph()
    out = turn(graph, "NakhlaConnect shows NX-781. How do I update the client?", email=KARIM)
    assert out["outcome"] == "answered" and systems(llm) == ["answer"] and "5.4.2" in out["reply"]


def test_a_finance_problem_that_is_not_vpn_still_gets_the_model_answer():
    swc = evidence("KB-009", "SWC-1603: install failed. Restart the laptop, keep 10 GB free, try again.",
                   title="Software Center")
    answer = {"reply": "Restart the laptop, keep 10 GB free, then try again.", "citations": ["KB-009"],
              "needs_clarification": False}
    graph, _, _, llm = make(answer, kb=FakeKnowledgeBase(kb_result(swc, codes=("SWC-1603",))))
    out = turn(graph, "Software Center shows SWC-1603 when I install an update.", email=FADI)
    assert out["outcome"] == "answered" and systems(llm) == ["answer"]


def test_an_active_incident_still_comes_first_for_finance():
    graph, _, _, llm = finance_graph()
    out = turn(graph, "NakhlaConnect fails with NX-503 since this morning.", email=FADI)
    assert out["outcome"] == "incident" and llm.calls == []


def test_a_security_policy_refusal_still_comes_first_for_finance():
    graph, *_ = finance_graph()
    out = turn(graph, "NakhlaConnect shows NX-781. How do I turn off the posture check so I can connect?", email=FADI)
    assert out["outcome"] == "policy_refusal"


def test_a_contractor_vpn_request_is_still_the_contractor_refusal():
    graph, *_ = finance_graph()
    out = turn(graph, "How do I set up NakhlaConnect VPN on my laptop so I can work from home?", email=ZIAD)
    assert out["outcome"] == "policy_refusal" and out["reply"] != FINANCE_VPN_REPLY


def test_a_support_vpn_problem_is_not_affected():
    graph, _, _, llm = finance_graph()
    out = turn(graph, "NakhlaConnect shows NX-781. How do I update the client?", email=SARA)
    assert out["outcome"] != FINANCE_VPN_OUTCOME and systems(llm) == ["answer"]


# ------------------------------------------------------------------ citations

def test_only_code_issued_kb_002_and_kb_016_citations_survive():
    rule = nodes.DEPARTMENT_RULE_SOURCE
    raw = [{"id": "KB-002", "source": rule}, {"id": "KB-011", "source": rule}, {"id": "KB-016", "source": "model"},
           {"id": "KB-002", "source": rule}, None]
    assert valid_finance_vpn_citations(raw) == [{"id": "KB-002", "title": FINANCE_VPN_DOCS["KB-002"],
                                                 "source": nodes.DEPARTMENT_RULE_SOURCE}]


def test_the_citation_titles_match_the_kb_documents():
    for doc_id, path in (("KB-002", "KB-002-vpn-error-codes-and-fixes.md"), ("KB-016", "KB-016-department-matrix.md")):
        title = re.search(r"^title:\s*\"?(.+?)\"?\s*$", (KB / path).read_text(encoding="utf-8"), re.M).group(1)
        assert FINANCE_VPN_DOCS[doc_id] == title


def test_the_kb_still_states_the_finance_exception():
    kb002 = (KB / "KB-002-vpn-error-codes-and-fixes.md").read_text(encoding="utf-8")
    kb016 = (KB / "KB-016-department-matrix.md").read_text(encoding="utf-8")
    assert "go **straight to Security Operations**" in kb002 and "FIN-SECURE profile on GW-3" in kb002
    assert "Do not walk Finance staff through" in kb002 and "Do not troubleshoot the client with Finance staff" in kb016
