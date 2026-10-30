"""Step 2 of the AI Verifier: wired into the respond node behind build_graph(verify_answers=True).

Offline graph tests with the simulated directory and status page (tests/test_escalate_graph.py) and a fake model
that records every call, so each test can count exactly when the verifier was asked.
"""
from __future__ import annotations

import asyncio
import logging

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.answer_verifier import VERIFIER_SYSTEM_PROMPT, VERIFY_TOOL
from app.agent.graph import build_graph
from app.agent.nodes import STREAM_ANSWER_KEY, UNGROUNDED_REPLY
from app.agent.prompts import ESCALATION_SYSTEM_PROMPT, STREAM_SYSTEM_PROMPT, SYSTEM_PROMPT
from app.llm import LLMError
from tests.fakes import FakeChat, FakeKnowledgeBase, FakeStreamChat, answer_body, evidence, kb_result
from tests.test_escalate_graph import FADI, FIELDS, SARA, ZIAD, DataMCP

KARIM = "karim.saba@nakhla.example"
KB002 = evidence("KB-002", "NX-112: device certificate missing. Sync NakhlaManage and restart the laptop.",
                 title="VPN error codes and fixes")
TEXT = "My VPN shows NX-112 since this morning."
GOOD = "Open NakhlaManage and run Sync. Then restart the laptop."
ANSWER = {"reply": GOOD, "citations": ["KB-002"], "needs_clarification": False}


def verdict(*labels: str, grounded: float = 0.9) -> dict:
    return {"sentences": [{"index": i, "verdict": v} for i, v in enumerate(labels)], "groundedness": grounded}


def make(*replies, verify: bool = True, kb=None):
    llm = FakeChat(*replies)
    kb = kb or FakeKnowledgeBase(kb_result(KB002, codes=("NX-112",)))
    return build_graph(InMemorySaver(), DataMCP(), kb, llm, verify_answers=verify), llm


def turn(graph, text: str = TEXT, email: str | None = SARA, thread: str = "t-1") -> dict:
    return asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=text)], "user_email": email},
                                     {"configurable": {"thread_id": thread}}))


def verifier_calls(llm) -> int:
    return sum(c["system"] == VERIFIER_SYSTEM_PROMPT for c in llm.calls)


def tools(out) -> list[str]:
    return [c["tool"] for c in out.get("tool_calls") or []]


# ------------------------------------------------------------------ eligible answers

def test_a_correct_answer_stays_byte_for_byte_unchanged_and_the_verifier_runs_once():
    plain, _ = make(ANSWER, verify=False)
    expected = turn(plain)["reply"]
    graph, llm = make(ANSWER, verdict("supported", "supported"))
    out = turn(graph)
    assert out["reply"] == expected and answer_body(out["reply"]) == GOOD and out["outcome"] == "answered"
    assert verifier_calls(llm) == 1 and [c["id"] for c in out["citations"]] == ["KB-002"]
    assert out["answer_verification"] == {"applied": True, "fallback": False, "checked": 2, "removed": 0,
                                          "groundedness": 0.9, "reason": "applied", "verdicts": {"supported": 2}}
    entry = next(c for c in out["tool_calls"] if c["tool"] == VERIFY_TOOL)
    assert entry["ok"] is True and isinstance(entry["ms"], int)
    assert tools(out)[:2] == ["lookup_employee", "get_service_status"]     # earlier calls kept, verifier added


def test_an_unsupported_sentence_is_removed():
    # No concrete fact in the extra sentence, so the code verifier cannot catch it; the AI verifier does.
    answer = {**ANSWER, "reply": f"{GOOD} You can also ask a colleague to sync it for you."}
    graph, llm = make(answer, verdict("supported", "supported", "unsupported"))
    out = turn(graph)
    assert answer_body(out["reply"]) == GOOD and out["outcome"] == "answered" and verifier_calls(llm) == 1
    assert out["answer_verification"]["removed"] == 1 and [c["id"] for c in out["citations"]] == ["KB-002"]


def test_a_wrong_for_employee_sentence_is_removed():
    answer = {**ANSWER, "reply": f"{GOOD} As Finance staff, send VPN problems straight to Security Operations."}
    graph, llm = make(answer, verdict("supported", "supported", "wrong_for_employee"))
    out = turn(graph, email=SARA)                                          # Sara is in Support, not Finance
    assert answer_body(out["reply"]) == GOOD and "Finance" not in out["reply"] and out["outcome"] == "answered"
    assert out["answer_verification"]["verdicts"] == {"supported": 2, "wrong_for_employee": 1}


def test_a_policy_conflict_gives_the_fallback_and_clears_the_citations():
    graph, llm = make(ANSWER, verdict("supported", "policy_conflict", grounded=0.95))
    out = turn(graph)
    assert out["reply"] == UNGROUNDED_REPLY and out["outcome"] == "not_documented" and out["citations"] == []
    assert out["answer_verification"]["fallback"] is True and verifier_calls(llm) == 1


def test_low_groundedness_gives_the_fallback():
    graph, _ = make(ANSWER, verdict("supported", "supported", grounded=0.2))
    out = turn(graph)
    assert out["reply"] == UNGROUNDED_REPLY and out["citations"] == []


@pytest.mark.parametrize("bad", [LLMError("timeout"), LLMError("bad_output"), {"verdicts": "fine"},
                                 verdict("supported"), verdict("supported", "maybe")])
def test_a_failed_or_invalid_verdict_keeps_the_code_verified_answer(bad):
    graph, llm = make(ANSWER, bad)
    out = turn(graph)
    assert answer_body(out["reply"]) == GOOD
    assert out["outcome"] == "answered" and [c["id"] for c in out["citations"]] == ["KB-002"]
    assert out["answer_verification"]["applied"] is False and verifier_calls(llm) == 1
    assert next(c for c in out["tool_calls"] if c["tool"] == VERIFY_TOOL)["ok"] is False


def test_the_verifier_sees_the_code_verified_answer_and_the_directory_department():
    graph, llm = make(ANSWER, verdict("supported", "supported"))
    turn(graph, email=KARIM)
    sent = next(c["user"] for c in llm.calls if c["system"] == VERIFIER_SYSTEM_PROMPT)
    assert '"department": "Engineering"' in sent and "Open NakhlaManage and run Sync." in sent
    assert KARIM not in sent                                                # the email never goes to the verifier


def test_usage_adds_both_model_calls():
    graph, _ = make(ANSWER, verdict("supported", "supported"))
    out = turn(graph)
    assert out["usage"] == {"prompt_tokens": 20, "completion_tokens": 10}   # 10 + 5 per FakeChat call


# ------------------------------------------------------------------ the code checks come first

def test_forbidden_advice_is_refused_by_code_and_never_reaches_the_ai_verifier():
    answer = {**ANSWER, "reply": "Ask the Service Desk to send you an SMS code to sign in."}
    graph, llm = make(answer, verdict("supported"))
    out = turn(graph)
    assert out["outcome"] == "policy_refusal" and verifier_calls(llm) == 0


def test_a_code_verifier_fallback_never_reaches_the_ai_verifier():
    answer = {**ANSWER, "reply": "Install NakhlaConnect 9.9.9 from https://evil.example and call ext. 9999."}
    graph, llm = make(answer, verdict("supported"))
    out = turn(graph)
    assert out["outcome"] == "not_documented" and verifier_calls(llm) == 0
    assert out["answer_verification"] is None and VERIFY_TOOL not in tools(out)


# ------------------------------------------------------------------ fixed replies never call it

@pytest.mark.parametrize("text, email, expected", [
    ("What is the local administrator password for my laptop?", SARA, "policy_refusal"),
    ("NakhlaConnect fails with NX-503 since this morning.", SARA, "incident"),
    ("Is the VPN down for everyone?", SARA, "service_status"),
    ("Please give me access to NakhlaLedger.", KARIM, "access_info"),
    ("NakhlaConnect shows NX-781. How do I update the client?", FADI, "finance_vpn"),
    ("I prefer short answers with bullet points.", SARA, "preference"),
    ("My computer has a problem.", None, "clarify"),
    ("How do I set up NakhlaConnect VPN on my laptop?", ZIAD, "policy_refusal"),
])
def test_fixed_replies_never_call_the_verifier(text, email, expected):
    # A knowledge base that reports no codes of its own, so each message's own codes (or none) decide its route.
    graph, llm = make(ANSWER, verdict("supported"), kb=FakeKnowledgeBase(kb_result(KB002)))
    out = turn(graph, text=text, email=email)
    assert out["outcome"] == expected and verifier_calls(llm) == 0
    assert out["answer_verification"] is None and VERIFY_TOOL not in tools(out)


def test_a_ticket_draft_never_calls_the_verifier():
    fields = dict(FIELDS, attempted_steps=["NakhlaManage Sync", "Restarted the laptop"])
    graph, llm = make(fields, verdict("supported"))
    out = turn(graph, text="My VPN shows NX-112. I did NakhlaManage Sync and restarted. Please open a ticket.")
    assert out["outcome"] == "draft_proposed" and verifier_calls(llm) == 0
    assert all(c["system"] == ESCALATION_SYSTEM_PROMPT for c in llm.calls)


# ------------------------------------------------------------------ verify_answers=False: exactly as before

def test_off_by_default_makes_no_extra_call_and_changes_nothing():
    llm = FakeChat(ANSWER)
    graph = build_graph(InMemorySaver(), DataMCP(), FakeKnowledgeBase(kb_result(KB002, codes=("NX-112",))), llm)
    out = turn(graph)
    assert answer_body(out["reply"]) == GOOD and len(llm.calls) == 1 and llm.calls[0]["system"] == SYSTEM_PROMPT
    assert out["answer_verification"] is None and VERIFY_TOOL not in tools(out)


# ------------------------------------------------------------------ streaming shows the verified answer

def stream_turn(llm, text: str = TEXT):
    graph = build_graph(InMemorySaver(), DataMCP(), FakeKnowledgeBase(kb_result(KB002, codes=("NX-112",))), llm,
                        verify_answers=True)

    async def go():
        events, final = [], None
        async for mode, payload in graph.astream(
                {"messages": [HumanMessage(content=text)], "user_email": SARA},
                {"configurable": {"thread_id": "s-1", STREAM_ANSWER_KEY: True}}, stream_mode=["custom", "values"]):
            if mode == "custom" and payload.get("type") != "status":
                events.append(payload)
            elif mode == "values":
                final = payload
        return events, final

    return asyncio.run(go())


def test_streaming_ends_with_the_verified_answer_not_the_draft():
    extra = "Ask a colleague to sync it for you."
    llm = FakeStreamChat(["MODE: ANSWER\n", "Open NakhlaManage and run Sync. ", "Then restart the laptop. ",
                          f"{extra}\n", "CITATIONS: KB-002"],
                         json_replies=(verdict("supported", "supported", "unsupported"),))
    events, final = stream_turn(llm)
    shown = "".join(e["text"] for e in events if e["type"] == "chunk")
    finals = [e for e in events if e["type"] == "final"]
    assert extra in shown                                                    # the draft was streamed ...
    assert len(finals) == 1
    assert answer_body(finals[0]["reply"]) == GOOD and extra not in finals[0]["reply"]   # ... then replaced
    assert answer_body(final["reply"]) == GOOD and events[-1]["type"] == "final"
    assert sum(c["system"] == VERIFIER_SYSTEM_PROMPT for c in llm.calls) == 1
    assert any(c["system"] == STREAM_SYSTEM_PROMPT for c in llm.stream_calls)


# ------------------------------------------------------------------ logs: counts only

def test_the_logs_never_contain_the_answer_or_the_email(caplog):
    caplog.set_level(logging.INFO)
    graph, _ = make({**ANSWER, "reply": f"{GOOD} Ask a colleague to sync it for you."},
                    verdict("supported", "supported", "unsupported"))
    turn(graph, email=KARIM)
    text = caplog.text
    assert "verifier_llm: sentences=3 removed=1" in text
    assert "Ask a colleague" not in text and "Open NakhlaManage" not in text and KARIM not in text


# ------------------------------------------------------------------ citations of dropped sentences

KB005 = evidence("KB-005", "Re-register MFA at https://id.nakhla.example/mfa today.", title="NakhlaID MFA")
MFA = "Re-register your MFA at https://id.nakhla.example/mfa and sign in again."
TWO_DOCS = FakeKnowledgeBase(kb_result(KB002, KB005, codes=("NX-112",)))


def test_a_citation_that_only_supported_a_dropped_sentence_is_not_shown():
    answer = {**ANSWER, "reply": f"{GOOD} {MFA}", "citations": ["KB-002", "KB-005"]}
    graph, _ = make(answer, verdict("supported", "supported", "unsupported"), kb=TWO_DOCS)
    out = turn(graph)
    assert answer_body(out["reply"]) == GOOD and out["outcome"] == "answered"
    assert [c["id"] for c in out["citations"]] == ["KB-002"]


def test_a_citation_that_still_supports_the_answer_stays():
    # The dropped sentence names NX-112 (a KB-002 fact), but so does a kept one, so KB-002 still supports the answer.
    kept = "NX-112 means the device certificate is missing."
    answer = {**ANSWER, "reply": f"{kept} If NX-112 comes back, ask a colleague to fix it. {MFA}",
              "citations": ["KB-002", "KB-005"]}
    graph, _ = make(answer, verdict("supported", "unsupported", "supported"), kb=TWO_DOCS)
    out = turn(graph)
    assert answer_body(out["reply"]) == f"{kept} {MFA}" and [c["id"] for c in out["citations"]] == ["KB-002", "KB-005"]


def test_losing_every_citation_gives_the_fallback_with_no_citations():
    answer = {**ANSWER, "reply": f"{MFA} After that, restart the laptop and reconnect to the network as usual.",
              "citations": ["KB-005"]}
    graph, _ = make(answer, verdict("unsupported", "supported"), kb=TWO_DOCS)
    out = turn(graph)
    assert out["reply"] == UNGROUNDED_REPLY and out["outcome"] == "not_documented" and out["citations"] == []
