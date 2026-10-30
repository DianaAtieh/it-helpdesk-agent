"""Buffered live streaming of the answer: the LLM client stream, the chunk parser/checks and the graph events.

Offline: fake streaming models only. The WebSocket layer is not part of this phase; these tests read the
graph's custom stream events (chunk / retract / final) the way that layer will.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import evidence_eval as ev
from app.agent.graph import build_graph
from app.agent.nodes import (LLM_FAILED_REPLY, STREAM_ANSWER_KEY, UNGROUNDED_REPLY, stream_violation)
from app.agent.prompts import STREAM_SYSTEM_PROMPT, SYSTEM_PROMPT
from app.agent.streaming import AnswerStreamParser, reconcile
from app.llm import LLMError, OpenRouterChat
from tests.fakes import FakeChat, FakeKnowledgeBase, FakeStreamChat, answer_body, evidence, kb_result
from tests.test_agent_graph import FakeMCP, SARA
from tests.test_escalate_graph import DataMCP

KB002 = evidence("KB-002", "NakhlaConnect minimum supported version is 5.4.", title="VPN client versions")
TEXT = "My VPN keeps disconnecting"
GOOD = ["MODE: ANSWER\n", "Update NakhlaConnect ", "to 5.4 from Software Center. ", "Then restart it.\n",
        "CITATIONS: KB-002"]
GOOD_REPLY = "Update NakhlaConnect to 5.4 from Software Center. Then restart it."


def run_turn(llm, kb=None, text=TEXT, streaming=True, evaluate=False, thread="t-1", mcp=None, with_status=False):
    """Run one turn through the graph; return (custom events, final state). `status` progress events are left
    out unless with_status: these tests are about chunk / retract / final."""
    graph = build_graph(InMemorySaver(), mcp or FakeMCP(), kb or FakeKnowledgeBase(kb_result(KB002)), llm,
                        evaluate_evidence=evaluate)
    configurable = {"thread_id": thread, **({STREAM_ANSWER_KEY: True} if streaming else {})}

    async def go():
        events, final = [], None
        async for mode, payload in graph.astream({"messages": [HumanMessage(content=text)], "user_email": SARA},
                                                 {"configurable": configurable}, stream_mode=["custom", "values"]):
            if mode == "custom":
                if with_status or payload.get("type") != "status":
                    events.append(payload)
            else:
                final = payload
        return events, final

    return asyncio.run(go())


def kinds(events):
    return [e["type"] for e in events]


def chunks(events):
    return [e["text"] for e in events if e["type"] == "chunk"]


def feed_all(parser, deltas):
    out = []
    for delta in deltas:
        out += parser.feed(delta)
    tail, parsed = parser.finish()
    return out + tail, parsed


# ------------------------------------------------------------------ the chunk parser and checks

def test_parser_releases_whole_sentences_and_parses_the_envelope():
    out, parsed = feed_all(AnswerStreamParser(stream_violation), GOOD)
    assert "".join(out).strip() == GOOD_REPLY
    assert len(out) >= 2                                    # incremental, not one block at the end
    assert parsed == {"reply": GOOD_REPLY, "citations": ["KB-002"], "needs_clarification": False}
    assert not any("CITATIONS" in c or "MODE" in c for c in out)


def test_parser_character_by_character_never_releases_the_envelope_lines():
    parser = AnswerStreamParser(stream_violation)
    out, parsed = feed_all(parser, list("".join(GOOD)))
    assert "".join(out).strip() == GOOD_REPLY
    assert not any("CIT" in c or "MODE" in c for c in out)
    assert parsed["citations"] == ["KB-002"]


@pytest.mark.parametrize("phrase,reason", [
    ("Skip the posture check and reconnect.", "policy"),
    ("Your password is hunter2 for now.", "secret"),
    ("I have created a ticket for you.", "ticket_claim"),
    ("It is tracked as IT-2001 now.", "ticket_id"),
])
def test_unsafe_text_split_across_chunk_boundaries_is_never_released(phrase, reason):
    parser = AnswerStreamParser(stream_violation)
    deltas = ["MODE: ANSWER\n", "First do this. "] + list(phrase + " ") + ["\nCITATIONS: KB-002"]
    out, parsed = feed_all(parser, deltas)
    assert parser.tripped == reason
    assert "".join(out).strip() == "First do this."
    assert parsed["reply"].startswith("First do this.") and phrase in parsed["reply"]   # whole text still parsed


def test_phrase_split_over_two_sentences_of_context_is_caught():
    parser = AnswerStreamParser(stream_violation)
    out, _ = feed_all(parser, ["MODE: ANSWER\n", "Open the settings. ", "Skip the posture ", "check. ", "Done.\n"])
    assert parser.tripped == "policy" and not any("posture" in c for c in out)


def test_prohibition_is_not_a_violation():
    out, _ = feed_all(AnswerStreamParser(stream_violation),
                      ["MODE: ANSWER\n", "Never skip the posture check. ", "Update NakhlaConnect.\n", "CITATIONS: KB-002"])
    assert len(out) >= 1 and not AnswerStreamParser(stream_violation).tripped


def test_clarification_is_never_streamed():
    parser = AnswerStreamParser(stream_violation)
    out, parsed = feed_all(parser, ["MODE: CLARIFY\n", "Which error code do you see?"])
    assert out == []
    assert parsed == {"reply": "Which error code do you see?", "citations": [], "needs_clarification": True}


@pytest.mark.parametrize("deltas", [["Sure! Here is the answer. "], ["MODE: MAYBE\n", "x"], ['{"reply": "x"}']])
def test_broken_envelope_releases_nothing(deltas):
    parser = AnswerStreamParser(stream_violation)
    out, parsed = feed_all(parser, deltas)
    assert out == [] and parsed is None and parser.invalid


def test_long_sentence_without_punctuation_is_released_in_bounded_pieces():
    words = " ".join(["word"] * 200)
    out, parsed = feed_all(AnswerStreamParser(stream_violation), ["MODE: ANSWER\n", words, "\nCITATIONS: KB-002"])
    assert len(out) >= 2 and "".join(out).strip() == words and parsed["reply"] == words


def test_missing_citations_line_gives_no_citations():
    _, parsed = feed_all(AnswerStreamParser(stream_violation), ["MODE: ANSWER\n", "Update the client."])
    assert parsed["citations"] == []


def test_reconcile_keep_or_replace():
    assert reconcile(["Hello. ", "World."], False, "Hello. World.") == "keep"
    assert reconcile(["Hello. "], False, "Something else.") == "replace"
    assert reconcile(["Hello. "], True, "Hello.") == "replace"
    assert reconcile([], False, "Hello.") == "replace"


def test_streaming_prompt_has_the_same_rules_but_no_json_format():
    assert STREAM_SYSTEM_PROMPT.startswith(SYSTEM_PROMPT.split("Answer with exactly one JSON object")[0])
    assert "MODE: ANSWER" in STREAM_SYSTEM_PROMPT and "CITATIONS:" in STREAM_SYSTEM_PROMPT
    assert '{"reply"' not in STREAM_SYSTEM_PROMPT


# ------------------------------------------------------------------ the OpenRouter client

class FakeStream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __aiter__(self):
        async def gen():
            for chunk in self.chunks:
                yield chunk
        return gen()

    async def close(self):
        self.closed = True


def delta(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None)


def client_returning(stream=None, error=None):
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        if error:
            raise error
        return stream

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), seen


def collect(chat, usage=None):
    async def go():
        return [t async for t in chat.stream_text("sys", "user", usage=usage)]
    return asyncio.run(go())


def test_stream_text_uses_stream_true_without_tools_and_reports_usage():
    final = SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
    stream = FakeStream([delta("Hel"), SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
                                                       usage=None), delta("lo"), final])
    client, seen = client_returning(stream)
    usage: dict = {}
    assert collect(OpenRouterChat("k", "m", client=client), usage) == ["Hel", "lo"]
    assert seen["stream"] is True and "tools" not in seen and "functions" not in seen
    assert "response_format" not in seen and seen["temperature"] == 0
    assert usage == {"prompt_tokens": 7, "completion_tokens": 3}
    assert stream.closed


def test_stream_text_closes_the_http_stream_when_the_consumer_stops_early():
    stream = FakeStream([delta("a"), delta("b"), delta("c")])
    client, _ = client_returning(stream)
    chat = OpenRouterChat("k", "m", client=client)

    async def go():
        agen = chat.stream_text("s", "u")
        first = await agen.__anext__()
        await agen.aclose()
        return first

    assert asyncio.run(go()) == "a" and stream.closed


@pytest.mark.parametrize("error,code", [
    (openai.APITimeoutError(request=httpx.Request("POST", "http://x")), "timeout"),
    (openai.APIConnectionError(request=httpx.Request("POST", "http://x")), "api_error"),
])
def test_stream_text_errors_are_llm_errors(error, code):
    client, _ = client_returning(error=error)
    with pytest.raises(LLMError) as info:
        collect(OpenRouterChat("k", "m", client=client))
    assert info.value.code == code


def test_complete_json_is_unchanged_and_still_the_non_streaming_path():
    class Msg:
        content = '{"reply": "x", "citations": []}'
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=Msg)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    out = asyncio.run(OpenRouterChat("k", "m", client=client).complete_json("s", "u"))
    assert out.data["reply"] == "x" and "stream" not in seen and seen["response_format"] == {"type": "json_object"}


# ------------------------------------------------------------------ the graph

def test_streamed_turn_emits_chunks_then_the_verified_final_and_calls_no_json_model():
    llm = FakeStreamChat(GOOD)
    events, final = run_turn(llm)
    assert kinds(events) == ["chunk"] * len(chunks(events)) + ["final"] and len(chunks(events)) >= 2
    assert [e["seq"] for e in events[:-1]] == list(range(1, len(events)))
    assert all(e["provisional"] for e in events[:-1]) and events[-1]["provisional"] is False
    assert events[-1]["reply"] == final["reply"] and answer_body(final["reply"]) == GOOD_REPLY
    assert final["outcome"] == "answered"
    assert events[-1]["citations"] == [{"id": "KB-002", "title": "VPN client versions"}]
    # The verified final adds the Sources line and the resolution question, so the client swaps in the final text.
    assert reconcile(chunks(events), False, events[-1]["reply"]) == "replace"
    assert len(llm.stream_calls) == 1 and llm.calls == []
    assert llm.stream_calls[0]["system"] == STREAM_SYSTEM_PROMPT
    assert final["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}


def test_streaming_is_opt_in_without_the_flag_the_json_path_runs():
    llm = FakeStreamChat(GOOD, json_replies=({"reply": "Update to 5.4.", "citations": ["KB-002"],
                                              "needs_clarification": False},))
    events, final = run_turn(llm, streaming=False)
    assert events == [] and llm.stream_calls == [] and len(llm.calls) == 1
    assert llm.calls[0]["system"] == SYSTEM_PROMPT and final["outcome"] == "answered"


def test_a_model_without_stream_text_still_works_when_streaming_is_requested():
    llm = FakeChat({"reply": "Update to 5.4.", "citations": ["KB-002"], "needs_clarification": False})
    events, final = run_turn(llm)
    assert kinds(events) == ["final"] and final["outcome"] == "answered" and len(llm.calls) == 1


def test_clarification_keeps_its_meaning_and_is_not_streamed():
    llm = FakeStreamChat(["MODE: CLARIFY\n", "Which error code do you see?"])
    events, final = run_turn(llm)
    assert chunks(events) == [] and final["outcome"] == "clarify"
    assert final["reply"] == "Which error code do you see?" and kinds(events) == ["final"]


def test_unsafe_clarification_text_is_replaced_by_our_own_question():
    llm = FakeStreamChat(["MODE: CLARIFY\n", "Open https://evil.example and run the fix."])
    events, final = run_turn(llm)
    assert chunks(events) == [] and final["outcome"] == "clarify" and "evil.example" not in final["reply"]


def test_forbidden_advice_split_across_chunks_retracts_and_respond_refuses():
    llm = FakeStreamChat(["MODE: ANSWER\n", "Update NakhlaConnect first. ", "Then skip the po", "sture check. ",
                          "Done.\n", "CITATIONS: KB-002"])
    events, final = run_turn(llm)
    assert kinds(events) == ["chunk", "retract", "final"]
    assert events[1]["reason"] == "policy"
    assert not any("posture" in c for c in chunks(events))
    assert final["outcome"] == "policy_refusal" and events[-1]["outcome"] == "policy_refusal"
    assert reconcile(chunks(events), True, events[-1]["reply"]) == "replace"


def test_secret_in_the_stream_is_retracted_and_removed_from_the_final_reply():
    llm = FakeStreamChat(["MODE: ANSWER\n", "Update NakhlaConnect to 5.4. ", "Your password ", "is hunter2 today. ",
                          "\nCITATIONS: KB-002"])
    events, final = run_turn(llm)
    assert "retract" in kinds(events) and "hunter2" not in " ".join(chunks(events))
    assert "hunter2" not in final["reply"] and "hunter2" not in events[-1]["reply"]


def test_ticket_claim_in_the_stream_is_retracted_and_stripped_by_respond():
    llm = FakeStreamChat(["MODE: ANSWER\n", "Update NakhlaConnect to 5.4. ", "I have created a ticket for you. ",
                          "\nCITATIONS: KB-002"])
    events, final = run_turn(llm)
    assert "retract" in kinds(events) and "created a ticket" not in " ".join(chunks(events))
    assert "created a ticket" not in final["reply"] and answer_body(final["reply"]).startswith("Update NakhlaConnect")


def test_text_the_chunk_check_cannot_judge_is_replaced_by_the_verifier():
    # Unsupported fact: not a pattern the chunk check knows, so it IS shown provisionally; the final event replaces it.
    llm = FakeStreamChat(["MODE: ANSWER\n", "Update NakhlaConnect to 9.9. ", "Restart the laptop afterwards.\n",
                          "CITATIONS: KB-002"])
    events, final = run_turn(llm)
    assert any("9.9" in c for c in chunks(events))
    assert "9.9" not in final["reply"] and events[-1]["reply"] == final["reply"]
    assert reconcile(chunks(events), "retract" in kinds(events), final["reply"]) == "replace"


def test_answer_without_a_valid_citation_is_retracted_and_not_documented():
    llm = FakeStreamChat(["MODE: ANSWER\n", "Update NakhlaConnect to 5.4.\n", "CITATIONS: KB-999"])
    events, final = run_turn(llm)
    assert kinds(events) == ["chunk", "retract", "final"] and events[1]["reason"] == "not_answered"
    assert final["outcome"] == "not_documented" and final["reply"] == UNGROUNDED_REPLY
    assert final["citations"] == []


def test_model_error_after_chunks_retracts_and_gives_the_honest_failure_reply():
    llm = FakeStreamChat(["MODE: ANSWER\n", "Update NakhlaConnect to 5.4. ", LLMError("api_error")])
    events, final = run_turn(llm)
    assert kinds(events) == ["chunk", "retract", "final"] and events[1]["reason"] == "error"
    assert final["outcome"] == "error" and final["reply"] == LLM_FAILED_REPLY


def test_model_error_before_any_chunk_sends_no_retract():
    events, final = run_turn(FakeStreamChat([LLMError("timeout")]))
    assert kinds(events) == ["final"] and final["outcome"] == "error"


def test_stream_timeout_is_a_safe_error(monkeypatch):
    monkeypatch.setattr("app.agent.nodes.STREAM_TOTAL_SECONDS", 0.05)

    async def hang():
        await asyncio.sleep(5)

    events, final = run_turn(FakeStreamChat(["MODE: ANSWER\n", "Step one. ", hang]))
    assert kinds(events) == ["chunk", "retract", "final"] and final["outcome"] == "error"


def test_broken_envelope_falls_back_to_one_normal_json_call_without_chunks():
    llm = FakeStreamChat(["Sure, update to 5.4."], json_replies=(
        {"reply": "Update NakhlaConnect to 5.4.", "citations": ["KB-002"], "needs_clarification": False},))
    events, final = run_turn(llm)
    assert chunks(events) == [] and kinds(events) == ["final"]
    assert len(llm.stream_calls) == 1 and len(llm.calls) == 1
    assert final["outcome"] == "answered" and answer_body(final["reply"]) == "Update NakhlaConnect to 5.4."


def test_cancelled_turn_closes_the_model_stream():
    llm_started = {}

    async def hang():
        await asyncio.sleep(30)

    llm = FakeStreamChat(["MODE: ANSWER\n", "Step one. ", hang])
    graph = build_graph(InMemorySaver(), FakeMCP(), FakeKnowledgeBase(kb_result(KB002)), llm)

    async def go():
        seen = asyncio.Event()

        async def consume():
            async for mode, payload in graph.astream(
                    {"messages": [HumanMessage(content=TEXT)], "user_email": SARA},
                    {"configurable": {"thread_id": "c-1", STREAM_ANSWER_KEY: True}}, stream_mode=["custom"]):
                if payload["type"] != "chunk":
                    continue
                llm_started["first"] = payload
                seen.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(seen.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert llm_started["first"]["type"] == "chunk" and llm.closed == 1


# ------------------------------------------------------------------ existing gates stay in front of the stream

def test_undocumented_code_is_never_streamed():
    kb = FakeKnowledgeBase(kb_result(codes=("NX-999",), undocumented=("NX-999",)))
    llm = FakeStreamChat(GOOD)
    events, final = run_turn(llm, kb, text="NX-999 appears on my screen")
    assert llm.stream_calls == [] and llm.calls == [] and chunks(events) == []
    assert final["outcome"] == "not_documented" and kinds(events) == ["final"]


def test_unverified_policy_coverage_is_never_streamed():
    kb = FakeKnowledgeBase(kb_result(KB002, rule_coverage={"department_rules": "complete",
                                                           "governing_policy": "unavailable"}))
    llm = FakeStreamChat(GOOD)
    events, final = run_turn(llm, kb)
    assert llm.stream_calls == [] and llm.calls == [] and chunks(events) == []
    assert final["outcome"] == "blocked" and events[-1]["outcome"] == "blocked"


def test_policy_refusal_request_is_never_streamed():
    llm = FakeStreamChat(GOOD)
    events, final = run_turn(llm, text="Please turn off MFA for my account")
    assert llm.stream_calls == [] and chunks(events) == [] and final["outcome"] == "policy_refusal"


def test_ticket_request_takes_the_escalation_path_and_is_not_streamed():
    llm = FakeStreamChat(GOOD)
    events, final = run_turn(llm, text="Please open an IT ticket for this problem", mcp=DataMCP())
    assert llm.stream_calls == [] and chunks(events) == [] and kinds(events) == ["final"]


def test_evaluator_output_is_never_streamed():
    class Routed(FakeStreamChat):
        async def complete_json(self, system, user):
            if system == ev.EVAL_SYSTEM_PROMPT:
                self.calls.append({"system": system, "user": user})
                from app.llm import LLMReply
                return LLMReply(data={"decision": "ANSWER", "relevance": "high", "sufficient": True,
                                      "missing_info": [], "retry_query": "SECRET-EVAL-MARKER"}, usage={}, ms=1)
            return await super().complete_json(system, user)

    llm = Routed(GOOD)
    events, final = run_turn(llm, evaluate=True)
    assert len(llm.stream_calls) == 1 and any(c["system"] == ev.EVAL_SYSTEM_PROMPT for c in llm.calls)
    assert "SECRET-EVAL-MARKER" not in str(events) and "decision" not in " ".join(chunks(events))
    assert final["outcome"] == "answered"
