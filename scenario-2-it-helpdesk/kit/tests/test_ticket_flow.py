"""The conversation flow around an answer and a ticket:

    issue -> answer citing inline ("According to KB-002, ..."), a "Sources: KB-002" line and the resolution question
          -> "it's fixed"        -> resolved, no ticket, no model call
          -> "open a ticket"     -> a clear request is drafted in one turn; only a missing description is asked for
                                    (over turns, enforced in code); a summary and an approval request when complete
          -> Approve             -> /api/approve files the ticket (API tests: tests/test_api_tickets.py fixtures)

Graph tests are offline (fake model, knowledge base and MCP, from tests/test_escalate_graph.py). Run from the kit
folder:

    python -m pytest tests/test_ticket_flow.py -v
"""
from __future__ import annotations

import pytest

from app.agent import escalation
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.escalation import (APPROVAL_REQUEST, ASK_DESCRIPTION_REPLY, ASK_STEPS_REPLY, DRAFT_READY_REPLY,
                                  NO_EMPLOYEE_REPLY, RESOLUTION_QUESTION, RESOLVED_REPLY,
                                  STEPS_NOT_PROVIDED)
from app.agent.graph import build_graph
from app.main import app, mcp
from app.schemas import ToolCall
from tests.fakes import FakeChat, FakeKnowledgeBase, kb_result
from tests.helpers import created_tickets, requires_api
from tests.test_api_tickets import KB002, approve, chat
from tests.test_escalate_graph import ANSWER, DIRECTORY, FADI, FIELDS, SARA, DataMCP, make, saved, systems, turn
from tests.test_ui_pages import ui

NO_STEPS = dict(FIELDS, attempted_steps=[], missing_question="")
NOTHING = dict(FIELDS, description="", attempted_steps=[], missing_question="")


# ------------------------------------------------------------- the answer

def test_an_answer_ends_with_its_sources_and_the_resolution_question():
    graph, mcp, kb, llm = make(ANSWER)
    out = turn(graph, "NakhlaConnect shows NX-112 since this morning.")
    assert out["outcome"] == "answered"
    assert out["reply"].endswith(f"Sources: KB-002\n\n{RESOLUTION_QUESTION}")
    # The model named no source inline, so code attaches it to the claim itself (one cited document supports it).
    assert out["reply"].startswith("According to KB-002, sync NakhlaManage, then restart the laptop.")
    assert [c["id"] for c in out["citations"]] == ["KB-002"]           # the citations field is unchanged


def test_an_inline_citation_by_the_model_is_kept_as_written():
    inline = dict(ANSWER, reply="According to KB-002, sync NakhlaManage and then restart the laptop.")
    graph, mcp, kb, llm = make(inline)
    out = turn(graph, "NakhlaConnect shows NX-112 since this morning.")
    assert out["outcome"] == "answered"
    assert out["reply"] == ("According to KB-002, sync NakhlaManage and then restart the laptop.\n\n"
                            f"Sources: KB-002\n\n{RESOLUTION_QUESTION}")


def test_the_ui_does_not_repeat_sources_that_are_already_in_the_message():
    cited = [{"id": "KB-002", "title": "VPN error codes and fixes"}]
    in_text = ui.side_panels({"reply": "According to KB-002, sync.\n\nSources: KB-002", "citations": cited})[0]
    assert "KB-002" not in in_text and "Cited inside the answer" in in_text
    # A code-issued citation the text does not name (the Finance VPN reply's KB-016) is still listed.
    finance = [{"id": "KB-002", "title": "VPN"}, {"id": "KB-016", "title": "Department matrix"}]
    panel = ui.side_panels({"reply": "Finance VPN problems go to Security Operations (KB-002).",
                            "citations": finance})[0]
    assert "KB-016" in panel and "`KB-002`" not in panel


# ------------------------------------------------------------- resolved: no ticket

def test_resolved_ends_the_flow_without_a_ticket_or_a_model_call():
    graph, mcp, kb, llm = make(ANSWER, FIELDS)
    turn(graph, "NakhlaConnect shows NX-112 since this morning.")
    out = turn(graph, "Yes, it works now, thanks!")
    assert (out["outcome"], out["reply"], out["draft"]) == ("resolved", RESOLVED_REPLY, None)
    assert systems(llm) == ["answer"] and "search_tickets" not in mcp.calls
    assert len(kb.searches) == 1                                        # no second search either


def test_still_broken_is_not_resolved():
    assert not escalation.reports_resolved(_after_answer("No, it's still not working."))
    assert not escalation.reports_resolved(_after_answer("It worked yesterday but not today"))
    assert escalation.reports_resolved(_after_answer("That fixed it"))
    assert escalation.reports_resolved(_after_answer("yes"))
    assert not escalation.reports_resolved(_after_answer("Not resolved, please open a ticket"))


def _after_answer(text: str) -> list:
    from langchain_core.messages import AIMessage, HumanMessage
    return [HumanMessage(content="NX-112"), AIMessage(content=f"Sync.\n\nSources: [KB-002]\n\n{RESOLUTION_QUESTION}"),
            HumanMessage(content=text)]


# ------------------------------------------------------------- mandatory fields, collected over turns

def test_a_ticket_request_with_nothing_described_asks_for_the_problem_and_drafts_nothing():
    graph, mcp, kb, llm = make(NOTHING)
    out = turn(graph, "Please open a ticket.")
    assert (out["outcome"], out["reply"], out["draft"]) == ("clarify", ASK_DESCRIPTION_REPLY, None)
    fields = saved(graph)["ticket_fields"]
    assert fields["missing"] == ["description"] and fields["asked"] == ASK_DESCRIPTION_REPLY


def test_no_attempted_steps_is_recorded_as_not_provided_and_is_never_a_step_taken():
    graph, mcp, kb, llm = make(NO_STEPS)
    out = turn(graph, "My VPN shows NX-112 since this morning. Please open a ticket.")
    assert out["outcome"] == "draft_proposed" and out["draft"]["attempted_steps"] == [STEPS_NOT_PROVIDED]
    assert "Steps already tried" not in out["reply"]                       # the summary never calls it a step
    assert "create_ticket" not in mcp.calls


def test_a_description_given_on_a_later_turn_completes_the_draft_and_is_kept():
    graph, mcp, kb, llm = make(NOTHING, dict(FIELDS, attempted_steps=["Restarted the laptop"]))
    first = turn(graph, "Please open a ticket.")
    assert (first["outcome"], first["reply"], first["draft"]) == ("clarify", ASK_DESCRIPTION_REPLY, None)
    second = turn(graph, "My VPN shows NX-112 since this morning, and I restarted the laptop.")
    assert second["outcome"] == "draft_proposed" and second["draft"]
    assert second["draft"]["description"].startswith(FIELDS["description"] + "\n\nDetails:")
    assert second["draft"]["attempted_steps"] == ["Restarted the laptop"]
    assert systems(llm) == ["escalate", "escalate"]


def test_the_model_question_is_used_only_when_it_asks_for_exactly_the_missing_fields():
    good = "What have you already tried?"
    assert escalation.missing_question(good, ["attempted_steps"]) == good
    assert escalation.missing_question("What is the problem, and what have you tried?",
                                       ["attempted_steps"]) == ASK_STEPS_REPLY        # asks for a given field too
    assert escalation.missing_question("Which OS are you on?", ["description"]) == ASK_DESCRIPTION_REPLY
    assert escalation.missing_question("", ["description"]) == ASK_DESCRIPTION_REPLY


def test_a_description_the_employee_never_gave_is_not_accepted():
    """Never invent: the model's description counts only when the employee's own words describe a problem."""
    graph, mcp, kb, llm = make(dict(FIELDS, missing_question=""))      # the model makes up a full description
    out = turn(graph, "Please open a ticket.")
    assert out["outcome"] == "clarify" and out["draft"] is None
    assert saved(graph)["ticket_fields"]["missing"] == ["description"]


def test_missing_fields_block_the_draft_on_every_turn_until_given():
    graph, mcp, kb, llm = make(NOTHING)
    for text in ("Please open a ticket.", "please", "ok"):
        out = turn(graph, text)
        assert out["outcome"] == "clarify" and out["draft"] is None
    assert saved(graph).get("draft") is None


# ------------------------------------------------------------- complete: summary and approval request, no ticket

def test_all_fields_give_a_summary_and_an_approval_request_but_no_ticket():
    graph, mcp, kb, llm = make(FIELDS)
    out = turn(graph, "It shows NX-112. I did NakhlaManage Sync and a restart. Please open a ticket.")
    assert out["outcome"] == "draft_proposed" and out["draft"]
    reply = out["reply"]
    assert reply.startswith(DRAFT_READY_REPLY) and reply.endswith(APPROVAL_REQUEST)
    for text in (FIELDS["subject"], "network_access", "Support", "NakhlaManage Sync; Restarted the laptop",
                 "Related articles: KB-002"):
        assert text in reply
    assert "create_ticket" not in mcp.calls and saved(graph)["ticket_fields"] is None


def test_finance_vpn_keeps_its_fixed_step_and_needs_no_steps_question():
    graph, mcp, kb, llm = make(NO_STEPS)
    out = turn(graph, "My VPN shows NX-112 since this morning, please open a ticket.", email=FADI)
    assert out["outcome"] == "draft_proposed"
    assert out["draft"]["category"] == "security" and out["draft"]["priority"] in ("P1", "P2")
    assert out["draft"]["attempted_steps"] == [escalation.FINANCE_VPN_STEP]


# ------------------------------------------------------------- approval files the ticket (real app + helpdesk API)

@requires_api
@pytest.mark.usefixtures("clean")
def test_a_clear_request_needs_one_turn_and_only_approval_files_exactly_one_ticket(client, monkeypatch):
    llm = FakeChat(dict(FIELDS, attempted_steps=["Restarted the laptop"]))
    graph = build_graph(app.state.checkpointer, mcp, FakeKnowledgeBase(kb_result(KB002, codes=("NX-112",))), llm)
    monkeypatch.setattr(app.state, "graph", graph)
    out = chat(client, "s-flow", "My VPN shows NX-112 and I restarted the laptop. Please open a ticket.")
    assert out["state"] == "await_approval" and out["reply"].endswith(APPROVAL_REQUEST)
    assert created_tickets() == []                                       # the summary alone files nothing
    action = out["pending_action"]["action_id"]
    done = approve(client, "s-flow", action)
    assert done["state"] == "ticket_created" and done["result"]["ticket_id"] in done["reply"]
    [ticket] = created_tickets()
    assert ticket["attempted_steps"] == ["Restarted the laptop"]
    assert ticket["description"].startswith(FIELDS["description"] + "\n\nDetails:")
    approve(client, "s-flow", action)                                    # a replayed approval: still one ticket
    assert len(created_tickets()) == 1


def test_an_error_code_alone_describes_the_problem_for_a_finance_vpn_ticket():
    """UI case: Fadi writes "nx-781", gets the Finance VPN reply, then "create me ticket please": the draft follows,
    with no question for a description he already gave (the code) or for steps (the Finance VPN fixed step)."""
    graph, mcp, kb, llm = make(dict(NO_STEPS, description="NakhlaConnect VPN shows error NX-781."))
    first = turn(graph, "nx-781", email=FADI)
    assert first["reply"] == escalation.FINANCE_VPN_REPLY
    out = turn(graph, "create me ticket please", email=FADI)
    assert out["outcome"] == "draft_proposed" and out["draft"]["category"] == "security"
    assert out["draft"]["attempted_steps"] == [escalation.FINANCE_VPN_STEP]
    assert not escalation.describes_problem("nx-781\ncreate me ticket please")   # only this route accepts it
    assert not escalation.describes_problem("create me ticket please")


# ------------------------------------------------------------- citations next to the claims they support

def test_each_claim_gets_only_the_source_that_supports_it_and_nothing_is_forced():
    from app.agent.answer_verifier import attribute_sentences
    evidence = [{"doc_id": "KB-002", "text": "NX-781: update to 5.4.2 from Software Center."},
                {"doc_id": "SP-002", "text": "Never edit client.ini to skip the version check."}]
    cited = [{"id": "KB-002"}, {"id": "SP-002"}]
    out = attribute_sentences("Update NakhlaConnect to 5.4.2 from Software Center. Do not edit the configuration. "
                              "According to SP-002, never skip the check.", cited, evidence)
    assert out == ("According to KB-002, update NakhlaConnect to 5.4.2 from Software Center. Do not edit the "
                   "configuration. "
                   "According to SP-002, never skip the check.")     # no fact to match, two sources: not forced
    assert attribute_sentences("Use version 9.9.9 instead.", cited, evidence) == "Use version 9.9.9 instead."
    both = [{"doc_id": "KB-002", "text": "Install 5.4.2."}, {"doc_id": "KB-009", "text": "Version 5.4.2 is in "
                                                                                       "Software Center."}]
    assert attribute_sentences("Install version 5.4.2 now.", [{"id": "KB-002"}, {"id": "KB-009"}], both) \
        == "According to KB-002 and KB-009, install version 5.4.2 now."
    # A "(KB-002)" the model wrote at the end of a sentence is said the same way; list markers stay in front.
    assert attribute_sentences("1. Restart NakhlaConnect (KB-002).", cited, evidence) \
        == "1. According to KB-002, restart NakhlaConnect."
    assert attribute_sentences("NakhlaConnect 5.4.2 is the minimum (KB-002).", cited, evidence) \
        == "According to KB-002, NakhlaConnect 5.4.2 is the minimum."      # a name keeps its capital


# ------------------------------------------------------------- the structured intake schema (TicketIntake)

def _missing_after(proposed: dict, employee: dict | None = None, text="My VPN shows NX-112 since this morning."):
    state = {"employee": employee or {}, "codes": [], "messages": []}
    values = escalation.prefill(escalation.merge_fields({}, proposed), state, text)
    return escalation.missing_fields(values, state, text, text)


COMPLETE = {"description": "NakhlaConnect shows NX-112 since this morning.", "device_type": "laptop",
            "operating_system": "Windows 11", "location": "Remote", "affected_users": 1, "priority": "P3",
            "attempted_steps": ["Restarted the laptop"]}


def test_each_mandatory_field_is_reported_missing_by_the_schema():
    assert _missing_after(COMPLETE) == []
    assert _missing_after(dict(COMPLETE, location=None)) == ["location"]
    assert _missing_after(dict(COMPLETE, affected_users=None)) == []          # the impact is never asked
    assert "description" in _missing_after(dict(COMPLETE, description=""), text="Please open a ticket.")


def test_details_the_employee_did_not_give_are_not_mandatory_and_never_guessed():
    for field in ("device_type", "operating_system", "priority", "attempted_steps"):
        assert _missing_after(dict(COMPLETE, **{field: None})) == []
    graph, mcp, kb, llm = make(dict(FIELDS, device_type=None, operating_system=None, priority=None))
    out = turn(graph, "NX-112 again after a sync and a restart. Please open a ticket.")
    assert out["outcome"] == "draft_proposed" and out["draft"]["priority"] == escalation.DEFAULT_PRIORITY
    assert "- Device:" not in out["draft"]["description"] and "- Device:" not in out["reply"]
    assert "- Urgency given:" not in out["draft"]["description"]


def test_invalid_values_are_never_stored():
    values = escalation.merge_fields({}, dict(COMPLETE, device_type="toaster", priority="urgent!!"))
    assert "device_type" not in values and "priority" not in values
    assert _missing_after(dict(COMPLETE, device_type="toaster", priority="urgent!!")) == []
    assert escalation.merge_fields({}, dict(COMPLETE, affected_users=True)).get("affected_users") is None   # dropped
    assert _missing_after(dict(COMPLETE, affected_users=True)) == []          # optional: the contract default (1)


def test_optional_fields_may_stay_empty():
    values = escalation.prefill(escalation.merge_fields({}, COMPLETE), {"codes": []}, "")
    intake = escalation.validated_intake(values, {}, "NakhlaConnect shows NX-112 since this morning.")
    assert intake is not None and intake.error_code is None and intake.device_name is None


def test_only_the_fields_the_employee_must_give_are_ever_asked():
    assert escalation.EMPLOYEE_FIELDS == ("description", "location")
    asked = escalation.ask_reply(list(escalation.EMPLOYEE_FIELDS))
    assert "how many" not in asked and "operating system" not in asked and "urgent" not in asked
    for question in ("How many people are affected?", "Which OS are you on?", "How urgent is it?",
                     "Which device is it on?"):
        assert escalation.missing_question(question, ["location"]) == escalation.ask_reply(["location"])


def test_the_employee_comes_from_the_directory_never_from_the_model():
    graph, mcp, kb, llm = make(dict(FIELDS, requester_email="samir.nassar@nakhla.example",
                                    department="Executive Office"))
    out = turn(graph, "It still shows NX-112 after a sync and a restart. Please open a ticket.")
    assert out["draft"]["requester_email"] == "sara.mikhail@nakhla.example" and out["draft"]["department"] == "Support"
    assert "- Employee: Sara Mikhail · E-1022 · sara.mikhail@nakhla.example" in out["reply"]
    assert "- Operating system: Windows 11" in out["reply"] and "- Location: Remote" in out["reply"]


def test_an_incomplete_intake_never_reaches_the_approval_gate():
    graph, mcp, kb, llm = make(dict(FIELDS, missing_question=""))
    for text in ("Please open a ticket.", "please just file it", "ok"):
        out = turn(graph, text)
        assert out["outcome"] == "clarify" and out["draft"] is None
    assert "create_ticket" not in mcp.calls


# ------------------------------------------------------------- the operating system: the employee's, else the directory's

def test_the_directory_operating_system_is_used_and_labelled_when_the_employee_gives_none():
    graph, mcp, kb, llm = make(dict(FIELDS, operating_system=None))
    out = turn(graph, "NX-112 again after a sync and a restart. Please open a ticket.")
    assert out["outcome"] == "draft_proposed"
    assert "- Operating system: Windows 11 24H2 (from directory)" in out["draft"]["description"]
    assert "- Operating system: Windows 11 24H2 (from directory)" in out["reply"]


def test_an_operating_system_the_employee_gave_beats_the_directory():
    graph, mcp, kb, llm = make(dict(FIELDS, operating_system="macOS 15"))
    out = turn(graph, "NX-112 on my macOS 15 laptop after a sync and a restart. Please open a ticket.")
    assert "- Operating system: macOS 15\n" in out["draft"]["description"]
    assert "from directory" not in out["draft"]["description"]


def test_an_operating_system_the_employee_never_mentioned_does_not_count_as_theirs():
    graph, mcp, kb, llm = make(dict(FIELDS, operating_system="macOS 15"))
    out = turn(graph, "NX-112 again after a sync and a restart. Please open a ticket.")
    assert "macOS" not in out["draft"]["description"]
    assert "- Operating system: Windows 11 24H2 (from directory)" in out["draft"]["description"]


def test_a_directory_record_without_an_operating_system_leaves_it_out(monkeypatch):
    monkeypatch.setitem(DIRECTORY, SARA, dict(DIRECTORY[SARA], device_os=None))
    graph, mcp, kb, llm = make(dict(FIELDS, operating_system=None))
    out = turn(graph, "NX-112 again after a sync and a restart. Please open a ticket.")
    assert out["outcome"] == "draft_proposed"
    assert "- Operating system:" not in out["draft"]["description"]


# ------------------------------------------------------------- nothing relaxed by the shorter intake

class DirectoryDown(DataMCP):
    """The directory cannot be reached; every other tool still answers."""

    async def call(self, tool, args=None):
        if tool == "lookup_employee":
            self.calls.append(tool)
            data = {"ok": False, "error": {"code": "helpdesk_unavailable"}}
            return data, ToolCall(tool=tool, ok=False, ms=1)
        return await super().call(tool, args)


def directory_down_graph(*replies):
    mcp, llm = DirectoryDown(), FakeChat(*replies)
    kb = FakeKnowledgeBase(kb_result(KB002, codes=("NX-112",)))
    return build_graph(InMemorySaver(), mcp, kb, llm), mcp, llm


def test_an_unavailable_directory_blocks_the_draft_and_nothing_is_prefilled():
    graph, mcp, llm = directory_down_graph(FIELDS)
    out = turn(graph, "NX-112 after a sync and a restart. Please open a ticket.")
    assert (out["outcome"], out["reply"], out["draft"]) == ("blocked", NO_EMPLOYEE_REPLY, None)
    assert llm.calls == [] and "create_ticket" not in mcp.calls


def test_an_unavailable_directory_never_lifts_a_security_refusal():
    graph, mcp, llm = directory_down_graph(FIELDS)
    out = turn(graph, "Please open a ticket to give me the local admin password.")
    assert out["draft"] is None and out["outcome"] in ("blocked", "policy_refusal") and llm.calls == []


def test_the_model_can_never_set_p1_and_the_rules_still_can():
    graph, mcp, kb, llm = make(dict(FIELDS, priority="P1"))
    out = turn(graph, "NX-112 after a sync and a restart. Please open a ticket.")
    assert out["draft"]["priority"] == "P2"                                 # capped, not a P1
    graph, mcp, kb, llm = make(dict(FIELDS, priority=None, affected_users=5))
    out = turn(graph, "NX-112 for 5 of us after a sync and a restart. Please open a ticket.")
    assert out["draft"]["priority"] == "P1"                                 # KB-015: more than 3 users, set by code


def test_no_urgency_uses_the_rules_and_the_employee_s_own_words():
    graph, mcp, kb, llm = make(dict(FIELDS, priority=None))
    out = turn(graph, "NX-112 after a sync and a restart, and I cannot work. Please open a ticket.")
    assert out["draft"]["priority"] == "P2"                                 # KB-015: blocked from working
    graph, mcp, kb, llm = make(dict(FIELDS, priority=None))
    out = turn(graph, "NX-112 after a sync and a restart. Please open a ticket.")
    assert out["draft"]["priority"] == escalation.DEFAULT_PRIORITY


# ------------------------------------------------------------- the short draft through the real approval gate

@requires_api
@pytest.mark.usefixtures("clean")
def test_a_draft_with_nothing_optional_passes_the_gate_and_is_filed_once(client, monkeypatch):
    bare = dict(FIELDS, device_type=None, operating_system=None, priority=None, attempted_steps=[])
    graph = build_graph(app.state.checkpointer, mcp, FakeKnowledgeBase(kb_result(KB002, codes=("NX-112",))),
                        FakeChat(bare))
    monkeypatch.setattr(app.state, "graph", graph)
    out = chat(client, "s-bare", "My VPN shows NX-112 since this morning. Please open a ticket.")
    assert out["state"] == "await_approval" and created_tickets() == []
    action = out["pending_action"]["action_id"]
    first = approve(client, "s-bare", action)
    again = approve(client, "s-bare", action)
    assert first["state"] == "ticket_created" and "already filed" in again["reply"]
    [ticket] = created_tickets()                                            # exactly one, valid for the ticket schema
    assert ticket["attempted_steps"] == [STEPS_NOT_PROVIDED] and ticket["requester_email"] == SARA


@requires_api
@pytest.mark.usefixtures("clean")
def test_a_rejected_short_draft_files_nothing(client, monkeypatch):
    graph = build_graph(app.state.checkpointer, mcp, FakeKnowledgeBase(kb_result(KB002, codes=("NX-112",))),
                        FakeChat(dict(FIELDS, device_type=None, priority=None, attempted_steps=[])))
    monkeypatch.setattr(app.state, "graph", graph)
    out = chat(client, "s-no", "My VPN shows NX-112 since this morning. Please open a ticket.")
    done = approve(client, "s-no", out["pending_action"]["action_id"], approved=False)
    assert "did not file" in done["reply"] and created_tickets() == [] and done.get("result") is None


# ------------------------------------------------------------- the one follow-up for a vague request

VAGUE = dict(FIELDS, description="My laptop is really slow today.", device_type=None, attempted_steps=[],
             error_code=None, missing_question="")
AFTER = dict(VAGUE, description="")
VAGUE_REQUEST = "My laptop is really slow today. Please open a ticket."
KB_NONE = kb_result(KB002)


def vague_graph(*replies):
    return make(*replies, kb=FakeKnowledgeBase(KB_NONE))


def test_a_vague_request_gets_one_combined_question_and_no_draft():
    graph, mcp, kb, llm = vague_graph(VAGUE)
    out = turn(graph, VAGUE_REQUEST)
    assert out["outcome"] == "clarify" and out["draft"] is None and "create_ticket" not in mcp.calls
    assert "error code" in out["reply"] and "tried" in out["reply"] and "don't know" in out["reply"]
    assert "which device" not in out["reply"]                  # "laptop" is in the request: never asked for


def test_i_dont_know_leads_to_a_draft_and_nothing_is_asked_twice():
    for answer in ("I don't know.", "Not applicable", "n/a"):
        graph, mcp, kb, llm = vague_graph(VAGUE, AFTER)
        first = turn(graph, VAGUE_REQUEST)
        out = turn(graph, answer)
        assert first["outcome"] == "clarify" and out["outcome"] == "draft_proposed"
        assert out["draft"]["attempted_steps"] == [STEPS_NOT_PROVIDED] and "- Error code:" not in out["draft"]["description"]
        assert "create_ticket" not in mcp.calls


def test_answers_are_kept_and_earlier_values_are_not_lost():
    graph, mcp, kb, llm = vague_graph(VAGUE, dict(AFTER, attempted_steps=["Restarted it twice"]))
    turn(graph, VAGUE_REQUEST)
    out = turn(graph, "I restarted it twice.")
    assert out["outcome"] == "draft_proposed" and out["draft"]["attempted_steps"] == ["Restarted it twice"]
    assert out["draft"]["description"].startswith(VAGUE["description"])


def test_only_details_the_employee_can_add_are_asked_and_directory_facts_never():
    graph, mcp, kb, llm = vague_graph(dict(VAGUE, description="Something is really slow today."))
    out = turn(graph, "Something is really slow today. Please open a ticket.")
    assert "which device" in out["reply"]
    assert "operating system" not in out["reply"] and "where you are" not in out["reply"]


def test_an_error_code_alone_is_not_a_clear_problem():
    graph, mcp, kb, llm = make(dict(FIELDS, missing_question=""))
    out = turn(graph, "NX-112. Please open a ticket.")
    assert (out["outcome"], out["reply"], out["draft"]) == ("clarify", ASK_DESCRIPTION_REPLY, None)


def test_an_error_code_or_step_the_employee_never_gave_does_not_skip_the_question():
    invented = dict(VAGUE, error_code="NX-999", attempted_steps=["Reinstalled the graphics driver"])
    graph, mcp, kb, llm = vague_graph(invented)
    out = turn(graph, VAGUE_REQUEST)
    assert out["outcome"] == "clarify" and out["draft"] is None


def test_a_finance_vpn_problem_is_never_asked_for_steps():
    graph, mcp, kb, llm = make(dict(FIELDS, attempted_steps=[], category="security"))
    out = turn(graph, "My VPN keeps dropping when I connect from home. Please open a ticket.", email=FADI)
    assert out["outcome"] == "draft_proposed" and out["draft"]["category"] == "security"
