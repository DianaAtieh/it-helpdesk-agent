"""The conversation flow around an answer and a ticket:

    issue -> answer citing inline ("According to KB-002, ..."), a "Sources: KB-002" line and the resolution question
          -> "it's fixed"        -> resolved, no ticket, no model call
          -> "open a ticket"     -> the mandatory employee fields (description, attempted_steps) are collected over
                                    turns, enforced in code; a summary and an approval request only when complete
          -> Approve             -> /api/approve files the ticket (API tests: tests/test_api_tickets.py fixtures)

Graph tests are offline (fake model, knowledge base and MCP, from tests/test_escalate_graph.py). Run from the kit
folder:

    python -m pytest tests/test_ticket_flow.py -v
"""
from __future__ import annotations

import pytest

from app.agent import escalation
from app.agent.escalation import (APPROVAL_REQUEST, ASK_BOTH_REPLY, ASK_DESCRIPTION_REPLY, ASK_STEPS_REPLY,
                                  DRAFT_READY_REPLY, RESOLUTION_QUESTION, RESOLVED_REPLY)
from app.agent.graph import build_graph
from app.main import app, mcp
from tests.fakes import FakeChat, FakeKnowledgeBase, kb_result
from tests.helpers import created_tickets, requires_api
from tests.test_api_tickets import KB002, approve, chat
from tests.test_escalate_graph import ANSWER, FADI, FIELDS, make, saved, systems, turn
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

def test_a_ticket_request_with_nothing_described_asks_for_both_fields_and_drafts_nothing():
    graph, mcp, kb, llm = make(NOTHING)
    out = turn(graph, "Please open a ticket.")
    assert (out["outcome"], out["reply"], out["draft"]) == ("clarify", ASK_BOTH_REPLY, None)
    fields = saved(graph)["ticket_fields"]
    assert fields["missing"] == ["description", "attempted_steps"] and fields["asked"] == ASK_BOTH_REPLY


def test_partial_information_asks_only_for_the_remaining_field_and_keeps_what_was_given():
    graph, mcp, kb, llm = make(NO_STEPS, dict(FIELDS, description="", attempted_steps=["Restarted the laptop"]))
    first = turn(graph, "My VPN shows NX-112 since this morning. Please open a ticket.")
    assert (first["outcome"], first["reply"], first["draft"]) == ("clarify", ASK_STEPS_REPLY, None)
    assert saved(graph)["ticket_fields"]["missing"] == ["attempted_steps"]
    # The model returns an EMPTY description on the second turn: the first turn's description is preserved.
    second = turn(graph, "I restarted the laptop.")
    assert second["outcome"] == "draft_proposed" and second["draft"]
    assert second["draft"]["description"].startswith(FIELDS["description"] + "\n\nDetails:")
    assert second["draft"]["attempted_steps"] == ["Restarted the laptop"]
    assert systems(llm) == ["escalate", "escalate"]
    # The model saw the value collected on the first turn.
    assert FIELDS["description"] in llm.calls[1]["user"]


def test_the_model_question_is_used_only_when_it_asks_for_exactly_the_missing_fields():
    good = "What have you already tried?"
    assert escalation.missing_question(good, ["attempted_steps"]) == good
    assert escalation.missing_question("What is the problem, and what have you tried?",
                                       ["attempted_steps"]) == ASK_STEPS_REPLY        # asks for a given field too
    assert escalation.missing_question("Which OS are you on?", ["description"]) == ASK_DESCRIPTION_REPLY
    assert escalation.missing_question("", ["description", "attempted_steps"]) == ASK_BOTH_REPLY


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
def test_after_the_fields_are_collected_only_approval_files_exactly_one_ticket(client, monkeypatch):
    llm = FakeChat(NO_STEPS, dict(FIELDS, description="", attempted_steps=["Restarted the laptop"]))
    graph = build_graph(app.state.checkpointer, mcp, FakeKnowledgeBase(kb_result(KB002, codes=("NX-112",))), llm)
    monkeypatch.setattr(app.state, "graph", graph)
    first = chat(client, "s-flow", "My VPN shows NX-112 since this morning. Please open a ticket.")
    assert first["state"] == "clarify" and first["pending_action"] is None and created_tickets() == []
    second = chat(client, "s-flow", "I restarted the laptop.")
    assert second["state"] == "await_approval" and second["reply"].endswith(APPROVAL_REQUEST)
    assert created_tickets() == []                                       # the summary alone files nothing
    action = second["pending_action"]["action_id"]
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
    assert escalation.describes_problem("nx-781\ncreate me ticket please")
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
    assert _missing_after(dict(COMPLETE, device_type=None)) == ["device_type"]
    assert _missing_after(dict(COMPLETE, operating_system=None)) == ["operating_system"]
    assert _missing_after(dict(COMPLETE, priority=None)) == ["priority"]            # the urgency
    assert _missing_after(dict(COMPLETE, affected_users=None)) == []          # the impact is never asked
    assert "description" in _missing_after(dict(COMPLETE, description=""), text="Please open a ticket.")


def test_invalid_values_are_never_stored_and_count_as_missing():
    assert _missing_after(dict(COMPLETE, device_type="toaster")) == ["device_type"]
    assert _missing_after(dict(COMPLETE, priority="urgent!!")) == ["priority"]
    assert escalation.merge_fields({}, dict(COMPLETE, affected_users=True)).get("affected_users") is None   # dropped
    assert _missing_after(dict(COMPLETE, affected_users=True)) == []          # optional: the contract default (1)


def test_optional_fields_may_stay_empty():
    values = escalation.prefill(escalation.merge_fields({}, COMPLETE), {"codes": []}, "")
    intake = escalation.validated_intake(values, {}, "NakhlaConnect shows NX-112 since this morning.")
    assert intake is not None and intake.error_code is None and intake.device_name is None


def test_the_directory_fills_the_location_but_the_operating_system_is_always_asked():
    employee = {"device_os": "macOS 15", "location": "Jerusalem HQ"}
    assert _missing_after(dict(COMPLETE, operating_system=None, location=None), employee) == ["operating_system"]
    assert "operating system" in escalation.ask_reply(["operating_system"])
    assert "how many" not in escalation.ask_reply(list(escalation.EMPLOYEE_FIELDS))     # never asked
    assert escalation.missing_question("How many people are affected, and which OS?", ["operating_system"]) \
        == escalation.ask_reply(["operating_system"])                                  # the model may not ask it


def test_missing_device_and_urgency_are_asked_for_together_and_kept_across_turns():
    first_json = dict(FIELDS, device_type=None, priority=None, missing_question="")
    second_json = dict(FIELDS, description="", device_type="desktop", priority="P3", attempted_steps=[])
    graph, mcp, kb, llm = make(first_json, second_json)
    first = turn(graph, "My VPN shows NX-112 since this morning. I restarted. Please open a ticket.")
    assert first["outcome"] == "clarify" and first["draft"] is None
    assert saved(graph)["ticket_fields"]["missing"] == ["device_type", "priority"]
    assert "which device" in first["reply"] and "how urgent" in first["reply"]
    assert "operating system" not in first["reply"] and "how many" not in first["reply"]   # given / never asked
    second = turn(graph, "It's my desktop, and I have a workaround.")
    assert second["outcome"] == "draft_proposed"
    assert second["draft"]["description"].startswith(FIELDS["description"])   # kept from the first turn
    assert "- Device: desktop" in second["draft"]["description"]
    assert second["draft"]["attempted_steps"] == FIELDS["attempted_steps"]      # kept from the first turn


def test_the_employee_comes_from_the_directory_never_from_the_model():
    graph, mcp, kb, llm = make(dict(FIELDS, requester_email="samir.nassar@nakhla.example",
                                    department="Executive Office"))
    out = turn(graph, "It still shows NX-112 after a sync and a restart. Please open a ticket.")
    assert out["draft"]["requester_email"] == "sara.mikhail@nakhla.example" and out["draft"]["department"] == "Support"
    assert "- Employee: Sara Mikhail · E-1022 · sara.mikhail@nakhla.example" in out["reply"]
    assert "- Operating system: Windows 11" in out["reply"] and "- Location: Remote" in out["reply"]


def test_an_incomplete_intake_never_reaches_the_approval_gate():
    graph, mcp, kb, llm = make(dict(FIELDS, device_type=None, missing_question=""))
    for text in ("NX-112 again. Please open a ticket.", "please just file it", "ok"):
        out = turn(graph, text)
        assert out["outcome"] == "clarify" and out["draft"] is None
    assert "create_ticket" not in mcp.calls
