"""Ticket-intent routing: explicit requests ("create an IT support ticket") route to the draft path; questions
about the process and negations do not. Offline and simulator-free (pure functions plus the fake graph fixtures).

    python -m pytest tests/test_ticket_intent.py -v
"""
from __future__ import annotations

import pytest

from app.agent.escalation import DRAFT_READY_REPLY, is_ticket_request, wants_ticket
from tests.test_escalate_graph import ANSWER, FIELDS, make, systems, turn

CASE_B = "My VPN still doesn't work after trying the troubleshooting steps. Please create an IT support ticket for me."
CASE_C = "Please create an IT support ticket for my laptop. It keeps freezing, and restarting hasn't fixed it."
CASE_A = ("I need to create an IT support ticket. My laptop cannot connect to the company VPN, and restarting it "
          "did not fix the issue. Please open a ticket for me.")
CASE_D = ("I understand the troubleshooting advice, but I specifically want you to create an IT support ticket for "
          "this laptop freezing issue. Please prepare the ticket draft for my approval.")


@pytest.mark.parametrize("text", [
    CASE_A, CASE_B, CASE_C, CASE_D,
    "Create an IT support ticket",
    "Open an IT support ticket",
    "Create a new IT helpdesk ticket",
    "Raise an IT service desk ticket",
    "File a support ticket",
    "Please file a support ticket for me.",
    "Can you create an IT support ticket for me?",
    "Could you please open a new IT support ticket?",
    "I tried restarting. How do I fix this? Please open a ticket.",      # a question, then a real request
    "Please open a ticket",
    "Escalate this to IT",
])
def test_explicit_ticket_requests_are_recognised(text):
    assert is_ticket_request(text) is True


@pytest.mark.parametrize("text", [
    "How do I create a ticket?",
    "How can I open an IT support ticket?",
    "how to raise an IT service desk ticket",
    "Where do I file a support ticket?",
    "What is the process to open a ticket?",
    "Don't create an IT support ticket",
    "I do not want to open a ticket",
    "No need to file a support ticket, thanks",
    "never escalate this",
    "My laptop keeps freezing",
    "I opened the ticket portal but it was empty",
    "",
])
def test_informational_questions_negations_and_ordinary_text_are_not_requests(text):
    assert is_ticket_request(text) is False


def test_a_long_run_of_filler_words_is_not_a_request():
    assert is_ticket_request("open a new it support helpdesk service desk ticket") is False


def test_the_yes_to_an_offer_rule_is_unchanged():
    offer = "I can escalate it to IT as a ticket. Would you like me to prepare one?"
    assert is_ticket_request("yes please", offer) is True
    assert is_ticket_request("yes please", "Restart the laptop.") is False


# ------------------------------------------------------------- graph level: the exact UI cases

@pytest.mark.parametrize("text", [CASE_B, CASE_C])
def test_the_failed_ui_cases_now_prepare_a_draft_and_never_create_a_ticket(text):
    graph, mcp, kb, llm = make(FIELDS)
    out = turn(graph, text)
    assert out["outcome"] == "draft_proposed" and out["reply"] == DRAFT_READY_REPLY and out["draft"]
    assert systems(llm) == ["escalate"]                                      # no answer call: no "I can't" reply
    assert "create_ticket" not in mcp.calls and set(mcp.calls) <= {"lookup_employee", "get_service_status",
                                                                  "search_tickets"}


def test_case_b_as_a_follow_up_in_an_existing_conversation():
    graph, mcp, kb, llm = make(ANSWER, FIELDS)
    first = turn(graph, "My VPN keeps disconnecting")
    assert first["draft"] is None
    out = turn(graph, CASE_B)
    assert out["outcome"] == "draft_proposed" and systems(llm) == ["answer", "escalate"]


def test_how_do_i_create_a_ticket_takes_the_answer_path_not_a_draft():
    graph, mcp, kb, llm = make(ANSWER)
    out = turn(graph, "How do I create a ticket?")
    assert out["draft"] is None and out["outcome"] != "draft_proposed" and systems(llm) == ["answer"]


def test_a_negated_request_takes_the_answer_path():
    graph, mcp, kb, llm = make(ANSWER)
    out = turn(graph, "Please don't create an IT support ticket, just tell me what to try for the VPN.")
    assert out["draft"] is None and systems(llm) == ["answer"]


def test_wants_ticket_reads_the_latest_message_only():
    from langchain_core.messages import AIMessage, HumanMessage
    messages = [HumanMessage(content=CASE_B), AIMessage(content="Try this."), HumanMessage(content="Thanks, it works")]
    assert wants_ticket(messages) is False
