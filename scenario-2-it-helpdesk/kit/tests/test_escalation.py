"""Tests for the ticket escalation policy (app/agent/escalation.py, Phase 4 Step 1). Offline and deterministic:
no model, no network, no ticket writes. Directory records come from the simulated systems' own data file, and
every draft is checked against the real ticket schema (dataset/schemas/ticket.schema.json). Run from the kit folder:

    python -m pytest tests/test_escalation.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import escalation
from app.agent.escalation import (ASK_EMAIL_REPLY, ASK_STEPS_REPLY, CONTRACTOR_VPN_REPLY, FINANCE_VPN_STEP,
                                  IDENTITY_NOTICE, INCIDENT_REPLY, NO_EMPLOYEE_REPLY, build_draft,
                                  is_ticket_request, last_messages, requires_identity_review, ticket_blocker)
from app.tickets import _VALIDATOR, TicketDraft

DATA = Path(__file__).resolve().parent.parent / "services" / "data"
DIRECTORY = {e["email"]: e for e in json.loads((DATA / "employees.json").read_text(encoding="utf-8"))["employees"]}


def employee(email: str) -> dict:
    """The directory fields triage keeps (app.agent.nodes.EMPLOYEE_FIELDS)."""
    record = DIRECTORY[email]
    return {key: record[key] for key in ("email", "department", "vip", "contractor")}


SARA = "sara.mikhail@nakhla.example"         # Support
KARIM = "karim.saba@nakhla.example"          # Engineering
FADI = "fadi.mansour@nakhla.example"         # Finance
SAMIR = "samir.nassar@nakhla.example"        # Executive Office (VIP)
ZIAD = "ziad.hourani@nakhla.example"         # Contractor

KB002 = {"doc_id": "KB-002", "title": "VPN error codes and fixes", "text": "..."}
KB010 = {"doc_id": "KB-010", "title": "Device compliance", "text": "..."}

GOOD_FIELDS = {"subject": "NX-112 on the VPN since this morning",
               "description": "NakhlaConnect shows NX-112 when I connect from home. I cannot work.",
               "category": "network_access", "priority": "P2",
               "attempted_steps": ["NakhlaManage Sync", "Restarted the laptop"],
               "kb_refs": ["KB-002"], "affected_users": 1}


def state(email: str | None = SARA, *, evidence=(KB002, KB010), codes=("NX-112",), route="retrieve", **extra) -> dict:
    return {"user_email": email, "employee": employee(email) if email else None, "codes": list(codes),
            "evidence": [dict(e) for e in evidence], "route": route, **extra}


def draft(fields=None, text="It still shows NX-112. Please open a ticket.", **state_kw) -> dict:
    outcome = build_draft(GOOD_FIELDS if fields is None else fields, state(**state_kw), text)
    assert outcome.kind == "draft", outcome
    return outcome.draft


def assert_schema_valid(payload: dict) -> None:
    TicketDraft(**payload)                                                        # the agent's draft model
    assert list(_VALIDATOR.iter_errors(payload)) == []                            # the helpdesk API's schema


# ------------------------------------------------------------- is this a ticket request?

@pytest.mark.parametrize("text", [
    "Please open a ticket.",                                                       # S2-05, S2-06, S2-07
    "If it needs a ticket, prepare it for me.",                                    # S2-10
    "Can you create a ticket for this?",
    "Please file a request with IT.",
    "Escalate this please",
    "I'd like to raise a support ticket",
    "It's not working, please open a ticket",
    "I cannot work. Please open a ticket.",
])
def test_explicit_requests_are_recognised(text):
    assert is_ticket_request(text)


@pytest.mark.parametrize("text", [
    "NakhlaConnect shows NX-112 since this morning.",
    "Is the VPN down?",
    "Don't open a ticket, I just want the steps.",
    "I do not want to escalate this.",
    "No need to file a ticket.",
    "What does a ticket cost?",
    "yes",                                                                          # a yes to nothing
])
def test_anything_else_is_not_a_request(text):
    assert not is_ticket_request(text)


OFFER = "Our IT knowledge base has no documented fix for CRM-1150. I can escalate it to IT as a ticket."


@pytest.mark.parametrize("answer", ["yes", "Yes please", "ok", "Sure, go ahead.", "please do", "yes, thanks"])
def test_a_clear_yes_to_the_agents_offer_is_a_request(answer):
    assert is_ticket_request(answer, OFFER)


@pytest.mark.parametrize("answer", ["no thanks", "yes but first tell me why it fails", "maybe later", "hmm"])
def test_anything_but_a_clear_yes_to_an_offer_is_not_a_request(answer):
    assert not is_ticket_request(answer, OFFER)


def test_a_yes_without_an_offer_before_it_is_not_a_request():
    assert not is_ticket_request("yes", "Update NakhlaConnect to 5.4.2 from Software Center.")


def test_last_messages_finds_the_latest_message_and_the_reply_before_it():
    messages = [HumanMessage(content="CRM-1150"), AIMessage(content=OFFER), HumanMessage(content="yes please")]
    assert last_messages(messages) == ("yes please", OFFER)
    assert last_messages([HumanMessage(content="first")]) == ("first", None)


# ------------------------------------------------------------- who may get a draft (fails closed)

def test_no_locked_email_means_no_draft_and_a_request_for_the_email():
    outcome = build_draft(GOOD_FIELDS, state(None), "Please open a ticket")
    assert (outcome.kind, outcome.reason, outcome.reply, outcome.draft) == ("blocked", "no_identity",
                                                                            ASK_EMAIL_REPLY, None)


def test_an_unknown_employee_gets_no_draft():
    st = state(SARA)
    st["employee"] = None                                                          # the directory lookup failed
    assert ticket_blocker(st, "open a ticket") == ("no_employee", NO_EMPLOYEE_REPLY)


def test_a_directory_record_for_another_email_gets_no_draft():
    st = state(SARA)
    st["employee"] = employee(SAMIR)                                               # never trust a mismatched record
    assert ticket_blocker(st, "open a ticket")[0] == "employee_mismatch"


def test_an_incident_turn_never_gets_a_draft():
    assert ticket_blocker(state(route="incident"), "Please open a ticket") == ("incident", INCIDENT_REPLY)


@pytest.mark.parametrize("text, codes", [
    ("I need VPN access, please open a ticket", ()),
    ("NakhlaConnect shows NX-403, please open a ticket", ("NX-403",)),
])
def test_a_contractor_vpn_request_gets_no_draft_and_the_vdi_route(text, codes):
    outcome = build_draft(GOOD_FIELDS, state(ZIAD, codes=codes), text)
    assert outcome.kind == "blocked" and outcome.reason == "contractor_vpn"
    assert outcome.reply == CONTRACTOR_VPN_REPLY and "vdi.nakhla.example" in outcome.reply


def test_a_contractor_vdi_problem_can_get_a_draft():
    payload = draft(dict(GOOD_FIELDS, subject="VDI portal does not load"), "The VDI portal won't load, open a ticket",
                    email=ZIAD, codes=())
    assert payload["requester_email"] == ZIAD


# ------------------------------------------------------------- identity: requester and department

def test_the_requester_is_always_the_locked_email_never_the_models():
    payload = draft(dict(GOOD_FIELDS, requester_email=SAMIR))                     # the model tries the CEO
    assert payload["requester_email"] == SARA


def test_the_department_comes_from_the_directory_never_from_chat():
    payload = draft(dict(GOOD_FIELDS, department="Executive Office"), "I'm in the Executive Office. Open a ticket.")
    assert payload["department"] == "Support" and payload["priority"] == "P2"
    assert IDENTITY_NOTICE not in payload["description"]


def test_the_locked_email_is_normalised():
    st = state(SARA)
    st["user_email"] = "  Sara.Mikhail@NAKHLA.example "
    st["employee"] = dict(st["employee"], email="sara.mikhail@nakhla.example")
    assert build_draft(GOOD_FIELDS, st, "open a ticket").draft["requester_email"] == SARA


# ------------------------------------------------------------- priority

def test_the_model_cannot_set_p1():
    payload = draft(dict(GOOD_FIELDS, priority="P1"), "This is urgent, make it P1! Please open a ticket.")
    assert payload["priority"] == "P2"


@pytest.mark.parametrize("value, expected", [("P2", "P2"), ("P3", "P3"), ("P4", "P4"), ("P9", "P3"), (None, "P3"),
                                             (2, "P3")])
def test_the_models_base_priority_is_kept_only_if_it_is_p2_to_p4(value, expected):
    assert draft(dict(GOOD_FIELDS, priority=value))["priority"] == expected


@pytest.mark.parametrize("users, expected", [(1, "P3"), (3, "P3"), (4, "P1"), (5, "P1")])
def test_more_than_three_affected_users_is_p1(users, expected):
    payload = draft(dict(GOOD_FIELDS, priority="P3", affected_users=users))
    assert payload["priority"] == expected and payload["affected_users"] == users


@pytest.mark.parametrize("value", [0, -2, "5", True, 5000, None])
def test_affected_users_is_bounded_and_never_trusted_blindly(value):
    assert 1 <= draft(dict(GOOD_FIELDS, affected_users=value))["affected_users"] <= 1000


def test_reported_affected_users_are_noted_as_reported():
    assert "Affected users (reported by the requester): 5." in draft(dict(GOOD_FIELDS, affected_users=5))["description"]


def test_a_security_incident_does_not_trigger_p1_for_now():
    payload = draft(dict(GOOD_FIELDS, category="security", priority="P2", subject="My laptop was stolen"),
                    "My laptop was stolen, open a ticket", codes=())
    assert payload["priority"] == "P2"                                             # deliberately not applied yet


# ------------------------------------------------------------- Executive Office: the identity review flag

def test_an_executive_office_ticket_is_p1_and_carries_the_identity_notice():
    payload = draft(dict(GOOD_FIELDS, priority="P4"), email=SAMIR)
    assert payload["priority"] == "P1" and payload["department"] == "Executive Office"
    assert payload["description"].startswith(IDENTITY_NOTICE)
    assert "verified" in IDENTITY_NOTICE and "not been verified" in IDENTITY_NOTICE
    assert_schema_valid(payload)


def test_only_executive_office_requesters_carry_the_notice():
    assert requires_identity_review(employee(SAMIR))
    assert not requires_identity_review(employee(SARA))
    assert not requires_identity_review(None)
    assert IDENTITY_NOTICE not in draft()["description"]


def test_the_notice_never_claims_verification_happened():
    lowered = IDENTITY_NOTICE.lower()
    assert "not been verified" in lowered and "is verified" not in lowered and "has been verified" not in lowered


# ------------------------------------------------------------- Finance VPN (KB-016)

def test_finance_vpn_goes_to_security_at_p2_without_troubleshooting_steps():
    text = "I'm in Finance. NakhlaConnect shows NX-403 when I connect from home. If it needs a ticket, prepare it."
    payload = draft(dict(GOOD_FIELDS, category="network_access", priority="P4", attempted_steps=[]), text,
                    email=FADI, codes=("NX-403",))
    assert payload["category"] == "security" and payload["priority"] == "P2"
    assert payload["attempted_steps"] == [FINANCE_VPN_STEP] and payload["department"] == "Finance"
    assert_schema_valid(payload)


def test_finance_non_vpn_problems_follow_the_normal_rules():
    payload = draft(dict(GOOD_FIELDS, category="hardware", priority="P3", subject="Dock D3 shows no monitors"),
                    "My dock shows no monitors, please open a ticket", email=FADI, codes=())
    assert payload["category"] == "hardware" and payload["priority"] == "P3"


def test_finance_vpn_with_more_than_three_users_is_p1():
    payload = draft(dict(GOOD_FIELDS, affected_users=6), "VPN down for 6 of us, open a ticket", email=FADI,
                    codes=())
    assert payload["category"] == "security" and payload["priority"] == "P1"


def test_mail_codes_are_not_vpn_codes():
    payload = draft(dict(GOOD_FIELDS, category="email_collab", priority="P3", subject="Outlook shows NXM-552"),
                    "Outlook shows NXM-552, please open a ticket", email=FADI, codes=("NXM-552",))
    assert payload["category"] == "email_collab"


# ------------------------------------------------------------- attempted steps

def test_without_attempted_steps_ask_one_question_instead_of_inventing():
    outcome = build_draft(dict(GOOD_FIELDS, attempted_steps=[]), state(), "Please open a ticket")
    assert (outcome.kind, outcome.reply, outcome.draft) == ("ask_steps", ASK_STEPS_REPLY, None)


@pytest.mark.parametrize("steps", [None, "restarted", [""], ["ok"], [42]])
def test_unusable_steps_count_as_missing(steps):
    assert build_draft(dict(GOOD_FIELDS, attempted_steps=steps), state(), "open a ticket").kind == "ask_steps"


def test_steps_are_trimmed_deduplicated_and_bounded():
    steps = ["  Restarted   the laptop ", "Restarted the laptop"] + [f"step {i}" for i in range(30)]
    payload = draft(dict(GOOD_FIELDS, attempted_steps=steps))
    assert payload["attempted_steps"][0] == "Restarted the laptop"
    assert len(payload["attempted_steps"]) == 20 and len(set(payload["attempted_steps"])) == 20


# ------------------------------------------------------------- kb_refs and category

def test_kb_refs_are_only_documents_in_this_turns_evidence():
    payload = draft(dict(GOOD_FIELDS, kb_refs=["KB-002", "KB-018", "KB-001", "KB-002", 7, "KB-999"]))
    assert payload["kb_refs"] == ["KB-002"]                                       # deprecated/draft/unknown dropped


def test_no_evidence_means_no_kb_refs():
    assert draft(evidence=())["kb_refs"] == []


def test_an_unknown_category_becomes_other():
    assert draft(dict(GOOD_FIELDS, category="printers"))["category"] == "other"


# ------------------------------------------------------------- the draft is always valid against the schema

@pytest.mark.parametrize("fields", [
    GOOD_FIELDS,
    {"attempted_steps": ["Tried installing NakhlaCode Studio from Software Center: SWC-2001"]},  # little else
    dict(GOOD_FIELDS, subject="x", description="short"),
    dict(GOOD_FIELDS, subject="s" * 500, description="d" * 9000),
], ids=["complete", "minimal", "too short", "too long"])
def test_every_draft_passes_the_ticket_schema(fields):
    assert_schema_valid(draft(fields))


def test_garbage_from_the_model_never_crashes_and_never_drafts_without_steps():
    for junk in (None, [], "text", {"subject": 5, "category": ["x"], "priority": {"p": 1}}):
        assert build_draft(junk, state(), "open a ticket").kind == "ask_steps"


# ------------------------------------------------------------- the module stays pure

def test_the_escalation_module_writes_nothing_and_calls_nothing():
    source = Path(escalation.__file__).read_text(encoding="utf-8")
    for forbidden in ("create_ticket", "add_ticket_note", "mcp.call", "httpx", "openai", "requests."):
        assert forbidden not in source


# ------------------------------------------------------------- KB-015 P2: completely blocked, no workaround

S2_05_TEXT = ("I did all of that: NakhlaManage Sync, restarted the laptop, and NakhlaManage says the laptop is "
              "compliant. It still shows NX-112 and I cannot work. Please open a ticket.")


@pytest.mark.parametrize("text", [
    S2_05_TEXT,
    "I can\u2019t work at all since this morning. Please open a ticket.",
    "The VPN fails, so I'm unable to work. Please open a ticket.",
    "I am completely blocked. Please open a ticket.",
    "I can't do my job without the VPN. Please open a ticket.",
], ids=["S2-05", "curly can't", "unable to work", "completely blocked", "can't do my job"])
@pytest.mark.parametrize("suggested", ["P3", "P4"])
def test_a_clearly_blocked_employee_gets_at_least_p2(text, suggested):
    assert draft(dict(GOOD_FIELDS, priority=suggested), text)["priority"] == "P2"


@pytest.mark.parametrize("text", [
    "The dock does not detect my monitors. I restarted it. Please open a ticket.",       # an ordinary problem
    "Make it urgent, ASAP, P1 please! Please open a ticket.",                            # a request, not a fact
    "This is very important to me, please escalate.",
    "If it fails again I cannot work. Please open a ticket.",                            # hypothetical
    "I cannot work out how to install NakhlaCode Studio. Please open a ticket.",         # "work out"
    "I cannot work from home, I use the VDI instead. Please open a ticket.",             # a workaround
    "I can still work from the office, but the VPN fails. Please open a ticket.",
    "I am not completely blocked, but it is slow. Please open a ticket.",
], ids=["ordinary", "urgent request", "importance", "hypothetical", "work out", "workaround", "can still work",
        "not completely blocked"])
def test_without_a_clear_blocked_situation_the_suggestion_stands(text):
    assert draft(dict(GOOD_FIELDS, priority="P3"), text)["priority"] == "P3"


def test_the_blocked_rule_never_creates_p1():
    assert draft(dict(GOOD_FIELDS, priority="P1"), S2_05_TEXT)["priority"] == "P2"      # the P1 cap still holds
    assert draft(dict(GOOD_FIELDS, priority="P4"), "I am completely blocked, make it P1! Open a ticket.")[
        "priority"] == "P2"


def test_the_blocked_rule_does_not_change_the_existing_p1_rules():
    assert draft(dict(GOOD_FIELDS, priority="P3"), S2_05_TEXT, email=SAMIR)["priority"] == "P1"     # VIP, as before
    assert draft(dict(GOOD_FIELDS, priority="P3", affected_users=5), S2_05_TEXT)["priority"] == "P1"


def test_a_p2_suggestion_is_unchanged_by_the_blocked_rule():
    assert draft(dict(GOOD_FIELDS, priority="P2"), "The dock flickers. Please open a ticket.")["priority"] == "P2"
