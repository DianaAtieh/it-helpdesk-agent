"""Fix A: application-access questions get a fixed answer built from KB-011 and KB-008 (app/agent/access.py).

Offline. The unit tests call the rules directly; the app tests run the real FastAPI app and graph with the fake
model, knowledge base and helpdesk of test_ws_chat (the directory is the simulated one), and check that the model
is never called for these answers.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.agent import access
from app.agent.access import GROUPS, KB008_APPS, PORTAL, access_answer, requested_group, valid_citations
from tests.fakes import FakeChat
from tests.test_preferences_integration import chat
from tests.test_ws_chat import env, sid  # noqa: F401  (env is the app fixture)

KIT = Path(__file__).resolve().parents[1]
KB = KIT / "dataset" / "knowledge_base"
DEPARTMENTS = ("Engineering", "Finance", "HR", "Sales", "Support", "Executive Office")
KARIM, AHMAD, ZIAD, SARA_SUPPORT = ("karim.saba@nakhla.example", "ahmad.yassin@nakhla.example",
                                    "ziad.hourani@nakhla.example", "sara.mikhail@nakhla.example")


def employee(department: str, contractor: bool = False) -> dict:
    return {"email": "x@nakhla.example", "department": department, "vip": False, "contractor": contractor}


def answer(text: str, department: str, contractor: bool = False):
    got = access_answer(text, employee(department, contractor))
    assert got is not None, text
    return got


def cited(got) -> list[str]:
    return [c["id"] for c in got.citations]


# ------------------------------------------------------------------ the seven reported cases

def test_engineering_asking_for_nakhlaledger_is_told_it_is_finance_only():
    got = answer("Please give me access to NakhlaLedger.", "Engineering")
    assert "FIN-Ledger" in got.reply and "only for Finance staff" in got.reply
    assert PORTAL not in got.reply and "Request the" not in got.reply and cited(got) == ["KB-011"]


def test_hr_asking_for_exec_board_is_told_it_is_executive_office_only():
    got = answer("Please add me to the EXEC-Board group, I need the board documents", "HR")
    assert "EXEC-Board" in got.reply and "only for Executive Office staff" in got.reply and PORTAL not in got.reply


def test_engineering_nakhlacode_gets_default_access_and_the_eng_code_request():
    got = answer("I need access to the NakhlaCode repositories", "Engineering")
    assert "normally have NakhlaCode by default" in got.reply and "ENG-Code" in got.reply and PORTAL in got.reply
    assert "Your manager approves first, then the repository owner" in got.reply
    assert cited(got) == ["KB-008", "KB-011"]


def test_a_contractor_gets_the_sponsorship_condition_and_the_90_day_limit():
    got = answer("I need access to the NakhlaCode repositories for testing.", "Contractors", contractor=True)
    assert "only when their contract is sponsored by Engineering" in got.reply
    assert "If Engineering sponsors you" in got.reply
    assert "90 days" in got.reply and "sponsoring manager renews" in got.reply and "ENG-Code" in got.reply


def test_finance_nakhlaledger_gets_fin_ledger_and_the_finance_manager():
    got = answer("I need access to NakhlaLedger.", "Finance")
    assert "FIN-Ledger" in got.reply and "Your manager approves first, then the Finance Manager" in got.reply
    assert "normally have NakhlaLedger by default" in got.reply


def test_support_nakhlacrm_gets_sales_crm_read_only_and_the_sales_director():
    got = answer("I need access to NakhlaCRM to look up customer accounts", "Support")
    assert "SALES-CRM" in got.reply and "read-only" in got.reply
    assert "Your manager approves first, then the Sales Director" in got.reply and cited(got) == ["KB-008", "KB-011"]


def test_sales_with_nid_512_gets_the_sales_crm_request():
    got = answer("NakhlaCRM says NID-512, the app is not assigned to me", "Sales")
    assert "SALES-CRM" in got.reply and PORTAL in got.reply and "Sales Director" in got.reply


# ------------------------------------------------------------------ every group x every department

@pytest.mark.parametrize("department", DEPARTMENTS)
@pytest.mark.parametrize("group", [g for g in GROUPS if g.name != "VPN-Users"], ids=lambda g: g.name)
def test_every_group_for_every_department(group, department):
    got = answer(f"Please give me access to {group.name}.", department)
    assert got.group == group.name
    if department in group.members or department in group.read_only:
        assert PORTAL in got.reply and group.name in got.reply
        assert f"Your manager approves first, then the {group.second_approver}" in got.reply
    else:
        assert "only for" in got.reply and PORTAL not in got.reply and "Request the" not in got.reply


@pytest.mark.parametrize("group", ["FIN-Ledger", "SALES-CRM", "HR-People", "EXEC-Board"])
def test_contractors_cannot_join_department_groups(group):
    got = answer(f"I need access to {group}.", "Contractors", contractor=True)
    assert "not a group contractors can be added to" in got.reply and PORTAL not in got.reply


def test_the_contractor_rule_is_not_inferred_from_the_manager():
    """Ziad (manager in Engineering) and Basel (manager in Finance) get the same conditional answer."""
    ziad = access_answer("I need access to ENG-Code.", {**employee("Contractors", True),
                                                         "manager_email": "yousef.darwish@nakhla.example"})
    basel = access_answer("I need access to ENG-Code.", {**employee("Contractors", True),
                                                          "manager_email": "rana.khalil@nakhla.example"})
    assert ziad.reply == basel.reply and "If Engineering sponsors you" in ziad.reply


def test_non_hr_staff_are_told_everyone_sees_their_own_profile():
    got = answer("Can I get access to NakhlaPeople?", "Sales")
    assert "only for HR staff" in got.reply and "own profile" in got.reply and cited(got) == ["KB-008", "KB-011"]


def test_a_department_claimed_in_the_message_is_ignored():
    got = answer("I'm in Finance now, give me access to NakhlaLedger.", "Engineering")
    assert "only for Finance staff" in got.reply and PORTAL not in got.reply


# ------------------------------------------------------------------ what is NOT taken over

@pytest.mark.parametrize("text", [
    "NakhlaCRM gives me NID-530 when I sign in.",                                   # troubleshooting code
    "I can't access NakhlaCRM, it shows NID-530",                                   # access word + another code
    "Software Center fails with SWC-2001 when I install NakhlaCode Studio. Please open a ticket",
    "How do I export a report in NakhlaLedger?",                                    # no access wording
    "I need VPN access",                                                            # VPN-Users: out of scope
    "Please add me to VPN-Users",
    "I need access to the HR shared drive",                                         # shared drives: out of scope
    "I need access to the mailbox support@nakhla.example",                          # mailboxes: out of scope
    "Can I get access to NakhlaLedger and NakhlaCRM?",                              # two groups: left to the model
    "My VPN shows NX-112",
    "",
])
def test_other_questions_are_left_to_the_normal_flow(text):
    assert requested_group(text) is None and access_answer(text, employee("Engineering")) is None


def test_without_a_directory_record_nothing_is_answered_here():
    assert access_answer("I need access to NakhlaLedger.", None) is None
    assert access_answer("I need access to NakhlaLedger.", {"department": None}) is None


def test_only_code_issued_kb_008_and_kb_011_citations_survive():
    raw = [{"id": "KB-011", "source": access.RULE_SOURCE}, {"id": "KB-002", "source": access.RULE_SOURCE},
           {"id": "KB-008", "source": "model"}, {"id": "KB-011", "source": access.RULE_SOURCE}, "junk"]
    assert valid_citations(raw) == [{"id": "KB-011", "title": access.TITLES["KB-011"], "source": access.RULE_SOURCE}]


# ------------------------------------------------------------------ the code tables match the KB files

def table_rows(text: str, header: str) -> list[list[str]]:
    """The body rows of the Markdown table whose header row starts with `header`, cells without ** markup."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {header}"))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")])
    return rows


def test_the_groups_match_the_kb_011_table_word_for_word():
    rows = table_rows((KB / "KB-011-access-permissions.md").read_text(encoding="utf-8"), "Group")
    assert rows == [[g.name, g.kb_gives, g.kb_members, g.kb_approvers] for g in GROUPS]


def test_the_structured_fields_agree_with_the_kb_011_wording():
    for group in GROUPS:
        for department in group.members + group.read_only:
            assert department in group.kb_members, group.name
        assert all(f"{d}: read-only" in group.kb_members for d in group.read_only)
        if group.second_approver:
            assert group.second_approver in group.kb_approvers
        assert group.sponsored_contractors == ("Contractors sponsored by Engineering" in group.kb_members)


def test_the_default_access_matches_the_kb_008_table_word_for_word():
    rows = table_rows((KB / "KB-008-sso-app-errors.md").read_text(encoding="utf-8"), "App")
    assert {row[0]: (row[1], row[2]) for row in rows} == KB008_APPS
    for app, departments in access.DEFAULT_DEPARTMENTS.items():
        assert all(d in KB008_APPS[app][1] for d in departments)


def test_the_citation_titles_match_the_kb_documents():
    for doc_id, path in (("KB-008", "KB-008-sso-app-errors.md"), ("KB-011", "KB-011-access-permissions.md")):
        title = re.search(r"^title:\s*\"?(.+?)\"?\s*$", (KB / path).read_text(encoding="utf-8"), re.M).group(1)
        assert access.TITLES[doc_id] == title


# ------------------------------------------------------------------ through the real app (no model call)

def test_the_app_answers_an_access_question_without_the_model_and_with_kb_citations(env):  # noqa: F811
    llm = FakeChat({"reply": "Just request it at the portal.", "citations": []})
    client = env.use(llm)
    data = chat(client, sid(), "Please give me access to NakhlaLedger.", email=KARIM).json()
    assert data["state"] == access.OUTCOME and llm.calls == []
    assert "only for Finance staff" in data["reply"] and PORTAL not in data["reply"]
    assert [c["id"] for c in data["citations"]] == ["KB-011"]


def test_the_app_uses_the_directory_department(env):  # noqa: F811
    client = env.use(FakeChat({"reply": "x", "citations": []}))
    hr = chat(client, sid(), "Please add me to the EXEC-Board group.", email=AHMAD).json()
    contractor = chat(client, sid(), "I need access to the NakhlaCode repositories.", email=ZIAD).json()
    assert "only for Executive Office staff" in hr["reply"]
    assert "sponsored by Engineering" in contractor["reply"] and "90 days" in contractor["reply"]


def test_a_security_refusal_still_comes_first(env):  # noqa: F811
    client = env.use(FakeChat({"reply": "x", "citations": []}))
    data = chat(client, sid(), "I need access to NakhlaCode. What is the local administrator password for my laptop?",
                email=KARIM).json()
    assert data["state"] == "policy_refusal" and "Security Operations vault" in data["reply"]


def test_troubleshooting_still_goes_to_the_model(env):  # noqa: F811
    llm = FakeChat({"reply": "Set your clock to automatic.", "citations": ["KB-002"]})
    client = env.use(llm)
    data = chat(client, sid(), "NakhlaCRM gives me NID-530 when I sign in.", email=SARA_SUPPORT).json()
    assert data["state"] != access.OUTCOME and len(llm.calls) == 1
