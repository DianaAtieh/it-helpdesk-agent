"""The NAKHLA dashboard (ui/app_ui.py): state badges, My Tickets, Service Status, navigation and page wiring.

Offline: helper functions and the Blocks layout only (no app, no model, no network). Histories are given in Gradio 6's
own format (content as a list of blocks) as well as plain strings.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

KIT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("app_ui_pages", KIT / "ui" / "app_ui.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

DRAFT = {"action_id": "a-1", "summary": "VPN NX-112",
         "payload": {"subject": "NakhlaConnect fails with NX-112", "priority": "P2", "category": "network_access"}}


def gradio_history(*pairs):
    """A history as Gradio 6 passes it to handlers."""
    return [{"role": role, "metadata": None, "content": [{"text": text, "type": "text"}]} for role, text in pairs]


# ------------------------------------------------------------------ message text and state badges

@pytest.mark.parametrize("content, expected", [
    ("plain", "plain"),
    ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "a\nb"),
    ({"type": "text", "text": "one"}, "one"),
    ([{"type": "file", "path": "x.png"}], None),
    (None, None),
])
def test_message_text_reads_both_formats(content, expected):
    assert ui.message_text(content) == expected


@pytest.mark.parametrize("state", ["answered", "clarify", "not_documented", "incident", "service_status",
                                   "access_info", "finance_vpn", "policy_refusal", "await_approval"])
def test_every_known_state_has_a_badge(state):
    badge = ui.state_badge(state)
    assert f"data-state='{state}'" in badge and ui.STATE_BADGES[state][0] in badge


def test_an_unknown_state_is_shown_by_name_not_dropped():
    assert "Some new state" in ui.state_badge("some_new_state") and "nk-neutral" in ui.state_badge("some_new_state")


def test_a_state_cannot_inject_html():
    badge = ui.state_badge("x'><script>alert(1)</script>")
    assert "<script" not in badge and "'>" not in badge.split("data-state='")[1].split("'")[0]
    assert ui.state_badge(None) == "" and ui.state_badge("") == ""


def test_the_last_response_panel_shows_the_badge_and_the_real_result():
    _, _, result_md = ui.side_panels({"state": "finance_vpn", "result": {"ticket_id": "IT-2042"}})
    assert "finance_vpn" in result_md and "IT-2042" in result_md


# ------------------------------------------------------------------ My Tickets: only confirmed tickets

def test_an_approved_ticket_is_recorded_once_with_the_approved_draft():
    history = gradio_history(("user", "Approve"), ("assistant", "Done. Your ticket is IT-2042. Network Operations "
                                                                "will contact you."))
    tickets, md = ui.record_ticket(history, DRAFT, [])
    assert tickets == [{"id": "IT-2042", "subject": "NakhlaConnect fails with NX-112", "priority": "P2",
                        "category": "network_access"}]
    assert "IT-2042" in md and "Network access" in md and "P2" in md
    again, _ = ui.record_ticket(history, DRAFT, tickets)                      # a double-click: still one entry
    assert again == tickets


@pytest.mark.parametrize("reply", [
    "OK, I did not file it. Nothing was created.",
    "I couldn't confirm that the ticket was created: the ticket system did not answer in time.",
    "The ticket was not created: the ticket system is unavailable right now.",
])
def test_a_rejected_or_failed_approval_records_nothing(reply):
    tickets, md = ui.record_ticket(gradio_history(("assistant", reply)), DRAFT, [])
    assert tickets == [] and md == ui.TICKETS_EMPTY


def test_the_empty_ticket_page_invents_nothing():
    assert ui.render_tickets([]) == ui.TICKETS_EMPTY and "IT-" not in ui.TICKETS_EMPTY


# ------------------------------------------------------------------ Service Status: the assistant's real answer

@pytest.mark.parametrize("state", ["service_status", "incident"])
def test_a_status_or_incident_answer_is_shown_on_the_status_page(state):
    result_md = ui.side_panels({"state": state})[2]
    out = ui.update_status_page(result_md, gradio_history(("assistant", "INC-2207: GW-2 degraded. Use GW-1.")))
    assert "INC-2207" in out and f"data-state='{state}'" in out


def test_other_answers_leave_the_status_page_alone():
    result_md = ui.side_panels({"state": "answered"})[2]
    assert ui.update_status_page(result_md, gradio_history(("assistant", "Update to 5.4.2."))) == ui.gr.skip()


def test_the_empty_status_page_invents_no_status():
    assert "INC-" not in ui.STATUS_EMPTY and "operational" not in ui.STATUS_EMPTY.lower()


# ------------------------------------------------------------------ navigation, profile and wiring

def test_navigation_shows_exactly_one_page():
    for page in ui.NAV_PAGES:
        shown = [column.visible for column in ui.show_page(page)]
        assert shown.count(True) == 1 and shown[ui.NAV_PAGES.index(page)] is True


def test_the_profile_card_shows_the_signed_in_email_and_a_logout_link():
    card = ui.profile_card(SimpleNamespace(username="sara.mikhail@nakhla.example"))
    assert "sara.mikhail@nakhla.example" in card and "/logout" in card and ">SM<" in card
    assert "Not signed in" in ui.profile_card(None)


def test_the_profile_card_escapes_the_name():
    assert "<script>" not in ui.profile_card(SimpleNamespace(username="<script>@x.y"))


def blocks(kind: str):
    return [b for b in ui.demo.blocks.values() if type(b).__name__ == kind]


def test_the_dashboard_has_the_brand_navigation_and_suggestions():
    radios = blocks("Radio")
    assert any(list(dict.fromkeys(c[0] for c in r.choices)) == list(ui.NAV_PAGES) for r in radios)
    labels = [b.value for b in blocks("Button")]
    assert all(text in labels for text in ui.SUGGESTIONS)
    assert "＋  New Conversation" in labels and "Approve" in labels and "Reject" in labels
    assert "NAKHLA" in ui.BRAND_HTML and "Intelligent IT Helpdesk" in ui.BRAND_HTML and "<svg" in ui.MARK_SVG


def test_suggestions_only_fill_the_message_box():
    msg = next(b for b in blocks("Textbox") if b.label == "Message")
    fills = [f for f in ui.demo.fns.values() if f.outputs == [msg] and not f.inputs]
    assert len(fills) >= len(ui.SUGGESTIONS)                    # they never call send / send_stream


def test_approval_records_tickets_only_after_decide():
    wired = [f for f in ui.demo.fns.values() if f.fn is ui.record_ticket]
    assert len(wired) == 1 and [type(c).__name__ for c in wired[0].inputs] == ["Chatbot", "State", "State"]


def test_the_ui_never_calls_the_session_reset_endpoint():
    source = (KIT / "ui" / "app_ui.py").read_text(encoding="utf-8").replace("/preferences/reset", "")
    assert '/reset"' not in source and "/admin" not in source


def test_the_status_button_asks_a_question_the_backend_treats_as_a_status_question():
    from app.agent.nodes import status_question
    assert status_question(ui.STATUS_QUESTION)


def test_the_sign_in_page_carries_the_brand_and_is_used_at_launch():
    page = ui.LOGIN_HTML
    assert "<svg" in page and "NAKHLA" in page and "Intelligent IT Helpdesk" in page and 'id="nk-login"' in page
    assert "<script" not in page.lower()                                   # styling and text only
    assert "ext. 4357" in page and "help.nakhla.example" in page           # contact details from KB-017
    styles = page.split("<style>")[1].split("</style>")[0]
    rules = [line.strip() for line in styles.splitlines() if "{" in line and not line.strip().startswith("@media")]
    scoped = ("#nk-login", ":has(#nk-login)", "footer")
    assert all(any(s in rule for s in scoped) for rule in rules)          # never styles the dashboard
    source = (KIT / "ui" / "app_ui.py").read_text(encoding="utf-8")
    assert "auth=users.check_login, auth_message=LOGIN_HTML" in source
