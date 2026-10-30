"""POST /api/session/start: a signed-in employee's NEW conversation is linked and given its session token before
any message, so the UI can load their saved preferences and tickets right away. Run from the kit folder:

    python -m pytest tests/test_session_start.py -v

The backend tests use the real app and the helpdesk API (tests/conftest.py); the UI tests are offline.
"""
from __future__ import annotations

import pytest

from app.main import SESSION_TOKEN_HEADER
from tests.helpers import requires_api
from tests.test_ui_pages import ui

SARA = "sara.mikhail@nakhla.example"
FADI = "fadi.mansour@nakhla.example"
REEM = "reem.halabi@nakhla.example"           # locked in the directory


def start(client, session_id: str, email: str):
    return client.post("/api/session/start", json={"session_id": session_id, "user": {"email": email}})


def chat(client, session_id: str, email: str | None, message: str = "hello"):
    body = {"session_id": session_id, "message": message, **({"user": {"email": email}} if email else {})}
    return client.post("/api/chat", json=body)


# ------------------------------------------------------------- backend

@requires_api
@pytest.mark.usefixtures("clean")
def test_an_active_employee_starts_a_new_conversation_and_can_read_their_preferences_at_once(client):
    r = start(client, "st-1", SARA)
    assert r.status_code == 200 and r.json() == {"session_id": "st-1", "status": "started"}
    assert r.headers.get(SESSION_TOKEN_HEADER)
    prefs = client.get("/preferences", headers={"X-Session-Id": "st-1"})            # token in the cookie jar
    assert prefs.status_code == 200 and "preferences" in prefs.json()
    assert client.get("/api/my-tickets", headers={"X-Session-Id": "st-1"}).status_code == 200


@requires_api
@pytest.mark.usefixtures("clean")
@pytest.mark.parametrize("email", ["nobody@nakhla.example", REEM])
def test_an_unknown_or_inactive_employee_cannot_start_a_conversation(client, email):
    r = start(client, "st-2", email)
    assert r.status_code == 403 and SESSION_TOKEN_HEADER not in r.headers
    assert client.get("/preferences", headers={"X-Session-Id": "st-2"}).status_code in (401, 403, 404)


@requires_api
@pytest.mark.usefixtures("clean")
def test_a_started_conversation_is_never_relinked_and_never_gets_a_second_token(client):
    first = start(client, "st-3", SARA)
    token = first.headers[SESSION_TOKEN_HEADER]
    again = start(client, "st-3", SARA)                          # the same employee, with the current token
    assert again.status_code == 200 and again.json()["status"] == "already_started"
    assert SESSION_TOKEN_HEADER not in again.headers
    other = start(client, "st-3", FADI)                          # someone else: refused, no token
    assert other.status_code == 409 and SESSION_TOKEN_HEADER not in other.headers
    client.cookies.clear()
    stranger = start(client, "st-3", SARA)                       # the right email but no token: refused too
    assert stranger.status_code == 409
    # The original token still works: nothing was replaced.
    r = client.get("/preferences", headers={"X-Session-Id": "st-3", SESSION_TOKEN_HEADER: token})
    assert r.status_code == 200


@requires_api
@pytest.mark.usefixtures("clean")
def test_chat_after_start_continues_the_linked_conversation_with_the_identity_lock(client):
    start(client, "st-4", SARA)
    ok = chat(client, "st-4", SARA)
    assert ok.status_code == 200 and ok.json()["state"] not in ("identity_mismatch", "unauthorized")
    assert SESSION_TOKEN_HEADER not in ok.headers                # no second token from the first chat turn
    assert chat(client, "st-4", FADI).json()["state"] == "identity_mismatch"
    client.cookies.clear()
    assert chat(client, "st-4", SARA).status_code == 403         # the session token is still required


@requires_api
@pytest.mark.usefixtures("clean")
def test_a_first_chat_that_wins_the_race_keeps_its_token_and_identity(client):
    first = chat(client, "st-5", SARA)
    token = first.headers[SESSION_TOKEN_HEADER]
    late = start(client, "st-5", SARA)                           # the UI's start arrives after the first message
    assert late.status_code == 200 and late.json()["status"] == "already_started"
    assert SESSION_TOKEN_HEADER not in late.headers
    assert start(client, "st-5", FADI).status_code == 409
    r = chat(client, "st-5", SARA, "still there?")
    assert r.status_code == 200 and r.json()["state"] != "identity_mismatch"
    assert client.get("/preferences", headers={"X-Session-Id": "st-5", SESSION_TOKEN_HEADER: token}).status_code == 200


# ------------------------------------------------------------- UI

class FakeRequest:
    def __init__(self, username):
        self.username = username


def test_the_ui_starts_the_session_for_the_signed_in_email_and_keeps_the_token(monkeypatch):
    sent = []
    monkeypatch.setattr(ui, "call_app", lambda path, body, token=None: (sent.append((path, body)) or
                                                                         ({"status": "started"}, "tok-1")))
    auth = ui.start_session("ui-new", "", FakeRequest(SARA))
    assert auth == {"session_id": "ui-new", "token": "tok-1"}
    assert sent == [("/api/session/start", {"session_id": "ui-new", "user": {"email": SARA}})]


def test_without_a_signed_in_email_nothing_is_sent_and_a_race_never_overwrites_the_token(monkeypatch):
    sent = []
    monkeypatch.setattr(ui, "call_app", lambda path, body, token=None: (sent.append(path) or ({}, None)))
    assert ui.start_session("ui-new", "", FakeRequest(None)) == ui.gr.skip() and sent == []
    # A first chat message already started it: no token in this answer, auth_state is left untouched.
    assert ui.start_session("ui-new", "", FakeRequest(SARA)) == ui.gr.skip() and sent == ["/api/session/start"]


def test_preferences_load_after_the_session_starts_without_a_message(monkeypatch):
    calls = []

    def fake(method, path, session_id, auth):
        calls.append((method, path))
        return True, {"preferences": {"format": "numbered", "verbosity": "short"}}

    monkeypatch.setattr(ui, "call_preferences", fake)
    panel = ui.load_preferences_on_start("ui-new", {"session_id": "ui-new", "token": "tok-1"})
    assert calls == [("GET", "/preferences")] and "numbered steps" in panel and "short" in panel
    assert ui.PREF_LOADING != ui.render_preferences({})          # the panel no longer starts as "no preferences"


def test_my_tickets_loads_from_the_ticket_endpoint_and_never_runs_a_chat_turn(monkeypatch):
    monkeypatch.setattr(ui, "call_app", lambda *a, **k: pytest.fail("no chat turn may be sent"))
    monkeypatch.setattr(ui, "call_preferences", lambda method, path, sid, auth: (path == "/api/my-tickets", {
        "tickets": [{"action_id": "a", "ticket_id": "IT-2001", "subject": "VPN", "status": "open",
                     "approval": "approved"}], "filter": {}}))
    html = ui.open_tickets_page(ui.NAV_PAGES[2], "ui-new", {"session_id": "ui-new", "token": "tok-1"})
    assert "nk-ticket-card" in html and "IT-2001" in html
    assert ui.open_tickets_page(ui.NAV_PAGES[0], "ui-new", None) == ui.gr.skip()
    assert not hasattr(ui, "MY_TICKETS_QUESTION")                # the hidden "Show me all my tickets" is gone


def test_every_page_load_starts_the_session_then_loads_preferences_and_tickets():
    names = [getattr(f.fn, "__name__", "") for f in ui.demo.fns.values()]
    for name in ("start_session", "load_preferences_on_start", "refresh_tickets_page"):
        assert name in names
