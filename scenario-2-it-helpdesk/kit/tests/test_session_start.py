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
    session_id, auth = ui.start_session("", FakeRequest(SARA))
    assert session_id.startswith("ui-") and auth == {"session_id": session_id, "token": "tok-1"}
    assert sent == [("/api/session/start", {"session_id": session_id, "user": {"email": SARA}})]
    assert ui.start_session("", FakeRequest(SARA))[0] != session_id          # every new chat: a new conversation


def test_without_a_signed_in_email_nothing_is_sent_and_a_refused_start_leaves_no_token(monkeypatch):
    sent = []
    monkeypatch.setattr(ui, "call_app", lambda path, body, token=None: (sent.append(path) or ({}, None)))
    session_id, auth = ui.start_session("", FakeRequest(None))
    assert session_id and auth is None and sent == []
    session_id, auth = ui.start_session("", FakeRequest(SARA))       # refused: the first chat message starts it
    assert session_id and auth is None and sent == ["/api/session/start"]


def test_the_conversation_id_has_one_owner_so_it_can_never_disagree_with_its_token():
    """The bug this prevents: gr.State(callable) is built once for all tabs and replaced by its own load event, so a
    load event reading it got a shared id, and the token never matched the tab's real id."""
    load = next(f for f in ui.demo.fns.values() if getattr(f.fn, "__name__", "") == "start_session")
    states = [b for b in load.outputs if type(b).__name__ == "State"]
    assert len(states) == 2 and states[0].value is None and not callable(states[0].value)
    assert states[0] not in load.inputs                              # it never reads the id: it creates it


@pytest.mark.parametrize("handler", ["send", "send_stream"])
def test_a_message_sent_before_the_page_has_started_stays_in_the_box(handler, monkeypatch):
    import asyncio
    import inspect
    monkeypatch.setattr(ui, "call_app", lambda *a, **k: pytest.fail("nothing may be sent"))
    result = getattr(ui, handler)("my vpn", [], None, "", None)
    if inspect.isasyncgen(result):
        async def first():
            return [item async for item in result]
        result = asyncio.run(first())[0]
    assert result[0] == ui.gr.skip() and result[1] == []


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


def test_a_failed_start_shows_a_clear_message_and_never_stays_on_loading(monkeypatch):
    monkeypatch.setattr(ui, "call_preferences", lambda *a: pytest.fail("no call without a session"))
    panel, tickets = ui.load_preferences_on_start("ui-new", None), ui.load_tickets_on_start("ui-new", None)
    assert ui.START_FAILED in panel and ui.PREF_LOADING not in panel
    assert ui.html.escape(ui.START_FAILED) in tickets


def test_a_failed_load_says_so_and_never_stays_on_loading(monkeypatch):
    auth = {"session_id": "ui-new", "token": "tok-1"}
    monkeypatch.setattr(ui, "call_preferences", lambda *a: (False, "The app did not answer."))
    assert "The app did not answer." in ui.load_preferences_on_start("ui-new", auth)
    assert "The app did not answer." in ui.load_tickets_on_start("ui-new", auth)
    monkeypatch.setattr(ui, "call_preferences", lambda *a: (True, {"preferences": "not a dict"}))
    assert "could not be read" in ui.load_preferences_on_start("ui-new", auth)


def test_empty_preferences_and_tickets_show_their_empty_states(monkeypatch):
    auth = {"session_id": "ui-new", "token": "tok-1"}
    monkeypatch.setattr(ui, "call_preferences", lambda method, path, sid, a: (True, {"preferences": {}, "tickets": [],
                                                                                    "filter": {}}))
    assert "No saved preferences" in ui.load_preferences_on_start("ui-new", auth)
    assert "You don't have any tickets" in ui.load_tickets_on_start("ui-new", auth)


def test_two_employees_get_their_own_session_token_and_data(monkeypatch):
    issued = {}

    def fake_app(path, body, token=None):
        issued[body["session_id"]] = body["user"]["email"]
        return {"status": "started"}, f"tok-{body['user']['email']}"

    def fake_get(method, path, session_id, auth):
        assert auth == {"session_id": session_id, "token": f"tok-{issued[session_id]}"}   # matched, never crossed
        return True, {"preferences": {"language": "en"}, "tickets": [{"action_id": issued[session_id],
                                                                      "subject": issued[session_id]}], "filter": {}}

    monkeypatch.setattr(ui, "call_app", fake_app)
    monkeypatch.setattr(ui, "call_preferences", fake_get)
    first = ui.start_session("", FakeRequest(SARA))
    second = ui.start_session("", FakeRequest("fadi.mansour@nakhla.example"))
    assert first[0] != second[0]
    assert SARA in ui.load_tickets_on_start(*first) and "fadi" not in ui.load_tickets_on_start(*first)
    assert "fadi" in ui.load_tickets_on_start(*second) and SARA not in ui.load_tickets_on_start(*second)


def test_only_one_load_event_initialises_the_session_and_the_panels_follow_it():
    fns = list(ui.demo.fns.values())
    assert sum(getattr(f.fn, "__name__", "") == "start_session" for f in fns) == 1
    session = next(f for f in fns if getattr(f.fn, "__name__", "") == "start_session").outputs[0]
    assert session.load_event_to_attach is None                      # no second load event of its own


def test_every_page_load_starts_the_session_then_loads_preferences_and_tickets():
    names = [getattr(f.fn, "__name__", "") for f in ui.demo.fns.values()]
    for name in ("start_session", "load_preferences_on_start", "load_tickets_on_start"):
        assert name in names


def test_employees_do_not_wait_for_each_other_and_nothing_per_user_is_shared():
    assert ui.demo._queue.default_concurrency_limit == 20
    chat = next(f for f in ui.demo.fns.values() if getattr(f.fn, "__name__", "") == "send_stream")
    assert chat.concurrency_limit == "default"                       # it uses that limit, not one of its own
    states = [b for b in ui.demo.blocks.values() if type(b).__name__ == "State"]
    assert len(states) == 7                                          # per employee values live in gr.State
