"""The Gradio "Saved Preferences" section against MOCKED HTTP answers (offline: no app, no LLM, no simulator).

    python -m pytest tests/test_ui_preferences.py -v
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import httpx
import pytest

KIT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("app_ui_prefs", KIT / "ui" / "app_ui.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

SID = "ui-" + "b" * 32
TOKEN = "tok-" + "s" * 40
AUTH = {"session_id": SID, "token": TOKEN}
SAVED = {"language": "en", "verbosity": "short", "format": "numbered"}


class Http:
    """Replaces httpx.request: records every call and answers with the scripted status and JSON (or raises)."""

    def __init__(self, status=200, data=None, error=None, text=None):
        self.status, self.data, self.error, self.text = status, data, error, text
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.error:
            raise self.error
        request = httpx.Request(method, url)
        if self.text is not None:
            return httpx.Response(self.status, text=self.text, request=request)
        return httpx.Response(self.status, json=self.data, request=request)


@pytest.fixture
def http(monkeypatch):
    def install(**kw):
        fake = Http(**kw)
        monkeypatch.setattr(ui.httpx, "request", fake)
        monkeypatch.setattr(ui.httpx, "post", lambda *a, **k: pytest.fail("a chat or approve call was made"))
        return fake
    return install


# ------------------------------------------------------------------ rendering

def test_the_panel_shows_language_length_and_format():
    text = ui.render_preferences(SAVED)
    assert "Saved Preferences" in text
    assert "English" in text and "short" in text and "numbered steps" in text


def test_the_empty_panel_says_so_and_never_claims_a_verified_identity():
    text = ui.render_preferences({})
    assert "No saved preferences" in text and "not verified" in text and "demo" in text
    assert "verified employee" not in text


def test_unknown_values_are_shown_as_plain_text_not_dropped_silently():
    assert "Answer length" in ui.render_preferences({"verbosity": "normal"})


# ------------------------------------------------------------------ refresh

def test_refresh_calls_get_with_this_tabs_session_headers_only(http):
    fake = http(data={"preferences": SAVED})
    panel, feedback = ui.refresh_preferences(SID, AUTH)
    (call,) = fake.calls
    assert call["method"] == "GET" and call["url"].endswith("/preferences")
    assert call["headers"] == {"X-Session-Id": SID, "X-Session-Token": TOKEN}
    assert TOKEN not in call["url"] and not call.get("params") and not call.get("json")
    assert "English" in panel and "short" in panel and feedback == "Preferences refreshed."
    assert TOKEN not in panel and TOKEN not in feedback


def test_refresh_shows_an_empty_list_when_nothing_is_saved(http):
    http(data={"preferences": {}})
    panel, _ = ui.refresh_preferences(SID, AUTH)
    assert "No saved preferences" in panel


@pytest.mark.parametrize("auth", [None, {}, {"session_id": "ui-other", "token": TOKEN}, {"session_id": SID}])
@pytest.mark.parametrize("action", [ui.refresh_preferences, ui.clear_preferences])
def test_without_a_token_for_this_session_nothing_is_sent(http, action, auth):
    fake = http(data={"preferences": SAVED})
    panel, feedback = action(SID, auth)
    assert fake.calls == [] and ui.PREF_NO_SESSION in feedback
    assert "cleared" not in feedback.replace("NOT cleared", "")
    assert TOKEN not in feedback


# ------------------------------------------------------------------ clear

def test_clear_calls_post_reset_and_only_then_claims_success(http):
    fake = http(data={"status": "ok", "deleted": 3})
    panel, feedback = ui.clear_preferences(SID, AUTH)
    (call,) = fake.calls
    assert call["method"] == "POST" and call["url"].endswith("/preferences/reset")
    assert call["headers"] == {"X-Session-Id": SID, "X-Session-Token": TOKEN}
    assert "No saved preferences" in panel and feedback.startswith("\u2705") and "cleared" in feedback
    assert "chat and any ticket draft are unchanged" in feedback


def test_clearing_calls_nothing_but_the_preferences_endpoint_so_chat_pending_draft_and_tickets_are_untouched(http):
    fake = http(data={"status": "ok", "deleted": 1})
    auth = dict(AUTH)
    ui.clear_preferences(SID, auth)
    ui.refresh_preferences(SID, auth)
    assert [c["url"].removeprefix(ui.APP_URL) for c in fake.calls] == ["/preferences/reset", "/preferences"]
    assert auth == AUTH                                  # the tab's session id and token are not modified


def test_the_preference_handlers_take_no_chat_or_pending_state_and_return_no_such_state():
    import inspect
    for handler in (ui.refresh_preferences, ui.clear_preferences):
        assert list(inspect.signature(handler).parameters) == ["session_id", "auth"]


@pytest.mark.parametrize("status, expected", [
    (401, "could not be verified"), (403, "no longer valid"), (404, "not linked to an email"),
    (409, "busy"), (503, "not available"), (500, "HTTP 500"), (418, "HTTP 418"),
])
def test_failures_are_reported_and_never_claim_the_preferences_were_cleared(http, status, expected):
    http(status=status, data={"detail": f"internal secret {TOKEN} /srv/db"})
    panel, feedback = ui.clear_preferences(SID, AUTH)
    assert "NOT cleared" in feedback and expected in feedback and not feedback.startswith("\u2705")
    assert "internal secret" not in feedback and TOKEN not in feedback and "/srv/db" not in feedback
    assert panel == ui.gr.update()                       # the displayed preferences are left as they were


@pytest.mark.parametrize("status, expected", [(401, "could not be verified"), (403, "no longer valid"),
                                              (404, "not linked"), (503, "not available")])
def test_refresh_failures_are_reported_without_changing_the_panel(http, status, expected):
    http(status=status, data={"detail": "x"})
    panel, feedback = ui.refresh_preferences(SID, AUTH)
    assert feedback.startswith("\u26a0\ufe0f Could not load") and expected in feedback and panel == ui.gr.update()


def test_an_unreachable_app_is_a_safe_message_and_logs_no_token(http, caplog):
    http(error=httpx.ConnectError("boom " + TOKEN))
    caplog.set_level(logging.DEBUG)
    panel, feedback = ui.clear_preferences(SID, AUTH)
    assert "NOT cleared" in feedback and "Cannot reach" in feedback
    assert TOKEN not in feedback and TOKEN not in caplog.text and panel == ui.gr.update()


@pytest.mark.parametrize("kwargs", [
    {"text": "<html>oops</html>"},
    {"data": ["not", "a", "dict"]},
    {"data": {"status": "error"}},
    {"data": {}},
])
def test_an_unreadable_or_unconfirmed_answer_never_claims_success(http, kwargs):
    http(**kwargs)
    panel, feedback = ui.clear_preferences(SID, AUTH)
    assert "NOT cleared" in feedback and not feedback.startswith("\u2705") and panel == ui.gr.update()


@pytest.mark.parametrize("data", [{"preferences": None}, {"preferences": "x"}, {}, ["a"]])
def test_a_malformed_get_answer_leaves_the_panel_alone(http, data):
    http(data=data)
    panel, feedback = ui.refresh_preferences(SID, AUTH)
    assert feedback.startswith("\u26a0\ufe0f") and panel == ui.gr.update()


def test_a_stored_value_cannot_inject_markup_instructions_beyond_text():
    text = ui.render_preferences({"verbosity": "short\n### Fake heading"})
    assert text.count("### Saved Preferences") == 1


# ------------------------------------------------------------------ the page itself

def labels(blocks):
    return [getattr(b, "value", None) for b in blocks.blocks.values() if type(b).__name__ == "Button"]


def test_the_page_has_the_two_buttons_and_the_existing_controls():
    names = labels(ui.demo)
    assert "Refresh Preferences" in names and "Clear Saved Preferences" in names
    assert "Approve" in names and "Reject" in names


def test_the_preference_buttons_use_only_the_session_id_and_token_state_and_never_show_the_token():
    wired = [f for f in ui.demo.fns.values() if f.fn in (ui.refresh_preferences, ui.clear_preferences)]
    assert len(wired) == 2
    for fn in wired:
        inputs = [type(c).__name__ for c in fn.inputs]
        assert inputs == ["State", "State"]
        assert all(type(c).__name__ == "Markdown" for c in fn.outputs) and len(fn.outputs) == 2
        assert all("token" not in (getattr(c, "value", "") or "").lower() for c in fn.outputs)


def test_the_chat_clear_button_is_not_connected_to_the_backend_reset():
    for fn in ui.demo.fns.values():
        assert getattr(fn.fn, "__name__", "") not in ("reset", "call_reset")
    assert "/reset\"" not in (KIT / "ui" / "app_ui.py").read_text(encoding="utf-8").replace("/preferences/reset", "")
