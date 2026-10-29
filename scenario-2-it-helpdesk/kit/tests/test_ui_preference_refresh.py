"""The Gradio UI: an empty email box, and the Preferences panel refreshed only after a CONFIRMED save.

Offline, against MOCKED HTTP (no app, no LLM, no simulator), like tests/test_ui_preferences.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest

from app.agent.preferences import SAVED_CONFIRMATION

KIT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("app_ui_refresh", KIT / "ui" / "app_ui.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

SID = "ui-" + "c" * 32
AUTH = {"session_id": SID, "token": "tok-" + "t" * 40}
SAVED = {"verbosity": "short", "format": "bullets"}
CONFIRMED = (f"Thanks for letting me know how you like my answers written.\n\n{SAVED_CONFIRMATION} answer length: "
             "short; answer format: bullet points. This only changes how I write answers.")


class Http:
    """Replaces httpx.request: records every call, answers with the scripted status and JSON (or raises)."""

    def __init__(self, status=200, data=None, error=None):
        self.status, self.data, self.error, self.calls = status, data, error, []

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.error:
            raise self.error
        return httpx.Response(self.status, json=self.data, request=httpx.Request(method, url))


@pytest.fixture
def http(monkeypatch):
    def install(**kw):
        fake = Http(**kw)
        monkeypatch.setattr(ui.httpx, "request", fake)
        monkeypatch.setattr(ui.httpx, "post", lambda *a, **k: pytest.fail("a chat or approve call was made"))
        return fake
    return install


def turn(reply: str, earlier: list | None = None) -> list:
    return (earlier or []) + [{"role": "user", "content": "x"}, {"role": "assistant", "content": reply}]


# ------------------------------------------------------------------ the email box

def test_there_is_no_prefilled_or_editable_email_the_login_decides_who_is_asking():
    boxes = [b for b in ui.demo.blocks.values()
             if type(b).__name__ == "Textbox" and "email" in (getattr(b, "label", "") or "").lower()]
    assert boxes == []                                   # the signed-in email is used (see tests/test_ui_login.py)
    assert "tariq" not in (KIT / "ui" / "app_ui.py").read_text(encoding="utf-8").lower()


# ------------------------------------------------------------------ refresh only after a confirmed save

def test_a_confirmed_save_reloads_the_stored_values_once(http):
    fake = http(data={"preferences": SAVED})
    history = turn(CONFIRMED)
    panel, feedback, seen = ui.refresh_after_confirmed_save(history, SID, AUTH, 0)
    assert len(fake.calls) == 1 and fake.calls[0]["method"] == "GET" and fake.calls[0]["url"].endswith("/preferences")
    assert panel == ui.render_preferences(SAVED) and "updated" in feedback.lower() and seen == len(history)
    again = ui.refresh_after_confirmed_save(history, SID, AUTH, seen)            # the same turn again: no request
    assert len(fake.calls) == 1 and again[0] == ui.gr.skip()


@pytest.mark.parametrize("reply", [
    "Update NakhlaConnect to 5.4 from Software Center.",                                  # an ordinary answer
    "I couldn't save your preference right now, so it was not saved.",                   # a failed save
    "I didn't save this as a lasting preference.",                                       # a skipped save
    "I will keep answers shorter and use bullet points from now on.",                    # model wording only
])
def test_no_request_is_sent_without_the_backend_confirmation(http, reply):
    fake = http(data={"preferences": SAVED})
    panel, feedback, seen = ui.refresh_after_confirmed_save(turn(reply), SID, AUTH, 0)
    assert fake.calls == [] and panel == ui.gr.skip() and feedback == ui.gr.skip() and seen == 2


@pytest.mark.parametrize("kw", [{"status": 503, "data": {"detail": "x"}},
                                {"error": httpx.ConnectError("down")},
                                {"status": 200, "data": {"unexpected": True}}])
def test_a_failed_refresh_leaves_the_panel_as_it_was(http, kw):
    fake = http(**kw)
    panel, feedback, seen = ui.refresh_after_confirmed_save(turn(CONFIRMED), SID, AUTH, 0)
    assert len(fake.calls) == 1 and panel == ui.gr.skip() and "Refresh Preferences" in feedback and seen == 2


def test_an_unchanged_chat_sends_nothing(http):
    fake = http(data={"preferences": SAVED})
    history = turn(CONFIRMED)
    assert ui.refresh_after_confirmed_save(history, SID, AUTH, len(history))[0] == ui.gr.skip()
    assert ui.refresh_after_confirmed_save([], SID, AUTH, 0)[0] == ui.gr.skip()
    assert fake.calls == []


def test_the_refresh_also_works_with_gradios_own_history_format(http):
    """Gradio 6 hands handlers each message's content as a list of blocks, not a plain string."""
    fake = http(data={"preferences": SAVED})
    history = [{"role": "user", "content": [{"type": "text", "text": "x"}]},
               {"role": "assistant", "metadata": None, "content": [{"text": CONFIRMED, "type": "text"}]}]
    panel, _, seen = ui.refresh_after_confirmed_save(history, SID, AUTH, 0)
    assert len(fake.calls) == 1 and panel == ui.render_preferences(SAVED) and seen == 2


def test_the_ui_and_the_backend_use_the_same_confirmation_text():
    assert ui.SAVED_CONFIRMATION == SAVED_CONFIRMATION


def test_the_refresh_runs_after_the_chat_turn_with_only_chat_and_state_inputs():
    wired = [f for f in ui.demo.fns.values() if f.fn is ui.refresh_after_confirmed_save]
    assert len(wired) == 1
    assert [type(c).__name__ for c in wired[0].inputs] == ["Chatbot", "State", "State", "State"]
    assert [type(c).__name__ for c in wired[0].outputs] == ["Markdown", "Markdown", "State"]
    # the existing Refresh / Clear buttons are unchanged (exactly their two handlers)
    assert len([f for f in ui.demo.fns.values() if f.fn in (ui.refresh_preferences, ui.clear_preferences)]) == 2
