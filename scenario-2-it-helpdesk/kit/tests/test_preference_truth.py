"""The backend, not the model, says whether a preference was saved.

main._save_preferences adds exactly one deterministic note for a style request, decided by what really happened
(saved only after PreferenceStore.set_many succeeded; otherwise not lasting / no email / invalid email /
unavailable / write failed; nothing at all for a blocked, failed or interrupted turn). The respond node removes
model sentences that promise to remember or keep following a style, so the model can never contradict the note.
Offline: the app runs for real with fake model, knowledge base and helpdesk (the env fixture of test_ws_chat).
"""
from __future__ import annotations

import pytest

import app.main as app_main
from app.agent.nodes import PREFERENCE_REPLY
from app.agent.preferences import SAVED_CONFIRMATION, is_preference_claim, strip_preference_claims
from app.agent.prompts import STREAM_SYSTEM_PROMPT, SYSTEM_PROMPT
from tests.fakes import FakeChat, FakeStreamChat, answer_body
from tests.test_agent_graph import SARA
from tests.test_answer_streaming import GOOD_REPLY
from tests.test_preferences_integration import chat, start, store
from tests.test_ws_chat import body, converse, env, final_of, sid  # noqa: F401  (env is the app fixture)

REPORTED = "I want shorter answers, with bullet points"
ONE_OFF = "Keep answers short and use bullet points."           # a command, never saved as a lasting preference
PROMISE = "I will keep answers shorter and use bullet points as you requested."
REST = "Let me know if you have any IT questions or issues."
INVALID = "enter your email"


@pytest.fixture(autouse=True)
def clean_preferences(env):  # noqa: F811
    real = store()                                   # a test may switch the store off; clean the real one after
    real.delete(SARA)
    yield
    real.delete(SARA)


def model(reply: str) -> FakeChat:
    return FakeChat({"reply": reply, "citations": ["KB-002"]})


# ------------------------------------------------------------------ the reported false promise

def test_a_model_promise_is_removed_when_nothing_was_saved(env):  # noqa: F811
    _, _, _, response = start(env, model(f"{PROMISE}\n{REST}"), ONE_OFF)
    reply = response.json()["reply"]
    assert store().get(SARA) == {}
    assert PROMISE not in reply and "as you requested" not in reply and SAVED_CONFIRMATION not in reply
    assert REST in reply and app_main.PREFERENCES_NOT_LASTING_NOTE in reply


def test_the_reported_message_never_claims_more_than_the_database_holds(env):  # noqa: F811
    _, _, _, response = start(env, model(f"{PROMISE}\n{REST}"), REPORTED)
    reply, saved = response.json()["reply"], store().get(SARA)
    assert PROMISE not in reply
    assert (SAVED_CONFIRMATION in reply) == bool(saved)       # confirmed exactly when it is really stored


def test_an_answer_that_is_only_the_promise_becomes_the_fixed_reply(env):  # noqa: F811
    _, _, _, response = start(env, model(PROMISE), ONE_OFF)
    data = response.json()
    assert data["reply"].startswith(PREFERENCE_REPLY) and data["citations"] == []
    assert app_main.PREFERENCES_NOT_LASTING_NOTE in data["reply"] and store().get(SARA) == {}


def test_a_model_copy_of_the_backend_confirmation_is_removed(env):  # noqa: F811
    forged = f"{GOOD_REPLY}\nSaved to your preferences: answer length: short."
    _, _, _, response = start(env, model(forged), ONE_OFF)
    reply = response.json()["reply"]
    assert SAVED_CONFIRMATION not in reply and store().get(SARA) == {}


# ------------------------------------------------------------------ every save outcome

def test_a_confirmed_save_is_the_only_confirmation_and_the_model_claim_still_goes(env):  # noqa: F811
    text = "My VPN keeps disconnecting. From now on, use numbered steps and keep answers short."
    _, _, _, response = start(env, model(f"{GOOD_REPLY}\nFrom now on I'll use numbered steps for you."), text)
    reply = response.json()["reply"]
    assert store().get(SARA) == {"verbosity": "short", "format": "numbered"}
    assert reply.count(SAVED_CONFIRMATION) == 1 and "I'll use numbered steps for you" not in reply
    assert app_main.PREFERENCES_NOT_LASTING_NOTE not in reply


def test_a_write_failure_is_reported_and_the_model_claim_goes(env, monkeypatch):  # noqa: F811
    client = env.use(model(f"{GOOD_REPLY}\nI've saved your preference for short answers."))

    def boom(*args, **kwargs):
        raise RuntimeError("disk I/O error")

    monkeypatch.setattr(store(), "set_many", boom)
    reply = chat(client, sid(), "My VPN keeps disconnecting. From now on, keep answers short.").json()["reply"]
    assert app_main.PREFERENCES_SAVE_FAILED_NOTE in reply and SAVED_CONFIRMATION not in reply
    assert "I've saved your preference" not in reply and store().get(SARA) == {}


def test_an_unavailable_store_is_reported_on_the_model_route(env, monkeypatch):  # noqa: F811
    client = env.use(model(GOOD_REPLY))
    monkeypatch.setattr(app_main.app.state, "preferences", None)
    reply = chat(client, sid(), "My VPN keeps disconnecting. From now on, keep answers short.").json()["reply"]
    assert app_main.PREFERENCES_UNAVAILABLE_NOTE in reply and SAVED_CONFIRMATION not in reply


def test_an_invalid_email_is_reported_and_nothing_is_saved(env):  # noqa: F811
    client = env.use(model(GOOD_REPLY))
    reply = chat(client, sid(), "Remember that I prefer short answers.", email=INVALID).json()["reply"]
    assert app_main.PREFERENCES_INVALID_EMAIL_NOTE in reply and SAVED_CONFIRMATION not in reply
    assert app_main.PREFERENCES_SAVE_FAILED_NOTE not in reply


def test_a_missing_email_on_the_model_route_is_reported(env):  # noqa: F811
    client = env.use(model(GOOD_REPLY))
    reply = chat(client, sid(), "NakhlaConnect shows NX-112. From now on, keep answers short.", email=None).json()
    assert app_main.PREFERENCES_NO_EMAIL_NOTE in reply["reply"] and SAVED_CONFIRMATION not in reply["reply"]


def test_blocked_failed_and_interrupted_turns_get_no_note_and_save_nothing(env):  # noqa: F811
    env.use(model(GOOD_REPLY))
    save, portal = app_main._save_preferences, env.client.portal
    text = "Remember that I prefer short answers."
    for outcome in ("error", "blocked", "policy_refusal", "busy", "identity_mismatch"):
        assert portal.call(save, text, SARA, outcome, None) == ""
    assert portal.call(save, text, SARA, "answered", lambda: False) == ""          # the client went away
    assert store().get(SARA) == {}


def test_the_note_for_each_skipped_case_is_decided_by_code(env, monkeypatch):  # noqa: F811
    env.use(model(GOOD_REPLY))
    save, portal = app_main._save_preferences, env.client.portal
    lasting = "Remember that I prefer short answers."
    assert portal.call(save, "My VPN keeps disconnecting", SARA, "answered", None) == ""       # no style asked
    assert portal.call(save, lasting, None, "preference", None) == ""          # the fixed reply already says it
    assert portal.call(save, lasting, None, "answered", None) == app_main.PREFERENCES_NO_EMAIL_NOTE
    assert portal.call(save, lasting, INVALID, "answered", None) == app_main.PREFERENCES_INVALID_EMAIL_NOTE
    assert portal.call(save, ONE_OFF, SARA, "answered", None) == app_main.PREFERENCES_NOT_LASTING_NOTE
    monkeypatch.setattr(app_main.app.state, "preferences", None)
    assert portal.call(save, lasting, SARA, "answered", None) == app_main.PREFERENCES_UNAVAILABLE_NOTE


# ------------------------------------------------------------------ unrelated answers are unchanged

def test_a_normal_answer_is_unchanged_and_gets_no_note(env):  # noqa: F811
    _, _, _, response = start(env, model(GOOD_REPLY))
    assert answer_body(response.json()["reply"]) == GOOD_REPLY


def test_an_informational_remember_is_kept(env):  # noqa: F811
    text = f"Remember to save your work before restarting. {GOOD_REPLY}"
    _, _, _, response = start(env, model(text))
    assert answer_body(response.json()["reply"]) == text


# ------------------------------------------------------------------ the WebSocket final reply

def test_a_streamed_promise_is_replaced_by_a_truthful_final_reply(env):  # noqa: F811
    llm = FakeStreamChat(["MODE: ANSWER\n", f"{PROMISE} ", "Update NakhlaConnect to 5.4 from Software Center.\n",
                          "CITATIONS: KB-002"])
    events = converse(env.use(llm), body(sid(), ONE_OFF))
    final = final_of(events)
    assert PROMISE not in final["reply"] and SAVED_CONFIRMATION not in final["reply"]
    assert app_main.PREFERENCES_NOT_LASTING_NOTE in final["reply"] and final["display"] == "replace"
    assert store().get(SARA) == {}


def test_the_prompt_says_the_model_cannot_save_preferences():
    for prompt in (SYSTEM_PROMPT, STREAM_SYSTEM_PROMPT):
        assert "cannot save, store or remember preferences" in prompt


# ------------------------------------------------------------------ the claim filter itself

@pytest.mark.parametrize("sentence", [
    PROMISE,
    "Going forward, I'll answer in bullet points.",
    "I've noted your preference for short answers.",
    "Your preference has been saved.",
    "Your preferences have been updated.",
    "From now on I will keep my replies brief.",
    "I will always use bullet points for you.",
    "Saved to your preferences: answer length: short.",
])
def test_preference_saving_claims_are_detected(sentence):
    assert is_preference_claim(sentence)


@pytest.mark.parametrize("sentence", [
    "I'll keep this short.",
    "Always restart the client before connecting.",
    "Remember to save your work before restarting.",
    "Open Software Center and install NakhlaConnect 5.4.2 or later.",
    "If it still fails, I can escalate it to IT as a ticket.",
    "I will use the steps from KB-002.",
])
def test_ordinary_answer_sentences_are_not_claims(sentence):
    assert not is_preference_claim(sentence)


def test_stripping_keeps_steps_and_line_breaks_and_leaves_clean_text_untouched():
    steps = "1. Open Software Center.\n2. Install NakhlaConnect 5.4.2.\n3. Reconnect to GW-2."
    assert strip_preference_claims(steps) == (steps, 0)
    assert strip_preference_claims(f"{steps}\nGoing forward, I'll answer in numbered steps.") == (steps, 1)
    assert strip_preference_claims(f"Sure. {PROMISE} {REST}") == (f"Sure. {REST}", 1)
