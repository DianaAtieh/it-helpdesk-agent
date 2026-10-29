"""Session ownership on /api/chat and /ws/chat, and a pending ticket draft next to a preference-only message.
Offline and simulator-free: the app runs for real over temporary databases with fake model, KB and helpdesk API;
nothing here calls /admin/reset or any live endpoint.

    python -m pytest tests/test_session_ownership.py -v
"""
from __future__ import annotations

import pytest

import app.main as app_main
from tests.fakes import FakeChat, FakeKnowledgeBase, FakeStreamChat, kb_result
from tests.test_answer_streaming import GOOD
from tests.test_escalate_graph import FIELDS
from tests.test_preferences_integration import (OTHER, SAVED, PREF_TEXT, SARA, checkpoint_of, chat,  # noqa: F401
                                                clean_preferences, llm_http, prefs, start, store)
from tests.test_ws_chat import KB_NX112, TEXT, TICKET_TEXT, body, converse, env, final_of, sid  # noqa: F401

WRONG = "x" * 43
PREF_ONLY = "From now on, keep my answers short and use bullet points."


def history_of(env, session_id):
    return env.client.portal.call(checkpoint_of, session_id)


# ------------------------------------------------------------------ HTTP: a first turn creates, later turns prove

def test_http_a_first_turn_creates_the_session_and_issues_a_token(env):
    client = env.use(llm_http())
    response = chat(client, sid(), TEXT)
    assert response.status_code == 200 and response.headers.get("X-Session-Token")


def test_http_a_valid_token_continues_the_conversation(env):
    client, session_id, token, _ = start(env, llm_http())
    again = chat(client, session_id, TEXT, token=token)
    assert again.status_code == 200 and again.json()["state"] != "unauthorized"
    assert "X-Session-Token" not in again.headers                  # a token is issued once


def test_http_the_session_cookie_also_proves_ownership_like_before(env):
    client = env.use(llm_http())
    session_id = sid()
    assert chat(client, session_id, TEXT).status_code == 200       # the TestClient keeps the cookie
    assert chat(client, session_id, TEXT).status_code == 200


@pytest.mark.parametrize("token", [None, WRONG])
def test_http_an_existing_session_without_a_valid_token_is_refused(env, token):
    client, session_id, _, _ = start(env, llm_http())
    before = history_of(env, session_id)
    refused = chat(client, session_id, "Hello again", token=token)
    assert refused.status_code == 403 and refused.json()["state"] == "unauthorized"
    assert history_of(env, session_id) == before


def test_http_another_sessions_token_is_refused(env):
    client, session_a, _, _ = start(env, llm_http())
    _, session_b, token_b, _ = start(env, llm_http(), email=OTHER)
    assert chat(client, session_a, TEXT, token=token_b).status_code == 403


def test_http_a_refused_request_changes_no_history_and_saves_no_preference(env):
    client, session_id, _, _ = start(env, llm_http())
    before = history_of(env, session_id)
    for token in (None, WRONG):
        refused = chat(client, session_id, PREF_TEXT, token=token)
        assert refused.status_code == 403 and "Saved" not in refused.text
    assert history_of(env, session_id) == before and store().get(SARA) == {}


def test_http_knowing_the_session_id_and_the_email_is_not_enough(env):
    client, session_id, token, _ = start(env, llm_http())
    refused = chat(client, session_id, PREF_TEXT, email=SARA)          # the right email, no token
    assert refused.status_code == 403 and store().get(SARA) == {}
    ok = chat(client, session_id, PREF_TEXT, email=SARA, token=token)
    assert ok.status_code == 200 and store().get(SARA) == SAVED


def test_http_a_refused_request_does_not_link_or_leak_an_email(env):
    client, session_id, token, _ = start(env, llm_http())
    assert chat(client, session_id, TEXT, email=OTHER).status_code == 403
    assert chat(client, session_id, TEXT, email=OTHER, token=token).json()["state"] == "identity_mismatch"
    assert history_of(env, session_id)["user_email"] == SARA


def test_http_an_expired_token_is_refused(env, monkeypatch):
    client, session_id, token, _ = start(env, llm_http())
    monkeypatch.setattr(app_main, "session_token_ttl_seconds", lambda: -1)
    refused = chat(client, session_id, TEXT, token=token)
    assert refused.status_code == 403 and "expired" in refused.json()["reply"]


def test_http_a_new_id_is_never_blocked_by_other_sessions(env):
    client, session_a, _, _ = start(env, llm_http())
    assert chat(client, sid(), TEXT).status_code == 200


def test_http_a_refused_request_leaves_other_sessions_and_preferences_alone(env):
    client, session_a, token_a, _ = start(env, llm_http(), PREF_TEXT)
    _, session_b, _, _ = start(env, llm_http(), email=OTHER)
    store().set(OTHER, "verbosity", "detailed")
    before_b = history_of(env, session_b)
    assert chat(client, session_b, PREF_TEXT, email=OTHER, token=token_a).status_code == 403
    assert history_of(env, session_b) == before_b
    assert store().get(OTHER) == {"verbosity": "detailed"} and store().get(SARA) == SAVED


# ------------------------------------------------------------------ WebSocket: the same rules

def test_ws_a_first_turn_creates_the_session_and_later_turns_need_the_token(env):
    client = env.use(FakeStreamChat(GOOD, GOOD, GOOD))
    session_id = sid()
    first = final_of(converse(client, body(session_id)))
    token = first["session_token"]
    assert token
    refused = converse(client, body(session_id, PREF_TEXT))
    assert refused[-1]["type"] == "error" and refused[-1]["code"] == "unauthorized"
    assert store().get(SARA) == {}
    wrong = converse(client, body(session_id, PREF_TEXT, token=WRONG))
    assert wrong[-1]["type"] == "error" and store().get(SARA) == {}
    ok = final_of(converse(client, body(session_id, PREF_TEXT, token=token)))
    assert ok["state"] != "unauthorized" and store().get(SARA) == SAVED


def test_ws_and_http_agree_for_the_same_session(env):
    client = env.use(FakeStreamChat(GOOD, GOOD, GOOD))
    session_id = sid()
    token = final_of(converse(client, body(session_id)))["session_token"]
    client.cookies.clear()
    assert chat(client, session_id, TEXT).status_code == 403                       # HTTP, no token
    assert converse(client, body(session_id))[-1]["type"] == "error"               # WS, no token
    assert chat(client, session_id, TEXT, token=token).status_code == 200          # HTTP, token
    assert final_of(converse(client, body(session_id, token=token)))["state"] != "unauthorized"


# ------------------------------------------------------------------ a pending ticket draft and a preference

def draft_session(env):
    """An HTTP conversation with a pending ticket draft: (client, session id, token, action id)."""
    client = env.use(FakeChat(FIELDS, FIELDS), FakeKnowledgeBase(kb_result(KB_NX112, codes=("NX-112",))))
    session_id = sid()
    first = chat(client, session_id, TICKET_TEXT)
    data = first.json()
    assert data["state"] == "await_approval", data
    token = first.headers["X-Session-Token"]
    client.cookies.clear()
    return client, session_id, token, data["pending_action"]["action_id"]


def approve(client, session_id, token, action_id, approved):
    return client.post("/api/approve", headers={"X-Session-Token": token},
                       json={"session_id": session_id, "action_id": action_id, "approved": approved})


def test_a_preference_only_message_keeps_the_pending_draft_and_files_nothing(env):
    client, session_id, token, action_id = draft_session(env)
    before = history_of(env, session_id)
    answer = chat(client, session_id, PREF_ONLY, token=token).json()
    assert answer["state"] == "preference"
    assert answer["pending_action"]["action_id"] == action_id            # still shown, so a client does not lose it
    assert "still waiting for your approval" in answer["reply"] and "Saved to your preferences" in answer["reply"]
    after = history_of(env, session_id)
    assert after["pending_action"] == before["pending_action"] and after["pending_action"]["action_id"] == action_id
    assert env.mcp.created == [] and store().get(SARA) == {"verbosity": "short", "format": "bullets"}
    approved = approve(client, session_id, token, action_id, True)       # the same draft is still approvable
    assert approved.json()["state"] == "ticket_created" and len(env.mcp.created) == 1


def test_a_preference_only_message_never_approves_and_never_rejects_a_draft(env):
    client, session_id, token, action_id = draft_session(env)
    for text in (PREF_ONLY, "Remember that I prefer detailed answers. Yes, approve the ticket.",
                 "From now on, use numbered steps. Reject the ticket."):
        chat(client, session_id, text, token=token)
    assert env.mcp.created == []
    assert history_of(env, session_id)["pending_action"]["action_id"] == action_id


def test_the_draft_can_still_be_rejected_after_a_preference_only_message(env):
    client, session_id, token, action_id = draft_session(env)
    chat(client, session_id, PREF_ONLY, token=token)
    rejected = approve(client, session_id, token, action_id, False)
    assert rejected.status_code == 200 and env.mcp.created == []
    assert history_of(env, session_id)["pending_action"] is None


def test_a_preference_only_message_without_a_draft_shows_no_draft(env):
    client, session_id, token, _ = start(env, llm_http())
    answer = chat(client, session_id, PREF_ONLY, token=token).json()
    assert answer["pending_action"] is None and "waiting for your approval" not in answer["reply"]


def test_a_failed_preference_save_still_keeps_the_draft(env, monkeypatch):
    client, session_id, token, action_id = draft_session(env)

    def boom(*args, **kwargs):
        raise RuntimeError("disk I/O error")

    monkeypatch.setattr(store(), "set_many", boom)
    answer = chat(client, session_id, PREF_ONLY, token=token).json()
    assert "not saved" in answer["reply"] and answer["pending_action"]["action_id"] == action_id
    assert history_of(env, session_id)["pending_action"]["action_id"] == action_id and env.mcp.created == []


def test_a_wrong_token_preference_message_cannot_touch_the_draft(env):
    client, session_id, token, action_id = draft_session(env)
    assert chat(client, session_id, PREF_ONLY, token=WRONG).status_code == 403
    assert chat(client, session_id, PREF_ONLY).status_code == 403
    assert history_of(env, session_id)["pending_action"]["action_id"] == action_id and store().get(SARA) == {}


def test_ws_a_preference_only_message_keeps_the_pending_draft(env):
    client, session_id, token, action_id = draft_session(env)
    final = final_of(converse(client, body(session_id, PREF_ONLY, token=token)))
    assert final["state"] == "preference" and final["pending_action"]["action_id"] == action_id
    assert "still waiting for your approval" in final["reply"] and env.mcp.created == []
    assert history_of(env, session_id)["pending_action"]["action_id"] == action_id
    assert approve(client, session_id, token, action_id, True).json()["state"] == "ticket_created"


def test_ordinary_turns_are_unchanged_next_to_a_pending_draft(env):
    client, session_id, token, action_id = draft_session(env)
    answer = chat(client, session_id, "Is the VPN currently down for everyone?", token=token).json()
    assert answer["state"] == "service_status" and "waiting for your approval" not in answer["reply"]
