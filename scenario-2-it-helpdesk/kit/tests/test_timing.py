"""Timing diagnostics for the WebSocket streaming path (offline): app/timing.py, the marks in the graph, the
streaming helper and /ws/chat, and the Gradio-side UITiming. Diagnostics only: no behaviour depends on them.

    python -m pytest tests/test_timing.py -v
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
from pathlib import Path

from app import timing
from tests.fakes import FakeStreamChat
from tests.test_ws_chat import (GOOD, GOOD_REPLY, SARA, TEXT, body, chunk_text, converse, env, final_of,  # noqa: F401
                                first_turn, sid)

LINE = re.compile(r"timing turn=(\w+) event=(\S+) t_ms=([\d.]+) dt_ms=([\d.-]+)")


def marks(caplog) -> dict[str, float]:
    out = {}
    for record in caplog.records:
        m = LINE.fullmatch(record.getMessage())
        if m:
            out[m.group(2)] = float(m.group(3))
    return out


def summary(caplog) -> dict[str, str]:
    for record in caplog.records:
        text = record.getMessage()
        if text.startswith("timing_summary "):
            return dict(pair.split("=", 1) for pair in text.split()[1:])
    return {}


# ------------------------------------------------------------------ the timer itself

def test_marks_are_once_only_monotonic_and_noop_without_a_timer(caplog):
    caplog.set_level(logging.INFO, logger="helpdesk-app.timing")
    timing.mark("nobody_listening")                              # no active timer in this context
    assert not caplog.records
    timer = timing.TurnTimer("abc")
    timer.mark("a")
    timer.mark("b")
    timer.mark("a")                                              # ignored: first occurrence wins
    assert list(timer.marks) == ["a", "b"] and timer.marks["b"] >= timer.marks["a"] >= 0
    assert [r.getMessage().split()[2] for r in caplog.records] == ["event=a", "event=b"]


def test_phases_are_derived_only_from_marks_that_exist():
    timer = timing.TurnTimer("abc")
    timer.marks = {"ws_request_received": 5.0, "triage_start": 7.0, "triage_end": 107.0,
                   "llm_request_started": 300.0, "llm_first_token": 900.0}
    phases = timer.phases()
    assert phases == {"receive_to_triage_ms": 2.0, "triage_ms": 100.0, "llm_first_token_ms": 600.0}


# ------------------------------------------------------------------ the server's marks on a real turn

def test_a_streamed_turn_logs_every_mark_in_order_and_a_summary(env, caplog):
    caplog.set_level(logging.INFO, logger="helpdesk-app.timing")

    async def pause():
        await asyncio.sleep(0.25)

    llm = FakeStreamChat(["MODE: ANSWER\n", "Update NakhlaConnect to 5.4 from Software Center. ", pause,
                          "Then restart it.\n", "CITATIONS: KB-002"])
    first_turn(env, llm)
    got = marks(caplog)
    order = ["ws_accept_start", "ws_accepted", "ws_request_received", "graph_start", "triage_start", "triage_end",
             "retrieve_start", "retrieve_end",
             "answer_start", "llm_request_started", "llm_first_token", "first_chunk_emitted", "ws_first_chunk_sent",
             "llm_stream_end", "answer_end", "respond_start", "respond_end", "graph_end", "ws_final_sent"]
    assert [m for m in order if m not in got] == []
    times = [got[m] for m in order]
    assert times == sorted(times)
    # Genuinely live: the first chunk was already sent while the model was still generating (250 ms pause).
    assert got["llm_stream_end"] - got["ws_first_chunk_sent"] >= 200
    s = summary(caplog)
    assert float(s["llm_generation_ms"]) >= 200 and s["chunks"] != "0" and s["state"] == "answered"
    assert float(s["request_to_first_chunk_ms"]) < float(s["request_to_final_ms"])


def test_a_slow_model_start_is_attributed_to_the_model_not_to_buffering(env, caplog):
    caplog.set_level(logging.INFO, logger="helpdesk-app.timing")

    async def slow_first_token():
        await asyncio.sleep(0.3)

    llm = FakeStreamChat([slow_first_token, "MODE: ANSWER\n", "Update NakhlaConnect to 5.4 from Software Center. ",
                          "Then restart it.\n", "CITATIONS: KB-002"])
    first_turn(env, llm)
    s = summary(caplog)
    assert float(s["llm_first_token_ms"]) >= 250
    assert float(s["buffering_ms"]) < 100                          # the parser released the sentence immediately


def test_buffering_time_is_visible_when_the_first_sentence_is_slow_to_complete(env, caplog):
    caplog.set_level(logging.INFO, logger="helpdesk-app.timing")

    async def slow_sentence_end():
        await asyncio.sleep(0.3)

    llm = FakeStreamChat(["MODE: ANSWER\n", "Update NakhlaConnect to 5.4 from Software", slow_sentence_end,
                          " Center. Then restart it.\n", "CITATIONS: KB-002"])
    first_turn(env, llm)
    s = summary(caplog)
    assert float(s["buffering_ms"]) >= 250                         # a first token long before the first chunk
    assert float(s["llm_first_token_ms"]) < 150


def test_a_policy_gated_turn_has_no_llm_marks(env, caplog):
    caplog.set_level(logging.INFO, logger="helpdesk-app.timing")
    first_turn(env, FakeStreamChat(GOOD), text="Please turn off MFA for my account")
    got = marks(caplog)
    assert "llm_request_started" not in got and "ws_first_chunk_sent" not in got and "ws_final_sent" in got


def test_timing_logs_contain_no_content_tokens_or_identifiers(env, caplog):
    caplog.set_level(logging.DEBUG)
    session_id = sid()
    client, _, events = first_turn(env, FakeStreamChat(GOOD), session_id=session_id)
    token = final_of(events)["session_token"]
    lines = [r.getMessage() for r in caplog.records if r.name == "helpdesk-app.timing"]
    assert lines and all(re.fullmatch(r"timing(_summary)? turn=\w+ .*", line) for line in lines)
    blob = "\n".join(lines)
    for secret in (token, session_id, SARA, TEXT, GOOD_REPLY, "NakhlaConnect", "KB-002"):
        assert secret not in blob
    turn_ids = {line.split()[1] for line in lines}
    assert turn_ids == {"turn=" + final_of(events)["turn_id"]}


def test_http_chat_does_not_log_timing(env, caplog):
    caplog.set_level(logging.INFO, logger="helpdesk-app.timing")
    from tests.fakes import FakeKnowledgeBase, kb_result
    from tests.test_agent_graph import KB002
    llm = FakeStreamChat(GOOD, json_replies=({"reply": "Update NakhlaConnect to 5.4.", "citations": ["KB-002"],
                                              "needs_clarification": False},))
    client = env.use(llm)
    r = client.post("/api/chat", json={"session_id": sid(), "message": TEXT, "user": {"email": SARA}})
    assert r.status_code == 200 and r.json()["state"] == "answered"
    assert not [r for r in caplog.records if r.name == "helpdesk-app.timing"]


# ------------------------------------------------------------------ the Gradio side

def load_ui():
    spec = importlib.util.spec_from_file_location("app_ui_timing", Path(__file__).resolve().parents[1] / "ui" / "app_ui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ui_logs_when_the_first_chunk_arrived_and_was_displayed(monkeypatch, caplog):
    from tests.test_ui_streaming import FakeServer, chunk, final
    ui = load_ui()
    caplog.set_level(logging.INFO, logger="helpdesk-ui.timing")

    async def late(ws):
        await asyncio.sleep(0.25)

    async def go():
        async with FakeServer([chunk("Update NakhlaConnect to 5.4. ", 1), late, chunk("Then restart it.", 2),
                               final(seq=3)]) as srv:
            monkeypatch.setattr(ui, "APP_URL", f"http://127.0.0.1:{srv.port}")
            async for _ in ui.send_stream("My VPN drops", [], "ui-" + "b" * 32, SARA, None):
                pass
    asyncio.run(go())
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("ui_timing"))
    fields = dict(p.split("=", 1) for p in line.split()[1:])
    assert fields["turn"] == "t1" and fields["chunks"] == "2"
    assert float(fields["max_chunk_gap_ms"]) >= 200
    order = ["connected_ms", "request_sent_ms", "first_event_received_ms", "first_chunk_received_ms",
             "first_chunk_displayed_ms", "final_received_ms", "final_displayed_ms"]
    values = [float(fields[k]) for k in order]
    assert values == sorted(values)
    assert float(fields["final_received_ms"]) - float(fields["first_chunk_received_ms"]) >= 200   # chunk came early
    assert "NakhlaConnect" not in line and "My VPN" not in line
