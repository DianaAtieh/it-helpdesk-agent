"""Lightweight per-turn timing for the WebSocket streaming path (diagnostics only; no behaviour depends on it).

A TurnTimer records named marks as milliseconds on the monotonic clock since the turn started, keyed by the
turn_id the client also sees in every event. It is reached through a ContextVar, so graph nodes and the
streaming helper call mark("name") without any plumbing; with no timer active (the HTTP path, other tests)
mark() does nothing.

Only names and numbers are ever logged: never prompts, answers, tokens, session ids, e-mails or ticket data.
Log lines (logger "helpdesk-app.timing"):
    timing turn=<id> event=<name> t_ms=<since turn start> dt_ms=<since the previous mark>
    timing_summary turn=<id> key=value ...                   the phases derived from the marks, once per turn
"""
from __future__ import annotations

import contextvars
import logging
import time

log = logging.getLogger("helpdesk-app.timing")

_current: contextvars.ContextVar["TurnTimer | None"] = contextvars.ContextVar("turn_timer", default=None)

# (summary key, from mark, to mark): the phases that explain where the latency of a streamed answer goes.
PHASES = (
    ("ws_accept_ms", "ws_accept_start", "ws_accepted"),                   # server side of the WebSocket handshake
    ("accept_to_request_ms", "ws_accepted", "ws_request_received"),       # client finishing its connect + sending the frame
    ("receive_to_triage_ms", "ws_request_received", "triage_start"),      # session lock, load, redaction, graph start
    ("triage_ms", "triage_start", "triage_end"),                          # directory + status page (MCP)
    ("triage_lookup_ms", "triage_lookup_start", "triage_lookup_end"),     # lookup_employee alone (first turn only)
    ("triage_status_ms", "triage_status_start", "triage_status_end"),     # get_service_status alone
    ("retrieve_ms", "retrieve_start", "retrieve_end"),                    # knowledge base (embeddings + search)
    ("embed_ms", "embed_start", "embed_end"),                             # the query embedding request alone
    ("search_ms", "embed_end", "retrieve_end"),                           # local search + policy filtering after it
    ("evaluate_ms", "evaluate_start", "evaluate_end"),                    # evidence evaluator (LLM), when enabled
    ("escalate_ms", "escalate_start", "escalate_end"),
    ("answer_prep_ms", "answer_start", "llm_request_started"),            # before the answer model is called
    ("llm_first_token_ms", "llm_request_started", "llm_first_token"),     # OpenRouter latency to the first token
    ("buffering_ms", "llm_first_token", "first_chunk_emitted"),           # sentence buffering + chunk checks
    ("emit_to_ws_sent_ms", "first_chunk_emitted", "ws_first_chunk_sent"),
    ("llm_generation_ms", "llm_request_started", "llm_stream_end"),       # the whole answer stream
    ("answer_ms", "answer_start", "answer_end"),
    ("respond_ms", "respond_start", "respond_end"),                       # verifier + guardrails
    ("graph_ms", "graph_start", "graph_end"),
    ("request_to_first_chunk_ms", "ws_request_received", "ws_first_chunk_sent"),
    ("request_to_final_ms", "ws_request_received", "ws_final_sent"),
)


class TurnTimer:
    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self.started = time.perf_counter()
        self.marks: dict[str, float] = {}
        self._last = 0.0

    def mark(self, name: str) -> None:
        """Record `name` the first time it happens in this turn (later calls are ignored)."""
        if name in self.marks:
            return
        t_ms = round((time.perf_counter() - self.started) * 1000, 1)
        self.marks[name] = t_ms
        log.info("timing turn=%s event=%s t_ms=%s dt_ms=%s", self.turn_id, name, t_ms, round(t_ms - self._last, 1))
        self._last = t_ms

    def phases(self) -> dict[str, float]:
        return {key: round(self.marks[b] - self.marks[a], 1)
                for key, a, b in PHASES if a in self.marks and b in self.marks}

    def log_summary(self, **extra: int | str) -> dict[str, float]:
        phases = self.phases()
        fields = " ".join(f"{k}={v}" for k, v in {**phases, **extra}.items())
        log.info("timing_summary turn=%s %s", self.turn_id, fields)
        return phases


def start_turn(turn_id: str) -> TurnTimer:
    timer = TurnTimer(turn_id)
    _current.set(timer)
    return timer


def mark(name: str) -> None:
    timer = _current.get()
    if timer is not None:
        timer.mark(name)
