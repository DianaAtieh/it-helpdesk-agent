"""Buffered live streaming of the answer (opt-in; used by the answer node when a turn asks for it).

The streaming answer is plain text in a small envelope (see prompts.STREAM_FORMAT):

    MODE: ANSWER | CLARIFY
    <text for the employee>
    CITATIONS: KB-002, SP-001        (ANSWER only)

AnswerStreamParser turns the model's deltas into (a) provisional chunks that are safe to show so far, and (b) at
the end the same {"reply", "citations", "needs_clarification"} dict the JSON path produces, so everything after
the model call (citation filtering, clarification check, respond) is unchanged.

What the chunk checks are, and are not. Text is released only in whole sentences (or, for a very long
sentence, up to the last whitespace after MAX_HOLD_CHARS), and every release is checked together with the end of
what was already released, so a forbidden phrase, secret or ticket claim split across deltas is still seen.
This is RISK REDUCTION, not verification: the facts and thresholds check (verifier), the citation check and the
final guardrails need the complete reply and only run in the respond node. Streamed text is therefore
PROVISIONAL. The authoritative reply is the "final" event; reconcile() says whether the client keeps what it
showed or replaces it. A chunk check that trips stops further chunks and asks the client to retract.
"""
from __future__ import annotations

import re
from typing import Callable

MAX_HOLD_CHARS = 240          # a sentence longer than this is released up to its last whitespace
CONTEXT_CHARS = 160           # end of the released text that is re-checked together with each new release
HEADER_MAX_CHARS = 64

_HEADER = re.compile(r"(?i)^\s*MODE\s*:\s*(ANSWER|CLARIFY)\s*$")
_TRAILER = re.compile(r"(?im)^[ \t]*CITATIONS[ \t]*:")
_TRAILER_WORD = "CITATIONS:"
_DOC_ID = re.compile(r"\b[A-Za-z]{2,6}-\d{1,5}\b")
_BOUNDARY = re.compile(r"[.!?]\s+|\n")

CHUNK, RETRACT, FINAL, STATUS = "chunk", "retract", "final", "status"

# Progress of the run, one per graph stage that actually starts. The labels are fixed generic text: nothing from
# the conversation, the directory, the tools or the evidence ever goes into a status event.
STATUS_LABELS = {
    "employee": "Checking employee information",
    "knowledge_base": "Searching the knowledge base",
    "evaluating": "Reviewing the retrieved information",
    "preparing": "Preparing the answer",
    "generating": "Generating the response",
    "ticket": "Preparing your ticket request",
}
NODE_STATUS = {"triage": "employee", "retrieve": "knowledge_base", "evaluate": "evaluating", "answer": "preparing",
               "escalate": "ticket"}      # graph node -> the stage announced when it starts


def status_event(stage: str) -> dict:
    return {"type": STATUS, "stage": stage, "label": STATUS_LABELS[stage]}


def chunk_event(seq: int, text: str) -> dict:
    return {"type": CHUNK, "seq": seq, "text": text, "provisional": True}


def retract_event(reason: str) -> dict:
    """Everything shown so far from this turn's chunks must be removed; the final event follows."""
    return {"type": RETRACT, "reason": reason}


def final_event(reply: str, citations: list[dict], outcome: str) -> dict:
    """The verified reply (respond node): the only text that is authoritative and stored in the history."""
    return {"type": FINAL, "reply": reply, "citations": citations, "outcome": outcome, "provisional": False}


def reconcile(chunks: list[str], retracted: bool, final_reply: str) -> str:
    """What a client does with the verified final reply: "keep" what it already shows, or "replace" it."""
    if retracted:
        return "replace"
    shown = "".join(chunks).strip()
    return "keep" if shown and shown == (final_reply or "").strip() else "replace"


class AnswerStreamParser:
    """Feed it the model's deltas; it returns chunks that passed `violation` (text -> reason | None)."""

    def __init__(self, violation: Callable[[str], str | None]) -> None:
        self._violation = violation
        self._raw = ""                    # only until the MODE line is complete
        self._body = ""                   # everything after the MODE line
        self._released = 0                # how much of _body was handed out as chunks
        self.mode: str | None = None
        self.invalid = False              # the model did not follow the envelope
        self.tripped: str | None = None   # reason of the first failed chunk check: no more chunks
        self.released_any = False

    @property
    def streaming(self) -> bool:
        return self.mode == "ANSWER" and not self.invalid and self.tripped is None

    def feed(self, delta: str) -> list[str]:
        if self.invalid or not delta:
            return []
        if self.mode is None:
            self._raw += delta
            newline = self._raw.find("\n")
            if newline < 0:
                self.invalid = len(self._raw) > HEADER_MAX_CHARS
                return []
            header = _HEADER.match(self._raw[:newline])
            if not header:
                self.invalid = True
                return []
            self.mode = header.group(1).upper()
            self._body = self._raw[newline + 1:].lstrip()
            self._raw = ""
        else:
            self._body += delta
        return self._release(final=False)

    def finish(self) -> tuple[list[str], dict | None]:
        """End of the model's output: the last chunks, and the parsed answer (None if the envelope was broken)."""
        if self.mode is None and not self.invalid:
            header = _HEADER.match(self._raw)
            if header:
                self.mode = header.group(1).upper()
            else:
                self.invalid = True
        if self.invalid:
            return [], None
        chunks = self._release(final=True)
        trailer = _TRAILER.search(self._body)
        text = self._body[:trailer.start()] if trailer else self._body
        ids: list[str] = []
        if trailer:
            first_line = self._body[trailer.end():].split("\n", 1)[0]
            ids = list(dict.fromkeys(_DOC_ID.findall(first_line)))
        return chunks, {"reply": text.strip(), "citations": ids, "needs_clarification": self.mode == "CLARIFY"}

    def _release(self, final: bool) -> list[str]:
        if not self.streaming:                # a clarification is never streamed: it is checked whole, at the end
            return []
        trailer = _TRAILER.search(self._body)
        region = self._body[self._released:trailer.start() if trailer else len(self._body)]
        if trailer or final:
            cut = len(region)
        else:
            line_start = region.rfind("\n") + 1       # a half-written "CITATIONS:" line is never released
            compact = re.sub(r"\s", "", region[line_start:]).upper()
            if compact and _TRAILER_WORD.startswith(compact):
                region = region[:line_start]
            cut = 0
            for boundary in _BOUNDARY.finditer(region):
                cut = boundary.end()
            if cut == 0 and len(region) > MAX_HOLD_CHARS:
                cut = max(region.rfind(" ", 0, MAX_HOLD_CHARS * 2), region.rfind("\n")) + 1
        candidate = region[:cut]
        if not candidate.strip():
            return []
        context = self._body[:self._released][-CONTEXT_CHARS:]
        reason = self._violation(context + candidate)
        if reason:
            self.tripped = reason
            return []
        self._released += cut
        self.released_any = True
        return [candidate]
