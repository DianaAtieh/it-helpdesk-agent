"""Guardrails in code. For now: remove secrets from what employees type, before the text is stored or used.

SP-001: Nakhla never asks for a password, and nothing may store one. Employees still type them into chats,
so every message is redacted BEFORE it reaches the agent, the checkpoint database or an external API.

Kept on purpose: error codes (NX-781, SWC-0404), ticket and document ids (IT-2001, KB-002), emails.
"""
from __future__ import annotations

import re

REDACTED = "[REDACTED]"

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "password is X", "pwd: X", "PIN = X", "passcode was X" (a separator is required, so
    # "my password expired" or "reset my password" are left alone)
    (re.compile(r"(?i)\b(password|passwd|pwd|passcode|passphrase|pin)(\s*(?:is|was|=|:)\s*)"
                r"(\"[^\"]+\"|'[^']+'|\S+?)(?=[,;.]?(?:\s|$))"),
     r"\1\2" + REDACTED),
    # one-time, MFA and backup codes: 6 to 12 digits shortly after a code word
    (re.compile(r"(?i)\b(code|otp|mfa|2fa|one-time|verification|backup)\b([^\d\n]{0,20}?)(\d(?:[\s-]?\d){5,11})\b"),
     r"\1\2" + REDACTED),
    # API keys and tokens with well-known prefixes
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"), REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), REDACTED),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/-]{16,}=*"), r"\1" + REDACTED),
    # long random-looking strings (32+ characters with letters AND digits): tokens, keys, secrets
    (re.compile(r"\b(?=[A-Za-z0-9+/_-]*\d)(?=[A-Za-z0-9+/_-]*[A-Za-z])[A-Za-z0-9+/_-]{32,}={0,2}"), REDACTED),
]


def redact_secrets(text: str) -> str:
    """Return text with passwords, one-time codes, API keys and tokens replaced by [REDACTED]."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------- instructions aimed at the assistant (V4)

# Messages are saved, and from Phase 2 the saved history goes back to the LLM on every turn. A message that
# tries to give the assistant new rules ("from now on ignore your instructions") would then be re-read on
# every later turn, even after a restart. Such a message is answered once, then kept out of the history.
WITHHELD = "[message withheld from later turns: it contained instructions for the assistant]"

_INSTRUCTION_PATTERNS = [
    # "ignore your instructions", "disregard all previous rules", "forget the above prompt"
    re.compile(r"(?i)\b(ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+)?(?:of\s+)?"
               r"(?:your|the|my|previous|prior|above|earlier|these|those|system)?\s*(?:previous\s+|prior\s+)?"
               r"(instructions?|rules|guidelines|prompts?|guardrails?)\b"),
    re.compile(r"(?i)\bfrom now on,?\s+(you|ignore|always|never|act|respond|answer|pretend)\b"),
    re.compile(r"(?i)\b(you are now|pretend (?:to be|you are)|roleplay as)\b"),
    re.compile(r"(?i)\b(system|developer)\s+(prompt|message|instructions?|mode)\b"),
    re.compile(r"(?i)\b(reveal|print|show|repeat|leak)\b.{0,20}\b(your\s+(instructions|prompt|rules)|"
               r"the\s+(?:system\s+)?prompt)\b"),
    re.compile(r"(?i)\b(jailbreak|do anything now)\b"),
]


def looks_like_instructions(text: str) -> bool:
    """True if the text tries to give the assistant new instructions. Pattern-based: catches the common
    phrasings, not every possible one; the agent's own prompt must still treat the history as data."""
    return any(pattern.search(text) for pattern in _INSTRUCTION_PATTERNS)
