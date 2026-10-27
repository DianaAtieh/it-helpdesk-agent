"""Unit tests for app/guardrails.py (no API, no network). Run from the kit folder:

    python -m pytest tests/test_guardrails.py -v
"""
from __future__ import annotations

import pytest

from app.guardrails import REDACTED, redact_secrets


@pytest.mark.parametrize("text,secret", [
    ("my laptop password is Hunter2-secret, VPN broken", "Hunter2-secret"),
    ("Password: 'correct horse battery'", "correct horse battery"),
    ("my pwd=abc123!", "abc123!"),
    ("the PIN is 4821", "4821"),
    ("my backup code is 482913", "482913"),
    ("the MFA code 123 456 did not work", "123 456"),
    ("verification code: 99887766", "99887766"),
    ("key sk-or-v1-abcdefghijklmnop1234", "sk-or-v1-abcdefghijklmnop1234"),
    ("token ghp_ABCDEFGHIJKLMNOPQRSTuvwx", "ghp_ABCDEFGHIJKLMNOPQRSTuvwx"),
    ("AKIAABCDEFGHIJKLMNOP is my AWS key", "AKIAABCDEFGHIJKLMNOP"),
    ("Authorization: Bearer abcdefghijklmnop1234567", "abcdefghijklmnop1234567"),
    ("secret 9f8e7d6c5b4a39281706f5e4d3c2b1a0ff", "9f8e7d6c5b4a39281706f5e4d3c2b1a0ff"),
])
def test_secrets_are_redacted(text, secret):
    out = redact_secrets(text)
    assert secret not in out and REDACTED in out


@pytest.mark.parametrize("text", [
    "NakhlaConnect shows NX-781, error code SWC-0404 and PRN-21",
    "my password expired yesterday, please reset my password",
    "ticket IT-2001 about KB-002 and SP-001, incident INC-2207",
    "email sara.mikhail@nakhla.example, client 5.4.2, gateway GW-2",
    "4 users affected since 09:30, the PIN pad on the printer is broken",
    "I need the MFA code reset procedure",
])
def test_ordinary_helpdesk_text_is_left_alone(text):
    assert redact_secrets(text) == text


def test_punctuation_after_a_password_is_kept():
    assert redact_secrets("password is Hunter2-secret, VPN broken") == "password is [REDACTED], VPN broken"
