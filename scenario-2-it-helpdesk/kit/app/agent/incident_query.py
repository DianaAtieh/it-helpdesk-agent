"""Questions about status-page incidents and gateways ("any ongoing incidents?", "what's happening with GW-2?").

Pure code on one message, no I/O. It does two things:
  * types identifiers: INC-n is a status-page incident, GW-n a gateway, IT-nnnn / REF-... a helpdesk ticket.
    An incident or gateway id is never an error code and never a ticket id.
  * reads a status question about incidents and gateways in two tiers:
      "clear": an incident noun or an INC-/GW- id together with a status ask. Code decides it, whatever the triage
               model said (the status page answers it; the model is never needed for that).
      "weak":  only a gateway noun or a service name with an ask. The triage model decides, as for any other
               message; code never claims it alone.
The caller (nodes.make_triage) keeps every other rule first: security and department rules, a verified error-code
incident, ticket requests, "show my tickets", troubleshooting questions and real error codes.
"""
from __future__ import annotations

import re

INCIDENT_ID = re.compile(r"(?<![A-Za-z0-9])INC-\d{2,6}(?![A-Za-z0-9])", re.IGNORECASE)
GATEWAY_ID = re.compile(r"(?<![A-Za-z0-9])GW-\d{1,3}(?![A-Za-z0-9])", re.IGNORECASE)

_NOUN = re.compile(r"(?i)\b(?:incidents?|outages?|disruptions?)\b")
_GATEWAY_NOUN = re.compile(r"(?i)\bgateways?\b")
_TICKET_WORD = re.compile(r"(?i)\btickets?\b")
# A question or request at the start of a sentence: "Is there ...", "Any ...", "Give me ...", "What's ...".
_ASK_START = re.compile(r"(?i)^\W*(?:is|are|was|were|has|have|does|do|any|anything|what|what's|whats|which|show|give|"
                        r"tell|list|check|can\s+you|could\s+you)\b")
# Words that ask for a state: needed when the message names only an id, so "Do I need GW-2 for the office?" is not one.
_STATE_WORDS = re.compile(r"(?i)\b(?:status|details?|happening|going\s+on|ongoing|active|latest|current\w*|updates?|"
                          r"right\s+now|affected|reported|down|degraded|unavailable|offline)\b")
_WEAK_WORDS = re.compile(r"(?i)\b(?:problems?|issues?|trouble)\b")
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


def incident_ids(text: str) -> list[str]:
    return list(dict.fromkeys(found.upper() for found in INCIDENT_ID.findall(text or "")))


def gateway_ids(text: str) -> list[str]:
    return list(dict.fromkeys(found.upper() for found in GATEWAY_ID.findall(text or "")))


def is_status_identifier(code: str) -> bool:
    """True for an INC-n or GW-n token: a status-page id, not an error code."""
    return bool(INCIDENT_ID.fullmatch(code or "") or GATEWAY_ID.fullmatch(code or ""))


def _asks(text: str) -> bool:
    return any(_ASK_START.search(sentence) for sentence in _SENTENCES.split(text))


def read_scope(text: str, topics_named: bool = False) -> dict | None:
    """What a status question about incidents or gateways asks, or None when the message is not one.

    {"tier": "clear" | "weak", "kind": "incident" | "gateway" | "general", "incidents": [...], "gateways": [...],
     "ticket_word": bool}. topics_named: the message names a service (vpn, mail ...), which only makes it weak."""
    text = text or ""
    incidents, gateways = incident_ids(text), gateway_ids(text)
    asks, states = _asks(text), bool(_STATE_WORDS.search(text))
    if (_NOUN.search(text) and (asks or states)) or ((incidents or gateways) and states):
        tier = "clear"
    elif (_GATEWAY_NOUN.search(text) or topics_named) and (asks or states or _WEAK_WORDS.search(text)):
        tier = "weak"
    else:
        return None
    return {"tier": tier, "kind": "incident" if incidents else "gateway" if gateways else "general",
            "incidents": incidents, "gateways": gateways,
            "ticket_word": bool(incidents and _TICKET_WORD.search(text))}
