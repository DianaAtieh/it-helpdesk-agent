"""Security policy precedence in code (Phase 3): deterministic, no model, no network, no ticket writes.

Manifest rule 1: "A document whose id starts with SP- overrides every KB- article on the same point." The
retriever removes deprecated, draft and non-authoritative text, but leaves same-point SP-vs-KB conflicts to the
agent, and the model's own suggestions are not checked by the retriever at all. This module enforces the two
security policies (SP-001, SP-002) at three points:

  check_request()         an employee asks for something a policy forbids: a fixed refusal, before any model call
  filter_evidence()       a non-SP chunk that RECOMMENDS a forbidden action is dropped before the model sees it
  find_forbidden_advice() a model-written reply that RECOMMENDS a forbidden action is replaced by the refusal

Only five rules, each taken from the policy text (nothing beyond SP-001/SP-002 and the directory's contractor
flag). A sentence that prohibits or warns against an action ("Do not edit the client configuration to skip the
check") is never a recommendation. A request that only touches a policy topic ("how do I reset my password?") is
not refused: it gets the normal grounded answer, still subject to the output check.

Refusals cite only the SP document of the matched rule. Those citations are issued by CODE, not by the model
(PolicyHit.citation marks them), so they may be shown even when that chunk was not retrieved this turn. Refusals
never claim a report or a ticket was made: SP-001 section 5 and SP-002 section 4 say such requests are reported to
Security Operations, but reporting is a write and is out of scope here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.agent.escalation import is_vdi_topic, is_vpn_topic

_MANIFEST = Path(__file__).resolve().parents[2] / "dataset" / "manifest.json"
TITLES: dict[str, str] = {d["id"]: d["title"]
                          for d in json.loads(_MANIFEST.read_text(encoding="utf-8"))["documents"]}

POLICY_SOURCE = "policy"          # how a code-issued citation is marked (vs. citations the model chose)

# ---------------------------------------------------------------- the fixed refusals (from the policy text)

REFUSALS = {
    "PHONE_RESET": (
        "I can't reset passwords, read out temporary passwords or change MFA devices over chat, phone or email: "
        "the IT Service Desk stopped doing that on 2026-02-01 (SP-001 section 2). Reset your password yourself at "
        "https://id.nakhla.example/reset and approve it with NakhlaKey (steps in KB-004). If self-service is "
        "impossible, an assisted reset needs a ticket, your manager's approval in that ticket, and your identity "
        "verified in person at an IT Service Desk counter or by video call with your employee badge visible."),
    "ADMIN_CREDENTIAL": (
        "I can't give out administrator, local-admin or service-account passwords: they are held in the Security "
        "Operations vault and are never disclosed through the Service Desk, chat, tickets or any assistant, "
        "whoever asks (SP-001 section 5). Engineering staff who need temporary admin rights can use the "
        "time-limited Elevate function in Software Center (SP-002 section 5)."),
    "SMS_CODE": (
        "SMS and voice-call codes are not allowed: they were switched off on 2026-03-01, and nobody at Nakhla can "
        "send you an SMS code (SP-002 section 1). If you don't have your phone, sign in with one of your one-time "
        "backup codes; the steps for a new phone are in KB-007."),
    "WEAKEN_SECURITY": (
        "I can't help with that: disabling disk encryption, antivirus or the NakhlaManage agent, bypassing or "
        "turning off the NakhlaConnect posture or version check, and using any VPN client other than "
        "NakhlaConnect are forbidden (SP-002 section 4). If NakhlaConnect says your version is too old, update it "
        "from Software Center (KB-002)."),
    "CONTRACTOR_VPN": (
        "Contractors have no VPN access: you work only through the Nakhla Workspace VDI portal at "
        "https://vdi.nakhla.example (SP-002 section 3). If the VDI portal itself doesn't work, tell me what you "
        "see and I can help with that."),
    "MFA_OPT_OUT": (
        "I can't help with turning off, skipping or getting an exemption from MFA: it is mandatory for every "
        "NakhlaID account, including contractors (SP-002 section 1). If NakhlaKey is the problem, for example a "
        "new phone, tell me what you see; the steps for a new phone are in KB-007."),
}
RULE_DOC = {"PHONE_RESET": "SP-001", "ADMIN_CREDENTIAL": "SP-001", "SMS_CODE": "SP-002",
            "WEAKEN_SECURITY": "SP-002", "CONTRACTOR_VPN": "SP-002", "MFA_OPT_OUT": "SP-002"}
RULE_SECTION = {"PHONE_RESET": "SP-001 section 2", "ADMIN_CREDENTIAL": "SP-001 section 5",
                "SMS_CODE": "SP-002 section 1", "WEAKEN_SECURITY": "SP-002 section 4",
                "CONTRACTOR_VPN": "SP-002 section 3", "MFA_OPT_OUT": "SP-002 section 1"}


@dataclass(frozen=True)
class PolicyHit:
    rule: str          # PHONE_RESET | ADMIN_CREDENTIAL | SMS_CODE | WEAKEN_SECURITY | CONTRACTOR_VPN | MFA_OPT_OUT
    doc_id: str        # the SP document the rule comes from
    section: str
    reply: str         # the fixed refusal

    @property
    def citation(self) -> dict:
        """The one citation a refusal may carry: the rule's own SP document, marked as issued by code."""
        return {"id": self.doc_id, "title": TITLES.get(self.doc_id, ""), "source": POLICY_SOURCE}


def hit(rule: str) -> PolicyHit:
    return PolicyHit(rule, RULE_DOC[rule], RULE_SECTION[rule], REFUSALS[rule])


# ---------------------------------------------------------------- text helpers

_CURLY = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_MARKUP = re.compile(r"[*`]")          # bold and code marks only: "_" is kept (posture_check=off)

# A sentence that prohibits or warns against an action is never a recommendation of it.
_PROHIBITION = re.compile(
    r"(?i)\b(?:do\s+not|don't|dont|never|must\s+not|mustn't|should\s+not|shouldn't|may\s+not|cannot|can't|"
    r"not\s+allowed|forbidden|prohibited|not\s+permitted|against\s+(?:the\s+)?(?:security\s+)?polic(?:y|ies)|"
    r"no\s+longer|not\s+possible|are\s+not|is\s+not|isn't|aren't|won't|will\s+not|refused)\b")
# A request in the negative ("I don't want to bypass the check") is not a request.
_NEGATED_REQUEST = re.compile(
    r"(?i)\b(?:don't|do\s+not|never|won't|will\s+not)\s+(?:want\s+to\s+|need\s+to\s+|plan\s+to\s+|intend\s+to\s+)?"
    r"(?:\w+\s+)?(?:bypass|skip|disable|turn|switch|reset|change|share|use|send|get|give)\b")


def sentences(text: str) -> list[str]:
    text = _MARKUP.sub("", (text or "").translate(_CURLY))
    return [s.strip() for s in _SENTENCES.split(text) if s.strip()]


def is_prohibition(sentence: str) -> bool:
    return bool(_PROHIBITION.search(sentence))


# ---------------------------------------------------------------- what an employee asks for

_ASK = r"(?:can|could|would|will)\s+you\s+(?:please\s+)?|please\s+|i\s+(?:need|want)\s+you\s+to\s+"
_CHANNEL = r"(?:over|by|on|via|through)\s+(?:the\s+)?(?:phone|telephone|chat|e-?mail)"

# SP-002 section 1: MFA is mandatory for every NakhlaID account. Turning it off, skipping it or being exempt
# from it is never something the desk or this agent may help with (or recommend).
_MFA = r"(?:mfa|2fa|two[-\s]factor(?:\s+authentication)?|multi[-\s]?factor(?:\s+authentication)?)"
_MFA_OPT_OUT = (
    rf"\b(?:disable|turn\s+off|switch\s+off|bypass|skip|get\s+around|get\s+past|waive|opt\s+out\s+of)\s+"
    rf"(?:the\s+|my\s+|our\s+|all\s+|their\s+|your\s+)?(?:account\s+|nakhlaid\s+)?(?:{_MFA}|nakhlakey(?:\s+prompts?)?)\b"
    rf"|\b(?:exempt|exemption|exception)\b[^.?!]{{0,30}}\b(?:from\s+)?(?:the\s+)?(?:{_MFA}|nakhlakey)\b"
    rf"|\b(?:sign|log)\s*in\s+without\s+(?:{_MFA}|nakhlakey)\b")

_REQUESTS: dict[str, re.Pattern] = {
    # SP-001 section 2: the desk does not reset passwords, read out temporary passwords or change MFA devices
    # over the phone, chat or email.
    "PHONE_RESET": re.compile(
        rf"(?i)(?:{_ASK})(?:reset|change|unlock\s+and\s+reset)\s+(?:my\s+)?(?:nakhlaid\s+)?password"
        rf"|\b(?:reset|change)\s+my\s+(?:nakhlaid\s+)?password\s+(?:for\s+me|{_CHANNEL})"
        rf"|\bpassword\s+reset\s+{_CHANNEL}"
        r"|\b(?:give|tell|send|read)\s+me\s+(?:a\s+|the\s+)?(?:temporary|temp|new|one-time)\s+password"
        rf"|(?:{_ASK})(?:change|move|reset|switch|remove)\s+my\s+(?:mfa|nakhlakey)\b"
        rf"|\b(?:change|move|reset|switch)\s+my\s+(?:mfa|nakhlakey)\s+(?:device\s+|phone\s+)?(?:for\s+me|{_CHANNEL})"),
    # SP-001 section 5: administrator, local-admin and service-account passwords are never disclosed.
    "ADMIN_CREDENTIAL": re.compile(
        r"(?i)\b(?:what\s+is|what's|give|gives|tell|tells|send|sends|share|shares|provide|provides|reveal|reveals|"
        r"need|needs|want|wants|have|know|get|gets)\b[^.?!]{0,40}?"
        r"\b(?:admin(?:istrator)?|local[\s-]?admin|root|service[\s-]?account)\s+(?:account\s+)?"
        r"(?:password|credentials?|login)\b"),
    # SP-002 section 1: SMS and voice-call codes are not allowed.
    "SMS_CODE": re.compile(
        r"(?i)\b(?:send|sends|sending|give|gives|get|gets|need|needs|want|wants|receive|receives|request|"
        r"requests|text|texts)\b[^.?!]{0,30}?"
        r"\b(?:sms|text[\s-]message|voice(?:[\s-]call)?)\s+(?:code|otp|verification\s+code)\b"
        r"|\b(?:send|text)\s+me\s+(?:a\s+|the\s+)?code\s+by\s+(?:sms|text)\b"),
    # SP-002 section 4: do not weaken device security.
    "WEAKEN_SECURITY": re.compile(
        r"(?i)\b(?:bypass|skip|disable|turn\s+off|switch\s+off|get\s+around|get\s+past|avoid)\s+(?:the\s+)?"
        r"(?:\w+\s+){0,2}?(?:posture|version|compliance)\s+check"
        r"|\b(?:disable|turn\s+off|switch\s+off|uninstall|remove|stop)\s+(?:the\s+)?"
        r"(?:disk\s+encryption|bitlocker|encryption|antivirus|anti-virus|nakhlamanage(?:\s+agent)?)\b"
        r"|\b(?:use|install|connect\s+with)\s+(?:a\s+|an\s+)?(?:another|other|different|my\s+own|own|personal|"
        r"private|third[-\s]party)\s+vpn\b"),
    "MFA_OPT_OUT": re.compile(r"(?i)" + _MFA_OPT_OUT),
}


def check_request(text: str, employee: dict | None = None,
                  codes: list[str] | tuple[str, ...] = ()) -> PolicyHit | None:
    """The rule an employee's message asks the agent to break, or None. Only clear requests count."""
    if isinstance(employee, dict) and employee.get("contractor") is True \
            and is_vpn_topic(text, codes) and not is_vdi_topic(text):
        return hit("CONTRACTOR_VPN")                     # SP-002 section 3, from the directory's contractor flag
    for sentence in sentences(text):
        if _NEGATED_REQUEST.search(sentence):
            continue
        for rule, pattern in _REQUESTS.items():
            if pattern.search(sentence):
                return hit(rule)
    return None


# ---------------------------------------------------------------- what a recommendation looks like

_ADVICE: dict[str, re.Pattern] = {
    "WEAKEN_SECURITY": re.compile(
        r"(?i)posture_?check\s*=\s*(?:off|false|0)"
        r"|\b(?:edit|change|modify|open)\b[^.]{0,60}\b(?:client\.ini|config(?:uration)?(?:\s+file)?)\b[^.]{0,60}"
        r"\b(?:skip|bypass|disable|turn\s+off|switch\s+off)\b"
        r"|\b(?:skip|bypass|disable|turn\s+off|switch\s+off)\s+(?:the\s+)?(?:\w+\s+){0,2}?"
        r"(?:posture|version|compliance)\s+check"
        r"|\b(?:disable|turn\s+off|uninstall)\s+(?:the\s+)?(?:disk\s+encryption|bitlocker|antivirus|"
        r"nakhlamanage(?:\s+agent)?)\b"),
    "SMS_CODE": re.compile(
        r"(?i)\b(?:ask|request|get|receive|use)\b[^.]{0,40}\b(?:sms|text[\s-]message|voice(?:[\s-]call)?)\s+"
        r"(?:code|otp)\b"
        r"|\b(?:we|i|the\s+(?:it\s+)?(?:service\s+)?desk|it|the\s+helpdesk)\s+(?:can|will)\s+send\s+(?:you\s+)?"
        r"(?:an?\s+)?(?:sms|text|voice)"),
    "PHONE_RESET": re.compile(
        r"(?i)\b(?:call|phone|ring|contact)\s+(?:the\s+)?(?:it\s+)?(?:service\s+)?desk\b[^.]{0,60}"
        r"\b(?:reset|temporary\s+password|read)"
        r"|\b(?:read|give)\s+(?:you\s+)?(?:a\s+|the\s+)?temporary\s+password"
        r"|\b(?:the\s+(?:it\s+)?(?:service\s+)?desk|we|i|it)\s+(?:can|will)\s+reset\s+your\s+password"),
    "ADMIN_CREDENTIAL": re.compile(
        r"(?i)\b(?:admin(?:istrator)?|local[\s-]?admin|root|service[\s-]?account)\s+(?:account\s+)?password\s+"
        r"(?:is|was|:|=)\s*\S"
        r"|\bpassword\s+for\s+(?:the\s+)?(?:local\s+)?(?:admin(?:istrator)?|root|service)\b[^.]{0,30}\b(?:is|:)\s*\S"),
    "MFA_OPT_OUT": re.compile(r"(?i)" + _MFA_OPT_OUT),
}


def find_forbidden_advice(text: str) -> PolicyHit | None:
    """The rule a reply (or a chunk) RECOMMENDS breaking, or None. Prohibitions and warnings never count."""
    for sentence in sentences(text):
        if is_prohibition(sentence):
            continue
        for rule, pattern in _ADVICE.items():
            if pattern.search(sentence):
                return hit(rule)
    return None


def filter_evidence(evidence: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop non-SP chunks that recommend a forbidden action (SP wins, manifest rule 1). Returns (kept, dropped
    chunk ids). SP chunks are never dropped."""
    kept, dropped = [], []
    for record in evidence or []:
        doc_id = str(record.get("doc_id", ""))
        if not doc_id.startswith("SP-") and find_forbidden_advice(str(record.get("text", ""))):
            dropped.append(str(record.get("chunk_id") or doc_id))
            continue
        kept.append(record)
    return kept, dropped
