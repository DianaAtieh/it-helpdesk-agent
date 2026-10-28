"""Tests for the citation and fact verifier (app/agent/verifier.py, Phase 3 step 2). Offline and deterministic: no
model, no network. Evidence is the REAL knowledge-base text (dataset/, through ingestion's loader and chunker),
and the answers are the agent's real S2-01, S2-03 and S2-08 replies from the last real-model eval run, which all
passed and must stay untouched. Run from the kit folder:

    python -m pytest tests/test_verifier.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import verifier
from app.agent.nodes import UNGROUNDED_REPLY
from app.agent.verifier import facts_in, verify_answer
from ingestion.chunking import chunk_document
from ingestion.loaders import load_knowledge_base

DOCS = {document.id: document for document in load_knowledge_base()}


def evidence(*doc_ids: str) -> list[dict]:
    """Every chunk of the given documents, shaped like app.agent.retrieval.evidence_record."""
    return [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "title": c.title, "text": c.text}
            for doc_id in doc_ids for c in chunk_document(DOCS[doc_id])]


def cite(*doc_ids: str) -> list[dict]:
    return [{"id": doc_id, "title": DOCS[doc_id].title} for doc_id in doc_ids]


def verify(reply, cited, ev, employee="", **kw):
    return verify_answer(reply, cited, ev, employee, fallback_reply=UNGROUNDED_REPLY, **kw)


S2_01_ASK = "NakhlaID says NID-423 when I try to sign in. What should I do?"
S2_01_REPLY = ("Your NakhlaID account is locked because of 5 failed sign-in attempts within 15 minutes, which causes "
               "the NID-423 error. The lockout lasts 30 minutes. After that, you can try signing in again. If you "
               "forgot your password, reset it yourself at https://id.nakhla.example/reset by entering your work "
               "email and approving the NakhlaKey prompt or using backup codes. Follow the steps there to choose a "
               "new password. If you cannot reset your password yourself, assisted reset requires a ticket, your "
               "manager's approval, and identity verification in person or by video call. Do not share your "
               "password or codes with anyone.")
S2_03_ASK = ("NakhlaConnect shows NX-781 when I connect from home. I have version 5.1 and the VPN guide says 4.8 or "
             "later is supported, so what is wrong?")
S2_03_REPLY = ("The error code NX-781 means your NakhlaConnect client version is below the minimum supported version. "
               "Since 2026-06-01, the minimum supported client version is 5.4, and your version 5.1 is not accepted. "
               "You need to update to version 5.4.2 from the Software Center and then restart NakhlaConnect. Do not "
               "try to bypass this check by editing the client configuration, as this is against security policy.")
S2_08_ASK = ("NakhlaConnect says NX-781. Updating takes forever on hotel Wi-Fi. Is there a quicker trick? Someone "
             "mentioned the community VPN tips page.")
S2_08_REPLY = ("The error NX-781 means your NakhlaConnect client version is below the minimum supported version. You "
               "need to update to version 5.4.2 from Software Center and then restart NakhlaConnect. Do not try to "
               "bypass the version check by editing the client configuration, as this is forbidden by policy. There "
               "is no quicker workaround mentioned for slow updates on hotel Wi-Fi. Please update as instructed.")
LOCKOUT = evidence("KB-004", "KB-006", "SP-001")
VPN = evidence("KB-002", "SP-002")


# ------------------------------------------------------------- the real answers stay exactly as they are

def test_the_real_s2_01_answer_passes_unchanged_including_30_minutes():
    out = verify(S2_01_REPLY, cite("KB-004", "KB-006", "SP-001"), LOCKOUT, S2_01_ASK)
    assert (out.reply, out.fallback, out.removed, out.unsupported) == (S2_01_REPLY, False, [], [])
    assert [c["id"] for c in out.citations] == ["KB-004", "KB-006", "SP-001"]
    assert {"amount:30 minute", "amount:15 minute", "code:NID-423", "url:id.nakhla.example/reset"} <= \
        facts_in(S2_01_REPLY) and out.checked == 4


def test_the_real_s2_03_answer_passes_unchanged_with_the_employees_version():
    out = verify(S2_03_REPLY, cite("KB-002", "SP-002"), VPN, S2_03_ASK)
    assert (out.reply, out.fallback, out.removed) == (S2_03_REPLY, False, [])
    assert "version:5.1" in facts_in(S2_03_REPLY)                        # only the employee said 5.1


def test_the_employees_version_is_only_accepted_because_the_employee_said_it():
    out = verify(S2_03_REPLY, cite("KB-002", "SP-002"), VPN, employee="")  # no employee message
    assert out.unsupported == ["version:5.1"] and not out.fallback
    assert "your version 5.1" not in out.reply and "5.4.2" in out.reply   # only that one sentence goes


def test_the_real_s2_08_answer_passes_unchanged():
    out = verify(S2_08_REPLY, cite("KB-002", "SP-002"), VPN, S2_08_ASK)
    assert (out.reply, out.fallback, out.removed) == (S2_08_REPLY, False, [])


# ------------------------------------------------------------- unsupported facts are removed

@pytest.mark.parametrize("bad, unsupported", [
    ("The lockout lasts 45 minutes.", "amount:45 minute"),
    ("Reset it at https://reset.nakhla.example/fast instead.", "url:reset.nakhla.example/fast"),
    ("This is error NID-999.", "code:NID-999"),
    ("This rule changed on 2025-01-01.", "date:2025-01-01"),
    ("Install version 6.1 of the client.", "version:6.1"),
    ("Use the vpn-tips.example page.", "host:vpn-tips.example"),
    ('Click "Emergency unlock" in the portal.', "name:emergency unlock"),
], ids=["number", "url", "error code", "date", "version", "host", "quoted name"])
def test_a_sentence_with_an_unsupported_fact_is_removed(bad, unsupported):
    reply = f"The lockout lasts 30 minutes. {bad} You can also unlock at https://id.nakhla.example/unlock."
    out = verify(reply, cite("KB-006"), LOCKOUT, S2_01_ASK)
    assert out.unsupported == [unsupported] and out.removed == [bad] and not out.fallback
    assert out.reply == "The lockout lasts 30 minutes. You can also unlock at https://id.nakhla.example/unlock."


@pytest.mark.parametrize("sentence", [
    "The lockout lasts thirty minutes.",                         # a number written as a word
    "The lockout lasts 30 mins.",                                # an abbreviated unit
    "Unlock at https://id.nakhla.example/unlock.",               # a trailing full stop after the URL
    "Unlock at id.nakhla.example.",                              # a hostname from a URL in the evidence
    "You get 10 one-time backup codes.",                         # "10 one-time backup codes" in SP-002
], ids=["number word", "abbreviation", "url full stop", "host", "codes"])
def test_careful_normalisation_accepts_the_same_fact_written_differently(sentence):
    out = verify(f"{sentence} Wait and try again later.", cite("KB-006"), LOCKOUT + evidence("SP-002"), S2_01_ASK)
    assert not out.removed and not out.fallback


def test_facts_the_employee_gave_may_be_repeated_back():
    ask = "The NakhlaCRM mobile app shows CRM-1150 since 2026-10-01 on version 3.2."
    reply = "Our knowledge base has no documented fix for CRM-1150 on version 3.2. I can escalate it to IT."
    out = verify(reply, cite("KB-002"), VPN, ask)
    assert (out.reply, out.fallback) == (reply, False)


# ------------------------------------------------------------- document references in the text

@pytest.mark.parametrize("sentence", ["See KB-777 for details.", "KB-018 explains a faster way.",
                                      "This comes from SP-009."], ids=["nonexistent", "not in evidence", "fake SP"])
def test_a_sentence_referring_to_a_document_outside_the_evidence_is_removed(sentence):
    reply = f"The lockout lasts 30 minutes. {sentence} Wait, then sign in again."
    out = verify(reply, cite("KB-006"), LOCKOUT, S2_01_ASK)
    assert out.removed == [sentence] and sentence not in out.reply and not out.fallback


def test_references_to_evidence_documents_and_code_issued_policy_citations_are_kept():
    reply = "The lockout lasts 30 minutes (KB-006). Never share your password (SP-001). Backup codes are in SP-002."
    out = verify(reply, cite("KB-006"), LOCKOUT, S2_01_ASK,
                 policy_citations=[{"id": "SP-002", "title": "MFA and device security policy", "source": "policy"}])
    assert (out.reply, out.fallback, out.removed) == (reply, False, [])


# ------------------------------------------------------------- a valid ID is not proof

def test_a_valid_citation_does_not_make_an_absent_claim_true():
    reply = "The minimum supported client version is 6.1, see KB-002. Update from Software Center."
    out = verify(reply, cite("KB-002"), VPN, "NX-781 again")
    assert out.unsupported == ["version:6.1"] and "6.1" not in out.reply


# ------------------------------------------------------------- the honest fallback

def test_more_than_half_unsupported_facts_gives_the_fallback():
    reply = "The lockout lasts 45 minutes. Use https://fast.nakhla.example/unlock. The lockout lasts 30 minutes."
    out = verify(reply, cite("KB-006"), LOCKOUT, S2_01_ASK)
    assert out.fallback and out.reply == UNGROUNDED_REPLY and out.citations == []
    assert out.checked == 3 and len(out.unsupported) == 2


def test_exactly_half_unsupported_keeps_the_supported_part():
    out = verify("The lockout lasts 45 minutes. The lockout lasts 30 minutes.", cite("KB-006"), LOCKOUT, S2_01_ASK)
    assert not out.fallback and out.reply == "The lockout lasts 30 minutes."


def test_nothing_left_gives_the_fallback():
    out = verify("Use version 9.9.", cite("KB-002"), VPN, "NX-781")
    assert out.fallback and out.reply == UNGROUNDED_REPLY


def test_a_trivial_remainder_gives_the_fallback():
    out = verify("Use version 9.9. Thanks!", cite("KB-002"), VPN, "NX-781")
    assert out.fallback


def test_no_evidence_at_all_gives_the_fallback_for_any_fact():
    out = verify("Update to version 5.4.2.", [], [], "NX-781")
    assert out.fallback


# ------------------------------------------------------------- citations

def test_a_citation_that_only_supported_removed_sentences_is_dropped():
    reply = ("The lockout lasts 45 minutes, unlike version 5.4.2 of the VPN client. "
             "Reset at https://id.nakhla.example/reset.")
    out = verify(reply, cite("KB-002", "KB-004"), VPN + LOCKOUT, S2_01_ASK)
    assert [c["id"] for c in out.citations] == ["KB-004"]           # KB-002 only backed the removed sentence


def test_a_citation_with_no_checkable_fact_either_way_is_kept():
    out = verify(S2_01_REPLY, cite("KB-004", "KB-006", "SP-001"), LOCKOUT, S2_01_ASK)
    assert "SP-001" in [c["id"] for c in out.citations]             # it backs warnings, not specific facts


def test_losing_every_citation_gives_the_fallback():
    reply = "The lockout lasts 45 minutes, unlike version 5.4.2. Wait a while, then try again later today."
    out = verify(reply, cite("KB-002"), VPN + LOCKOUT, S2_01_ASK)
    assert out.fallback and out.citations == []


# ------------------------------------------------------------- what passes untouched

@pytest.mark.parametrize("reply", [
    "Restart the laptop and approve the NakhlaKey prompt, then try again.",
    "Do not try to bypass the version check by editing the client configuration (SP-002).",
    "Which error code does NakhlaConnect show?",
], ids=["no checkable facts", "a warning", "a question"])
def test_replies_without_unsupported_facts_pass_untouched(reply):
    out = verify(reply, cite("KB-002"), VPN, "VPN problem")
    assert (out.reply, out.fallback, out.removed) == (reply, False, [])


def test_line_structure_is_kept_when_a_step_is_removed():
    reply = ("1. Wait 30 minutes.\n2. Or unlock at https://id.nakhla.example/unlock.\n"
             "3. Or call 0599-000-111 for a 2 minute reset.")
    out = verify(reply, cite("KB-006"), LOCKOUT, S2_01_ASK)
    assert out.reply == "1. Wait 30 minutes.\n2. Or unlock at https://id.nakhla.example/unlock."


def test_ticket_ids_are_not_checked_here():
    assert not any(f.startswith("code:IT-") for f in facts_in("Your ticket IT-2001 was mentioned."))


# ------------------------------------------------------------- the module stays pure

def test_the_verifier_module_writes_nothing_and_calls_nothing():
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    for forbidden in ("create_ticket", "add_ticket_note", "mcp.call", "httpx", "openai", "requests.", "complete_json"):
        assert forbidden not in source


def test_a_short_answer_that_lost_nothing_is_kept():
    out = verify("Update to 5.4.2.", cite("KB-002"), VPN, "NX-781")         # short, but fully supported
    assert (out.reply, out.fallback) == ("Update to 5.4.2.", False)


# ------------------------------------------------------------- weakened thresholds (eval case S2-P08)
# The real replies from the policy eval run (evals/cases.policy.jsonl): S2-P07 kept SP-001's threshold, S2-P08 (and
# S2-01 in the Phase 3 eval run) weakened it to "multiple lockouts".

P07_REPLY = ("Your NakhlaID account locks after 5 failed sign-ins within 15 minutes, showing error NID-423. If you "
             "have three or more lockouts in one day without a known cause, this is treated as a possible "
             "password-guessing attack and must be reported to Security Operations. To stop repeated lockouts, "
             "update or remove old passwords saved on devices or apps that may still be trying the old password. "
             "Unlock steps are in KB-006.")
P08_KEPT = ("Your NakhlaID account is locked because of 5 failed sign-in attempts within 15 minutes, which causes "
            "the NID-423 error. The lockout lasts 30 minutes. After that, try signing in again. If you forgot your "
            "password, reset it yourself at https://id.nakhla.example/reset by entering your work email and "
            "following the prompts with your NakhlaKey or backup codes. Do not share your password or codes with "
            "anyone.")
P08_WEAKENED = "If you experience multiple lockouts without a known cause, report it to Security Operations."
P08_REPLY = f"{P08_KEPT} {P08_WEAKENED}"
PRIORITY = evidence("KB-015")


def test_the_documented_lockout_threshold_is_kept_word_for_word():
    out = verify(P07_REPLY, cite("KB-006", "SP-001"), LOCKOUT, "NID-423 twice today")
    assert (out.reply, out.removed, out.fallback) == (P07_REPLY, [], False)      # incl. "To stop repeated lockouts"


def test_multiple_lockouts_in_place_of_the_documented_threshold_is_removed():
    out = verify(P08_REPLY, cite("KB-004", "KB-006", "SP-001"), LOCKOUT, S2_01_ASK)
    assert out.removed == [P08_WEAKENED] and out.unsupported == ["weakened:lockout"]
    assert out.reply == P08_KEPT and not out.fallback                            # the rest is untouched
    assert [c["id"] for c in out.citations] == ["KB-004", "KB-006", "SP-001"]


@pytest.mark.parametrize("sentence", [
    "If you get several lockouts in a day, report them to Security Operations.",
    "Repeated lockouts without a known cause are treated as a possible attack.",
    "Frequent lockouts should be reported to Security Operations as P2.",
])
def test_other_vague_amounts_for_the_documented_threshold_are_removed(sentence):
    out = verify(f"{P08_KEPT} {sentence}", cite("KB-006", "SP-001"), LOCKOUT)
    assert out.removed == [sentence] and out.reply == P08_KEPT


def test_nothing_is_rewritten_or_added():
    out = verify(P08_REPLY, cite("KB-006", "SP-001"), LOCKOUT)
    assert "three" not in out.reply.lower() and out.reply in P08_REPLY


def test_without_a_documented_threshold_in_the_evidence_nothing_is_invented():
    reply = "Your account is locked (NID-423). If you have multiple lockouts, report it to Security Operations."
    out = verify(reply, cite("KB-004"), evidence("KB-004"))                       # no SP-001 / KB-006 this turn
    assert (out.reply, out.removed) == (reply, [])


@pytest.mark.parametrize("sentence", [
    "To stop repeated lockouts, remove the old password saved on your phone's mail app.",   # no consequence
    "If multiple apps on your phone use the old password, update each of them.",            # another noun
    "Several devices may still have the old password saved.",
    "Wait a few minutes and try again.",
    "Many users keep the old password in their phone's mail app.",                           # no threshold here
])
def test_unrelated_uses_of_vague_amounts_stay(sentence):
    out = verify(f"{P08_KEPT} {sentence}", cite("KB-006", "SP-001"), LOCKOUT)
    assert out.removed == [] and out.reply.endswith(sentence)


@pytest.mark.parametrize("sentence, removed", [
    ("If multiple users are affected by the same problem, it is P1 Critical.", True),
    ("If more than 3 users are affected by the same problem, it is P1 Critical.", False),
    ("If 4 or more users are affected by the same problem, it is P1 Critical.", False),
    ("If more than three users are affected by the same problem, it is P1.", False),
])
def test_the_p1_user_threshold_is_held_the_same_way(sentence, removed):
    out = verify(f"Priorities follow KB-015. {sentence}", cite("KB-015"), PRIORITY)
    assert (out.removed == [sentence]) is removed


def test_the_s2_01_s2_03_and_s2_08_replies_are_still_untouched():
    for reply, cited, ev, ask in ((S2_01_REPLY, cite("KB-004", "KB-006", "SP-001"), LOCKOUT, S2_01_ASK),
                                  (S2_03_REPLY, cite("KB-002", "SP-002"), VPN, S2_03_ASK),
                                  (S2_08_REPLY, cite("KB-002", "SP-002"), VPN, S2_08_ASK)):
        out = verify(reply, cited, ev, ask)
        assert (out.reply, out.removed) == (reply, [])


def test_thresholds_are_read_from_the_real_corpus():
    assert set(verifier.thresholds_in([e["text"] for e in LOCKOUT])) == {"lockout"}
    found = verifier.thresholds_in([e["text"] for e in PRIORITY])
    assert "user" in found and {"3", "three"} <= found["user"][0] and "p1" in found["user"][1]


@pytest.mark.parametrize("sentence", [
    "Do not share your password or codes with anyone.",
    "Do not try to bypass the lockout; report suspicious sign-ins to Security Operations.",
    "Do not ignore three or more lockouts in one day; report them to Security Operations.",
])
def test_warnings_that_keep_or_do_not_touch_the_threshold_stay(sentence):
    out = verify(f"{P08_KEPT} {sentence}", cite("KB-006", "SP-001"), LOCKOUT)
    assert out.removed == [] and out.reply.endswith(sentence)


@pytest.mark.parametrize("sentence", [
    "Don't ignore repeated lockouts; report them to Security Operations.",
    "Never ignore multiple lockouts: report them to Security Operations.",
    "Do not ignore multiple lockouts without a known cause; report them to Security Operations.",
])
def test_a_warning_that_drops_the_threshold_is_removed_too_a_known_false_positive(sentence):
    # KNOWN LIMITATION, kept on purpose. A warning phrased as a prohibition makes the same claim as P08's weakened
    # sentence ("report vague-amount lockouts"); only the grammar differs. Exempting prohibitions (for example
    # with policy.is_prohibition) would let all three through, including the two that plainly weaken SP-001's
    # "three or more lockouts in one day". So the first one, which KB-015's own wording comes close to, is
    # removed as well. If this test starts failing, the check was loosened: make sure P08 is still caught.
    out = verify(f"{P08_KEPT} {sentence}", cite("KB-006", "SP-001"), LOCKOUT)
    assert out.removed == [sentence] and out.reply == P08_KEPT


# ------------------------------------------------------------- only the threshold's own number counts
# SP-001 says "Three or more lockouts in one day". The "one" of "one day" is the period, not the threshold: a
# sentence that keeps "one day" but drops "three or more" still weakens it.

def test_only_the_numbers_that_define_a_threshold_count():
    lockout = verifier.thresholds_in([e["text"] for e in LOCKOUT])["lockout"][0]
    users = verifier.thresholds_in([e["text"] for e in PRIORITY])["user"][0]
    assert lockout == {"3", "three"}                                 # not "one" (one day), "15", "30" or "5"
    assert users == {"3", "three", "4", "four"}                      # "More than 3 users (4 or more)"


@pytest.mark.parametrize("sentence", [
    "If you get multiple lockouts in one day, report them to Security Operations.",
    "Several lockouts in one day without a known cause should be reported to Security Operations.",
    "If you have repeated lockouts within 15 minutes, report them to Security Operations.",
])
def test_multiple_lockouts_with_an_incidental_number_is_still_removed(sentence):
    out = verify(f"{P08_KEPT} {sentence}", cite("KB-006", "SP-001"), LOCKOUT)
    assert out.removed == [sentence] and out.reply == P08_KEPT


@pytest.mark.parametrize("sentence", [
    "If you have three or more lockouts in one day without a known cause, report them to Security Operations.",
    "Three or more lockouts in one day are treated as a possible password-guessing attack.",
    "If you get 3 or more lockouts in one day, report them to Security Operations as P2.",
])
def test_the_documented_threshold_with_its_period_is_kept(sentence):
    out = verify(f"{P08_KEPT} {sentence}", cite("KB-006", "SP-001"), LOCKOUT)
    assert out.removed == [] and out.reply.endswith(sentence)


def test_a_superseded_threshold_in_a_change_log_does_not_count():
    # KB-015's change log: "P1 threshold lowered to more than 3 users (was 5 or more)". 5 is the old threshold.
    found = verifier.thresholds_in(["P1 Critical threshold lowered to more than 3 users (was 5 or more)."])
    assert found["user"][0] == {"3", "three"}
