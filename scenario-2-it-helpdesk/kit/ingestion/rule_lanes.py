"""Rule lanes (Phase 2D): department exceptions and governing security policies, kept apart from the main evidence.

Given the validated inventory, the evidence signal and the caller's department, build_lane() returns the chunks that
carry the applicable rules. The lanes are ADDITIVE: they never change or consume the top_k evidence, and they never
influence ranking. Their contents are original indexed chunks (text, offsets and citation untouched), selected only
by inventory pointers; the retriever makes no decision about what a rule means.

Safety properties:
  * topical applicability comes from the signal (evidence that is itself eligible), never from the rule chunk's own
    semantic score; department applicability is an exact match on the caller-supplied department (never inferred);
  * every anchor is re-checked with the normal eligibility rules, so draft, deprecated, superseded, non-authoritative
    and department-mismatched chunks can never appear as lane items (they are listed as omitted, with the reason);
  * primary anchors are always returned (their number is bounded by inventory validation); the budget only limits
    SUPPORTING anchors, and every omission is reported, never silent;
  * an anchor already in the main evidence is referenced (in_main_evidence) and its text is not duplicated.

Coverage values (per lane):
  complete              every primary anchor of every applicable inventory rule is returned or referenced.
                        Complete relative to the validated inventory only, not proof that no other rule exists.
  partial               a primary anchor could not be returned (reason listed); a warning is raised.
  department_required   the department is unknown and department rules could apply; only metadata is returned.
  no_inventory_rule     the topic is established but the inventory has no applicable rule. NOT proof that none exists.
  topic_not_established no eligible evidence establishes a topic, so no rule is attached.
  unavailable           the inventory failed validation (or is disabled); a warning is raised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Collection, Mapping

from ingestion.eligibility import ACTIONABLE, CONDITIONAL, Eligibility
from ingestion.hybrid import HybridResult
from ingestion.rule_inventory import DEPARTMENT_EXCEPTION, GOVERNING_POLICY, PRIMARY, Rule, RuleInventory

LANE_DEPARTMENT = "department_rules"
LANE_GOVERNING = "governing_policy"
_KIND = {LANE_DEPARTMENT: DEPARTMENT_EXCEPTION, LANE_GOVERNING: GOVERNING_POLICY}

COMPLETE = "complete"
PARTIAL = "partial"
DEPARTMENT_REQUIRED = "department_required"
NO_INVENTORY_RULE = "no_inventory_rule"
TOPIC_NOT_ESTABLISHED = "topic_not_established"
UNAVAILABLE = "unavailable"

DEFAULT_MAX_DEPARTMENT_RULES = 3
DEFAULT_MAX_GOVERNING_POLICY = 2

NOTE_RULES = ("Rule lanes (department_rules, governing_policy) hold authoritative rule chunks selected from a validated "
              "inventory. Coverage 'complete' is relative to that inventory, and 'no_inventory_rule' is not proof that no "
              "exception exists. The Agent or verifier decides how a rule applies; retrieval makes no such decision.")


@dataclass(frozen=True)
class RuleItem:
    rule_ids: tuple[str, ...]
    role: str  # primary if any listed rule needs it as a primary anchor, else supporting
    in_main_evidence: bool  # also in the main evidence; the text is not repeated in the answer view
    requires_department_verification: bool
    hybrid: HybridResult  # untouched Phase 2B result for the chunk: text, offsets, citation, scores

    @property
    def chunk_id(self) -> str:
        return self.hybrid.chunk_id

    @property
    def doc_id(self) -> str:
        return self.hybrid.chunk.doc_id

    @property
    def citation(self) -> str:
        return self.hybrid.citation


@dataclass(frozen=True)
class OmittedAnchor:
    rule_id: str
    chunk_id: str
    citation: str | None
    role: str
    reason: str  # BUDGET, NOT_IN_POOL, or an eligibility reason code (DRAFT, APPLICABILITY_MISMATCH, ...)


@dataclass(frozen=True)
class RuleIndexEntry:
    """Metadata only, used while the department is unknown. Carries no source text."""
    rule_id: str
    department: str | None
    topic: str
    chunks: tuple[tuple[str, str], ...]  # (chunk_id, citation) of the primary anchors


@dataclass(frozen=True)
class RuleLane:
    lane: str
    coverage: str
    items: tuple[RuleItem, ...] = ()
    omitted: tuple[OmittedAnchor, ...] = ()
    index: tuple[RuleIndexEntry, ...] = ()
    rule_ids: tuple[str, ...] = ()  # inventory rules that applied
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerRuleItem:
    """Answer-facing lane record. text is None when the chunk is already in the main evidence (see chunk_id there)."""
    lane: str
    rule_ids: tuple[str, ...]
    role: str
    in_main_evidence: bool
    requires_department_verification: bool
    chunk_id: str
    doc_id: str
    citation: str
    text: str | None
    title: str
    version: str
    last_updated: str
    status: str
    authoritative: bool
    relative_path: str
    page: int
    heading_path: tuple[str, ...]
    char_start: int
    char_end: int
    doc_sha256: str


def answer_item(lane: str, item: RuleItem) -> AnswerRuleItem:
    chunk = item.hybrid.chunk
    return AnswerRuleItem(
        lane=lane, rule_ids=item.rule_ids, role=item.role, in_main_evidence=item.in_main_evidence,
        requires_department_verification=item.requires_department_verification, chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id, citation=chunk.citation, text=None if item.in_main_evidence else chunk.text,
        title=chunk.title, version=chunk.version, last_updated=chunk.last_updated, status=chunk.status,
        authoritative=chunk.authoritative, relative_path=chunk.relative_path, page=chunk.page,
        heading_path=chunk.heading_path, char_start=chunk.char_start, char_end=chunk.char_end,
        doc_sha256=chunk.doc_sha256)


def unavailable_lane(lane: str, reason: str) -> RuleLane:
    warning = (f"RULE INVENTORY UNAVAILABLE ({reason}): {lane} could not be checked. Do not treat the absence of "
               f"department exceptions or governing policies as 'none apply'; verify against the knowledge base.")
    return RuleLane(lane, UNAVAILABLE, warnings=(warning,))


def build_lane(lane: str, inventory: RuleInventory, signal: Collection[str], department: str | None, *,
               by_chunk: Mapping[str, HybridResult], classify_doc: Callable[[str], Eligibility],
               main_ids: Collection[str], max_supporting: int) -> RuleLane:
    """signal: ids of the eligible, relevant chunks (plus applicability-mismatched current ones) that establish topics."""
    if not signal:
        return RuleLane(lane, TOPIC_NOT_ESTABLISHED)
    kind = _KIND[lane]
    topics = inventory.established_topics(signal)
    rules = inventory.activated(kind, topics, department)
    if not rules:
        return RuleLane(lane, NO_INVENTORY_RULE)
    rule_ids = tuple(r.rule_id for r in rules)

    if kind == DEPARTMENT_EXCEPTION and department is None:
        index = tuple(RuleIndexEntry(
            r.rule_id, r.department, r.topic,
            tuple((a.chunk_id, by_chunk[a.chunk_id].citation) for a in r.primaries if a.chunk_id in by_chunk))
            for r in rules)
        warning = ("Department unknown (never inferred from the question): rules exist for "
                   f"{', '.join(sorted({r.department for r in rules if r.department}))} on this topic. Verify the user's "
                   "department from the directory and search again with the department before advising.")
        return RuleLane(lane, DEPARTMENT_REQUIRED, index=index, rule_ids=rule_ids, warnings=(warning,))

    chosen: dict[str, list] = {}  # chunk_id -> [role, rule ids, requires verification]
    omitted: list[OmittedAnchor] = []
    missing_primary: list[OmittedAnchor] = []

    def check(rule: Rule, anchor) -> tuple[HybridResult | None, str | None, Eligibility | None]:
        result = by_chunk.get(anchor.chunk_id)
        if result is None:
            return None, "NOT_IN_POOL", None
        verdict = classify_doc(result.chunk.doc_id)
        if verdict.tier not in (ACTIONABLE, CONDITIONAL):
            return result, (verdict.reasons[0] if verdict.reasons else "NOT_AUTHORITATIVE"), verdict
        return result, None, verdict

    def add(rule: Rule, anchor, verdict: Eligibility) -> None:
        entry = chosen.setdefault(anchor.chunk_id, [anchor.role, [], False])
        if rule.rule_id not in entry[1]:
            entry[1].append(rule.rule_id)
        entry[2] = entry[2] or verdict.tier == CONDITIONAL
        if anchor.role == PRIMARY:
            entry[0] = PRIMARY

    for rule in rules:  # primaries first: guaranteed, never dropped for budget
        for anchor in rule.primaries:
            result, reason, verdict = check(rule, anchor)
            if reason:
                record = OmittedAnchor(rule.rule_id, anchor.chunk_id, result.citation if result else None, PRIMARY, reason)
                omitted.append(record)
                missing_primary.append(record)
            else:
                add(rule, anchor, verdict)

    def text_bearing() -> int:
        return sum(1 for chunk_id in chosen if chunk_id not in main_ids)

    for rule in rules:
        for anchor in rule.anchors:
            if anchor.role == PRIMARY:
                continue
            result, reason, verdict = check(rule, anchor)
            if reason:
                omitted.append(OmittedAnchor(rule.rule_id, anchor.chunk_id, result.citation if result else None,
                                             anchor.role, reason))
            elif anchor.chunk_id in chosen or anchor.chunk_id in main_ids:
                add(rule, anchor, verdict)
            elif text_bearing() >= max_supporting:
                omitted.append(OmittedAnchor(rule.rule_id, anchor.chunk_id, result.citation, anchor.role, "BUDGET"))
            else:
                add(rule, anchor, verdict)

    items = tuple(RuleItem(tuple(ids), role, chunk_id in main_ids, needs, by_chunk[chunk_id])
                  for chunk_id, (role, ids, needs) in chosen.items())
    warnings = tuple(f"Primary rule evidence unavailable for {m.rule_id}: {m.chunk_id} ({m.reason}). Rule coverage is "
                     f"partial; do not treat the returned rules as the whole exception." for m in missing_primary)
    return RuleLane(lane, PARTIAL if missing_primary else COMPLETE, items, tuple(omitted), (), rule_ids, warnings)
