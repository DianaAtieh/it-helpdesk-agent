"""Hard evidence-eligibility rules (Phase 2C). Pure functions over stored document metadata: no I/O.

Eligibility is independent of relevance: a high semantic score or an exact error-code match can never
make an ineligible chunk usable. Rules, in order:

  * status other than current/deprecated/draft          -> excluded  INVALID_STATUS
  * status draft                                        -> excluded  DRAFT
  * status deprecated                                   -> excluded  DEPRECATED
  * named in `supersedes` by another CURRENT document   -> excluded  SUPERSEDED_BY:<id>
  * empty applies_to                                    -> excluded  INVALID_APPLICABILITY
  * department known and not in a scoped applies_to     -> excluded  APPLICABILITY_MISMATCH
  * otherwise (status current):
        authoritative is not True                       -> tier context_only
        scoped applies_to and department unknown        -> tier conditional (applicability unresolved)
        else                                            -> tier actionable

The department is only ever what the caller supplies; it is never inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

VALID_STATUSES = ("current", "deprecated", "draft")

ACTIONABLE = "actionable"
CONDITIONAL = "conditional"
CONTEXT_ONLY = "context_only"
EXCLUDED = "excluded"

APPLICABLE = "applicable"
UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class DocumentFacts:
    doc_id: str
    status: str
    authoritative: bool
    supersedes: str | None
    applies_to: tuple[str, ...]


@dataclass(frozen=True)
class Eligibility:
    tier: str
    reasons: tuple[str, ...]  # non-empty exactly when tier == EXCLUDED
    applicability: str  # APPLICABLE or UNRESOLVED (meaningful for non-excluded tiers)
    applicable_departments: tuple[str, ...]  # the scoped list, or () when the document applies to all


def is_scoped(applies_to: Iterable[str]) -> bool:
    return "all" not in {value.strip().lower() for value in applies_to}


def superseded_by(documents: Iterable[DocumentFacts]) -> Mapping[str, str]:
    """doc_id -> id of the current document that supersedes it (lowest id wins if several do)."""
    result: dict[str, str] = {}
    for doc in sorted(documents, key=lambda d: d.doc_id):
        target = (doc.supersedes or "").strip()
        if doc.status == "current" and target and target != doc.doc_id:
            result.setdefault(target, doc.doc_id)
    return result


def classify(doc: DocumentFacts, superseders: Mapping[str, str], department: str | None) -> Eligibility:
    reasons: list[str] = []
    if doc.status not in VALID_STATUSES:
        reasons.append("INVALID_STATUS")
    elif doc.status == "draft":
        reasons.append("DRAFT")
    elif doc.status == "deprecated":
        reasons.append("DEPRECATED")
    if doc.doc_id in superseders:
        reasons.append(f"SUPERSEDED_BY:{superseders[doc.doc_id]}")
    scoped = bool(doc.applies_to) and is_scoped(doc.applies_to)
    departments = tuple(doc.applies_to) if scoped else ()
    if not doc.applies_to:
        reasons.append("INVALID_APPLICABILITY")
    elif scoped and department is not None and department.strip().lower() not in {v.strip().lower() for v in doc.applies_to}:
        reasons.append("APPLICABILITY_MISMATCH")
    if reasons:
        return Eligibility(EXCLUDED, tuple(reasons), UNRESOLVED if scoped and department is None else APPLICABLE, departments)
    applicability = UNRESOLVED if scoped and department is None else APPLICABLE
    if doc.authoritative is not True:
        return Eligibility(CONTEXT_ONLY, (), applicability, departments)
    return Eligibility(CONDITIONAL if applicability == UNRESOLVED else ACTIONABLE, (), applicability, departments)
