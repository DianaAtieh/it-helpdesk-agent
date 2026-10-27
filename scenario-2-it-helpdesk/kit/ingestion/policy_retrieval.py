"""Policy-aware evidence selection on top of hybrid retrieval (Phase 2C).

Pipeline (deterministic, one query-embedding call, read-only):
  1. Candidate pool = the COMPLETE index, ranked by HybridRetriever (nothing is dropped before step 2).
  2. Eligibility (ingestion.eligibility) is decided from stored metadata only. Relevance never overrides it.
  3. Relevance among eligible chunks: exact error-code match (bypasses the floor, but only for eligible
     chunks) or semantic score >= min_semantic_score (default 0.25, calibrated on 11 real queries).
  4. actionable + conditional share the top_k budget; context_only is capped separately (default 3);
     diagnostics (excluded chunks) never use the answer budget.

Known limitations (documented, not implemented):
  1. Out-of-scope queries can still retrieve semantically similar authoritative documents.
  2. Same-point SP-vs-KB conflicts require downstream verification.
  3. Sufficiency ("sufficient" etc.) only says eligible, relevant-scoring evidence exists; it does not
     guarantee that the evidence actually resolves the user's problem.

Not solved here, and left to the downstream agent/verifier:
  * whether the question is in scope: the floor separates clearly unrelated queries (top score <= 0.17
    on the calibration set) but NOT in-domain-sounding out-of-scope ones (e.g. "reset my Netflix password"
    scored 0.53 against KB-004);
  * same-point conflicts between SP- and KB- content among current documents. Eligibility removes
    deprecated/draft/superseded text structurally; `governing_policy_chunks` only lists the relevant
    actionable SP chunks so the verifier can apply "SP overrides KB" - it does not boost them in the ranking
    and does not decide whether two chunks address the same point;
  * whether an error code is documented does not mean any step fixes it ("documented" = the code appears
    in eligible evidence as a whole token).
Retrieved text is untrusted data. Excluded text is withheld unless explicitly requested for diagnostics.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ingestion.conflicts import ConflictAnnotation, KnownConflict, ManifestError, annotate, load_known_conflicts
from ingestion.eligibility import (ACTIONABLE, CONDITIONAL, CONTEXT_ONLY, EXCLUDED, UNRESOLVED, DocumentFacts, classify,
                                   superseded_by)
from ingestion.embeddings import EmbeddingProvider
from ingestion.hybrid import HybridResult, HybridRetriever
from ingestion.retrieval import QueryError

DEFAULT_MIN_SEMANTIC_SCORE = 0.25
DEFAULT_MAX_CONTEXT = 3
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "dataset" / "manifest.json"

NOTE_CODES = ("A documented error code only means the code appears in eligible evidence as a whole token; "
              "it does not confirm that any step fixes it.")
NOTE_CONFLICTS = ("Same-point conflicts between security policies (SP-) and KB articles are not resolved here: "
                  "SP governs, and the downstream verifier must apply that rule.")
NOTE_SCOPE = ("Whether the question is in scope is not validated; a semantic score does not establish that the "
              "evidence answers the question.")
NOTE_DEPARTMENT = ("Items with requires_department_verification apply only to applicable_departments; verify the "
                   "user's department from the directory before using them.")


@dataclass(frozen=True)
class EvidenceItem:
    tier: str  # actionable | conditional | context_only
    policy_rank: int  # 1-based within its group (actionable+conditional share one order; context_only its own)
    hybrid: HybridResult  # untouched Phase 2B result: text, offsets, citation, scores, exact-code evidence
    applicability: str  # applicable | unresolved
    requires_department_verification: bool
    applicable_departments: tuple[str, ...]  # scoped department list, () when the document applies to all

    @property
    def chunk_id(self) -> str:
        return self.hybrid.chunk_id

    @property
    def doc_id(self) -> str:
        return self.hybrid.chunk.doc_id

    @property
    def citation(self) -> str:
        return self.hybrid.citation

    @property
    def documents_query_codes(self) -> tuple[str, ...]:
        return self.hybrid.matched_codes


@dataclass(frozen=True)
class ExcludedItem:
    """Diagnostics only. Carries no source text unless include_diagnostic_text was requested."""
    chunk_id: str
    doc_id: str
    status: str
    authoritative: bool
    reasons: tuple[str, ...]
    semantic_score: float
    semantic_rank: int
    matched_codes: tuple[str, ...]
    citation: str
    untrusted_diagnostic_text: str | None = None


@dataclass(frozen=True)
class ApplicabilityReport:
    department: str | None
    department_known: bool
    requires_department_verification: bool  # some returned answer evidence is conditional on the department
    unresolved_documents: tuple[str, ...]
    required_departments: tuple[str, ...]


@dataclass(frozen=True)
class Sufficiency:
    level: str  # sufficient (>=1 actionable) | conditional_only | context_only | none
    relevant_actionable: int
    relevant_conditional: int
    relevant_context_only: int
    eligible_below_relevance: int  # eligible chunks that were neither above the floor nor exact matches
    excluded_relevant: int  # relevant-scoring chunks removed by eligibility (see diagnostics)
    truncated_by_top_k: int  # relevant actionable/conditional chunks that did not fit in top_k
    best_eligible_score: float | None
    min_semantic_score: float


@dataclass(frozen=True)
class AnswerEvidence:
    """Plain, answer-facing record. Built only from actionable and conditional evidence."""
    chunk_id: str
    doc_id: str
    tier: str
    text: str  # original source text, untrusted data
    citation: str
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
    semantic_score: float
    documents_query_codes: tuple[str, ...]
    applicability: str
    requires_department_verification: bool
    applicable_departments: tuple[str, ...]


@dataclass(frozen=True)
class AnswerView:
    evidence: tuple[AnswerEvidence, ...]
    query_codes: tuple[str, ...]
    code_coverage: Mapping[str, str]
    undocumented_codes: tuple[str, ...]
    department: str | None
    requires_department_verification: bool
    sufficiency: Sufficiency
    scope_validated: bool  # always False: scope and same-point conflicts are the verifier's job
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PolicyResponse:
    actionable: tuple[EvidenceItem, ...]
    conditional: tuple[EvidenceItem, ...]
    context_only: tuple[EvidenceItem, ...]
    diagnostics: tuple[ExcludedItem, ...]
    query_codes: tuple[str, ...]
    code_coverage: Mapping[str, str]  # code -> documented | undocumented (never "fix confirmed")
    applicability: ApplicabilityReport
    sufficiency: Sufficiency
    governing_policy_chunks: tuple[str, ...]  # relevant actionable SP chunks; a view, not a ranking boost
    conflict_annotations: tuple[ConflictAnnotation, ...]  # opt-in, diagnostic only
    annotation_warning: str | None
    department: str | None
    pool_size: int
    pool_complete: bool

    @property
    def undocumented_codes(self) -> tuple[str, ...]:
        return tuple(code for code, status in self.code_coverage.items() if status == "undocumented")

    def _answer(self, item: EvidenceItem) -> AnswerEvidence:
        chunk = item.hybrid.chunk
        return AnswerEvidence(
            chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, tier=item.tier, text=chunk.text, citation=chunk.citation,
            title=chunk.title, version=chunk.version, last_updated=chunk.last_updated, status=chunk.status,
            authoritative=chunk.authoritative, relative_path=chunk.relative_path, page=chunk.page,
            heading_path=chunk.heading_path, char_start=chunk.char_start, char_end=chunk.char_end,
            doc_sha256=chunk.doc_sha256, semantic_score=item.hybrid.semantic_score,
            documents_query_codes=item.documents_query_codes, applicability=item.applicability,
            requires_department_verification=item.requires_department_verification,
            applicable_departments=item.applicable_departments)

    def for_answering(self) -> AnswerView:
        """The only supported input for answer generation: actionable and conditional evidence, nothing else."""
        items = sorted((*self.actionable, *self.conditional), key=lambda i: i.policy_rank)
        needs = self.applicability.requires_department_verification
        notes = [NOTE_CODES, NOTE_CONFLICTS, NOTE_SCOPE] + ([NOTE_DEPARTMENT] if needs else [])
        return AnswerView(tuple(self._answer(i) for i in items), self.query_codes, self.code_coverage,
                          self.undocumented_codes, self.department, needs, self.sufficiency, False, tuple(notes))

    def context_for_background(self) -> tuple[AnswerEvidence, ...]:
        """Current but non-authoritative text. Background only: never instructions, never cited as a basis."""
        return tuple(self._answer(i) for i in self.context_only)


def _date_ordinal(value: str) -> int:
    try:
        return date.fromisoformat(value).toordinal()
    except ValueError:
        return 0


def _version_parts(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return ()


def _rank_key(result: HybridResult):
    chunk = result.chunk
    return (-len(result.matched_codes), -round(result.semantic_score, 9), -_date_ordinal(chunk.last_updated),
            tuple(-part for part in _version_parts(chunk.version)), chunk.chunk_id)


class PolicyRetriever:
    def __init__(self, db_path: Path | str, provider: EmbeddingProvider, *,
                 min_semantic_score: float = DEFAULT_MIN_SEMANTIC_SCORE, max_context: int = DEFAULT_MAX_CONTEXT,
                 manifest_path: Path | str | None = None) -> None:
        if (isinstance(min_semantic_score, bool) or not isinstance(min_semantic_score, (int, float))
                or not math.isfinite(min_semantic_score) or not -1.0 <= min_semantic_score <= 1.0):
            raise QueryError("min_semantic_score must be a finite number between -1 and 1")
        if not isinstance(max_context, int) or isinstance(max_context, bool) or max_context < 0:
            raise QueryError("max_context must be a non-negative integer")
        self.min_semantic_score = float(min_semantic_score)
        self.max_context = max_context
        self.hybrid = HybridRetriever(db_path, provider)
        facts: dict[str, DocumentFacts] = {}
        for row in self.hybrid.semantic.chunk_rows:
            facts.setdefault(row["doc_id"], DocumentFacts(
                row["doc_id"], row["status"], bool(row["authoritative"]), row["supersedes"],
                tuple(json.loads(row["applies_to"]))))
        self._facts = facts
        self._superseders = superseded_by(facts.values())
        self._conflicts: tuple[KnownConflict, ...] = ()
        self.annotation_warning: str | None = None
        if manifest_path is not None:
            try:
                self._conflicts = load_known_conflicts(manifest_path)
            except ManifestError as exc:
                self.annotation_warning = f"conflict annotations disabled: {exc}"

    @property
    def skipped_invalid(self) -> list[tuple[str, str]]:
        return self.hybrid.skipped_invalid

    @property
    def dimensions(self) -> int | None:
        return self.hybrid.dimensions

    @property
    def provider(self) -> EmbeddingProvider:
        return self.hybrid.provider

    def __len__(self) -> int:
        return len(self.hybrid)

    def search(self, query: str, top_k: int = 5, *, department: str | None = None,
               include_diagnostic_text: bool = False) -> PolicyResponse:
        if department is not None and (not isinstance(department, str) or not department.strip()):
            raise QueryError("department must be a non-empty string or None")
        department = department.strip() if department is not None else None
        self.hybrid.semantic.check_search_args(query, top_k)  # the pool search below uses its own top_k
        pool = self.hybrid.search(query, top_k=len(self.hybrid))  # one embedding call
        answer, context, diagnostics = [], [], []
        eligible_below = 0
        for result in pool.results:
            chunk = result.chunk
            verdict = classify(self._facts[chunk.doc_id], self._superseders, department)
            relevant = bool(result.matched_codes) or result.semantic_score >= self.min_semantic_score
            if verdict.tier == EXCLUDED:
                if relevant:
                    text = chunk.text if include_diagnostic_text else None
                    diagnostics.append(ExcludedItem(
                        chunk.chunk_id, chunk.doc_id, chunk.status, chunk.authoritative, verdict.reasons,
                        result.semantic_score, result.semantic_rank, result.matched_codes, chunk.citation, text))
            elif not relevant:
                eligible_below += 1
            elif verdict.tier == CONTEXT_ONLY:
                context.append((result, verdict))
            else:
                answer.append((result, verdict))
        answer.sort(key=lambda pair: _rank_key(pair[0]))
        context.sort(key=lambda pair: _rank_key(pair[0]))
        kept = answer[:top_k]
        actionable, conditional = [], []
        for rank, (result, verdict) in enumerate(kept, start=1):
            item = EvidenceItem(verdict.tier, rank, result, verdict.applicability, verdict.tier == CONDITIONAL,
                                verdict.applicable_departments)
            (conditional if verdict.tier == CONDITIONAL else actionable).append(item)
        context_items = tuple(EvidenceItem(CONTEXT_ONLY, rank, result, verdict.applicability,
                                           verdict.applicability == UNRESOLVED, verdict.applicable_departments)
                              for rank, (result, verdict) in enumerate(context[:self.max_context], start=1))
        diagnostics.sort(key=lambda d: (-round(d.semantic_score, 9), d.chunk_id))

        documented = {code for result, _ in answer for code in result.matched_codes}
        coverage = MappingProxyType({code: "documented" if code in documented else "undocumented"
                                     for code in pool.query_codes})
        unresolved = tuple(dict.fromkeys(i.doc_id for i in (*actionable, *conditional) if i.requires_department_verification))
        required = tuple(dict.fromkeys(d for i in conditional for d in i.applicable_departments))
        applicability = ApplicabilityReport(department, department is not None, bool(conditional), unresolved, required)
        level = ("sufficient" if actionable else "conditional_only" if conditional
                 else "context_only" if context_items else "none")
        best = max((r.semantic_score for r, _ in answer), default=None)
        sufficiency = Sufficiency(level, sum(v.tier == ACTIONABLE for _, v in answer),
                                  sum(v.tier == CONDITIONAL for _, v in answer), len(context), eligible_below,
                                  len(diagnostics), len(answer) - len(kept), best, self.min_semantic_score)
        governing = tuple(i.chunk_id for i in actionable if i.doc_id.startswith("SP-"))
        relevant_docs = {r.chunk.doc_id for r, _ in (*answer, *context)} | {d.doc_id for d in diagnostics}
        annotations = annotate(self._conflicts, relevant_docs) if self._conflicts else ()
        return PolicyResponse(tuple(actionable), tuple(conditional), context_items, tuple(diagnostics), pool.query_codes,
                              coverage, applicability, sufficiency, governing, annotations, self.annotation_warning,
                              department, len(pool.results), len(pool.results) == len(self.hybrid))
