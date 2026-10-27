"""Offline tests for ingestion.policy_retrieval and the policy CLI mode.

Fake embeddings and temporary SQLite indexes only; no network. Relevance tests use CONTROLLED cosine scores:
after indexing, stored vectors are overwritten so that each chunk has an exact chosen cosine to the fixed
query vector. That lets the tests replay the real calibration numbers (11 real queries, 2026-10-27) on the
real 117-chunk corpus without any API call.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ingestion.chunking import chunk_document
from ingestion.embeddings import EmbeddingError, FakeEmbeddings
from ingestion.hybrid import HybridRetriever
from ingestion.index_store import IndexStore, index_documents, pack_vector
from ingestion.loaders import load_knowledge_base
from ingestion.policy_retrieval import (DEFAULT_MANIFEST, DEFAULT_MAX_CONTEXT, DEFAULT_MIN_SEMANTIC_SCORE, NOTE_CODES,
                                        NOTE_CONFLICTS, NOTE_DEPARTMENT, NOTE_SCOPE, AnswerView, ExcludedItem,
                                        PolicyRetriever)
from ingestion.retrieval import QueryError
from ingestion.search import main
from ingestion.tests.test_hybrid import synthetic

DIMS = 64
_REAL = None


def real_docs():
    global _REAL
    if _REAL is None:
        _REAL = load_knowledge_base()
    return _REAL


class FixedQuery(FakeEmbeddings):
    """Embeds every query as the first basis vector, so stored cosines are fully controlled."""

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0] + [0.0] * (DIMS - 1) for _ in texts]


class Scored:
    """A temporary index in which every chunk has a chosen cosine similarity to FixedQuery's vector."""

    def __init__(self, test: unittest.TestCase, documents, doc_scores=None, default=0.05, chunk_scores=None):
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.db = self.dir / "index.db"
        fake = FakeEmbeddings(dimensions=DIMS)
        store = IndexStore(self.db, "fake", fake.model)
        index_documents(store, documents, fake)
        self.chunks = {c.chunk_id: c for d in documents for c in chunk_document(d)}
        doc_scores, chunk_scores = doc_scores or {}, chunk_scores or {}
        try:
            for i, (chunk_id, chunk) in enumerate(sorted(self.chunks.items())):
                cosine = chunk_scores.get(chunk_id, doc_scores.get(chunk.doc_id, default))
                vector = np.zeros(DIMS)
                vector[0], vector[1 + i % (DIMS - 1)] = cosine, math.sqrt(1 - cosine * cosine)
                store.conn.execute("UPDATE embeddings SET vector = ? WHERE chunk_id = ?",
                                   (pack_vector(vector.astype("<f4")), chunk_id))
        finally:
            store.close()

    def retriever(self, **kw) -> PolicyRetriever:
        return PolicyRetriever(self.db, FixedQuery(dimensions=DIMS), **kw)

    def of(self, *doc_ids: str):
        return [c for c in self.chunks.values() if c.doc_id in doc_ids]

    def digest(self) -> tuple:
        stat = self.db.stat()
        return hashlib.sha256(self.db.read_bytes()).hexdigest(), stat.st_mtime_ns, stat.st_size


def docs_in(items) -> set[str]:
    return {i.doc_id for i in items}


VPN = {"KB-001": 0.54, "KB-018": 0.52, "KB-002": 0.40}  # replays the real VPN query: deprecated/draft outscore KB-002
PASSWORD = {"KB-005": 0.68, "SP-001": 0.63, "KB-004": 0.55, "KB-006": 0.51}  # deprecated KB-005 ranks first


class HardExclusionTests(unittest.TestCase):
    def setUp(self):
        self.index = Scored(self, real_docs(), VPN)

    def test_vpn_conflict_draft_and_deprecated_never_become_evidence_even_when_they_outscore_current(self):
        response = self.index.retriever().search("minimum vpn client", top_k=3, department="Engineering")
        self.assertEqual(len(response.actionable), 3)
        self.assertEqual(docs_in(response.actionable), {"KB-002"})
        self.assertEqual(response.conditional, ())
        self.assertEqual(docs_in(response.diagnostics), {"KB-001", "KB-018"})
        reasons = {d.doc_id: d.reasons for d in response.diagnostics}
        self.assertEqual(reasons["KB-001"], ("DEPRECATED", "SUPERSEDED_BY:KB-002"))
        self.assertEqual(reasons["KB-018"], ("DRAFT",))
        self.assertEqual(len(response.diagnostics), len(self.index.of("KB-001", "KB-018")))

    def test_full_pool_crowding_the_top_semantic_chunks_are_all_ineligible_yet_current_evidence_is_found(self):
        top = sorted(self.index.chunks.values(), key=lambda c: -VPN.get(c.doc_id, 0))[:8]
        self.assertTrue(all(c.doc_id in ("KB-001", "KB-018") for c in top))
        response = self.index.retriever().search("minimum vpn client", top_k=2, department="Engineering")
        self.assertEqual(len(response.actionable), 2)
        self.assertEqual(response.sufficiency.level, "sufficient")
        self.assertEqual(response.pool_size, len(self.index.chunks))
        self.assertTrue(response.pool_complete)

    def test_budget_overflow_is_reported_not_silent(self):
        response = self.index.retriever().search("minimum vpn client", top_k=3, department="Engineering")
        self.assertEqual(response.sufficiency.truncated_by_top_k, len(self.index.of("KB-002")) - 3)
        self.assertGreater(response.sufficiency.truncated_by_top_k, 0)

    def test_password_conflict_deprecated_phone_reset_is_excluded_and_current_policy_survives(self):
        index = Scored(self, real_docs(), PASSWORD)
        response = index.retriever().search("reset by phone", top_k=20)
        self.assertEqual(docs_in(response.actionable), {"SP-001", "KB-004", "KB-006"})
        self.assertEqual(docs_in(response.diagnostics), {"KB-005"})
        self.assertEqual({d.reasons for d in response.diagnostics}, {("DEPRECATED", "SUPERSEDED_BY:KB-004")})
        self.assertEqual(response.actionable[0].doc_id, "SP-001")
        self.assertTrue(response.governing_policy_chunks)
        self.assertTrue(all(c.startswith("SP-001:") for c in response.governing_policy_chunks))

    def test_a_superseder_that_is_not_current_removes_nothing(self):
        docs = [synthetic("KB-801", "Old but still current procedure.", ()),
                synthetic("KB-802", "Draft that claims to replace it.", (), status="draft", authoritative=False,
                          supersedes="KB-801")]
        response = Scored(self, docs, {"KB-801": 0.4, "KB-802": 0.6}).retriever().search("q")
        self.assertEqual(docs_in(response.actionable), {"KB-801"})
        self.assertEqual({d.reasons for d in response.diagnostics}, {("DRAFT",)})

    def test_invalid_status_is_excluded(self):
        docs = [synthetic("KB-801", "Odd status.", (), status="archived", authoritative=True)]
        response = Scored(self, docs, {"KB-801": 0.9}).retriever().search("q")
        self.assertEqual((response.actionable, response.conditional, response.context_only), ((), (), ()))
        self.assertEqual(response.diagnostics[0].reasons, ("INVALID_STATUS",))

    def test_excluded_text_is_metadata_only_by_default_and_labelled_when_requested(self):
        retriever = self.index.retriever()
        plain = retriever.search("q", top_k=5, department="Engineering")
        self.assertTrue(plain.diagnostics)
        self.assertTrue(all(d.untrusted_diagnostic_text is None for d in plain.diagnostics))
        self.assertNotIn("text", {f.name for f in dataclasses.fields(ExcludedItem)})
        explicit = retriever.search("q", top_k=5, department="Engineering", include_diagnostic_text=True)
        by_id = {d.chunk_id: d for d in explicit.diagnostics}
        for chunk in self.index.of("KB-018"):
            self.assertEqual(by_id[chunk.chunk_id].untrusted_diagnostic_text, chunk.text)

    def test_draft_prompt_injection_content_never_reaches_the_answer_view(self):
        poison = {c.text for c in self.index.of("KB-018", "KB-001")}
        response = self.index.retriever().search("my vpn client is rejected", top_k=10, department="Engineering",
                                                 include_diagnostic_text=True)
        view = response.for_answering()
        self.assertFalse(poison & {e.text for e in view.evidence})
        self.assertTrue(docs_in(view.evidence) <= {"KB-002"})
        self.assertFalse(poison & {e.text for e in response.context_for_background()})
        self.assertTrue(any(d.untrusted_diagnostic_text in poison for d in response.diagnostics))
        self.assertEqual({f.name for f in dataclasses.fields(AnswerView)},
                         {"evidence", "query_codes", "code_coverage", "undocumented_codes", "department",
                          "requires_department_verification", "sufficiency", "scope_validated", "notes"})


class ContextOnlyTests(unittest.TestCase):
    def setUp(self):
        self.extra_ids = [f"KB-90{n}" for n in range(1, 5)]
        self.docs = [*real_docs(), *(synthetic(i, f"Background note {i}.", (), authoritative=False)
                                     for i in self.extra_ids)]
        self.scores = {"KB-017": 0.6, **{i: 0.6 for i in self.extra_ids}}
        self.index = Scored(self, self.docs, self.scores)
        self.faq = len(self.index.of("KB-017", *self.extra_ids))  # KB-017 alone has only 2 chunks

    def test_non_authoritative_current_text_is_context_only_and_never_promoted(self):
        response = self.index.retriever().search("faq question")
        self.assertEqual((response.actionable, response.conditional), ((), ()))
        self.assertTrue(docs_in(response.context_only) <= {"KB-017", *self.extra_ids})
        self.assertEqual(response.sufficiency.level, "context_only")
        self.assertEqual(response.for_answering().evidence, ())
        self.assertTrue(all(i.tier == "context_only" for i in response.context_only))

    def test_context_only_is_capped_separately_and_reported(self):
        self.assertGreater(self.faq, DEFAULT_MAX_CONTEXT)
        response = self.index.retriever().search("faq question", top_k=1)
        self.assertEqual(len(response.context_only), DEFAULT_MAX_CONTEXT)
        self.assertEqual(response.sufficiency.relevant_context_only, self.faq)
        self.assertEqual(len(self.index.retriever(max_context=1).search("q").context_only), 1)
        none = self.index.retriever(max_context=0).search("q")
        self.assertEqual((none.context_only, none.sufficiency.level), ((), "none"))

    def test_context_is_reachable_only_through_the_explicit_background_accessor(self):
        response = self.index.retriever().search("faq question")
        background = response.context_for_background()
        self.assertEqual(len(background), DEFAULT_MAX_CONTEXT)
        self.assertTrue(all(e.tier == "context_only" and not e.authoritative for e in background))
        self.assertNotIn(background[0].chunk_id, {e.chunk_id for e in response.for_answering().evidence})

    def test_context_does_not_consume_the_answer_budget_and_answer_evidence_does_not_consume_context(self):
        index = Scored(self, self.docs, {**self.scores, "KB-004": 0.5, "KB-001": 0.9})
        response = index.retriever().search("q", top_k=1)
        self.assertEqual(len(response.actionable), 1)
        self.assertEqual(len(response.context_only), DEFAULT_MAX_CONTEXT)
        self.assertTrue(response.diagnostics)  # high-scoring deprecated KB-001 consumed nothing


class DepartmentTests(unittest.TestCase):
    def setUp(self):
        self.index = Scored(self, real_docs(), {"KB-002": 0.4})
        self.kb002 = real_docs()
        self.departments = next(d.applies_to for d in real_docs() if d.id == "KB-002")

    def test_unknown_department_keeps_scoped_evidence_conditional_with_explicit_requirements(self):
        response = self.index.retriever().search("vpn", top_k=3)
        self.assertEqual(response.actionable, ())
        self.assertEqual(len(response.conditional), 3)
        for item in response.conditional:
            self.assertTrue(item.requires_department_verification)
            self.assertEqual((item.applicability, item.applicable_departments), ("unresolved", tuple(self.departments)))
        view = response.for_answering()
        self.assertTrue(view.requires_department_verification)
        self.assertIn(NOTE_DEPARTMENT, view.notes)
        self.assertTrue(all(e.requires_department_verification and e.applicable_departments for e in view.evidence))
        self.assertEqual(response.applicability.required_departments, tuple(self.departments))
        self.assertEqual(response.applicability.unresolved_documents, ("KB-002",))
        self.assertFalse(response.applicability.department_known)
        self.assertEqual(response.sufficiency.level, "conditional_only")

    def test_known_listed_department_makes_the_evidence_actionable_without_a_verification_flag(self):
        for department in ("Finance", "finance", " Engineering "):
            with self.subTest(department=department):
                response = self.index.retriever().search("vpn", top_k=3, department=department)
                self.assertEqual(response.conditional, ())
                self.assertEqual(len(response.actionable), 3)
                view = response.for_answering()
                self.assertFalse(view.requires_department_verification)
                self.assertNotIn(NOTE_DEPARTMENT, view.notes)
                self.assertTrue(all(not e.requires_department_verification for e in view.evidence))

    def test_known_department_mismatch_is_excluded_with_a_reason(self):
        for department in ("Contractors", "Marketing"):
            with self.subTest(department=department):
                response = self.index.retriever().search("vpn", top_k=3, department=department)
                self.assertEqual((response.actionable, response.conditional), ((), ()))
                self.assertEqual({d.reasons for d in response.diagnostics}, {("APPLICABILITY_MISMATCH",)})
                self.assertEqual(response.sufficiency.level, "none")

    def test_department_is_never_inferred_from_the_query(self):
        provider = FixedQuery(dimensions=DIMS)
        retriever = PolicyRetriever(self.index.db, provider)
        response = retriever.search("I work in Finance and I am a Contractor", top_k=3)
        self.assertIsNone(response.department)
        self.assertEqual(len(response.conditional), 3)

    def test_invalid_department_is_rejected_before_any_embedding_call(self):
        provider = FixedQuery(dimensions=DIMS)
        retriever = PolicyRetriever(self.index.db, provider)
        for bad in ("", "   ", 5, ["Finance"]):
            with self.assertRaises(QueryError):
                retriever.search("vpn", department=bad)  # type: ignore[arg-type]
        self.assertEqual(provider.calls, [])

    def test_all_applicability_documents_are_unaffected_by_department(self):
        index = Scored(self, real_docs(), {"KB-004": 0.5})
        for department in (None, "Contractors", "Finance"):
            response = index.retriever().search("password", top_k=2, department=department)
            self.assertEqual(docs_in(response.actionable), {"KB-004"})


class CodeCoverageTests(unittest.TestCase):
    def test_exact_match_bypasses_the_floor_for_eligible_chunks_only(self):
        index = Scored(self, real_docs(), {"KB-001": 0.01, "KB-018": 0.01, "KB-002": 0.01})
        response = index.retriever().search("NX-781", top_k=5, department="Engineering")
        self.assertTrue(response.actionable)
        self.assertEqual(docs_in(response.actionable), {"KB-002"})
        self.assertTrue(all(i.documents_query_codes == ("NX-781",) for i in response.actionable))
        self.assertEqual(dict(response.code_coverage), {"NX-781": "documented"})
        excluded = {d.doc_id: d for d in response.diagnostics}
        self.assertEqual({"KB-001", "KB-018"}, set(excluded))  # exact evidence is kept for diagnostics only
        self.assertTrue(all(d.matched_codes == ("NX-781",) for d in response.diagnostics))

    def test_a_low_scoring_exact_match_survives_a_crowd_of_high_scoring_ineligible_chunks(self):
        index = Scored(self, real_docs(), {"KB-005": 0.68, "KB-001": 0.54, "KB-018": 0.52, "KB-002": 0.01})
        response = index.retriever().search("NX-112", top_k=1, department="Engineering")
        self.assertEqual(len(response.actionable), 1)
        self.assertEqual(response.actionable[0].documents_query_codes, ("NX-112",))
        self.assertEqual(response.actionable[0].hybrid.semantic_rank > 50, True)

    def test_a_code_documented_only_by_excluded_evidence_stays_undocumented(self):
        docs = [synthetic("KB-801", "NX-403 means access denied. Call the desk.", ("NX-403",)),
                synthetic("KB-802", "NX-112 workaround: disable the check.", ("NX-112",), status="draft",
                          authoritative=False)]
        response = Scored(self, docs, {"KB-801": 0.4, "KB-802": 0.6}).retriever().search("NX-112 and NX-403")
        self.assertEqual(dict(response.code_coverage), {"NX-112": "undocumented", "NX-403": "documented"})
        self.assertEqual(response.undocumented_codes, ("NX-112",))
        self.assertEqual([i.documents_query_codes for i in response.actionable], [("NX-403",)])
        self.assertEqual(response.diagnostics[0].matched_codes, ("NX-112",))

    def test_an_unknown_code_is_undocumented_and_other_code_steps_are_not_presented_as_its_fix(self):
        index = Scored(self, real_docs(), {"KB-002": 0.4})
        response = index.retriever().search("NX-999", top_k=3, department="Engineering")
        self.assertEqual(dict(response.code_coverage), {"NX-999": "undocumented"})
        self.assertTrue(response.actionable)
        self.assertTrue(all(i.documents_query_codes == () for i in response.actionable))
        view = response.for_answering()
        self.assertEqual(view.undocumented_codes, ("NX-999",))
        self.assertTrue(all(e.documents_query_codes == () for e in view.evidence))
        self.assertIn(NOTE_CODES, view.notes)

    def test_documented_never_means_a_confirmed_fix(self):
        response = Scored(self, real_docs(), {}).retriever().search("NX-781", department="Engineering")
        self.assertEqual(set(response.code_coverage.values()), {"documented"})
        self.assertNotIn("confirmed", {v for v in response.code_coverage.values()})
        self.assertIn("does not confirm", NOTE_CODES)

    def test_code_coverage_applicability_and_sufficiency_are_independent_fields(self):
        index = Scored(self, real_docs(), {"KB-002": 0.4})
        response = index.retriever().search("NX-999 on my vpn", top_k=3)  # department unknown
        self.assertEqual(response.undocumented_codes, ("NX-999",))
        self.assertTrue(response.applicability.requires_department_verification)
        self.assertEqual(response.sufficiency.level, "conditional_only")
        view = response.for_answering()
        self.assertEqual((view.undocumented_codes, view.requires_department_verification), (("NX-999",), True))

    def test_query_without_codes_has_empty_coverage(self):
        response = Scored(self, real_docs(), {"KB-004": 0.5}).retriever().search("forgot password")
        self.assertEqual((response.query_codes, dict(response.code_coverage), response.undocumented_codes), ((), {}, ()))


class RelevanceTests(unittest.TestCase):
    # (document, cosine from the real calibration, label: truly relevant to that query, returned at floor 0.25)
    LABELLED = [
        ("SP-001", 0.0634, False, False),  # "recipe for chocolate cake"
        ("KB-003", 0.0533, False, False),  # "who won the world cup"
        ("KB-004", 0.1472, False, False),  # "change a flat tire"
        ("KB-010", 0.3741, True, True),  # weakest relevant chunk of "laptop not compliant"
        ("KB-007", 0.3503, True, True),  # weakest relevant chunk of "MFA on new phone"
        ("KB-013", 0.4247, True, True),  # weakest relevant chunk of "cannot print"
        ("KB-004", 0.5288, False, True),  # "reset my Netflix password": irrelevant but ABOVE the floor
    ]

    def test_labelled_query_chunk_pairs(self):
        for doc_id, cosine, relevant, returned in self.LABELLED:
            with self.subTest(doc=doc_id, cosine=cosine, relevant=relevant):
                index = Scored(self, real_docs(), {doc_id: cosine})
                response = index.retriever().search("q", top_k=50)
                self.assertEqual(bool(response.actionable), returned)
                if not returned:
                    self.assertEqual((response.sufficiency.level, response.sufficiency.relevant_actionable), ("none", 0))

    def test_the_floor_cannot_detect_an_in_domain_sounding_out_of_scope_query(self):
        response = Scored(self, real_docs(), {"KB-004": 0.5288}).retriever().search("reset my Netflix password")
        self.assertEqual(response.sufficiency.level, "sufficient")  # NOT solved by the threshold
        view = response.for_answering()
        self.assertFalse(view.scope_validated)
        self.assertIn(NOTE_SCOPE, view.notes)
        self.assertIn(NOTE_CONFLICTS, view.notes)

    def test_floor_boundary_and_configuration(self):
        docs = [synthetic("KB-801", "Above the floor.", ()), synthetic("KB-802", "Below the floor.", ())]
        index = Scored(self, docs, {"KB-801": 0.2501, "KB-802": 0.2499})
        self.assertEqual(DEFAULT_MIN_SEMANTIC_SCORE, 0.25)
        self.assertEqual(docs_in(index.retriever().search("q").actionable), {"KB-801"})
        self.assertEqual(index.retriever(min_semantic_score=0.5).search("q").actionable, ())
        self.assertEqual(docs_in(index.retriever(min_semantic_score=0.1).search("q", top_k=5).actionable),
                         {"KB-801", "KB-802"})

    def test_no_relevant_evidence_returns_nothing_and_never_pads_with_unrelated_current_documents(self):
        response = Scored(self, real_docs(), {}).retriever().search("what is the capital of australia", top_k=5)
        self.assertEqual((response.actionable, response.conditional, response.context_only, response.diagnostics),
                         ((), (), (), ()))
        self.assertEqual(response.sufficiency.level, "none")
        self.assertGreater(response.sufficiency.eligible_below_relevance, 0)
        self.assertIsNone(response.sufficiency.best_eligible_score)
        self.assertEqual(response.for_answering().evidence, ())

    def test_sufficiency_reports_the_best_eligible_score(self):
        response = Scored(self, real_docs(), {"KB-004": 0.55, "KB-006": 0.4}).retriever().search("q")
        self.assertAlmostEqual(response.sufficiency.best_eligible_score, 0.55, places=5)


class RankingAndBudgetTests(unittest.TestCase):
    def test_top_k_is_shared_by_actionable_and_conditional_evidence(self):
        index = Scored(self, real_docs(), {"SP-001": 0.5, "KB-004": 0.45, "KB-002": 0.4})
        response = index.retriever().search("q", top_k=4)
        self.assertEqual(len(response.actionable) + len(response.conditional), 4)
        ranks = [i.policy_rank for i in (*response.actionable, *response.conditional)]
        self.assertEqual(sorted(ranks), [1, 2, 3, 4])
        self.assertEqual([e.policy_rank for e in []], [])
        view = response.for_answering()
        self.assertEqual([e.doc_id for e in view.evidence][0], "SP-001")
        self.assertEqual(view.evidence[0].semantic_score >= view.evidence[-1].semantic_score, True)

    def test_security_policy_is_not_boosted_above_a_more_relevant_kb_chunk(self):
        index = Scored(self, real_docs(), {"SP-001": 0.30, "KB-004": 0.55, "SP-002": 0.10})
        response = index.retriever().search("q", top_k=20)
        self.assertEqual(response.actionable[0].doc_id, "KB-004")
        self.assertNotIn("SP-002", docs_in((*response.actionable, *response.context_only)))
        self.assertTrue(all(c.startswith("SP-001:") for c in response.governing_policy_chunks))

    def test_ordering_is_deterministic_and_ties_break_on_chunk_id(self):
        docs = [synthetic("KB-821", "Identical text.", ()), synthetic("KB-820", "Identical text.", ())]
        index = Scored(self, docs, {"KB-820": 0.4, "KB-821": 0.4})
        first = [i.chunk_id for i in index.retriever().search("q").actionable]
        again = [i.chunk_id for i in index.retriever().search("q").actionable]
        self.assertEqual(first, again)
        self.assertEqual(first, sorted(first))

    def test_equal_scores_prefer_the_later_last_updated_then_the_higher_version(self):
        docs = [synthetic("KB-830", "Same text.", (), last_updated="2025-01-01", version="1.0"),
                synthetic("KB-831", "Same text.", (), last_updated="2026-01-01", version="1.0"),
                synthetic("KB-832", "Same text.", (), last_updated="2026-01-01", version="1.10"),
                synthetic("KB-833", "Same text.", (), last_updated="2026-01-01", version="1.9")]
        response = Scored(self, docs, default=0.4).retriever().search("q", top_k=4)
        self.assertEqual([i.doc_id for i in response.actionable], ["KB-832", "KB-833", "KB-831", "KB-830"])

    def test_malformed_parameters_are_rejected_without_embedding(self):
        index = Scored(self, real_docs(), {"KB-004": 0.5})
        for bad in (float("nan"), float("inf"), 2.0, -2.0, True, "0.25", None):
            with self.assertRaises(QueryError):
                index.retriever(min_semantic_score=bad)  # type: ignore[arg-type]
        for bad in (-1, True, 1.5, None):
            with self.assertRaises(QueryError):
                index.retriever(max_context=bad)  # type: ignore[arg-type]
        provider = FixedQuery(dimensions=DIMS)
        retriever = PolicyRetriever(index.db, provider)
        for query, top_k in (("", 5), ("   ", 5), (None, 5), ("ok", 0), ("ok", True)):
            with self.assertRaises(QueryError):
                retriever.search(query, top_k=top_k)  # type: ignore[arg-type]
        self.assertEqual(provider.calls, [])

    def test_provider_failure_propagates(self):
        class Down(FixedQuery):
            def embed(self, texts):
                raise EmbeddingError("provider is down")

        index = Scored(self, real_docs(), {"KB-004": 0.5})
        with self.assertRaises(EmbeddingError):
            PolicyRetriever(index.db, Down(dimensions=DIMS)).search("q")

    def test_one_embedding_call_and_the_database_is_never_modified(self):
        index = Scored(self, real_docs(), VPN)
        before = index.digest()
        provider = FixedQuery(dimensions=DIMS)
        retriever = PolicyRetriever(index.db, provider, manifest_path=DEFAULT_MANIFEST)
        retriever.search(" NX-781 vpn ", department="Finance", include_diagnostic_text=True)
        self.assertEqual(provider.calls, [["NX-781 vpn"]])
        self.assertEqual(index.digest(), before)


class PreservationAndInvariantTests(unittest.TestCase):
    """Real 20-document corpus embedded with the fake provider; every chunk is relevant (floor -1)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.db = Path(cls.tmp.name) / "index.db"
        cls.fake = FakeEmbeddings(dimensions=DIMS)
        store = IndexStore(cls.db, "fake", cls.fake.model)
        index_documents(store, real_docs(), cls.fake)
        store.close()
        cls.chunks = {c.chunk_id: c for d in real_docs() for c in chunk_document(d)}
        cls.pages = {d.id: {p.page: p.text for p in d.pages} for d in real_docs()}

    def retriever(self, **kw) -> PolicyRetriever:
        return PolicyRetriever(self.db, FakeEmbeddings(dimensions=DIMS), min_semantic_score=-1.0, **kw)

    def test_phase_2b_results_citations_offsets_and_scores_are_preserved_exactly(self):
        policy = self.retriever().search("NX-781 vpn client", top_k=200, department="Finance")
        hybrid = {r.chunk_id: r for r in HybridRetriever(self.db, FakeEmbeddings(dimensions=DIMS))
                  .search("NX-781 vpn client", top_k=len(self.chunks)).results}
        items = (*policy.actionable, *policy.conditional, *policy.context_only)
        self.assertTrue(items)
        for item in items:
            self.assertEqual(item.hybrid, hybrid[item.chunk_id])
            chunk = item.hybrid.chunk
            self.assertEqual(chunk.text, self.pages[chunk.doc_id][chunk.page][chunk.char_start:chunk.char_end])
        for e in policy.for_answering().evidence:
            twin = hybrid[e.chunk_id]
            self.assertEqual((e.text, e.citation, e.char_start, e.char_end, e.page, e.semantic_score),
                             (twin.chunk.text, twin.chunk.citation, twin.chunk.char_start, twin.chunk.char_end,
                              twin.chunk.page, twin.semantic_score))
            self.assertEqual((e.status, e.authoritative, e.doc_sha256, e.version, e.heading_path),
                             (twin.chunk.status, twin.chunk.authoritative, twin.chunk.doc_sha256, twin.chunk.version,
                              twin.chunk.heading_path))

    def test_invariants_hold_for_every_department_and_query(self):
        ineligible_docs = {"KB-001", "KB-005", "KB-018"}
        for department in (None, "Finance", "Engineering", "Contractors", "HR", "Sales", "Executive Office"):
            for query in ("vpn nx-781 client", "password reset by phone", "printer PRN-07", "mfa new phone", "wifi"):
                with self.subTest(department=department, query=query):
                    r = self.retriever(max_context=500).search(query, top_k=500, department=department)
                    answer = (*r.actionable, *r.conditional)
                    for item in answer:
                        chunk = item.hybrid.chunk
                        self.assertEqual((chunk.status, chunk.authoritative), ("current", True))
                        self.assertNotIn(chunk.doc_id, ineligible_docs | {"KB-017"})
                        self.assertEqual(item.requires_department_verification, department is None and
                                         chunk.applies_to != ("all",))
                    self.assertEqual(docs_in(r.context_only) - {"KB-017"}, set())
                    ids = [i.chunk_id for i in (*answer, *r.context_only)] + [d.chunk_id for d in r.diagnostics]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(set(ids), set(self.chunks))  # nothing silently dropped
                    self.assertEqual(r.sufficiency.eligible_below_relevance, 0)
                    if department == "Contractors":
                        self.assertNotIn("KB-002", docs_in(answer))

    def test_manifest_annotations_are_opt_in_diagnostic_and_not_answer_facing(self):
        self.assertEqual(self.retriever().search("vpn", top_k=5).conflict_annotations, ())
        on = self.retriever(manifest_path=DEFAULT_MANIFEST).search("vpn password phone", top_k=5)
        by_id = {n.id: n for n in on.conflict_annotations}
        self.assertTrue({"C1", "C2"} <= set(by_id))
        self.assertEqual(set(by_id["C1"].present_documents), {"KB-001", "KB-002", "KB-018"})
        self.assertTrue(all(n.diagnostic_only for n in on.conflict_annotations))
        self.assertNotIn("conflict_annotations", {f.name for f in dataclasses.fields(AnswerView)})
        self.assertIsNone(on.annotation_warning)

    def test_unloadable_manifest_fails_safely_with_a_warning(self):
        for path in (Path(self.tmp.name) / "missing.json", self.write_bad()):
            r = self.retriever(manifest_path=path).search("vpn", top_k=3)
            self.assertEqual(r.conflict_annotations, ())
            self.assertIn("conflict annotations disabled", r.annotation_warning or "")
            self.assertEqual(len(r.actionable) + len(r.conditional), 3)

    def write_bad(self) -> Path:
        path = Path(self.tmp.name) / "bad.json"
        path.write_text('{"known_conflicts": [{"id": "C1"}]}', encoding="utf-8")
        return path


class PolicyCliTests(unittest.TestCase):
    def setUp(self):
        self.index = Scored(self, real_docs(), VPN)

    def run_cli(self, *argv, provider=None):
        lines: list[str] = []
        code = main([*argv, "--provider", "fake", "--db", str(self.index.db)],
                    provider=provider or FixedQuery(dimensions=DIMS), out=lines.append)
        return code, "\n".join(lines)

    def test_policy_is_the_default_mode_and_flags_unresolved_department(self):
        code, text = self.run_cli("vpn version", "--top-k", "2")
        self.assertEqual(code, 0)
        self.assertIn("mode=policy", text)
        self.assertIn("[conditional]", text)
        self.assertIn("REQUIRES DEPARTMENT VERIFICATION", text)
        self.assertIn("department: unknown (not inferred)", text)
        self.assertIn("NOT resolved here", text)
        self.assertNotIn("[actionable]", text)

    def test_known_department_gives_actionable_evidence_and_default_output_hides_excluded_text(self):
        code, text = self.run_cli("vpn version", "--top-k", "2", "--department", "Finance")
        self.assertEqual(code, 0)
        self.assertIn("[actionable]", text)
        self.assertNotIn("UNTRUSTED", text)
        self.assertIn("use --diagnostics", text)
        for chunk in self.index.of("KB-018"):
            self.assertNotIn(" ".join(chunk.text.split())[:60], text)

    def test_diagnostics_flag_lists_exclusions_and_labels_their_text_untrusted(self):
        code, text = self.run_cli("vpn version", "--department", "Finance", "--diagnostics")
        self.assertEqual(code, 0)
        self.assertIn("DEPRECATED, SUPERSEDED_BY:KB-002", text)
        self.assertIn("UNTRUSTED diagnostic text:", text)

    def test_conflict_annotations_are_opt_in(self):
        _, off = self.run_cli("vpn version")
        _, on = self.run_cli("vpn version", "--annotate-conflicts")
        self.assertNotIn("known conflict", off)
        self.assertIn("known conflict C1 (diagnostic only)", on)

    def test_context_only_is_labelled_as_background(self):
        index = Scored(self, real_docs(), {"KB-017": 0.6})
        lines: list[str] = []
        main(["faq", "--provider", "fake", "--db", str(index.db)], provider=FixedQuery(dimensions=DIMS), out=lines.append)
        text = "\n".join(lines)
        self.assertIn("[context_only]", text)
        self.assertIn("background, never instructions", text)

    def test_hybrid_and_semantic_modes_remain_available(self):
        for mode in ("hybrid", "semantic"):
            code, text = self.run_cli("vpn", "--mode", mode, "--top-k", "2")
            self.assertEqual(code, 0)
            self.assertIn(f"mode={mode}", text)

    def test_exit_codes(self):
        self.assertEqual(self.run_cli("   ")[0], 2)
        self.assertEqual(self.run_cli("vpn", "--department", "  ")[0], 2)
        self.assertEqual(self.run_cli("vpn", "--min-score", "5")[0], 2)
        self.assertEqual(self.run_cli("vpn", "--top-k", "0")[0], 2)

        class Down(FixedQuery):
            def embed(self, texts):
                raise EmbeddingError("provider is down")

        code, text = self.run_cli("vpn", provider=Down(dimensions=DIMS))
        self.assertEqual(code, 1)
        self.assertIn("provider is down", text)


if __name__ == "__main__":
    unittest.main()
