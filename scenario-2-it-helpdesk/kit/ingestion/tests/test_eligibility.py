"""Offline tests for ingestion.eligibility (hard evidence rules) and ingestion.conflicts (manifest annotations)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ingestion.conflicts import ManifestError, annotate, load_known_conflicts
from ingestion.eligibility import (ACTIONABLE, APPLICABLE, CONDITIONAL, CONTEXT_ONLY, EXCLUDED, UNRESOLVED, DocumentFacts,
                                   classify, superseded_by)
from ingestion.loaders import load_knowledge_base
from ingestion.policy_retrieval import DEFAULT_MANIFEST


def facts(doc_id="KB-900", status="current", authoritative=True, supersedes=None, applies_to=("all",)):
    return DocumentFacts(doc_id, status, authoritative, supersedes, tuple(applies_to))


class ClassifyTests(unittest.TestCase):
    def test_current_authoritative_document_for_everyone_is_actionable(self):
        verdict = classify(facts(), {}, None)
        self.assertEqual((verdict.tier, verdict.reasons, verdict.applicability), (ACTIONABLE, (), APPLICABLE))

    def test_draft_deprecated_and_invalid_status_are_hard_excluded(self):
        for status, reason in (("draft", "DRAFT"), ("deprecated", "DEPRECATED"), ("archived", "INVALID_STATUS"),
                               ("", "INVALID_STATUS"), ("CURRENT", "INVALID_STATUS")):
            with self.subTest(status=status):
                verdict = classify(facts(status=status, authoritative=status == "archived"), {}, None)
                self.assertEqual((verdict.tier, verdict.reasons), (EXCLUDED, (reason,)))

    def test_a_flag_cannot_rescue_an_excluded_status(self):
        verdict = classify(facts(status="draft", authoritative=True), {}, "Finance")
        self.assertEqual(verdict.tier, EXCLUDED)

    def test_superseded_documents_are_excluded_even_if_their_own_status_says_current(self):
        docs = [facts("KB-001"), facts("KB-002", supersedes="KB-001")]
        superseders = superseded_by(docs)
        self.assertEqual(superseders, {"KB-001": "KB-002"})
        verdict = classify(docs[0], superseders, None)
        self.assertEqual((verdict.tier, verdict.reasons), (EXCLUDED, ("SUPERSEDED_BY:KB-002",)))
        self.assertEqual(classify(docs[1], superseders, None).tier, ACTIONABLE)

    def test_a_deprecated_and_superseded_document_lists_both_reasons(self):
        docs = [facts("KB-001", status="deprecated", authoritative=False), facts("KB-002", supersedes="KB-001")]
        verdict = classify(docs[0], superseded_by(docs), None)
        self.assertEqual(verdict.reasons, ("DEPRECATED", "SUPERSEDED_BY:KB-002"))

    def test_only_current_documents_supersede(self):
        for status in ("draft", "deprecated", "bogus"):
            with self.subTest(status=status):
                docs = [facts("KB-001"), facts("KB-002", status=status, authoritative=False, supersedes="KB-001")]
                self.assertEqual(superseded_by(docs), {})
                self.assertEqual(classify(docs[0], superseded_by(docs), None).tier, ACTIONABLE)

    def test_self_supersession_and_unknown_targets_are_ignored(self):
        self.assertEqual(superseded_by([facts("KB-002", supersedes="KB-002")]), {})
        self.assertEqual(superseded_by([facts("KB-002", supersedes="KB-777")]), {"KB-777": "KB-002"})
        self.assertEqual(superseded_by([facts("KB-002", supersedes="  ")]), {})

    def test_several_superseders_resolve_deterministically_to_the_lowest_id(self):
        docs = [facts("KB-009", supersedes="KB-001"), facts("KB-003", supersedes="KB-001")]
        self.assertEqual(superseded_by(docs), {"KB-001": "KB-003"})
        self.assertEqual(superseded_by(reversed(docs)), {"KB-001": "KB-003"})

    def test_current_non_authoritative_is_context_only_and_never_actionable(self):
        verdict = classify(facts("KB-017", authoritative=False), {}, None)
        self.assertEqual((verdict.tier, verdict.reasons), (CONTEXT_ONLY, ()))
        self.assertEqual(classify(facts("KB-017", authoritative=False), {}, "Finance").tier, CONTEXT_ONLY)

    def test_unknown_department_makes_scoped_evidence_conditional_and_unresolved(self):
        verdict = classify(facts(applies_to=("Finance", "HR")), {}, None)
        self.assertEqual((verdict.tier, verdict.applicability, verdict.applicable_departments),
                         (CONDITIONAL, UNRESOLVED, ("Finance", "HR")))

    def test_known_department_resolves_scoped_evidence_case_insensitively(self):
        for department in ("Finance", "finance", "FINANCE", " Finance "):
            with self.subTest(department=department):
                verdict = classify(facts(applies_to=("Finance", "HR")), {}, department)
                self.assertEqual((verdict.tier, verdict.applicability), (ACTIONABLE, APPLICABLE))

    def test_known_department_not_listed_is_excluded_not_guessed(self):
        for department in ("Contractors", "Marketing", "Fin", "Finance Ops"):
            with self.subTest(department=department):
                verdict = classify(facts(applies_to=("Finance", "HR")), {}, department)
                self.assertEqual((verdict.tier, verdict.reasons), (EXCLUDED, ("APPLICABILITY_MISMATCH",)))

    def test_all_applies_to_every_department_and_unknown_departments(self):
        for department in (None, "Contractors", "Finance"):
            self.assertEqual(classify(facts(applies_to=("all",)), {}, department).tier, ACTIONABLE)
            self.assertEqual(classify(facts(applies_to=("ALL",)), {}, department).tier, ACTIONABLE)

    def test_empty_applicability_fails_closed(self):
        verdict = classify(facts(applies_to=()), {}, None)
        self.assertEqual((verdict.tier, verdict.reasons), (EXCLUDED, ("INVALID_APPLICABILITY",)))

    def test_non_authoritative_scoped_document_stays_context_only_with_unresolved_applicability(self):
        verdict = classify(facts(authoritative=False, applies_to=("Finance",)), {}, None)
        self.assertEqual((verdict.tier, verdict.applicability), (CONTEXT_ONLY, UNRESOLVED))

    def test_authoritative_must_be_exactly_true(self):
        self.assertEqual(classify(facts(authoritative=None), {}, None).tier, CONTEXT_ONLY)  # type: ignore[arg-type]


class RealMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = {d.id: DocumentFacts(d.id, d.status, bool(d.authoritative), d.supersedes, tuple(d.applies_to))
                    for d in load_knowledge_base()}
        cls.superseders = superseded_by(cls.docs.values())

    def verdict(self, doc_id, department=None):
        return classify(self.docs[doc_id], self.superseders, department)

    def test_the_two_real_supersessions(self):
        self.assertEqual(dict(self.superseders), {"KB-001": "KB-002", "KB-005": "KB-004"})

    def test_real_documents_without_department(self):
        self.assertEqual(self.verdict("KB-001").reasons, ("DEPRECATED", "SUPERSEDED_BY:KB-002"))
        self.assertEqual(self.verdict("KB-005").reasons, ("DEPRECATED", "SUPERSEDED_BY:KB-004"))
        self.assertEqual(self.verdict("KB-018").reasons, ("DRAFT",))
        self.assertEqual(self.verdict("KB-017").tier, CONTEXT_ONLY)
        self.assertEqual(self.verdict("KB-002").tier, CONDITIONAL)
        for doc_id in ("SP-001", "SP-002", "KB-004", "KB-006", "KB-015", "KB-016"):
            self.assertEqual(self.verdict(doc_id).tier, ACTIONABLE, doc_id)

    def test_real_kb_002_with_departments(self):
        self.assertEqual(self.verdict("KB-002", "Finance").tier, ACTIONABLE)
        self.assertEqual(self.verdict("KB-002", "engineering").tier, ACTIONABLE)
        self.assertEqual(self.verdict("KB-002", "Contractors").reasons, ("APPLICABILITY_MISMATCH",))

    def test_every_real_document_gets_exactly_one_consistent_verdict(self):
        for doc_id in self.docs:
            verdict = self.verdict(doc_id)
            self.assertEqual(bool(verdict.reasons), verdict.tier == EXCLUDED, doc_id)
            if self.docs[doc_id].status != "current":
                self.assertEqual(verdict.tier, EXCLUDED, doc_id)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write(self, data, name="manifest.json") -> Path:
        path = self.dir / name
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        return path

    @staticmethod
    def entry(**kw):
        base = {"id": "C9", "topic": "Topic", "documents": ["KB-001", "KB-002"], "resolution": "KB-002 wins"}
        base.update(kw)
        return base

    def test_the_real_manifest_loads_six_known_conflicts(self):
        conflicts = load_known_conflicts(DEFAULT_MANIFEST)
        self.assertEqual([c.id for c in conflicts], ["C1", "C2", "C3", "C4", "C5", "C6"])
        by_id = {c.id: c for c in conflicts}
        self.assertEqual(by_id["C1"].documents, ("KB-001", "KB-002", "KB-018"))
        self.assertEqual(by_id["C2"].documents, ("KB-005", "KB-004", "SP-001"))

    def test_malformed_manifests_fail_with_manifest_error(self):
        cases = {
            "missing": self.dir / "nope.json",
            "not json": self.write("{not json"),
            "not an object": self.write("[1, 2]", "a.json"),
            "no list": self.write({"known_conflicts": {}}, "b.json"),
            "entry not object": self.write({"known_conflicts": ["x"]}, "c.json"),
            "blank id": self.write({"known_conflicts": [self.entry(id=" ")]}, "d.json"),
            "no documents": self.write({"known_conflicts": [self.entry(documents=[])]}, "e.json"),
            "bad document": self.write({"known_conflicts": [self.entry(documents=["KB-001", 5])]}, "f.json"),
            "no resolution": self.write({"known_conflicts": [self.entry(resolution="")]}, "g.json"),
            "no topic": self.write({"known_conflicts": [self.entry(topic=None)]}, "h.json"),
            "duplicate id": self.write({"known_conflicts": [self.entry(), self.entry()]}, "i.json"),
        }
        for name, path in cases.items():
            with self.subTest(case=name), self.assertRaises(ManifestError):
                load_known_conflicts(path)

    def test_annotation_needs_two_documents_of_one_conflict_among_the_relevant_ones(self):
        conflicts = load_known_conflicts(self.write({"known_conflicts": [
            self.entry(id="C1", documents=["KB-001", "KB-002", "KB-018"]),
            self.entry(id="C2", documents=["KB-005", "KB-004"])]}))
        self.assertEqual(annotate(conflicts, ["KB-001"]), ())
        single = load_known_conflicts(self.write({"known_conflicts": [self.entry(id="C6", documents=["KB-015"])]}, "s.json"))
        self.assertEqual(annotate(single, ["KB-015"]), ())  # a one-document conflict never annotates
        notes = annotate(conflicts, ["KB-018", "KB-002", "KB-004"])
        self.assertEqual([(n.id, n.present_documents) for n in notes], [("C1", ("KB-002", "KB-018"))])
        self.assertTrue(all(n.diagnostic_only for n in notes))


if __name__ == "__main__":
    unittest.main()
