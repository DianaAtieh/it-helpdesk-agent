"""Offline tests for ingestion.retrieval and ingestion.search (fake embeddings, temporary SQLite files). No network.

    .venv\\Scripts\\python.exe -m unittest discover -s ingestion/tests -t . -v
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ingestion.chunking import chunk_document
from ingestion.embeddings import EmbeddingError, FakeEmbeddings
from ingestion.index_store import IndexStore, index_documents, pack_vector
from ingestion.loaders import load_knowledge_base
from ingestion.retrieval import MAX_QUERY_CHARS, QueryError, RetrievalError, SemanticRetriever
from ingestion.search import main, pinned_model

DIMS = 64
_DOCS = None


def docs():
    global _DOCS
    if _DOCS is None:
        _DOCS = load_knowledge_base()
    return _DOCS


class IndexCase(unittest.TestCase):
    """Each test gets its own fake-embedding index of the 20 real documents."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = Path(tmp.name) / "index.db"
        self.fake = FakeEmbeddings(dimensions=DIMS)
        store = IndexStore(self.db, "fake", self.fake.model)
        index_documents(store, docs(), self.fake)
        store.close()
        self.chunks = {c.chunk_id: c for d in docs() for c in chunk_document(d)}

    def retriever(self, provider=None) -> SemanticRetriever:
        return SemanticRetriever(self.db, provider or FakeEmbeddings(dimensions=DIMS))

    def sql(self, statement: str, params: tuple = ()) -> None:
        conn = IndexStore(self.db, "fake", self.fake.model)
        try:
            conn.conn.execute(statement, params)
        finally:
            conn.close()


class RankingTests(IndexCase):
    def test_a_chunks_own_embedded_text_ranks_first_with_score_one(self):
        retriever = self.retriever()
        for chunk_id in sorted(self.chunks)[::9]:
            chunk = self.chunks[chunk_id]
            best = retriever.search(chunk.contextual_text, top_k=3)[0]
            self.assertEqual(best.chunk_id, chunk_id)
            self.assertAlmostEqual(best.score, 1.0, places=5)

    def test_results_are_best_first_ranked_and_limited_to_top_k(self):
        results = self.retriever().search("VPN keeps disconnecting", top_k=7)
        self.assertEqual([r.rank for r in results], list(range(1, 8)))
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len({r.chunk_id for r in results}), 7)

    def test_top_k_larger_than_the_index_returns_everything_once(self):
        retriever = self.retriever()
        results = retriever.search("anything", top_k=10_000)
        self.assertEqual(len(results), len(retriever))
        self.assertEqual(len(results), len(self.chunks))

    def test_scores_match_an_independent_numpy_cosine(self):
        retriever = self.retriever()
        query = "printer error"
        results = retriever.search(query, top_k=len(self.chunks))
        q = np.asarray(self.fake.embed([query])[0], dtype=np.float64)
        q /= np.linalg.norm(q)
        for r in results[:10]:
            stored = np.asarray(self.fake.embed([self.chunks[r.chunk_id].contextual_text])[0], dtype=np.float64)
            stored /= np.linalg.norm(stored)
            self.assertAlmostEqual(r.score, float(q @ stored), places=5)

    def test_ordering_is_deterministic_across_retrievers_and_calls(self):
        first = [(r.chunk_id, r.score) for r in self.retriever().search("reset my password", 10)]
        retriever = self.retriever()
        self.assertEqual(first, [(r.chunk_id, r.score) for r in retriever.search("reset my password", 10)])
        self.assertEqual(first, [(r.chunk_id, r.score) for r in retriever.search("reset my password", 10)])

    def test_ties_break_on_chunk_id(self):
        ids = sorted(self.chunks)
        source = ids[40]
        blob = IndexStore(self.db, "fake", self.fake.model)
        vector = bytes(blob.conn.execute("SELECT vector FROM embeddings WHERE chunk_id = ?", (source,)).fetchone()[0])
        blob.close()
        twins = [ids[3], ids[77], ids[100]]
        for twin in twins:
            self.sql("UPDATE embeddings SET vector = ? WHERE chunk_id = ?", (vector, twin))
        results = self.retriever().search(self.chunks[source].contextual_text, top_k=10)
        tied = [r for r in results if abs(r.score - results[0].score) < 1e-9]
        self.assertEqual(len(tied), 4)
        self.assertEqual([r.chunk_id for r in tied], sorted([source, *twins]))

    def test_query_is_stripped_and_embedded_exactly_once(self):
        provider = FakeEmbeddings(dimensions=DIMS)
        self.retriever(provider).search("   VPN down  \n", top_k=2)
        self.assertEqual(provider.calls, [["VPN down"]])


class CitationMetadataTests(IndexCase):
    def test_results_carry_original_text_and_full_citation_metadata(self):
        retriever = self.retriever()
        by_doc = {d.id: d for d in docs()}
        for r in retriever.search("deprecated vpn client", top_k=len(self.chunks)):
            chunk = self.chunks[r.chunk_id]
            doc = by_doc[r.doc_id]
            self.assertEqual(r.text, chunk.text)
            self.assertNotEqual(r.text, chunk.contextual_text)
            self.assertEqual(r.context, chunk.context)
            page = next(p.text for p in doc.pages if p.page == r.page)
            self.assertEqual(r.text, page[r.char_start:r.char_end])
            self.assertEqual((r.title, r.version, r.last_updated, r.status, r.authoritative),
                             (doc.title, doc.version, str(doc.last_updated), doc.status, bool(doc.authoritative)))
            self.assertEqual((r.owner, r.supersedes, r.applies_to, r.tags),
                             (doc.owner, doc.supersedes, tuple(doc.applies_to), tuple(doc.tags)))
            self.assertEqual((r.relative_path, r.format, r.doc_sha256, r.heading_path, r.error_codes),
                             (doc.relative_path, doc.format, doc.sha256, chunk.heading_path, chunk.error_codes))
            self.assertIn(r.doc_id, r.citation)
            self.assertIn(f"page {r.page}", r.citation)

    def test_draft_and_deprecated_chunks_are_returned_not_filtered(self):
        statuses = {r.status for r in self.retriever().search("anything", top_k=10_000)}
        self.assertEqual(statuses, {"current", "deprecated", "draft"})

    def test_non_authoritative_flag_is_preserved(self):
        results = self.retriever().search("anything", top_k=10_000)
        self.assertTrue({r.doc_id for r in results if not r.authoritative} >= {"KB-017"})
        self.assertTrue(any(r.authoritative for r in results))

    def test_untrusted_instructions_are_returned_as_inert_text(self):
        results = self.retriever().search("anything", top_k=10_000)
        hostile = [r for r in results if r.doc_id == "KB-018"]
        self.assertTrue(hostile)
        self.assertEqual({r.status for r in hostile}, {"draft"})
        self.assertTrue(all(isinstance(r.text, str) for r in hostile))


class ValidationTests(IndexCase):
    def test_empty_and_blank_queries_are_rejected_without_calling_the_provider(self):
        provider = FakeEmbeddings(dimensions=DIMS)
        retriever = self.retriever(provider)
        for bad in ("", "   ", "\n\t"):
            with self.assertRaises(QueryError):
                retriever.search(bad)
        with self.assertRaises(QueryError):
            retriever.search(None)  # type: ignore[arg-type]
        self.assertEqual(provider.calls, [])

    def test_overlong_query_and_bad_top_k_are_rejected(self):
        provider = FakeEmbeddings(dimensions=DIMS)
        retriever = self.retriever(provider)
        with self.assertRaises(QueryError):
            retriever.search("x" * (MAX_QUERY_CHARS + 1))
        for bad in (0, -1, True, 2.5):
            with self.assertRaises(QueryError):
                retriever.search("ok", top_k=bad)  # type: ignore[arg-type]
        self.assertEqual(provider.calls, [])

    def test_missing_index_is_an_error_and_no_file_is_created(self):
        missing = self.db.parent / "nope.db"
        with self.assertRaises(RetrievalError) as ctx:
            SemanticRetriever(missing, FakeEmbeddings(dimensions=DIMS))
        self.assertIn("build it first", str(ctx.exception))
        self.assertFalse(missing.exists())

    def test_empty_or_non_sqlite_file_is_an_error(self):
        for content in (b"", b"this is not a database at all" * 20):
            path = self.db.parent / "bad.db"
            path.write_bytes(content)
            with self.assertRaises(RetrievalError):
                SemanticRetriever(path, FakeEmbeddings(dimensions=DIMS))
            self.assertEqual(path.read_bytes(), content)

    def test_model_provider_and_dimension_mismatches_are_rejected(self):
        with self.assertRaises(RetrievalError) as ctx:
            SemanticRetriever(self.db, FakeEmbeddings(dimensions=DIMS, model="other-model"))
        self.assertIn("model", str(ctx.exception))
        with self.assertRaises(RetrievalError) as ctx:
            SemanticRetriever(self.db, FakeEmbeddings(dimensions=32))
        self.assertIn("32", str(ctx.exception))
        real_kind = FakeEmbeddings(dimensions=DIMS)
        real_kind.kind = "openrouter"
        with self.assertRaises(RetrievalError) as ctx:
            SemanticRetriever(self.db, real_kind)
        self.assertIn("never mixed", str(ctx.exception))

    def test_query_vector_with_wrong_dimensions_is_rejected(self):
        class Drifting(FakeEmbeddings):
            def embed(self, texts):
                self.expected_dimensions = None
                return [[1.0] * 8 for _ in texts]

        retriever = self.retriever()
        retriever.provider = Drifting(dimensions=DIMS)
        with self.assertRaises(RetrievalError):
            retriever.search("hello")

    def test_provider_failures_and_degenerate_query_vectors_propagate(self):
        class Failing(FakeEmbeddings):
            def embed(self, texts):
                raise EmbeddingError("simulated outage")

        class Zero(FakeEmbeddings):
            def embed(self, texts):
                return [[0.0] * DIMS for _ in texts]

        retriever = self.retriever()
        for provider in (Failing(dimensions=DIMS), Zero(dimensions=DIMS)):
            retriever.provider = provider
            with self.assertRaises(EmbeddingError):
                retriever.search("hello")

    def test_index_with_no_embeddings_is_an_error(self):
        self.sql("DELETE FROM embeddings")
        with self.assertRaises(RetrievalError):
            self.retriever()


class InvalidVectorTests(IndexCase):
    def corrupt(self, chunk_id: str, blob: bytes) -> None:
        self.sql("UPDATE embeddings SET vector = ? WHERE chunk_id = ?", (blob, chunk_id))

    def test_invalid_vectors_are_skipped_reported_and_never_returned(self):
        ids = sorted(self.chunks)
        unit = np.zeros(DIMS, dtype="<f4")
        unit[0] = 1.0
        bad = {
            ids[0]: pack_vector(unit)[:-1],  # truncated
            ids[1]: pack_vector(np.full(DIMS, np.nan, dtype="<f4")),
            ids[2]: pack_vector(unit * 5),  # not unit length
            ids[3]: pack_vector(np.zeros(DIMS, dtype="<f4")),
            ids[4]: pack_vector(np.full(DIMS, np.inf, dtype="<f4")),
        }
        for chunk_id, blob in bad.items():
            self.corrupt(chunk_id, blob)
        retriever = self.retriever()
        self.assertEqual({c for c, _ in retriever.skipped_invalid}, set(bad))
        self.assertEqual(len(retriever), len(ids) - len(bad))
        returned = {r.chunk_id for r in retriever.search("anything", top_k=10_000)}
        self.assertEqual(returned & set(bad), set())
        self.assertTrue(all(math.isfinite(r.score) for r in retriever.search("anything", top_k=5)))

    def test_vector_from_another_model_or_dimension_is_skipped(self):
        ids = sorted(self.chunks)
        self.sql("UPDATE embeddings SET model = 'someone-else' WHERE chunk_id = ?", (ids[0],))
        self.sql("UPDATE embeddings SET dimensions = 32 WHERE chunk_id = ?", (ids[1],))
        retriever = self.retriever()
        self.assertEqual({c for c, _ in retriever.skipped_invalid}, {ids[0], ids[1]})

    def test_all_vectors_invalid_is_an_error(self):
        self.sql("UPDATE embeddings SET vector = zeroblob(length(vector))")
        with self.assertRaises(RetrievalError) as ctx:
            self.retriever()
        self.assertIn("no usable embeddings", str(ctx.exception))


class ReadOnlyTests(IndexCase):
    def state(self):
        stat = self.db.stat()
        return (hashlib.sha256(self.db.read_bytes()).hexdigest(), stat.st_mtime_ns, stat.st_size,
                sorted(p.name for p in self.db.parent.iterdir()))

    def test_searching_never_modifies_the_database(self):
        before = self.state()
        retriever = self.retriever()
        for query in ("vpn", "printer", "password reset"):
            retriever.search(query, top_k=5)
        self.assertEqual(self.state(), before)

    def test_the_retriever_holds_no_open_database_handle(self):
        retriever = self.retriever()
        self.db.unlink()  # would fail on Windows if the file were still open
        self.assertEqual(len(retriever.search("still works from the loaded snapshot", 3)), 3)


class CliTests(IndexCase):
    def run_cli(self, *argv, provider="default") -> tuple[int, list[str]]:
        lines: list[str] = []
        provider = FakeEmbeddings(dimensions=DIMS) if provider == "default" else provider
        code = main([*argv], provider=provider, out=lines.append)
        return code, lines

    def test_search_prints_ranked_results_with_citations_and_flags(self):
        code, lines = self.run_cli("My VPN keeps disconnecting", "--mode", "semantic", "--provider", "fake",
                                   "--db", str(self.db), "--top-k", "3")
        self.assertEqual(code, 0)
        text = "\n".join(lines)
        self.assertIn("3 result(s) from 117 indexed chunks", text)
        self.assertEqual(sum(1 for line in lines if line[:3] in ("1. ", "2. ", "3. ")), 3)
        self.assertIn("status=", text)
        self.assertIn("authoritative=", text)
        self.assertIn("page 1", text)

    def test_full_flag_prints_the_complete_chunk_text(self):
        chunk = self.chunks[sorted(self.chunks)[10]]
        code, lines = self.run_cli(chunk.contextual_text, "--mode", "semantic", "--provider", "fake",
                                   "--db", str(self.db), "--top-k", "1", "--full")
        self.assertEqual(code, 0)
        self.assertIn(chunk.text, "\n".join(lines))

    def test_cli_error_exit_codes(self):
        base = ["--provider", "fake", "--db", str(self.db)]
        self.assertEqual(self.run_cli("   ", *base)[0], 2)
        self.assertEqual(self.run_cli("vpn", "--top-k", "0", *base)[0], 2)
        self.assertEqual(self.run_cli("vpn", "--provider", "fake", "--db", str(self.db.parent / "none.db"))[0], 2)
        self.assertEqual(self.run_cli("vpn", *base, provider=FakeEmbeddings(dimensions=DIMS, model="x"))[0], 2)

        class Down(FakeEmbeddings):
            def embed(self, texts):
                raise EmbeddingError("provider is down")

        code, lines = self.run_cli("vpn", *base, provider=Down(dimensions=DIMS))
        self.assertEqual(code, 1)
        self.assertIn("provider is down", "\n".join(lines))

    def test_cli_warns_about_skipped_vectors(self):
        self.sql("UPDATE embeddings SET vector = zeroblob(length(vector)) WHERE chunk_id = ?", (sorted(self.chunks)[0],))
        code, lines = self.run_cli("vpn", "--provider", "fake", "--db", str(self.db))
        self.assertEqual(code, 0)
        self.assertTrue(any(line.startswith("warning:") and "1 invalid" in line for line in lines))

    def test_pinned_model_is_read_from_the_index_without_changing_it(self):
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        self.assertEqual(pinned_model(self.db), self.fake.model)
        self.assertIsNone(pinned_model(self.db.parent / "missing.db"))
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), before)

    def test_default_openrouter_provider_uses_the_models_pinned_in_the_index(self):
        # No network: a missing key stops before any request, proving the CLI built an OpenRouter provider.
        import os
        from unittest import mock
        with mock.patch("ingestion.embeddings._load_env", return_value=True), mock.patch.dict(os.environ, {}, clear=True):
            lines: list[str] = []
            code = main(["vpn", "--db", str(self.db)], out=lines.append)
        self.assertEqual(code, 2)
        self.assertTrue(any("OPENROUTER_API_KEY" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
