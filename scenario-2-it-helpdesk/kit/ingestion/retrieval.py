"""Read-only semantic retrieval over the SQLite index built in Phase 1C (Phase 2A).

The query is embedded with the provider that matches the index's pinned model, scored against the
stored unit vectors with NumPy (cosine similarity is a dot product for unit vectors) and returned
best-first. Ties break on chunk_id, so results are deterministic.

Deliberately not here: status/authority/department policy, answering. Exact error-code matching lives
in ingestion.hybrid, which builds on the public helpers below.
Every result carries the stored policy metadata (status, authoritative, ...) as plain data for the
later policy layer, and chunk text is returned untouched and must be treated as untrusted.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ingestion.embeddings import EmbeddingProvider, normalize_vector
from ingestion.index_store import IndexMismatchError, IndexStore, vector_problem

MAX_QUERY_CHARS = 2000  # keeps one query far below the embedding model's input limit and cost


class RetrievalError(RuntimeError):
    """The index is missing, unreadable, empty or incompatible with the provider."""


class QueryError(ValueError):
    """The query or search parameters are invalid."""


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float  # cosine similarity in [-1, 1]
    chunk_id: str
    doc_id: str
    text: str  # original source text, exactly as chunked (never the added context)
    context: str  # heading breadcrumb that was added for embedding; not source text
    heading_path: tuple[str, ...]
    page: int
    char_start: int
    char_end: int
    title: str
    version: str
    last_updated: str
    status: str
    authoritative: bool
    owner: str | None
    supersedes: str | None
    applies_to: tuple[str, ...]
    tags: tuple[str, ...]
    error_codes: tuple[str, ...]  # declared codes that literally appear in this chunk
    source_path: str
    relative_path: str
    format: str
    doc_sha256: str

    @property
    def citation(self) -> str:
        section = f", {' > '.join(self.heading_path)}" if self.heading_path else ""
        return f"{self.doc_id} v{self.version} ({self.relative_path}, page {self.page}{section})"


class SemanticRetriever:
    """Loads a snapshot of the index once; the database is opened read-only and never written."""

    def __init__(self, db_path: Path | str, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self.db_path = Path(db_path)
        self.skipped_invalid: list[tuple[str, str]] = []  # (chunk_id, reason) of vectors that were not searched
        if not self.db_path.is_file():
            raise RetrievalError(f"no index at {self.db_path}: build it first with python -m ingestion.build_index")
        try:
            store = IndexStore(self.db_path, provider.kind, provider.model, read_only=True)
        except IndexMismatchError as exc:
            raise RetrievalError(str(exc)) from None
        except sqlite3.Error as exc:
            raise RetrievalError(f"{self.db_path} is not a readable index: {exc}") from None
        try:
            self.dimensions = store.dimensions
            self._load(store)
        except sqlite3.Error as exc:
            raise RetrievalError(f"{self.db_path} could not be read: {exc}") from None
        finally:
            store.close()
        if provider.expected_dimensions is not None and provider.expected_dimensions != self.dimensions:
            raise RetrievalError(f"provider produces {provider.expected_dimensions}-dimensional vectors but the "
                                 f"index holds {self.dimensions}-dimensional vectors")

    def _load(self, store: IndexStore) -> None:
        if self.dimensions is None:
            raise RetrievalError(f"{self.db_path} is empty: no embeddings have been indexed")
        rows = store.conn.execute(
            "SELECT c.*, e.vector AS vec, e.dimensions AS vec_dims, e.model AS vec_model "
            "FROM chunks c JOIN embeddings e USING (chunk_id) ORDER BY c.chunk_id").fetchall()
        vectors, self._chunks = [], []
        for row in rows:
            problem = vector_problem(bytes(row["vec"]), row["vec_dims"], self.dimensions, row["vec_model"], store.model)
            if problem:
                self.skipped_invalid.append((row["chunk_id"], problem))
                continue
            vectors.append(np.frombuffer(bytes(row["vec"]), dtype="<f4"))
            self._chunks.append(row)
        if not vectors:
            detail = f" ({len(self.skipped_invalid)} invalid vectors skipped)" if self.skipped_invalid else ""
            raise RetrievalError(f"{self.db_path} has no usable embeddings{detail}")
        self._matrix = np.stack(vectors).astype(np.float64)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunk_rows(self) -> list[sqlite3.Row]:
        """The searchable chunk rows, ordered by chunk_id; positions match score_all()."""
        return self._chunks

    @staticmethod
    def check_search_args(query: str, top_k: int) -> str:
        """Validate the search arguments and return the stripped query."""
        if not isinstance(query, str) or not query.strip():
            raise QueryError("query is empty")
        query = query.strip()
        if len(query) > MAX_QUERY_CHARS:
            raise QueryError(f"query is {len(query)} characters; the limit is {MAX_QUERY_CHARS}")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise QueryError("top_k must be a positive integer")
        return query

    def score_all(self, query: str) -> np.ndarray:
        """Embed an already validated query once and return its cosine similarity to every chunk."""
        (raw,) = self.provider.embed([query])  # EmbeddingError propagates unchanged
        vector = normalize_vector(raw).astype(np.float64)
        if vector.shape[0] != self.dimensions:
            raise RetrievalError(f"query embedding has {vector.shape[0]} dimensions but the index holds {self.dimensions}")
        return self._matrix @ vector

    def order(self, scores: np.ndarray) -> list[int]:
        """Chunk positions best-first. Rounding only decides ordering among near-identical scores."""
        return sorted(range(len(scores)), key=lambda i: (-round(float(scores[i]), 9), self._chunks[i]["chunk_id"]))

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query = self.check_search_args(query, top_k)
        scores = self.score_all(query)
        return [self.result_at(i, rank, float(scores[i]))
                for rank, i in enumerate(self.order(scores)[:top_k], start=1)]

    def result_at(self, position: int, rank: int, score: float) -> SearchResult:
        return self._result(rank, self._chunks[position], score)

    @staticmethod
    def _result(rank: int, row: sqlite3.Row, score: float) -> SearchResult:
        return SearchResult(
            rank=rank, score=score, chunk_id=row["chunk_id"], doc_id=row["doc_id"], text=row["text"],
            context=row["context"], heading_path=tuple(json.loads(row["heading_path"])), page=row["page"],
            char_start=row["char_start"], char_end=row["char_end"], title=row["title"], version=row["version"],
            last_updated=row["last_updated"], status=row["status"], authoritative=bool(row["authoritative"]),
            owner=row["owner"], supersedes=row["supersedes"], applies_to=tuple(json.loads(row["applies_to"])),
            tags=tuple(json.loads(row["tags"])), error_codes=tuple(json.loads(row["chunk_error_codes"])),
            source_path=row["source_path"], relative_path=row["relative_path"], format=row["format"],
            doc_sha256=row["doc_sha256"])
