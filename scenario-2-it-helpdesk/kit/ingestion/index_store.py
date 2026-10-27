"""SQLite persistence for chunk metadata and embedding vectors (Phase 1C).

One index database is pinned to one (provider kind, model, dimensions). Documents are replaced
atomically, one transaction per document, so a failure never leaves a half-written document.
Nothing here ranks or filters by policy: status, authoritative, version and the other validated
metadata are stored as data for the later retrieval phase.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Sequence

import numpy as np

from ingestion.chunking import CHUNKER_VERSION, Chunk, ChunkConfig, chunk_document
from ingestion.embeddings import EmbeddingError, EmbeddingProvider, normalize_vector
from ingestion.loaders import Document

SCHEMA_VERSION = 1
MAX_CONSECUTIVE_FAILURES = 3

SCHEMA = """
CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, version TEXT NOT NULL, last_updated TEXT NOT NULL,
    status TEXT NOT NULL, authoritative INTEGER NOT NULL, owner TEXT, supersedes TEXT,
    applies_to TEXT NOT NULL, tags TEXT NOT NULL, error_codes TEXT NOT NULL,
    source_path TEXT NOT NULL, relative_path TEXT NOT NULL, format TEXT NOT NULL, doc_sha256 TEXT NOT NULL,
    chunker_version TEXT NOT NULL, fingerprint TEXT NOT NULL, indexed_at TEXT NOT NULL);
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL, page INTEGER NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
    text TEXT NOT NULL, context TEXT NOT NULL, content_hash TEXT NOT NULL, heading_path TEXT NOT NULL,
    chunk_error_codes TEXT NOT NULL,
    title TEXT NOT NULL, version TEXT NOT NULL, last_updated TEXT NOT NULL, status TEXT NOT NULL,
    authoritative INTEGER NOT NULL, owner TEXT, supersedes TEXT, applies_to TEXT NOT NULL, tags TEXT NOT NULL,
    source_path TEXT NOT NULL, relative_path TEXT NOT NULL, format TEXT NOT NULL, doc_sha256 TEXT NOT NULL);
CREATE TABLE embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model TEXT NOT NULL, dimensions INTEGER NOT NULL, embedded_text_sha256 TEXT NOT NULL,
    vector BLOB NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX idx_chunks_doc ON chunks(doc_id);
CREATE INDEX idx_chunks_status ON chunks(status);
CREATE INDEX idx_embeddings_hash ON embeddings(embedded_text_sha256);
"""


class IndexMismatchError(RuntimeError):
    """The index was built with a different provider, model, dimension or schema."""


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_vector(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def unpack_vector(blob: bytes, dimensions: int) -> np.ndarray:
    if len(blob) != 4 * dimensions:
        raise ValueError(f"vector blob has {len(blob)} bytes, expected {4 * dimensions}")
    return np.frombuffer(blob, dtype="<f4")


def vector_problem(blob: bytes, dimensions: int, pinned: int | None, row_model: str, model: str) -> str | None:
    """Why a stored vector must not be trusted or reused, or None if it is sound."""
    if row_model != model:
        return f"model {row_model!r} is not {model!r}"
    if pinned is None or dimensions != pinned:
        return f"dimensions {dimensions} do not match the index ({pinned})"
    try:
        vector = unpack_vector(blob, dimensions)
    except ValueError as exc:
        return str(exc)
    if not np.isfinite(vector).all():
        return "vector has NaN or infinite values"
    if abs(float(np.linalg.norm(vector)) - 1.0) > 1e-4:
        return "vector is not unit length"
    return None


def _document_row(doc: Document) -> dict:
    return {
        "doc_id": doc.id, "title": doc.title, "version": doc.version, "last_updated": str(doc.last_updated),
        "status": doc.status, "authoritative": int(bool(doc.authoritative)), "owner": doc.owner,
        "supersedes": doc.supersedes, "applies_to": _json(list(doc.applies_to)), "tags": _json(list(doc.tags)),
        "error_codes": _json(list(doc.error_codes)), "source_path": str(doc.source_path),
        "relative_path": doc.relative_path, "format": doc.format, "doc_sha256": doc.sha256,
    }


def _chunk_row(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "ordinal": chunk.ordinal, "page": chunk.page,
        "char_start": chunk.char_start, "char_end": chunk.char_end, "text": chunk.text, "context": chunk.context,
        "content_hash": chunk.content_hash, "heading_path": _json(list(chunk.heading_path)),
        "chunk_error_codes": _json(list(chunk.error_codes)), "title": chunk.title, "version": chunk.version,
        "last_updated": str(chunk.last_updated), "status": chunk.status,
        "authoritative": int(bool(chunk.authoritative)), "owner": chunk.owner, "supersedes": chunk.supersedes,
        "applies_to": _json(list(chunk.applies_to)), "tags": _json(list(chunk.tags)),
        "source_path": str(chunk.source_path), "relative_path": chunk.relative_path, "format": chunk.format,
        "doc_sha256": chunk.doc_sha256,
    }


def embedded_text_hash(chunk: Chunk) -> str:
    return _sha256(chunk.contextual_text)


def document_fingerprint(doc: Document, chunks: Sequence[Chunk]) -> str:
    """Hash of everything stored for a document: metadata, chunk ids, text, context and offsets.

    A change in any of them (not only the file hash) makes the document stale.
    """
    payload = {
        "chunker_version": CHUNKER_VERSION, "document": _document_row(doc),
        "chunks": [dict(_chunk_row(c), embedded_text_sha256=embedded_text_hash(c)) for c in chunks],
    }
    return _sha256(_json(payload))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IndexStore:
    def __init__(self, path: Path | str, provider_kind: str, model: str, *, read_only: bool = False) -> None:
        self.path = str(path)
        self.provider_kind, self.model = provider_kind, model
        self.read_only = read_only
        if read_only:  # the file is opened read-only, so nothing (schema, meta, timestamps) can change
            uri = Path(self.path).resolve().as_uri() + "?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        else:
            self.conn = sqlite3.connect(self.path, isolation_level=None)  # explicit transactions
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row
        try:
            self._init_or_check()
        except BaseException:
            self.conn.close()  # never leak the file handle when the index is rejected
            raise

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "IndexStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _init_or_check(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0 and self.read_only:
            raise IndexMismatchError(f"{self.path} contains no index")
        if version == 0:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in SCHEMA.split(";"):
                    if statement.strip():
                        self.conn.execute(statement)
                self.conn.executemany("INSERT INTO index_meta VALUES (?, ?)", [
                    ("provider_kind", self.provider_kind), ("model", self.model),
                    ("vector_format", "float32-le-normalized")])
                self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
        elif version != SCHEMA_VERSION:
            raise IndexMismatchError(f"{self.path}: schema version {version}, this code expects {SCHEMA_VERSION}")
        meta = self.meta()
        for key, wanted in (("provider_kind", self.provider_kind), ("model", self.model)):
            if meta.get(key) != wanted:
                raise IndexMismatchError(
                    f"{self.path} was built with {key}={meta.get(key)!r}, not {wanted!r}. Real and fake or "
                    "differently modelled vectors are never mixed: use another --db path or delete this file.")

    def meta(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM index_meta")}

    @property
    def dimensions(self) -> int | None:
        value = self.meta().get("dimensions")
        return int(value) if value else None

    def document_ids(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT doc_id FROM documents ORDER BY doc_id")]

    def document_state(self, doc_id: str) -> tuple[str, int] | None:
        row = self.conn.execute(
            "SELECT d.fingerprint, (SELECT COUNT(*) FROM chunks c JOIN embeddings e USING (chunk_id) "
            "WHERE c.doc_id = d.doc_id) FROM documents d WHERE d.doc_id = ?", (doc_id,)).fetchone()
        return (row[0], row[1]) if row else None

    def chunk_count(self, doc_id: str | None = None) -> int:
        if doc_id is None:
            return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()[0]

    def vectors_by_text_hash(self, hashes: Iterable[str]) -> tuple[dict[str, bytes], int]:
        """Reusable vectors by embedded-text hash, plus how many stored candidates were rejected.

        A candidate is reused only if it is for this model, matches the pinned dimensions, is
        finite and unit length. Anything else is ignored, so the text is embedded again.
        """
        found: dict[str, bytes] = {}
        rejected = 0
        pinned = self.dimensions
        for h in dict.fromkeys(hashes):
            for row in self.conn.execute(
                    "SELECT vector, dimensions, model FROM embeddings WHERE embedded_text_sha256 = ?", (h,)):
                if vector_problem(bytes(row["vector"]), row["dimensions"], pinned, row["model"], self.model) is None:
                    found[h] = bytes(row["vector"])
                    break
                rejected += 1
        return found, rejected

    def get_vector(self, chunk_id: str) -> np.ndarray:
        row = self.conn.execute("SELECT vector, dimensions FROM embeddings WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if row is None:
            raise KeyError(chunk_id)
        return unpack_vector(row["vector"], row["dimensions"])

    def pin_dimensions(self, dimensions: int) -> None:
        pinned = self.dimensions
        if pinned is None:
            self.conn.execute("INSERT INTO index_meta VALUES ('dimensions', ?)", (str(dimensions),))
        elif pinned != dimensions:
            raise IndexMismatchError(f"index holds {pinned}-dimensional vectors, provider returned {dimensions}")

    def replace_document(self, doc: Document, chunks: Sequence[Chunk], vectors: dict[str, bytes],
                         dimensions: int | None) -> int:
        """Atomically replace one document's rows. Returns how many old chunks were removed."""
        self._require_writable()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            if dimensions is not None:
                self.pin_dimensions(dimensions)
            pinned = self.dimensions
            for chunk in chunks:
                blob = vectors[embedded_text_hash(chunk)]
                if pinned is None or len(blob) != 4 * pinned:
                    raise IndexMismatchError(f"vector for {chunk.chunk_id} does not match the pinned dimensions")
            removed = self.chunk_count(doc.id)
            self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc.id,))  # cascades to chunks, embeddings
            row = dict(_document_row(doc), chunker_version=CHUNKER_VERSION,
                       fingerprint=document_fingerprint(doc, chunks), indexed_at=_now())
            self._insert("documents", row)
            for chunk in chunks:
                self._insert("chunks", _chunk_row(chunk))
                self._insert("embeddings", {
                    "chunk_id": chunk.chunk_id, "model": self.model, "dimensions": pinned,
                    "embedded_text_sha256": embedded_text_hash(chunk),
                    "vector": vectors[embedded_text_hash(chunk)], "created_at": _now()})
            self.conn.execute("COMMIT")
            return removed
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    def _require_writable(self) -> None:
        if self.read_only:
            raise IndexMismatchError("index was opened read-only")

    def delete_document(self, doc_id: str) -> int:
        self._require_writable()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            removed = self.chunk_count(doc_id)
            self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self.conn.execute("COMMIT")
            return removed
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    def _insert(self, table: str, row: dict) -> None:
        columns = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self.conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(row.values()))


@dataclass
class IndexReport:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    not_attempted: list[str] = field(default_factory=list)
    stale_documents: list[str] = field(default_factory=list)  # in the index but no longer in the dataset
    pruned: list[str] = field(default_factory=list)
    chunks_written: int = 0
    chunks_removed: int = 0
    texts_embedded: int = 0   # texts sent to the provider (API usage)
    vectors_reused: int = 0
    vectors_rejected: int = 0  # stored vectors that failed validation and were not reused
    dry_run: bool = False
    prune_skipped: bool = False  # --prune was requested but withheld because indexing was incomplete

    @property
    def ok(self) -> bool:
        return not self.failed and not self.not_attempted


def index_documents(store: IndexStore, docs: Sequence[Document], provider: EmbeddingProvider,
                    config: ChunkConfig = ChunkConfig(), *, prune: bool = False, dry_run: bool = False,
                    log: Callable[[str], None] = lambda message: None) -> IndexReport:
    """Chunk, embed and index documents. Idempotent: unchanged documents cost no API calls."""
    report = IndexReport(dry_run=dry_run)
    consecutive_failures = 0
    for position, doc in enumerate(docs):
        chunks = chunk_document(doc, config)
        fingerprint = document_fingerprint(doc, chunks)
        state = store.document_state(doc.id)
        if state == (fingerprint, len(chunks)):
            report.skipped.append(doc.id)
            continue
        hashes = [embedded_text_hash(c) for c in chunks]
        vectors, rejected = store.vectors_by_text_hash(hashes)
        report.vectors_rejected += rejected
        reused = len(vectors)
        missing = list(dict.fromkeys(h for h in hashes if h not in vectors))
        text_for = {embedded_text_hash(c): c.contextual_text for c in chunks}
        dimensions = None
        if dry_run:
            report.texts_embedded += len(missing)
            report.vectors_reused += reused
            report.indexed.append(doc.id)
            continue
        try:
            if missing:
                raw = provider.embed([text_for[h] for h in missing])
                if len(raw) != len(missing):
                    raise EmbeddingError(f"provider returned {len(raw)} vectors for {len(missing)} inputs")
                normalized = [normalize_vector(v) for v in raw]
                dimensions = len(normalized[0])
                vectors.update({h: pack_vector(v) for h, v in zip(missing, normalized)})
            report.chunks_removed += store.replace_document(doc, chunks, vectors, dimensions)
        except EmbeddingError as exc:
            report.failed[doc.id] = str(exc)
            consecutive_failures += 1
            log(f"FAILED {doc.id}: {exc}")
            if exc.fatal or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                report.not_attempted.extend(d.id for d in docs[position + 1:])
                break
            continue
        except IndexMismatchError as exc:
            report.failed[doc.id] = str(exc)
            report.not_attempted.extend(d.id for d in docs[position + 1:])
            break
        consecutive_failures = 0
        report.indexed.append(doc.id)
        report.chunks_written += len(chunks)
        report.texts_embedded += len(missing)
        report.vectors_reused += reused
        log(f"indexed {doc.id}: {len(chunks)} chunks, {len(missing)} embedded, {reused} reused")
    known = {d.id for d in docs}
    report.stale_documents = [d for d in store.document_ids() if d not in known]
    # Pruning is destructive, so it only runs after a fully successful pass over every document.
    report.prune_skipped = prune and not dry_run and not report.ok
    if prune and not dry_run and report.ok:
        for doc_id in report.stale_documents:
            report.chunks_removed += store.delete_document(doc_id)
            report.pruned.append(doc_id)
        report.stale_documents = []
    return report


def verify_index(store: IndexStore, docs: Sequence[Document] | None = None,
                 config: ChunkConfig = ChunkConfig()) -> list[str]:
    """Offline integrity check (no network). Returns human-readable problems; empty means healthy."""
    problems: list[str] = []
    dims = store.dimensions
    q = store.conn.execute
    for (chunk_id,) in q("SELECT chunk_id FROM chunks WHERE chunk_id NOT IN (SELECT chunk_id FROM embeddings)"):
        problems.append(f"chunk without embedding: {chunk_id}")
    for (chunk_id,) in q("SELECT chunk_id FROM chunks WHERE doc_id NOT IN (SELECT doc_id FROM documents)"):
        problems.append(f"orphan chunk: {chunk_id}")
    for row in q("SELECT chunk_id, text, content_hash, char_start, char_end FROM chunks"):
        if _sha256(row["text"]) != row["content_hash"]:
            problems.append(f"content hash mismatch: {row['chunk_id']}")
        if row["char_end"] - row["char_start"] != len(row["text"]):
            problems.append(f"offset/length mismatch: {row['chunk_id']}")
    for row in q("SELECT e.chunk_id, e.vector, e.dimensions, e.model, e.embedded_text_sha256, c.context, c.text "
                 "FROM embeddings e JOIN chunks c USING (chunk_id)"):
        issue = vector_problem(bytes(row["vector"]), row["dimensions"], dims, row["model"], store.model)
        if issue:
            problems.append(f"{row['chunk_id']}: {issue}")
        # Same property the indexer hashed, so the formatting rule (including empty context) lives only in Chunk.
        stored = SimpleNamespace(context=row["context"], text=row["text"])
        if _sha256(Chunk.contextual_text.fget(stored)) != row["embedded_text_sha256"]:
            problems.append(f"embedded text hash does not match context+text: {row['chunk_id']}")
    if docs is not None:
        indexed = set(store.document_ids())
        for doc in docs:
            if doc.id not in indexed:
                problems.append(f"document missing from index: {doc.id}")
                continue
            chunks = chunk_document(doc, config)
            if store.document_state(doc.id) != (document_fingerprint(doc, chunks), len(chunks)):
                problems.append(f"document stale or incomplete: {doc.id}")
        for doc_id in sorted(indexed - {d.id for d in docs}):
            problems.append(f"document no longer in dataset: {doc_id}")
    return problems
