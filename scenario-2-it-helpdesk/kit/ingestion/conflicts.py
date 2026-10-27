"""Optional known-conflict annotations from dataset/manifest.json (Phase 2C). Diagnostic metadata only.

The manifest lists documented conflicts (C1-C6) and how the precedence rules resolve them. When two or
more documents of one conflict appear among the relevant chunks, the conflict is attached to the
response as `diagnostic_only` metadata. This does NOT detect or resolve conflicts between arbitrary
chunks and is never part of the answer-facing view; the downstream verifier owns conflict resolution.
A malformed manifest raises ManifestError; callers degrade to "no annotations" with a warning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MAX_MANIFEST_BYTES = 2_000_000


class ManifestError(ValueError):
    """The manifest is missing, unreadable or does not have the expected structure."""


@dataclass(frozen=True)
class KnownConflict:
    id: str
    topic: str
    documents: tuple[str, ...]
    resolution: str


@dataclass(frozen=True)
class ConflictAnnotation:
    id: str
    topic: str
    documents: tuple[str, ...]  # all documents the manifest lists for this conflict
    present_documents: tuple[str, ...]  # those that appear among the relevant chunks
    resolution: str  # manifest text, informational only
    diagnostic_only: bool = True


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{where} must be a non-empty string")
    return value


def load_known_conflicts(path: Path | str) -> tuple[KnownConflict, ...]:
    path = Path(path)
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ManifestError(f"{path} is larger than {MAX_MANIFEST_BYTES} bytes")
        data = json.loads(path.read_text(encoding="utf-8"))
    except ManifestError:
        raise
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from None
    entries = data.get("known_conflicts") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ManifestError(f"{path}: 'known_conflicts' must be a list")
    conflicts, seen = [], set()
    for index, entry in enumerate(entries):
        where = f"known_conflicts[{index}]"
        if not isinstance(entry, dict):
            raise ManifestError(f"{where} must be an object")
        conflict_id = _text(entry.get("id"), f"{where}.id")
        if conflict_id in seen:
            raise ManifestError(f"duplicate conflict id {conflict_id}")
        seen.add(conflict_id)
        documents = entry.get("documents")
        if not isinstance(documents, list) or not documents or not all(isinstance(d, str) and d for d in documents):
            raise ManifestError(f"{where}.documents must be a non-empty list of document ids")
        conflicts.append(KnownConflict(conflict_id, _text(entry.get("topic"), f"{where}.topic"),
                                       tuple(dict.fromkeys(documents)), _text(entry.get("resolution"), f"{where}.resolution")))
    return tuple(conflicts)


def annotate(conflicts: Iterable[KnownConflict], relevant_doc_ids: Iterable[str]) -> tuple[ConflictAnnotation, ...]:
    present = set(relevant_doc_ids)
    found = []
    for conflict in sorted(conflicts, key=lambda c: c.id):
        here = tuple(d for d in conflict.documents if d in present)
        if len(here) >= 2:
            found.append(ConflictAnnotation(conflict.id, conflict.topic, conflict.documents, here, conflict.resolution))
    return tuple(found)
