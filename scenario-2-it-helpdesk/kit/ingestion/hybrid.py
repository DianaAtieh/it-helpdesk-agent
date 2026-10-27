"""Hybrid candidate retrieval: semantic similarity plus exact error-code matches (Phase 2B).

Ranking (deterministic, applied to the union of candidates):
  1. number of the query's error codes found in the chunk's ORIGINAL text, as whole tokens (more first);
  2. semantic cosine similarity (higher first, compared at 9 decimals);
  3. chunk_id (ascending).
Candidates are the semantic top_k plus every chunk with an exact match; after ranking, at most top_k
are returned. A query without codes therefore ranks exactly like SemanticRetriever.search.

Evidence is never merged into one number: each result keeps its true cosine `semantic_score` (and
`semantic_rank` over the whole index) and, separately, the exact-match evidence (`matched_codes`,
`match_type`). Exact matching reads only the source text, never the added heading context and never the
substring-derived chunk metadata. No status, authority or applicability policy is applied here:
draft, deprecated and non-authoritative chunks stay visible and carry their metadata for Phase 2C.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ingestion.embeddings import EmbeddingProvider
from ingestion.error_codes import code_prefixes, codes_in_text, extract_codes
from ingestion.retrieval import SearchResult, SemanticRetriever


@dataclass(frozen=True)
class HybridResult:
    rank: int  # position in the hybrid ordering (1-based)
    chunk: SearchResult  # untouched Phase 2A result; chunk.rank and chunk.score are the semantic rank and cosine
    semantic_score: float  # cosine similarity of the query to this chunk (never a blended value)
    semantic_rank: int  # rank over the whole index by cosine alone
    matched_codes: tuple[str, ...]  # query codes found as whole tokens in chunk.text
    query_codes: tuple[str, ...]  # all codes recognised in the query
    match_type: str  # "complete" (all query codes), "partial", or "none"
    from_semantic_top_k: bool  # the chunk is in the semantic top_k

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def citation(self) -> str:
        return self.chunk.citation


@dataclass(frozen=True)
class HybridResponse:
    results: tuple[HybridResult, ...]
    query_codes: tuple[str, ...]  # recognised codes, canonical form, first-seen order
    unmatched_codes: tuple[str, ...]  # recognised codes found in no chunk (reported, never fuzzy-matched)


class HybridRetriever:
    def __init__(self, db_path: Path | str, provider: EmbeddingProvider) -> None:
        self.semantic = SemanticRetriever(db_path, provider)
        rows = self.semantic.chunk_rows
        self.prefixes = code_prefixes(code for row in rows for code in json.loads(row["chunk_error_codes"]))

    @property
    def skipped_invalid(self) -> list[tuple[str, str]]:
        return self.semantic.skipped_invalid

    @property
    def dimensions(self) -> int | None:
        return self.semantic.dimensions

    @property
    def provider(self) -> EmbeddingProvider:
        return self.semantic.provider

    def __len__(self) -> int:
        return len(self.semantic)

    def search(self, query: str, top_k: int = 5) -> HybridResponse:
        query = self.semantic.check_search_args(query, top_k)
        query_codes = extract_codes(query, self.prefixes)
        scores = self.semantic.score_all(query)  # the only embedding call
        order = self.semantic.order(scores)
        semantic_rank = {position: rank for rank, position in enumerate(order, start=1)}
        rows = self.semantic.chunk_rows
        matched = {position: codes_in_text(rows[position]["text"], query_codes) for position in range(len(rows))} \
            if query_codes else {}
        exact = {position for position, codes in matched.items() if codes}
        top_semantic = set(order[:top_k])
        candidates = exact | top_semantic
        ranked = sorted(candidates, key=lambda i: (-len(matched.get(i, ())), -round(float(scores[i]), 9),
                                                   rows[i]["chunk_id"]))
        results = []
        for rank, position in enumerate(ranked[:top_k], start=1):
            codes = matched.get(position, ())
            kind = "none" if not codes else "complete" if len(codes) == len(query_codes) else "partial"
            score = float(scores[position])
            results.append(HybridResult(
                rank=rank, chunk=self.semantic.result_at(position, semantic_rank[position], score),
                semantic_score=score, semantic_rank=semantic_rank[position], matched_codes=codes,
                query_codes=query_codes, match_type=kind, from_semantic_top_k=position in top_semantic))
        found = {code for codes in matched.values() for code in codes}
        return HybridResponse(tuple(results), query_codes, tuple(c for c in query_codes if c not in found))
