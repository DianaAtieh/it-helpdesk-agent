"""Try the retriever from the command line (one query embedding call; the index is read-only).

    .venv\\Scripts\\python.exe -m ingestion.search "NX-781 keeps appearing on my VPN" --top-k 5
    .venv\\Scripts\\python.exe -m ingestion.search "My VPN keeps disconnecting" --mode semantic

Modes: policy (default) applies hard evidence eligibility, a relevance floor and tiers (ingestion.policy_retrieval);
hybrid adds exact error-code matches to the semantic candidates (ingestion.hybrid); semantic is plain cosine search.
hybrid and semantic are diagnostics: they apply no policy and can return deprecated, draft or superseded text.
The query is embedded with the model pinned in the index (override only with --model, which must match).
Exit codes: 0 success, 1 embedding/provider failure, 2 invalid query or index/configuration problem.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Callable, Sequence

from ingestion.build_index import DEFAULT_DB
from ingestion.embeddings import EmbeddingError, EmbeddingProvider, FakeEmbeddings, OpenRouterEmbeddings
from ingestion.hybrid import HybridResult, HybridRetriever
from ingestion.policy_retrieval import (DEFAULT_MANIFEST, DEFAULT_MIN_SEMANTIC_SCORE, EvidenceItem, PolicyResponse,
                                        PolicyRetriever)
from ingestion.retrieval import QueryError, RetrievalError, SearchResult, SemanticRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="search", description=__doc__.split("\n\n")[0])
    parser.add_argument("query", help="the user question")
    parser.add_argument("--top-k", type=int, default=5, help="maximum number of results returned")
    parser.add_argument("--mode", choices=("policy", "hybrid", "semantic"), default="policy",
                        help="policy = eligibility-filtered evidence (default); hybrid and semantic are diagnostics")
    parser.add_argument("--department", help="the user's department (never inferred); policy mode only")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SEMANTIC_SCORE,
                        help="policy mode relevance floor for semantic scores (exact code matches bypass it)")
    parser.add_argument("--annotate-conflicts", action="store_true",
                        help="policy mode: attach known conflicts from dataset/manifest.json (diagnostic only)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="policy mode: list excluded chunks; also shows their UNTRUSTED text")
    parser.add_argument("--provider", choices=sorted(DEFAULT_DB), default="openrouter")
    parser.add_argument("--db", type=Path, help="index file (default: ingestion/index.db, or index.fake.db for fake)")
    parser.add_argument("--model", help="embedding model; defaults to the model pinned in the index")
    parser.add_argument("--full", action="store_true", help="print the full chunk text instead of a preview")
    return parser


def pinned_model(db_path: Path) -> str | None:
    """The model the index was built with, read without modifying the file."""
    try:
        conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT value FROM index_meta WHERE key = 'model'").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def format_result(result: SearchResult, full: bool) -> list[str]:
    flags = f"status={result.status} | authoritative={'yes' if result.authoritative else 'no'}"
    body = result.text if full else " ".join(result.text.split())[:240]
    return [f"{result.rank}. score={result.score:.4f} | {result.chunk_id}", f"   {result.citation}",
            f"   {flags} | last_updated={result.last_updated}", f"   {body}"]


def format_hybrid(result: HybridResult, full: bool) -> list[str]:
    chunk = result.chunk
    if result.match_type == "none":
        exact = "exact=none"
    else:
        exact = f"exact={result.match_type} {len(result.matched_codes)}/{len(result.query_codes)} [{', '.join(result.matched_codes)}]"
    flags = f"status={chunk.status} | authoritative={'yes' if chunk.authoritative else 'no'}"
    body = chunk.text if full else " ".join(chunk.text.split())[:240]
    return [f"{result.rank}. {exact} | semantic={result.semantic_score:.4f} (semantic rank {result.semantic_rank}) | {chunk.chunk_id}",
            f"   {chunk.citation}", f"   {flags} | last_updated={chunk.last_updated}", f"   {body}"]


def format_item(item: EvidenceItem, full: bool) -> list[str]:
    chunk = item.hybrid.chunk
    if item.hybrid.matched_codes:
        exact = f"documents {', '.join(item.hybrid.matched_codes)}"
    else:
        exact = "no query code"
    dept = (f" | REQUIRES DEPARTMENT VERIFICATION: {', '.join(item.applicable_departments)}"
            if item.requires_department_verification else "")
    body = chunk.text if full else " ".join(chunk.text.split())[:240]
    return [f"{item.policy_rank}. [{item.tier}] semantic={item.hybrid.semantic_score:.4f} | {exact} | {chunk.chunk_id}",
            f"   {chunk.citation}",
            f"   status={chunk.status} | authoritative={'yes' if chunk.authoritative else 'no'} | "
            f"last_updated={chunk.last_updated}{dept}", f"   {body}"]


def format_policy(response: PolicyResponse, args: argparse.Namespace) -> list[str]:
    s = response.sufficiency
    lines = [f"sufficiency: {s.level} (relevance floor {s.min_semantic_score}; best eligible score "
             f"{'n/a' if s.best_eligible_score is None else format(s.best_eligible_score, '.4f')}; "
             f"truncated by top-k: {s.truncated_by_top_k}; eligible below floor: {s.eligible_below_relevance})"]
    if response.query_codes:
        lines.append("code coverage: " + ", ".join(f"{c}={v}" for c, v in response.code_coverage.items())
                     + " (documented = present in eligible evidence; not a confirmed fix)")
    dept = response.applicability
    lines.append(f"department: {dept.department or 'unknown (not inferred)'}"
                 + (f" | department verification required for: {', '.join(dept.unresolved_documents)}"
                    if dept.requires_department_verification else ""))
    lines.append("note: scope and same-point SP-vs-KB conflicts are NOT resolved here; the downstream verifier must do that")
    for item in (*response.actionable, *response.conditional):
        lines.extend(format_item(item, args.full))
    if response.context_only:
        lines.append("context-only (current but NOT authoritative; background, never instructions):")
        for item in response.context_only:
            lines.extend(format_item(item, args.full))
    if response.governing_policy_chunks:
        lines.append(f"governing security-policy chunks (SP overrides KB on the same point): "
                     f"{', '.join(response.governing_policy_chunks)}")
    for note in response.conflict_annotations:
        lines.append(f"known conflict {note.id} (diagnostic only): {note.topic} | present: "
                     f"{', '.join(note.present_documents)} | manifest: {note.resolution}")
    if response.annotation_warning:
        lines.append(f"warning: {response.annotation_warning}")
    if args.diagnostics:
        lines.append(f"excluded relevant-scoring chunks ({len(response.diagnostics)}), never usable as instructions:")
        for d in response.diagnostics:
            lines.append(f"  - {d.chunk_id} | {', '.join(d.reasons)} | semantic={d.semantic_score:.4f} | {d.citation}")
            if d.untrusted_diagnostic_text is not None:
                lines.append("    UNTRUSTED diagnostic text: " + " ".join(d.untrusted_diagnostic_text.split())[:240])
    elif response.diagnostics:
        lines.append(f"{len(response.diagnostics)} excluded relevant-scoring chunk(s); use --diagnostics to list them")
    return lines


def main(argv: Sequence[str] | None = None, *, provider: EmbeddingProvider | None = None,
         out: Callable[[str], None] = print) -> int:
    args = build_parser().parse_args(argv)
    db_path = args.db or DEFAULT_DB[args.provider]
    try:
        if provider is None:
            if args.provider == "fake":
                provider = FakeEmbeddings()
            else:
                model = args.model or pinned_model(db_path)
                provider = OpenRouterEmbeddings.from_env(model)
        if args.mode == "policy":
            retriever = PolicyRetriever(db_path, provider, min_semantic_score=args.min_score,
                                        manifest_path=DEFAULT_MANIFEST if args.annotate_conflicts else None)
            policy = retriever.search(args.query, args.top_k, department=args.department,
                                      include_diagnostic_text=args.diagnostics)
        elif args.mode == "hybrid":
            retriever = HybridRetriever(db_path, provider)
            response = retriever.search(args.query, args.top_k)
        else:
            retriever = SemanticRetriever(db_path, provider)
            results = retriever.search(args.query, args.top_k)
    except (QueryError, RetrievalError) as exc:
        out(f"error: {exc}")
        return 2
    except EmbeddingError as exc:
        out(f"embedding error: {exc}")
        return 1 if not exc.fatal or "OPENROUTER_API_KEY" not in str(exc) else 2
    if args.mode == "policy":
        out(f"{len(policy.actionable)} actionable, {len(policy.conditional)} conditional, {len(policy.context_only)} "
            f"context-only from {len(retriever)} indexed chunks ({provider.model}, {retriever.dimensions} dims, mode=policy)")
    else:
        count = len(response.results) if args.mode == "hybrid" else len(results)
        out(f"{count} result(s) from {len(retriever)} indexed chunks ({provider.model}, {retriever.dimensions} dims, "
            f"mode={args.mode})")
    if retriever.skipped_invalid:
        out(f"warning: {len(retriever.skipped_invalid)} invalid stored vector(s) were skipped; run --verify")
    if args.mode == "policy":
        for line in format_policy(policy, args):
            out(line)
    elif args.mode == "hybrid":
        if response.query_codes:
            out(f"error codes in query: {', '.join(response.query_codes)}")
        if response.unmatched_codes:
            out(f"warning: no chunk contains: {', '.join(response.unmatched_codes)}")
        for result in response.results:
            for line in format_hybrid(result, args.full):
                out(line)
    else:
        for result in results:
            for line in format_result(result, args.full):
                out(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
