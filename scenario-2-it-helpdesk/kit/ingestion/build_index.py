"""One-command offline ingestion: load -> chunk -> embed -> store in SQLite.

    .venv\\Scripts\\python.exe -m ingestion.build_index            # real embeddings via OpenRouter
    .venv\\Scripts\\python.exe -m ingestion.build_index --dry-run  # plan only, no API calls
    .venv\\Scripts\\python.exe -m ingestion.build_index --verify   # offline integrity check
    .venv\\Scripts\\python.exe -m ingestion.build_index --provider fake   # offline, separate index file

Exit codes: 0 success, 1 partial failure or verification problems, 2 configuration error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from ingestion.chunking import ChunkConfig
from ingestion.embeddings import (DEFAULT_MODEL, EmbeddingError, EmbeddingProvider, FakeEmbeddings,
                                  OpenRouterEmbeddings, RetryPolicy, resolve_model)
from ingestion.index_store import IndexMismatchError, IndexReport, IndexStore, index_documents, verify_index
from ingestion.loaders import DocumentLoadError, load_knowledge_base

INGESTION_DIR = Path(__file__).resolve().parent
DEFAULT_DB = {"openrouter": INGESTION_DIR / "index.db", "fake": INGESTION_DIR / "index.fake.db"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_index", description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", choices=sorted(DEFAULT_DB), default="openrouter")
    parser.add_argument("--model", help=f"embedding model (default: $EMBEDDING_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--db", type=Path, help="index file (default: ingestion/index.db, or index.fake.db for fake)")
    parser.add_argument("--prune", action="store_true", help="delete indexed documents that left the dataset")
    parser.add_argument("--dry-run", action="store_true", help="report what would be embedded; no API calls, no writes")
    parser.add_argument("--verify", action="store_true", help="check the existing index offline and exit")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=4, help="total tries per request (single retry policy)")
    return parser


def make_provider(args: argparse.Namespace) -> EmbeddingProvider:
    if args.provider == "fake":
        return FakeEmbeddings()
    return OpenRouterEmbeddings.from_env(args.model, batch_size=args.batch_size,
                                         retry=RetryPolicy(max_attempts=args.max_attempts))


def print_report(report: IndexReport, out: Callable[[str], None]) -> None:
    out(("DRY RUN - nothing written. " if report.dry_run else "") + (
        f"indexed {len(report.indexed)} | skipped (unchanged) {len(report.skipped)} | "
        f"failed {len(report.failed)} | not attempted {len(report.not_attempted)}"))
    out(f"chunks written {report.chunks_written} | old chunks removed {report.chunks_removed} | "
        f"texts sent to provider {report.texts_embedded} | vectors reused {report.vectors_reused}"
        + (f" | corrupt cached vectors ignored {report.vectors_rejected}" if report.vectors_rejected else ""))
    for doc_id, message in report.failed.items():
        out(f"  FAILED {doc_id}: {message}")
    if report.not_attempted:
        out("  NOT ATTEMPTED: " + ", ".join(report.not_attempted))
    if report.stale_documents:
        suffix = " (NOT pruned: indexing did not complete cleanly)" if report.prune_skipped else ""
        out("  in index but not in dataset (use --prune to remove)" + suffix + ": " + ", ".join(report.stale_documents))
    if report.pruned:
        out("  pruned: " + ", ".join(report.pruned))


def main(argv: Sequence[str] | None = None, *, provider: EmbeddingProvider | None = None,
         out: Callable[[str], None] = print) -> int:
    args = build_parser().parse_args(argv)
    offline_only = args.verify or args.dry_run  # these never need a key or the network
    try:
        docs = load_knowledge_base()
        if provider is None and not offline_only:
            provider = make_provider(args)
    except (DocumentLoadError, EmbeddingError) as exc:
        out(f"configuration error: {exc}")
        return 2
    if provider is not None:
        kind, model = provider.kind, provider.model
    else:
        kind = args.provider
        model = resolve_model(args.model) if kind == "openrouter" else FakeEmbeddings().model
    db_path = args.db or DEFAULT_DB[kind]
    if args.verify and not Path(db_path).exists():
        out(f"configuration error: no index at {db_path}")
        return 2
    if args.dry_run and not Path(db_path).exists():
        db_path = ":memory:"  # a dry run never creates files
    if provider is None:
        provider = FakeEmbeddings(model=model)  # placeholder: never called in verify or dry-run mode
    read_only = offline_only and db_path != ":memory:"  # dry-run and verify can never modify an existing index
    try:
        with IndexStore(db_path, kind, model, read_only=read_only) as store:
            if args.verify:
                problems = verify_index(store, docs)
                out(f"{db_path}: {store.chunk_count()} chunks, {len(store.document_ids())} documents, "
                    f"{len(problems)} problem(s)")
                for problem in problems:
                    out(f"  {problem}")
                return 1 if problems else 0
            report = index_documents(store, docs, provider, ChunkConfig(), prune=args.prune,
                                     dry_run=args.dry_run, log=out)
    except IndexMismatchError as exc:
        out(f"configuration error: {exc}")
        return 2
    print_report(report, out)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
