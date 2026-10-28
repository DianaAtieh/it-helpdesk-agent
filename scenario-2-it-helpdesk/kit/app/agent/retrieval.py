"""Knowledge-base search for the agent, on top of ingestion.policy_retrieval.PolicyRetriever (used as is).

  * The retriever is built once (load_knowledge_base) and searched in a worker thread: search() is
    synchronous (one embedding call plus numpy work on the in-memory index; it never touches SQLite).
  * Only PolicyResponse.for_answering() is used: actionable and conditional evidence, plus the department
    and governing-policy rule lanes. Context-only, deprecated, draft and superseded text never gets here,
    so it can never be cited.
  * The result is plain dicts and strings: checkpoint-safe, and easy to fake in tests (tests/fakes.py).
  * Evidence text is untrusted data. The answer node must present it as data, never as instructions.

Unknown error codes (CRM-1150, sample case S2-04): the retriever only recognises codes whose prefix exists
in the index, so a code like CRM-1150 is silently dropped and the evidence looks "sufficient" on unrelated
text. Here every strict code (UPPERCASE letters, a hyphen, digits) in the question is also checked against the
evidence text: a code that no evidence mentions is reported as undocumented. ingestion/ is not changed.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ingestion.build_index import DEFAULT_DB
from ingestion.embeddings import DEFAULT_BASE_URL, EmbeddingError, OpenRouterEmbeddings, RetryPolicy
from ingestion.error_codes import codes_in_text
from ingestion.policy_retrieval import PolicyRetriever
from ingestion.retrieval import MAX_QUERY_CHARS, QueryError, RetrievalError
from ingestion.search import pinned_model

TOP_K = 5
DEFAULT_INDEX = Path(DEFAULT_DB["openrouter"])

# Live chat waits for the query embedding while the employee waits for the reply (and the session is locked),
# so it gives up sooner than index building: 2 attempts of at most 10 s each, instead of ingestion's defaults
# (4 attempts, 30 s), which are tuned for long offline batch runs. Index building is unchanged.
LIVE_EMBED_ATTEMPTS = 2
LIVE_EMBED_TIMEOUT_S = 10.0

# Only the unambiguous form: "CRM-1150", not "crm 1150" or "Win 11". The retriever's own, more lenient
# matching still applies to codes whose prefix is in the index.
_STRICT_CODE = re.compile(r"(?<![A-Za-z0-9_-])[A-Z]{2,5}-\d{2,5}(?![A-Za-z0-9_]|-\d)", re.ASCII)


class KnowledgeBaseError(RuntimeError):
    """The knowledge base could not be searched.

    code: unavailable (no index, no key, bad index) | search_failed (the embedding call failed)
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


@dataclass(frozen=True)
class KBResult:
    evidence: tuple[dict, ...]          # citeable chunks only (see evidence_record)
    codes: tuple[str, ...]              # every error code found in the question
    undocumented_codes: tuple[str, ...]  # codes that no evidence mentions
    sufficiency: str                    # sufficient | conditional_only | context_only | none (the retriever's)
    department: str | None              # the department the search was scoped to
    requires_department_verification: bool
    rule_warnings: tuple[str, ...]
    ms: int

    @property
    def doc_ids(self) -> frozenset[str]:
        """The documents an answer may cite."""
        return frozenset(e["doc_id"] for e in self.evidence)

    def to_state(self) -> dict:
        """Plain values for the graph state (checkpoint-safe)."""
        return {"evidence": [dict(e) for e in self.evidence], "codes": list(self.codes),
                "undocumented_codes": list(self.undocumented_codes), "sufficiency": self.sufficiency,
                "department": self.department,
                "requires_department_verification": self.requires_department_verification,
                "rule_warnings": list(self.rule_warnings)}


class KnowledgeBase(Protocol):
    """What the agent needs from the knowledge base. tests/fakes.py has a fake with the same shape."""

    async def search(self, query: str, department: str | None = None) -> KBResult: ...


def strict_codes(text: str) -> tuple[str, ...]:
    """Error codes written as UPPERCASE-digits, de-duplicated, in order of first appearance."""
    return tuple(dict.fromkeys(_STRICT_CODE.findall(text or "")))


def evidence_record(item, source: str) -> dict:
    """One citeable chunk as a plain dict. `source` is "evidence" or the name of a rule lane."""
    return {
        "doc_id": item.doc_id, "chunk_id": item.chunk_id, "title": item.title, "citation": item.citation,
        "text": item.text, "status": item.status, "authoritative": bool(item.authoritative), "source": source,
        "requires_department_verification": bool(item.requires_department_verification),
        "applicable_departments": list(getattr(item, "applicable_departments", ()) or ()),
    }


def build_result(response, query: str, ms: int) -> KBResult:
    """Turn a PolicyResponse into a KBResult. Pure: no I/O, so it is tested with real retriever output types."""
    view = response.for_answering()
    evidence: dict[str, dict] = {}
    for item in view.evidence:
        evidence.setdefault(item.chunk_id, evidence_record(item, "evidence"))
    # Rule lanes add the department exceptions and governing policies. Their text is None when the chunk is
    # already in the main evidence. Defence in depth: only current, authoritative chunks.
    for item in (*view.department_rules, *view.governing_policy):
        if item.text is not None and item.status == "current" and item.authoritative:
            evidence.setdefault(item.chunk_id, evidence_record(item, item.lane))
    records = tuple(evidence.values())

    codes = tuple(dict.fromkeys((*view.query_codes, *strict_codes(query))))
    texts = [e["text"] for e in records]
    undocumented = tuple(code for code in codes
                         if code in view.undocumented_codes
                         or not any(codes_in_text(text, (code,)) for text in texts))
    return KBResult(records, codes, undocumented, view.sufficiency.level, view.department,
                    bool(view.requires_department_verification), tuple(view.rule_warnings), ms)


class PolicyKnowledgeBase:
    """The real knowledge base: ingestion's PolicyRetriever over ingestion/index.db."""

    def __init__(self, retriever: PolicyRetriever, top_k: int = TOP_K) -> None:
        self._retriever = retriever
        self.top_k = top_k

    async def search(self, query: str, department: str | None = None) -> KBResult:
        query = (query or "").strip()[:MAX_QUERY_CHARS]       # messages may be longer than a query may be
        department = (department or "").strip() or None
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(self._retriever.search, query, self.top_k, department=department)
        except QueryError as exc:
            raise KnowledgeBaseError("search_failed", str(exc)) from None
        except (EmbeddingError, RetrievalError) as exc:
            raise KnowledgeBaseError("search_failed", exc.__class__.__name__) from None
        return build_result(response, query, round((time.perf_counter() - start) * 1000))


def live_embeddings(model: str | None) -> OpenRouterEmbeddings:
    """The query-embedding client for live chat: ingestion's OpenRouterEmbeddings with the live limits above.

    Raises KnowledgeBaseError("unavailable") without OPENROUTER_API_KEY. Builds the client; sends nothing.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise KnowledgeBaseError("unavailable", "OPENROUTER_API_KEY is not set")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL,
                    max_retries=0, timeout=LIVE_EMBED_TIMEOUT_S)     # retries live in RetryPolicy only
    return OpenRouterEmbeddings.from_env(model, retry=RetryPolicy(max_attempts=LIVE_EMBED_ATTEMPTS), client=client)


def load_knowledge_base(db_path: Path | str = DEFAULT_INDEX) -> PolicyKnowledgeBase:
    """Build the real knowledge base once (at app startup). Raises KnowledgeBaseError("unavailable")."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise KnowledgeBaseError("unavailable", f"no index at {db_path}: run python -m ingestion.build_index")
    try:
        provider = live_embeddings(pinned_model(db_path))
        return PolicyKnowledgeBase(PolicyRetriever(db_path, provider))
    except (EmbeddingError, RetrievalError, QueryError) as exc:
        raise KnowledgeBaseError("unavailable", str(exc)) from None
