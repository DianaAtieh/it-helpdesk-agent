"""Fake chat model and fake knowledge base for the app tests: deterministic, offline, free.

They have the same shape as app.llm.ChatModel and app.agent.retrieval.KnowledgeBase, and record every call so
a test can check what the agent sent (for example: that retrieved text went in as data).
"""
from __future__ import annotations

from app.agent.retrieval import KBResult, KnowledgeBaseError
from app.llm import LLMError, LLMReply


class FakeChat:
    """Returns the scripted JSON objects in order (the last one repeats); an Exception in the script is raised."""

    def __init__(self, *replies: dict | Exception, usage: dict | None = None) -> None:
        self.replies = list(replies) or [{"reply": "ok", "citations": []}]
        self.usage = usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 5}
        self.calls: list[dict] = []

    async def complete_json(self, system: str, user: str) -> LLMReply:
        self.calls.append({"system": system, "user": user})
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return LLMReply(data=dict(reply), usage=dict(self.usage), ms=1)


def evidence(doc_id: str, text: str, *, title: str = "", source: str = "evidence",
             chunk_id: str | None = None) -> dict:
    """One evidence record, shaped like app.agent.retrieval.evidence_record."""
    return {"doc_id": doc_id, "chunk_id": chunk_id or f"{doc_id}#1", "title": title or doc_id,
            "citation": f"{doc_id} > {title or doc_id}", "text": text, "status": "current", "authoritative": True,
            "source": source, "requires_department_verification": False, "applicable_departments": []}


def kb_result(*records: dict, codes: tuple[str, ...] = (), undocumented: tuple[str, ...] = (),
              sufficiency: str = "sufficient", department: str | None = None) -> KBResult:
    return KBResult(tuple(records), codes, undocumented, sufficiency if records else "none", department,
                    False, (), 1)


class FakeKnowledgeBase:
    """Returns the given KBResult (or raises the given error) and records every search."""

    def __init__(self, result: KBResult | Exception | None = None) -> None:
        self.result = result if result is not None else kb_result()
        self.searches: list[dict] = []

    async def search(self, query: str, department: str | None = None) -> KBResult:
        self.searches.append({"query": query, "department": department})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


__all__ = ["FakeChat", "FakeKnowledgeBase", "KnowledgeBaseError", "LLMError", "evidence", "kb_result"]
