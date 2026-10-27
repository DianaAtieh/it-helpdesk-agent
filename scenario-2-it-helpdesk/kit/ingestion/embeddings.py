"""Embedding providers for the offline ingestion pipeline (Phase 1C).

``OpenRouterEmbeddings`` calls the OpenAI-compatible OpenRouter ``/embeddings`` endpoint.
``FakeEmbeddings`` is a deterministic offline stand-in for tests and dry runs.

Retries live in one place only: the OpenAI SDK client is created with ``max_retries=0`` and
``RetryPolicy`` is the single bounded policy. API keys are read from the environment and never
logged, stored, or included in error messages.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

DEFAULT_MODEL = "openai/text-embedding-3-small"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
KNOWN_DIMENSIONS = {DEFAULT_MODEL: 1536}
KIT_DIR = Path(__file__).resolve().parents[1]


class EmbeddingError(RuntimeError):
    """An embedding request failed. ``fatal`` means retrying or continuing cannot help (bad key/model)."""

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


class EmbeddingProvider(Protocol):
    kind: str  # "openrouter" or "fake": real and fake vectors must never share an index
    model: str
    expected_dimensions: int | None

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff: max_attempts is the total number of tries, not extra retries."""
    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 20.0

    def delay(self, attempt: int, retry_after: float | None, jitter: Callable[[], float]) -> float:
        backoff = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
        if retry_after is not None:
            return min(self.max_delay, max(0.0, retry_after))
        return backoff * (0.5 + jitter() / 2)


def validate_vectors(vectors: Sequence[Sequence[float]], count: int, expected_dimensions: int | None) -> list[list[float]]:
    """Reject wrong counts, ragged or wrong-sized vectors, and non-finite values."""
    if len(vectors) != count:
        raise EmbeddingError(f"provider returned {len(vectors)} vectors for {count} inputs")
    result = [[float(x) for x in v] for v in vectors]
    dims = {len(v) for v in result}
    if len(dims) != 1 or 0 in dims:
        raise EmbeddingError(f"provider returned vectors of inconsistent sizes {sorted(dims)}")
    size = dims.pop()
    if expected_dimensions is not None and size != expected_dimensions:
        raise EmbeddingError(f"expected {expected_dimensions} dimensions but the provider returned {size}", fatal=True)
    if any(not math.isfinite(x) for v in result for x in v):
        raise EmbeddingError("provider returned NaN or infinite values")
    return result


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    """Unit-length little-endian float32, so cosine similarity is a plain dot product later."""
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm == 0.0:
        raise EmbeddingError("cannot normalise a zero or non-finite vector")
    return (array / norm).astype("<f4")


def _status_of(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        return float(headers.get("retry-after")) if headers else None
    except (TypeError, ValueError):
        return None


def _is_transient(exc: Exception) -> bool:
    status = _status_of(exc)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


def _load_env() -> bool:
    from dotenv import load_dotenv
    load_dotenv(KIT_DIR / ".env", override=False)  # never overrides variables already set
    return True


def resolve_model(model: str | None = None) -> str:
    """Explicit argument, then $EMBEDDING_MODEL (kit .env included), then the project default."""
    _load_env()
    return model or os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL


class OpenRouterEmbeddings:
    kind = "openrouter"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, *,
                 batch_size: int = 32, retry: RetryPolicy = RetryPolicy(), expected_dimensions: int | None = None,
                 client=None, sleep: Callable[[float], None] = time.sleep,
                 jitter: Callable[[], float] = random.random) -> None:
        if not api_key and client is None:
            raise EmbeddingError("OPENROUTER_API_KEY is not set", fatal=True)
        if batch_size < 1 or retry.max_attempts < 1:
            raise ValueError("batch_size and max_attempts must be at least 1")
        self.model = model
        self.batch_size = batch_size
        self.retry = retry
        self.expected_dimensions = expected_dimensions if expected_dimensions is not None else KNOWN_DIMENSIONS.get(model)
        self._sleep, self._jitter = sleep, jitter
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=30.0)
        self._client = client

    @classmethod
    def from_env(cls, model: str | None = None, **kwargs) -> "OpenRouterEmbeddings":
        _load_env()
        return cls(os.getenv("OPENROUTER_API_KEY", ""), resolve_model(model),
                   os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL, **kwargs)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(list(texts[i:i + self.batch_size])))
        return validate_vectors(vectors, len(texts), self.expected_dimensions) if texts else []

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                response = self._client.embeddings.create(model=self.model, input=batch, encoding_format="float")
                ordered = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in ordered] != list(range(len(batch))):
                    raise EmbeddingError("provider returned unexpected embedding indexes")
                return validate_vectors([item.embedding for item in ordered], len(batch), self.expected_dimensions)
            except EmbeddingError:
                raise
            except Exception as exc:  # classified below; the message never includes request details
                status = _status_of(exc)
                label = f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "")
                if not _is_transient(exc):
                    raise EmbeddingError(f"embedding request failed: {label}", fatal=True) from None
                if attempt == self.retry.max_attempts:
                    raise EmbeddingError(f"embedding request failed after {attempt} attempts: {label}") from None
                self._sleep(self.retry.delay(attempt, _retry_after(exc), self._jitter))
        raise AssertionError("unreachable")


class FakeEmbeddings:
    """Deterministic hash-based vectors. Same text and model name give the same vector, offline."""
    kind = "fake"

    def __init__(self, dimensions: int = 64, model: str = "fake-hash-embedding") -> None:
        self.model = model
        self.expected_dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        out: list[float] = []
        counter = 0
        while len(out) < self.expected_dimensions:
            digest = hashlib.sha256(f"{self.model}\x1f{counter}\x1f{text}".encode("utf-8")).digest()
            out.extend(b / 127.5 - 1.0 for b in digest)
            counter += 1
        return out[:self.expected_dimensions]
