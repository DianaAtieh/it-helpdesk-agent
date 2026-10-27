"""Offline tests for ingestion.embeddings. No network, no API key.

    .venv\\Scripts\\python.exe -m unittest discover -s ingestion/tests -t . -v

The optional online smoke test runs only with RUN_ONLINE_EMBEDDING_TEST=1 and a key in the environment
or the kit .env; it embeds one short string (a fraction of a cent).
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ingestion.embeddings import (DEFAULT_MODEL, EmbeddingError, FakeEmbeddings, OpenRouterEmbeddings, RetryPolicy,
                                  normalize_vector, resolve_model, validate_vectors)

SECRET = "sk-or-test-SECRET-123"


class HTTPFailure(Exception):
    def __init__(self, status: int, retry_after: str | None = None) -> None:
        super().__init__(f"boom with {SECRET} in the message")
        self.status_code = status
        self.response = SimpleNamespace(headers={"retry-after": retry_after} if retry_after else {})


class APIConnectionError(Exception):  # same class name the SDK uses for network failures
    pass


class StubClient:
    """Plays back a script of responses; an Exception entry is raised instead of returned."""

    def __init__(self, script, dims: int = 4) -> None:
        self.script, self.dims, self.calls = list(script), dims, []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0) if self.script else "ok"
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(kwargs)
        return self.ok(kwargs["input"])

    def ok(self, inputs, reverse: bool = False):
        items = [SimpleNamespace(index=i, embedding=[float(i + 1)] * self.dims) for i in range(len(inputs))]
        return SimpleNamespace(data=list(reversed(items)) if reverse else items)


def provider(client, **kwargs) -> tuple[OpenRouterEmbeddings, list[float]]:
    sleeps: list[float] = []
    kwargs.setdefault("expected_dimensions", client.dims)
    p = OpenRouterEmbeddings("key", "test/model", client=client, sleep=sleeps.append, jitter=lambda: 1.0, **kwargs)
    return p, sleeps


class ValidationTests(unittest.TestCase):
    def test_rejects_wrong_count_ragged_nan_and_wrong_dimensions(self):
        with self.assertRaises(EmbeddingError):
            validate_vectors([[1.0, 2.0]], 2, None)
        with self.assertRaises(EmbeddingError):
            validate_vectors([[1.0, 2.0], [1.0]], 2, None)
        with self.assertRaises(EmbeddingError):
            validate_vectors([[1.0, float("nan")]], 1, None)
        with self.assertRaises(EmbeddingError):
            validate_vectors([[float("inf"), 1.0]], 1, None)
        with self.assertRaises(EmbeddingError) as ctx:
            validate_vectors([[1.0, 2.0]], 1, 1536)
        self.assertTrue(ctx.exception.fatal)
        self.assertEqual(validate_vectors([[1, 2]], 1, 2), [[1.0, 2.0]])

    def test_normalize_gives_unit_little_endian_float32(self):
        v = normalize_vector([3.0, 4.0])
        self.assertEqual(v.dtype, np.dtype("<f4"))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)
        self.assertEqual(v.tolist(), np.asarray([0.6, 0.8], dtype="<f4").tolist())
        for bad in ([0.0, 0.0], [float("nan"), 1.0]):
            with self.assertRaises(EmbeddingError):
                normalize_vector(bad)


class FakeProviderTests(unittest.TestCase):
    def test_deterministic_sized_and_recorded(self):
        fake = FakeEmbeddings(dimensions=70)
        a, b = fake.embed(["one", "two"]), FakeEmbeddings(dimensions=70).embed(["one", "two"])
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 70)
        self.assertNotEqual(a[0], a[1])
        self.assertEqual(fake.calls, [["one", "two"]])
        self.assertEqual(fake.kind, "fake")
        self.assertNotEqual(FakeEmbeddings(model="other").embed(["one"]), FakeEmbeddings().embed(["one"]))


class OpenRouterProviderTests(unittest.TestCase):
    def test_request_payload_has_no_dimensions_override(self):
        client = StubClient([])
        p, _ = provider(client)
        p.embed(["a", "b"])
        self.assertEqual(client.calls, [{"model": "test/model", "input": ["a", "b"], "encoding_format": "float"}])

    def test_default_model_expects_1536_dimensions(self):
        p = OpenRouterEmbeddings("key", client=StubClient([]))
        self.assertEqual(p.model, DEFAULT_MODEL)
        self.assertEqual(p.expected_dimensions, 1536)
        with self.assertRaises(EmbeddingError) as ctx:
            p.embed(["a"])  # the stub returns 4-dimensional vectors
        self.assertTrue(ctx.exception.fatal)

    def test_batches_and_restores_order(self):
        client = StubClient([])
        p, _ = provider(client, batch_size=2)
        out = p.embed(["a", "b", "c", "d", "e"])
        self.assertEqual([len(c["input"]) for c in client.calls], [2, 2, 1])
        self.assertEqual(len(out), 5)
        reversed_client = StubClient([lambda kw: reversed_client.ok(kw["input"], reverse=True)])
        p2, _ = provider(reversed_client)
        self.assertEqual([v[0] for v in p2.embed(["a", "b", "c"])], [1.0, 2.0, 3.0])

    def test_empty_input_makes_no_call(self):
        client = StubClient([])
        p, _ = provider(client)
        self.assertEqual(p.embed([]), [])
        self.assertEqual(client.calls, [])

    def test_transient_errors_are_retried_with_backoff(self):
        client = StubClient([HTTPFailure(429), APIConnectionError("net"), HTTPFailure(503)])
        p, sleeps = provider(client, retry=RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=20.0))
        self.assertEqual(len(p.embed(["a"])), 1)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(sleeps, [1.0, 2.0, 4.0])

    def test_retry_after_header_is_honoured_and_capped(self):
        client = StubClient([HTTPFailure(429, "7"), HTTPFailure(429, "999")])
        p, sleeps = provider(client, retry=RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=20.0))
        p.embed(["a"])
        self.assertEqual(sleeps, [7.0, 20.0])

    def test_attempts_are_bounded(self):
        client = StubClient([HTTPFailure(500)] * 10)
        p, sleeps = provider(client, retry=RetryPolicy(max_attempts=3))
        with self.assertRaises(EmbeddingError) as ctx:
            p.embed(["a"])
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(len(sleeps), 2)
        self.assertFalse(ctx.exception.fatal)
        self.assertIn("HTTP 500", str(ctx.exception))

    def test_auth_and_client_errors_are_fatal_and_not_retried(self):
        for status in (400, 401, 403, 404):
            client = StubClient([HTTPFailure(status)] * 3)
            p, sleeps = provider(client)
            with self.assertRaises(EmbeddingError) as ctx:
                p.embed(["a"])
            self.assertTrue(ctx.exception.fatal, status)
            self.assertEqual((len(client.calls), sleeps), (1, []), status)

    def test_error_messages_never_contain_secrets(self):
        client = StubClient([HTTPFailure(401)])
        p, _ = provider(client)
        with self.assertRaises(EmbeddingError) as ctx:
            p.embed(["a"])
        self.assertNotIn(SECRET, str(ctx.exception))
        self.assertNotIn(SECRET, repr(ctx.exception))

    def test_bad_index_set_is_rejected(self):
        client = StubClient([lambda kw: SimpleNamespace(data=[SimpleNamespace(index=5, embedding=[1.0] * 4)])])
        p, _ = provider(client)
        with self.assertRaises(EmbeddingError):
            p.embed(["a"])

    def test_missing_key_is_a_fatal_configuration_error(self):
        with mock.patch("ingestion.embeddings._load_env", return_value=True), \
                mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EmbeddingError) as ctx:
                OpenRouterEmbeddings.from_env()
        self.assertTrue(ctx.exception.fatal)
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))

    def test_sdk_client_is_built_without_its_own_retries(self):
        with mock.patch("openai.OpenAI") as sdk:
            OpenRouterEmbeddings("key", base_url="https://example.invalid/v1")
        kwargs = sdk.call_args.kwargs
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(kwargs["base_url"], "https://example.invalid/v1")

    def test_model_resolution_order(self):
        with mock.patch("ingestion.embeddings._load_env", return_value=True), \
                mock.patch.dict(os.environ, {"EMBEDDING_MODEL": "env/model"}):
            self.assertEqual(resolve_model("arg/model"), "arg/model")
            self.assertEqual(resolve_model(), "env/model")
        with mock.patch("ingestion.embeddings._load_env", return_value=True), \
                mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_model(), DEFAULT_MODEL)


@unittest.skipUnless(os.getenv("RUN_ONLINE_EMBEDDING_TEST") == "1", "set RUN_ONLINE_EMBEDDING_TEST=1 to call the API")
class OnlineSmokeTest(unittest.TestCase):
    def test_real_embedding_has_expected_dimensions(self):
        p = OpenRouterEmbeddings.from_env(batch_size=1, retry=RetryPolicy(max_attempts=2))
        if not os.getenv("OPENROUTER_API_KEY"):
            self.skipTest("no OPENROUTER_API_KEY configured")
        (vector,) = p.embed(["VPN error NX-781 smoke test"])
        self.assertEqual(len(vector), p.expected_dimensions or len(vector))
        self.assertAlmostEqual(float(np.linalg.norm(normalize_vector(vector))), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
