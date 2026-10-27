"""Tests for app/llm.py: the chat model client (OpenRouter through the openai library). Offline: the openai
client is replaced by a fake, so no request leaves the machine. Run from the kit folder:

    python -m pytest tests/test_llm.py -v
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.llm import LLMError, OpenRouterChat, parse_json_object


class FakeOpenAI:
    """Stands in for openai.AsyncOpenAI: records the create() arguments, returns or raises what it is given."""

    def __init__(self, content: str | None = '{"reply": "hi", "citations": []}', error: Exception | None = None,
                 usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30)) -> None:
        self.kwargs: dict = {}
        outer = self

        class Completions:
            async def create(self, **kwargs):
                outer.kwargs = kwargs
                if error is not None:
                    raise error
                message = SimpleNamespace(content=content)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

        self.chat = SimpleNamespace(completions=Completions())


def ask(client: FakeOpenAI):
    return asyncio.run(OpenRouterChat("key", "openai/gpt-4.1-mini", client=client).complete_json("sys", "usr"))


REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


# ------------------------------------------------------------- what is sent

def test_the_model_gets_no_tools_only_text_in_and_json_out():
    client = FakeOpenAI()
    ask(client)
    assert "tools" not in client.kwargs and "functions" not in client.kwargs and "tool_choice" not in client.kwargs
    assert client.kwargs["response_format"] == {"type": "json_object"}
    assert client.kwargs["temperature"] == 0 and client.kwargs["model"] == "openai/gpt-4.1-mini"
    assert client.kwargs["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]


def test_the_answer_and_the_token_usage_come_back():
    reply = ask(FakeOpenAI())
    assert reply.data == {"reply": "hi", "citations": []}
    assert reply.usage == {"prompt_tokens": 120, "completion_tokens": 30}
    assert isinstance(reply.ms, int)


def test_missing_usage_is_an_empty_dict():
    assert ask(FakeOpenAI(usage=None)).usage == {}


# ------------------------------------------------------------- parsing

def test_a_json_fence_around_the_answer_is_accepted():
    assert parse_json_object('```json\n{"reply": "x"}\n```') == {"reply": "x"}


@pytest.mark.parametrize("content", [None, "", "   ", "not json", "[1, 2]", '"just a string"'])
def test_anything_but_a_json_object_is_bad_output(content):
    with pytest.raises(LLMError) as err:
        ask(FakeOpenAI(content=content))
    assert err.value.code == "bad_output"


def test_no_choices_is_bad_output():
    client = FakeOpenAI()

    async def empty(**kwargs):
        return SimpleNamespace(choices=[], usage=None)

    client.chat.completions.create = empty
    with pytest.raises(LLMError) as err:
        ask(client)
    assert err.value.code == "bad_output"


# ------------------------------------------------------------- failures are honest error codes

def test_a_timeout_is_reported_as_timeout():
    with pytest.raises(LLMError) as err:
        ask(FakeOpenAI(error=openai.APITimeoutError(request=REQUEST)))
    assert err.value.code == "timeout"


def test_a_connection_or_api_error_is_reported_as_api_error():
    with pytest.raises(LLMError) as err:
        ask(FakeOpenAI(error=openai.APIConnectionError(request=REQUEST)))
    assert err.value.code == "api_error"


@pytest.mark.parametrize("key, model", [("", "openai/gpt-4.1-mini"), ("key", ""), ("", "")])
def test_without_a_key_or_a_model_the_client_is_not_configured(key, model):
    with pytest.raises(LLMError) as err:
        OpenRouterChat(key, model)
    assert err.value.code == "not_configured"


def test_from_env_reads_model_key_and_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setenv("MODEL", "openai/gpt-4.1-mini")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.invalid/api/v1")
    chat = OpenRouterChat.from_env()                     # builds the client; sends nothing
    assert chat.model == "openai/gpt-4.1-mini"
    assert str(chat._client.base_url).startswith("https://example.invalid/api/v1")
    monkeypatch.delenv("MODEL")
    with pytest.raises(LLMError):
        OpenRouterChat.from_env()
