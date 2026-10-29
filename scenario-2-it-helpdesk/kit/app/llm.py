"""The chat model, through OpenRouter with the openai library (pinned in requirements.txt).

Text in, one JSON object out, nothing else: the model is given NO tools, so it cannot write a ticket or
call any other tool. Ticket writes stay in /api/approve (app/main.py), behind the human approval gate and the
session token. Settings come from the kit's .env: MODEL (the chat model), OPENROUTER_API_KEY and
OPENROUTER_BASE_URL (shared with the embeddings in ingestion/).

Every failure is an LLMError with a short code, so the caller can give an honest reply instead of a guess.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_TOKENS = 800

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class LLMError(RuntimeError):
    """The chat model could not give a usable answer.

    code: not_configured | timeout | api_error | bad_output
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


@dataclass(frozen=True)
class LLMReply:
    data: dict                                   # the parsed JSON object
    usage: dict = field(default_factory=dict)    # {"prompt_tokens", "completion_tokens"} when the API reports them
    ms: int = 0


class ChatModel(Protocol):
    """What the agent needs from a chat model. tests/fakes.py has a fake with the same shape."""

    async def complete_json(self, system: str, user: str) -> LLMReply: ...


class StreamingChatModel(ChatModel, Protocol):
    """Optional extra: a model that can also generate plain text incrementally (see OpenRouterChat.stream_text)."""

    def stream_text(self, system: str, user: str, *, usage: dict | None = None) -> AsyncIterator[str]: ...


def parse_json_object(text: str | None) -> dict:
    """The model's answer as a dict. Accepts a ```json fence around it; anything else is bad_output."""
    if not text or not text.strip():
        raise LLMError("bad_output", "empty answer")
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except ValueError:
        raise LLMError("bad_output", "the answer is not JSON") from None
    if not isinstance(data, dict):
        raise LLMError("bad_output", "the answer is not a JSON object")
    return data


class OpenRouterChat:
    def __init__(self, api_key: str, model: str, base_url: str = DEFAULT_BASE_URL, *,
                 timeout: float = DEFAULT_TIMEOUT_S, max_retries: int = 1, max_tokens: int = DEFAULT_MAX_TOKENS,
                 client=None) -> None:
        if client is None:
            if not api_key or not model:
                raise LLMError("not_configured", "OPENROUTER_API_KEY and MODEL must both be set")
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    @classmethod
    def from_env(cls, **kwargs) -> "OpenRouterChat":
        """MODEL, OPENROUTER_API_KEY and OPENROUTER_BASE_URL (the app loads the kit's .env at startup)."""
        return cls(os.getenv("OPENROUTER_API_KEY", ""), os.getenv("MODEL", ""),
                   os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL, **kwargs)

    async def complete_json(self, system: str, user: str) -> LLMReply:
        import openai

        start = time.perf_counter()
        try:
            # No `tools` and no `functions`, on purpose: the model can only return text.
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=self.max_tokens,
            )
        except openai.APITimeoutError:
            raise LLMError("timeout", "the chat model did not answer in time") from None
        except openai.APIError as exc:
            raise LLMError("api_error", exc.__class__.__name__) from None
        ms = round((time.perf_counter() - start) * 1000)
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("bad_output", "no choices in the answer")
        data = parse_json_object(choices[0].message.content)
        usage = getattr(response, "usage", None)
        tokens = ({"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
                  if usage is not None else {})
        return LLMReply(data=data, usage=tokens, ms=ms)

    async def stream_text(self, system: str, user: str, *, usage: dict | None = None) -> AsyncIterator[str]:
        """The answer as plain text, yielded as the model generates it (stream=True). Opt-in: used only by the
        streaming answer path; complete_json is untouched. Same safety as complete_json: no tools, errors are
        LLMError. `usage`, if given, is filled in when the provider reports token counts (last chunk).
        Closing the iterator early (a cancelled turn) closes the HTTP stream."""
        import openai

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=self.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
        except openai.APITimeoutError:
            raise LLMError("timeout", "the chat model did not answer in time") from None
        except openai.APIError as exc:
            raise LLMError("api_error", exc.__class__.__name__) from None
        try:
            async for chunk in response:
                reported = getattr(chunk, "usage", None)
                if reported is not None and usage is not None:
                    usage.update(prompt_tokens=reported.prompt_tokens, completion_tokens=reported.completion_tokens)
                for choice in getattr(chunk, "choices", None) or []:
                    text = getattr(getattr(choice, "delta", None), "content", None)
                    if text:
                        yield text
        except openai.APITimeoutError:
            raise LLMError("timeout", "the chat model stopped answering") from None
        except openai.APIError as exc:
            raise LLMError("api_error", exc.__class__.__name__) from None
        finally:
            await response.close()
