"""Step 3 of the AI Verifier: the app turns it on with ANSWER_VERIFICATION (on unless set to false).

Runs the app's real startup (lifespan) with a recording build_graph, a no-op MCP client, no knowledge base and no
chat model: nothing is called, nothing leaves the machine. The app's state is restored afterwards, so this never
disturbs the shared test client.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

import app.main as app_main
from app.main import answer_verification_enabled


class IdleMCP:
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def start_app(monkeypatch, tmp_path, value: str | None) -> dict:
    """Run the lifespan once; return the keyword arguments the app gave build_graph."""
    if value is None:
        monkeypatch.delenv("ANSWER_VERIFICATION", raising=False)
    else:
        monkeypatch.setenv("ANSWER_VERIFICATION", value)
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(app_main, "mcp", IdleMCP())
    monkeypatch.setattr(app_main, "load_knowledge_base", lambda: None)
    monkeypatch.setattr(app_main.OpenRouterChat, "from_env", classmethod(lambda cls: None))
    seen: dict = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return "graph"

    monkeypatch.setattr(app_main, "build_graph", spy)
    saved = {k: getattr(app_main.app.state, k, None) for k in ("graph", "checkpointer", "preferences", "kb", "llm")}

    async def run():
        async with app_main.lifespan(app_main.app):
            pass

    try:
        asyncio.run(run())
    finally:
        for key, value_ in saved.items():
            setattr(app_main.app.state, key, value_)
    return seen


@pytest.mark.parametrize("value", [None, "", "true", "1", "yes", "on", " TRUE "])
def test_answer_verification_is_on_by_default_and_when_set(value, monkeypatch):
    if value is None:
        monkeypatch.delenv("ANSWER_VERIFICATION", raising=False)
    else:
        monkeypatch.setenv("ANSWER_VERIFICATION", value)
    assert answer_verification_enabled() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", " False "])
def test_answer_verification_can_be_turned_off(value, monkeypatch):
    monkeypatch.setenv("ANSWER_VERIFICATION", value)
    assert answer_verification_enabled() is False


def test_the_app_builds_its_graph_with_the_verifier_on_by_default(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="helpdesk-app")
    assert start_app(monkeypatch, tmp_path, None)["verify_answers"] is True
    assert "answer verification: on" in caplog.text


def test_the_app_builds_its_graph_without_the_verifier_when_turned_off(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="helpdesk-app")
    assert start_app(monkeypatch, tmp_path, "false")["verify_answers"] is False
    assert "answer verification: off" in caplog.text


def test_the_example_env_documents_the_setting_as_on():
    from pathlib import Path
    text = (Path(app_main.__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    assert "\nANSWER_VERIFICATION=true\n" in text
