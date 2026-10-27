"""Tests for the SQLite checkpointer lifecycle (app/main.py lifespan). Run from the kit folder:

    python -m pytest tests -v

They do not need the helpdesk API: the MCP server starts without it. The database is the temporary
one from tests/conftest.py, never app/data/checkpoints.db.
"""
from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.main import DEFAULT_CHECKPOINT_DB, KIT, app, checkpoint_db_path


def test_the_checkpointer_is_open_and_set_up_while_the_app_runs(client):
    checkpointer = app.state.checkpointer
    assert isinstance(checkpointer, AsyncSqliteSaver)
    assert checkpointer.is_setup


def test_tests_use_the_temporary_database_not_the_real_one(client, checkpoint_db):
    async def database_file() -> str:
        async with app.state.checkpointer.conn.execute("PRAGMA database_list") as cursor:
            return (await cursor.fetchone())[2]

    opened = Path(client.portal.call(database_file)).resolve()
    assert opened == checkpoint_db.resolve()
    assert opened != DEFAULT_CHECKPOINT_DB.resolve()
    assert checkpoint_db.is_file()


def test_setup_created_the_checkpoint_tables(client):
    async def tables() -> set[str]:
        async with app.state.checkpointer.conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
            return {row[0] for row in await cur.fetchall()}

    assert {"checkpoints", "writes"} <= client.portal.call(tables)


def test_checkpoint_db_path_defaults_and_resolves_relative_paths_under_the_kit(monkeypatch):
    monkeypatch.delenv("CHECKPOINT_DB", raising=False)
    assert checkpoint_db_path() == DEFAULT_CHECKPOINT_DB

    monkeypatch.setenv("CHECKPOINT_DB", "data/other.db")
    assert checkpoint_db_path() == KIT / "data" / "other.db"

    absolute = (KIT / "somewhere" / "x.db").resolve()
    monkeypatch.setenv("CHECKPOINT_DB", str(absolute))
    assert checkpoint_db_path() == absolute
