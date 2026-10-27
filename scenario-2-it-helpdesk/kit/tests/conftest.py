"""Fixtures for the app tests. Run from the kit folder: python -m pytest tests -v

The tests need the helpdesk API running (uvicorn services.helpdesk_api:app --port 8102) and are
skipped otherwise. They call the API's test-only /admin/reset, so tickets you created by hand on
that API are cleared. They never touch the real checkpoint database (app/data/checkpoints.db).
"""
from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app, delete_all_threads, sessions
from tests.helpers import API


@pytest.fixture(scope="session", autouse=True)
def checkpoint_db(tmp_path_factory):
    """A throw-away checkpoint database for the whole test run. The app reads CHECKPOINT_DB at startup."""
    path = tmp_path_factory.mktemp("checkpoints") / "checkpoints.db"
    previous = os.environ.get("CHECKPOINT_DB")
    os.environ["CHECKPOINT_DB"] = str(path)
    yield path
    if previous is None:
        os.environ.pop("CHECKPOINT_DB", None)
    else:
        os.environ["CHECKPOINT_DB"] = previous


@pytest.fixture(scope="session")
def client(checkpoint_db):
    with TestClient(app) as c:          # starts the app's lifespan once: checkpointer + MCP server
        yield c


@pytest.fixture
def clean(client):
    """Every test starts fresh: helpdesk API seed data, no in-memory sessions, no saved conversations.

    Deleting the checkpoint threads matters: test files reuse session ids such as "s-1", and a thread
    left by an earlier test would carry its history and its locked user_email into the next one.
    """
    httpx.post(f"{API}/admin/reset", timeout=5).raise_for_status()
    sessions.clear()
    client.portal.call(delete_all_threads, app.state.checkpointer)
    yield
    httpx.delete(f"{API}/admin/faults", timeout=5)
