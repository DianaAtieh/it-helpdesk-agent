"""Fixtures for the app tests. Run from the kit folder: python -m pytest tests -v

The tests need the helpdesk API running (uvicorn services.helpdesk_api:app --port 8102) and are
skipped otherwise. They call the API's test-only /admin/reset, so tickets you created by hand on
that API are cleared.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app, sessions
from tests.helpers import API


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:          # starts the app's lifespan once, so the MCP server runs
        yield c


@pytest.fixture
def clean():
    httpx.post(f"{API}/admin/reset", timeout=5).raise_for_status()
    sessions.clear()
    yield
    httpx.delete(f"{API}/admin/faults", timeout=5)
