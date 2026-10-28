"""Offline tests for the MCP boundary: path-argument validation, the read/write allowlist and .env loading.

No API, no MCP subprocess and no model are needed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.mcp_client import APPROVED_WRITE_TOOLS, READ_TOOLS, HelpdeskMCP
from services import helpdesk_mcp as server

GOOD_EMAIL = "sara.mikhail@nakhla.example"


@pytest.fixture
def api_calls(monkeypatch) -> list[tuple]:
    """Replace the HTTP layer of the MCP server; records (method, path) of every request it would make."""
    seen: list[tuple] = []

    async def fake_call(method, path, **kwargs):
        seen.append((method, path))
        return True, {"ok": True}

    monkeypatch.setattr(server, "_call", fake_call)
    return seen


# ------------------------------------------------------------- B1: path arguments

BAD_EMAILS = ["../tickets?limit=50", "../../admin/tickets", "x@nakhla.example/../../tickets", "a%2f..%2f..%2fadmin",
              "sara.mikhail@nakhla.example/../x", "sara@other.example", "", "  ", "sara mikhail@nakhla.example",
              "..\\admin@nakhla.example", "a/b@nakhla.example"]
BAD_TICKET_IDS = ["../../admin/log", "IT-1033/../../admin", "IT-1033/notes", "IT-12", "1033", "",
                  "it-%2e%2e", "IT-1033?x=1", "IT-10333"]


@pytest.mark.parametrize("email", BAD_EMAILS)
def test_lookup_employee_rejects_malformed_emails_without_an_http_request(api_calls, email):
    result = asyncio.run(server.lookup_employee(email))
    assert result["ok"] is False and result["error"]["code"] == "invalid_argument"
    assert api_calls == []


@pytest.mark.parametrize("ticket_id", BAD_TICKET_IDS)
def test_ticket_tools_reject_malformed_ids_without_an_http_request(api_calls, ticket_id):
    for result in (asyncio.run(server.get_ticket(ticket_id)), asyncio.run(server.add_ticket_note(ticket_id, "hello"))):
        assert result["ok"] is False and result["error"]["code"] == "invalid_argument"
    assert api_calls == []


def test_valid_arguments_still_reach_the_api_on_the_expected_paths(api_calls):
    asyncio.run(server.lookup_employee(f"  {GOOD_EMAIL.upper()} "))
    asyncio.run(server.get_ticket("IT-1033"))
    asyncio.run(server.add_ticket_note(" it-2001 ", "note"))
    assert api_calls == [("GET", f"/api/directory/{GOOD_EMAIL}"), ("GET", "/api/tickets/IT-1033"),
                         ("POST", "/api/tickets/IT-2001/notes")]


def test_path_segments_are_url_encoded_as_defence_in_depth():
    assert server._path_segment("a/../b%2f") == "a%2F..%2Fb%252f"
    assert server._path_segment(GOOD_EMAIL) == GOOD_EMAIL


# ------------------------------------------------------------- B2: read/write allowlist

class RecordingSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call_tool(self, tool, args, read_timeout_seconds=None):
        self.calls.append(tool)
        return SimpleNamespace(structuredContent={"ok": True, "ticket": {"ticket_id": "IT-2001"}}, content=[])


def started_client() -> tuple[HelpdeskMCP, RecordingSession]:
    client, session = HelpdeskMCP(), RecordingSession()
    client._session = session
    return client, session


def test_the_allowlist_is_exactly_the_three_read_tools_and_create_ticket():
    assert READ_TOOLS == {"lookup_employee", "get_service_status", "search_tickets"}
    assert APPROVED_WRITE_TOOLS == {"create_ticket"}


@pytest.mark.parametrize("tool", sorted(READ_TOOLS))
def test_read_tools_work_without_approval(tool):
    client, session = started_client()
    data, call = asyncio.run(client.call(tool, {}))
    assert data["ok"] is True and call.ok and session.calls == [tool]


@pytest.mark.parametrize("tool", ["create_ticket", "add_ticket_note", "get_ticket", "delete_ticket", "", "CREATE_TICKET"])
def test_unapproved_calls_to_write_or_unknown_tools_never_reach_the_server(tool):
    client, session = started_client()
    data, call = asyncio.run(client.call(tool, {"subject": "x"}))
    assert data["ok"] is False and data["error"]["code"] == "tool_not_permitted" and not call.ok
    assert session.calls == []


@pytest.mark.parametrize("tool", ["add_ticket_note", "get_ticket", "lookup_employee_x", "search_tickets "])
def test_approval_does_not_unlock_other_tools(tool):
    client, session = started_client()
    data, _ = asyncio.run(client.call(tool, {}, approved=True))
    assert data["error"]["code"] == "tool_not_permitted" and session.calls == []


def test_approved_ticket_creation_still_works():
    client, session = started_client()
    data, call = asyncio.run(client.call("create_ticket", {"subject": "x"}, approved=True))
    assert data["ok"] is True and data["ticket"]["ticket_id"] == "IT-2001" and call.ok
    assert session.calls == ["create_ticket"]


def test_refusal_is_decided_before_the_session_state():
    data, _ = asyncio.run(HelpdeskMCP().call("create_ticket", {}))              # not started, not approved
    assert data["error"]["code"] == "tool_not_permitted"


def test_the_approve_endpoint_is_the_only_caller_that_passes_approved():
    from pathlib import Path
    app_dir = Path(__file__).resolve().parent.parent / "app"
    users = [p.name for p in app_dir.rglob("*.py") if "approved=True" in p.read_text(encoding="utf-8")
             and p.name != "mcp_client.py"]
    assert users == ["main.py"]


# ------------------------------------------------------------- B4: .env loading

def test_dotenv_loading_is_limited_to_helpdesk_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=sk-or-secret\nAPP_ADMIN_TOKEN=admin-secret\nAWS_SECRET_ACCESS_KEY=aws\n"
                   "# comment\nHELPDESK_API_URL=http://127.0.0.1:9999\nHELPDESK_API_KEY = \"k-from-env\"\n",
                   encoding="utf-8")
    for name in ("OPENROUTER_API_KEY", "APP_ADMIN_TOKEN", "AWS_SECRET_ACCESS_KEY", "HELPDESK_API_URL",
                 "HELPDESK_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    server._load_dotenv(env)

    import os
    assert os.environ["HELPDESK_API_URL"] == "http://127.0.0.1:9999" and os.environ["HELPDESK_API_KEY"] == "k-from-env"
    assert not {"OPENROUTER_API_KEY", "APP_ADMIN_TOKEN", "AWS_SECRET_ACCESS_KEY"} & set(os.environ)


def test_dotenv_does_not_override_values_already_set(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HELPDESK_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("HELPDESK_API_KEY", "from-process")
    server._load_dotenv(env)
    import os
    assert os.environ["HELPDESK_API_KEY"] == "from-process"


def test_a_missing_dotenv_is_fine(tmp_path):
    server._load_dotenv(tmp_path / "nope.env")
