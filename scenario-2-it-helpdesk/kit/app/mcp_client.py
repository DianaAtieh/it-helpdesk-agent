"""Client for the kit's helpdesk MCP server (services/helpdesk_mcp.py), over stdio.

One MCP session is started together with our app and reused for every call, so
we do not pay the server's start-up time on each request. Every call is timed
and returned as a ToolCall for the ChatResponse's tool_calls.

The write tool create_ticket also goes through here, but only /api/approve may call it (approved=True),
never the LLM; every other tool outside READ_TOOLS is refused before reaching the server.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from app.schemas import ToolCall

KIT = Path(__file__).resolve().parent.parent
CALL_TIMEOUT = timedelta(seconds=25)   # the MCP server itself gives up on the API after 20 s

log = logging.getLogger("helpdesk-app.mcp")

# Authorization is decided here, in code. Graph nodes get the read tools only; the one write tool the app
# uses (create_ticket) also needs approved=True, which only /api/approve passes. Any other tool is refused.
READ_TOOLS = frozenset({"lookup_employee", "get_service_status", "search_tickets"})
APPROVED_WRITE_TOOLS = frozenset({"create_ticket"})
# The app's own MCP server for the employee's ticket history (app/tickets_mcp.py): one read tool, no writes.
MY_TICKETS_TOOLS = frozenset({"get_my_tickets"})


def _tool_json(result: Any) -> dict:
    """Read a tool result as a dict (structured content first, then the JSON text). Same as check_kit.py."""
    data = getattr(result, "structuredContent", None)
    if isinstance(data, dict):
        return data["result"] if set(data) == {"result"} and isinstance(data["result"], dict) else data
    for part in result.content or []:
        if getattr(part, "type", "") == "text":
            try:
                return json.loads(part.text)
            except ValueError:
                return {"ok": False, "error": {"code": "bad_tool_output", "message": part.text[:200]}}
    return {"ok": False, "error": {"code": "empty_tool_output", "message": "The tool returned nothing."}}


def server_env() -> dict[str, str]:
    """Least privilege: what the OS needs to run Python, plus the HELPDESK_* settings. Nothing else.

    Our other secrets (OPENROUTER_API_KEY, APP_ADMIN_TOKEN, cloud credentials) stay out of the
    subprocess. Note: when the kit's .env file exists, the MCP server also reads it by itself.
    """
    return {**get_default_environment(),
            **{key: value for key, value in os.environ.items() if key.startswith("HELPDESK_")}}


def tickets_server_env() -> dict[str, str]:
    """The ticket-history server needs only where its database is (TICKETS_DB), nothing else."""
    return {**get_default_environment(), **({"TICKETS_DB": os.environ["TICKETS_DB"]} if "TICKETS_DB" in os.environ
                                            else {})}


class HelpdeskMCP:
    def __init__(self, module: str = "services.helpdesk_mcp", read_tools: frozenset[str] = READ_TOOLS,
                 write_tools: frozenset[str] = APPROVED_WRITE_TOOLS, env=server_env) -> None:
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._module, self._read, self._write, self._env = module, read_tools, write_tools, env

    async def start(self) -> None:
        params = StdioServerParameters(command=sys.executable, args=["-m", self._module],
                                       cwd=str(KIT), env=self._env())
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        names = sorted(t.name for t in (await self._session.list_tools()).tools)
        log.info("MCP server %s started, tools: %s", self._module, ", ".join(names))

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def call(self, tool: str, args: dict | None = None, *, approved: bool = False) -> tuple[dict, ToolCall]:
        """Call one MCP tool. Never raises: failures come back as {"ok": False, "error": {...}}.

        Only READ_TOOLS are allowed by default; APPROVED_WRITE_TOOLS need approved=True. Refused calls
        never reach the MCP server.
        """
        start = time.perf_counter()
        if tool not in self._read and not (approved and tool in self._write):
            log.warning("MCP call refused: tool=%s approved=%s", tool, approved)
            data = {"ok": False, "error": {"code": "tool_not_permitted",
                                           "message": "This tool is not permitted from this code path."}}
        elif self._session is None:
            data = {"ok": False, "error": {"code": "mcp_not_started", "message": "The MCP server is not running."}}
        else:
            try:
                result = await self._session.call_tool(tool, args or {}, read_timeout_seconds=CALL_TIMEOUT)
                data = _tool_json(result)
            except Exception as exc:  # broken pipe, timeout, server crash
                log.exception("MCP call %s failed", tool)
                data = {"ok": False, "error": {"code": "mcp_error", "message": str(exc) or exc.__class__.__name__}}
        ms = round((time.perf_counter() - start) * 1000)
        ok = bool(data.get("ok"))
        log.info("tool=%s ok=%s ms=%d", tool, ok, ms)
        return data, ToolCall(tool=tool, ok=ok, ms=ms)


def my_tickets_mcp() -> HelpdeskMCP:
    """The client for app/tickets_mcp.py: get_my_tickets only, never a write."""
    return HelpdeskMCP("app.tickets_mcp", read_tools=MY_TICKETS_TOOLS, write_tools=frozenset(),
                       env=tickets_server_env)
