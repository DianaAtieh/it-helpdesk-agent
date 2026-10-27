"""Client for the kit's helpdesk MCP server (services/helpdesk_mcp.py), over stdio.

One MCP session is started together with our app and reused for every call, so
we do not pay the server's start-up time on each request. Every call is timed
and returned as a ToolCall for the ChatResponse's tool_calls.

The write tools (create_ticket, add_ticket_note) also go through here, but only
/api/approve may call them, never the LLM.
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
from mcp.client.stdio import stdio_client

from app.schemas import ToolCall

KIT = Path(__file__).resolve().parent.parent
CALL_TIMEOUT = timedelta(seconds=25)   # the MCP server itself gives up on the API after 20 s

log = logging.getLogger("helpdesk-app.mcp")


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


class HelpdeskMCP:
    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        # Pass our environment on: by default the MCP SDK starts the server with a minimal env,
        # and the server needs HELPDESK_API_URL and HELPDESK_API_KEY.
        params = StdioServerParameters(command=sys.executable, args=["-m", "services.helpdesk_mcp"],
                                       cwd=str(KIT), env=dict(os.environ))
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        names = sorted(t.name for t in (await self._session.list_tools()).tools)
        log.info("MCP server started, tools: %s", ", ".join(names))

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def call(self, tool: str, args: dict | None = None) -> tuple[dict, ToolCall]:
        """Call one MCP tool. Never raises: failures come back as {"ok": False, "error": {...}}."""
        start = time.perf_counter()
        if self._session is None:
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
