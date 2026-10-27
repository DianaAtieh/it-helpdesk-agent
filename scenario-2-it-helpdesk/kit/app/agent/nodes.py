"""Graph nodes. Each node takes the state and returns only the fields it changes.

Nodes get their dependencies (the MCP client) from a factory, so tests can pass a fake client.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

from langchain_core.messages import AIMessage

from app.agent.state import AgentState


class MCPClient(Protocol):
    """What the nodes need from app.mcp_client.HelpdeskMCP: call(tool, args) -> (data, ToolCall)."""

    async def call(self, tool: str, args: dict | None = None) -> tuple[dict, Any]: ...


def describe_status(data: dict) -> str:
    """Turn a get_service_status result into a short reply."""
    if not data.get("ok"):
        code = (data.get("error") or {}).get("code", "unknown")
        return f"I could not check the status page right now ({code}). Please try again in a moment."
    status = data["status"]
    active = [i for i in status.get("incidents", []) if i.get("status") == "active"]
    if not active:
        return "The status page shows no active incidents right now."
    lines = ["The status page shows these active incidents:"]
    for inc in active:
        line = f"- {inc['id']} ({inc.get('priority', '?')}): {inc.get('title', '')}"
        if inc.get("workaround"):
            line += f". Workaround: {inc['workaround']}"
        lines.append(line)
    return "\n".join(lines)


def make_check_status(mcp: MCPClient) -> Callable[[AgentState], Awaitable[dict]]:
    """The walking-skeleton node: read the status page over MCP and reply with it."""

    async def check_status(state: AgentState) -> dict:
        data, call = await mcp.call("get_service_status")
        reply = describe_status(data)
        incidents = (data.get("status") or {}).get("incidents", []) if data.get("ok") else []
        active = next((i for i in incidents if i.get("status") == "active"), None)
        return {
            "messages": [AIMessage(content=reply)],
            "reply": reply,
            "active_incident": active,
            "tool_calls": [call.model_dump()],      # plain dict: checkpoint-safe
        }

    return check_status
