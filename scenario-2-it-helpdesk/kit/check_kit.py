"""Smoke test for the IT Helpdesk kit: dataset, helpdesk API, MCP server over stdio.

    uvicorn services.helpdesk_api:app --port 8102     # terminal 1 (from the kit folder)
    python check_kit.py                               # terminal 2

Prints one line per check and ends with "Kit ready." when everything works.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
TOOLS = ["add_ticket_note", "create_ticket", "get_service_status", "get_ticket", "lookup_employee", "search_tickets"]


def load_env() -> None:
    env = KIT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def ok(message: str) -> None:
    print(f"✅ {message}")


def fail(message: str, hint: str) -> None:
    print(f"❌ {message}\n   -> {hint}")
    sys.exit(1)


def tool_json(result) -> dict:
    """Read a tool result as a dict (structured content first, then the JSON text)."""
    data = getattr(result, "structuredContent", None)
    if isinstance(data, dict):
        return data["result"] if set(data) == {"result"} and isinstance(data["result"], dict) else data
    for part in result.content or []:
        if getattr(part, "type", "") == "text":
            try:
                return json.loads(part.text)
            except ValueError:
                return {"ok": False, "error": {"message": part.text[:200]}}
    return {}


async def check_mcp(url: str, key: str) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # Pass the environment on: by default the MCP SDK starts the server with a minimal env.
    env = {**os.environ, "HELPDESK_API_URL": url, "HELPDESK_API_KEY": key}
    params = StdioServerParameters(command=sys.executable, args=["-m", "services.helpdesk_mcp"], cwd=str(KIT), env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = sorted(t.name for t in (await session.list_tools()).tools)
            if names != TOOLS:
                fail(f"MCP server lists {names}", f"expected {TOOLS}. Did you edit services/helpdesk_mcp.py?")
            ok(f"MCP server over stdio lists {len(names)} tools: {', '.join(names)}")
            data = tool_json(await session.call_tool("lookup_employee", {"email": "tariq.hamdan@nakhla.example"}))
            if not data.get("ok"):
                fail(f"MCP read call failed: {data.get('error')}", "Check HELPDESK_API_URL and HELPDESK_API_KEY.")
            emp = data["employee"]
            ok(f"MCP read call works: lookup_employee -> {emp['name']} ({emp['department']}, "
               f"NakhlaConnect {emp['vpn_client_version']})")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env()
    url = (os.getenv("HELPDESK_API_URL") or "http://localhost:8102").rstrip("/")
    key = os.getenv("HELPDESK_API_KEY") or "dev-key-change-me"

    if sys.version_info < (3, 10):
        fail(f"Python {sys.version.split()[0]}", "Use Python 3.10 or newer (the kit is tested on 3.12).")
    try:
        import fastapi, httpx, jsonschema, mcp, pymupdf, yaml  # noqa: F401,E401
    except ImportError as exc:
        fail(f"Missing package: {exc.name}", "Activate your venv and run: pip install -r requirements.txt")
    ok(f"Python {sys.version.split()[0]} and the kit packages are installed")

    manifest = json.loads((KIT / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    files = [p for p in (KIT / "dataset" / "knowledge_base").iterdir() if p.suffix in (".md", ".pdf")]
    tickets = json.loads((KIT / "dataset" / "tickets" / "historical_tickets.json").read_text(encoding="utf-8"))["tickets"]
    if len(files) != manifest["counts"]["documents"] or len(tickets) != manifest["counts"]["tickets"]:
        fail("The dataset is incomplete", "Copy the whole kit/dataset folder, then run: python dataset/validate_dataset.py")
    ok(f"Dataset: {len(files)} KB documents ({sum(p.suffix == '.pdf' for p in files)} PDF), "
       f"{len(tickets)} historical tickets")

    import httpx
    try:
        httpx.get(f"{url}/health", timeout=5).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        fail(f"The helpdesk API is not answering at {url}/health ({exc.__class__.__name__})",
             "Start it from the kit folder: uvicorn services.helpdesk_api:app --port 8102")
    ok(f"Helpdesk API is up at {url}")
    r = httpx.get(f"{url}/api/status", headers={"X-API-Key": key}, timeout=10)
    if r.status_code == 401:
        fail("The API rejected the X-API-Key", "Use the same HELPDESK_API_KEY for the service, the MCP server and your app (.env).")
    if r.status_code != 200:
        fail(f"GET /api/status returned {r.status_code}", "Clear faults: curl.exe -X DELETE http://localhost:8102/admin/faults")
    active = r.json().get("active_incidents", [])
    ok(f"REST call with X-API-Key works (status page: {len(active)} active incident(s) {', '.join(active)})")

    try:
        asyncio.run(check_mcp(url, key))
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        fail(f"MCP over stdio failed: {exc!r}", "Run it alone to see the error: python -m services.helpdesk_mcp")
    print("✅ Kit ready.")


if __name__ == "__main__":
    main()
