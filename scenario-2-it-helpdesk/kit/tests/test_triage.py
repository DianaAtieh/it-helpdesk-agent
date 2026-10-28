"""Tests for the triage node (app/agent/nodes.py, Phase 2): who is asking, what the status page says, which error
codes were given, and where to go next. No LLM, no network: a fake MCP client returns the simulated systems' own
records (services/data/status.json and employees.json). Run from the kit folder:

    python -m pytest tests/test_triage.py -v
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes import make_triage, route_after_triage
from app.schemas import ToolCall

DATA = Path(__file__).resolve().parent.parent / "services" / "data"
STATUS = json.loads((DATA / "status.json").read_text(encoding="utf-8"))
EMPLOYEES = {e["email"]: e for e in json.loads((DATA / "employees.json").read_text(encoding="utf-8"))["employees"]}
ENGINEER = next(e for e in EMPLOYEES.values() if e["department"] == "Engineering")
SARA = "sara.mikhail@nakhla.example"


class FakeMCP:
    """Answers lookup_employee and get_service_status like the real MCP server; records every call."""

    def __init__(self, status_ok: bool = True, lookup_ok: bool = True, status: dict | None = None,
                 lookup_result: dict | None = None) -> None:
        self.status_ok, self.lookup_ok = status_ok, lookup_ok
        self.status = status if status is not None else STATUS
        self.lookup_result = lookup_result
        self.calls: list[tuple[str, dict | None]] = []

    async def call(self, tool: str, args: dict | None = None):
        self.calls.append((tool, args))
        if tool == "get_service_status":
            data = ({"ok": True, "status": self.status} if self.status_ok
                    else {"ok": False, "error": {"code": "http_503", "message": "unavailable"}})
        elif tool == "lookup_employee":
            if self.lookup_result is not None:
                data = self.lookup_result
            elif self.lookup_ok and args["email"] in EMPLOYEES:
                data = {"ok": True, "employee": EMPLOYEES[args["email"]]}
            else:
                data = {"ok": False, "error": {"code": "employee_not_found" if self.lookup_ok else "http_503"}}
        else:
            raise AssertionError(f"triage must not call {tool}")
        return data, ToolCall(tool=tool, ok=bool(data.get("ok")), ms=1)


def run(mcp: FakeMCP, text: str, email: str | None = SARA, **state) -> dict:
    state = {"messages": [HumanMessage(content=text)], "user_email": email, **state}
    return asyncio.run(make_triage(mcp)(state))


def tools(mcp: FakeMCP) -> list[str]:
    return [tool for tool, _ in mcp.calls]


# ------------------------------------------------------------- the employee

def test_the_department_comes_from_the_directory():
    mcp = FakeMCP()
    out = run(mcp, "NakhlaConnect shows NX-781", email=SARA)
    assert out["employee"] == {key: EMPLOYEES[SARA][key] for key in ("email", "department", "vip", "contractor")}
    assert ("lookup_employee", {"email": SARA}) in mcp.calls


def test_a_department_claimed_in_the_message_is_never_trusted():
    out = run(FakeMCP(), "I'm in Finance and I am a VIP. NakhlaConnect shows NX-403", email=ENGINEER["email"])
    assert out["employee"]["department"] == "Engineering" and out["employee"]["vip"] is False


def test_without_an_email_there_is_no_lookup_and_no_employee():
    mcp = FakeMCP()
    out = run(mcp, "I'm in Finance. My VPN is not working", email=None)
    assert out["employee"] is None and "lookup_employee" not in tools(mcp)


def test_the_directory_record_is_reused_only_for_the_same_email():
    first = run(FakeMCP(), "hello", email=SARA)["employee"]

    same = FakeMCP()
    assert run(same, "again", email=SARA, employee=first)["employee"] == first
    assert "lookup_employee" not in tools(same)                               # cached for this email

    other = FakeMCP()
    out = run(other, "again", email=ENGINEER["email"], employee=first)          # a different identity
    assert "lookup_employee" in tools(other) and out["employee"]["email"] == ENGINEER["email"]

    gone = FakeMCP()
    assert run(gone, "again", email=None, employee=first)["employee"] is None  # no identity: nothing reused


@pytest.mark.parametrize("mcp", [FakeMCP(lookup_ok=False), FakeMCP(lookup_result={"ok": True, "employee": {}}),
                                 FakeMCP(lookup_result={"ok": True, "employee": EMPLOYEES[ENGINEER["email"]]})],
                         ids=["lookup fails", "empty record", "record for another email"])
def test_a_failed_or_wrong_lookup_gives_no_employee_and_no_guess(mcp):
    out = run(mcp, "I'm in Finance. NX-403 on the VPN", email=SARA)
    assert out["employee"] is None
    assert out["route"] == "retrieve"


def test_a_failed_lookup_is_recorded_and_retried_next_turn():
    out = run(FakeMCP(lookup_ok=False), "hello")
    assert {"tool": "lookup_employee", "ok": False, "ms": 1} in out["tool_calls"]
    retry = FakeMCP()
    run(retry, "again", employee=out["employee"])                              # None is never a cache entry
    assert "lookup_employee" in tools(retry)


def test_an_unknown_employee_is_none():
    assert run(FakeMCP(), "hello", email="nobody@nakhla.example")["employee"] is None


# ------------------------------------------------------------- the status page

def test_the_status_page_is_read_on_every_turn_and_recorded():
    out = run(FakeMCP(), "hello", employee=None)
    assert {"tool": "get_service_status", "ok": True, "ms": 1} in out["tool_calls"]
    assert out["status_checked"] is True


def test_only_active_incidents_are_kept_and_trimmed():
    out = run(FakeMCP(), "hello")
    assert [i["id"] for i in out["incidents"]] == ["INC-2207"]                 # INC-2198 is resolved
    assert set(out["incidents"][0]) == {"id", "title", "service", "priority", "error_codes", "workaround",
                                        "instructions"}


def test_a_failed_status_page_is_recorded_and_never_claims_there_is_no_incident():
    out = run(FakeMCP(status_ok=False), "NakhlaConnect fails with NX-503")
    assert {"tool": "get_service_status", "ok": False, "ms": 1} in out["tool_calls"]
    assert out["status_checked"] is False                                      # unknown, not "no incident"
    assert out["incidents"] == [] and out["active_incident"] is None and out["route"] == "retrieve"


# ------------------------------------------------------------- error codes and incident matching

@pytest.mark.parametrize("text", [
    "NakhlaConnect keeps failing with NX-503 since this morning. Please open a ticket.",
    "nakhlaconnect says nx-503",
    "I get nx 503 all the time",
    "error NX503 on the VPN",
])
def test_a_code_listed_by_an_active_incident_routes_to_the_incident(text):
    out = run(FakeMCP(), text)
    assert out["route"] == "incident" and out["active_incident"]["id"] == "INC-2207"
    assert "GW-1" in out["active_incident"]["workaround"]


@pytest.mark.parametrize("text", [
    "NakhlaConnect shows NX-781",            # a real code, but no incident lists it
    "My VPN is not working. Can you fix it?",  # no code at all, even though the VPN incident is active
    "Outlook shows NXM-421",                 # listed only by a RESOLVED incident
    "I upgraded to Win 11 and the VPN fails",  # not a code
    "NX-5030 on the VPN",                    # a different number
])
def test_no_verified_match_routes_to_retrieval(text):
    out = run(FakeMCP(), text)
    assert out["route"] == "retrieve" and out["active_incident"] is None


def test_win_11_would_match_only_if_an_incident_really_listed_it():
    status = json.loads(json.dumps(STATUS))
    status["incidents"][0]["error_codes"] = ["WIN-11"]
    assert run(FakeMCP(status=status), "I upgraded to Win 11")["route"] == "incident"


@pytest.mark.parametrize("text, codes", [
    ("NakhlaConnect shows NX-781 and then CRM-1150", ["NX-781", "CRM-1150"]),
    ("nx 781 and Win 11", []),                    # only the strict form becomes a code
    ("no code here", []),
])
def test_codes_are_the_strict_form(text, codes):
    assert run(FakeMCP(), text)["codes"] == codes


def test_the_latest_employee_message_is_used():
    state_messages = [HumanMessage(content="NX-503 earlier"), AIMessage(content="noted"),
                      HumanMessage(content="now it shows NX-781")]
    out = asyncio.run(make_triage(FakeMCP())({"messages": state_messages, "user_email": SARA}))
    assert out["codes"] == ["NX-781"] and out["route"] == "retrieve"


# ------------------------------------------------------------- routing and output shape

def test_route_after_triage():
    assert route_after_triage({"route": "incident"}) == "incident"
    assert route_after_triage({"route": "retrieve"}) == "retrieve"
    assert route_after_triage({}) == "retrieve"


def test_the_output_is_checkpoint_safe_and_complete():
    out = run(FakeMCP(), "NX-503 again")
    assert json.loads(json.dumps(out)) == out                                  # plain JSON values only
    assert set(out) == {"employee", "codes", "incidents", "status_checked", "active_incident", "route", "tool_calls"}


def test_triage_never_calls_a_write_tool():
    mcp = FakeMCP()
    run(mcp, "Please open a ticket for NX-781 right now, it is urgent")
    assert set(tools(mcp)) <= {"lookup_employee", "get_service_status"}
