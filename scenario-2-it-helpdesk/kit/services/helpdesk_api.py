"""Simulated IT helpdesk API of Nakhla Systems (a fictional company) - the system of record.

    uvicorn services.helpdesk_api:app --port 8102        (run from the kit folder)

Business endpoints (header X-API-Key required):
    GET  /api/directory/{email}          employee record, 404 if unknown
    GET  /api/status                     IT status page: services and incidents
    GET  /api/tickets?requester_email=&q=&status=&limit=   search the ticket archive
    GET  /api/tickets/{ticket_id}        one ticket
    POST /api/tickets                    create a ticket (body: dataset/schemas/ticket.schema.json, Idempotency-Key header)
    POST /api/tickets/{ticket_id}/notes  add a note ({"note": "...", "author": "..."}, Idempotency-Key header)
Test-only admin endpoints (no key, for eval runners - your app must not call them):
    POST /admin/reset · POST /admin/faults · DELETE /admin/faults · GET /admin/faults
    GET  /admin/log · GET /admin/tickets (?all=1) · GET /admin/notes
Env: HELPDESK_API_KEY (default dev-key-change-me), HELPDESK_TODAY (YYYY-MM-DD, default real today),
     HELPDESK_DATA_DIR (default services/data), HELPDESK_DATASET_DIR (default dataset).
State is in memory, seeded from JSON at start-up and on POST /admin/reset.
"""
from __future__ import annotations

import asyncio
import copy
import datetime as dt
import hashlib
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator
from starlette.exceptions import HTTPException as StarletteHTTPException

HERE = Path(__file__).resolve().parent
KIT = HERE.parent


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from the kit's .env without overriding real env vars."""
    env = KIT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()
API_KEY = os.getenv("HELPDESK_API_KEY") or "dev-key-change-me"
DATA_DIR = Path(os.getenv("HELPDESK_DATA_DIR") or HERE / "data")
DATASET_DIR = Path(os.getenv("HELPDESK_DATASET_DIR") or KIT / "dataset")

FAULT_TARGETS = {"create_ticket", "directory", "status", "search_tickets", "get_ticket", "add_note"}
FAULT_MODES = {"error_500", "error_503", "timeout", "slow"}
ROUTING = {"network_access": "Network Operations", "identity_access": "Identity & Access Management",
           "hardware": "Endpoint Engineering", "software": "Endpoint Engineering",
           "email_collab": "Collaboration Services", "security": "Security Operations",
           "onboarding": "IT Service Desk", "other": "IT Service Desk"}
STOPWORDS = {"the", "and", "for", "with", "my", "is", "a", "an", "to", "of", "on", "in", "it", "i", "at", "not",
             "can", "cannot", "can't", "does", "doesn't", "when", "this", "that", "from", "me", "how", "what", "after"}

app = FastAPI(
    title="Nakhla Systems helpdesk API (simulated)",
    version="1.0",
    description="Simulated IT helpdesk of Nakhla Systems, a fictional company. Business endpoints need the "
                "X-API-Key header. /admin/* endpoints are test-only.",
)


# ----------------------------------------------------------------- state
def today() -> dt.date:
    value = (os.getenv("HELPDESK_TODAY") or "").strip()
    return dt.date.fromisoformat(value) if value else dt.date.today()


def now_iso() -> str:
    return f"{today().isoformat()}T{dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')}Z"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_dates(text: str) -> str:
    """Replace {TODAY} and {TODAY-N} in the status page with real dates."""
    def sub(m: re.Match) -> str:
        days = int(m.group(1) or 0)
        return (today() - dt.timedelta(days=days)).isoformat()
    return re.sub(r"\{TODAY(?:-(\d+))?\}", sub, text)


SCHEMA = _read_json(DATASET_DIR / "schemas" / "ticket.schema.json")
CREATE_VALIDATOR = Draft202012Validator(SCHEMA)
STATE: dict[str, Any] = {}


def reset_state() -> None:
    manifest = _read_json(DATASET_DIR / "manifest.json")
    superseded_by = {d["supersedes"]: d["id"] for d in manifest["documents"] if d.get("supersedes")}
    STATE.clear()
    STATE.update({
        "employees": {e["email"]: e for e in _read_json(DATA_DIR / "employees.json")["employees"]},
        "status": json.loads(_render_dates((DATA_DIR / "status.json").read_text(encoding="utf-8"))),
        "archive": _read_json(DATASET_DIR / "tickets" / "historical_tickets.json")["tickets"],
        "created": [],
        "notes_added": [],
        "next_id": 2001,
        "idempotency": {},
        "faults": {},
        "log": deque(maxlen=500),
        "docs": {d["id"]: {"title": d["title"], "status": d["status"],
                           "superseded_by": superseded_by.get(d["id"])} for d in manifest["documents"]},
    })
    STATE["status"].pop("_comment", None)


reset_state()


def all_tickets() -> list[dict]:
    return STATE["archive"] + STATE["created"]


def find_ticket(ticket_id: str) -> dict | None:
    tid = ticket_id.strip().upper()
    return next((t for t in all_tickets() if t["ticket_id"] == tid), None)


# ----------------------------------------------------------------- helpers
def error(status: int, code: str, message: str, **extra) -> JSONResponse:
    headers = {"Retry-After": "5"} if status == 503 else None
    return JSONResponse({"error": {"code": code, "message": message, **extra}}, status_code=status, headers=headers)


async def fault(request: Request, operation: str) -> JSONResponse | None:
    """Mark the operation for the log and apply an armed fault, if any."""
    request.state.operation = operation
    armed = STATE["faults"].get(operation)
    if not armed or armed["remaining"] <= 0:
        return None
    armed["remaining"] -= 1
    if armed["remaining"] <= 0:
        STATE["faults"].pop(operation, None)
    mode = armed["mode"]
    if mode == "error_500":
        return error(500, "internal_error", f"Simulated failure: {operation} crashed. Nothing was changed.")
    if mode == "error_503":
        return error(503, "service_unavailable", f"Simulated outage: {operation} is temporarily unavailable. Nothing was changed.")
    if mode == "timeout":
        await asyncio.sleep(15)
        return error(504, "gateway_timeout", f"Simulated timeout: {operation} took too long. Nothing was changed.")
    await asyncio.sleep(3)  # slow: succeed after a delay
    return None


def body_hash(body: Any) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def field_errors(body: dict) -> list[dict]:
    out = []
    for err in sorted(CREATE_VALIDATOR.iter_errors(body), key=lambda e: list(e.absolute_path)):
        if err.validator == "required":
            field = re.search(r"'(.+?)' is a required property", err.message)
            out.append({"field": field.group(1) if field else "?", "message": "is required"})
        elif err.validator == "additionalProperties":
            out.append({"field": ",".join(re.findall(r"'([^']+)'", err.message)) or "?",
                        "message": "is not a ticket field (see dataset/schemas/ticket.schema.json)"})
        else:
            path = ".".join(str(p) for p in err.absolute_path) or "body"
            out.append({"field": path, "message": err.message})
    return out


# ----------------------------------------------------------------- middleware and errors
@app.middleware("http")
async def auth_and_log(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path
    if path.startswith("/api/") and request.headers.get("x-api-key") != API_KEY:
        response = error(401, "unauthorized", "Missing or wrong X-API-Key header.")
    else:
        response = await call_next(request)
    if path not in ("/admin/log", "/favicon.ico"):
        STATE["log"].append({
            "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "method": request.method, "path": path, "query": str(request.url.query) or None,
            "status": response.status_code, "ms": round((time.perf_counter() - start) * 1000, 1),
            "operation": getattr(request.state, "operation", None),
        })
    return response


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException):
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    return error(exc.status_code, code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def request_error(_: Request, exc: RequestValidationError):
    fields = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body", "message": e["msg"]} for e in exc.errors()]
    return error(422, "validation_error", "The request is invalid.", fields=fields)


# ----------------------------------------------------------------- business endpoints
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": "helpdesk"}


@app.get("/api/directory/{email}", tags=["directory"])
async def get_employee(email: str, request: Request):
    if (resp := await fault(request, "directory")) is not None:
        return resp
    employee = STATE["employees"].get(email.strip().lower())
    if employee is None:
        return error(404, "employee_not_found", f"No employee with email {email} in the Nakhla directory.")
    return employee


@app.get("/api/status", tags=["status"])
async def get_status(request: Request):
    if (resp := await fault(request, "status")) is not None:
        return resp
    status = copy.deepcopy(STATE["status"])
    active = [i["id"] for i in status["incidents"] if i["status"] == "active"]
    return {"generated_at": now_iso(), "active_incidents": active, **status}


@app.get("/api/tickets", tags=["tickets"])
async def search_tickets(request: Request, requester_email: str | None = None, q: str | None = None,
                         status: str | None = None, limit: int = 10):
    if (resp := await fault(request, "search_tickets")) is not None:
        return resp
    if status and status not in ("open", "resolved", "closed"):
        return error(422, "validation_error", "status must be open, resolved or closed.",
                     fields=[{"field": "status", "message": "must be open, resolved or closed"}])
    if not 1 <= limit <= 50:
        return error(422, "validation_error", "limit must be between 1 and 50.",
                     fields=[{"field": "limit", "message": "must be between 1 and 50"}])
    tickets = all_tickets()
    if requester_email:
        tickets = [t for t in tickets if t["requester_email"] == requester_email.strip().lower()]
    if status:
        tickets = [t for t in tickets if t["status"] == status]
    scored = [(0, t) for t in tickets]
    if q and q.strip():
        terms = {w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", q.lower()) if len(w) >= 2 and w not in STOPWORDS}
        scored = []
        for t in tickets:
            hay = f"{t['subject']} {t['description']}".lower()
            score = sum(1 for w in terms if (w in hay if len(w) >= 3 else re.search(rf"\b{re.escape(w)}\b", hay)))
            if score:
                scored.append((score, t))
    scored.sort(key=lambda st: (st[0], st[1]["created_at"]), reverse=True)
    results = [copy.deepcopy(t) for _, t in scored[:limit]]
    return {"tickets": results, "count": len(results), "total_matches": len(scored)}


@app.get("/api/tickets/{ticket_id}", tags=["tickets"])
async def get_ticket(ticket_id: str, request: Request):
    if (resp := await fault(request, "get_ticket")) is not None:
        return resp
    ticket = find_ticket(ticket_id)
    if ticket is None:
        return error(404, "ticket_not_found", f"No ticket {ticket_id}.")
    return copy.deepcopy(ticket)


@app.post("/api/tickets", status_code=201, tags=["tickets"])
async def create_ticket(request: Request):
    if (resp := await fault(request, "create_ticket")) is not None:
        return resp
    try:
        body = await request.json()
    except ValueError:
        return error(400, "invalid_json", "The body must be a JSON object.")
    if not isinstance(body, dict):
        return error(400, "invalid_json", "The body must be a JSON object.")
    if isinstance(body.get("requester_email"), str):
        body["requester_email"] = body["requester_email"].strip().lower()
    if isinstance(body.get("kb_refs"), list):
        body["kb_refs"] = [r.strip().upper() if isinstance(r, str) else r for r in body["kb_refs"]]

    key = request.headers.get("idempotency-key")
    if key:
        seen = STATE["idempotency"].get(("create_ticket", key))
        if seen:
            if seen["hash"] != body_hash(body):
                return error(409, "idempotency_conflict",
                             "This Idempotency-Key was already used with a different ticket body.")
            return JSONResponse(copy.deepcopy(seen["response"]), status_code=201, headers={"Idempotent-Replayed": "true"})

    errors = field_errors(body)
    employee = STATE["employees"].get(body.get("requester_email", ""))
    if isinstance(body.get("requester_email"), str) and employee is None and not any(
            e["field"] == "requester_email" for e in errors):
        errors.append({"field": "requester_email", "message": "is not an employee in the directory"})
    if employee and isinstance(body.get("department"), str) and body["department"] != employee["department"]:
        errors.append({"field": "department", "message": f"does not match the directory ({employee['department']})"})
    for ref in body.get("kb_refs", []) if isinstance(body.get("kb_refs"), list) else []:
        if isinstance(ref, str) and re.fullmatch(r"(KB|SP)-\d{3}", ref) and ref not in STATE["docs"]:
            errors.append({"field": "kb_refs", "message": f"{ref} is not a document in the knowledge base"})
    if errors:
        return error(422, "validation_error", "The ticket is invalid. Fix the fields listed.", fields=errors)

    warnings = []
    for ref in body["kb_refs"]:
        doc = STATE["docs"][ref]
        if doc["status"] == "deprecated":
            warnings.append(f"{ref} is deprecated" + (f" (superseded by {doc['superseded_by']})" if doc["superseded_by"] else ""))
        elif doc["status"] == "draft":
            warnings.append(f"{ref} is an unreviewed draft")
    ticket = {
        "ticket_id": f"IT-{STATE['next_id']}",
        "subject": body["subject"], "description": body["description"],
        "category": body["category"], "priority": body["priority"], "status": "open",
        "requester_email": body["requester_email"], "department": body["department"],
        "affected_users": body.get("affected_users", 1),
        "assigned_team": ROUTING[body["category"]],
        "attempted_steps": body["attempted_steps"], "kb_refs": body["kb_refs"],
        "resolution": None, "created_at": now_iso(), "resolved_at": None, "notes": [],
    }
    if warnings:
        ticket["warnings"] = warnings
    STATE["next_id"] += 1
    STATE["created"].append(ticket)
    if key:
        STATE["idempotency"][("create_ticket", key)] = {"hash": body_hash(body), "response": copy.deepcopy(ticket)}
    return JSONResponse(copy.deepcopy(ticket), status_code=201, headers={"Location": f"/api/tickets/{ticket['ticket_id']}"})


@app.post("/api/tickets/{ticket_id}/notes", status_code=201, tags=["tickets"])
async def add_note(ticket_id: str, request: Request):
    if (resp := await fault(request, "add_note")) is not None:
        return resp
    ticket = find_ticket(ticket_id)
    if ticket is None:
        return error(404, "ticket_not_found", f"No ticket {ticket_id}.")
    try:
        body = await request.json()
    except ValueError:
        return error(400, "invalid_json", "The body must be a JSON object.")
    note = body.get("note") if isinstance(body, dict) else None
    author = (body.get("author") if isinstance(body, dict) else None) or "IT assistant"
    if not isinstance(note, str) or not note.strip() or len(note) > 2000:
        return error(422, "validation_error", "note must be 1-2000 characters.",
                     fields=[{"field": "note", "message": "must be 1-2000 characters"}])
    key = request.headers.get("idempotency-key")
    if key:
        seen = STATE["idempotency"].get(("add_note", key))
        if seen:
            if seen["hash"] != body_hash({"t": ticket["ticket_id"], **body}):
                return error(409, "idempotency_conflict", "This Idempotency-Key was already used with a different note.")
            return JSONResponse(copy.deepcopy(seen["response"]), status_code=201, headers={"Idempotent-Replayed": "true"})
    entry = {"at": now_iso(), "author": str(author)[:80], "text": note.strip()}
    ticket["notes"].append(entry)
    STATE["notes_added"].append({"ticket_id": ticket["ticket_id"], **entry})
    response = {"ticket_id": ticket["ticket_id"], "note": entry, "notes_count": len(ticket["notes"])}
    if key:
        STATE["idempotency"][("add_note", key)] = {"hash": body_hash({"t": ticket["ticket_id"], **body}),
                                                   "response": response}
    return JSONResponse(response, status_code=201)


# ----------------------------------------------------------------- test-only admin endpoints
@app.post("/admin/reset", tags=["admin (test-only)"])
async def admin_reset():
    reset_state()
    return {"ok": True, "archive_tickets": len(STATE["archive"]), "created_tickets": 0, "today": today().isoformat()}


@app.post("/admin/faults", tags=["admin (test-only)"])
async def admin_set_fault(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return error(400, "invalid_json", "The body must be a JSON object.")
    target, mode, count = body.get("target"), body.get("mode"), body.get("count", 1)
    if target not in FAULT_TARGETS or mode not in FAULT_MODES or not isinstance(count, int) or not 1 <= count <= 100:
        return error(422, "validation_error", f"target must be one of {sorted(FAULT_TARGETS)}, mode one of "
                     f"{sorted(FAULT_MODES)}, count 1-100.")
    STATE["faults"][target] = {"mode": mode, "remaining": count}
    return {"ok": True, "faults": STATE["faults"]}


@app.get("/admin/faults", tags=["admin (test-only)"])
async def admin_get_faults():
    return {"faults": STATE["faults"]}


@app.delete("/admin/faults", tags=["admin (test-only)"])
async def admin_clear_faults():
    STATE["faults"].clear()
    return {"ok": True, "faults": {}}


@app.get("/admin/log", tags=["admin (test-only)"])
async def admin_log(limit: int = 100):
    entries = list(STATE["log"])[-max(1, min(limit, 500)):]
    return {"entries": entries, "count": len(entries)}


@app.get("/admin/tickets", tags=["admin (test-only)"])
async def admin_tickets(all: bool = False):  # noqa: A002 - query name fixed by the spec
    tickets = all_tickets() if all else STATE["created"]
    return {"tickets": copy.deepcopy(tickets), "count": len(tickets)}


@app.get("/admin/notes", tags=["admin (test-only)"])
async def admin_notes():
    return {"notes": copy.deepcopy(STATE["notes_added"]), "count": len(STATE["notes_added"])}
