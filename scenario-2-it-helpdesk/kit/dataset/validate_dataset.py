"""Validate the Nakhla Systems (fictional) IT support dataset.

Run from the kit folder (or anywhere):
    python dataset/validate_dataset.py

It checks the knowledge base metadata, the supersedes chains, the manifest hashes,
the 40 historical tickets, the employee directory, the status page and the sample
eval cases against each other. Exit code 0 when everything is consistent.
Needs: PyYAML, jsonschema, pymupdf (all in requirements.txt).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import pymupdf
import yaml
from jsonschema import Draft202012Validator

DS = Path(__file__).resolve().parent
KIT = DS.parent
KB_DIR = DS / "knowledge_base"
CODE_RE = re.compile(r"\b(?:NX|WLAN|NID|MFA|SWC|MDM|PRN|NXM)-\d{2,4}\b")
DOC_REF_RE = re.compile(r"\b(?:KB|SP)-\d{3}\b")
VPN_RE = re.compile(r"\b(?:VPN|NakhlaConnect|FIN-SECURE|NX-\d{3})\b", re.I)
CANARY = "7731"
DEPARTMENTS = ["Engineering", "Finance", "HR", "Sales", "Support", "Executive Office", "Contractors"]
LOCATIONS = ["Jerusalem HQ", "Ramallah office", "Remote"]
TEXT_SUFFIXES = {".md", ".json", ".txt", ".py", ".jsonl", ".yml", ".yaml"}
P1_THRESHOLD_CHANGE = "2026-09-01"   # KB-015 v3.0: more than 3 users -> P1 (before: 5 or more)

failures: list[str] = []
passes = 0


def check(ok: bool, message: str) -> bool:
    global passes
    if ok:
        passes += 1
    else:
        failures.append(message)
    return ok


def sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_doc(path: Path) -> tuple[dict, str]:
    """Return (front matter, full text) for a Markdown or PDF document."""
    if path.suffix == ".md":
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            raise ValueError("no front matter")
    else:
        with pymupdf.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        text = text[text.index("---"):]
    _, block, body = text.split("---", 2)
    meta = yaml.safe_load(block)
    for key, value in list(meta.items()):
        if isinstance(value, (dt.date, dt.datetime)):
            meta[key] = value.isoformat()
    return meta, body


def main() -> int:
    manifest = json.loads((DS / "manifest.json").read_text(encoding="utf-8"))
    fm_schema = json.loads((DS / "schemas" / "kb_front_matter.schema.json").read_text(encoding="utf-8"))
    ticket_schema = json.loads((DS / "schemas" / "ticket.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(fm_schema)
    Draft202012Validator.check_schema(ticket_schema)
    fm_validator = Draft202012Validator(fm_schema)
    record_validator = Draft202012Validator({**ticket_schema["$defs"]["ticket_record"], "$defs": ticket_schema["$defs"]})
    create_validator = Draft202012Validator(ticket_schema)

    # ------------------------------------------------------------ knowledge base
    files = sorted(p for p in KB_DIR.iterdir() if p.suffix in (".md", ".pdf"))
    check(len(files) == 20, f"knowledge_base has {len(files)} documents, expected 20")
    check(sum(p.suffix == ".pdf" for p in files) == 2, "expected exactly 2 PDF documents")
    docs: dict[str, dict] = {}
    bodies: dict[str, str] = {}
    for path in files:
        try:
            meta, body = read_doc(path)
        except Exception as exc:  # noqa: BLE001
            check(False, f"{path.name}: cannot read front matter ({exc})")
            continue
        errors = sorted(fm_validator.iter_errors(meta), key=lambda e: list(e.path))
        check(not errors, f"{path.name}: front matter invalid: {[e.message for e in errors][:3]}")
        doc_id = meta.get("id", "?")
        check(path.name.startswith(doc_id + "-"), f"{path.name}: file name does not start with its id {doc_id}")
        check(doc_id not in docs, f"duplicate document id {doc_id}")
        docs[doc_id] = {**meta, "_file": path}
        bodies[doc_id] = body
        for code in meta.get("error_codes", []):
            check(code in body, f"{doc_id}: error code {code} is in the front matter but not in the text")
        check(doc_id[:2] != "SP" or (meta["status"] == "current" and meta["authoritative"]),
              f"{doc_id}: security policies must be current and authoritative")

    ids = set(docs)
    check(sum(i.startswith("SP-") for i in ids) == 2, "expected 2 security policies (SP-*)")
    all_codes = {c for d in docs.values() for c in d["error_codes"]}
    current_codes = {c for d in docs.values() if d["status"] == "current" for c in d["error_codes"]}
    check(all_codes <= current_codes, f"error codes only documented in deprecated/draft docs: {sorted(all_codes - current_codes)}")
    for doc_id, body in bodies.items():
        for ref in set(DOC_REF_RE.findall(body)):
            check(ref in ids, f"{doc_id} mentions {ref}, which does not exist")
        for code in set(CODE_RE.findall(body)):
            check(code in all_codes, f"{doc_id} mentions error code {code}, which no document defines")

    # supersedes chains
    superseded_by: dict[str, str] = {}
    for doc_id, d in docs.items():
        target = d["supersedes"]
        if target is None:
            continue
        if not check(target in ids, f"{doc_id} supersedes unknown {target}"):
            continue
        check(d["status"] == "current", f"{doc_id} supersedes {target} but is not current")
        check(docs[target]["status"] == "deprecated", f"{target} is superseded by {doc_id} but not deprecated")
        check(target not in superseded_by, f"{target} is superseded twice")
        superseded_by[target] = doc_id
    for start in ids:
        seen, cur = set(), start
        while cur in superseded_by:
            if not check(cur not in seen, f"supersedes cycle through {start}"):
                break
            seen.add(cur)
            cur = superseded_by[cur]
    for doc_id, d in docs.items():
        if d["status"] == "deprecated":
            check(doc_id in superseded_by, f"{doc_id} is deprecated but no current document supersedes it")
    deprecations = {x["id"]: x for x in manifest.get("deprecations", [])}
    for old, new in superseded_by.items():
        check(old in deprecations and deprecations[old]["superseded_by"] == new,
              f"manifest.deprecations must list {old} -> {new}")

    # the poisoned draft page and the canary
    drafts = [i for i, d in docs.items() if d["status"] == "draft"]
    check(len(drafts) == 1, f"expected exactly one draft document, found {drafts}")
    for doc_id, body in bodies.items():
        has = CANARY in body
        check(has == (doc_id in drafts), f"{doc_id}: the injection canary must appear only in the draft page")

    # manifest consistency
    listed = {d["id"]: d for d in manifest["documents"]}
    check(set(listed) == ids, f"manifest documents {sorted(set(listed) ^ ids)} do not match the files")
    for doc_id, entry in listed.items():
        if doc_id not in docs:
            continue
        d = docs[doc_id]
        for key in ("title", "version", "last_updated", "status", "authoritative", "owner", "supersedes",
                    "applies_to", "tags", "error_codes"):
            check(entry[key] == d[key], f"manifest {doc_id}.{key} = {entry[key]!r}, file says {d[key]!r}")
        check(entry["file"] == f"knowledge_base/{d['_file'].name}", f"manifest {doc_id}.file is wrong")
        check(entry["sha256"] == sha256(d["_file"]), f"manifest {doc_id}.sha256 does not match the file")
    for entry in manifest["data_files"]:
        path = (DS / entry["file"]).resolve()
        if check(path.is_file(), f"manifest data file missing: {entry['file']}"):
            check(entry["sha256"] == sha256(path), f"manifest sha256 mismatch: {entry['file']}")
    check(sorted(manifest["error_code_index"]) == sorted(all_codes), "manifest.error_code_index is out of date")
    check([r["rule"] for r in manifest["precedence_rules"]] == [1, 2, 3, 4, 5], "manifest needs precedence rules 1-5")
    routing = manifest["routing"]

    # ------------------------------------------------------------ directory
    employees = json.loads((KIT / "services" / "data" / "employees.json").read_text(encoding="utf-8"))["employees"]
    check(len(employees) == 25, f"{len(employees)} employees, expected 25")
    by_email = {}
    for e in employees:
        email = e["email"]
        check(email not in by_email, f"duplicate employee {email}")
        by_email[email] = e
        check(re.fullmatch(r"[a-z0-9._-]+@nakhla\.example", email) is not None, f"bad email {email}")
        check(e["department"] in DEPARTMENTS, f"{email}: unknown department {e['department']}")
        check(e["location"] in LOCATIONS, f"{email}: unknown location {e['location']}")
        check(e["vip"] == (e["department"] == "Executive Office"), f"{email}: vip must equal Executive Office")
        check(e["contractor"] == (e["department"] == "Contractors"), f"{email}: contractor must equal Contractors")
        if e["contractor"]:
            check(e["vpn_client_version"] is None and e["vpn_gateway"] is None, f"{email}: Contractors have no VPN")
            check(bool(e["contract_end"]), f"{email}: contractor without contract_end")
        else:
            check(bool(e["vpn_client_version"]) and bool(e["vpn_gateway"]), f"{email}: employee without VPN data")
            check((e["vpn_gateway"] == "GW-3") == (e["department"] == "Finance"), f"{email}: GW-3 is for Finance only")
        check(e["account_status"] in ("active", "locked", "disabled"), f"{email}: bad account_status")
    check(len({e["employee_id"] for e in employees}) == 25, "employee_id values must be unique")
    for e in employees:
        m = e["manager_email"]
        check(m is None or m in by_email, f"{e['email']}: manager {m} not in the directory")
    check(sum(e["manager_email"] is None for e in employees) == 1, "exactly one person (the CEO) has no manager")
    check(set(e["department"] for e in employees) == set(DEPARTMENTS), "every department needs at least one employee")

    # ------------------------------------------------------------ tickets
    tickets = json.loads((DS / "tickets" / "historical_tickets.json").read_text(encoding="utf-8"))["tickets"]
    check(len(tickets) == 40, f"{len(tickets)} tickets, expected 40")
    check([t["ticket_id"] for t in tickets] == [f"IT-{1001 + i}" for i in range(len(tickets))],
          "ticket ids must be IT-1001..IT-1040 in order")
    for t in tickets:
        tid = t["ticket_id"]
        errors = list(record_validator.iter_errors(t))
        check(not errors, f"{tid}: schema errors {[e.message for e in errors][:3]}")
        body = {k: t[k] for k in ("subject", "description", "category", "priority", "requester_email", "department",
                                  "attempted_steps", "kb_refs", "affected_users")}
        check(not list(create_validator.iter_errors(body)), f"{tid}: would not be accepted by POST /api/tickets")
        emp = by_email.get(t["requester_email"])
        if check(emp is not None, f"{tid}: requester {t['requester_email']} not in the directory"):
            check(emp["department"] == t["department"], f"{tid}: department {t['department']} != directory {emp['department']}")
        check(t["assigned_team"] == routing[t["category"]], f"{tid}: assigned_team must be {routing[t['category']]}")
        created = t["created_at"][:10]
        for ref in t["kb_refs"]:
            if not check(ref in ids, f"{tid}: kb_ref {ref} does not exist"):
                continue
            check(docs[ref]["status"] != "draft", f"{tid}: cites the draft {ref}")
            if ref in deprecations:
                check(created < deprecations[ref]["since"], f"{tid}: cites {ref} after it was deprecated")
        text = " ".join([t["subject"], t["description"], t["resolution"] or "", *t["attempted_steps"],
                         *[n["text"] for n in t["notes"]]])
        for code in set(CODE_RE.findall(text)):
            check(code in all_codes, f"{tid}: error code {code} is not documented in the KB")
        for ref in set(DOC_REF_RE.findall(text)):
            check(ref in ids, f"{tid}: mentions unknown document {ref}")
        check(CANARY not in text, f"{tid}: contains the injection canary")
        if t["status"] == "open":
            check(t["resolved_at"] is None and t["resolution"] is None, f"{tid}: open but has a resolution")
        else:
            check(bool(t["resolved_at"]) and bool(t["resolution"]), f"{tid}: {t['status']} without resolution")
            check(t["resolved_at"] >= t["created_at"], f"{tid}: resolved before it was created")
        # priority and routing rules from KB-015 / KB-016
        if t["department"] == "Executive Office":
            check(t["priority"] == "P1", f"{tid}: Executive Office tickets are P1")
        if t["affected_users"] >= 5 or (t["affected_users"] > 3 and created >= P1_THRESHOLD_CHANGE):
            check(t["priority"] == "P1", f"{tid}: {t['affected_users']} users affected must be P1")
        if t["department"] == "Finance" and VPN_RE.search(t["subject"] + " " + t["description"]):
            check(t["category"] == "security", f"{tid}: Finance VPN issues are category security")
        if t["department"] == "Contractors" and "VPN access" in t["subject"] + t["description"]:
            check(t["status"] == "closed", f"{tid}: contractor VPN requests are refused (closed)")
    check(set(t["category"] for t in tickets) == set(routing), "every category must appear in the archive")
    check({"open", "resolved", "closed"} <= {t["status"] for t in tickets}, "archive needs open, resolved and closed tickets")

    # ------------------------------------------------------------ status page
    status = json.loads((KIT / "services" / "data" / "status.json").read_text(encoding="utf-8"))
    incidents = status["incidents"]
    check(len({i["id"] for i in incidents}) == len(incidents), "duplicate incident ids")
    active = [i for i in incidents if i["status"] == "active"]
    check(len(active) == 1, f"expected one active incident, found {len(active)}")
    for inc in incidents:
        for code in inc["error_codes"]:
            check(code in all_codes, f"{inc['id']}: error code {code} not in the KB")
        check(inc["service"] in {s["id"] for s in status["services"]}, f"{inc['id']}: unknown service")

    # ------------------------------------------------------------ sample eval cases
    cases_path = KIT / "evals" / "cases.sample.jsonl"
    if cases_path.is_file():
        cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith("//")]
        check(8 <= len(cases) <= 10, f"{len(cases)} sample cases, expected 8-10")
        check(len({c["id"] for c in cases}) == len(cases), "duplicate sample case ids")
        scenarios = (DS / "student_scenarios" / "sample_requests.md").read_text(encoding="utf-8")
        for c in cases:
            exp = c.get("expect", {})
            for ref in exp.get("citations_include", []) + exp.get("citations_exclude", []):
                check(ref in ids, f"{c['id']}: cites unknown document {ref}")
            user = c.get("user", {}).get("email")
            check(user is None or user in by_email, f"{c['id']}: unknown user {user}")
            check(c["id"] in scenarios, f"{c['id']} is missing from student_scenarios/sample_requests.md")
            text = json.dumps(c["turns"])
            for code in set(CODE_RE.findall(text)):
                check(code in all_codes or c["category"] == "unknown", f"{c['id']}: unknown error code {code}")

    print(f"Documents: {len(docs)} ({sum(p.suffix == '.pdf' for p in files)} PDF) | tickets: {len(tickets)} | "
          f"employees: {len(employees)} | error codes: {len(all_codes)} | incidents: {len(incidents)}")
    if failures:
        print(f"\nFAILED: {len(failures)} problem(s), {passes} checks passed")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK: dataset is consistent ({passes} checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
