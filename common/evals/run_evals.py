"""Shared eval runner for the Final Mini-Project (all three scenarios).

It plays test cases against YOUR system through the API contract
(POST /api/chat, POST /api/approve, POST /api/voice) and checks:
  - what your agent said and proposed   (the reply, pending_action, result, citations)
  - what really happened in the systems  (the simulated systems' /admin endpoints)

Usage (from your repo root, with your app and the simulated system running):
  python evals/run_evals.py                                  # all cases in evals/cases.sample.jsonl
  python evals/run_evals.py --cases evals/my_cases.jsonl     # your own cases
  python evals/run_evals.py --only S2-03,S2-05 --verbose     # a few cases, show every turn
  python evals/run_evals.py --repeat 3                       # run each case 3 times (flakiness)
  python evals/run_evals.py --judge                          # also grade "judge" criteria with an LLM

URLs come from flags or env (defaults in brackets):
  --app       APP_URL           [http://localhost:8000]   your entry point
  --clinic    CLINIC_API_URL    [http://localhost:8101]   Scenario 1
  --helpdesk  HELPDESK_API_URL  [http://localhost:8102]   Scenario 2
  --travel    TRAVEL_API_URL    [http://localhost:8103]   Scenario 3

Writes evals/scorecard.md and evals/scorecard.json. Exit code 0 only if every case passed.
The case format is documented in the mini-project's common/CONTRACT.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

try:
    import httpx
except ImportError:  # httpx comes with the openai package
    sys.exit("This runner needs httpx:  pip install httpx")

HERE = Path(__file__).resolve().parent
DEFAULTS = {
    "APP": ("APP_URL", "http://localhost:8000"),
    "CLINIC_API": ("CLINIC_API_URL", "http://localhost:8101"),
    "HELPDESK_API": ("HELPDESK_API_URL", "http://localhost:8102"),
    "TRAVEL_API": ("TRAVEL_API_URL", "http://localhost:8103"),
}


# ----------------------------------------------------------------- helpers
def load_env() -> None:
    """Load the nearest .env (for the --judge key) without needing python-dotenv."""
    for folder in [Path.cwd(), *Path.cwd().parents][:4]:
        env = folder / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def fill(text: str, urls: dict) -> str:
    for name, value in urls.items():
        text = text.replace("{" + name + "}", value.rstrip("/"))
    return text


def as_list(data) -> list:
    """Admin endpoints may return a list or {"bookings": [...]} - find the list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return value
    return []


def get_path(record, dotted: str):
    cur = record
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def record_matches(record, wanted: dict) -> bool:
    for key, value in wanted.items():
        got = get_path(record, key)
        if isinstance(value, str) and isinstance(got, str):
            if value.lower() not in got.lower():
                return False
        elif isinstance(value, str) and isinstance(got, list):
            if not any(value.lower() == str(x).lower() for x in got):
                return False
        elif got != value:
            return False
    return True


def citation_ids(resp: dict) -> list[str]:
    out = []
    for c in resp.get("citations") or []:
        out.append(str(c.get("id", c) if isinstance(c, dict) else c).upper())
    return out


def resolve_file(path: str, cases_file: Path) -> Path:
    for base in (Path.cwd(), cases_file.parent, cases_file.parent.parent):
        if (base / path).is_file():
            return base / path
    raise FileNotFoundError(path)


def mime_for(path: Path) -> str:
    return {".wav": "audio/wav", ".mp3": "audio/mpeg", ".webm": "audio/webm",
            ".ogg": "audio/ogg", ".m4a": "audio/mp4"}.get(path.suffix.lower(), "application/octet-stream")


# ----------------------------------------------------------------- one case
def run_case(case: dict, urls: dict, client: httpx.Client, cases_file: Path,
             judge, verbose: bool) -> dict:
    cid = case.get("id", "?")
    session = f"eval-{cid}-{uuid.uuid4().hex[:6]}"
    out = {"id": cid, "category": case.get("category", "other"), "passed": False,
           "checks": [], "turns": [], "latencies": [], "cost_usd": 0.0, "error": None}

    # 1. setup: reset and configure the simulated systems
    for step in case.get("setup", []):
        try:
            r = client.request(step.get("method", "POST"), fill(step["url"], urls),
                               json=step.get("json"), timeout=20)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"setup failed: {step.get('url')} -> {exc}"
            return out

    # 2. play the turns against YOUR app
    last: dict = {}
    pending = None
    last_action_id = None
    tools_seen: list[str] = []
    app = urls["APP"].rstrip("/")
    for turn in case.get("turns", []):
        t0 = time.perf_counter()
        try:
            if "say" in turn:
                body = {"session_id": session, "message": turn["say"]}
                if turn.get("user") or case.get("user"):
                    body["user"] = turn.get("user") or case.get("user")
                r = client.post(f"{app}/api/chat", json=body)
            elif "approve" in turn:
                if not pending:
                    out["error"] = "the case wanted to approve/reject, but the last reply had no pending_action"
                    break
                last_action_id = pending.get("action_id")
                r = client.post(f"{app}/api/approve", json={
                    "session_id": session, "action_id": last_action_id,
                    "approved": bool(turn["approve"])})
            elif "approve_again" in turn:
                # the double click: send the SAME approval again (same action_id)
                if not last_action_id:
                    out["error"] = "approve_again needs an earlier approve turn in the same case"
                    break
                r = client.post(f"{app}/api/approve", json={
                    "session_id": session, "action_id": last_action_id, "approved": True})
            elif "voice" in turn:
                audio = resolve_file(turn["voice"], cases_file)
                with audio.open("rb") as fh:
                    r = client.post(f"{app}/api/voice", data={"session_id": session},
                                    files={"audio": (audio.name, fh, mime_for(audio))})
            else:
                out["error"] = f"unknown turn type: {turn}"
                break
            seconds = time.perf_counter() - t0
            try:
                last = r.json()
                if not isinstance(last, dict):
                    last = {"reply": str(last)[:500]}
            except ValueError:
                last = {"reply": r.text[:500]}
            if r.status_code >= 400:
                last.setdefault("reply", "")
                last["_http_status"] = r.status_code
        except (httpx.HTTPError, FileNotFoundError) as exc:
            seconds = time.perf_counter() - t0
            last = {"reply": "", "_http_error": str(exc)}
        out["latencies"].append(round(seconds, 2))
        pending = last.get("pending_action") or None
        for tc in last.get("tool_calls") or []:
            tools_seen.append(str(tc.get("tool", tc) if isinstance(tc, dict) else tc))
        usage = last.get("usage") or {}
        try:
            out["cost_usd"] += float(usage.get("cost_usd") or usage.get("cost") or 0)
        except (TypeError, ValueError):
            pass
        out["turns"].append({"turn": turn, "seconds": round(seconds, 2),
                             "status": last.get("_http_status", 200),
                             "reply": (last.get("reply") or "")[:800],
                             "transcript": last.get("transcript"),
                             "pending_action": pending, "result": last.get("result"),
                             "citations": citation_ids(last)})
        if verbose:
            print(f"\n     > {json.dumps(turn, ensure_ascii=False)[:100]}  ({seconds:.1f}s)")
            if last.get("transcript"):
                print(f"       heard: {last['transcript'][:160]!r}")
            print(f"       < {(last.get('reply') or '')[:220]!r}")
            if pending:
                print(f"       < pending_action: {pending.get('type')} - {str(pending.get('summary', ''))[:120]}")
            if last.get("result"):
                print(f"       < result: {json.dumps(last['result'], ensure_ascii=False)[:120]}")
    if out["error"]:
        return out

    # 3. check the last response and the systems
    exp = case.get("expect", {})
    reply = (last.get("reply") or "").lower()
    checks = out["checks"]

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    if "_http_status" in last or "_http_error" in last:
        check("http", False, f"last call failed: {last.get('_http_status') or last.get('_http_error')}")
    if "reply_contains_any" in exp:
        words = exp["reply_contains_any"]
        check("reply_contains_any", any(w.lower() in reply for w in words), f"wanted one of {words}")
    if "reply_contains_all" in exp:
        missing = [w for w in exp["reply_contains_all"] if w.lower() not in reply]
        check("reply_contains_all", not missing, f"missing {missing}")
    if "reply_not_contains" in exp:
        bad = [w for w in exp["reply_not_contains"] if w.lower() in reply]
        check("reply_not_contains", not bad, f"must not say {bad}")
    if "reply_matches" in exp:
        check("reply_matches", re.search(exp["reply_matches"], last.get("reply") or "", re.I) is not None,
              f"regex {exp['reply_matches']}")
    if "pending_action" in exp:
        want = exp["pending_action"]
        if want == "none":
            check("pending_action", pending is None, f"expected none, got {pending and pending.get('type')}")
        elif want == "any":
            check("pending_action", pending is not None, "expected a pending_action")
        else:
            check("pending_action", bool(pending) and pending.get("type") == want,
                  f"expected {want}, got {pending and pending.get('type')}")
    result = last.get("result") or {}
    if "result_has" in exp:
        missing = [k for k in exp["result_has"] if not (isinstance(result, dict) and result.get(k))]
        check("result_has", not missing, f"result is missing {missing}")
    if exp.get("result_absent"):
        check("result_absent", not result, f"expected no result, got {result}")
    cites = citation_ids(last)
    if "citations_include" in exp:
        missing = [c for c in exp["citations_include"] if c.upper() not in cites]
        check("citations_include", not missing, f"missing {missing}, got {cites}")
    if "citations_exclude" in exp:
        bad = [c for c in exp["citations_exclude"] if c.upper() in cites]
        check("citations_exclude", not bad, f"must not cite {bad}")
    if "citations_min" in exp:
        check("citations_min", len(cites) >= exp["citations_min"], f"{len(cites)} citations")
    if "tools_used_include" in exp:
        missing = [t for t in exp["tools_used_include"] if t not in tools_seen]
        check("tools_used_include", not missing, f"never called {missing} (saw {sorted(set(tools_seen))})")
    if "max_latency_s" in exp:
        worst = max(out["latencies"] or [0])
        check("max_latency_s", worst <= exp["max_latency_s"], f"slowest turn {worst:.1f}s")
    for st in exp.get("state", []):
        url = fill(st["get"], urls)
        try:
            records = as_list(client.get(url, timeout=20).json())
        except Exception as exc:  # noqa: BLE001
            check("state", False, f"GET {url} failed: {exc}")
            continue
        label = "state " + url.split("/admin/")[-1]
        if "len" in st:
            check(label, len(records) == st["len"], f"{len(records)} records, expected {st['len']}")
        if "len_min" in st:
            check(label, len(records) >= st["len_min"], f"{len(records)} records, expected >= {st['len_min']}")
        if "len_max" in st:
            check(label, len(records) <= st["len_max"], f"{len(records)} records, expected <= {st['len_max']}")
        if "contains" in st:
            check(label, any(record_matches(r, st["contains"]) for r in records),
                  f"no record matches {st['contains']}")
        if "not_contains" in st:
            check(label, not any(record_matches(r, st["not_contains"]) for r in records),
                  f"a record matches {st['not_contains']}")
    if "judge" in exp:
        if judge is None:
            checks.append({"check": "judge", "ok": None, "detail": "skipped (run with --judge)"})
        else:
            ok, why = judge(exp["judge"], out["turns"])
            check("judge", ok, why)

    graded = [c for c in checks if c["ok"] is not None]
    out["passed"] = bool(graded) and all(c["ok"] for c in graded)
    if not graded:
        out["error"] = "this case has no checks - add an 'expect' block (or run with --judge)"
    return out


# ----------------------------------------------------------------- LLM judge
def make_judge(model: str):
    from openai import OpenAI  # only needed with --judge

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        sys.exit("--judge needs OPENROUTER_API_KEY in your environment or .env")
    llm = OpenAI(api_key=key, base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))

    def judge(criterion: str, turns: list) -> tuple[bool, str]:
        transcript = json.dumps(turns, ensure_ascii=False, indent=1)[:12000]
        msg = ("You grade one conversation with an AI agent against ONE criterion. "
               "Be strict and literal. Answer with JSON only: {\"pass\": true|false, \"reason\": \"one sentence\"}.\n\n"
               f"CRITERION: {criterion}\n\nCONVERSATION (turns, replies, proposed actions, results):\n{transcript}")
        try:
            r = llm.chat.completions.create(model=model, temperature=0,
                                            messages=[{"role": "user", "content": msg}],
                                            response_format={"type": "json_object"})
            data = json.loads(r.choices[0].message.content)
            return bool(data.get("pass")), str(data.get("reason", ""))[:300]
        except Exception as exc:  # noqa: BLE001
            return False, f"judge error: {exc}"

    return judge


# ----------------------------------------------------------------- report
def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]


def write_scorecard(results: list[dict], args, started: float) -> tuple[str, dict]:
    total = len(results)
    passed = sum(r["passed"] for r in results)
    lat = [x for r in results for x in r["latencies"]]
    cats: dict[str, list[bool]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r["passed"])
    cost = sum(r["cost_usd"] for r in results)
    summary = {
        "cases_file": str(args.cases), "runs": total, "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "turn_latency_avg_s": round(statistics.mean(lat), 2) if lat else 0.0,
        "turn_latency_p95_s": round(p95(lat), 2),
        "cost_usd_reported": round(cost, 4) if cost else None,
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(cats.items())},
        "wall_time_s": round(time.time() - started, 1),
    }
    lines = ["# Eval scorecard", "",
             f"Cases file: `{Path(args.cases).name}` · runs: **{total}** · passed: **{passed}** "
             f"(**{summary['pass_rate']:.0%}**)", "",
             f"Turn latency: avg **{summary['turn_latency_avg_s']} s**, p95 **{summary['turn_latency_p95_s']} s**"
             + (f" · reported cost: **${cost:.4f}**" if cost else " · cost: not reported by the app"), "",
             "| case | category | result | why it failed |", "|---|---|---|---|"]
    for r in results:
        why = r["error"] or "; ".join(f"{c['check']}: {c['detail']}" for c in r["checks"] if c["ok"] is False)
        lines.append(f"| {r['id']} | {r['category']} | {'✅' if r['passed'] else '❌'} | {why.replace('|', '/')[:200]} |")
    lines += ["", "| category | passed |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["by_category"].items()]
    return "\n".join(lines) + "\n", summary


# ----------------------------------------------------------------- main
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env()
    ap = argparse.ArgumentParser(description="Run eval cases against your agent's API contract.")
    ap.add_argument("--cases", type=Path, default=HERE / "cases.sample.jsonl")
    ap.add_argument("--app")
    ap.add_argument("--clinic")
    ap.add_argument("--helpdesk")
    ap.add_argument("--travel")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--category", help="run one category only")
    ap.add_argument("--repeat", type=int, default=1, help="run every case N times")
    ap.add_argument("--timeout", type=float, default=180, help="seconds per turn")
    ap.add_argument("--judge", action="store_true", help="grade 'judge' criteria with an LLM")
    ap.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "openai/gpt-4.1-mini"))
    ap.add_argument("--out", type=Path, default=HERE / "scorecard.md")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    args = ap.parse_args()

    urls = {}
    flags = {"APP": args.app, "CLINIC_API": args.clinic, "HELPDESK_API": args.helpdesk, "TRAVEL_API": args.travel}
    for name, (env, default) in DEFAULTS.items():
        urls[name] = flags[name] or os.getenv(env) or default

    if not args.cases.is_file():
        sys.exit(f"No cases file at {args.cases}")
    cases = []
    for n, line in enumerate(args.cases.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("//"):
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"{args.cases.name} line {n} is not valid JSON: {exc}")
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        cases = [c for c in cases if c.get("id") in wanted]
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
    if args.list:
        for c in cases:
            print(f"{c.get('id', '?'):8} {c.get('category', ''):16} {c.get('description', '')}")
        return 0
    if not cases:
        sys.exit("No cases selected.")

    client = httpx.Client(timeout=args.timeout)
    try:
        client.get(urls["APP"].rstrip("/") + "/health", timeout=10).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Your app is not answering at {urls['APP']}/health ({exc}).\n"
                 "Start it first (and the simulated system), or pass --app.")
    used = {name for name in urls if name != "APP" and any("{" + name + "}" in json.dumps(c) for c in cases)}
    for name in sorted(used):
        try:
            client.get(urls[name].rstrip("/") + "/health", timeout=10).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"The simulated system {name} is not answering at {urls[name]}/health ({exc}).")

    judge = make_judge(args.judge_model) if args.judge else None
    started = time.time()
    results = []
    print(f"Running {len(cases)} case(s) x {args.repeat} against {urls['APP']}\n")
    for rep in range(args.repeat):
        for case in cases:
            label = str(case.get("id")) + (f" (run {rep + 1})" if args.repeat > 1 else "")
            print(f"  {label:18} {case.get('category', ''):16} ", end="", flush=True)
            res = run_case(case, urls, client, args.cases, judge, args.verbose)
            results.append(res)
            took = sum(res["latencies"])
            verdict = ("PASS" if res["passed"] else "FAIL") + f"  ({took:.1f}s)"
            print(f"\n     = {verdict}" if args.verbose else verdict)
            if not res["passed"]:
                if res["error"]:
                    print(f"     -> {res['error']}")
                for c in res["checks"]:
                    if c["ok"] is False:
                        print(f"     -> {c['check']}: {c['detail']}")
            if args.verbose:
                for c in res["checks"]:
                    if c["ok"] is None:
                        print(f"     .  {c['check']}: {c['detail']}")

    md, summary = write_scorecard(results, args, started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nPassed {summary['passed']}/{summary['runs']} ({summary['pass_rate']:.0%}) · "
          f"turn latency avg {summary['turn_latency_avg_s']}s, p95 {summary['turn_latency_p95_s']}s")
    print(f"By category: {summary['by_category']}")
    print(f"Scorecard written to {args.out}")
    return 0 if summary["passed"] == summary["runs"] else 1


if __name__ == "__main__":
    sys.exit(main())
