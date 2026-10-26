<!-- JHF-BRAND -->
<div align="center" style="padding:28px 20px; background:#ffffff; border:2px solid #e0e0e0; border-radius:12px;">
  <p style="margin:0 0 16px 0;">
    <img src="../../assets/jhf-logo.png" alt="Jerusalem High-Tech Foundry (JHF)" height="54" style="vertical-align:middle; margin:0 22px;" />
    <img src="../../assets/comcec-logo.png" alt="COMCEC - Cooperation for Development" height="72" style="vertical-align:middle; margin:0 22px;" />
  </p>
  <h1 style="color:#1a3c5e; margin:6px 0;">Agentic AI Bootcamp</h1>
  <h3 style="color:#0078d4; margin:4px 0; font-weight:600;">The Ship Week &middot; The API contract</h3>
  <hr style="border:0; border-top:1px solid #0078d4; width:60%; margin:16px auto;" />
  <p style="font-size:14px; color:#555; margin:6px 0;">
    <strong>Lead Trainers</strong><br/>
    <a href="https://www.linkedin.com/in/alaaldin-ahmed-260266150" target="_blank">Alaaldin Ahmed</a> &nbsp;|&nbsp;
    <a href="https://il.linkedin.com/in/mohammad-abu-alhalawe" target="_blank">Mohammad Abu Alhalaweh</a>
  </p>
  <p style="font-size:12.5px; color:#777; margin:8px 0 0 0;">
    Organized by <strong>Jerusalem High-Tech Foundry (JHF)</strong> &nbsp;&middot;&nbsp; In partnership with <strong>COMCEC</strong>
  </p>
</div>

# The API contract — and the test-case format

Every team's system speaks the same small HTTP contract. Three things depend on it:
**your UI**, **your eval runner**, and **the hidden tests** we run at the end. Keep
it exactly as written: the field names, the paths, the meaning.

> Think of it as the contract a client signs with your API. Inside your system
> you may use anything (LangGraph, CrewAI, plain Python). The outside stays fixed.

---

## 1. Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/health` | — | `200 {"status": "ok"}` |
| `POST` | `/api/chat` | `{"session_id": "s-1", "message": "…", "user": {…}}` (`user` optional) | a **ChatResponse** |
| `POST` | `/api/approve` | `{"session_id": "s-1", "action_id": "act-7f3a", "approved": true}` | a **ChatResponse** |
| `POST` | `/api/voice` | multipart form: `audio` = file, `session_id` = text | a **ChatResponse** plus `transcript`, `audio_b64`, `audio_mime` — **Scenario 1 only** |

Run it on port **8000** by default (`uvicorn app.main:app --port 8000`, or your own layout).

## 2. The ChatResponse

```jsonc
{
  "session_id": "s-1",
  "reply": "What the user reads (and in Scenario 1, hears).",
  "state": "await_confirmation",            // optional: where your workflow is now
  "pending_action": {                        // ONLY when you propose a write and wait for a human
    "action_id": "act-7f3a",                 // you generate it; it is also your idempotency key
    "type": "create_booking",                // create_booking | cancel_booking | create_ticket | save_trip
    "summary": "Dental check-up, Tue 16:30, Dr. Rania Khoury, for Lina Haddad",
    "payload": { "slot_id": "SL-…", "service_id": "dental_checkup", "patient_phone": "0599000111" }
  },
  "result": { "booking_id": "BK-1042" },     // ONLY after the system of record confirmed the write
  "citations": [ { "id": "KB-004", "title": "VPN connection failures", "url": null } ],
  "tool_calls": [ { "tool": "find_slots", "ok": true, "ms": 132 } ],
  "usage": { "prompt_tokens": 1830, "completion_tokens": 212, "cost_usd": 0.0011 }   // optional; the runner sums it
}
```

| Field | Rule |
|---|---|
| `reply` | Always present. Plain text the user can read. |
| `pending_action` | `null` unless you are waiting for approval. **One** pending action at a time. The `summary` is what a human approves, so it must say exactly what will happen. |
| `result` | `null` unless a write **really** succeeded. Copy the ID from the system's response. Never generate one. |
| `citations` | Scenario 2: the KB documents behind the answer. Scenario 3: the sources behind the facts. Scenario 1: `[]` is fine. |
| `tool_calls` | Every tool your agent called in this turn: name, success, milliseconds. This is your observability and it is graded. |
| `usage` | Optional but recommended: lets your scorecard report cost per run. |

## 3. The approval gate (all scenarios)

```
user  ──▶ /api/chat ──▶ your agent decides ──▶ reply + pending_action {action_id, type, payload}
user  ──▶ /api/approve {action_id, approved: true}
                         └─▶ YOUR CODE (not the LLM) calls the write tool over MCP
                             with exactly that payload and Idempotency-Key = action_id
                         ◀── system of record: 201 {id}  → reply + result {id}
                         ◀── system of record: 409/503   → honest reply, result null
user  ──▶ /api/approve {approved: false} ──▶ nothing written, the agent offers the next step
```

- The write tool is **not** in the LLM's tool list. The LLM proposes, your code commits.
- `approved: true` for an unknown or already used `action_id` returns a clear
  reply and writes nothing again.
- After a failed write, the session continues. The user can retry or choose another option.

## 4. The simulated systems (what your agent talks to)

| | Scenario 1 | Scenario 2 | Scenario 3 |
|---|---|---|---|
| Service | Clinic API `:8101` | Helpdesk API `:8102` | Travel API `:8103` |
| MCP server | `python -m services.clinic_mcp` | `python -m services.helpdesk_mcp` | `python -m services.travel_mcp` |
| Collection (tests read it) | `GET /admin/bookings` | `GET /admin/tickets` | `GET /admin/trips` |

All three use `X-API-Key` on business endpoints and JSON errors with real HTTP
codes, and they share the **test-only** admin endpoints: `POST /admin/reset`,
`POST /admin/faults`, `DELETE /admin/faults`, `GET /admin/log`. Your **app must
not call `/admin/*`**. Those endpoints belong to the tests. Each kit's README has
the full list.

**Switch on a failure** (to test your error handling):

```powershell
# Windows PowerShell: the next create_booking call answers 503
Invoke-RestMethod -Method Post -Uri http://localhost:8101/admin/faults -ContentType 'application/json' `
  -Body '{"target": "create_booking", "mode": "error_503", "count": 1}'
```

```bash
# macOS / Linux
curl -X POST http://localhost:8101/admin/faults -H "Content-Type: application/json" \
  -d '{"target": "create_booking", "mode": "error_503", "count": 1}'
```

Modes: `error_500`, `error_503`, `timeout` (waits 15 s, then 504), `slow` (waits 3 s, then succeeds).

## 5. The test-case format (`evals/*.jsonl`)

One JSON object per line. The runner plays the `turns` against **your** API, then
checks the last response and the state of the simulated system.

```jsonc
{
  "id": "S1-03",
  "category": "conflict",
  "description": "Tuesday 17:00 is taken: offer 16:30 or 18:00 and book nothing.",
  "setup": [ {"method": "POST", "url": "{CLINIC_API}/admin/reset"} ],
  "turns": [
    {"say": "Book me a dental check-up next Tuesday at 5 PM. I'm Lina Haddad, 0599000111."}
  ],
  "expect": {
    "reply_contains_any": ["16:30", "18:00"],
    "reply_not_contains": ["BK-"],
    "pending_action": "none",
    "state": [ {"get": "{CLINIC_API}/admin/bookings", "len": 0} ]
  }
}
```

**Turns:** `{"say": "…"}` → `/api/chat` · `{"approve": true}` or `{"approve": false}`
→ `/api/approve` for the latest `pending_action` · `{"approve_again": true}` → the **same**
approval sent a second time (the double click: your system must write once, not twice) ·
`{"voice": "samples/audio/en_01.wav"}` → `/api/voice` (Scenario 1). A case may carry
`"user": {…}` (sent with every `say`).

**Checks (all optional, all must pass):**

| Check | Passes when |
|---|---|
| `reply_contains_any` / `reply_contains_all` | the last reply contains one / all of the strings (case-insensitive) |
| `reply_not_contains` | none of the strings appear |
| `reply_matches` | the regular expression matches |
| `pending_action` | `"none"`, `"any"`, or a type such as `"create_ticket"` |
| `result_has` / `result_absent` | `result` has these keys / `result` is empty |
| `citations_include` / `citations_exclude` / `citations_min` | about the ids in `citations` |
| `tools_used_include` | these tool names appear in `tool_calls` during the case |
| `max_latency_s` | the slowest turn took at most this many seconds |
| `state` | `GET` an admin collection: `len`, `len_min`, `len_max`, `contains` / `not_contains` (`{"field": "value"}`, dotted keys like `patient.phone`, text matches by substring) |
| `judge` | an LLM grades this sentence against the conversation, only with `--judge` |

**Categories** (use them, the scorecard groups by them): `happy_path`,
`clarification`, `conflict`, `rejection`, `tool_failure`, `guardrail`, `memory`,
`grounding`, `policy_conflict`, `unknown`, `revision`, `budget`, `voice`.

**Run it:**

```powershell
python evals/run_evals.py                             # the sample cases
python evals/run_evals.py --cases evals/my_cases.jsonl
python evals/run_evals.py --only S1-03 -v             # one case, every turn printed
python evals/run_evals.py --repeat 3                  # is it stable?
python evals/run_evals.py --judge                     # also the LLM-judged criteria
```

It writes `evals/scorecard.md` (paste it into your README) and `evals/scorecard.json`.

> **Write cases that a wrong system fails.** "The reply is not empty" proves
> nothing. "No booking exists after a rejection" proves a lot. Prefer checks on
> the **system state** over checks on wording.

---

<div align="center" style="padding:14px; border-top:2px solid #0078d4; margin-top:34px;">
  <p style="margin:0 0 8px 0;">
    <img src="../../assets/jhf-logo.png" alt="JHF" height="28" style="vertical-align:middle; margin:0 14px; background:#ffffff; padding:6px 10px; border-radius:6px;" />
    <img src="../../assets/comcec-logo.png" alt="COMCEC" height="40" style="vertical-align:middle; margin:0 14px; background:#ffffff; padding:6px 10px; border-radius:6px;" />
  </p>
  <p style="color:#888; font-size:13px; margin:0;">
    <strong>JHF Agentic AI Bootcamp</strong> &mdash; The Ship Week · The API contract<br/>
    Lead Trainers: <a href="https://www.linkedin.com/in/alaaldin-ahmed-260266150">Alaaldin Ahmed</a> &amp; <a href="https://il.linkedin.com/in/mohammad-abu-alhalawe">Mohammad Abu Alhalaweh</a><br/>
    Organized by Jerusalem High-Tech Foundry (JHF) &middot; In partnership with COMCEC
  </p>
</div>
