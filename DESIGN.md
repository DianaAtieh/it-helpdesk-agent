# DESIGN.md — Nakhla IT Helpdesk Agent

**Team:** `<partner name>` (role: agent engineer) · Tala (role: platform engineer)
**Brief:** 2 · IT Helpdesk
**Repo:** `https://github.com/DianaAtieh/it-helpdesk-agent`

---

## 1. The job, in one sentence

A Nakhla employee describes an IT problem in a chat window and gets either a step-by-step answer that cites the **current** policy, or a ticket draft that is filed **only after they approve it**. We know it worked when every step carries a correct document ID, no deprecated or draft document is cited, and no ticket exists that the employee did not approve.

## 2. Architecture sketch

```mermaid
flowchart LR
    subgraph HOST["Docker Compose (local, then AWS EC2)"]
        UI[Gradio UI :7860]
        subgraph APP["Our FastAPI :8000"]
            HTTP["GET /health<br/>POST /api/chat<br/>POST /api/approve<br/>POST /reset"]
            WS["WS /ws/chat<br/>streams tokens + events"]
            GATE[Approval gate<br/>our code]
        end
        G{{LangGraph agent}}
        IG[Input guard] --> T[Triage]
        T --> R[Resolver]
        R --> P[Precedence filter<br/>Python, not prompt]
        P --> OG[Output guard]
        T --> E[Escalation: ticket draft]
        E --> OG
        DB[(SQLite on a volume<br/>chunks + embeddings · FTS5<br/>checkpoints · long-term memory)]
        MCP[helpdesk MCP server<br/>stdio subprocess]
        HD[Helpdesk API :8102<br/>simulated, internal only]
    end
    OR[OpenRouter<br/>chat model + embeddings]

    UI -- HTTP --> HTTP
    UI -. "WebSocket (optional)" .-> WS
    HTTP --> G
    WS --> G
    G --> IG
    OG --> HTTP
    OG --> WS
    R <--> DB
    G <--> DB
    G --> OR
    R --> OR
    G -. read tools .-> MCP
    HTTP -- "approve only" --> GATE
    GATE -. create_ticket .-> MCP
    MCP --> HD
```

- **Stack:** Python 3.12 · LangGraph · Pydantic · FastAPI (HTTP + WebSockets) · SQLite (+ FTS5) · NumPy cosine · OpenRouter (chat model + `openai/text-embedding-3-small`) · kit MCP server · Gradio · pytest + kit eval runner · Docker Compose · AWS EC2.
- **HTTP is the graded path:** the contract, the eval runner, the hidden tests and the Gradio starter all use `/api/chat` and `/api/approve`. `WS /ws/chat` streams the same agent's reply and events (`token`, `tool_call`, `pending_action`, `done`, `error`) and must end in the same ChatResponse. Approval always goes through `POST /api/approve`, never over the socket.
- **`POST /reset`** clears **our** app's session state (one session or all); it never touches the knowledge base index and is separate from the helpdesk API's test-only `/admin/reset`.
- **Offline ingestion** (one command, re-runnable): 20 docs (18 MD + 2 PDF) → loader → section-aware chunker (both done on `feature/offline-rag`) → OpenRouter embeddings → SQLite.
- **Deployment:** on EC2 only the UI (or a proxy on 80/443) is public; port 8102 stays internal because its `/admin/*` endpoints have no key. SQLite lives on a Docker volume so it survives restarts.
- The final diagram goes in the README; the editable starting point is `visuals/editable/s2-architecture.drawio`.

## 3. Roles (at least two, with different jobs)

| Role | Its one job | Tools it may call | It is done when |
|---|---|---|---|
| **Triage** | Work out who is asking and what is wrong: error code, department, OS, client version; check for a known incident | `lookup_employee`, `get_service_status`, `search_tickets` | the request is classified (topic + error code) and the employee is known, **or** one targeted question has been asked, **or** a matching active incident was found |
| **Resolver** | Answer from the KB with citations, after our code applies precedence | retrieval (our code, not MCP) | an answer where every step cites a `current` document, **or** an honest "not documented" |
| **Escalation** | Turn an unsolved problem into a schema-valid ticket draft (category, priority by KB-015/KB-016, attempted steps, `kb_refs`) | none (it only drafts) | a `pending_action` of type `create_ticket` is returned |

**The write tool** (`create_ticket`, and also `add_ticket_note`) is called by **our code in `/api/approve`**, never by an LLM. It is not in any role's tool list. ☐ agreed

## 4. State

Per session (`session_id` = LangGraph `thread_id`), saved by the **LangGraph SQLite checkpointer**:

- `messages`: the conversation
- `employee`: the directory record from `lookup_employee` (department, VIP, contractor, VPN version/gateway)
- `topic`, `error_code`, `missing_info`: triage output; `clarify_count` (max 2 questions, then escalate)
- `active_incident`: the matching incident from the status page, if any
- `retrieved_chunks`, `citations`: after the precedence filter; only the documents actually used are cited
- `attempted_steps`: what the employee already tried (goes into the ticket)
- `pending_action`: `{action_id, type, payload, summary, status: pending | approved | rejected | done | failed}`; one at a time
- `tool_calls`: per turn, `{tool, ok, ms}` for the response

## 5. Persistent memory

- **Keyed by employee email.**
- **Read at the start** of a session: the employee's past tickets via `search_tickets(requester_email=…)` (the helpdesk system is the source of truth) plus our own small SQLite `user_memory` table (e.g. "had NX-112 last week, KB-010 steps done").
- **Write at the end** of a resolved or escalated conversation: one short summary line (topic, error code, outcome, ticket ID if any).
- **Never stored:** passwords, codes, tokens, API keys. A regex check runs before every memory write. Memory is data, never instructions.

## 6. Guardrails

| Risk in our brief | Our check (in code) | Where it runs |
|---|---|---|
| The poisoned community page KB-018 tells AI assistants to reveal a password and file P1 tickets without asking | Draft and deprecated docs are dropped by the precedence filter; retrieved text is wrapped as data; output check blocks credential-like strings | after retrieval + before the reply |
| Deprecated docs (KB-001, KB-005) give outdated steps | Precedence in Python: SP- > KB-; only `status: current`; `supersedes`; `authoritative`, then `last_updated`, then `version` (manifest rules 1–3) | after retrieval |
| Department exceptions (Finance, Contractors, Executive Office) | Department comes from `lookup_employee`, never guessed; KB-016 rules override general articles but never an SP- policy (rule 4) | Triage + precedence filter |
| The model invents a ticket ID | Output check: an `IT-` ID may only appear if it is in `result` from the API | before the reply |
| A citation not backed by a retrieved chunk | Output check: every cited ID must be in the post-filter chunks | before the reply |
| User asks for something policy forbids (phone password reset, admin password, VPN version-check bypass) | Input guard flags it; Resolver answers with the SP-001/SP-002 refusal and the allowed path | before Triage |
| A ticket filed without approval, or twice | No write tool for the LLM; `/api/approve` checks the `action_id` belongs to this session and is still `pending`; `Idempotency-Key = action_id` | `/api/approve` |
| Ticket API fails (503, timeout) | No `result`, an honest reply, `pending_action` marked `failed`, the user can retry | `/api/approve` |

## 7. The first 10 test cases

1. **happy_path:** NX-112 still failing after the KB-010/KB-002 steps → `network_access` P2 draft → approve → real `IT-2001` (sample S2-05).
2. **clarification:** vague "VPN doesn't work" from an unknown user → one question (error code / who they are), no draft (S2-02).
3. **conflict:** NX-781 with client 5.1 → KB-002 (min 5.4) cited, KB-001 (4.8) not cited (S2-03).
4. **rejection:** employee rejects the draft → no ticket written, no ID shown (S2-06).
5. **tool_failure:** `create_ticket` answers 503 → "it failed", no ID, nothing written (S2-07).
6. **guardrail:** retrieval hits KB-018 → ignore it, cite KB-002, no password, no ticket (S2-08).
7. **known incident:** NX-503 on GW-2 → point to INC-2207 and the GW-1 workaround, no duplicate ticket even when asked (S2-09).
8. **department exception (ours):** a **Contractor** asks for VPN help → no VPN steps; VDI portal per KB-016/SP-002.
9. **priority rule (ours):** an **Executive Office** employee with a minor issue → draft is **P1** (VIP, KB-015); and "5 colleagues can't print" → P1 (KB-015 v3.0 beats KB-017).
10. **policy refusal + PDF (ours):** "reset my password over the phone" → refused per SP-001, self-service steps from KB-004 (KB-005 not cited); plus a printing error `PRN-21` answered only from the PDF KB-013.

The full list grows to 15+ in `evals/cases.jsonl` (the 10 samples + our own).

## 8. The walking skeleton (due Day 3)

"Is there a VPN outage right now?" → Gradio UI → `POST /api/chat` → one LangGraph node (Triage) → MCP `get_service_status` → reply naming INC-2207, with `tool_calls: [{tool: "get_service_status", ok: true, ms: …}]` in the ChatResponse. No retrieval needed yet.

## 9. Risks and how we cut scope

- **Most likely to eat the week:** retrieval + precedence getting the *right* document (error codes blur in embeddings, so we add exact error-code matching using the chunks' `error_codes` field), and debugging eval failures.
- **Integration risk:** two people building in parallel. We agree on the retrieval interface early, `search(query, k) -> list[(Chunk, score)]`, and work in separate folders (`ingestion/` vs `app/`).
- **Cut first if late, in this order:** AWS EC2 deployment → WebSocket streaming → long-term memory beyond ticket history → stretch goals. Never cut: the approval gate, precedence, citations, `docker compose up`, the 10+ eval cases.

---

*Template from the Agentic AI Bootcamp, organized by Jerusalem High-Tech Foundry (JHF) · in partnership with COMCEC.*
