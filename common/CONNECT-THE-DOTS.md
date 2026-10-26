<!-- JHF-BRAND -->
<div align="center" style="padding:28px 20px; background:#ffffff; border:2px solid #e0e0e0; border-radius:12px;">
  <p style="margin:0 0 16px 0;">
    <img src="../../assets/jhf-logo.png" alt="Jerusalem High-Tech Foundry (JHF)" height="54" style="vertical-align:middle; margin:0 22px;" />
    <img src="../../assets/comcec-logo.png" alt="COMCEC - Cooperation for Development" height="72" style="vertical-align:middle; margin:0 22px;" />
  </p>
  <h1 style="color:#1a3c5e; margin:6px 0;">Agentic AI Bootcamp</h1>
  <h3 style="color:#0078d4; margin:4px 0; font-weight:600;">The Ship Week &middot; Connect the dots</h3>
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

# Connect the dots — every module, one system

You didn't take thirteen separate courses. You collected the parts of **one**
machine. This week you assemble it. For every layer of your system: the module that
taught it, the file worth reopening, and what it becomes in each brief.

<p align="center"><img src="../visuals/connect-the-dots.png" alt="Modules mapped to the layers of a real agentic system" width="900"/></p>

## The eight layers of a real agentic system

| # | Layer | The question it answers | Module | Reopen this file |
|:--:|---|---|:--:|---|
| 1 | **Interface** | How does a person use it? | — | your brief's `kit/ui/app_ui.py` (a starter you can replace) |
| 2 | **Entry point** | How does anyone (a UI, a test, another team) call it? | M12 | `Module-12-Deployment/solution-code/app.py` · [CONTRACT.md](CONTRACT.md) |
| 3 | **Orchestration** | Who does what, in which order, until when? | M1 · M2 · M3 · M7 | `Module-01/solution-code/agent_loop_solution.py` · `Module-02/starter-code/hello_langgraph_starter.py` · `Module-07-Multi-Agent-Systems/ex1-the-team/team_solution.py` · `ex2-the-manager/manager_solution.py` |
| 4 | **Tools** | How does it act on the world, safely and swappably? | M5 | `Module-05/lab-2-agent-over-mcp/mcp_client_agent_SOLUTION.py` · `Module-05/homework/helpdesk_server.py` |
| 5 | **Knowledge & memory** | What does it know, and what does it remember about you? | M6 · M13 | `Module-06-Memory-Systems/shared/memory_store.py` · `solution-code/memory_agent_solution.py` |
| 6 | **Guardrails & approvals** | What must it never do, and who says yes? | M11 · M4 | `Module-11-Guardrails-Safety-and-Cost/shared/checks.py` · `shared/redteam.py` · `Module-04-Reflection-and-Robustness/solution-code/reflection_agent_solution.py` |
| 7 | **Evaluation & observability** | How do we know it works, and why it failed? | M10 | `Module-10-Evaluation-and-Observability/solution-code/eval_harness_solution.py` · `shared/tracing.py` · `common/evals/run_evals.py` |
| 8 | **Deployment** | Can someone else start it with one command? | M12 · M13 | `Module-12-Deployment/README.md` (Dockerfile, pins) · Module 13 if you want a cloud twin |

## What each layer becomes in your brief

| Layer | 1 · Voice Receptionist | 2 · IT Helpdesk | 3 · Trip Planning Crew |
|---|---|---|---|
| **Interface** | microphone + transcript + approval card + spoken reply | chat + citations panel + ticket-draft card | trip form + itinerary table + approval card |
| **Orchestration** | a LangGraph **state machine**: collect → check → propose → await approval → book | **triage → retrieve → answer → verify → escalate** | a CrewAI **crew**: coordinator, researcher, budget analyst, itinerary planner, verifier |
| **Roles (≥ 2)** | e.g. intake + scheduler (+ a confirm-back check) | e.g. triage + resolver + escalation writer | at least 3 agents with visible responsibilities |
| **Tools via MCP** | `clinic_mcp`: `find_slots`, `get_patient`, `create_booking`… | `helpdesk_mcp`: `lookup_employee`, `get_service_status`, `create_ticket`… | `travel_mcp`: `find_attractions`, `get_opening_hours`, `travel_time`, `save_trip`… |
| **Real API** | speech-to-text (OpenRouter) + text-to-speech | embeddings (OpenRouter) | Wikivoyage, OpenStreetMap, Open-Meteo, Frankfurter, OSRM |
| **Short-term memory** | the details collected so far, the proposed slots | the troubleshooting so far, what the user tried | the request, the current plan, the requested changes |
| **Persistent memory** | the returning patient's profile (`get_patient` / `save_patient`) | the requester's ticket history | the traveller's saved preferences and trips |
| **Knowledge** | clinic info and policies (`clinic_info`) | **the 20-document knowledge base, your retrieval** | Wikivoyage listings with sources |
| **Approval gate** | `create_booking`, `cancel_booking` | `create_ticket` | `save_trip` |
| **The trap** | a request to diagnose; a closed day; a taken slot; an instruction hidden in speech | an outdated policy; a poisoned wiki page; an answer that is not in the KB | a closed museum; two prices for one ticket; an instruction inside a listing |
| **Reflection (M4)** | read the slot back before proposing it | check every step is supported by a cited document | the verifier agent checks hours, travel times and the budget |
| **Evaluation** | the sample cases + yours, including a voice case | the sample cases + yours, including the citations | the sample cases + yours, in replay mode |

## Patterns from the course you'll want

| Pattern | Where you met it | Use it for |
|---|---|---|
| **Step and cost caps** | M1, M11 | every loop and every crew (`max_iter`, a turn budget) |
| **Structured outputs** (Pydantic) | M2, M7 | the proposed action, the ticket draft, the itinerary: never parse free text |
| **Manager + specialists** | M7 exercise 2 | the coordinator that routes and merges |
| **Untrusted content** | M5, M7, M11 | everything retrieved, scraped or transcribed is data, never instructions |
| **Idempotency** | M12 | `Idempotency-Key = action_id` on every write |
| **Fail loudly, recover politely** | M11 | tool errors become honest replies, never invented results |
| **Measure, change one thing, re-measure** | M10 | your "one failure we fixed" |

> **A2A is optional this week.** Use it only if a real boundary exists (for example,
> your verifier is a separate service another team could call). Otherwise keep
> your roles in one process and spend the time on evaluation.

---

<div align="center" style="padding:14px; border-top:2px solid #0078d4; margin-top:34px;">
  <p style="margin:0 0 8px 0;">
    <img src="../../assets/jhf-logo.png" alt="JHF" height="28" style="vertical-align:middle; margin:0 14px; background:#ffffff; padding:6px 10px; border-radius:6px;" />
    <img src="../../assets/comcec-logo.png" alt="COMCEC" height="40" style="vertical-align:middle; margin:0 14px; background:#ffffff; padding:6px 10px; border-radius:6px;" />
  </p>
  <p style="color:#888; font-size:13px; margin:0;">
    <strong>JHF Agentic AI Bootcamp</strong> &mdash; The Ship Week · Connect the dots<br/>
    Lead Trainers: <a href="https://www.linkedin.com/in/alaaldin-ahmed-260266150">Alaaldin Ahmed</a> &amp; <a href="https://il.linkedin.com/in/mohammad-abu-alhalawe">Mohammad Abu Alhalaweh</a><br/>
    Organized by Jerusalem High-Tech Foundry (JHF) &middot; In partnership with COMCEC
  </p>
</div>
