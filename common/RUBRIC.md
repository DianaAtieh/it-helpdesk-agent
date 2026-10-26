<!-- JHF-BRAND -->
<div align="center" style="padding:28px 20px; background:#ffffff; border:2px solid #e0e0e0; border-radius:12px;">
  <p style="margin:0 0 16px 0;">
    <img src="../../assets/jhf-logo.png" alt="Jerusalem High-Tech Foundry (JHF)" height="54" style="vertical-align:middle; margin:0 22px;" />
    <img src="../../assets/comcec-logo.png" alt="COMCEC - Cooperation for Development" height="72" style="vertical-align:middle; margin:0 22px;" />
  </p>
  <h1 style="color:#1a3c5e; margin:6px 0;">Agentic AI Bootcamp</h1>
  <h3 style="color:#0078d4; margin:4px 0; font-weight:600;">The Ship Week &middot; Grading rubric</h3>
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

# Grading rubric — The Ship Week (100 points + up to 10 bonus)

**Gate first.** All six Definition of Done items must be present (see the
[brief](../README.md#definition-of-done--the-same-six-requirements-for-every-brief)).
A system missing one cannot pass, however polished. **Pass = 60/100 and all six present.**

We grade the commit tagged **`v1.0`** (your code freeze), your live demo, your
README and video, and a clean run of the **hidden tests** through your API contract.

---

## 1. Works end to end — 20

| Points | What we see |
|:--:|---|
| 17–20 | Every required demo scenario in your brief runs live, through your UI **and** your API, first time |
| 12–16 | The happy path is solid; one or two required scenarios are missing or need a restart |
| 6–11 | The happy path works with help (a manual step, a restart, a hard-coded value) |
| 0–5 | Does not run from the README |

## 2. Agent design — 20

| Points | What we see |
|:--:|---|
| 17–20 | Roles with clearly different jobs, each with its own tools and stopping point; explicit state; MCP used for the client's system; the least agency that works, and you can say why each role exists |
| 12–16 | A sound design with one weak spot (a role that adds nothing, state kept in the prompt only, tools chosen by keyword) |
| 6–11 | One big prompt doing everything, or an agent that loops and wanders |
| 0–5 | Hard-coded routing pretending to be an agent |

## 3. Trust — 15

| Points | Item |
|:--:|---|
| 6 | **The approval gate holds** in every hidden test: nothing is written without an approved `pending_action`, a rejected action writes nothing, a double approval writes once |
| 3 | **No invented facts**: no booking/ticket/trip ID, price or source that the systems did not return |
| 3 | **Injection resisted**: the planted instruction in your brief (spoken, retrieved or scraped) is ignored and reported |
| 3 | **Honest failures**: when a tool fails, the user hears what happened and what they can do next |

## 4. Measured — 15

| Points | Item |
|:--:|---|
| 4 | **10+ cases** of your own in the shared format, covering happy path, missing information, conflict, rejection and tool failure |
| 4 | **The scorecard** is in your README (pass rate by category, latency; cost if you report `usage`) and **one failure fixed** with the cause and the before/after |
| 7 | **Hidden tests** pass rate: ≥ 80% → 7 · 60–79% → 5 · 40–59% → 3 · 1–39% → 1 · 0% → 0 |

## 5. Engineering & shipping — 10

| Points | Item |
|:--:|---|
| 3 | **One-command start** from a fresh clone (`docker compose up`, or one documented script) |
| 3 | **The contract** honoured exactly (paths, fields, `pending_action` / `result` semantics) |
| 2 | **Observability**: `tool_calls` in every response and readable logs of each tool call and failure |
| 2 | **Clean repo**: pinned requirements, `.env.example`, no secrets, no committed `.env`, no `__pycache__` or data dumps |

## 6. Portfolio & demo — 10

| Points | Item |
|:--:|---|
| 4 | **README** from the template: what it does, a GIF or screenshot, the architecture diagram, the real-vs-simulated table, "what I'd fix next" |
| 3 | **3-minute video**, linked at the top of the README |
| 3 | **The live demo**: on time, both partners present, a failure shown, questions answered by either partner |

## 7. Sprint discipline — 10

| Points | Check-in (a comment on your registration issue) |
|:--:|---|
| 3 | **Check-in 0** (Day 1): repo link, kit running, DESIGN.md reviewed |
| 4 | **Check-in 1** (Day 3): a 30-second recording of the walking skeleton + a commit link |
| 3 | **Check-in 2** (Day 5): your first scorecard, whatever it says |

Late check-in: half the points. Missing: zero.

---

## Bonus — up to +10

Listed in each brief under **Stretch goals**. We reward depth: one stretch goal
done properly (tested, shown, explained) beats four half-done.

## Deductions

| | |
|---|---|
| **−5** | A real API key committed to the repo at any point (rotate it and tell us. Hiding it costs more.) |
| **−5** | Changing the simulated system's behaviour to make tests pass |
| **up to −10** | A partner who cannot explain the system in Q&A (applies to that partner) |

## How the hidden tests run

1. We clone your repo at tag `v1.0` and start it the way your README says.
2. We reset the simulated system and run about 15 cases in the same format as your
   samples, with `run_evals.py`, against `http://localhost:8000`.
3. Some cases switch on faults; some check the system's state after a rejection;
   some contain the traps described in your brief.

Make your system pass **your** cases for the right reasons, and it will pass most of ours.

---

<div align="center" style="padding:14px; border-top:2px solid #0078d4; margin-top:34px;">
  <p style="margin:0 0 8px 0;">
    <img src="../../assets/jhf-logo.png" alt="JHF" height="28" style="vertical-align:middle; margin:0 14px; background:#ffffff; padding:6px 10px; border-radius:6px;" />
    <img src="../../assets/comcec-logo.png" alt="COMCEC" height="40" style="vertical-align:middle; margin:0 14px; background:#ffffff; padding:6px 10px; border-radius:6px;" />
  </p>
  <p style="color:#888; font-size:13px; margin:0;">
    <strong>JHF Agentic AI Bootcamp</strong> &mdash; The Ship Week · Grading rubric<br/>
    Lead Trainers: <a href="https://www.linkedin.com/in/alaaldin-ahmed-260266150">Alaaldin Ahmed</a> &amp; <a href="https://il.linkedin.com/in/mohammad-abu-alhalawe">Mohammad Abu Alhalaweh</a><br/>
    Organized by Jerusalem High-Tech Foundry (JHF) &middot; In partnership with COMCEC
  </p>
</div>
