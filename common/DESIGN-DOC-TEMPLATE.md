# DESIGN.md — one page, written on Day 1

> Copy this file to the root of your repo as `DESIGN.md` and fill it in **before**
> you write agent code. Bullet points are fine. Show it to your Lead Trainer at the
> kickoff session; that review is Check-in 0.
> A good design doc is short, specific, and full of decisions.

**Team:** `<name 1>` (role: agent engineer) · `<name 2>` (role: platform engineer)
**Brief:** `1 Voice Receptionist | 2 IT Helpdesk | 3 Trip Planning Crew`
**Repo:** `https://github.com/<owner>/<product-name>`

---

## 1. The job, in one sentence

> *Who* uses it, to *do what*, and how *we know it worked*.
> Example: "A patient books a clinic appointment by voice in under two minutes,
> and no booking is ever made that the patient did not approve."

## 2. Architecture sketch

Draw it (a photo of paper is fine for Day 1; the final one goes in your README).
It must show: **UI → your API → agents/roles → MCP server → the client's system**,
plus **memory**, **the approval gate**, and **where the guardrails sit**.
Start from your brief's blueprint in `Final-Mini-Project/visuals/` if it helps.

<!-- Save the photo or export as docs/architecture-day1.png, then delete these comment marks:
![sketch](docs/architecture-day1.png)
-->

## 3. Roles (at least two, with different jobs)

| Role | Its one job | Tools it may call | It is done when |
|---|---|---|---|
| e.g. Intake | collect service, day, time, name, phone | `list_services`, `get_patient` | all required fields known |
| e.g. Scheduler | find and propose real slots | `find_slots` | 1–3 real slots proposed |
| | | | |

**The write tool** (`create_booking` / `create_ticket` / `save_trip`) is called by
**our code in `/api/approve`**, never by an LLM. ☐ agreed

## 4. State

What does a session remember between turns? List the fields
(e.g. `service_id, date, time_window, patient, proposed_slots, pending_action`).
Where does it live (memory dict, SQLite, LangGraph checkpointer)?

## 5. Persistent memory

What do we remember **across** sessions, keyed by what (phone, email, traveller id)?
When do we write it, and when do we read it?

## 6. Guardrails

| Risk in our brief | Our check (in code) | Where it runs |
|---|---|---|
| e.g. the model invents a booking ID | an output check: an ID only appears if it came from `result` | before the reply |
| e.g. instructions hidden in retrieved text | strip/flag, never follow | after retrieval |
| | | |

## 7. The first 10 test cases

One line each: the category and the one thing it proves. At least one each of
**happy path, missing information, conflict, rejection, tool failure**.

1. happy_path:
2. clarification:
3. conflict:
4. rejection:
5. tool_failure:
6. 
7. 
8. 
9. 
10. 

## 8. The walking skeleton (due Day 3)

The thinnest path that goes all the way through: *which* single request, through
*which* role, calling *which* MCP tool, returning *what*?

## 9. Risks and how we cut scope

The thing most likely to eat the week, and what we drop first if we are late.
(Stretch goals go **last**.)

---

*Template from the Agentic AI Bootcamp, organized by Jerusalem High-Tech Foundry (JHF) · in partnership with COMCEC.*
