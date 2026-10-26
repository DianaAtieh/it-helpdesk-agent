<!--
  PORTFOLIO README TEMPLATE — copy this to README.md at the root of your repo.
  Replace everything in <angle brackets>. Delete these comments.
  Rule of thumb: a recruiter reads the first screen only. Make it count:
  what it does, a GIF or screenshot, and a "run it" block that works.
-->

# <Product name> — <one line: what it does, for whom>

<!-- A short GIF (10–20 s) of the happy path. Record with ScreenToGif, OBS or the Xbox Game Bar (Win+G),
     save it as docs/demo.gif, then delete these comment marks so the line below shows:
![demo](docs/demo.gif)
-->

**<One paragraph: the problem, who has it, and what this agent does about it. No buzzwords.>**

▶ **3-minute demo video:** <link> · 🧪 **Eval pass rate:** <x/y cases> · 🧰 **Stack:** <LangGraph | CrewAI> · MCP · FastAPI · <your model>

---

## What it does

- <capability 1 — a verb and a result: "Books appointments by voice, in English or Arabic">
- <capability 2>
- <capability 3>
- **Asks before it acts:** every <booking | ticket | saved trip> is proposed first and written only after you approve it.

## Architecture

<!-- Export your diagram to docs/architecture.png (start from an editable file in the course repo:
     Final-Mini-Project/visuals/editable/, open it at https://app.diagrams.net), then delete these comment marks:
![architecture](docs/architecture.png)
-->

<3–6 bullets: the roles and what each owns · how the agents reach the client's system (MCP) · where memory lives · where the guardrails and the approval gate sit.>

| Component | Real or simulated? | Notes |
|---|---|---|
| LLM (<model>) via OpenRouter | **Real** | |
| <Speech-to-text / embeddings / travel data> | **Real** | <provider> |
| <Clinic calendar / ticket system / trip store> | **Simulated** | a REST API + MCP server with fault injection, provided by the bootcamp kit |
| <anything else> | | |

## Run it

```bash
git clone https://github.com/<owner>/<repo>.git && cd <repo>
cp .env.example .env          # add your OPENROUTER_API_KEY
docker compose up --build     # the simulated system + this app
# open http://localhost:<port>
```

<Or the local, non-Docker steps: venv, pip install, start the service, start the app.>

## How it behaves — the approval gate

<Two or three sentences and, ideally, a screenshot: the proposed action card, Approve / Reject.>

## Evaluation

<Paste evals/scorecard.md here, or the summary and a link to it.>

| Category | Passed |
|---|---|
| happy_path | x/y |
| … | |

**The failure we fixed:** <case id> failed because <cause>. We changed <what>. Before → after: <x/y → x/y>.

## Guardrails and safety

- <check 1: what it stops, where it runs>
- <check 2>
- <what happens when the <calendar | ticket API | data source> is down>

## Cost and latency

<Average turn latency, p95, tokens or dollars per conversation, from your scorecard.>

## What I'd fix next

- <honest limitation 1>
- <honest limitation 2>

## Team

| | Built |
|---|---|
| **<Name 1>** ([GitHub](<link>) · [LinkedIn](<link>)) | <what you owned> |
| **<Name 2>** ([GitHub](<link>) · [LinkedIn](<link>)) | <what you owned> |

## Acknowledgements

Built during the **Agentic AI Bootcamp** — organized by Jerusalem High-Tech Foundry (JHF), in partnership with COMCEC.
<Data attribution lines, if you used real data — copy them from the kit README.>
