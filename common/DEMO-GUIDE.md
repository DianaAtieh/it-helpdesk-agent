<!-- JHF-BRAND -->
<div align="center" style="padding:28px 20px; background:#ffffff; border:2px solid #e0e0e0; border-radius:12px;">
  <p style="margin:0 0 16px 0;">
    <img src="../../assets/jhf-logo.png" alt="Jerusalem High-Tech Foundry (JHF)" height="54" style="vertical-align:middle; margin:0 22px;" />
    <img src="../../assets/comcec-logo.png" alt="COMCEC - Cooperation for Development" height="72" style="vertical-align:middle; margin:0 22px;" />
  </p>
  <h1 style="color:#1a3c5e; margin:6px 0;">Agentic AI Bootcamp</h1>
  <h3 style="color:#0078d4; margin:4px 0; font-weight:600;">The Ship Week &middot; The demo and the video</h3>
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

# The demo — 6 minutes live, 3 minutes on video

You are showing a product to an engineering manager who has never seen it. You are
not telling the story of your week.

## The 6-minute live demo (+ 2 minutes Q&A)

| Time | Who | What | Tip |
|---|---|---|---|
| 0:00–0:45 | Partner A | **The problem and the user.** One sentence each. Then the architecture diagram: point at the roles, the MCP server, memory, the approval gate. | Start with the user, not the framework. |
| 0:45–2:45 | Partner A | **The happy path, live, through your UI.** Show the proposed action, approve it, show the ID that came back, then show it in the system (`/admin/<collection>` or the kit's log). | Use a script you rehearsed. Type the request beforehand and paste it. |
| 2:45–4:15 | Partner B | **Two things going wrong**, chosen from your brief's demo list: a conflict or missing information, **and** a tool failure (switch a fault on live with one command). Show that nothing false was written and the user got an honest answer. | This is where trust is earned. Most teams skip it. Don't. |
| 4:15–5:15 | Partner B | **The scorecard.** The pass rate, the categories, and the **one failure you fixed**: the cause, the change, the before/after. | Show the file, not a slide. |
| 5:15–6:00 | Both | **What's real and what's simulated, and what you'd fix next.** | Honest limitations score points. |

**Q&A (2 minutes).** Expect: *"Why two agents and not one?"*, *"What stops it
from booking without approval?"*, *"Show me where the guardrail is in the code."*,
*"What does a run cost?"* Each partner must be able to answer about the whole system.

**Before you present:**
- [ ] `docker compose up` from a fresh clone works on the laptop you present from
- [ ] The simulated system is reset (`POST /admin/reset`) and faults are cleared
- [ ] Font size 150% in the terminal and the browser; notifications off
- [ ] A **backup**: your 3-minute video, ready to play if the Wi-Fi dies
- [ ] The fault command for the failure demo is in a text file, ready to paste

## The 3-minute video (for your portfolio)

Recorded once, linked at the top of your README. People who never meet you will watch it.

1. **0:00–0:20** — the problem and who it's for (your face or your voice, then the UI).
2. **0:20–1:50** — the happy path, end to end, including the approval card.
3. **1:50–2:30** — one failure handled well (conflict, rejection or outage).
4. **2:30–3:00** — the architecture diagram and the scorecard, then the repo link.

Tools: OBS Studio, the Xbox Game Bar (`Win+G`), Clipchamp, or a Teams meeting you
record by yourself. Upload it to YouTube as **unlisted** (or to your repo's
release) and put the link in the README. Keep it under 3:00. Nobody watches minute four.

---

<div align="center" style="padding:14px; border-top:2px solid #0078d4; margin-top:34px;">
  <p style="margin:0 0 8px 0;">
    <img src="../../assets/jhf-logo.png" alt="JHF" height="28" style="vertical-align:middle; margin:0 14px; background:#ffffff; padding:6px 10px; border-radius:6px;" />
    <img src="../../assets/comcec-logo.png" alt="COMCEC" height="40" style="vertical-align:middle; margin:0 14px; background:#ffffff; padding:6px 10px; border-radius:6px;" />
  </p>
  <p style="color:#888; font-size:13px; margin:0;">
    <strong>JHF Agentic AI Bootcamp</strong> &mdash; The Ship Week · The demo and the video<br/>
    Lead Trainers: <a href="https://www.linkedin.com/in/alaaldin-ahmed-260266150">Alaaldin Ahmed</a> &amp; <a href="https://il.linkedin.com/in/mohammad-abu-alhalawe">Mohammad Abu Alhalaweh</a><br/>
    Organized by Jerusalem High-Tech Foundry (JHF) &middot; In partnership with COMCEC
  </p>
</div>
