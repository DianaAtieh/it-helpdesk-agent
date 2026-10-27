"""Gradio UI starter for the IT Helpdesk Agent - a small client of YOUR API contract.

It never talks to the helpdesk API or the LLM directly. It only calls your app:
    POST {APP_URL}/api/chat      {"session_id", "message", "user": {"email"}}
    POST {APP_URL}/api/approve   {"session_id", "action_id", "approved"}
Run your app first (port 8000), then:   python ui/app_ui.py   ->  http://localhost:7860
Replace it with your own UI if you like; keep the contract.
"""
from __future__ import annotations

import os
import uuid

import gradio as gr
import httpx

APP_URL = os.getenv("APP_URL", "http://localhost:8000").rstrip("/")
NO_CARD = "_No ticket draft waiting for approval._"

# ---------------------------------------------------------------- palm theme
# Nakhla means "palm tree": frond greens, trunk browns and a warm sand background.
PALM = gr.themes.Color(
    name="palm", c50="#f1f7ee", c100="#dcebd3", c200="#bcd9ab", c300="#93c07a", c400="#6aa64f",
    c500="#4d8b34", c600="#3b7028", c700="#2f5a21", c800="#26481c", c900="#1e3a17", c950="#11220c")
TRUNK = gr.themes.Color(
    name="trunk", c50="#faf5ef", c100="#f1e5d4", c200="#e2c9a8", c300="#cfa877", c400="#b88851",
    c500="#9c6d3a", c600="#7f562d", c700="#654426", c800="#4f3520", c900="#3d291a", c950="#22160d")
SAND = gr.themes.Color(
    name="sand", c50="#fbf8f2", c100="#f5efe3", c200="#e9dfcc", c300="#d8c9ad", c400="#b9a685",
    c500="#998665", c600="#7a694e", c700="#5f523d", c800="#463c2d", c900="#2f281e", c950="#1c1812")

THEME = gr.themes.Soft(primary_hue=PALM, secondary_hue=TRUNK, neutral_hue=SAND).set(
    body_background_fill="*neutral_50",
    body_background_fill_dark="*primary_950",
    block_background_fill="white",
    block_border_color="*neutral_200",
    block_title_text_color="*secondary_700",
    block_label_text_color="*secondary_700",
    input_background_fill="*neutral_50",
    input_border_color_focus="*primary_400",
    border_color_accent="*primary_400",
    color_accent_soft="*primary_100",                 # the employee's chat bubbles
    background_fill_secondary="*neutral_100",         # the assistant's chat bubbles
    link_text_color="*primary_700",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_border_color="*primary_700",
    button_primary_text_color="white",
    button_cancel_background_fill="*secondary_600",   # Reject (variant="stop") in trunk brown
    button_cancel_background_fill_hover="*secondary_700",
    button_cancel_border_color="*secondary_700",
    button_cancel_text_color="white",
)

CSS = """
#palm-header {
    background: linear-gradient(135deg, #2f5a21 0%, #3b7028 55%, #7f562d 100%);
    border-radius: 14px; padding: 18px 24px; color: #fbf8f2;
    border-bottom: 5px solid #654426;
}
#palm-header h2 { margin: 0 0 4px 0; color: #fbf8f2; font-size: 1.6rem; }
#palm-header p { margin: 0; color: #dcebd3; font-size: 0.92rem; }
#palm-header code { background: rgba(0, 0, 0, 0.25); color: #f1e5d4; padding: 1px 6px; border-radius: 6px; }
.palm-panel { border-left: 4px solid #9c6d3a !important; border-radius: 10px !important; }
.palm-card { border-left: 4px solid #3b7028 !important; border-radius: 10px !important; }
.palm-panel h3, .palm-card h3 { color: #654426; }
"""

HEADER = f"""
<div id="palm-header">
  <h2>🌴 Nakhla IT Helpdesk</h2>
  <p>IT support assistant for Nakhla Systems (a fictional company). The helpdesk system behind it is
  simulated. Talking to <code>{APP_URL}</code>.</p>
</div>
"""


def call_app(path: str, body: dict) -> dict:
    try:
        r = httpx.post(f"{APP_URL}{path}", json=body, timeout=180)
    except httpx.HTTPError as exc:
        return {"reply": f"Cannot reach your app at {APP_URL} ({exc.__class__.__name__}). Is it running?"}
    try:
        data = r.json()
    except ValueError:
        data = {"reply": r.text[:500]}
    if r.status_code >= 400:
        data = {"reply": f"Your app answered HTTP {r.status_code}: {str(data)[:400]}"}
    return data if isinstance(data, dict) else {"reply": str(data)}


def side_panels(data: dict) -> tuple[str, str, str]:
    cites = data.get("citations") or []
    cite_md = "### Sources\n" + ("\n".join(
        f"- **{c.get('id', '?')}** {c.get('title', '')}" if isinstance(c, dict) else f"- {c}" for c in cites)
        or "_No knowledge-base article cited in the last answer._")
    calls = data.get("tool_calls") or []
    tools_md = "### Tool calls (last turn)\n" + ("\n".join(
        f"- `{t.get('tool', '?')}` {'ok' if t.get('ok') else 'FAILED'} · {t.get('ms', '?')} ms"
        for t in calls if isinstance(t, dict)) or "_none_")
    result = data.get("result") or {}
    state = data.get("state")
    result_md = ""
    if result:
        result_md = "### Result\n" + "\n".join(f"- **{k}**: `{v}`" for k, v in result.items())
    if state:
        result_md += f"\n\n_Workflow state: {state}_"
    return cite_md, tools_md, result_md


def ticket_card(pending: dict | None) -> str:
    if not pending:
        return NO_CARD
    payload = pending.get("payload") or {}
    lines = [f"### Ticket draft - waiting for your approval", f"**{pending.get('summary', '')}**", ""]
    for key in ("subject", "category", "priority", "department", "requester_email"):
        if key in payload:
            lines.append(f"- **{key}**: {payload[key]}")
    if payload.get("attempted_steps"):
        lines.append("- **attempted steps**: " + "; ".join(map(str, payload["attempted_steps"])))
    if "kb_refs" in payload:
        lines.append("- **KB refs**: " + (", ".join(payload["kb_refs"]) or "none"))
    if payload.get("description"):
        lines += ["", f"> {payload['description']}"]
    return "\n".join(lines)


def show(data: dict, history: list):
    history = history + [{"role": "assistant", "content": data.get("reply") or "(empty reply)"}]
    pending = data.get("pending_action") or None
    cite_md, tools_md, result_md = side_panels(data)
    buttons = gr.update(interactive=bool(pending))
    return history, pending, ticket_card(pending), buttons, buttons, cite_md, tools_md, result_md


def send(message: str, history: list, session_id: str, email: str):
    if not message.strip():
        return ("", history, *[gr.skip()] * 7)
    body = {"session_id": session_id, "message": message}
    if email.strip():
        body["user"] = {"email": email.strip().lower()}
    history = history + [{"role": "user", "content": message}]
    return ("", *show(call_app("/api/chat", body), history))


def decide(approved: bool, history: list, session_id: str, pending: dict | None):
    if not pending:
        return show({"reply": "There is no draft to approve or reject."}, history)
    history = history + [{"role": "user", "content": "Approve" if approved else "Reject"}]
    data = call_app("/api/approve", {"session_id": session_id, "action_id": pending.get("action_id"),
                                     "approved": approved})
    return show(data, history)


with gr.Blocks(title="Nakhla IT Helpdesk") as demo:
    session = gr.State(lambda: f"ui-{uuid.uuid4().hex}")  # 128 random bits: the id is all that proves the session
    pending_state = gr.State(None)
    gr.HTML(HEADER)
    with gr.Row():
        with gr.Column(scale=3):
            email = gr.Textbox(label="Your work email", value="tariq.hamdan@nakhla.example")
            chat = gr.Chatbot(label="Chat", height=460)
            msg = gr.Textbox(label="Message", placeholder="Describe your IT problem, e.g. NakhlaConnect shows NX-781",
                             submit_btn=True)
        with gr.Column(scale=2):
            with gr.Group(elem_classes="palm-card"):
                card = gr.Markdown(NO_CARD, padding=True)
                with gr.Row():
                    approve_btn = gr.Button("Approve", variant="primary", interactive=False)
                    reject_btn = gr.Button("Reject", variant="stop", interactive=False)
            result = gr.Markdown("", elem_classes="palm-panel")
            cites = gr.Markdown("### Sources\n_No answer yet._", padding=True, elem_classes="palm-panel")
            tools = gr.Markdown("### Tool calls (last turn)\n_none_", padding=True, elem_classes="palm-panel")
    outputs = [chat, pending_state, card, approve_btn, reject_btn, cites, tools, result]
    msg.submit(send, [msg, chat, session, email], [msg, *outputs])
    approve_btn.click(lambda h, s, p: decide(True, h, s, p), [chat, session, pending_state], outputs)
    reject_btn.click(lambda h, s, p: decide(False, h, s, p), [chat, session, pending_state], outputs)

if __name__ == "__main__":
    demo.launch(server_port=int(os.getenv("UI_PORT", "7860")), theme=THEME, css=CSS)
