"""Gradio UI starter for the IT Helpdesk Agent - a small client of YOUR API contract.

It never talks to the helpdesk API or the LLM directly. It only calls your app:
    POST {APP_URL}/api/chat      fallback only, when the WebSocket is unavailable
    WS   {APP_URL}/ws/chat       one chat turn per connection; answer text streams in while the model generates it
    POST {APP_URL}/api/approve   {"session_id", "action_id", "approved"}  + header X-Session-Token
The app sends a session token (X-Session-Token response header) on the first chat turn of a conversation;
approving or rejecting needs it back. It is kept in this tab's server-side state, next to the session id it
belongs to, and never shown, logged or put in a URL. A page refresh starts a new conversation (and token).
Run your app first (port 8000), then:   python ui/app_ui.py   ->  http://localhost:7860
Replace it with your own UI if you like; keep the contract.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid

import gradio as gr
import httpx
import websockets.asyncio.client
import websockets.exceptions

ui_log = logging.getLogger("helpdesk-ui.timing")
APP_URL = os.getenv("APP_URL", "http://127.0.0.1:8000").rstrip("/")   # 127.0.0.1: "localhost" costs ~2 s on Windows (::1 first)
WS_OPEN_SECONDS = float(os.getenv("UI_WS_OPEN_SECONDS") or 10)
WS_IDLE_SECONDS = float(os.getenv("UI_WS_IDLE_SECONDS") or 120)   # longest silence while the app works on a turn
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


def call_app(path: str, body: dict, token: str | None = None) -> tuple[dict, str | None]:
    """POST to our app. Returns the JSON answer and the session token the app issued in this answer, if any.

    No cookie jar on purpose: one Gradio server serves every browser tab, so a shared jar could mix the
    tokens of different users. Each tab sends its own token as a header instead.
    """
    headers = {"X-Session-Token": token} if token else {}
    try:
        r = httpx.post(f"{APP_URL}{path}", json=body, headers=headers, timeout=180)
    except httpx.HTTPError as exc:
        return {"reply": f"Cannot reach your app at {APP_URL} ({exc.__class__.__name__}). Is it running?"}, None
    try:
        data = r.json()
    except ValueError:
        data = {"reply": r.text[:500]}
    if r.status_code >= 400 and not (isinstance(data, dict) and data.get("reply")):
        data = {"reply": f"Your app answered HTTP {r.status_code}: {str(data)[:400]}"}
    return (data if isinstance(data, dict) else {"reply": str(data)}), r.headers.get("X-Session-Token")


class WSUnavailable(Exception):
    """The WebSocket could not be opened, or died before our message was sent: nothing happened server-side,
    so the turn may be retried over POST /api/chat."""


def ws_url() -> str:
    base = APP_URL
    return ("wss" if base.startswith("https://") else "ws") + "://" + base.split("://", 1)[-1] + "/ws/chat"


class UITiming:
    """Client-side timing of one streamed turn, relative to the moment the UI started it (monotonic clock).

    Joined to the server's `timing` log lines by turn_id (the two processes' clocks are not comparable, so only
    the turn_id and the elapsed values are). Logs names and numbers only: no message, answer or token.
    "displayed" means Gradio asked for the next update after receiving the previous one (the generator was
    resumed); it does not include the browser's own rendering time."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.marks: dict[str, float] = {}
        self.turn_id: str | None = None
        self.chunks = 0
        self.max_gap_ms = 0.0
        self._last_chunk: float | None = None

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, round((time.perf_counter() - self.started) * 1000, 1))

    def chunk_received(self) -> None:
        now = time.perf_counter()
        self.chunks += 1
        self.mark("first_chunk_received")
        if self._last_chunk is not None:
            self.max_gap_ms = max(self.max_gap_ms, round((now - self._last_chunk) * 1000, 1))
        self._last_chunk = now

    def log(self) -> None:
        fields = " ".join(f"{k}_ms={v}" for k, v in self.marks.items())
        ui_log.info("ui_timing turn=%s chunks=%s max_chunk_gap_ms=%s %s", self.turn_id, self.chunks,
                    self.max_gap_ms, fields)


async def ws_turn(body: dict, token: str | None, timing: UITiming | None = None):
    """One chat turn over /ws/chat. An async generator of UI events:
        ("chunk", text)             provisional text, in order
        ("retract", reason)         drop everything shown so far
        ("final", data, token)      the authoritative answer (the ChatResponse fields); token only on a first turn
        ("error", message)          a safe message from the app, or ours
    Raises WSUnavailable only while nothing has been sent. The token goes in a handshake header, never the URL.
    """
    import websockets
    headers = {"X-Session-Token": token} if token else {}
    timing = timing or UITiming()
    try:
        timing.mark("connect_start")
        ws = await websockets.asyncio.client.connect(ws_url(), additional_headers=headers, proxy=None,
                                                     open_timeout=WS_OPEN_SECONDS, max_size=2 ** 20)
        timing.mark("connected")
        try:
            await ws.send(json.dumps({"type": "chat", **body}))
            timing.mark("request_sent")
        except Exception:
            await ws.close()
            raise
    except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
        raise WSUnavailable(exc.__class__.__name__) from None
    turn_id, last_seq, done = None, 0, False
    try:
        while not done:
            try:
                raw = await asyncio.wait_for(ws.recv(), WS_IDLE_SECONDS)
                event = json.loads(raw)
                if not isinstance(event, dict) or not isinstance(event.get("seq"), int):
                    raise ValueError("bad event")
            except asyncio.TimeoutError:
                yield "error", "The assistant took too long to answer. Please try again."
                return
            except websockets.exceptions.ConnectionClosed:
                yield "error", "The connection to the assistant was lost before the answer was complete."
                return
            except (ValueError, TypeError):
                yield "error", "The assistant sent an unreadable answer. Please try again."
                return
            if turn_id is None:
                turn_id = event.get("turn_id")
                timing.turn_id = turn_id if isinstance(turn_id, str) else None
                timing.mark("first_event_received")
            if event.get("turn_id") != turn_id or event["seq"] <= last_seq:
                continue                                  # another turn's or a replayed / out-of-order event
            last_seq = event["seq"]
            kind = event.get("type")
            if kind == "chunk" and isinstance(event.get("text"), str):
                timing.chunk_received()
                yield "chunk", event["text"]
            elif kind == "retract":
                yield "retract", str(event.get("reason") or "")
            elif kind == "final":
                done = True
                timing.mark("final_received")
                issued = event.get("session_token")
                data = {k: v for k, v in event.items() if k not in ("type", "seq", "turn_id", "provisional",
                                                                    "display", "session_token")}
                yield "final", data, issued if isinstance(issued, str) else None
            elif kind == "error":
                done = True
                yield "error", str(event.get("message") or "Something went wrong.")[:500]
    finally:
        await ws.close()                                  # also on UI cancellation (GeneratorExit)


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


def send(message: str, history: list, session_id: str, email: str, auth: dict | None):
    """Sync fallback path (POST /api/chat), kept for the case where the WebSocket is unavailable."""
    if not message.strip():
        return ("", history, *[gr.skip()] * 7, gr.skip())
    body = {"session_id": session_id, "message": message}
    if email.strip():
        body["user"] = {"email": email.strip().lower()}
    history = history + [{"role": "user", "content": message}]
    data, issued = call_app("/api/chat", body)
    if issued:                                         # only the first turn of a conversation issues one
        auth = {"session_id": session_id, "token": issued}
    return ("", *show(data, history), auth)


STREAMING_CURSOR = " ▌"                  # marks provisional text: it is replaced by the verified reply
FALLBACK_NOTE = "_(Live streaming is unavailable, so this answer was not streamed.)_\n\n"


async def send_stream(message: str, history: list, session_id: str, email: str, auth: dict | None):
    """The chat handler: shows the answer while the model is still generating it (WebSocket /ws/chat)."""
    if not message.strip():
        yield ("", history, *[gr.skip()] * 7, gr.skip())
        return
    body = {"session_id": session_id, "message": message}
    if email.strip():
        body["user"] = {"email": email.strip().lower()}
    base = history + [{"role": "user", "content": message}]
    token = auth["token"] if auth and auth.get("session_id") == session_id else None
    skip = [gr.skip()] * 7
    shown = ""
    timing = UITiming()
    yield "", base, *skip, gr.skip()
    try:
        async with contextlib.aclosing(ws_turn(body, token, timing)) as turn:
            async for event in turn:
                kind = event[0]
                if kind == "chunk":
                    shown += event[1]
                    yield gr.skip(), base + [{"role": "assistant", "content": shown + STREAMING_CURSOR}], *skip, gr.skip()
                    timing.mark("first_chunk_displayed")
                elif kind == "retract":
                    shown = ""
                    yield gr.skip(), base, *skip, gr.skip()
                elif kind == "final":
                    _, data, issued = event
                    if issued:
                        auth = {"session_id": session_id, "token": issued}
                    yield gr.skip(), *show(data, base), auth
                    timing.mark("final_displayed")
                else:                   # an error: say so, keep the pending draft and the side panels as they are
                    yield gr.skip(), base + [{"role": "assistant", "content": event[1]}], *skip, gr.skip()
    except WSUnavailable:
        data, issued = await asyncio.to_thread(call_app, "/api/chat", body, None)
        if issued:
            auth = {"session_id": session_id, "token": issued}
        data = {**data, "reply": FALLBACK_NOTE + str(data.get("reply") or "")}
        yield gr.skip(), *show(data, base), auth
    finally:
        timing.log()


def decide(approved: bool, history: list, session_id: str, pending: dict | None, auth: dict | None):
    if not pending:
        return show({"reply": "There is no draft to approve or reject."}, history)
    history = history + [{"role": "user", "content": "Approve" if approved else "Reject"}]
    # Only ever the token issued for THIS tab's session id.
    token = auth["token"] if auth and auth.get("session_id") == session_id else None
    data, _ = call_app("/api/approve", {"session_id": session_id, "action_id": pending.get("action_id"),
                                        "approved": approved}, token)
    return show(data, history)


with gr.Blocks(title="Nakhla IT Helpdesk") as demo:
    session = gr.State(lambda: f"ui-{uuid.uuid4().hex}")  # 128 random bits: the id is all that proves the session
    pending_state = gr.State(None)
    auth_state = gr.State(None)       # {"session_id", "token"}: per tab, server-side, never sent to the browser
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
    msg.submit(send_stream, [msg, chat, session, email, auth_state], [msg, *outputs, auth_state])
    approve_btn.click(lambda h, s, p, a: decide(True, h, s, p, a), [chat, session, pending_state, auth_state], outputs)
    reject_btn.click(lambda h, s, p, a: decide(False, h, s, p, a), [chat, session, pending_state, auth_state], outputs)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    demo.launch(server_port=int(os.getenv("UI_PORT", "7860")), theme=THEME, css=CSS)
