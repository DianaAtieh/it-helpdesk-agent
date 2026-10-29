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
import html
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

import gradio as gr
import httpx
import websockets.asyncio.client
import websockets.exceptions

ui_log = logging.getLogger("helpdesk-ui.timing")
APP_URL = os.getenv("APP_URL", "http://127.0.0.1:8000").rstrip("/")   # 127.0.0.1: "localhost" costs ~2 s on Windows (::1 first)
WS_OPEN_SECONDS = float(os.getenv("UI_WS_OPEN_SECONDS") or 10)
WS_IDLE_SECONDS = float(os.getenv("UI_WS_IDLE_SECONDS") or 120)   # longest silence while the app works on a turn
NO_CARD = ("#### No ticket waiting\nWhen the assistant proposes a ticket, it appears here for you to review. "
           "Nothing is submitted until you click Approve.")

# ---------------------------------------------------------------- NAKHLA brand and theme
# Deep forest and palm greens, soft sage, a warm off-white, charcoal text and a small gold accent. The palm-and-shield
# mark (ui/assets/nakhla-mark.svg) is drawn in SVG, so it needs no image service. Fonts load from Google Fonts in
# the browser; offline, the system UI font is used.
ASSETS = Path(__file__).resolve().parent / "assets"
MARK_FILE = ASSETS / "nakhla-mark.svg"
MARK_SVG = MARK_FILE.read_text(encoding="utf-8") if MARK_FILE.is_file() else ""

FOREST = gr.themes.Color(
    name="forest", c50="#f2f7f3", c100="#dce9df", c200="#bcd6c3", c300="#8fbb9d", c400="#5c9c74",
    c500="#237a57", c600="#1d6a4b", c700="#185a40", c800="#154a37", c900="#123c32", c950="#0b2620")
STONE = gr.themes.Color(
    name="stone", c50="#f7f8f4", c100="#eff1eb", c200="#e3e8e2", c300="#cfd6cf", c400="#a3ada7",
    c500="#75817b", c600="#5d6863", c700="#47514c", c800="#323b37", c900="#202a27", c950="#141b19")
FONT = [gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"]
FONT_MONO = [gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"]

THEME = gr.themes.Base(primary_hue=FOREST, secondary_hue=FOREST, neutral_hue=STONE,
                       radius_size=gr.themes.sizes.radius_lg, text_size=gr.themes.sizes.text_md,
                       font=FONT, font_mono=FONT_MONO).set(
    body_background_fill="#f7f8f4",
    body_text_color="#202a27",
    body_text_color_subdued="#75817b",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f2f5f0",          # the assistant's chat bubbles
    color_accent_soft="#dce9df",                  # the employee's chat bubbles
    border_color_primary="#e3e8e2",
    border_color_accent="#237a57",
    block_background_fill="#ffffff",
    block_border_color="#e3e8e2",
    block_border_width="1px",
    block_radius="16px",
    block_shadow="0 1px 2px rgba(18, 60, 50, 0.05)",
    block_label_text_color="#75817b",
    block_title_text_color="#202a27",
    input_background_fill="#ffffff",
    input_border_color="#d5ddd6",
    input_border_color_focus="#237a57",
    input_shadow_focus="0 0 0 4px rgba(35, 122, 87, 0.14)",
    input_placeholder_color="#9aa49f",
    chatbot_text_size="*text_md",
    link_text_color="#237a57",
    code_background_fill="#eef3ef",
    button_border_width="1px",
    button_large_radius="12px",
    button_medium_radius="12px",
    button_small_radius="10px",
    button_primary_background_fill="#237a57",
    button_primary_background_fill_hover="#123c32",
    button_primary_border_color="#237a57",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f2f7f3",
    button_secondary_border_color="#d5ddd6",
    button_secondary_text_color="#202a27",
    button_cancel_background_fill="#ffffff",      # Reject and Clear (variant="stop")
    button_cancel_background_fill_hover="#fbf1ef",
    button_cancel_border_color="#e2b7b0",
    button_cancel_text_color="#a3392e",
)

CSS = """
footer { display: none !important; }
.gradio-container { max-width: 100% !important; width: 100% !important; padding: 0 !important; margin: 0 !important; }
#nk-shell { gap: 0 !important; min-height: 100vh; align-items: stretch; flex-wrap: nowrap; }

/* ---- sidebar ---- */
#nk-sidebar { background: #123c32; padding: 22px 16px 18px 16px !important; gap: 14px !important;
              min-height: 100vh; position: sticky; top: 0; align-self: flex-start; border: none !important; }
#nk-sidebar .block, #nk-sidebar .form { background: transparent !important; border: none !important; box-shadow: none !important; }
.nk-brand { display: flex; align-items: center; gap: 12px; padding: 2px 6px 14px 6px;
            border-bottom: 1px solid rgba(220, 233, 223, 0.14); }
.nk-brand svg { width: 44px; height: 44px; flex-shrink: 0; filter: drop-shadow(0 4px 10px rgba(0, 0, 0, 0.25)); }
.nk-wordmark { font-family: "Plus Jakarta Sans", Inter, sans-serif; font-weight: 800; letter-spacing: 0.18em;
               color: #f7f8f4; font-size: 1.15rem; line-height: 1.1; }
.nk-tagline { color: #9cc5aa; font-size: 0.74rem; letter-spacing: 0.02em; margin-top: 3px; white-space: nowrap; }
#nk-new { background: #237a57 !important; border-color: #2f8f67 !important; color: #fff !important; font-weight: 600 !important; }
#nk-new:hover { background: #2f8f67 !important; }
#nk-nav { margin-top: 2px; }
#nk-nav .wrap { display: flex !important; flex-direction: column !important; gap: 4px !important; }
#nk-nav label { display: flex !important; align-items: center; gap: 10px; width: 100%; margin: 0 !important;
                background: transparent !important; border: 1px solid transparent !important; border-radius: 11px !important;
                padding: 10px 12px !important; color: #c9dccf !important; font-weight: 550; font-size: 0.93rem;
                box-shadow: none !important; cursor: pointer; transition: background 0.15s ease, color 0.15s ease; }
#nk-nav label:hover { background: rgba(220, 233, 223, 0.08) !important; color: #ffffff !important; }
#nk-nav label.selected, #nk-nav label:has(input:checked) { background: rgba(220, 233, 223, 0.13) !important;
                color: #ffffff !important; border-color: rgba(213, 168, 79, 0.55) !important; }
#nk-nav input[type="radio"] { display: none !important; }
#nk-nav label::before { content: ""; width: 18px; height: 18px; flex-shrink: 0; background: currentColor; opacity: 0.9;
                        -webkit-mask: var(--nk-icon) center / contain no-repeat; mask: var(--nk-icon) center / contain no-repeat; }
#nk-nav label:nth-of-type(1) { --nk-icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'/></svg>"); }
#nk-nav label:nth-of-type(2) { --nk-icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='22 12 18 12 15 21 9 3 6 12 2 12'/></svg>"); }
#nk-nav label:nth-of-type(3) { --nk-icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M2 9a3 3 0 0 0 0 6v2a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-2a3 3 0 0 0 0-6V7a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2z'/><path d='M13 5v2M13 17v2M13 11v2'/></svg>"); }
#nk-nav label:nth-of-type(4) { --nk-icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><path d='M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg>"); }
.nk-nav-title { color: #75a38a; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
                padding: 6px 8px 0 8px; }
.nk-profile { display: flex; align-items: center; gap: 10px; padding: 12px; border-radius: 14px;
              background: rgba(220, 233, 223, 0.07); border: 1px solid rgba(220, 233, 223, 0.12); }
.nk-avatar { width: 36px; height: 36px; border-radius: 50%; background: #d5a84f; color: #123c32; font-weight: 800;
             display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-size: 0.9rem; }
.nk-profile-name { color: #f7f8f4; font-size: 0.82rem; font-weight: 600; word-break: break-all; line-height: 1.25; }
.nk-profile-meta { color: #9cc5aa; font-size: 0.72rem; margin-top: 2px; }
.nk-profile-meta a { color: #d5a84f !important; text-decoration: none; font-weight: 600; }
.nk-sidebar-foot { color: #5f8a74; font-size: 0.68rem; padding: 0 8px; }

/* ---- main area ---- */
#nk-main { padding: 22px 26px 18px 26px !important; gap: 14px !important; min-width: 0; }
.nk-page-head h1 { font-family: "Plus Jakarta Sans", Inter, sans-serif; font-size: 1.45rem; font-weight: 750;
                   letter-spacing: -0.02em; color: #123c32; margin: 0; }
.nk-page-head p { color: #75817b; margin: 4px 0 0 0; font-size: 0.92rem; }
.nk-chip { display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em; color: #8a6420;
           background: #f7ecd2; border: 1px solid #ecd49c; border-radius: 999px; padding: 2px 9px; margin-left: 8px;
           vertical-align: 3px; }

/* chat */
#nk-chat { border-radius: 18px !important; border: 1px solid #e3e8e2 !important; background: #ffffff !important; }
#nk-chat .placeholder, #nk-chat [class*="placeholder"] { display: flex; align-items: center; justify-content: center; height: 100%; }
.nk-welcome { text-align: center; max-width: 560px; margin: 0 auto; padding: 18px; }
.nk-welcome svg { width: 64px; height: 64px; margin: 0 auto 10px auto; display: block;
                  filter: drop-shadow(0 8px 18px rgba(18, 60, 50, 0.25)); animation: nk-float 5s ease-in-out infinite; }
.nk-welcome h2 { font-family: "Plus Jakarta Sans", Inter, sans-serif; font-size: 1.7rem; font-weight: 750; color: #123c32;
                 letter-spacing: -0.02em; margin: 0 0 6px 0; border: none; }
.nk-welcome p { color: #75817b; margin: 0; font-size: 0.97rem; line-height: 1.55; }
@keyframes nk-float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }
#nk-suggest { display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px !important; }
#nk-chat:has(.message-row) ~ #nk-suggest { display: none !important; }   /* suggestions only for an empty chat */
#nk-suggest button { justify-content: flex-start !important; text-align: left !important; min-height: 54px;
                     background: #ffffff !important; border: 1px solid #e3e8e2 !important; border-radius: 14px !important;
                     color: #202a27 !important; font-weight: 550 !important; font-size: 0.9rem !important;
                     padding: 12px 14px 12px 46px !important; position: relative; transition: all 0.15s ease; }
#nk-suggest button:hover { border-color: #8fbb9d !important; background: #f2f7f3 !important; transform: translateY(-1px);
                           box-shadow: 0 6px 16px rgba(18, 60, 50, 0.08); }
#nk-suggest button::before { content: ""; position: absolute; left: 14px; top: 50%; transform: translateY(-50%);
                             width: 22px; height: 22px; background: #237a57;
                             -webkit-mask: var(--nk-icon) center / contain no-repeat; mask: var(--nk-icon) center / contain no-repeat; }
#nk-suggest button:nth-of-type(1) { --nk-icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><rect x='3' y='11' width='18' height='11' rx='2'/><path d='M7 11V7a5 5 0 0 1 10 0v4'/></svg>"); }
#nk-suggest button:nth-of-type(2) { --nk-icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21 2l-2 2m-7.6 7.6a5.5 5.5 0 1 1-7.8 7.8 5.5 5.5 0 0 1 7.8-7.8zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4'/></svg>"); }
#nk-suggest button:nth-of-type(3) { --nk-icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><rect x='2' y='3' width='20' height='14' rx='2'/><line x1='8' y1='21' x2='16' y2='21'/><line x1='12' y1='17' x2='12' y2='21'/></svg>"); }
#nk-suggest button:nth-of-type(4) { --nk-icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z'/><line x1='12' y1='9' x2='12' y2='13'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg>"); }
#nk-msg { border-radius: 16px !important; box-shadow: 0 4px 18px rgba(18, 60, 50, 0.07) !important; }
#nk-msg textarea { font-size: 0.98rem !important; }
#nk-chat .message-buttons, #nk-chat .message-buttons-left, #nk-chat .message-buttons-right { display: none !important; }

/* context panel */
#nk-context { gap: 12px !important; min-width: 320px; }
.nk-card { border-radius: 16px !important; }
.nk-card-title { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #75817b;
                 padding: 2px 2px 0 2px; }
#nk-ticket { background: #ffffff !important; border: 1px solid #e3e8e2 !important; border-top: 3px solid #d5a84f !important;
             border-radius: 16px !important; padding: 6px 12px 10px 12px !important; gap: 6px !important;
             box-shadow: 0 1px 2px rgba(18, 60, 50, 0.05); }
#nk-ticket > .block, #nk-ticket .form { border: none !important; box-shadow: none !important; background: transparent !important; }
.nk-acc .nk-md h3 { display: none !important; }   /* the accordion label is the heading */
#nk-ticket h3 { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #8a6420; margin: 2px 0 8px 0; }
#nk-ticket h4 { font-size: 1.02rem; font-weight: 700; color: #123c32; margin: 4px 0 8px 0; }
#nk-ticket ul { list-style: none; padding: 0; margin: 0; }
#nk-ticket li { display: flex; gap: 8px; padding: 6px 0; border-bottom: 1px dashed #e3e8e2; font-size: 0.88rem; }
#nk-ticket li:last-child { border-bottom: none; }
#nk-ticket blockquote { border-left: 3px solid #8fbb9d; background: #f2f7f3; border-radius: 0 10px 10px 0;
                        padding: 8px 12px; margin: 10px 0 0 0; font-style: normal; color: #323b37; }
.nk-approval-note { font-size: 0.76rem; color: #75817b; }
#nk-approve-row { gap: 10px; }
#nk-side h3, #nk-context .nk-md h3 { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
                                      color: #75817b; margin: 2px 0 8px 0; }
#nk-context ul { padding-left: 0; list-style: none; margin: 0; }
#nk-context li { padding: 6px 0; font-size: 0.88rem; border-bottom: 1px dashed #e3e8e2; }
#nk-context li:last-child { border-bottom: none; }
.nk-state { display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem; font-weight: 700; border-radius: 999px;
            padding: 4px 11px; border: 1px solid transparent; }
.nk-state::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.nk-ok { color: #1d6a4b; background: #e3f1e7; border-color: #bcd6c3; }
.nk-info { color: #285f7a; background: #e6f0f5; border-color: #c4dbe7; }
.nk-warn { color: #8a6420; background: #f7ecd2; border-color: #ecd49c; }
.nk-alert { color: #a3392e; background: #fbecea; border-color: #efc6c0; }
.nk-gold { color: #7a5718; background: #f4e3bd; border-color: #e3c783; }
.nk-neutral { color: #47514c; background: #eff1eb; border-color: #e3e8e2; }
.nk-muted, .nk-muted p { font-size: 0.78rem !important; color: #75817b !important; }
code { font-family: var(--font-mono); font-size: 0.84em; padding: 1px 6px; border-radius: 6px; }

/* pages */
.nk-panel { background: #ffffff; border: 1px solid #e3e8e2; border-radius: 18px; padding: 18px 20px; }
.nk-panel h3 { font-family: "Plus Jakarta Sans", Inter, sans-serif; color: #123c32; font-size: 1.05rem; margin: 0 0 6px 0; }
.nk-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
.nk-tile { background: #ffffff; border: 1px solid #e3e8e2; border-radius: 16px; padding: 16px; }
.nk-tile b { color: #123c32; display: block; margin-bottom: 4px; }
.nk-tile span { color: #5d6863; font-size: 0.88rem; line-height: 1.5; }
.nk-empty { text-align: center; padding: 34px 16px; color: #75817b; }
.nk-empty b { display: block; color: #202a27; font-size: 1rem; margin-bottom: 4px; }
#nk-status-md, #nk-tickets-md { background: #ffffff; border: 1px solid #e3e8e2; border-radius: 18px; padding: 6px 10px; }
#nk-tickets-md table { width: 100%; }

/* ---- responsive ---- */
@media (max-width: 1100px) { #nk-context { min-width: 0; } }
@media (max-width: 900px) {
  #nk-shell { flex-direction: column !important; flex-wrap: wrap; }
  #nk-sidebar { min-height: 0; position: static; width: 100% !important; min-width: 0 !important; padding: 14px !important; }
  #nk-nav .wrap { flex-direction: row !important; flex-wrap: wrap; }
  #nk-nav label { width: auto; }
  .nk-sidebar-foot { display: none; }
  #nk-main { padding: 14px !important; }
  #nk-suggest { grid-template-columns: 1fr; }
}
"""

# A page load starts a new conversation (a new session id and token); the login cookie is kept.
NEW_CHAT = "() => { window.location.reload(); }"

BRAND_HTML = f"""
<div class="nk-brand">
  {MARK_SVG}
  <div><div class="nk-wordmark">NAKHLA</div><div class="nk-tagline">Intelligent IT Helpdesk</div></div>
</div>
"""

WELCOME = f"""<div class="nk-welcome">
{MARK_SVG}
<h2>How can we help you today?</h2>
<p>Your intelligent IT support assistant is ready to help. Answers come from the Nakhla IT knowledge base, with their
sources, and a ticket is only submitted after you approve it.</p>
</div>"""

SUGGESTIONS = ("I can't connect to the VPN.", "I need access to an application.", "My laptop is running slowly.",
               "I need help with an error message.")
NAV_PAGES = ("Helpdesk Chat", "Service Status", "My Tickets", "Help Center")
# Worded so the backend recognises it as a status question (a status cue plus named services, see
# app.agent.nodes.status_question); tests/test_ui_pages.py checks that.
STATUS_QUESTION = ("What is the current status of the VPN, email, Wi-Fi, sign-in, printing, Software Center "
                   "and VDI services?")

# ---------------------------------------------------------------- the sign-in page (Gradio's login, restyled)
# Gradio draws its own login form and does not load the app's CSS there, but it renders auth_message as HTML. So the
# brand panel (left) and the styles for the card (right) travel inside that message, scoped with :has(#nk-login) so
# they can never touch the dashboard. Text only; no script (scripts in this HTML would not run anyway).
_CHECK_ICON = ("<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.2' stroke-linecap='round' "
               "stroke-linejoin='round'><path d='M20 6 9 17l-5-5'/></svg>")
LOGIN_HTML = """
<style>
  body:has(#nk-login), .gradio-container:has(#nk-login), .main:has(#nk-login), .wrap:has(#nk-login) {
    background: #f7f8f4 !important; }
  .gradio-container:has(#nk-login) { max-width: 100% !important; width: 100% !important; margin: 0 !important; }
  .panel:has(#nk-login) > :has(> #nk-login) { order: -1; }   /* the brand header first (it only shows on mobile) */
  footer { display: none !important; }
  .wrap:has(#nk-login) { min-height: 100vh !important; padding: 40px 6vw 40px calc(44vw + 6vw) !important;
                         display: flex !important; align-items: center !important; justify-content: center !important; }
  .panel:has(#nk-login) { background: #ffffff !important; border: 1px solid #e3e8e2 !important; border-radius: 22px !important;
                          box-shadow: 0 30px 70px rgba(18, 60, 50, 0.13) !important; padding: 36px 34px 28px 34px !important;
                          width: 100% !important; max-width: 440px !important; min-width: 0 !important; gap: 14px !important; }
  .panel:has(#nk-login) > h2 { font-size: 0 !important; margin: 0 0 6px 0 !important; }
  .panel:has(#nk-login) > h2::before { content: "Welcome back"; display: block; color: #123c32; letter-spacing: -0.02em;
                                       font: 750 1.65rem/1.2 "Plus Jakarta Sans", Inter, system-ui, sans-serif; }
  .panel:has(#nk-login) > h2::after { content: "Sign in with your Nakhla company email and password.";
                                      display: block; margin-top: 6px; color: #75817b;
                                      font: 400 0.93rem/1.5 Inter, system-ui, sans-serif; }
  .panel:has(#nk-login) .form { border: none !important; background: transparent !important; gap: 14px !important;
                                box-shadow: none !important; }
  .panel:has(#nk-login) .form > div { border: none !important; background: transparent !important; padding: 0 !important; }
  .panel:has(#nk-login) [data-testid="block-info"] { font-size: 0 !important; }
  .panel:has(#nk-login) [data-testid="block-info"]::after { font: 600 0.82rem/1.4 Inter, system-ui, sans-serif; color: #323b37;
                                                           display: block; margin-bottom: 6px; }
  .panel:has(#nk-login) label, .panel:has(#nk-login) .input-container, .panel:has(#nk-login) .form > div {
    overflow: visible !important; }
  .panel:has(#nk-login) .input-container { padding-bottom: 2px !important; }
  .panel:has(#nk-login) .form > div:nth-child(1) [data-testid="block-info"]::after { content: "Company email"; }
  .panel:has(#nk-login) .form > div:nth-child(2) [data-testid="block-info"]::after { content: "Password"; }
  .panel:has(#nk-login) input { border: 1px solid #d5ddd6 !important; border-radius: 12px !important; background: #fbfcfa !important;
                                padding: 12px 14px !important; font-size: 0.97rem !important; box-sizing: border-box !important; }
  .panel:has(#nk-login) input:focus { border-color: #237a57 !important; box-shadow: 0 0 0 4px rgba(35, 122, 87, 0.14) !important; }
  .panel:has(#nk-login) button { background: #237a57 !important; border: 1px solid #237a57 !important; border-radius: 12px !important;
                                 color: #ffffff !important; font-size: 0 !important; min-height: 48px; margin-top: 4px; }
  .panel:has(#nk-login) button::after { content: "Sign in"; font: 650 1rem/1 Inter, system-ui, sans-serif; }
  .panel:has(#nk-login) button:hover { background: #123c32 !important; }
  #nk-login { position: fixed; top: 0; left: 0; bottom: 0; width: 44vw; z-index: 1; box-sizing: border-box; overflow: hidden;
              padding: 54px 56px 40px 56px; display: flex; flex-direction: column; justify-content: space-between;
              color: #dce9df; font-family: Inter, system-ui, sans-serif;
              background: radial-gradient(circle at 85% 12%, rgba(213, 168, 79, 0.18), transparent 32%),
                          radial-gradient(circle at 10% 90%, rgba(35, 122, 87, 0.55), transparent 45%), #123c32; }
  #nk-login::after { content: ""; position: absolute; right: -120px; bottom: -120px; width: 420px; height: 420px; opacity: 0.07;
                     border-radius: 50%; border: 60px solid #dce9df; }
  #nk-login .nkl-brand { display: flex; align-items: center; gap: 14px; }
  #nk-login .nkl-brand svg { width: 58px; height: 58px; filter: drop-shadow(0 8px 18px rgba(0, 0, 0, 0.3)); }
  #nk-login .nkl-word { font: 800 1.45rem/1 "Plus Jakarta Sans", Inter, sans-serif; letter-spacing: 0.2em; color: #f7f8f4; }
  #nk-login .nkl-tag { color: #9cc5aa; font-size: 0.82rem; margin-top: 6px; letter-spacing: 0.02em; }
  #nk-login h1 { font: 750 2.15rem/1.18 "Plus Jakarta Sans", Inter, sans-serif; color: #f7f8f4; letter-spacing: -0.02em;
                 margin: 0 0 14px 0; max-width: 470px; }
  #nk-login h1 span { color: #d5a84f; }
  #nk-login .nkl-lead { color: #b9d3c1; font-size: 1rem; line-height: 1.6; max-width: 450px; margin: 0 0 26px 0; }
  #nk-login ul { list-style: none; padding: 0; margin: 0; display: grid; gap: 12px; max-width: 450px; }
  #nk-login li { display: flex; align-items: flex-start; gap: 12px; font-size: 0.95rem; color: #e6efe8; line-height: 1.45; }
  #nk-login li i { flex-shrink: 0; width: 26px; height: 26px; border-radius: 8px; background: rgba(213, 168, 79, 0.16);
                   color: #d5a84f; display: flex; align-items: center; justify-content: center; }
  #nk-login li i svg { width: 15px; height: 15px; }
  #nk-login .nkl-foot { font-size: 0.78rem; color: #8db59b; line-height: 1.6; border-top: 1px solid rgba(220, 233, 223, 0.14);
                        padding-top: 16px; }
  #nk-login .nkl-foot b { color: #e6efe8; font-weight: 600; }
  @media (max-width: 900px) {
    .wrap:has(#nk-login) { padding: 0 0 32px 0 !important; flex-direction: column !important; justify-content: flex-start !important; }
    #nk-login { position: static; width: auto; margin: -36px -34px 10px -34px; padding: 26px 26px 22px 26px;
                border-radius: 22px 22px 0 0; }
    #nk-login h1, #nk-login .nkl-lead, #nk-login ul { display: none; }
    #nk-login .nkl-foot { margin-top: 14px; }
    .panel:has(#nk-login) { border-radius: 22px !important; margin: 16px !important; width: auto !important; }
  }
</style>
<div id="nk-login">
  <div class="nkl-brand">{{MARK}}<div><div class="nkl-word">NAKHLA</div><div class="nkl-tag">Intelligent IT Helpdesk</div></div></div>
  <div>
    <h1>IT support that knows <span>your company.</span></h1>
    <p class="nkl-lead">Describe the problem in your own words. The assistant answers from the Nakhla IT knowledge base
    and applies your department's rules.</p>
    <ul>
      <li><i>{{CHECK}}</i><span>Answers grounded in the IT knowledge base, always with their sources</span></li>
      <li><i>{{CHECK}}</i><span>Your department's rules and exceptions applied from the company directory</span></li>
      <li><i>{{CHECK}}</i><span>Tickets are drafted for you, and nothing is filed until you approve it</span></li>
    </ul>
  </div>
  <div class="nkl-foot"><b>IT Service Desk</b> · ext. 4357 · help.nakhla.example · 08:00-18:00 on business days<br>
  Demo environment: Nakhla Systems is fictional and the helpdesk is simulated.</div>
</div>
""".replace("{{MARK}}", MARK_SVG).replace("{{CHECK}}", _CHECK_ICON)


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
        ("status", label)           which stage the agent is in (transient; before the first chunk)
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
            if kind == "status" and isinstance(event.get("label"), str):
                timing.mark("first_status_received")
                yield "status", event["label"][:STATUS_MAX_CHARS]
            elif kind == "chunk" and isinstance(event.get("text"), str):
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


# How each backend state is shown (the response "state" field, unchanged). An unknown state is shown by its own
# name in a neutral badge, never dropped. Only [a-z_] from the state reaches the HTML.
STATE_BADGES = {
    "answered": ("Answered from the knowledge base", "ok"),
    "clarify": ("Needs a little more detail", "info"),
    "not_documented": ("Not documented", "warn"),
    "incident": ("Known incident", "alert"),
    "service_status": ("Service status", "info"),
    "access_info": ("Access information", "info"),
    "finance_vpn": ("Finance VPN · Security Operations", "warn"),
    "policy_refusal": ("Security policy", "alert"),
    "await_approval": ("Waiting for your approval", "gold"),
    "draft_proposed": ("Ticket draft proposed", "gold"),
    "preference": ("Preference", "info"),
    "blocked": ("Blocked", "alert"),
    "error": ("Something went wrong", "alert"),
    "unauthorized": ("Not authorized", "alert"),
    "busy": ("Still working on the previous message", "warn"),
    "identity_mismatch": ("Identity mismatch", "alert"),
}
PRIORITY_LABELS = {"P1": "P1 · critical", "P2": "P2 · high", "P3": "P3 · normal", "P4": "P4 · low"}
CATEGORY_LABELS = {"network_access": "Network access", "identity_access": "Identity & access", "hardware": "Hardware",
                   "software": "Software", "email_collab": "Email & collaboration", "security": "Security",
                   "onboarding": "Onboarding", "other": "Other"}


def state_badge(state: str | None) -> str:
    """An HTML badge for a response state ("" when there is none)."""
    safe = re.sub(r"[^a-z_]", "", str(state or "").lower())
    if not safe:
        return ""
    label, kind = STATE_BADGES.get(safe, (safe.replace("_", " ").capitalize(), "neutral"))
    return f"<span class='nk-state nk-{kind}' data-state='{safe}'>{label}</span>"


def side_panels(data: dict) -> tuple[str, str, str]:
    cites = data.get("citations") or []
    cite_md = "### Sources\n" + ("\n".join(
        f"- `{c.get('id', '?')}` {c.get('title', '')}" if isinstance(c, dict) else f"- `{c}`" for c in cites)
        or "_No knowledge-base article was cited in the last answer._")
    calls = data.get("tool_calls") or []
    tools_md = "### Tool calls (last turn)\n" + ("\n".join(
        f"- {'✅' if t.get('ok') else '❌'} `{t.get('tool', '?')}` · {t.get('ms', '?')} ms"
        + ("" if t.get("ok") else " · **failed**")
        for t in calls if isinstance(t, dict)) or "_none_")
    result = data.get("result") or {}
    result_md = "### Last response\n" + (state_badge(data.get("state")) or "_No response yet._")
    if result:
        result_md += "\n\n" + "\n".join(f"- **{str(k).replace('_', ' ').capitalize()}** `{v}`"
                                        for k, v in result.items())
    return cite_md, tools_md, result_md


def ticket_card(pending: dict | None) -> str:
    if not pending:
        return NO_CARD
    payload = pending.get("payload") or {}
    lines = ["### Ticket draft · waiting for your approval",
             f"#### {payload.get('subject') or pending.get('summary', '')}", ""]
    if "priority" in payload:
        lines.append(f"- **Priority** {PRIORITY_LABELS.get(payload['priority'], payload['priority'])}")
    if "category" in payload:
        lines.append(f"- **Category** {CATEGORY_LABELS.get(payload['category'], payload['category'])}")
    if payload.get("department"):
        lines.append(f"- **Department** {payload['department']}")
    if payload.get("requester_email"):
        lines.append(f"- **Requester** `{payload['requester_email']}`")
    if payload.get("attempted_steps"):
        lines.append("- **Already tried** " + "; ".join(map(str, payload["attempted_steps"])))
    if "kb_refs" in payload:
        lines.append("- **Related articles** " + (" ".join(f"`{ref}`" for ref in payload["kb_refs"]) or "none"))
    if payload.get("description"):
        lines += ["", f"> {payload['description']}"]
    return "\n".join(lines)


def show(data: dict, history: list):
    history = history + [{"role": "assistant", "content": data.get("reply") or "(empty reply)"}]
    pending = data.get("pending_action") or None
    cite_md, tools_md, result_md = side_panels(data)
    buttons = gr.update(interactive=bool(pending))
    return history, pending, ticket_card(pending), buttons, buttons, cite_md, tools_md, result_md


def login_email(email: str, request: gr.Request | None = None) -> str:
    """Who the chat talks for: the email this browser signed in with (Gradio login, ui/users.py), else `email`."""
    signed_in = getattr(request, "username", None)
    return (signed_in if isinstance(signed_in, str) and signed_in.strip() else email or "").strip().lower()


def send(message: str, history: list, session_id: str, email: str, auth: dict | None,
         request: gr.Request | None = None):
    """Sync fallback path (POST /api/chat), kept for the case where the WebSocket is unavailable."""
    if not message.strip():
        return ("", history, *[gr.skip()] * 7, gr.skip())
    body = {"session_id": session_id, "message": message}
    email = login_email(email, request)
    if email:
        body["user"] = {"email": email}
    history = history + [{"role": "user", "content": message}]
    token = auth["token"] if auth and auth.get("session_id") == session_id else None
    data, issued = call_app("/api/chat", body, token)
    if issued:                                         # only the first turn of a conversation issues one
        auth = {"session_id": session_id, "token": issued}
    return ("", *show(data, history), auth)


STREAMING_CURSOR = " ▌"                  # marks provisional text: it is replaced by the verified reply
STATUS_MAX_CHARS = 80
STATUS_PREFIX = "⏳ "                    # a status is a temporary bubble: never part of the history or the reply


def status_bubble(label: str) -> dict:
    return {"role": "assistant", "content": f"{STATUS_PREFIX}_{label}…_"}


FALLBACK_NOTE = "_(Live streaming is unavailable, so this answer was not streamed.)_\n\n"


async def send_stream(message: str, history: list, session_id: str, email: str, auth: dict | None,
                      request: gr.Request | None = None):
    """The chat handler: shows the answer while the model is still generating it (WebSocket /ws/chat)."""
    if not message.strip():
        yield ("", history, *[gr.skip()] * 7, gr.skip())
        return
    body = {"session_id": session_id, "message": message}
    email = login_email(email, request)
    if email:
        body["user"] = {"email": email}
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
                if kind == "status":
                    if not shown:       # only while no answer text is on screen; `base` never contains a status
                        yield gr.skip(), base + [status_bubble(event[1])], *skip, gr.skip()
                        timing.mark("first_status_displayed")
                elif kind == "chunk":
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
        data, issued = await asyncio.to_thread(call_app, "/api/chat", body, token)
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


# ---------------------------------------------------------------- saved preferences (long-term, per employee)
# These call GET /preferences and POST /preferences/reset with THIS tab's session id and token (headers only).
# They never touch the chat, the pending ticket draft or the session: "Clear chat" in Gradio only empties the
# visible chat box, POST /reset is not used here, and clearing preferences does not do either.
PREF_LABELS = (("language", "Language"), ("verbosity", "Answer length"), ("format", "Answer format"))
PREF_VALUES = {"en": "English", "short": "short", "normal": "normal", "detailed": "detailed",
               "numbered": "numbered steps", "bullets": "bullet points", "prose": "plain prose"}
PREF_NOTE = ("_Preferences only change how answers are written, never facts, policies or tickets. "
             "The email you type is not verified: this is a demo without login._")
PREF_NO_SESSION = "Send a message first, so this conversation is linked to your email."
PREF_ERRORS = {
    401: "This conversation could not be verified. Send a message first.",
    403: "This conversation's session is no longer valid. Start a new chat and try again.",
    404: "This conversation is not linked to an email yet. Send a message with your email first.",
    409: "The conversation is busy. Try again in a moment.",
    503: "Preferences are not available right now.",
}


def render_preferences(preferences: dict | None) -> str:
    rows = [f"- **{label}:** {PREF_VALUES.get(preferences[key], preferences[key])}"
            for key, label in PREF_LABELS if preferences and key in preferences]
    body = "\n".join(rows) if rows else "_No saved preferences._"
    return f"### Saved Preferences\n{body}\n\n{PREF_NOTE}"


def call_preferences(method: str, path: str, session_id: str, auth: dict | None) -> tuple[bool, dict | str]:
    """(True, JSON) when the app confirmed with HTTP 200, else (False, a safe message). Without a token for THIS
    session id nothing is sent. The token goes in a header only; nothing about the request or answer is logged."""
    if not auth or auth.get("session_id") != session_id or not auth.get("token"):
        return False, PREF_NO_SESSION
    headers = {"X-Session-Id": session_id, "X-Session-Token": auth["token"]}
    try:
        r = httpx.request(method, f"{APP_URL}{path}", headers=headers, timeout=30)
    except httpx.HTTPError as exc:
        ui_log.warning("preferences request failed: %s", exc.__class__.__name__)
        return False, f"Cannot reach the app at {APP_URL}. Is it running?"
    if r.status_code != 200:
        return False, PREF_ERRORS.get(r.status_code, f"The app answered HTTP {r.status_code}.")
    try:
        data = r.json()
    except ValueError:
        return False, "The app sent an answer that could not be read."
    return (True, data) if isinstance(data, dict) else (False, "The app sent an answer that could not be read.")


def refresh_preferences(session_id: str, auth: dict | None):
    """-> (saved preferences panel, feedback). On failure the panel is left as it was."""
    ok, data = call_preferences("GET", "/preferences", session_id, auth)
    if not ok:
        return gr.update(), f"⚠️ Could not load your preferences. {data}"
    saved = data.get("preferences")
    if not isinstance(saved, dict):
        return gr.update(), "⚠️ Could not load your preferences. The app sent an answer that could not be read."
    return render_preferences(saved), "Preferences refreshed."


def clear_preferences(session_id: str, auth: dict | None):
    """-> (saved preferences panel, feedback). Success is claimed only when the app confirms it."""
    ok, data = call_preferences("POST", "/preferences/reset", session_id, auth)
    if not ok:
        return gr.update(), f"⚠️ Your preferences were NOT cleared. {data}"
    if data.get("status") != "ok":
        return gr.update(), "⚠️ Your preferences were NOT cleared. The app did not confirm it."
    return render_preferences({}), "✅ Your saved preferences were cleared. Your chat and any ticket draft are unchanged."


# The backend's save confirmation: main.py adds it only after the database write succeeded, and the respond node
# removes it from model text, so the model's wording alone can never trigger a refresh. Keep it in step with
# app.agent.preferences.SAVED_CONFIRMATION (tests/test_ui_preference_refresh.py checks that).
SAVED_CONFIRMATION = "Saved to your preferences:"


def message_text(content) -> str | None:
    """A chat message's text. Gradio 6 hands handlers the history with each message's content as a list of blocks
    ([{"type": "text", "text": "..."}]); messages the UI built itself have a plain string. None if there is no text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        parts = [item if isinstance(item, str) else item.get("text") for item in content
                 if isinstance(item, str) or (isinstance(item, dict) and isinstance(item.get("text"), str))]
        return "\n".join(parts) if parts else None
    return None


def refresh_after_confirmed_save(history: list, session_id: str, auth: dict | None, seen: int):
    """Runs after every chat turn -> (saved preferences panel, feedback, seen).

    Reloads the panel from GET /preferences (the stored values) only when the turn's final reply carries the
    backend's save confirmation; otherwise it sends nothing. `seen` is the chat length already handled, so the same
    turn is never refreshed twice. If the request fails the panel is left as it was and the chat is unaffected."""
    history = history or []
    if len(history) == seen:
        return gr.skip(), gr.skip(), seen
    last = history[-1]
    reply = message_text(last.get("content")) if isinstance(last, dict) and last.get("role") == "assistant" else None
    if not isinstance(reply, str) or SAVED_CONFIRMATION not in reply:
        return gr.skip(), gr.skip(), len(history)
    ok, data = call_preferences("GET", "/preferences", session_id, auth)
    saved = data.get("preferences") if ok and isinstance(data, dict) else None
    if not isinstance(saved, dict):
        return gr.skip(), "⚠️ Could not refresh your preferences automatically. Use Refresh Preferences.", len(history)
    return render_preferences(saved), "Preferences updated.", len(history)


def signed_in_text(request: gr.Request | None = None) -> str:
    """The header line for the signed-in employee (empty when the UI runs without a login)."""
    email = login_email("", request)
    return f"Signed in as **{email}** · [Log out](/logout)" if email else ""


def profile_card(request: gr.Request | None = None) -> str:
    """The sidebar profile for the signed-in employee (from the Gradio login), with a log-out link."""
    email = login_email("", request)
    if not email:
        return ("<div class='nk-profile'><div class='nk-avatar'>?</div><div><div class='nk-profile-name'>Not signed in"
                "</div><div class='nk-profile-meta'>No login on this page</div></div></div>")
    initials = "".join(part[:1] for part in email.split("@")[0].split(".")[:2]).upper() or "?"
    return (f"<div class='nk-profile'><div class='nk-avatar'>{html.escape(initials)}</div><div>"
            f"<div class='nk-profile-name'>{html.escape(email)}</div>"
            "<div class='nk-profile-meta'>Signed in · <a href='/logout'>Log out</a></div></div></div>")


def show_page(page: str):
    """Navigation: only the chosen page is visible."""
    return [gr.Column(visible=page == name) for name in NAV_PAGES]


# ---------------------------------------------------------------- My Tickets: only what this conversation filed
# The app offers no "list my tickets" endpoint, so this page lists only tickets approved in this browser session,
# using the ticket ID from the helpdesk's own confirmation and the draft the employee approved. Nothing is invented.
TICKET_ID = re.compile(r"\b(IT-\d{3,6})\b")
TICKETS_EMPTY = ("<div class='nk-empty'><b>No tickets filed in this conversation yet</b>When the assistant proposes "
                 "a ticket, review it in the Ticket approval card and click Approve. Filed tickets appear here.</div>")


def render_tickets(tickets: list | None) -> str:
    if not tickets:
        return TICKETS_EMPTY
    rows = ["| Ticket | Subject | Priority | Category |", "|---|---|---|---|"]
    for t in tickets:
        subject = str(t.get("subject") or "").replace("|", "/")
        rows.append(f"| `{t['id']}` | {subject} | {PRIORITY_LABELS.get(t.get('priority'), t.get('priority') or '')} | "
                    f"{CATEGORY_LABELS.get(t.get('category'), t.get('category') or '')} |")
    return "\n".join(rows)


def record_ticket(history: list, draft: dict | None, tickets: list | None):
    """After Approve: add the ticket the helpdesk confirmed (its ID is in the backend's reply) to this
    conversation's list, once. A failed or rejected approval has no ticket ID and adds nothing."""
    tickets = list(tickets or [])
    last = (history or [None])[-1]
    reply = message_text(last.get("content")) if isinstance(last, dict) and last.get("role") == "assistant" else None
    match = TICKET_ID.search(reply) if isinstance(reply, str) else None
    if match and match.group(1) not in {t["id"] for t in tickets}:
        payload = (draft or {}).get("payload") or {}
        tickets.append({"id": match.group(1), "subject": payload.get("subject") or (draft or {}).get("summary", ""),
                        "priority": payload.get("priority"), "category": payload.get("category")})
    return tickets, render_tickets(tickets)


# ---------------------------------------------------------------- Service Status: the assistant's real answer
STATUS_EMPTY = ("<div class='nk-empty'><b>No status checked in this conversation yet</b>Ask the assistant with the "
                "button below. Its answer, read from the IT status page, appears here.</div>")


def update_status_page(result_md: str, history: list):
    """After a chat turn: when the backend answered a status question or matched an incident, show that answer."""
    state = re.search(r"data-state='([a-z_]+)'", result_md or "")
    last = (history or [None])[-1]
    reply = message_text(last.get("content")) if isinstance(last, dict) and last.get("role") == "assistant" else None
    if not state or state.group(1) not in ("service_status", "incident") or not isinstance(reply, str):
        return gr.skip()
    return f"{state_badge(state.group(1))}\n\n{reply}"


PAGE_HEAD = "<div class='nk-page-head'><h1>{title}</h1><p>{text}</p></div>"
HELP_HTML = """
<div class="nk-grid">
  <div class="nk-tile"><b>What the assistant can help with</b><span>VPN and Wi-Fi errors, sign-in and MFA problems,
  account lockouts, software installs and requests, application access, devices, mail and printing. Include the exact
  error code you see (for example <code>NX-781</code>) for the fastest answer.</span></div>
  <div class="nk-tile"><b>How tickets work</b><span>If a problem needs IT, the assistant prepares a ticket draft. You
  review it in the Ticket approval card. Nothing is submitted until you click Approve, and Reject files nothing.</span></div>
  <div class="nk-tile"><b>Your security</b><span>The assistant never asks for or reveals passwords, codes or admin
  credentials, and requests that would weaken security are refused. Who you are comes from your login, not from what
  you type.</span></div>
  <div class="nk-tile"><b>Contact the Service Desk</b><span>Portal <code>https://help.nakhla.example</code> or ext.
  4357, staffed 08:00-18:00 on business days. Option 1 on the phone line is the 24/7 line for critical incidents.
  <br><em>Source: KB-017, IT Service Desk FAQ</em></span></div>
  <div class="nk-tile"><b>Who can see my tickets?</b><span>You, your manager and the IT teams working on them.
  <br><em>Source: KB-017, IT Service Desk FAQ</em></span></div>
  <div class="nk-tile"><b>Where answers come from</b><span>Every answer is grounded in the Nakhla IT knowledge base and
  lists its sources. If something is not documented, the assistant says so instead of guessing.</span></div>
</div>
"""


with gr.Blocks(title="NAKHLA · Intelligent IT Helpdesk", fill_width=True) as demo:
    session = gr.State(lambda: f"ui-{uuid.uuid4().hex}")  # 128 random bits: the id is all that proves the session
    pending_state = gr.State(None)
    auth_state = gr.State(None)       # {"session_id", "token"}: per tab, server-side, never sent to the browser
    prefs_seen = gr.State(0)          # chat length already checked for a save confirmation (refresh_after_confirmed_save)
    email = gr.State("")              # no email box: the chat talks for the signed-in email (login_email)
    last_draft = gr.State(None)       # the last ticket draft shown, so My Tickets can describe what was approved
    tickets_state = gr.State([])      # tickets approved in this conversation (My Tickets)

    with gr.Row(elem_id="nk-shell", equal_height=False):
        # ------------------------------------------------------------ sidebar
        with gr.Column(scale=0, min_width=264, elem_id="nk-sidebar"):
            gr.HTML(BRAND_HTML, padding=False)
            new_btn = gr.Button("＋  New Conversation", elem_id="nk-new")
            gr.HTML("<div class='nk-nav-title'>Workspace</div>", padding=False)
            nav = gr.Radio(list(NAV_PAGES), value=NAV_PAGES[0], label="Navigation", show_label=False,
                           container=False, elem_id="nk-nav")
            profile = gr.HTML("", padding=False)
            gr.HTML("<div class='nk-sidebar-foot'>Nakhla Systems is fictional; the helpdesk behind this assistant "
                    "is simulated.</div>", padding=False)

        # ------------------------------------------------------------ main area
        with gr.Column(scale=1, elem_id="nk-main"):
            with gr.Column(visible=True) as page_chat:
                gr.HTML(PAGE_HEAD.format(title="Helpdesk Chat", text="Describe your problem in your own words and "
                                         "include any error code you see."), padding=False)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7, min_width=380):
                        chat = gr.Chatbot(label="Chat", show_label=False, height="64vh", layout="bubble",
                                          placeholder=WELCOME, buttons=[], elem_id="nk-chat",
                                          avatar_images=(None, str(MARK_FILE)) if MARK_FILE.is_file() else None)
                        with gr.Row(elem_id="nk-suggest"):
                            suggest_btns = [gr.Button(text, variant="secondary") for text in SUGGESTIONS]
                        msg = gr.Textbox(label="Message", show_label=False, submit_btn=True, autofocus=True,
                                         lines=1, max_lines=6, elem_id="nk-msg",
                                         placeholder="Describe your IT problem, e.g. NakhlaConnect shows NX-781")
                    with gr.Column(scale=4, min_width=320, elem_id="nk-context"):
                        with gr.Column(elem_id="nk-ticket", elem_classes="nk-card"):
                            card = gr.Markdown(NO_CARD, padding=True)
                            with gr.Row(elem_id="nk-approve-row"):
                                approve_btn = gr.Button("Approve", variant="primary", interactive=False)
                                reject_btn = gr.Button("Reject", variant="stop", interactive=False)
                            gr.Markdown("Nothing is submitted until you click Approve. Reject files nothing.",
                                        elem_classes="nk-approval-note", padding=True)
                        result = gr.Markdown("### Last response\n_No response yet._", padding=True,
                                             elem_classes="nk-md")
                        with gr.Accordion("Sources", open=True, elem_classes="nk-acc"):
                            cites = gr.Markdown("### Sources\n_No answer yet._", elem_classes="nk-md")
                        with gr.Accordion("Answer preferences", open=False, elem_classes="nk-acc"):
                            prefs_panel = gr.Markdown(render_preferences({}))
                            with gr.Row():
                                prefs_refresh_btn = gr.Button("Refresh Preferences", size="sm")
                                prefs_clear_btn = gr.Button("Clear Saved Preferences", variant="stop", size="sm")
                            prefs_feedback = gr.Markdown("")
                        with gr.Accordion("Technical details", open=False, elem_classes="nk-acc"):
                            tools = gr.Markdown("### Tool calls (last turn)\n_none_", elem_classes="nk-md")
                            signed_in = gr.Markdown("", elem_classes="nk-muted")
                            gr.Markdown(f"Connected to `{APP_URL}`", elem_classes="nk-muted")

            with gr.Column(visible=False) as page_status:
                gr.HTML(PAGE_HEAD.format(title="Service Status", text="Read live from the IT status page by the "
                                         "assistant. Nothing here is simulated or invented."), padding=False)
                status_md = gr.Markdown(STATUS_EMPTY, elem_id="nk-status-md", padding=True)
                status_btn = gr.Button("Ask the assistant for the current status", variant="primary")

            with gr.Column(visible=False) as page_tickets:
                gr.HTML(PAGE_HEAD.format(title="My Tickets", text="Tickets you approved in this conversation, with the "
                                         "ID the helpdesk confirmed. The helpdesk offers no full ticket list to "
                                         "this app."), padding=False)
                tickets_md = gr.Markdown(TICKETS_EMPTY, elem_id="nk-tickets-md", padding=True)

            with gr.Column(visible=False) as page_help:
                gr.HTML(PAGE_HEAD.format(title="Help Center", text="How the Nakhla assistant works and how to reach "
                                         "the IT Service Desk."), padding=False)
                gr.HTML(HELP_HTML, padding=False)
                help_chat_btn = gr.Button("Start a conversation", variant="primary")

    pages = [page_chat, page_status, page_tickets, page_help]
    outputs = [chat, pending_state, card, approve_btn, reject_btn, cites, tools, result]
    turn = msg.submit(send_stream, [msg, chat, session, email, auth_state], [msg, *outputs, auth_state])
    turn.then(refresh_after_confirmed_save, [chat, session, auth_state, prefs_seen],
              [prefs_panel, prefs_feedback, prefs_seen])
    turn.then(update_status_page, [result, chat], status_md)
    approve_btn.click(lambda h, s, p, a: decide(True, h, s, p, a), [chat, session, pending_state, auth_state],
                      outputs).then(record_ticket, [chat, last_draft, tickets_state], [tickets_state, tickets_md])
    reject_btn.click(lambda h, s, p, a: decide(False, h, s, p, a), [chat, session, pending_state, auth_state], outputs)
    pending_state.change(lambda p: p if p else gr.skip(), pending_state, last_draft)
    prefs_refresh_btn.click(refresh_preferences, [session, auth_state], [prefs_panel, prefs_feedback])
    prefs_clear_btn.click(clear_preferences, [session, auth_state], [prefs_panel, prefs_feedback])
    # Page-only helpers: none of them talks to the app.
    for button in suggest_btns:
        button.click(lambda text=button.value: text, None, msg)
    nav.change(show_page, nav, pages)
    status_btn.click(lambda: (STATUS_QUESTION, NAV_PAGES[0]), None, [msg, nav])
    help_chat_btn.click(lambda: NAV_PAGES[0], None, nav)
    new_btn.click(None, None, None, js=NEW_CHAT)
    demo.load(signed_in_text, None, signed_in)
    demo.load(profile_card, None, profile)

if __name__ == "__main__":
    import users                      # ui/users.py: the demo login accounts (python ui/users.py add-all)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not users.account_emails():
        raise SystemExit("No login accounts yet. Create them first: python ui/users.py add-all")
    demo.launch(server_port=int(os.getenv("UI_PORT", "7860")), theme=THEME, css=CSS,
                auth=users.check_login, auth_message=LOGIN_HTML)
