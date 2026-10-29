"""The Gradio UI login (ui/users.py + app_ui.py): demo accounts with one shared password, each employee signing in
with their own email, and the chat always talking for the email that signed in.

Offline: a temporary accounts database, the real company directory (read only), no app, no LLM, no network.
"""
from __future__ import annotations

import asyncio
import importlib.util
import secrets
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

KIT = Path(__file__).resolve().parents[1]


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, KIT / "ui" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


users = load("ui_users_login", "users.py")
ui = load("app_ui_login", "app_ui.py")

ACTIVE = sorted(users.active_employees())
TARIQ, RANA, HIBA = "tariq.hamdan@nakhla.example", "rana.khalil@nakhla.example", "hiba.qasem@nakhla.example"


@pytest.fixture(autouse=True)
def accounts_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UI_USERS_DB", str(tmp_path / "users.db"))
    return tmp_path / "users.db"


@pytest.fixture
def shared():
    """A demo password for this test only (never a real or committed one)."""
    return "demo-" + secrets.token_urlsafe(12)


def locked_employee() -> str:
    import json
    data = json.loads(users.DIRECTORY.read_text(encoding="utf-8"))
    rows = data["employees"] if isinstance(data, dict) else data
    return next(row["email"].lower() for row in rows if row.get("account_status") != "active")


# ------------------------------------------------------------------ one shared password, separate emails

def test_several_employees_sign_in_with_the_same_shared_password(shared):
    users.set_password([TARIQ, RANA, HIBA], shared)
    assert all(users.check_login(email, shared) for email in (TARIQ, RANA, HIBA))
    assert users.check_login("Tariq.Hamdan@Nakhla.Example ", shared)          # email case and spaces do not matter


def test_a_wrong_password_or_an_unknown_email_is_refused(shared):
    users.set_password([TARIQ], shared)
    assert not users.check_login(TARIQ, shared + "x")
    assert not users.check_login(TARIQ, "")
    assert not users.check_login("stranger@nakhla.example", shared)
    assert not users.check_login(RANA, shared)                                 # active, but no account yet


def test_only_active_directory_employees_can_get_an_account(shared):
    with pytest.raises(ValueError):
        users.set_password(["stranger@nakhla.example"], shared)
    with pytest.raises(ValueError):
        users.set_password([locked_employee()], shared)
    assert users.account_emails() == []


def test_an_employee_locked_after_the_account_was_made_cannot_sign_in(shared, accounts_db):
    users.set_password([TARIQ], shared)
    with sqlite3.connect(accounts_db) as conn:                    # an account that predates the directory lock
        conn.execute("INSERT INTO users SELECT ?, salt, password_hash FROM users WHERE email = ?",
                     (locked_employee(), TARIQ))
    assert not users.check_login(locked_employee(), shared)


def test_passwords_are_stored_only_as_salted_scrypt_hashes(shared, accounts_db):
    users.set_password([TARIQ, RANA], shared)
    assert shared.encode() not in accounts_db.read_bytes()
    with sqlite3.connect(accounts_db) as conn:
        rows = conn.execute("SELECT salt, password_hash FROM users").fetchall()
    assert len({salt for salt, _ in rows}) == 2 and len({digest for _, digest in rows}) == 2   # same password
    assert all(len(bytes.fromhex(digest)) == 32 for _, digest in rows)


def test_a_short_password_is_refused(shared):
    with pytest.raises(ValueError):
        users.set_password([TARIQ], "short")


# ------------------------------------------------------------------ the setup command

def test_add_all_creates_every_active_employee_with_one_hidden_prompt(shared, monkeypatch, capsys):
    prompts = []
    monkeypatch.setattr(users.getpass, "getpass", lambda prompt="": prompts.append(prompt) or shared)
    assert users.main(["add-all"]) == 0
    assert len(prompts) == 1                                                   # asked once, hidden
    assert users.account_emails() == ACTIVE and locked_employee() not in users.account_emails()
    assert shared not in capsys.readouterr().out
    assert all(users.check_login(email, shared) for email in ACTIVE)


def test_add_one_employee_and_refuse_an_unknown_one(shared, monkeypatch):
    monkeypatch.setattr(users.getpass, "getpass", lambda prompt="": shared)
    assert users.main(["add", TARIQ]) == 0 and users.account_emails() == [TARIQ]
    assert users.main(["add", "stranger@nakhla.example"]) == 1 and users.account_emails() == [TARIQ]


def test_the_password_is_not_in_the_source_or_the_config():
    for path in (KIT / "ui" / "users.py", KIT / "ui" / "app_ui.py", KIT / ".env.example"):
        text = path.read_text(encoding="utf-8").lower()
        assert "demo_password" not in text and "shared_password =" not in text


# ------------------------------------------------------------------ each session talks for its signed-in email

def signed_in(email: str):
    return SimpleNamespace(username=email)


def test_each_session_sends_its_own_signed_in_email(monkeypatch):
    sent = []
    monkeypatch.setattr(ui, "call_app", lambda path, body, token=None: (sent.append(body) or {"reply": "ok"}, None))
    ui.send("VPN fails", [], "ui-" + "a" * 32, "", None, request=signed_in(TARIQ))
    ui.send("VPN fails", [], "ui-" + "b" * 32, "", None, request=signed_in("Rana.Khalil@nakhla.example"))
    assert [b["user"]["email"] for b in sent] == [TARIQ, RANA]
    assert [b["session_id"] for b in sent] == ["ui-" + "a" * 32, "ui-" + "b" * 32]


def test_the_streaming_chat_also_uses_the_signed_in_email(monkeypatch):
    sent = []

    async def fake_turn(body, token, timing=None):
        sent.append(body)
        yield "final", {"reply": "ok"}, None

    monkeypatch.setattr(ui, "ws_turn", fake_turn)

    async def run(email, session_id):
        async for _ in ui.send_stream("hello", [], session_id, "", None, request=signed_in(email)):
            pass

    asyncio.run(run(HIBA, "ui-" + "c" * 32))
    asyncio.run(run(TARIQ, "ui-" + "d" * 32))
    assert [b["user"]["email"] for b in sent] == [HIBA, TARIQ]


def test_a_typed_email_cannot_replace_the_signed_in_one():
    assert ui.login_email("hiba.qasem@nakhla.example", signed_in(TARIQ)) == TARIQ
    assert ui.login_email("", None) == ""                                      # no login: no identity is invented


def test_the_header_shows_who_is_signed_in_with_a_logout_link():
    text = ui.signed_in_text(signed_in(TARIQ))
    assert TARIQ in text and "/logout" in text
    assert ui.signed_in_text(None) == ""
