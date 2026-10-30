"""Demo login accounts for the Gradio UI (ui/app_ui.py): company email + password, checked before the chat opens.

UI only: this protects the web page, not the API. The backend, the API contract, the evals and services/ are unchanged.
An account can exist only for an ACTIVE employee of the company directory (services/data/employees.json, read only),
and every login checks the directory again, so a locked employee cannot sign in. For an easy demo, all employees may
share one demo password, but each still signs in with their own email, and that email is who the chat talks for.

Passwords are stored as salted hashlib.scrypt hashes (a new random salt per account) in a small SQLite file:
ui/users.db, or UI_USERS_DB. The plain-text password is never stored, logged, or kept in the code, .env or the repo.

    python ui/users.py add-all        # every active employee, one shared demo password (asked once, hidden)
    python ui/users.py add <email>    # one employee (password asked, hidden)
    python ui/users.py list           # the emails that have an account (never a password or a hash)
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
DIRECTORY = KIT / "services" / "data" / "employees.json"
MIN_PASSWORD_CHARS = 8
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}


def db_path() -> Path:
    """UI_USERS_DB (read when called, so tests can point it elsewhere), else ui/users.db."""
    path = Path(os.getenv("UI_USERS_DB") or KIT / "ui" / "users.db")
    return path if path.is_absolute() else KIT / path


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, salt TEXT NOT NULL, "
                 "password_hash TEXT NOT NULL)")
    return conn


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT).hex()


def active_employees(directory: Path = DIRECTORY) -> set[str]:
    """The lower-case emails of the directory's ACTIVE employees (read only)."""
    data = json.loads(Path(directory).read_text(encoding="utf-8"))
    rows = data.get("employees", []) if isinstance(data, dict) else data
    return {str(row["email"]).strip().lower() for row in rows
            if isinstance(row, dict) and row.get("email") and row.get("account_status") == "active"}


def set_password(emails: list[str], password: str) -> int:
    """Create or update the accounts of `emails` with this password. Returns how many. Raises ValueError for a
    password shorter than MIN_PASSWORD_CHARS or an email that is not an active employee."""
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_CHARS:
        raise ValueError(f"the password must have at least {MIN_PASSWORD_CHARS} characters")
    allowed = active_employees()
    emails = [email.strip().lower() for email in emails]
    unknown = [email for email in emails if email not in allowed]
    if unknown:
        raise ValueError(f"not an active employee in the directory: {', '.join(unknown)}")
    with closing(_connect()) as conn, conn:
        for email in emails:
            salt = secrets.token_bytes(16)          # its own salt: the same password gives a different hash
            conn.execute("INSERT INTO users (email, salt, password_hash) VALUES (?, ?, ?) ON CONFLICT (email) "
                         "DO UPDATE SET salt = excluded.salt, password_hash = excluded.password_hash",
                         (email, salt.hex(), _hash(password, salt)))
    return len(emails)


def account_emails() -> list[str]:
    with closing(_connect()) as conn:
        return [row[0] for row in conn.execute("SELECT email FROM users ORDER BY email")]


def check_login(username: str, password: str) -> bool:
    """Gradio's auth function: True only for an active employee with an account and the right password."""
    if not isinstance(username, str) or not isinstance(password, str) or not password:
        return False
    email = username.strip().lower()
    try:
        with closing(_connect()) as conn:
            row = conn.execute("SELECT salt, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    except sqlite3.Error:
        return False
    if row is None:
        _hash(password, b"\0" * 16)                 # about the same time as a real check: no hint the email exists
        return False
    if email not in active_employees():             # locked (or removed) in the directory since the account was made
        return False
    return hmac.compare_digest(_hash(password, bytes.fromhex(row[0])), row[1])


def main(argv: list[str]) -> int:
    try:
        if argv == ["add-all"]:
            emails = sorted(active_employees())
            password = getpass.getpass(f"Shared demo password for {len(emails)} active employees: ")
            print(f"Created or updated {set_password(emails, password)} accounts in {db_path()}.")
        elif len(argv) == 2 and argv[0] == "add":
            password = getpass.getpass(f"Password for {argv[1]}: ")
            set_password([argv[1]], password)
            print(f"Created or updated the account of {argv[1].strip().lower()}.")
        elif argv == ["list"]:
            print("\n".join(account_emails()) or "No accounts yet.")
        else:
            print(__doc__.split("\n\n")[-1])
            return 2
    except ValueError as exc:
        print(f"Nothing was changed: {exc}.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
