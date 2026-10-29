"""Long-term user preferences: a small, deterministic store plus a rule-based detector (Phase 1, standalone).

What is stored: for one employee (the normalised email), up to three PRESENTATION preferences, each a value from a
fixed list. Nothing else: no chat text, no free-form values. The whitelist is enforced here, in Python, on every
write, so no caller (UI, API or model output) can store anything outside it.

    language   en
    verbosity  short | normal | detailed
    format     numbered | bullets | prose

The store is its own SQLite file (app/data/preferences.db, or PREFERENCES_DB), separate from the checkpoint
database, so a session reset or the checkpoint clean-up can never touch it. Every operation is scoped to one
employee; there is no "delete everything" operation. A preference expires 180 days after its last update.

extract_preferences() finds CLEAR, PERSISTENT requests in a message with fixed rules (no model, no network). It
prefers to miss a preference over saving a wrong one: temporary, negated, quoted, hypothetical, ambiguous,
conflicting, secret-looking and instruction-like messages save nothing.

Identity note: without a login the email is only a claim (see main.py), so these presentation-only values are the
only thing that may hang off it. They must never influence facts, policy, routing or approvals.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping

from app.guardrails import looks_like_instructions, redact_secrets

KIT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PREFERENCES_DB = KIT / "app" / "data" / "preferences.db"
EXPIRY_DAYS = 180
MAX_MESSAGE_CHARS = 1000

ALLOWED: dict[str, frozenset[str]] = {
    "language": frozenset({"en"}),
    "verbosity": frozenset({"short", "normal", "detailed"}),
    "format": frozenset({"numbered", "bullets", "prose"}),
}

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"          # fixed width, so text order equals time order
_EMAIL = re.compile(r"^[^@\s,;<>\"']+@[^@\s,;<>\"']+\.[^@\s,;<>\"']+$")


class PreferenceError(ValueError):
    """An invalid employee identifier, preference key or preference value."""


def preferences_db_path() -> Path:
    """PREFERENCES_DB (read when called, so tests can point it elsewhere); relative paths are under the kit."""
    path = Path(os.getenv("PREFERENCES_DB") or DEFAULT_PREFERENCES_DB)
    return path if path.is_absolute() else KIT / path


def normalize_email(email: str) -> str:
    """The identifier used for every lookup: trimmed and lower case. Raises PreferenceError if not an email."""
    if not isinstance(email, str):
        raise PreferenceError("the employee identifier must be an email address")
    value = email.strip().lower()
    if len(value) > 254 or not _EMAIL.match(value):
        raise PreferenceError("the employee identifier must be an email address")
    return value


def validate_preference(key: str, value: str) -> tuple[str, str]:
    """(key, value) if both are on the whitelist, else PreferenceError. Exact lower-case values only."""
    if not isinstance(key, str) or key not in ALLOWED:
        raise PreferenceError(f"unsupported preference key: {str(key)[:40]!r}")
    if not isinstance(value, str) or value not in ALLOWED[key]:
        raise PreferenceError(f"unsupported value for {key}: {str(value)[:40]!r}")
    return key, value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("the clock must return a timezone-aware datetime")
    return moment.astimezone(timezone.utc).strftime(_TIME_FORMAT)


class PreferenceStore:
    """The preferences database. Safe to share between threads and between instances of the same file.

    Each operation opens its own short-lived connection (SQLite serialises writers; a busy timeout and a
    process-wide lock per file make concurrent requests wait instead of failing). Calls are synchronous: async
    callers should run them with asyncio.to_thread. Every statement is parameterised.
    """

    _locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, path: str | os.PathLike | None = None, clock: Callable[[], datetime] | None = None) -> None:
        self.path = Path(path) if path is not None else preferences_db_path()
        self._clock = clock or _utc_now
        with self._locks_guard:
            self._lock = self._locks.setdefault(str(self.path.resolve()), threading.Lock())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    # ------------------------------------------------------------------ plumbing

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.execute("PRAGMA busy_timeout = 15000")
        return conn

    def init_schema(self) -> None:
        """Create the table if it is missing. Safe to call any number of times."""
        with self._lock, closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS user_preferences ("
                    " employee_email TEXT NOT NULL,"
                    " pref_key       TEXT NOT NULL,"
                    " pref_value     TEXT NOT NULL,"
                    " updated_at     TEXT NOT NULL,"
                    " PRIMARY KEY (employee_email, pref_key))")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user_preferences_updated "
                             "ON user_preferences (updated_at)")

    def _cutoff(self) -> str:
        return _stamp(self._clock() - timedelta(days=EXPIRY_DAYS))

    # ------------------------------------------------------------------ operations

    def set(self, email: str, key: str, value: str) -> None:
        """Save or update one preference (its 180 days start again)."""
        self.set_many(email, {key: value})

    def set_many(self, email: str, preferences: Mapping[str, str]) -> dict[str, str]:
        """Save or update several preferences in one transaction. Everything is validated first, so one bad
        entry saves nothing. Returns what was saved."""
        employee = normalize_email(email)
        checked = dict(validate_preference(key, value) for key, value in preferences.items())
        if not checked:
            return {}
        now = _stamp(self._clock())
        with self._lock, closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO user_preferences (employee_email, pref_key, pref_value, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (employee_email, pref_key) "
                "DO UPDATE SET pref_value = excluded.pref_value, updated_at = excluded.updated_at",
                [(employee, key, value, now) for key, value in checked.items()])
        return checked

    def get(self, email: str) -> dict[str, str]:
        """This employee's current preferences ({} if none). Expired ones are never returned and are deleted."""
        employee = normalize_email(email)
        cutoff = self._cutoff()
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM user_preferences WHERE employee_email = ? AND updated_at < ?",
                         (employee, cutoff))
            rows = conn.execute("SELECT pref_key, pref_value FROM user_preferences WHERE employee_email = ?",
                                (employee,)).fetchall()
        # A value outside the whitelist (a hand-edited file) is dropped, never passed on.
        return {key: value for key, value in sorted(rows) if key in ALLOWED and value in ALLOWED[key]}

    def delete(self, email: str, keys: list[str] | tuple[str, ...] | None = None) -> int:
        """Delete this employee's preferences (all of them, or only `keys`). Never touches another employee.
        Returns how many rows were deleted."""
        employee = normalize_email(email)
        if keys is None:
            sql, args = "DELETE FROM user_preferences WHERE employee_email = ?", (employee,)
        else:
            for key in keys:
                if key not in ALLOWED:
                    raise PreferenceError(f"unsupported preference key: {str(key)[:40]!r}")
            if not keys:
                return 0
            marks = ",".join("?" * len(keys))
            sql, args = (f"DELETE FROM user_preferences WHERE employee_email = ? AND pref_key IN ({marks})",
                         (employee, *keys))
        with self._lock, closing(self._connect()) as conn, conn:
            return conn.execute(sql, args).rowcount

    def purge_expired(self) -> int:
        """Delete every preference last updated more than 180 days ago. Returns how many."""
        with self._lock, closing(self._connect()) as conn, conn:
            return conn.execute("DELETE FROM user_preferences WHERE updated_at < ?", (self._cutoff(),)).rowcount


# ---------------------------------------------------------------------- explicit preference detection

_SENTENCES = re.compile(r"(?<=[.!?;])\s+|\n+")
# Text inside quotes, backticks or brackets is someone's words or an example, never the employee's own request.
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|‘[^’]*’|`[^`]*`|(?<![\w])'[^']*'(?![\w])|\([^)]*\)|\[[^\]]*\]")

# The request must say it lasts: "remember that I prefer", "from now on", "always", "in future" ...
_PERSISTENT = re.compile(
    r"(?i)\b(remember\s+(that|to)|please\s+remember|keep\s+in\s+mind\s+that|from\s+now\s+on|from\s+here\s+on|"
    r"going\s+forward|in\s+(the\s+)?future|for\s+(all\s+)?future|every\s+time|each\s+time|by\s+default|"
    r"as\s+a\s+default|permanently|always|i\s+(always|generally|consistently)\s+prefer|"
    r"my\s+(standing|permanent|default)\s+preference)\b")
# ... or a clear first-person statement of a standing preference: "I prefer short answers.", "I want answers in
# bullet points.", "I would like detailed answers.", "I'd like numbered steps." It counts only when it is about
# answers in general (a plural target such as "answers" or "steps", not "this answer"), so a one-off command
# ("Keep answers short", "Make it shorter") or "I want a short answer" is still not saved. Every other check below
# (temporary, negated, hypothetical, quoted, policy/ticket topics, secrets, steering) still applies to it.
_NATURAL = re.compile(r"(?i)\bi\s+(?:really\s+)?(?:prefer|want|would\s+like)\b|\bi['’]d\s+(?:really\s+)?like\b")
_GENERIC_TARGET = re.compile(r"(?i)\b(answers|replies|responses|explanations|instructions|steps|bullets|"
                             r"bullet[- ]points|paragraphs|prose)\b")
_OFF_TOPIC_TARGET = re.compile(r"(?i)\b(about|regarding|concerning)\b")      # "detailed answers about the VPN"
# "would" is also a hypothetical word below; in "I would like" / "I'd like" it is the preference itself.
_WOULD_LIKE = re.compile(r"(?i)\bi\s+would\s+(?:really\s+)?like\b|\bi['’]d\s+(?:really\s+)?like\b")


def _lasting(sentence: str) -> bool:
    """The sentence says the preference lasts: an explicit phrase, or a clear first-person statement about answers."""
    if _PERSISTENT.search(sentence):
        return True
    return bool(_NATURAL.search(sentence) and _GENERIC_TARGET.search(sentence)
                and not _OFF_TOPIC_TARGET.search(sentence))


def _for_checks(sentence: str) -> str:
    """The sentence as the temporary / negation / hypothetical checks see it ("I would like" -> "I like")."""
    return _WOULD_LIKE.sub("I like", sentence)


_TEMPORARY = re.compile(
    r"(?i)\b(for\s+(this|that|the\s+next|now|today|tonight|a\s+(moment|while|minute))|this\s+(one|time|answer|"
    r"reply|response|question|ticket|issue|conversation|session|chat)|just\s+(this|now|once)|only\s+(this|now|"
    r"once|today)|right\s+now|at\s+the\s+moment|for\s+once|one[- ]time|temporar\w+|next\s+(answer|reply|"
    r"response|time)|today|tonight)\b")
_NEGATION = re.compile(
    r"(?i)\b(not|no|never|none|nor|without|stop|stopped|cease|don'?t|dont|doesn'?t|didn'?t|won'?t|wouldn'?t|"
    r"can'?t|cannot|isn'?t|aren'?t|wasn'?t|instead\s+of|rather\s+than|unless|except|no\s+longer)\b|n['’]t\b")
_HYPOTHETICAL = re.compile(
    r"(?i)\b(if|would|could|might|may|maybe|perhaps|possibly|suppose|supposing|imagine|hypothetical\w*|"
    r"what\s+if|should|can\s+you|could\s+you|whether|wish|someday|sometimes|sometime|occasionally|usually|"
    r"often|typically|when\s+i\s+say|example|e\.g\.|how\s+(do|can|to|would|about)|what\s+(is|does|about)|"
    r"why|who|which)\b")
# Words that make a sentence about something other than presentation: never saved from such a sentence.
_FORBIDDEN_TOPIC = re.compile(
    r"(?i)\b(ignore|disregard|override|bypass|skip|disable|circumvent|jailbreak|policy|policies|rules?|"
    r"guardrails?|approv\w*|reject\w*|system|prompt|admin\w*|credentials?|passwords?|passwd|passcode|token|"
    r"secret|api[- ]?keys?|ticket\w*|escalat\w*|security|permission\w*|authori[sz]\w*|verif\w*|cite|citations?|"
    r"knowledge\s+base|answer\s+anything|pretend|roleplay|act\s+as|say|says|claim|insist|lie|tell\s+me\s+that)\b")
# One of these ANYWHERE in the message means someone is steering the assistant: the whole message saves nothing.
_STEERING = re.compile(
    r"(?i)\b(ignore|disregard|override|bypass|circumvent|disable|jailbreak|guardrails?|system\s+prompt|pretend|"
    r"roleplay|act\s+as|without\s+(approval|verif\w*|checking))\b")

_VALUE_RULES: dict[str, list[tuple[str, re.Pattern]]] = {
    "verbosity": [
        ("short", re.compile(r"(?i)\b(short|shorter|brief|briefly|concise|succinct|terse|compact)\b")),
        ("detailed", re.compile(r"(?i)\b(detailed|thorough|in[- ]depth|comprehensive|verbose|long|longer|"
                                r"elaborate)\b")),
        ("normal", re.compile(r"(?i)\b(normal|standard|medium|regular)[- ](length|detail|size)\b|"
                              r"\bdefault\s+(length|detail)\b")),
    ],
    "format": [
        ("numbered", re.compile(r"(?i)\b(numbered|step[- ]by[- ]step|numerically|ordered\s+(list|steps))\b")),
        ("bullets", re.compile(r"(?i)\b(bullets?|bulleted|bullet[- ]points?)\b")),
        ("prose", re.compile(r"(?i)\b(prose|paragraphs?|plain\s+text|flowing\s+text)\b")),
    ],
    "language": [
        ("en", re.compile(r"(?i)\b(in|using|use|with)\s+english\b|\benglish\s+(only|language|please)\b|"
                          r"\b(explain|answer|respond|reply|write|speak|communicate)\w*\s+(\w+\s+){0,4}english\b")),
    ],
}
# These words make "short" / "long" a statement about the answer's length, not about something else.
_ABOUT_ANSWERS = re.compile(r"(?i)\b(answers?|replies|reply|responses?|explanations?|explain\w*|steps?|"
                            r"instructions?|troubleshoot\w*|guides?|summar\w*|messages?|texts?|write|keep|"
                            r"give|make|be|use|show|format|list\w*)\b")


# The global injection filter (app.guardrails, unchanged) flags "from now on" followed by always / never / answer /
# respond ..., which also matches a plain style request ("From now on, answer in bullet points."). For preference
# detection only, that lead-in is set aside before the filter runs when the next word is a style verb, so the rest
# of the message is still checked in full. "From now on, ignore your rules" and "from now on you are ..." are not
# style verbs, stay flagged here, and save nothing. The message itself is still handled by main.py as before.
_PREFERENCE_LEAD_IN = re.compile(
    r"(?i)\bfrom\s+now\s+on,?\s+(?=(?:always|never|answer|respond|reply|use|give|write|keep|explain|format)\b)")


def _clean(message: str) -> str | None:
    """The message without quoted parts, or None if it must not be mined at all."""
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
        return None
    if redact_secrets(message) != message:                        # looks like it contains a secret
        return None
    if looks_like_instructions(_PREFERENCE_LEAD_IN.sub("", message)):
        return None                                               # aimed at the assistant's own rules
    if _STEERING.search(message):
        return None
    return _QUOTED.sub(" ", message)


def _values_in(sentence: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for key, rules in _VALUE_RULES.items():
        matches = [value for value, pattern in rules if pattern.search(sentence)]
        if len(matches) == 1:                                     # two different values = ambiguous = none
            found[key] = matches[0]
    if "verbosity" in found and not _ABOUT_ANSWERS.search(sentence):
        del found["verbosity"]
    return found


def extract_preferences(message: str) -> dict[str, str]:
    """The persistent preferences the employee clearly asked for in this message, as {key: value} ({} if none).

    A sentence counts only if it says the preference lasts ("remember that", "from now on", "always" ...) and is
    not temporary, negated, a question, hypothetical, about policy/tickets/security, or ambiguous. Every
    supported preference in a qualifying sentence is returned. The same key asked for with two different values
    in one message is dropped. Messages with secrets, instruction-like text or quotes around the request save
    nothing. The result is always valid for PreferenceStore.set_many."""
    text = _clean(message)
    if text is None:
        return {}
    result: dict[str, str] = {}
    conflicted: set[str] = set()
    for sentence in _SENTENCES.split(text):
        sentence = sentence.strip()
        checked = _for_checks(sentence)
        if (not sentence or "?" in sentence or not _lasting(sentence)
                or _TEMPORARY.search(checked) or _NEGATION.search(checked)
                or _HYPOTHETICAL.search(checked) or _FORBIDDEN_TOPIC.search(checked)):
            continue
        for key, value in _values_in(sentence).items():
            if result.get(key, value) != value:
                conflicted.add(key)
            result[key] = value
    return {key: value for key, value in result.items() if key not in conflicted
            and value in ALLOWED.get(key, ())}


# Words that may remain in a preference sentence once the persistence phrase ("from now on") and the preference
# values ("short", "bullet points", "in English") are taken out. Anything else, such as "VPN", "laptop", an error
# code or a number, means the sentence also says something else, so the message is mixed, not preference-only.
_FILLER = frozenset("""
a an the my me i it them that to and also please all your you with in of on
prefer prefers preferred preference preferences like want need set make keep use using give show be write
respond reply replies answer answers answering explain explaining explanation explanations
step steps point points instruction instructions troubleshoot troubleshooting guide guides message messages response responses
format formatting style text list lists summary summaries
""".split())
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def _residual_words(sentence: str) -> list[str]:
    rest = _NATURAL.sub(" ", _PERSISTENT.sub(" ", sentence))
    for rules in _VALUE_RULES.values():
        for _, pattern in rules:
            rest = pattern.sub(" ", rest)
    return [word for word in _WORD.findall(rest.lower()) if word not in _FILLER]


def is_preference_only(message: str) -> bool:
    """True only if the WHOLE message is a clear, persistent preference request and nothing else.

    Deliberately conservative, built on the same checks as extract_preferences: every sentence must be a
    qualifying preference sentence (lasting, not temporary / negated / a question / hypothetical / about
    policy or tickets, with at least one supported value) AND contain no other words than preference vocabulary.
    Quotes or brackets anywhere, secrets, instruction-like text or steering words, an over-long message, or any
    second topic (an IT problem, an error code, a ticket request) make it False: such a message takes the normal
    route, and extract_preferences may still save its clear preference sentences."""
    if not isinstance(message, str) or _QUOTED.search(message) or _clean(message) is None:
        return False
    sentences = [s.strip() for s in _SENTENCES.split(message) if s.strip()]
    if not sentences:
        return False
    for sentence in sentences:
        checked = _for_checks(sentence)
        if (not sentence or "?" in sentence or not _lasting(sentence)
                or _TEMPORARY.search(checked) or _NEGATION.search(checked)
                or _HYPOTHETICAL.search(checked) or _FORBIDDEN_TOPIC.search(checked)):
            return False
        if not _values_in(sentence) or _residual_words(sentence):
            return False
    return bool(extract_preferences(message))


# ---------------------------------------------------------------------- what the agent and the employee see

_LABELS = {"language": "language", "verbosity": "answer length", "format": "answer format"}
_VALUE_WORDS = {"en": "English", "short": "short", "normal": "normal", "detailed": "detailed",
                "numbered": "numbered steps", "bullets": "bullet points", "prose": "plain paragraphs"}


def safe_presentation(preferences: object) -> dict[str, str]:
    """Only whitelisted key/value pairs of a preferences mapping ({} for anything else). This is the single filter
    between stored or checkpointed data and the model prompt."""
    if not isinstance(preferences, Mapping):
        return {}
    return {key: preferences[key] for key in ALLOWED
            if isinstance(preferences.get(key), str) and preferences[key] in ALLOWED[key]}


def describe_preferences(preferences: Mapping[str, str]) -> str:
    """Fixed wording for a set of preferences, e.g. "answer length: short; answer format: numbered steps"."""
    return "; ".join(f"{_LABELS[key]}: {_VALUE_WORDS[value]}" for key, value in safe_presentation(preferences).items())


# ---------------------------------------------------------------------- what the model may NOT claim
# Only main.py saves preferences, and only its own note says what happened. The model never knows whether a save
# happened, so a sentence in a model answer that promises to remember, save or keep following a style ("I will
# keep answers shorter and use bullet points as you requested.") is removed by the respond node. A sentence counts
# only with BOTH a first-person promise (or a "preference saved" statement) AND a lasting / preference context, so
# ordinary answers ("Remember to save your work.") and a style used in this reply only ("I'll keep this short.")
# are left alone. The backend's own confirmation wording is reserved too: a model sentence carrying it is removed,
# so in the final reply it can only come from main.py.
SAVED_CONFIRMATION = "Saved to your preferences:"
_CLAIM_PROMISE = re.compile(
    r"(?i)\b(i\s*(?:will|'ll|’ll|shall|am\s+going\s+to|have|'ve|’ve)\s+(?:now\s+|also\s+|definitely\s+|always\s+|"
    r"make\s+sure\s+to\s+)*(?:remember|save|store|note|record|keep|use|stick|continue|apply|update|make\s+sure|"
    r"format|write|answer|respond|reply|give|provide)\w*"
    r"|preferences?\s+(?:has|have|is|are)\s+(?:been\s+)?(?:saved|stored|noted|recorded|updated|set)"
    r"|noted\s+your\s+preferences?)\b")
_CLAIM_CONTEXT = re.compile(
    r"(?i)\b(preferences?|prefer\w*|remember\w*|from\s+now\s+on|from\s+here\s+on|going\s+forward|future|"
    r"next\s+time|always|every\s+time|each\s+time|by\s+default|as\s+(?:you\s+)?(?:requested|asked|wished|prefer))\b")
_CLAIM_SPLIT = re.compile(r"(?<=[.!?])\s+")


def is_preference_claim(sentence: str) -> bool:
    """True for a sentence that claims a preference was saved or will be followed in later answers."""
    sentence = sentence or ""
    if SAVED_CONFIRMATION.lower() in sentence.lower():
        return True
    return bool(_CLAIM_PROMISE.search(sentence) and _CLAIM_CONTEXT.search(sentence))


def strip_preference_claims(text: str) -> tuple[str, int]:
    """(the reply without preference-saving claims, how many sentences were removed). The text is returned
    unchanged when there are none; line breaks (numbered steps) are kept and a line that was only a claim goes."""
    if not isinstance(text, str) or not text:
        return text, 0
    removed, lines = 0, []
    for line in text.split("\n"):
        kept = []
        for sentence in _CLAIM_SPLIT.split(line):
            if sentence.strip() and is_preference_claim(sentence):
                removed += 1
            else:
                kept.append(sentence)
        new = " ".join(kept).rstrip()
        if line.strip() and not new.strip():
            continue
        lines.append(new)
    if not removed:
        return text, 0
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), removed


# Asking for an answer style at all, lasting or not: a request word plus a supported value. main.py uses it only to
# say honestly that such a request was NOT saved; it never saves anything itself.
_STYLE_REQUEST = re.compile(
    r"(?i)\b(i\s+(?:really\s+)?(?:want|prefer|like|need|would\s+like)|i['’]d\s+(?:like|prefer)|give\s+me|"
    r"can\s+you|could\s+you|please|make|keep|use|answer|reply|respond|write)\b")


def asks_for_presentation(message: str) -> bool:
    """True if the message asks for an answer length, format or language (whether or not it can be saved)."""
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
        return False
    text = _QUOTED.sub(" ", message)
    return bool(_STYLE_REQUEST.search(text)) and any(_values_in(s) for s in _SENTENCES.split(text) if s.strip())
