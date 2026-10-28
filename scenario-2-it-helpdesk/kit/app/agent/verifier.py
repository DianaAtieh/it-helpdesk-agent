"""Citation and fact verifier (Phase 3): deterministic, no model, no network, no ticket writes.

A valid document ID is not proof that an answer is grounded. For a model-written answer, this checks the parts
that CAN be checked mechanically:

  * inline document references (KB-xxx, SP-xxx) must be documents in this turn's evidence, or the SP document of
    a code-issued policy citation; a sentence with any other reference is removed;
  * concrete facts in each sentence (error codes, versions, dates, URLs and hostnames, numbers with a unit,
    "quoted" or `backticked` names) must appear in the evidence text or in the employee's own message (an
    employee's "I have version 5.1" may be repeated back); a sentence with an unsupported fact is removed;
  * a documented threshold ("three or more lockouts in one day ... report them") must not be restated with a
    vague amount ("multiple lockouts ... report it"); such a sentence is removed (see "weakened thresholds");
  * if more than half of the checkable facts were unsupported, or what remains is empty or trivial, the whole
    reply becomes the caller's honest fallback;
  * a citation is dropped when the only facts it supported were in removed sentences.

What it does NOT do: judge paraphrased claims that contain no concrete fact ("restart the laptop", "approve the
prompt"). That would need semantic matching or another model call. The caller applies this only to model-written
answers; fixed replies, refusals, clarifications, incident replies and ticket drafts are not passed in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- normalisation

_CURLY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "ʼ": "'"})
_BOLD = re.compile(r"\*+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "ninety": 90, "hundred": 100,
}
_UNITS = {
    "second": "second", "seconds": "second", "sec": "second", "secs": "second",
    "minute": "minute", "minutes": "minute", "min": "minute", "mins": "minute",
    "hour": "hour", "hours": "hour", "hr": "hour", "hrs": "hour",
    "day": "day", "days": "day", "business day": "business day", "business days": "business day",
    "week": "week", "weeks": "week", "month": "month", "months": "month", "year": "year", "years": "year",
    "character": "character", "characters": "character", "user": "user", "users": "user",
    "attempt": "attempt", "attempts": "attempt", "sign-in": "sign-in", "sign-ins": "sign-in",
    "code": "code", "codes": "code", "password": "password", "passwords": "password",
    "gb": "gb", "mb": "mb", "%": "%", "percent": "%",
}

_DOC_REF = re.compile(r"\b(?:KB|SP)-\d{3}\b")
_ERROR_CODE = re.compile(r"(?<![A-Za-z0-9_-])([A-Z]{2,5}-\d{2,5})(?![A-Za-z0-9_]|-\d)")
_VERSION = re.compile(r"(?<![\w.])(\d+\.\d+(?:\.\d+)*)(?![\w]|\.\d)")
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_URL = re.compile(r"https?://[^\s)>\]\"'`]+", re.I)
_HOST = re.compile(r"(?<![\w@./-])((?:[a-z0-9-]+\.)+(?:example|com|net|org|io))(?![\w-])", re.I)
_QUOTED = re.compile(r'"([^"\n]{2,60})"|`([^`\n]{2,60})`')
_UNIT_WORDS = "|".join(sorted((re.escape(u) for u in _UNITS), key=len, reverse=True))
_NUMBER = r"\d+|" + "|".join(_NUMBER_WORDS)
_NUMBER_UNIT = re.compile(rf"(?i)\b({_NUMBER})\s*-?\s*({_UNIT_WORDS})(?![\w-])")


def _clean(text: str) -> str:
    """Curly quotes straightened, bold marks removed, whitespace collapsed (backticks are kept for _QUOTED)."""
    return " ".join(_BOLD.sub("", (text or "").translate(_CURLY)).split())


def _url_key(url: str) -> str:
    url = re.sub(r"(?i)^https?://", "", url.strip())
    return url.rstrip("/.,;:!?").lower()


def _number_units(text: str) -> set[str]:
    out = set()
    for number, unit in _NUMBER_UNIT.findall(text):
        value = int(number) if number.isdigit() else _NUMBER_WORDS[number.lower()]
        out.add(f"{value} {_UNITS[unit.lower()]}")
    return out


def facts_in(text: str) -> set[str]:
    """Every checkable fact in a text, as normalised keys (kind:value). Inline document IDs are separate."""
    text = _clean(text)
    keys = {f"code:{c}" for c in _ERROR_CODE.findall(text) if not _DOC_REF.fullmatch(c) and not c.startswith("IT-")}
    keys |= {f"version:{v}" for v in _VERSION.findall(text)}
    keys |= {f"date:{d}" for d in _DATE.findall(text)}
    urls = _URL.findall(text)
    keys |= {f"url:{_url_key(u)}" for u in urls}
    without_urls = _URL.sub(" ", text)
    keys |= {f"host:{h.lower()}" for h in _HOST.findall(without_urls)}
    keys |= {f"amount:{a}" for a in _number_units(text)}
    keys |= {f"name:{(q or b).strip().lower()}" for q, b in _QUOTED.findall(text)}
    return keys


# ---------------------------------------------------------------- weakened thresholds
#
# A documented threshold ("Three or more lockouts in one day ... report them to Security Operations") must not be
# restated with a vague amount ("If you have multiple lockouts, report it to Security Operations"): the sentence
# has no number, so the fact check above cannot see it. Deliberately narrow:
#   * only thresholds written "N or more X", "more than N X" or "at least N X" in this turn's evidence;
#   * not time or size units ("more than 30 days", "at least 14 characters" are already number facts);
#   * a reply sentence counts only if it uses a vague amount for the SAME noun and also states the consequence
#     that the evidence attaches to the threshold (an anchor word from the evidence's threshold sentence:
#     report, Security Operations, a priority, treated as ...). "To stop repeated lockouts, remove old saved
#     passwords" names no consequence and is kept;
#   * a sentence that carries the threshold's own number is kept ("three or more lockouts"); an incidental number
#     from the same evidence sentence ("in one day") does not count.
# Such a sentence is removed like any unsupported one; nothing is rewritten or added.

_THRESHOLD_NUMBER = rf"(?i)\b(?:({_NUMBER})\s+or\s+more|more\s+than\s+({_NUMBER})|at\s+least\s+({_NUMBER}))"
_THRESHOLD = re.compile(_THRESHOLD_NUMBER + r"\s+([a-z][a-z-]+)")
_NOT_THRESHOLD_NOUNS = {"second", "minute", "hour", "day", "week", "month", "year", "time", "character"}
_VAGUE = re.compile(r"(?i)\b(?:multiple|several|repeated|numerous|many|frequent|a\s+few|a\s+number\s+of)"
                    r"\s+([a-z][a-z-]+)(?:\s+([a-z][a-z-]+))?")       # the noun, or a word and then the noun
_ANCHORS = ("report", "security operations", "treated as", "considered", "escalat", "critical",
            "p1", "p2", "p3", "p4")


def _singular(word: str) -> str:
    word = word.lower()
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def _number_forms(text: str) -> set[str]:
    """Every number in a text, as digits and as words ("3", "three")."""
    words = {v: k for k, v in _NUMBER_WORDS.items()}
    out = set()
    for token in re.findall(rf"(?i)\b(?:{_NUMBER})\b", text):
        value = int(token) if token.isdigit() else _NUMBER_WORDS[token.lower()]
        out |= {str(value), words.get(value, str(value))}
    return out


_SUPERSEDED = re.compile(r"(?i)\b(?:was|were|previously|formerly)\s*$")


def _threshold_numbers(sentence: str) -> set[str]:
    """Only the numbers that define the current threshold ("Three or more", "more than 3", "(4 or more)"), never
    incidental ones in the same sentence ("in one day", "within 15 minutes") or a superseded one ("(was 5 or
    more)" in KB-015's change log)."""
    return {forms for match in re.finditer(_THRESHOLD_NUMBER, sentence)
            if not _SUPERSEDED.search(sentence[:match.start()])
            for number in match.groups() if number for forms in _number_forms(number)}


def thresholds_in(texts: list[str]) -> dict[str, tuple[set[str], set[str]]]:
    """{noun: (numbers, anchors)} for every documented threshold: the numbers that define the threshold and the
    consequence words of the evidence sentence that states it."""
    out: dict[str, tuple[set[str], set[str]]] = {}
    for text in texts:
        for sentence in _SENTENCE_END.split(_clean(text)):
            for match in _THRESHOLD.finditer(sentence):
                noun = _singular(match.group(4))
                if noun in _NOT_THRESHOLD_NOUNS:
                    continue
                lower = sentence.lower()
                anchors = {a for a in _ANCHORS if re.search(rf"\b{a}", lower)}
                if not anchors:
                    continue                         # a threshold with no stated consequence is not checked
                numbers, known = out.setdefault(noun, (set(), set()))
                numbers |= _threshold_numbers(sentence)
                known |= anchors
    return out


def weakened(sentence: str, thresholds: dict[str, tuple[set[str], set[str]]]) -> list[str]:
    """The nouns whose documented threshold this sentence restates with a vague amount ([] if none)."""
    clean = _clean(sentence)
    lower = clean.lower()
    found = []
    for match in _VAGUE.finditer(clean):
        noun = next((_singular(w) for w in match.groups() if w and _singular(w) in thresholds), None)
        if noun is None:
            continue
        numbers, anchors = thresholds[noun]
        if not any(re.search(rf"\b{a}", lower) for a in anchors):
            continue                                 # not about the threshold's consequence
        if _number_forms(clean) & numbers:
            continue                                 # the documented number is there
        found.append(noun)
    return found


class _Haystack:
    """The facts a source (the evidence, or the employee's message) supports."""

    def __init__(self, texts: list[str]) -> None:
        cleaned = [_clean(t) for t in texts]
        self.keys: set[str] = set()
        for text in cleaned:
            self.keys |= facts_in(text)
        self.lower = " ".join(cleaned).lower().replace("`", "")
        self.hosts = {_url_key(u).split("/")[0] for t in cleaned for u in _URL.findall(t)}

    def supports(self, key: str) -> bool:
        if key in self.keys:
            return True
        kind, value = key.split(":", 1)
        if kind == "name":                       # a quoted name only needs to appear somewhere in the text
            return value in self.lower
        if kind == "host":                       # a hostname is supported by a URL on that host
            return value in self.hosts
        if kind == "url":                        # a URL is supported by the same URL with a longer path
            return any(k.startswith(f"url:{value}") for k in self.keys)
        return False


# ---------------------------------------------------------------- verification

@dataclass(frozen=True)
class Verified:
    reply: str                                   # the checked reply (or the fallback)
    citations: list[dict]                        # the citations still supported
    fallback: bool                               # True if the honest fallback replaced the reply
    removed: list[str] = field(default_factory=list)       # sentences removed
    unsupported: list[str] = field(default_factory=list)   # fact keys / references that were not supported
    checked: int = 0                                       # how many checkable facts and references there were


MIN_REMAINING_CHARS = 25


_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*$")


def _split(reply: str) -> list[list[str]]:
    """Lines, each a list of sentences: the line structure of the reply is kept. A list marker such as "3." stays
    with the sentence after it, so removing that sentence never leaves the marker behind."""
    lines = []
    for line in reply.split("\n"):
        pieces, marker = [], ""
        for piece in _SENTENCE_END.split(line):
            if not piece.strip():
                continue
            if _LIST_MARKER.match(piece):
                marker = f"{marker}{piece.strip()} "
                continue
            pieces.append(f"{marker}{piece}")
            marker = ""
        if marker:
            pieces.append(marker.strip())
        lines.append(pieces)
    return lines


def verify_answer(reply: str, citations: list[dict], evidence: list[dict], employee_text: str, *,
                  fallback_reply: str, policy_citations: list[dict] | tuple = ()) -> Verified:
    """Check a model-written answer against this turn's evidence and the employee's message (see the module text).

    citations:        the answer's citations ({"id", ...}), already limited to documents in this turn's evidence
    evidence:         this turn's evidence records ({"doc_id", "text", ...})
    employee_text:    the employee's message(s) for this turn: their own facts may be repeated back
    fallback_reply:   the caller's honest fallback, used when too little of the answer is supported
    policy_citations: code-issued SP citations ({"id"}): their documents may be referred to inline
    """
    evidence = [e for e in evidence or [] if isinstance(e, dict)]
    by_doc: dict[str, list[str]] = {}
    for record in evidence:
        by_doc.setdefault(str(record.get("doc_id", "")), []).append(str(record.get("text", "")))
    all_evidence = _Haystack([t for texts in by_doc.values() for t in texts])
    employee = _Haystack([employee_text or ""])
    doc_haystacks = {doc_id: _Haystack(texts) for doc_id, texts in by_doc.items()}
    known_docs = set(by_doc) | {str(c.get("id")) for c in policy_citations or [] if isinstance(c, dict)}
    thresholds = thresholds_in([t for texts in by_doc.values() for t in texts])

    removed, unsupported, kept_lines = [], [], []
    checked = 0
    removed_facts: set[str] = set()
    kept_facts: set[str] = set()
    for line in _split(reply or ""):
        kept = []
        for sentence in line:
            clean = _clean(sentence)
            refs = set(_DOC_REF.findall(clean))
            facts = facts_in(clean)
            weak = [f"weakened:{noun}" for noun in weakened(clean, thresholds)]
            checked += len(refs) + len(facts) + len(weak)
            bad = sorted(r for r in refs if r not in known_docs)
            bad += sorted(f for f in facts if not (all_evidence.supports(f) or employee.supports(f)))
            bad += weak
            if bad:
                unsupported += bad
                removed.append(sentence.strip())
                removed_facts |= facts
            else:
                kept.append(sentence.strip())
                kept_facts |= facts
        if kept:
            kept_lines.append(" ".join(kept))
    text = "\n".join(kept_lines).strip()

    # A citation goes when it supported facts only in removed sentences, and none in what is left.
    def still_supported(citation: dict) -> bool:
        haystack = doc_haystacks.get(str(citation.get("id")))
        if haystack is None:
            return False
        helped_removed = any(haystack.supports(f) for f in removed_facts)
        helps_kept = any(haystack.supports(f) for f in kept_facts)
        return helps_kept or not helped_removed

    kept_citations = [c for c in citations or [] if isinstance(c, dict) and still_supported(c)]

    too_unsupported = checked > 0 and len(unsupported) * 2 > checked
    # "Trivial" is about what is LEFT after removals: a short answer that lost nothing is kept as it is.
    trivial = bool(removed) and (len(text) < MIN_REMAINING_CHARS or not re.search(r"[A-Za-z]{3,}", text))
    trivial = trivial or not text.strip()
    lost_citations = bool(citations) and not kept_citations
    if too_unsupported or trivial or lost_citations:
        return Verified(fallback_reply, [], True, removed, unsupported, checked)
    return Verified(text, kept_citations, False, removed, unsupported, checked)
