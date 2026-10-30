"""Which error codes in a message are the employee's own error, and which are only mentioned for comparison.

"My VPN shows NX-504. Is this the same as NX-503?" has ONE actual error (NX-504); NX-503 is a reference. Deterministic,
conservative, no model. A code is a REFERENCE only when a comparison word sits directly in front of it ("same as",
"different from", "not", ...). In a sentence that compares things, a code that is not clearly the employee's own
("shows NX-504", "fails with NX-504") is UNCERTAIN. Everything else is ACTUAL, so "NX-503 and NX-112 both fail" keeps
both. When no code is clearly actual and some are uncertain, the caller asks which code the employee sees.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent.retrieval import strict_codes

# Same shape as nodes._LOOSE_CODE: also "nx 503" / "NX503", so a loosely written referenced code is recognised too.
_CODE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,5})[-\s]?(\d{2,5})(?![A-Za-z0-9])", re.ASCII)
_SENTENCES = re.compile(r"(?<=[.!?;])\s+|\n+")

_FILLER = r"(?:(?:the|this|that|an?|other|old|earlier|previous|known|error|code|one|message|number)\s+){0,3}"
# A comparison word directly in front of the code: the code is a reference.
_REFERENCE_BEFORE = re.compile(
    r"\b(?:same\s+as|similar\s+to|different\s+(?:from|than|to)|differs?\s+from|compared?\s+(?:to|with)|versus|vs\.?|"
    r"unlike|instead\s+of|rather\s+than|other\s+than|not)\s+" + _FILLER + r"$", re.I)
# A verb or phrase that says the code is what the employee sees.
_OWNED_BEFORE = re.compile(
    r"\b(?:shows?|showing|shown|displays?|displaying|gets?|getting|got|sees?|seeing|saw|receiv(?:e|es|ed|ing)|"
    r"have|having|hit|hitting|with|my\s+(?:error|code)|error\s+code)\s+(?:(?:the|same|an?|error|code|message|number)\s+){0,2}$",
    re.I)
# The sentence compares things.
_COMPARES = re.compile(
    r"\b(?:the\s+same|same\s+as|similar|differen(?:t|ce)|differs?|compar\w*|versus|vs|unlike|instead\s+of|"
    r"rather\s+than|other\s+than)\b", re.I)


@dataclass(frozen=True)
class CodeSplit:
    actual: tuple[str, ...] = ()          # the employee's own error codes
    referenced: tuple[str, ...] = ()      # mentioned only for comparison or background
    uncertain: tuple[str, ...] = ()       # in a comparison, not clearly the employee's own (only when nothing is actual)

    @property
    def not_actual(self) -> frozenset[str]:
        return frozenset((*self.referenced, *self.uncertain))


def _normalise(match: re.Match) -> str:
    return f"{match.group(1).upper()}-{match.group(2)}"


def classify_codes(text: str) -> CodeSplit:
    kinds: dict[str, set[str]] = {}
    order: list[str] = []
    for sentence in _SENTENCES.split(text or ""):
        compares = bool(_COMPARES.search(sentence))
        for match in _CODE.finditer(sentence):
            code, before = _normalise(match), sentence[:match.start()]
            if _REFERENCE_BEFORE.search(before):
                kind = "referenced"
            elif compares and not _OWNED_BEFORE.search(before):
                kind = "uncertain"
            else:
                kind = "actual"
            kinds.setdefault(code, set()).add(kind)
            if code not in order:
                order.append(code)
    actual = [c for c in order if "actual" in kinds[c]]
    referenced = [c for c in order if c not in actual and "referenced" in kinds[c]]
    uncertain = [c for c in order if c not in actual and c not in referenced]
    # With one clearly actual code, the other codes of a comparison are references, not a reason to ask.
    if set(actual) & set(strict_codes(text)):
        referenced, uncertain = [*referenced, *uncertain], []
    return CodeSplit(tuple(actual), tuple(referenced), tuple(uncertain))


def is_ambiguous(split: CodeSplit, text: str) -> bool:
    """True when a real (strict-form) code is mentioned but none is clearly the employee's own."""
    strict = set(strict_codes(text))
    return not (set(split.actual) & strict) and bool(set(split.uncertain) & strict)


def without_codes(text: str, codes) -> str:
    """The text with these codes (any written form) removed, so a comparison code can't match anything later."""
    drop = set(codes)
    if not drop:
        return text or ""
    return _CODE.sub(lambda m: "" if _normalise(m) in drop else m.group(0), text or "")
