"""Exact error-code extraction and whole-token matching (Phase 2B). Pure functions: no I/O, no network.

Error codes in the dataset follow the front-matter schema: 2-4 capital letters, a hyphen, 2-4 digits
(NX-781, NXM-421, SWC-0404, PRN-07). Leading zeros are significant: SWC-0404 is not SWC-404.

A string in a query is treated as a code only if
  * it is not embedded in a longer identifier (no letter, digit, "_" or "-" directly before it, and no
    letter, digit, "_" or "-<digit>" directly after it), and
  * its letter prefix is one of the code prefixes present in the index (so KB-004, SP-001, "ext 4357" or
    "Windows 11" are never codes).
User-typed variants (nx-781, "NX 781", nx781) are normalised to NX-781. Codes are never fuzzy-matched.
"""
from __future__ import annotations

import re
from typing import Iterable

_QUERY_CODE = re.compile(
    r"(?<![A-Za-z0-9_-])([A-Za-z]{2,4})[-\u00a0 ]?(\d{2,4})(?![A-Za-z0-9_]|-\d)", re.ASCII)
_CANONICAL = re.compile(r"[A-Z]{2,4}-\d{2,4}", re.ASCII)


def code_prefixes(codes: Iterable[str]) -> frozenset[str]:
    """Letter prefixes of the codes that exist in the index."""
    return frozenset(code.split("-", 1)[0] for code in codes if _CANONICAL.fullmatch(code))


def extract_codes(query: str, prefixes: Iterable[str]) -> tuple[str, ...]:
    """Canonical codes in the query, de-duplicated, in order of first appearance."""
    if not isinstance(query, str):
        return ()
    known = frozenset(prefixes)
    found: dict[str, None] = {}
    for match in _QUERY_CODE.finditer(query):
        prefix = match.group(1).upper()
        if prefix in known:
            found.setdefault(f"{prefix}-{match.group(2)}", None)
    return tuple(found)


def token_pattern(code: str) -> re.Pattern[str]:
    """Matches the canonical code as a whole token in source text (case-sensitive, hyphen required)."""
    if not _CANONICAL.fullmatch(code):
        raise ValueError(f"not a canonical error code: {code!r}")
    return re.compile(r"(?<![A-Za-z0-9_-])" + re.escape(code) + r"(?![A-Za-z0-9_]|-\d)", re.ASCII)


def codes_in_text(text: str, codes: Iterable[str]) -> tuple[str, ...]:
    """The given codes that occur in text as whole tokens, in the order given."""
    return tuple(code for code in codes if token_pattern(code).search(text))
