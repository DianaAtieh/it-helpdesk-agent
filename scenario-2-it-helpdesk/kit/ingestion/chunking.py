"""Deterministic, section-aware chunking of loaded knowledge-base documents (Phase 1B).

Design rules:
- Pages are hard boundaries: a chunk never spans two pages (``PageText``).
- Markdown is split on ``#`` headings; PDFs use a narrow heading rule (see ``_is_pdf_heading``).
- A section is packed from whole blocks (paragraph, list item, table row, blockquote); a block is
  only cut when it alone exceeds ``max_chars``.
- ``Chunk.text`` is always the exact slice ``PageText.text[char_start:char_end]``. Repeated
  headings and table headers are retrieval context only and live in ``Chunk.context``; they are
  never mixed into ``text``, so offsets and citations stay faithful to the source.
- Overlap repeats whole trailing blocks of the previous chunk, only inside one section.
- Nothing is filtered by status: draft and deprecated documents are chunked like any other, and
  their ``status`` is carried on every chunk. Document text is data and is never interpreted.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

from ingestion.loaders import Document

CHUNKER_VERSION = "1"

# A PDF line shorter than this is assumed not to be a wrapped line (the PDFs wrap at ~100 chars),
# so the following plain line is not treated as a continuation of a list item.
_PDF_WRAP_MIN = 90
_PDF_HEADING_MAX = 70

_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_PDF_NUMBERED = re.compile(r"^(\d+)\.[ \t]+[A-Z]")
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)[ \t]+")
_TABLE_SEP = re.compile(r"^\|[ \t:|-]+\|[ \t]*$")
_SENTENCE_END = re.compile(r"[.!?](?=\s)")


@dataclass(frozen=True)
class ChunkConfig:
    """``max_chars`` bounds the source text of a chunk; ``context`` is added on top of it."""
    max_chars: int = 1200      # ~300 tokens; 104 of 107 dataset sections fit in one chunk
    overlap_chars: int = 150   # ~12% of max_chars: carries one step or sentence over a split

    def __post_init__(self) -> None:
        if self.max_chars < 100:
            raise ValueError(f"max_chars must be at least 100, got {self.max_chars}")
        if not 0 <= self.overlap_chars < self.max_chars // 2:
            raise ValueError(
                f"overlap_chars must be >= 0 and < max_chars // 2 ({self.max_chars // 2}), got {self.overlap_chars}")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str                  # {doc_id}:p{page}:{ordinal:03d}:{hash8}
    doc_id: str
    ordinal: int                   # position within the document, 0-based
    page: int                      # 1-based page the text came from (Markdown: 1)
    char_start: int                # offsets into PageText.text of that page
    char_end: int
    text: str                      # exactly PageText.text[char_start:char_end]
    heading_path: tuple[str, ...]  # headings enclosing the chunk, outermost first
    context: str                   # ADDED retrieval context, not source text
    content_hash: str              # sha256 of text
    title: str
    version: str
    last_updated: str
    status: str
    authoritative: bool
    owner: str
    supersedes: str | None
    applies_to: tuple[str, ...]
    tags: tuple[str, ...]
    error_codes: tuple[str, ...]   # the document's error codes that occur in this chunk's text
    source_path: Path
    relative_path: str
    format: str
    doc_sha256: str

    @property
    def contextual_text(self) -> str:
        """Context followed by the source text: what a later embedding step should encode."""
        return f"{self.context}\n\n{self.text}"


@dataclass(frozen=True)
class _Unit:
    """A block of source text; ``header`` is the (start, end) of its table header, if any."""
    start: int
    end: int
    header: tuple[int, int] | None = None


@dataclass
class _Section:
    path: tuple[str, ...]
    heading: _Unit | None
    units: list[_Unit]


def _is_pdf_heading(line: str, expected: int, has_next: bool) -> bool:
    """Narrow PDF rule: ``Change log``, or a short, unpunctuated, sequentially numbered line."""
    if not has_next:
        return False
    if line.strip() == "Change log":
        return True
    match = _PDF_NUMBERED.match(line)
    return bool(match and int(match.group(1)) == expected and len(line) <= _PDF_HEADING_MAX
                and not line.rstrip().endswith((".", ":", ";", ",")))


class _PageParser:
    """Splits one page into sections of blocks, keeping the heading stack across pages."""

    def __init__(self, fmt: str) -> None:
        self.fmt = fmt
        self.stack: list[tuple[int, str]] = []
        self.expected_number = 1

    def path(self) -> tuple[str, ...]:
        return tuple(h for _, h in self.stack)

    def _heading(self, lines: list[tuple[int, int, str]], i: int) -> tuple[int, str] | None:
        raw = lines[i][2]
        if self.fmt == "md":
            m = _MD_HEADING.match(raw)
            return (len(m.group(1)), m.group(2).strip()) if m else None
        has_next = any(line[2].strip() for line in lines[i + 1:])
        return (1, raw.strip()) if _is_pdf_heading(raw, self.expected_number, has_next) else None

    def _kind(self, lines: list[tuple[int, int, str]], i: int) -> str:
        raw = lines[i][2]
        if not raw.strip():
            return "blank"
        if self._heading(lines, i):
            return "heading"
        if raw.startswith("|"):
            return "table"
        if raw.startswith(">"):
            return "quote"
        if _LIST_ITEM.match(raw):
            return "item"
        return "text"

    def parse(self, text: str) -> list[_Section]:
        lines: list[tuple[int, int, str]] = []
        pos = 0
        for raw in text.split("\n"):
            lines.append((pos, pos + len(raw), raw))
            pos += len(raw) + 1
        sections = [_Section(self.path(), None, [])]
        i, n = 0, len(lines)
        while i < n:
            kind = self._kind(lines, i)
            if kind == "blank":
                i += 1
            elif kind == "heading":
                level, title = self._heading(lines, i)  # type: ignore[misc]
                if self.fmt == "pdf" and _PDF_NUMBERED.match(lines[i][2]):
                    self.expected_number += 1
                while self.stack and self.stack[-1][0] >= level:
                    self.stack.pop()
                self.stack.append((level, title))
                sections.append(_Section(self.path(), _Unit(lines[i][0], lines[i][1]), []))
                i += 1
            elif kind == "table":
                j = i
                while j < n and self._kind(lines, j) == "table":
                    j += 1
                sections[-1].units.extend(self._table_units(lines, i, j))
                i = j
            elif kind == "quote":
                j = i
                while j < n and self._kind(lines, j) == "quote":
                    j += 1
                sections[-1].units.append(_Unit(lines[i][0], lines[j - 1][1]))
                i = j
            elif kind == "item":
                j = i + 1
                while j < n and self._kind(lines, j) == "text" and self._continues(lines, j):
                    j += 1
                sections[-1].units.append(_Unit(lines[i][0], lines[j - 1][1]))
                i = j
            else:
                j = i
                while j < n and self._kind(lines, j) == "text":
                    j += 1
                sections[-1].units.append(_Unit(lines[i][0], lines[j - 1][1]))
                i = j
        for section in sections:
            if section.heading is not None:
                if section.units:  # glue the heading to its first block so it is never orphaned
                    section.units[0] = replace(section.units[0], start=section.heading.start)
                else:
                    section.units.append(section.heading)
        return [s for s in sections if s.units]

    def _continues(self, lines: list[tuple[int, int, str]], j: int) -> bool:
        return self.fmt == "md" or len(lines[j - 1][2]) >= _PDF_WRAP_MIN

    @staticmethod
    def _table_units(lines: list[tuple[int, int, str]], i: int, j: int) -> list[_Unit]:
        header = None
        first = i
        if j - i >= 2 and _TABLE_SEP.match(lines[i + 1][2]):
            header = (lines[i][0], lines[i + 1][1])
            first = i + 2  # the header and separator travel with the first data row
        units = []
        if header and first < j:
            units.append(_Unit(lines[i][0], lines[first][1], header))
            first += 1
        elif header:
            units.append(_Unit(lines[i][0], lines[i + 1][1], header))
        units.extend(_Unit(lines[k][0], lines[k][1], header) for k in range(first, j))
        return units


def _split_oversized(page: str, unit: _Unit, max_chars: int) -> list[_Unit]:
    """Cut a block longer than max_chars at the best boundary; pieces keep exact offsets."""
    pieces: list[_Unit] = []
    pos = unit.start
    while unit.end - pos > max_chars:
        window = page[pos:pos + max_chars]
        cut = _best_cut(window, max_chars // 4)
        end = pos + cut
        # Whitespace at the cut stays in the previous piece (while it fits), so pieces tile the block exactly.
        while end < unit.end and end - pos < max_chars and page[end].isspace():
            end += 1
        pieces.append(replace(unit, start=pos, end=end))
        pos = end
    pieces.append(replace(unit, start=pos, end=unit.end))
    return pieces


def _best_cut(window: str, minimum: int) -> int:
    """Prefer a line end after a sentence, then a sentence end, then any line end, then a space."""
    def last(positions: list[int]) -> int | None:
        usable = [p for p in positions if p >= minimum]
        return max(usable) if usable else None

    newline_after_sentence = [m.start() for m in re.finditer(r"(?<=[.!?:;])\n", window)]
    sentence_ends = [m.end() for m in _SENTENCE_END.finditer(window)]
    newlines = [m.start() for m in re.finditer(r"\n", window)]
    spaces = [m.start() for m in re.finditer(r"[ \t]", window)]
    for candidates in (newline_after_sentence, sentence_ends, newlines, spaces):
        cut = last(candidates)
        if cut is not None:
            return cut
    return len(window)  # no whitespace at all: hard cut


def _group_units(units: list[_Unit], config: ChunkConfig) -> list[list[_Unit]]:
    """Pack consecutive units into groups of at most max_chars, with whole-block overlap."""
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    for unit in units:
        if current and unit.end - current[0].start > config.max_chars:
            groups.append(current)
            current = _overlap(current, unit, config)
        current.append(unit)
    groups.append(current)
    return groups


def _overlap(previous: list[_Unit], upcoming: _Unit, config: ChunkConfig) -> list[_Unit]:
    carried: list[_Unit] = []
    for unit in reversed(previous):
        if (previous[-1].end - unit.start > config.overlap_chars
                or upcoming.end - unit.start > config.max_chars):
            break
        carried.insert(0, unit)
    return carried


def _context(doc: Document, path: tuple[str, ...], first: _Unit, page: str) -> str:
    parts = [f"{doc.id}: {doc.title}"] + [h for h in path if not h.startswith(doc.id)]
    context = " > ".join(parts)
    if first.header and first.start > first.header[0]:  # table continues: repeat its header
        context += "\n" + page[first.header[0]:first.header[1]]
    return context


def chunk_document(doc: Document, config: ChunkConfig = ChunkConfig()) -> list[Chunk]:
    """Chunk one document deterministically. Draft and deprecated documents are chunked too."""
    chunks: list[Chunk] = []
    parser = _PageParser(doc.format)
    for page in doc.pages:
        for section in parser.parse(page.text):
            units: list[_Unit] = []
            for unit in section.units:
                if unit.end - unit.start > config.max_chars:
                    units.extend(_split_oversized(page.text, unit, config.max_chars))
                else:
                    units.append(unit)
            for group in _group_units(units, config):
                start, end = group[0].start, group[-1].end
                text = page.text[start:end]
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                ordinal = len(chunks)
                digest = hashlib.sha256("\x1f".join(
                    [CHUNKER_VERSION, doc.id, doc.version, str(page.page), str(start), str(end), content_hash]
                ).encode("utf-8")).hexdigest()[:8]
                chunks.append(Chunk(
                    chunk_id=f"{doc.id}:p{page.page}:{ordinal:03d}:{digest}", doc_id=doc.id, ordinal=ordinal,
                    page=page.page, char_start=start, char_end=end, text=text, heading_path=section.path,
                    context=_context(doc, section.path, group[0], page.text), content_hash=content_hash,
                    title=doc.title, version=doc.version, last_updated=doc.last_updated, status=doc.status,
                    authoritative=doc.authoritative, owner=doc.owner, supersedes=doc.supersedes,
                    applies_to=doc.applies_to, tags=doc.tags,
                    error_codes=tuple(c for c in doc.error_codes if c in text),
                    source_path=doc.source_path, relative_path=doc.relative_path, format=doc.format,
                    doc_sha256=doc.sha256,
                ))
    return chunks


def chunk_documents(docs: list[Document], config: ChunkConfig = ChunkConfig()) -> list[Chunk]:
    """Chunk every document in the given order; nothing is filtered by status."""
    return [chunk for doc in docs for chunk in chunk_document(doc, config)]
