"""Load and validate knowledge-base documents (Markdown and PDF).

Document text is treated as untrusted data: it is only parsed (yaml.safe_load), never executed.
All documents are loaded, including deprecated and draft ones; ``status`` is kept explicit so
later stages can enforce the precedence rules.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

KIT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = KIT_DIR / "dataset"
KB_DIR = DATASET_DIR / "knowledge_base"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
SCHEMA_PATH = DATASET_DIR / "schemas" / "kb_front_matter.schema.json"

SUPPORTED_SUFFIXES = (".md", ".pdf")
TEXT_SUFFIXES = {".md"}
MANIFEST_META_KEYS = ("title", "version", "last_updated", "status", "authoritative", "owner",
                      "supersedes", "applies_to", "tags", "error_codes")
_FRONT_MATTER_RE = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL | re.MULTILINE)


# running page footer, e.g. "Nakhla Systems (fictional) · KB-012 v1.4 · current · Owner: X · page 2 of 2"
_PAGE_FOOTER_RE = re.compile(r"^Nakhla Systems \(fictional\).*\bpage \d+ of \d+[ \t]*$", re.MULTILINE)


class DocumentLoadError(ValueError):
    """A document or the dataset is invalid; the message names the file and the problem."""


@dataclass(frozen=True)
class PageText:
    """Cleaned text of one source page (1-based); Markdown documents have a single page 1."""
    page: int
    text: str


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    version: str
    last_updated: str
    status: str
    authoritative: bool
    owner: str
    supersedes: str | None
    applies_to: tuple[str, ...]
    tags: tuple[str, ...]
    error_codes: tuple[str, ...]
    source_path: Path
    relative_path: str          # as in the manifest, e.g. knowledge_base/KB-002-....md
    format: str                 # "md" or "pdf"
    pages: tuple[PageText, ...]  # per-page text for citation-aware chunking
    body: str                    # all pages joined by a blank line
    sha256: str

    @property
    def page_refs(self) -> tuple[int, ...]:
        """1-based pages holding body text; (1,) for Markdown."""
        return tuple(p.page for p in self.pages)


def file_sha256(path: Path) -> str:
    """SHA-256 of the file bytes; text files use LF line endings (same as the manifest)."""
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_schema(schema_path: Path = SCHEMA_PATH) -> Draft202012Validator:
    try:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise DocumentLoadError(f"{schema_path}: cannot read schema ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise DocumentLoadError(f"{schema_path}: schema is not valid JSON ({exc})") from exc
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    try:
        return json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise DocumentLoadError(f"{manifest_path}: cannot read manifest ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise DocumentLoadError(f"{manifest_path}: manifest is not valid JSON ({exc})") from exc


def _split_front_matter(text: str, path: Path, anchored: bool) -> tuple[dict, str, tuple[int, int]]:
    """Return (metadata, text after the block, span of the block in the normalised text)."""
    match = _FRONT_MATTER_RE.match(text) if anchored else _FRONT_MATTER_RE.search(text)
    if match is None:
        where = "start of the file" if anchored else "DOCUMENT CONTROL block on page 1"
        raise DocumentLoadError(f"{path}: no front matter between '---' lines at the {where}")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise DocumentLoadError(f"{path}: front matter is not valid YAML ({exc})") from exc
    if not isinstance(meta, dict):
        raise DocumentLoadError(f"{path}: front matter must be a YAML mapping")
    for key, value in list(meta.items()):
        if isinstance(value, (dt.date, dt.datetime)):
            meta[key] = value.isoformat()
    return meta, text[match.end():], match.span()


def _clean(body: str) -> str:
    return "\n".join(line.rstrip() for line in body.strip().split("\n"))


def _read_markdown(path: Path) -> tuple[dict, tuple[PageText, ...]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"{path}: not valid UTF-8 ({exc})") from exc
    meta, body, _ = _split_front_matter(text.replace("\r\n", "\n"), path, anchored=True)
    return meta, (PageText(1, _clean(body)),)


def _read_pdf(path: Path) -> tuple[dict, tuple[PageText, ...]]:
    try:
        import pymupdf
    except ImportError as exc:
        raise DocumentLoadError(f"{path}: PyMuPDF is not installed (pip install -r requirements.txt)") from exc
    try:
        with pymupdf.open(path) as doc:
            pages = [page.get_text().replace("\r\n", "\n") for page in doc]
    except Exception as exc:  # noqa: BLE001 - pymupdf raises several error types
        raise DocumentLoadError(f"{path}: cannot read PDF ({exc})") from exc
    if not pages:
        raise DocumentLoadError(f"{path}: PDF has no pages")
    meta, _, (start, end) = _split_front_matter(pages[0], path, anchored=False)
    # drop the DOCUMENT CONTROL box (heading and YAML) from page 1; keep the rest
    pages[0] = pages[0][:start].replace("DOCUMENT CONTROL", "") + pages[0][end:]
    result = []
    for number, text in enumerate(pages, start=1):
        text = _clean(_PAGE_FOOTER_RE.sub("", text))
        if text:
            result.append(PageText(number, text))
    if not result:
        raise DocumentLoadError(f"{path}: PDF has no text after the DOCUMENT CONTROL block")
    return meta, tuple(result)


def load_document(path: Path, validator: Draft202012Validator | None = None,
                  dataset_dir: Path = DATASET_DIR) -> Document:
    """Load one Markdown or PDF document and validate its metadata against the schema."""
    path = Path(path)
    if not path.is_file():
        raise DocumentLoadError(f"{path}: file not found")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentLoadError(f"{path}: unsupported format {suffix!r} (expected .md or .pdf)")
    meta, pages = _read_markdown(path) if suffix == ".md" else _read_pdf(path)
    body = "\n\n".join(p.text for p in pages if p.text)

    validator = validator or load_schema()
    errors = sorted(validator.iter_errors(meta), key=lambda e: [str(p) for p in e.path])
    if errors:
        details = "; ".join(f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:5])
        raise DocumentLoadError(f"{path}: front matter invalid: {details}")
    if not path.name.startswith(meta["id"] + "-"):
        raise DocumentLoadError(f"{path}: file name must start with its id {meta['id']}-")
    if not body:
        raise DocumentLoadError(f"{path}: document body is empty")
    for code in meta["error_codes"]:
        if code not in body:
            raise DocumentLoadError(f"{path}: error code {code} is in the front matter but not in the text")
    if meta["id"].startswith("SP-") and not (meta["status"] == "current" and meta["authoritative"]):
        raise DocumentLoadError(f"{path}: security policies must be current and authoritative")

    try:
        relative = path.resolve().relative_to(Path(dataset_dir).resolve()).as_posix()
    except ValueError:
        relative = f"knowledge_base/{path.name}"
    return Document(
        id=meta["id"], title=meta["title"], version=meta["version"], last_updated=meta["last_updated"],
        status=meta["status"], authoritative=meta["authoritative"], owner=meta["owner"],
        supersedes=meta["supersedes"], applies_to=tuple(meta["applies_to"]), tags=tuple(meta["tags"]),
        error_codes=tuple(meta["error_codes"]), source_path=path, relative_path=relative,
        format=suffix[1:], pages=pages, body=body, sha256=file_sha256(path),
    )


def validate_against_manifest(documents: list[Document], manifest: dict) -> None:
    """Check ids, metadata, file paths and SHA-256 hashes against manifest.json."""
    problems: list[str] = []
    ids = [d.id for d in documents]
    for doc_id in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"duplicate document id {doc_id}")
    listed = {entry["id"]: entry for entry in manifest.get("documents", [])}
    if set(listed) != set(ids):
        problems.append(f"manifest and files disagree on ids: {sorted(set(listed) ^ set(ids))}")
    by_id = {d.id: d for d in documents}
    for doc_id, entry in listed.items():
        doc = by_id.get(doc_id)
        if doc is None:
            continue
        for key in MANIFEST_META_KEYS:
            value = getattr(doc, key)
            value = list(value) if isinstance(value, tuple) else value
            if entry.get(key) != value:
                problems.append(f"{doc_id}: manifest {key} = {entry.get(key)!r}, file says {value!r}")
        if entry.get("file") != doc.relative_path:
            problems.append(f"{doc_id}: manifest file {entry.get('file')!r}, loaded {doc.relative_path!r}")
        if entry.get("format") != doc.format:
            problems.append(f"{doc_id}: manifest format {entry.get('format')!r}, file is {doc.format!r}")
        if entry.get("sha256") != doc.sha256:
            problems.append(f"{doc_id}: sha256 does not match the manifest (file changed?)")
    for doc in documents:
        if doc.supersedes is not None and doc.supersedes not in by_id:
            problems.append(f"{doc.id}: supersedes unknown document {doc.supersedes}")
    if problems:
        raise DocumentLoadError("manifest validation failed:\n  - " + "\n  - ".join(problems))


def load_knowledge_base(kb_dir: Path = KB_DIR, manifest_path: Path = MANIFEST_PATH,
                        schema_path: Path = SCHEMA_PATH) -> list[Document]:
    """Load every document (any status), validate against schema and manifest, sorted by id."""
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        raise DocumentLoadError(f"{kb_dir}: knowledge base directory not found")
    manifest = load_manifest(manifest_path)
    validator = load_schema(schema_path)
    paths = sorted(p for p in kb_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    expected = manifest.get("counts", {}).get("documents")
    if expected is not None and len(paths) != expected:
        raise DocumentLoadError(f"{kb_dir}: found {len(paths)} documents, manifest expects {expected}")
    dataset_dir = Path(manifest_path).resolve().parent
    documents = [load_document(p, validator, dataset_dir) for p in paths]
    validate_against_manifest(documents, manifest)
    return sorted(documents, key=lambda d: d.id)
