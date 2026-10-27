"""Hand-curated rule inventory (Phase 2D): pointers to authoritative rule chunks, validated at load time.

The inventory names, for each department exception and each governing security policy, WHICH indexed chunks
carry the rule. It never contains rule text and never a decision: the Agent receives the original chunk text and
citation. It is data, not logic, so no document, ticket or query can add a rule.

Everything is validated against the real dataset files and the index; any problem raises InventoryError and the
retriever then reports the rule lanes as `unavailable` (never as "no exception applies"):

  * strict structure: unknown keys, wrong types, duplicate ids and oversize files are rejected;
  * every anchor resolves to exactly ONE indexed chunk (document, last heading, quote that occurs in the stored text);
  * anchors and topic selectors point at current, authoritative documents only;
  * departments come from the dataset schema vocabulary, and every department rule matches an entry of
    manifest.json `department_exceptions` (primary anchors must be among that entry's source documents, and every
    manifest exception must have a rule, so Sales and Support, which have none, get none);
  * primary anchors are guaranteed in the answer lane, so their number per rule is bounded.

Topic: a rule is only considered when its topic is established by evidence that is itself eligible (see
policy_retrieval). Topics are explicit selectors (a document, or one section of a document) so that, for example,
an MFA section of SP-002 cannot establish the VPN topic. The reserved topic "*" ("whatever the issue") is
established by any evidence at all and is allowed for department rules only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Iterable, Mapping

DEPARTMENT_EXCEPTION = "department_exception"
GOVERNING_POLICY = "governing_policy"
KINDS = (DEPARTMENT_EXCEPTION, GOVERNING_POLICY)
PRIMARY = "primary"
SUPPORTING = "supporting"
ANY_TOPIC = "*"

MAX_PRIMARY_DEPARTMENT = 3  # primaries per department rule: always returned, so bounded by validation
MAX_PRIMARY_GOVERNING = 2
MAX_FILE_BYTES = 1_000_000
MIN_QUOTE_CHARS = 12

KIT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = Path(__file__).with_name("rule_inventory.json")
DEFAULT_MANIFEST = KIT_DIR / "dataset" / "manifest.json"
DEFAULT_SCHEMA = KIT_DIR / "dataset" / "schemas" / "kb_front_matter.schema.json"

_ID = re.compile(r"^[A-Z0-9][A-Z0-9-]*$")


class InventoryError(ValueError):
    """The rule inventory is missing, malformed, or does not match the dataset and the index."""


@dataclass(frozen=True)
class ChunkRef:
    chunk_id: str
    doc_id: str
    heading_path: tuple[str, ...]
    text: str
    status: str
    authoritative: bool


@dataclass(frozen=True)
class Anchor:
    doc_id: str
    heading: str
    quote: str
    role: str
    chunk_id: str  # resolved against the index


@dataclass(frozen=True)
class Rule:
    rule_id: str
    kind: str
    department: str | None
    topic: str
    anchors: tuple[Anchor, ...]

    @property
    def primaries(self) -> tuple[Anchor, ...]:
        return tuple(a for a in self.anchors if a.role == PRIMARY)


@dataclass(frozen=True)
class RuleInventory:
    version: int
    rules: tuple[Rule, ...]
    topic_chunks: Mapping[str, frozenset[str]]  # topic name -> chunk ids that establish it

    def established_topics(self, signal: Collection[str]) -> frozenset[str]:
        """Topics established by the given chunk ids (ANY_TOPIC whenever the signal is non-empty)."""
        ids = set(signal)
        topics = {name for name, chunk_ids in self.topic_chunks.items() if chunk_ids & ids}
        if ids:
            topics.add(ANY_TOPIC)
        return frozenset(topics)

    def activated(self, kind: str, topics: Collection[str], department: str | None = None) -> tuple[Rule, ...]:
        """Rules of `kind` whose topic is established, in inventory order. A known department selects only its
        own rules (case-insensitive exact match); an unknown department selects all of them."""
        wanted = department.strip().lower() if department is not None else None
        return tuple(rule for rule in self.rules
                     if rule.kind == kind and rule.topic in topics
                     and (wanted is None or rule.department is None or rule.department.lower() == wanted))


def _read_json(path: Path | str, what: str):
    path = Path(path)
    try:
        if path.stat().st_size > MAX_FILE_BYTES * 2:
            raise InventoryError(f"{what} {path.name} is larger than {MAX_FILE_BYTES * 2} bytes")
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InventoryError(f"cannot read {what} {path}: {exc.strerror or exc}") from None
    except ValueError as exc:
        if isinstance(exc, InventoryError):
            raise
        raise InventoryError(f"{what} {path.name} is not valid JSON: {exc}") from None


def _obj(value, where: str, required: Iterable[str], optional: Iterable[str] = ()) -> dict:
    if not isinstance(value, dict):
        raise InventoryError(f"{where} must be an object")
    required, optional = tuple(required), tuple(optional)
    missing = [key for key in required if key not in value]
    unknown = [key for key in value if key not in required + optional]
    if missing:
        raise InventoryError(f"{where} is missing: {', '.join(missing)}")
    if unknown:
        raise InventoryError(f"{where} has unknown keys: {', '.join(sorted(unknown))}")
    return value


def _text(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InventoryError(f"{where} must be a non-empty string")
    return value


def _schema_departments(schema_path: Path | str) -> frozenset[str]:
    schema = _read_json(schema_path, "schema")
    try:
        values = schema["properties"]["applies_to"]["items"]["enum"]
    except (KeyError, TypeError):
        raise InventoryError("schema has no applies_to enum") from None
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise InventoryError("schema applies_to enum is malformed")
    return frozenset(v for v in values if v != "all")


def _manifest_exceptions(manifest_path: Path | str) -> Mapping[str, frozenset[str]]:
    manifest = _read_json(manifest_path, "manifest")
    entries = manifest.get("department_exceptions") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise InventoryError("manifest has no department_exceptions list")
    result: dict[str, frozenset[str]] = {}
    for position, entry in enumerate(entries):
        where = f"manifest department_exceptions[{position}]"
        if not isinstance(entry, dict) or not isinstance(entry.get("department"), str):
            raise InventoryError(f"{where} needs a department")
        source = entry.get("source")
        if not isinstance(source, list) or not source or not all(isinstance(s, str) and s for s in source):
            raise InventoryError(f"{where} needs a non-empty source list")
        result[entry["department"]] = frozenset(source)
    return result


def _resolve(chunks: list[ChunkRef], doc_id: str, heading: str | None, quote: str | None, where: str) -> list[ChunkRef]:
    found = [c for c in chunks if c.doc_id == doc_id and (heading is None or (c.heading_path and c.heading_path[-1] == heading))
             and (quote is None or quote in c.text)]
    for chunk in found:
        if chunk.status != "current" or not chunk.authoritative:
            raise InventoryError(f"{where}: {chunk.chunk_id} is not a current, authoritative chunk "
                                 f"(status={chunk.status}, authoritative={chunk.authoritative})")
    return found


def load_inventory(path: Path | str, chunks: Iterable[ChunkRef], *, manifest_path: Path | str = DEFAULT_MANIFEST,
                   schema_path: Path | str = DEFAULT_SCHEMA) -> RuleInventory:
    chunk_list = list(chunks)
    data = _obj(_read_json(path, "rule inventory"), "inventory", ("inventory_version", "topics", "rules"))
    if data["inventory_version"] != 1 or isinstance(data["inventory_version"], bool):
        raise InventoryError("inventory_version must be 1")
    departments = _schema_departments(schema_path)
    manifest = _manifest_exceptions(manifest_path)

    topics_raw = data["topics"]
    if not isinstance(topics_raw, dict) or not topics_raw:
        raise InventoryError("topics must be a non-empty object")
    topic_chunks: dict[str, frozenset[str]] = {}
    for name, selectors in topics_raw.items():
        if name == ANY_TOPIC or not isinstance(name, str) or not name.strip():
            raise InventoryError(f"invalid topic name {name!r}")
        if not isinstance(selectors, list) or not selectors:
            raise InventoryError(f"topic {name} needs a non-empty list of selectors")
        ids: set[str] = set()
        for position, selector in enumerate(selectors):
            where = f"topic {name}[{position}]"
            item = _obj(selector, where, ("doc_id",), ("heading",))
            found = _resolve(chunk_list, _text(item["doc_id"], where), item.get("heading"), None, where)
            if not found:
                raise InventoryError(f"{where} matches no indexed chunk")
            ids.update(c.chunk_id for c in found)
        topic_chunks[name] = frozenset(ids)

    rules_raw = data["rules"]
    if not isinstance(rules_raw, list) or not rules_raw:
        raise InventoryError("rules must be a non-empty list")
    rules: list[Rule] = []
    seen: set[str] = set()
    for position, raw in enumerate(rules_raw):
        where = f"rules[{position}]"
        item = _obj(raw, where, ("rule_id", "kind", "topic", "anchors"), ("department",))
        rule_id = _text(item["rule_id"], where)
        if not _ID.match(rule_id) or rule_id in seen:
            raise InventoryError(f"{where}: rule_id {rule_id!r} is malformed or duplicated")
        seen.add(rule_id)
        where = rule_id
        kind = item["kind"]
        if kind not in KINDS:
            raise InventoryError(f"{where}: kind must be one of {', '.join(KINDS)}")
        topic = _text(item["topic"], f"{where}.topic")
        if topic != ANY_TOPIC and topic not in topic_chunks:
            raise InventoryError(f"{where}: unknown topic {topic!r}")
        department = item.get("department")
        if kind == DEPARTMENT_EXCEPTION:
            if department not in departments:
                raise InventoryError(f"{where}: department {department!r} is not in the schema vocabulary")
            if department not in manifest:
                raise InventoryError(f"{where}: manifest.json has no department_exceptions entry for {department}")
        else:
            if department is not None:
                raise InventoryError(f"{where}: a governing policy must not name a department")
            if topic == ANY_TOPIC:
                raise InventoryError(f"{where}: the any-topic marker is only allowed for department rules")
        anchors_raw = item["anchors"]
        if not isinstance(anchors_raw, list) or not anchors_raw:
            raise InventoryError(f"{where}: anchors must be a non-empty list")
        anchors: list[Anchor] = []
        used: set[str] = set()
        for index, anchor_raw in enumerate(anchors_raw):
            a_where = f"{where}.anchors[{index}]"
            anchor = _obj(anchor_raw, a_where, ("doc_id", "heading", "quote", "role"))
            doc_id, heading, quote = (_text(anchor[k], f"{a_where}.{k}") for k in ("doc_id", "heading", "quote"))
            if len(quote) < MIN_QUOTE_CHARS:
                raise InventoryError(f"{a_where}: quote is shorter than {MIN_QUOTE_CHARS} characters")
            if anchor["role"] not in (PRIMARY, SUPPORTING):
                raise InventoryError(f"{a_where}: role must be primary or supporting")
            found = _resolve(chunk_list, doc_id, heading, quote, a_where)
            if len(found) != 1:
                raise InventoryError(f"{a_where}: {doc_id} / {heading!r} / quote resolves to {len(found)} chunks, expected 1")
            if found[0].chunk_id in used:
                raise InventoryError(f"{a_where}: chunk {found[0].chunk_id} is listed twice in the rule")
            used.add(found[0].chunk_id)
            if kind == DEPARTMENT_EXCEPTION and anchor["role"] == PRIMARY and doc_id not in manifest[department]:
                raise InventoryError(f"{a_where}: primary anchor {doc_id} is not a manifest source for {department}")
            anchors.append(Anchor(doc_id, heading, quote, anchor["role"], found[0].chunk_id))
        rule = Rule(rule_id, kind, department, topic, tuple(anchors))
        limit = MAX_PRIMARY_DEPARTMENT if kind == DEPARTMENT_EXCEPTION else MAX_PRIMARY_GOVERNING
        if not 1 <= len(rule.primaries) <= limit:
            raise InventoryError(f"{where}: needs 1 to {limit} primary anchors, has {len(rule.primaries)}")
        rules.append(rule)

    covered = {r.department for r in rules if r.kind == DEPARTMENT_EXCEPTION}
    for department in sorted(set(manifest) - covered):
        raise InventoryError(f"manifest department exception for {department} has no rule in the inventory")
    return RuleInventory(1, tuple(rules), topic_chunks)
