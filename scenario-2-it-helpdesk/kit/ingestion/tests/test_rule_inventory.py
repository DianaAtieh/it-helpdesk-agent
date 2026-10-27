"""Offline tests for ingestion.rule_inventory: the real inventory validates, tampered variants are rejected."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ingestion.chunking import chunk_document
from ingestion.loaders import load_knowledge_base
from ingestion.rule_inventory import ChunkRef, DEFAULT_INVENTORY, InventoryError, load_inventory

_REFS = None


def refs() -> list[ChunkRef]:
    global _REFS
    if _REFS is None:
        _REFS = [ChunkRef(c.chunk_id, c.doc_id, c.heading_path, c.text, c.status, c.authoritative)
                 for d in load_knowledge_base() for c in chunk_document(d)]
    return _REFS


def real_data() -> dict:
    return json.loads(DEFAULT_INVENTORY.read_text(encoding="utf-8"))


class InventoryTests(unittest.TestCase):
    def load(self, data, chunks=None, **kw):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "inv.json"
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        return load_inventory(path, refs() if chunks is None else chunks, **kw)

    def rule(self, data, rule_id):
        return next(r for r in data["rules"] if r["rule_id"] == rule_id)

    def assertRejected(self, data, text="", **kw):
        with self.assertRaises(InventoryError) as ctx:
            self.load(data, **kw)
        self.assertIn(text, str(ctx.exception))

    def test_real_inventory_loads(self):
        inv = load_inventory(DEFAULT_INVENTORY, refs())
        self.assertEqual({r.department for r in inv.rules if r.department},
                         {"Finance", "Contractors", "Executive Office", "Engineering", "HR"})
        self.assertFalse({r.department for r in inv.rules} & {"Sales", "Support"})
        finance = next(r for r in inv.rules if r.department == "Finance")
        self.assertEqual(len(finance.primaries), 3)

    def test_activation_is_exact_and_case_insensitive(self):
        inv = load_inventory(DEFAULT_INVENTORY, refs())
        topics = frozenset({"vpn", "*"})
        names = lambda dept: {r.department for r in inv.activated("department_exception", topics, dept)}  # noqa: E731
        self.assertEqual(names("finance"), {"Finance"})
        self.assertEqual(names("Finance"), {"Finance"})
        self.assertEqual(names("Sales"), set())
        self.assertEqual(names("Fin"), set())
        self.assertEqual(names(None), {"Finance", "Contractors", "Executive Office"})

    def test_mfa_section_does_not_establish_vpn(self):
        inv = load_inventory(DEFAULT_INVENTORY, refs())
        sp2_mfa = next(c.chunk_id for c in refs() if c.doc_id == "SP-002" and c.heading_path[-1].startswith("1."))
        topics = inv.established_topics([sp2_mfa])
        self.assertIn("mfa", topics)
        self.assertNotIn("vpn", topics)
        self.assertEqual(inv.established_topics([]), frozenset())

    def test_tampering_is_rejected(self):
        d = real_data()
        d["extra"] = 1
        self.assertRejected(d, "unknown keys")
        d = real_data()
        d["inventory_version"] = 2
        self.assertRejected(d, "inventory_version")
        d = real_data()
        self.rule(d, "DEPT-FINANCE-VPN")["anchors"][0]["quote"] = "this text is not in any chunk"
        self.assertRejected(d, "resolves to 0 chunks")
        d = real_data()
        self.rule(d, "DEPT-FINANCE-VPN")["anchors"][0]["quote"] = "short"
        self.assertRejected(d, "shorter")
        d = real_data()
        self.rule(d, "DEPT-FINANCE-VPN")["anchors"][0].update(doc_id="KB-018", heading="Exceptions")
        self.assertRejected(d)
        d = real_data()
        self.rule(d, "DEPT-FINANCE-VPN")["department"] = "Marketing"
        self.assertRejected(d, "schema vocabulary")
        d = real_data()
        self.rule(d, "DEPT-FINANCE-VPN")["department"] = "all"
        self.assertRejected(d, "schema vocabulary")
        d = real_data()
        self.rule(d, "DEPT-FINANCE-VPN")["topic"] = "nonexistent"
        self.assertRejected(d, "unknown topic")
        d = real_data()
        self.rule(d, "GOV-SP002-VPN-DEVICES")["department"] = "Finance"
        self.assertRejected(d, "must not name a department")
        d = real_data()
        d["rules"].append(copy.deepcopy(d["rules"][0]))
        self.assertRejected(d, "duplicated")
        d = real_data()
        d["rules"] = [r for r in d["rules"] if r.get("department") != "HR"]
        self.assertRejected(d, "HR")
        d = real_data()
        for a in self.rule(d, "DEPT-FINANCE-VPN")["anchors"]:
            a["role"] = "supporting"
        self.assertRejected(d, "primary anchors")
        d = real_data()
        self.rule(d, "DEPT-FINANCE-VPN")["anchors"][3]["role"] = "primary"
        self.assertRejected(d, "manifest source")
        d = real_data()
        self.rule(d, "GOV-SP002-VPN-DEVICES")["topic"] = "*"
        self.assertRejected(d, "any-topic")

    def test_too_many_primaries(self):
        d = real_data()
        rule = self.rule(d, "DEPT-CONTRACTORS-VPN")
        for a in rule["anchors"]:
            a["role"] = "primary"
        rule["anchors"] = [a for a in rule["anchors"] if a["doc_id"] in ("KB-016", "SP-002", "KB-002")]
        rule["anchors"].append({"doc_id": "KB-002", "heading": "Gateways", "role": "primary",
                                "quote": "Finance users cannot change gateway."})
        self.assertRejected(d, "primary anchors")

    def test_ambiguous_quote_and_missing_chunks(self):
        d = real_data()
        self.rule(d, "DEPT-FINANCE-VPN")["anchors"][0]["heading"] = "Nonexistent heading"
        self.assertRejected(d, "resolves to 0 chunks")
        d = real_data()
        d["topics"]["vpn"][0]["doc_id"] = "KB-999"
        self.assertRejected(d, "matches no indexed chunk")
        self.assertRejected(real_data(), "resolves to 0 chunks", chunks=[c for c in refs() if c.doc_id != "KB-016"])

    def test_non_current_chunk_is_rejected(self):
        chunks = [ChunkRef(c.chunk_id, c.doc_id, c.heading_path, c.text, "draft" if c.doc_id == "KB-015" else c.status,
                           c.authoritative) for c in refs()]
        self.assertRejected(real_data(), "not a current, authoritative", chunks=chunks)

    def test_file_problems(self):
        self.assertRejected("{not json", "not valid JSON")
        self.assertRejected("[]", "must be an object")
        with self.assertRaises(InventoryError):
            load_inventory(Path(tempfile.gettempdir()) / "definitely-missing-inventory.json", refs())

    def test_manifest_and_schema_problems_fail_safely(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        bad = Path(tmp.name) / "m.json"
        bad.write_text("{}", encoding="utf-8")
        self.assertRejected(real_data(), "department_exceptions", manifest_path=bad)
        self.assertRejected(real_data(), "", schema_path=bad)


if __name__ == "__main__":
    unittest.main()
