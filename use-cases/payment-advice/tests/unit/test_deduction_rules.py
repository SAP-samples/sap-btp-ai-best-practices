"""
Unit tests for the DEDUCTION_RULES table contract and deduction_rules access module.

Tests cover:
  - Table contract registration in hana_schema (column types, primary key).
  - save_playbook HITL gate: unconfirmed stages without writing; confirmed persists
    with revision = current + 1.
  - delete_playbook HITL gate: unconfirmed previews without writing; confirmed deletes
    and reports deleted=True.

Run from the repo root:
    PYTHONPATH=api python -m unittest tests.unit.test_deduction_rules -v
"""

import unittest
from app.payment_advice import hana_schema
from app.payment_advice.deduction_rules import get_playbook, save_playbook, Playbook
from tests.unit.fakes import FakeEngine  # existing helper used by UC-01 tests


class DeductionRulesTest(unittest.TestCase):
    def test_deduction_rules_contract_registered(self):
        self.assertIn(hana_schema.DEDUCTION_RULES, hana_schema.TABLE_CONTRACTS)
        self.assertIn(hana_schema.DEDUCTION_RULES, hana_schema.TABLE_DDL)
        cols = hana_schema.TABLE_CONTRACTS[hana_schema.DEDUCTION_RULES]
        self.assertTrue(cols["CLIENT_KEY"].primary_key)
        self.assertEqual(cols["PLAYBOOK_TEXT"].data_type, "NCLOB")
        self.assertEqual(cols["ANCHORS_JSON"].data_type, "NCLOB")
        self.assertEqual(cols["REVISION"].data_type, "INTEGER")

    def test_save_unconfirmed_stages_without_writing(self):
        engine = FakeEngine(rows=[])   # no existing row
        result = save_playbook(engine, "fabrikam", "text", {"codes": {}}, user_confirmed=False)
        self.assertTrue(result["staged"])
        self.assertEqual(result["next_revision"], 1)
        self.assertEqual(engine.executed_writes, [])   # nothing persisted

    def test_save_confirmed_increments_revision(self):
        engine = FakeEngine(rows=[{"REVISION": 2}])    # existing revision 2
        result = save_playbook(engine, "fabrikam", "text", {"codes": {}}, user_confirmed=True, updated_by="peter")
        self.assertFalse(result["staged"])
        self.assertEqual(result["revision"], 3)
        self.assertTrue(engine.executed_writes)        # persisted

    def test_delete_unconfirmed_does_not_write(self):
        from app.payment_advice.deduction_rules import delete_playbook
        engine = FakeEngine(rows=[{"REVISION": 1}])    # row exists
        result = delete_playbook(engine, "fabrikam", user_confirmed=False)
        self.assertTrue(result["staged"])
        self.assertTrue(result["would_delete"])
        self.assertEqual(engine.executed_writes, [])

    def test_delete_confirmed_removes_row(self):
        from app.payment_advice.deduction_rules import delete_playbook
        engine = FakeEngine(rows=[{"REVISION": 1}])
        result = delete_playbook(engine, "fabrikam", user_confirmed=True)
        self.assertFalse(result["staged"])
        self.assertTrue(result["deleted"])
        self.assertTrue(engine.executed_writes)


if __name__ == "__main__":
    unittest.main()
