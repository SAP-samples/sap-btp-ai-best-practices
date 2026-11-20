"""Save writes reviewer corrections (persistence and validation patched)."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from app.orders import service
from app.orders.validation import ValidationResult


def persisted_state():
    """Stored order: one header field and one line, both from the agent."""
    return {
        "header": {"STATUS": "NEEDS_REVIEW", "SOLD_TO": "0000100001", "SHIP_TO": None, "PO_NUMBER": "PO-1"},
        "items": [{"ITEM_NO": 10, "LINE_TYPE": "MATERIAL", "MATERIAL": None, "QUANTITY": 2.0, "UOM": "PC"}],
        "field_meta": {(10, "MATERIAL"): {"RAW_VALUE": "YEW F236", "CHANGED_BY": "AGENT", "CONFIDENCE": 0.2}},
        "flags": [], "multiple_orders": False, "classification": "STANDARD", "message": {},
    }


class CorrectionTests(unittest.TestCase):
    """Changed fields become USER fields and are logged with the printed value."""

    def test_save_records_corrections_for_changed_fields_only(self):
        state = persisted_state()
        logged = {}

        @contextmanager
        def fake_begin():
            yield "conn"

        with patch.object(service, "load_persisted_state", return_value=state), \
             patch.object(service, "evaluate", return_value=(ValidationResult([], False), [])), \
             patch.object(service, "persist_evaluation"), \
             patch.object(service, "get_engine") as engine, \
             patch.object(service.repository, "insert_corrections", side_effect=lambda conn, oid, rows: logged.update(rows=rows)):
            engine.return_value.begin = fake_begin
            _, corrections = service.save_user_edits(
                "order-1", {"ship_to": "100002", "po_number": "PO-1"},
                [{"item_no": 10, "material": "7000006", "quantity": 2}])

        fields = {(c["ITEM_NO"], c["FIELD_NAME"]): c for c in corrections}
        self.assertEqual(set(fields), {(0, "SHIP_TO"), (10, "MATERIAL")})   # unchanged values are not corrections
        self.assertEqual(fields[(10, "MATERIAL")]["RAW_VALUE"], "YEW F236")
        self.assertEqual(fields[(10, "MATERIAL")]["USER_VALUE"], "000000000007000006")
        self.assertEqual(fields[(0, "SHIP_TO")]["SOLD_TO"], "0000100001")
        self.assertEqual(logged["rows"], corrections)
        meta = state["field_meta"][(10, "MATERIAL")]
        self.assertEqual((meta["CHANGED_BY"], meta["CONFIDENCE"]), ("USER", 1.0))


if __name__ == "__main__":
    unittest.main()
