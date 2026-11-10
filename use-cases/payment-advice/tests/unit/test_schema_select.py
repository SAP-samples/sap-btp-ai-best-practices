"""
Unit tests for schema selection and generation (offline).

No SAP/HANA/LLM network: the completion callable is faked and HANA/SAP helpers are
patched. Covers document sampling, field-definition generation (normalization,
unique labels, valid types), and the criticality-tiered selection branches.

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from app.payment_advice import schema_select as S  # noqa: E402
from app.payment_advice.customers import CustomerSchema, Customer  # noqa: E402


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [type("C", (), {"message": _Msg(content)})()]


class FakeCreate:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)

    def __call__(self, **kwargs):
        return _Resp(self.contents.pop(0))


class Sampling(unittest.TestCase):
    def test_tabular_sample_has_header_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.csv"
            with src.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["PO", "Amount"])
                w.writerows([[i, i * 2] for i in range(30)])
            text = S.sample_document_text(src, max_rows=10)
            self.assertIn("PO", text)
            self.assertLessEqual(text.count("\n"), 10)  # header + <=10 rows

    def test_opaque_sample_reads_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.txt"
            src.write_text("Payment number: 5000423510\nGross Terms Net\n", encoding="utf-8")
            self.assertIn("Payment number", S.sample_document_text(src))


class FieldGeneration(unittest.TestCase):
    def test_generation_normalizes_and_dedupes(self) -> None:
        content = json.dumps(
            {
                "header": [
                    {"name": "Payment Number", "label": "Payment Number", "type": "string"},
                    {"name": "Total", "label": "Total", "type": "integer"},  # -> number
                ],
                "line_items": [
                    {"name": "Invoice #", "label": "Total", "type": "number"},  # dup label -> disambiguated
                    {"name": "Invoice #", "label": "x", "type": "string"},  # dup name -> skipped
                ],
            }
        )
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):  # swallow the usage event line
            header, line = S.generate_schema_fields(
                "some sample", client_key="acme", openai_create=FakeCreate([content])
            )
        self.assertEqual([f.name for f in header], ["Payment_Number", "Total"])
        self.assertEqual(header[1].formattingType, "number")  # integer normalized
        self.assertEqual([f.name for f in line], ["Invoice_#"])  # '#' is SAP-permitted; dup name skipped
        labels = [f.label for f in header + line]
        self.assertEqual(len(labels), len(set(labels)), "labels unique across header+line")

    def test_country_region_type_becomes_string(self) -> None:
        content = json.dumps({"header": [], "line_items": [
            {"name": "region", "label": "Rg", "type": "country/region"}]})
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            _, line = S.generate_schema_fields("sample", client_key="acme", openai_create=FakeCreate([content]))
        self.assertEqual(line[0].formattingType, "string")

    def test_retyped_definitions_copy_fields_and_change_one_type(self) -> None:
        details = {
            "headerFields": [{"name": "document_no", "label": "Document No.", "formattingType": "string",
                              "formatting": {"decimalSeparator": "."}}],
            "lineItemFields": [{"name": "region", "label": "Rg", "description": "Region", "formattingType": "country/region"},
                               {"name": "gross", "label": "Gross", "formattingType": "currency"}],
        }
        header, lines, previous = S.retyped_field_definitions(details, "region", "string")
        self.assertEqual(previous, "country/region")
        self.assertEqual(header, [{"name": "document_no", "label": "Document No.", "formattingType": "string"}])
        self.assertEqual(lines[0], {"name": "region", "label": "Rg", "description": "Region", "formattingType": "string"})
        self.assertEqual(lines[1]["formattingType"], "currency")
        with self.assertRaisesRegex(ValueError, "not in schema version"):
            S.retyped_field_definitions(details, "province", "string")

    def test_empty_sample_raises(self) -> None:
        with self.assertRaises(ValueError):
            S.generate_schema_fields("   ", client_key="acme", openai_create=FakeCreate(["{}"]))


class TieredSelection(unittest.TestCase):
    def test_concurrent_canonical_creation_uses_shared_guard(self):
        """First-use requests must not create/configure the shared schema twice."""
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import contextmanager
        lock, created = threading.Lock(), []
        client = mock.Mock()

        def lookup(*args, **kwargs):
            """Model a remote lookup slow enough for overlapping first-use requests."""
            found = {'id': 'canonical'} if created else None
            time.sleep(.02)
            return found

        def create(**kwargs):
            """Record remote schema creation, not the guarding implementation."""
            created.append('canonical')
            return {'id': 'canonical'}

        @contextmanager
        def guard():
            """Stand in for the HANA guard tested separately against real HANA."""
            with lock:
                yield

        client.get_schema_by_name.side_effect = lookup
        client.create_schema.side_effect = create
        client.list_schema_versions.return_value = [{'version': '1', 'state': 'ACTIVE'}]

        def ensure():
            """Execute real canonical selection concurrently."""
            return S.ensure_canonical_schema(client, dox_client_id='test', canonical_guard=guard)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: ensure(), range(2)))
        self.assertEqual(results, [('canonical', '1'), ('canonical', '1')])
        self.assertEqual(created, ['canonical'])
        self.assertTrue(lock.acquire(blocking=False), 'Schema lock survived setup')
        lock.release()

    def test_promotion_ignores_old_canonical_binding(self):
        """A newly priority customer must generate its own schema despite an old canonical binding."""
        old = CustomerSchema('example', 'canonical', '1', True, 'canonical')
        new = CustomerSchema('example', 'dedicated', '1', False, 'dedicated')
        with mock.patch.object(S, 'get_customer', return_value=Customer('example', 'Example', True, 'active')), \
             mock.patch.object(S, 'get_bound_schema', return_value=old), \
             mock.patch.object(S, 'sample_document_text', return_value='Advice'), \
             mock.patch.object(S, 'generate_schema_fields', return_value=([], [])), \
             mock.patch.object(S, 'create_dedicated_schema', return_value=new) as create:
            result = S.select_schema(None, None, 'example', 'a.txt', dox_client_id='test', assume_yes=True)
        create.assert_called_once()
        self.assertEqual(result.schema_id, 'dedicated')

    def test_active_schema_is_not_reconfigured(self):
        """Reusing an activated schema must never replace its existing field contract."""
        client = mock.Mock()
        client.get_schema_by_name.return_value = {'id': 'existing'}
        client.list_schema_versions.return_value = [{'version': '2', 'state': 'ACTIVE'}]
        self.assertEqual(S.ensure_canonical_schema(client, dox_client_id='test'), ('existing', '2'))
        client.configure_schema_version.assert_not_called()

    def test_bound_schema_is_reused(self) -> None:
        bound = CustomerSchema("fabrikam", "sid", "3", is_canonical=False, source="dedicated")
        with mock.patch.object(S, "get_customer", return_value=Customer("fabrikam", "Fabrikam", True, "active")), \
             mock.patch.object(S, "get_bound_schema", return_value=bound):
            sel = S.select_schema(None, None, "fabrikam", "x.pdf", dox_client_id="ai4u_payment_advice")
        self.assertEqual(sel.status, "ready")
        self.assertEqual((sel.schema_id, sel.schema_version, sel.source), ("sid", "3", "dedicated"))

    def test_non_critical_uses_canonical(self) -> None:
        with mock.patch.object(S, "get_customer", return_value=None), \
             mock.patch.object(S, "get_bound_schema", return_value=None), \
             mock.patch.object(S, "ensure_canonical_schema", return_value=("canon", "1")) as ensure, \
             mock.patch.object(S, "bind_schema") as bind:
            sel = S.select_schema(None, object(), "some_new_client", "x.pdf", dox_client_id="ai4u_payment_advice")
        ensure.assert_called_once()
        bind.assert_called_once()
        self.assertEqual((sel.status, sel.is_canonical, sel.source), ("ready", True, "canonical"))

    def test_critical_without_yes_proposes(self) -> None:
        with mock.patch.object(S, "get_customer", return_value=Customer("globex", "Globex", True, "active")), \
             mock.patch.object(S, "get_bound_schema", return_value=None), \
             mock.patch.object(S, "sample_document_text", return_value="sample"), \
             mock.patch.object(S, "generate_schema_fields", return_value=(["h"], ["l"])), \
             mock.patch.object(S, "create_dedicated_schema") as created:
            sel = S.select_schema(None, object(), "globex", "x.xlsx", dox_client_id="ai4u_payment_advice", assume_yes=False)
        created.assert_not_called()  # nothing written to SAP without confirmation
        self.assertEqual(sel.status, "proposed")
        self.assertEqual(sel.proposed_header, ["h"])

    def test_critical_with_yes_creates(self) -> None:
        binding = CustomerSchema("globex", "new", "1", is_canonical=False, source="dedicated")
        with mock.patch.object(S, "get_customer", return_value=Customer("globex", "Globex", True, "active")), \
             mock.patch.object(S, "get_bound_schema", return_value=None), \
             mock.patch.object(S, "sample_document_text", return_value="sample"), \
             mock.patch.object(S, "generate_schema_fields", return_value=(["h"], ["l"])), \
             mock.patch.object(S, "create_dedicated_schema", return_value=binding) as created:
            sel = S.select_schema(None, object(), "globex", "x.xlsx", dox_client_id="ai4u_payment_advice", assume_yes=True)
        created.assert_called_once()
        self.assertEqual((sel.status, sel.schema_id, sel.source), ("ready", "new", "dedicated"))


if __name__ == "__main__":
    unittest.main()
