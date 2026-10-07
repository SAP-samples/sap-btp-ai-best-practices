"""Offline checks for the dedicated Document AI schema administration (describe, plan, prepare, publish, discard)."""
import unittest
from types import SimpleNamespace
from unittest import mock

from dox_client import FieldDefinition

from app.payment_advice import schema_admin as A
from app.payment_advice.customers import CustomerSchema, bind_schema
from tests.unit.fakes import FakeEngine

HEADER = [{"name": "document_no", "label": "Document No.", "formattingType": "string"}]
LINE = [{"name": "region", "label": "Rg", "formattingType": "string"},
        {"name": "net", "label": "Net", "formattingType": "number"}]
BOUND = CustomerSchema("contoso", "schema-1", "2", False, "dedicated")


def customer(priority=True):
    """A registered customer record."""
    return SimpleNamespace(display_name="Contoso", is_critical=priority)


def dox(versions=(("2", "active"),)):
    """Document AI client double: version 2 is bound, details return HEADER/LINE."""
    client = mock.Mock()
    client.list_schema_versions.return_value = [{"version": v, "state": s} for v, s in versions]
    client.get_schema_version_details.return_value = {"headerFields": HEADER, "lineItemFields": LINE, "state": "active"}
    client.get_schema_by_name.return_value = None
    client.create_schema_version.return_value = {"version": "3"}
    return client


class Patched(unittest.TestCase):
    """Patches the HANA customer functions used by schema_admin."""
    def patch(self, priority=True, bound=BOUND):
        for name, value in (("get_customer", customer(priority)), ("get_bound_schema", bound)):
            patcher = mock.patch.object(A, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        binder = mock.patch.object(A, "bind_schema")
        self.bind = binder.start()
        self.addCleanup(binder.stop)


class DescribeTests(Patched):
    """Only priority customers own an editable schema; the canonical schema is read-only."""
    def test_dedicated_schema_is_editable_with_fields_and_versions(self):
        self.patch()
        report = A.describe(None, dox(), "contoso", "cid")
        self.assertTrue(report["editable"])
        self.assertEqual(report["schema"], {"schema_id": "schema-1", "version": "2", "source": "dedicated", "canonical": False})
        self.assertEqual([f["name"] for f in report["line_fields"]], ["region", "net"])
        self.assertEqual(report["versions"], [{"version": "2", "state": "active"}])

    def test_non_priority_customer_is_not_editable(self):
        self.patch(priority=False, bound=CustomerSchema("acme", "canon", "1", True, "canonical"))
        report = A.describe(None, dox(), "acme", "cid")
        self.assertFalse(report["editable"])
        self.assertIn("priority", report["note"])


class PlanTests(unittest.TestCase):
    """Explicit changes produce a complete, validated field set and a diff."""
    current = A.CurrentSchema("contoso", "schema-1", "2", HEADER, LINE)

    def test_add_update_remove(self):
        plan = A.plan_changes(self.current, add=[{"name": "payee name", "scope": "header", "description": "title row"}],
                              update=[{"name": "net", "type": "currency"}], remove=["document_no"])
        self.assertEqual([f["name"] for f in plan.header], ["payee_name"])
        self.assertEqual(plan.diff["added"][0]["name"], "payee_name")
        self.assertEqual(plan.diff["changed"], [{"name": "net", "scope": "line", "changes": {"formattingType": ["number", "currency"]}}])
        self.assertEqual(plan.diff["removed"][0]["name"], "document_no")
        self.assertEqual(plan.base_version, "2")

    def test_invalid_changes_are_refused(self):
        for kwargs, message in (({"add": [{"name": "x", "label": "Rg"}]}, "label"),
                                ({"add": [{"name": "x", "type": "country/region"}]}, "type"),
                                ({"add": [{"name": "region", "scope": "line"}]}, "already exists"),
                                ({"remove": ["nope"]}, "no such field"),
                                ({"remove": ["region", "net"]}, "line-item field"),
                                ({"add": [{"name": "x", "scope": "footer"}]}, "scope")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(A.SchemaAdminError, message):
                A.plan_changes(self.current, **kwargs)

    def test_live_contoso_shape_same_name_in_both_scopes_and_legacy_type(self):
        contoso = A.CurrentSchema("contoso", "s", "2", HEADER, [{"name": "document_no", "label": "Document No"},
                                                             {"name": "province", "label": "Province",
                                                              "formattingType": "country/region"}])
        plan = A.plan_changes(contoso, add=[{"name": "payee_name", "scope": "header", "label": "Payee"}])
        self.assertEqual([f["name"] for f in plan.diff["added"]], ["payee_name"])
        self.assertEqual(plan.diff["changed"], [])
        with self.assertRaisesRegex(A.SchemaAdminError, "say which"):
            A.plan_changes(contoso, remove=["document_no"])
        plan = A.plan_changes(contoso, remove=[{"name": "document_no", "scope": "line"}])
        self.assertEqual(plan.diff["removed"], [{"scope": "line", "name": "document_no", "label": "Document No"}])

    def test_proposal_from_sample_merges_by_default(self):
        proposed = ([FieldDefinition(name="document_no", label="Doc"), FieldDefinition(name="payee_name", label="Payee")],
                    [FieldDefinition(name="gross", label="Gross", formattingType="number")])
        with mock.patch.object(A, "sample_document_text", return_value="text"), \
                mock.patch.object(A, "generate_schema_fields", return_value=proposed):
            plan = A.propose_from_sample(self.current, "a.pdf")
        self.assertEqual([f["name"] for f in plan.header], ["document_no", "payee_name"])
        self.assertEqual([f["name"] for f in plan.line], ["region", "net", "gross"])
        self.assertEqual(plan.diff["removed"], [])


class PrepareTests(Patched):
    """A new version is created (or a leftover draft reused), activated, but never the bound one."""
    plan = A.SchemaPlan("contoso", "2", HEADER + [{"name": "payee_name", "label": "Payee"}], LINE, {})

    def test_creates_new_version_with_full_field_set(self):
        self.patch()
        client = dox()
        result = A.prepare(None, client, self.plan, "cid")
        self.assertEqual(result, {"schema_id": "schema-1", "version": "3", "created_schema": False, "reused_version": False})
        client.configure_schema_version.assert_called_once_with(
            "schema-1", "3", client_id="cid", header_fields=self.plan.header, line_item_fields=LINE,
            replace=True, activate=True)
        self.bind.assert_not_called()  # production stays on version 2

    def test_reuses_leftover_draft_newer_than_bound(self):
        self.patch()
        client = dox(versions=(("1", "inactive"), ("2", "active"), ("3", "draft")))
        self.assertEqual(A.prepare(None, client, self.plan, "cid")["version"], "3")
        client.create_schema_version.assert_not_called()

    def test_stale_plan_is_refused(self):
        self.patch(bound=CustomerSchema("contoso", "schema-1", "4", False, "dedicated"))
        with self.assertRaisesRegex(A.SchemaAdminError, "make a new plan"):
            A.prepare(None, dox(), self.plan, "cid")

    def test_first_schema_is_created_under_the_customer_name(self):
        self.patch(bound=None)
        client = dox(versions=(("1", "draft"),))
        client.create_schema.return_value = {"id": "new-schema"}
        result = A.prepare(None, client, A.SchemaPlan("contoso", None, HEADER, LINE, {}), "cid")
        self.assertEqual((result["schema_id"], result["version"], result["created_schema"]), ("new-schema", "1", True))
        self.assertEqual(client.create_schema.call_args.kwargs["schema_name"], "payment_advice_client_contoso")


class PublishTests(Patched):
    """Publishing binds first, then retires the previous version; only for priority customers."""
    def test_binds_then_deactivates_previous_and_clears_label_cache(self):
        from app.payment_advice.pipeline import _SCHEMA_FIELDS_CACHE

        self.patch()
        client = dox(versions=(("2", "active"), ("3", "active")))
        calls = mock.Mock()
        calls.attach_mock(self.bind, "bind")
        calls.attach_mock(client.deactivate_schema_version, "deactivate")
        _SCHEMA_FIELDS_CACHE[("schema-1", "3")] = []
        result = A.publish(None, client, "contoso", "3", "cid")
        self.assertEqual(result["previous_version"], "2")
        self.assertEqual([c[0] for c in calls.mock_calls], ["bind", "deactivate"])
        self.assertEqual(self.bind.call_args.args[1], CustomerSchema("contoso", "schema-1", "3", False, "dedicated"))
        self.assertNotIn(("schema-1", "3"), _SCHEMA_FIELDS_CACHE)

    def test_non_priority_customer_is_refused(self):
        self.patch(priority=False)
        with self.assertRaisesRegex(A.SchemaAdminError, "priority"):
            A.publish(None, dox(versions=(("2", "active"), ("3", "active"))), "contoso", "3", "cid")
        self.bind.assert_not_called()

    def test_discard_refuses_the_published_version(self):
        self.patch()
        with self.assertRaisesRegex(A.SchemaAdminError, "published version"):
            A.discard(None, dox(), "contoso", "2", "cid")


class BindingTests(unittest.TestCase):
    """Re-binding an existing version refreshes CREATED_AT so it becomes the effective binding (rollback)."""
    def test_rebind_refreshes_created_at(self):
        engine = FakeEngine(rows=[{"n": 1}])
        bind_schema(engine, BOUND)
        self.assertIn('"CREATED_AT" = CURRENT_UTCTIMESTAMP', engine.executed_writes[-1])


if __name__ == "__main__":
    unittest.main()
