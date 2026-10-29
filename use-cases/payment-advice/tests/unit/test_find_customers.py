# tests/unit/test_find_customers.py
"""
Unit tests for find_customers (UC-02 fuzzy customer search).

Tests the SQL contract: CONTAINS...FUZZY, bound params :q/:n, and row->tuple
mapping.  No live HANA required — the FakeEngine drives the assertions.
"""
import unittest

from app.payment_advice.customers import find_customers
from tests.unit.fakes import FakeEngine


class FindCustomersTest(unittest.TestCase):
    def test_builds_fuzzy_contains_query_and_maps_rows(self):
        engine = FakeEngine(rows=[
            {"CLIENT_KEY": "fabrikam", "DISPLAY_NAME": "Fabrikam", "SCORE": 0.9},
            {"CLIENT_KEY": "adventure_works", "DISPLAY_NAME": "Adventure Works", "SCORE": 0.5},
        ])
        out = find_customers(engine, "walmrt", limit=5)
        sql = engine.last_sql.upper()
        self.assertIn("CONTAINS", sql)
        self.assertIn("FUZZY(", sql)
        self.assertIn("SCORE()", sql)
        self.assertEqual(engine.last_params, {"q": "walmrt", "n": 5})
        self.assertEqual(out[0], ("fabrikam", "Fabrikam", 0.9))

    def test_empty_query_returns_empty_list(self):
        """Empty or blank query must short-circuit without touching the engine."""
        engine = FakeEngine(rows=[])
        self.assertEqual(find_customers(engine, ""), [])
        self.assertEqual(find_customers(engine, "   "), [])
        # No SQL should have been emitted.
        self.assertIsNone(engine.last_sql)

    def test_two_rows_returned(self):
        """All rows in the result set are mapped to tuples."""
        engine = FakeEngine(rows=[
            {"CLIENT_KEY": "fabrikam", "DISPLAY_NAME": "Fabrikam", "SCORE": 0.9},
            {"CLIENT_KEY": "adventure_works", "DISPLAY_NAME": "Adventure Works", "SCORE": 0.5},
        ])
        out = find_customers(engine, "depot", limit=10)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1], ("adventure_works", "Adventure Works", 0.5))
