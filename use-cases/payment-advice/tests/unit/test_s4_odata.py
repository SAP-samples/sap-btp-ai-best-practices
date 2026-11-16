"""Offline checks for complete OData reads and URL-sized filters."""
import unittest

from app.s4.odata import S4IncompleteRead, chunk_values, or_filter, read_all


class PagedClient:
    """Serves a fixed list of rows page by page, optionally reporting a total that differs from it."""
    def __init__(self, rows, reported=None, with_count=True):
        self.rows, self.reported, self.with_count, self.calls = rows, reported, with_count, []

    def get_json(self, service, path, params):
        self.calls.append(dict(params))
        start, size = int(params.get("$skip", 0)), int(params["$top"])
        body = {"results": self.rows[start:start + size]}
        if self.with_count:
            body["__count"] = str(len(self.rows) if self.reported is None else self.reported)
        return {"d": body}


class ReadAllTests(unittest.TestCase):
    """A collection is read to its end, and a short answer is an error, never a partial result."""
    def test_all_pages_are_read_in_a_stable_order(self):
        client = PagedClient([{"n": i} for i in range(5)])
        rows = read_all(client, "SRV", "/Set", {"$filter": "x eq 1"}, order_by="n", page_size=2)
        self.assertEqual([r["n"] for r in rows], [0, 1, 2, 3, 4])
        self.assertEqual([c.get("$skip") for c in client.calls], [None, "2", "4"])
        self.assertTrue(all(c["$orderby"] == "n" and c["$inlinecount"] == "allpages" and c["$filter"] == "x eq 1"
                            for c in client.calls))

    def test_fewer_rows_than_s4_reported_raise(self):
        with self.assertRaises(S4IncompleteRead) as caught:
            read_all(PagedClient([{"n": 1}], reported=3), "SRV", "/Set", {}, order_by="n")
        self.assertEqual((caught.exception.received, caught.exception.expected), (1, 3))

    def test_service_without_a_count_stops_at_the_first_short_page(self):
        client = PagedClient([{"n": 1}, {"n": 2}, {"n": 3}], with_count=False)
        self.assertEqual(len(read_all(client, "SRV", "/Set", {}, order_by="n", page_size=2)), 3)
        self.assertEqual(len(client.calls), 2)


class FilterTests(unittest.TestCase):
    """Long value lists are split so that no filter outgrows the URL."""
    def test_values_are_quoted_and_joined(self):
        self.assertEqual(or_filter("Customer", ["10", "O'B"]), "(Customer eq '10' or Customer eq 'O''B')")

    def test_chunks_respect_the_character_budget_and_the_term_limit(self):
        values = [f"REF{i:013d}" for i in range(10)]  # each term costs 45 characters
        self.assertEqual([len(c) for c in chunk_values(values, "DocumentReferenceID", budget=100)], [2] * 5)
        self.assertEqual([len(c) for c in chunk_values(values, "DocumentReferenceID", max_terms=4)], [4, 4, 2])
        self.assertEqual(chunk_values(["X" * 500], "DocumentReferenceID", budget=10), [["X" * 500]])
        self.assertEqual(chunk_values([], "DocumentReferenceID"), [])


if __name__ == "__main__":
    unittest.main()
