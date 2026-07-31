"""Regression tests for the HANA Knowledge Graph notebook helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import re
import json


NOTEBOOK_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NOTEBOOK_DIRECTORY))

from kg_notebook_helpers import (
    build_predicate_select_query,
    build_run_graph_iri,
    create_run_id,
    predicate_label,
)


class BuildRunGraphIriTests(unittest.TestCase):
    """Test named-graph isolation for independent notebook runs."""

    def test_encodes_run_and_graph_name_in_a_namespaced_iri(self) -> None:
        """A graph IRI must isolate a run and safely encode its graph label."""
        graph_iri = build_run_graph_iri("run 2026/07", "motor A")

        self.assertEqual(
            graph_iri,
            "<https://kg-demo.example/runs/run%202026%2F07/motor%20A>",
        )

    def test_creates_a_unique_uuid_based_run_id(self) -> None:
        """Each execution must receive an isolated graph-run namespace."""
        first_run_id = create_run_id()
        second_run_id = create_run_id()

        self.assertRegex(first_run_id, r"^[0-9a-f]{32}$")
        self.assertRegex(second_run_id, r"^[0-9a-f]{32}$")
        self.assertNotEqual(first_run_id, second_run_id)


class PredicateQueryTests(unittest.TestCase):
    """Test predicate-based retrieval for the graph RAG scenarios."""

    def test_returns_no_query_when_no_predicates_are_selected(self) -> None:
        """An empty LLM selection must not produce an invalid FILTER() query."""
        query = build_predicate_select_query(
            "<https://kg-demo.example/runs/run-001/motor-a>",
            [],
        )

        self.assertIsNone(query)

    def test_queries_predicates_with_values_and_returns_human_labels(self) -> None:
        """Selected predicates must be queried as IRIs while labels stay readable."""
        predicate = "http://example.com/property/has%20efficiency%20rating"
        query = build_predicate_select_query(
            "<https://kg-demo.example/runs/run-001/motor-a>",
            [predicate],
        )

        self.assertIn("VALUES ?p { <http://example.com/property/has%20efficiency%20rating> }", query)
        self.assertNotIn("FILTER(", query)
        self.assertEqual(predicate_label(predicate), "has efficiency rating")


class NotebookIntegrationTests(unittest.TestCase):
    """Test the notebook's contract with its safe HANA graph helpers."""

    @staticmethod
    def _load_notebook() -> dict:
        """Load the target notebook without executing any of its cells."""
        notebook_path = NOTEBOOK_DIRECTORY / (
            "KG-RDF-creation-grounding-visualisation-BP07-BP08-BestPractice.ipynb"
        )
        return json.loads(notebook_path.read_text())

    def test_every_notebook_code_cell_compiles(self) -> None:
        """Notebook source must be syntactically valid before a HANA test run."""
        notebook = self._load_notebook()

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(
                    "".join(cell.get("source", [])),
                    f"notebook-cell-{index}",
                    "exec",
                )

    def test_notebook_uses_user_owned_run_graphs_and_predicate_queries(self) -> None:
        """The notebook must avoid shared graph IRIs and obsolete relation filtering."""
        notebook = self._load_notebook()
        code_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertIn("from kg_notebook_helpers import", code_source)
        self.assertIn("sys.path.insert(0, str(notebook_directory))", code_source)
        self.assertIn("SELECT CURRENT_USER FROM DUMMY", code_source)
        self.assertIn("SELECT DISTINCT ?p", code_source)
        self.assertIn("build_predicate_select_query", code_source)
        self.assertIn("DROP GRAPH", code_source)
        self.assertNotIn('base="http://graph/"', code_source)
        self.assertNotIn("str | None", code_source)


if __name__ == "__main__":
    unittest.main()
