"""Naming and structure rules for every HANA table the app creates."""

import re
import unittest

from app.db.schema import APP_TABLES, ECC_TABLES, SEQUENCES, create_table_sql

ECC_PATTERN = re.compile(r"^(?!POAGENT)[A-Z][A-Z0-9_]*_POAGENT$")
APP_PATTERN = re.compile(r"^POAGENT_[A-Z_]+$")


class SchemaNamingTests(unittest.TestCase):
    """Mock ECC tables keep SAP names + suffix; app tables use the generic prefix."""

    def test_table_names_follow_convention(self) -> None:
        """ECC mirrors end in _POAGENT, app tables start with POAGENT_."""
        for name in ECC_TABLES:
            self.assertRegex(name, ECC_PATTERN)
        for name in APP_TABLES:
            self.assertRegex(name, APP_PATTERN)
        for name in SEQUENCES:
            self.assertTrue(name.endswith("_POAGENT"), name)

    def test_every_table_has_valid_primary_key(self) -> None:
        """Primary-key columns exist in the table and are NOT NULL."""
        for name, table in {**ECC_TABLES, **APP_TABLES}.items():
            self.assertTrue(table.primary_key, name)
            types = dict(table.columns)
            for col in table.primary_key:
                self.assertIn(col, types, f"{name}.{col}")
                self.assertIn("NOT NULL", types[col], f"{name}.{col} must be NOT NULL")

    def test_create_sql_renders_quoted_identifiers(self) -> None:
        """DDL quotes identifiers and declares the primary key."""
        sql = create_table_sql("KNA1_POAGENT", ECC_TABLES["KNA1_POAGENT"])
        self.assertTrue(sql.startswith('CREATE COLUMN TABLE "KNA1_POAGENT"'))
        self.assertIn('PRIMARY KEY ("KUNNR")', sql)


if __name__ == "__main__":
    unittest.main()
