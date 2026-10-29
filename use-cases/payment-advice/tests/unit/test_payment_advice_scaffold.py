"""
Unit tests for the Payment Advice Extractor scaffold (step 1).

Pure, offline checks only: no SAP HANA, SAP Document AI, or LLM calls. They guard
the config/table/seed logic that would otherwise only fail at runtime against a
live tenant.

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Make ``app`` and ``dox_client`` importable when run from the repo root.
_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from app.payment_advice import config as config_mod  # noqa: E402
from app.payment_advice.config import PaymentAdviceSettings  # noqa: E402
from app.payment_advice.customers import (  # noqa: E402
    load_seed_customers,
    normalize_client_key,
)
from app.payment_advice.hana_schema import (  # noqa: E402
    TABLE_CONTRACTS,
    TABLE_DDL,
)

# A minimal but structurally valid SAP Document AI service key for tests.
_FAKE_SERVICE_KEY = (
    '{"url": "https://dox.example.com",'
    ' "uaa": {"url": "https://token.example.com",'
    ' "clientid": "cid", "clientsecret": "sec"}}'
)


class TableContractDdlConsistency(unittest.TestCase):
    """Every contracted column must appear in its DDL, and PKs must line up."""

    def test_every_contract_column_is_in_ddl(self) -> None:
        for table, columns in TABLE_CONTRACTS.items():
            ddl = TABLE_DDL[table]
            for column in columns:
                self.assertIn(f'"{column}"', ddl, f"{column} missing from {table} DDL")

    def test_primary_keys_declared_in_ddl(self) -> None:
        # CUSTOMERS: single-column inline PK on CLIENT_KEY.
        customers = next(t for t in TABLE_DDL if t.endswith("_CUSTOMERS"))
        self.assertIn('"CLIENT_KEY" NVARCHAR(60) NOT NULL PRIMARY KEY', TABLE_DDL[customers])
        # CUSTOMER_SCHEMAS: composite PK.
        schemas = next(t for t in TABLE_DDL if t.endswith("_CUSTOMER_SCHEMAS"))
        self.assertIn(
            'PRIMARY KEY ("CLIENT_KEY", "SCHEMA_ID", "SCHEMA_VERSION")',
            TABLE_DDL[schemas],
        )

    def test_contract_pk_flags_match_ddl_intent(self) -> None:
        schemas = next(t for t in TABLE_CONTRACTS if t.endswith("_CUSTOMER_SCHEMAS"))
        pk_cols = {c for c, contract in TABLE_CONTRACTS[schemas].items() if contract.primary_key}
        self.assertEqual(pk_cols, {"CLIENT_KEY", "SCHEMA_ID", "SCHEMA_VERSION"})


class SeedIntegrity(unittest.TestCase):
    """The JSON seed list loads, validates and can be overridden by env var."""

    def _write(self, content: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(content)
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_packaged_list_is_the_anonymized_demo(self) -> None:
        with mock.patch.dict(os.environ, {"PAYMENT_ADVICE_SEED_CUSTOMERS_PATH": ""}):
            self.assertEqual(load_seed_customers(), (("northwind", "Northwind"),))

    def test_env_var_overrides_packaged_list(self) -> None:
        path = self._write('[{"client_key": "acme", "display_name": "Acme"}]')
        with mock.patch.dict(os.environ, {"PAYMENT_ADVICE_SEED_CUSTOMERS_PATH": path}):
            self.assertEqual(load_seed_customers(), (("acme", "Acme"),))

    def test_rejects_bad_entries(self) -> None:
        for content in ('{"client_key": "acme"}', '[{"client_key": "Acme Co", "display_name": "Acme"}]',
                        '[{"client_key": "acme", "display_name": "A"}, {"client_key": "acme", "display_name": "B"}]',
                        '[{"client_key": "acme"}]', "not json"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                load_seed_customers(self._write(content))


class ClientKeyNormalization(unittest.TestCase):
    """normalize_client_key folds display names into stable keys."""

    def test_cases(self) -> None:
        cases = {
            "Fabrikam": "fabrikam",
            "  Globex  ": "globex",
            "Tailspin Toys": "tailspin_toys",
            "Example Distributors": "example_distributors",
            "WoodGrove": "woodgrove",
            "Adventure-Works": "adventure_works",
            "US01_Fabrikam": "us01_fabrikam",
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_client_key(raw), expected)

    def test_empty_raises(self) -> None:
        for bad in ("", "   ", "!!!"):
            with self.assertRaises(ValueError):
                normalize_client_key(bad)


class ConfigLoading(unittest.TestCase):
    """Config loads the service key from a file and enforces the model allowlist."""

    def _clean_env(self) -> dict[str, str]:
        # Remove any ambient service-key / model env so the file path is used.
        drop = {
            "DOCUMENT_AI_SERVICE_KEY_BASE64",
            "DOCUMENT_AI_SERVICE_KEY_JSON",
            "DOCUMENT_AI_SERVICE_KEY_PATH",
            "DOCUMENT_AI_CLIENT_ID",
            "PAYMENT_ADVICE_MAPPER_MODEL",
            "PAYMENT_ADVICE_AGENT_MODEL",
            "PAYMENT_ADVICE_OUT_DIR",
        }
        return {k: v for k, v in os.environ.items() if k not in drop}

    def test_loads_from_file_with_defaults(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(_FAKE_SERVICE_KEY)
            key_path = handle.name
        try:
            with mock.patch.dict(os.environ, self._clean_env(), clear=True):
                settings = PaymentAdviceSettings.from_env(service_key_path=key_path)
            self.assertEqual(settings.dox_client_id, "ai4u_payment_advice")
            self.assertEqual(settings.mapper_model, "gpt-5.6-luna")
            self.assertEqual(settings.max_pages, 100)
            self.assertEqual(settings.max_line_items, 2000)
            self.assertEqual(settings.max_columns, 49)
        finally:
            os.unlink(key_path)

    def test_bad_model_rejected(self) -> None:
        env = self._clean_env()
        env["PAYMENT_ADVICE_MAPPER_MODEL"] = "not-a-real-model"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(_FAKE_SERVICE_KEY)
            key_path = handle.name
        try:
            with mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError):
                    PaymentAdviceSettings.from_env(service_key_path=key_path)
        finally:
            os.unlink(key_path)

    def test_missing_key_raises(self) -> None:
        with mock.patch.dict(os.environ, self._clean_env(), clear=True):
            with self.assertRaises(ValueError):
                PaymentAdviceSettings.from_env(service_key_path="/nonexistent/key.json")

    def test_allowed_models_constant(self) -> None:
        self.assertEqual(config_mod.ALLOWED_MODELS, ("gpt-5.6-luna", "gemini-3.1-flash-lite"))


if __name__ == "__main__":
    unittest.main()
