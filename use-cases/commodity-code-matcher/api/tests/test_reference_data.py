from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import reference_data


class ReferenceDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(
            os.environ,
            {
                "HANA_COMMODITY_CATALOG_TABLE": "REFERENCE_COMMODITY_CATALOG",
                "HANA_UNSPSC_MAPPING_TABLE": "REFERENCE_UNSPSC_MAPPING",
                "HANA_SUPPLIER_GROUPS_TABLE": "REFERENCE_SUPPLIER_GROUPS",
                "HANA_REFERENCE_DATA_VERSION": "synthetic-v1",
            },
            clear=False,
        )
        self.env_patch.start()
        reference_data._CACHE.clear()

    def tearDown(self) -> None:
        self.env_patch.stop()
        reference_data._CACHE.clear()

    def test_load_reference_data_validates_and_drops_version_column(self) -> None:
        class _FakeConnection:
            def close(self) -> None:
                return None

        frames = {
            "REFERENCE_COMMODITY_CATALOG": pd.DataFrame(
                [{"CODE": "RC0001", "DESCRIPTION_LOW_LEVEL": "Item 1", "DATA_VERSION": "synthetic-v1"}]
            ),
            "REFERENCE_UNSPSC_MAPPING": pd.DataFrame(
                [{"REFERENCE_CODE": "RC0001", "UNSPSC_CODE_DESCRIPTION": "Hardware", "DATA_VERSION": "synthetic-v1"}]
            ),
            "REFERENCE_SUPPLIER_GROUPS": pd.DataFrame(
                [{"SUPPLIER_NAME": "SUPPLIER_01", "MATERIAL_GROUP": "RC0001", "DATA_VERSION": "synthetic-v1"}]
            ),
        }

        with patch.object(reference_data, "_connect", return_value=_FakeConnection()), patch.object(
            reference_data,
            "_fetch_dataframe",
            side_effect=lambda _conn, _schema, table: frames[table].copy(),
        ):
            bundle = reference_data.load_reference_data(force_refresh=True)

        self.assertEqual(bundle.data_version, "synthetic-v1")
        self.assertNotIn("DATA_VERSION", bundle.catalog_df.columns)
        self.assertNotIn("DATA_VERSION", bundle.unspsc_df.columns)
        self.assertNotIn("DATA_VERSION", bundle.supplier_groups_df.columns)

    def test_load_reference_data_rejects_mismatched_versions(self) -> None:
        class _FakeConnection:
            def close(self) -> None:
                return None

        frames = {
            "REFERENCE_COMMODITY_CATALOG": pd.DataFrame(
                [{"CODE": "RC0001", "DESCRIPTION_LOW_LEVEL": "Item 1", "DATA_VERSION": "synthetic-v1"}]
            ),
            "REFERENCE_UNSPSC_MAPPING": pd.DataFrame(
                [{"REFERENCE_CODE": "RC0001", "UNSPSC_CODE_DESCRIPTION": "Hardware", "DATA_VERSION": "synthetic-v2"}]
            ),
            "REFERENCE_SUPPLIER_GROUPS": pd.DataFrame(
                [{"SUPPLIER_NAME": "SUPPLIER_01", "MATERIAL_GROUP": "RC0001", "DATA_VERSION": "synthetic-v1"}]
            ),
        }

        with patch.object(reference_data, "_connect", return_value=_FakeConnection()), patch.object(
            reference_data,
            "_fetch_dataframe",
            side_effect=lambda _conn, _schema, table: frames[table].copy(),
        ):
            with self.assertRaises(reference_data.ReferenceDataError):
                reference_data.load_reference_data(force_refresh=True)


if __name__ == "__main__":
    unittest.main()
