from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from doc_extraction.embedding import matcher
from doc_extraction.embedding.matcher import run_community_code_matching


class MatcherTests(unittest.TestCase):
    def test_run_matching_with_neutral_reference_tables(self) -> None:
        def fake_embed(_self, texts):
            vectors = []
            for text in texts:
                lowered = str(text).lower()
                if "laptop" in lowered or "computing" in lowered or "hardware" in lowered:
                    vectors.append([1.0, 0.0, 0.0])
                else:
                    vectors.append([0.0, 1.0, 0.0])
            return np.array(vectors, dtype="float32")

        line_items = pd.DataFrame(
            [
                {
                    "file": "sample.pdf",
                    "line_index": 1,
                    "description": "generic laptop device",
                    "usageSummary": "portable computing hardware",
                    "Vendor": "SUPPLIER_01",
                }
            ]
        )
        catalog = pd.DataFrame(
            [
                {
                    "CODE": "RC0001",
                    "DESCRIPTION_LOW_LEVEL": "RC0001 CLASSIFICATION ITEM 001",
                    "DESCRIPTION_HIGH_LEVEL": "CATEGORY_01",
                    "PROCUREMENT_GROUP": "DOMAIN_01",
                    "DETAILED_DESCRIPTION_KEYWORDS": "generic laptop device, portable computing hardware",
                },
                {
                    "CODE": "RC0002",
                    "DESCRIPTION_LOW_LEVEL": "RC0002 CLASSIFICATION ITEM 002",
                    "DESCRIPTION_HIGH_LEVEL": "CATEGORY_02",
                    "PROCUREMENT_GROUP": "DOMAIN_02",
                    "DETAILED_DESCRIPTION_KEYWORDS": "factory wrench, industrial maintenance",
                },
            ]
        )
        unspsc = pd.DataFrame(
            [
                {
                    "REFERENCE_CODE": "RC0001",
                    "UNSPSC_CODE_DESCRIPTION": "computer hardware",
                    "REFERENCE_CODE_DESCRIPTION": "RC0001 CLASSIFICATION ITEM 001",
                }
            ]
        )
        supplier_groups = pd.DataFrame(
            [
                {
                    "SUPPLIER_NAME": "SUPPLIER_01",
                    "BUSINESS_PARTNER_ID": "BP_0001",
                    "MATERIAL_GROUP": "RC0001",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "result.xlsx"
            with patch.object(matcher._ProxyEmbeddings, "embed", new=fake_embed):
                final_output, output_df = run_community_code_matching(
                    line_items=line_items,
                    community_codes_path=catalog,
                    unspsc_context_path=unspsc,
                    supplier_groups_path=supplier_groups,
                    output_path=output_path,
                    llm_verify=False,
                    top_k_codes=2,
                )
            self.assertTrue(Path(final_output).exists())
            self.assertEqual(output_df.loc[0, "Business_Partner_ID"], "BP_0001")
            self.assertEqual(output_df.loc[0, "Code_Desc"], "RC0001")
            self.assertEqual(output_df.loc[0, "UNSPSC_CODE_DESCRIPTION_Desc"], "computer hardware")
            self.assertIn("RC0001", output_df.loc[0, "Codes_Desc_Top5"])


if __name__ == "__main__":
    unittest.main()
