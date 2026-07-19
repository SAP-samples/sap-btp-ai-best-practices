from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.extraction_service import (
    ExtractionConfig,
    ExtractionResult,
    _run_embedding_pipeline,
    _serialize_joule_line_items,
    run_extraction_for_paths,
)


class ExtractionServiceTests(unittest.TestCase):
    """Unit tests for Joule result shaping and genuine-row validation."""

    def test_placeholder_rows_are_removed_before_matching(self) -> None:
        """All-null extraction placeholders never reach commodity-code matching."""

        extracted = pd.DataFrame(
            [
                {"description": None, "netAmount": None, "quantity": None, "unitPrice": None},
                {"description": "Brake pads", "netAmount": 100.0, "quantity": 2, "unitPrice": 50.0},
            ]
        )
        matcher = Mock(return_value=(Path("result.xlsx"), extracted.iloc[[1]].copy()))

        with patch("app.services.extraction_service.load_reference_data") as reference_data, patch(
            "app.services.extraction_service._extract_with_llm", return_value=(pd.DataFrame(), extracted)
        ), patch("app.services.extraction_service.run_community_code_matching", matcher):
            reference_data.return_value = SimpleNamespace(
                catalog_df=pd.DataFrame(),
                unspsc_df=pd.DataFrame(),
                supplier_groups_df=pd.DataFrame(),
                data_version="v1",
            )
            _run_embedding_pipeline([Path("sample.pdf")], ExtractionConfig())

        matched_rows = matcher.call_args.kwargs["line_items"]
        self.assertEqual(matched_rows["description"].tolist(), ["Brake pads"])

    def test_only_placeholder_rows_fail_before_matching(self) -> None:
        """A document without a genuine extracted line fails instead of matching blank data."""

        placeholders = pd.DataFrame(
            [{"description": None, "netAmount": None, "quantity": None, "unitPrice": None}]
        )
        matcher = Mock()
        with patch("app.services.extraction_service.load_reference_data") as reference_data, patch(
            "app.services.extraction_service._extract_with_llm", return_value=(pd.DataFrame(), placeholders)
        ), patch("app.services.extraction_service.run_community_code_matching", matcher):
            reference_data.return_value = SimpleNamespace(
                catalog_df=pd.DataFrame(),
                unspsc_df=pd.DataFrame(),
                supplier_groups_df=pd.DataFrame(),
                data_version="v1",
            )
            with self.assertRaisesRegex(RuntimeError, "No genuine line items"):
                _run_embedding_pipeline([Path("sample.pdf")], ExtractionConfig())

        matcher.assert_not_called()

    def test_joule_serializer_returns_exact_fields_and_preserves_values(self) -> None:
        """Every enriched row becomes the exact seven-field Joule contract."""

        enriched = pd.DataFrame(
            [
                {
                    "description": "Brake pads",
                    "netAmount": 100,
                    "quantity": 2,
                    "unitPrice": 50.5,
                    "LLM_Suggestion_Desc": "RC0001",
                    "LLM_Confidence_Desc": 0.91,
                    "LLM_Reason_Desc": "Best semantic match.",
                    "file": "ignored.pdf",
                },
                {
                    "description": None,
                    "netAmount": pd.NA,
                    "quantity": None,
                    "unitPrice": None,
                    "LLM_Suggestion_Desc": "UNSURE",
                    "LLM_Confidence_Desc": 0.55,
                    "LLM_Reason_Desc": "Fallback heuristic (LLM unavailable)",
                },
            ],
            dtype=object,
        )

        result = _serialize_joule_line_items(enriched)

        self.assertEqual(
            result,
            [
                {
                    "description": "Brake pads",
                    "net_amount": 100,
                    "quantity": 2,
                    "unit_price": 50.5,
                    "ai_suggested_commodity_code": "RC0001",
                    "ai_confidence_score": "91%",
                    "ai_reasoning": "Best semantic match.",
                },
                {
                    "description": "Not detected",
                    "net_amount": "Not detected",
                    "quantity": "Not detected",
                    "unit_price": "Not detected",
                    "ai_suggested_commodity_code": "UNSURE",
                    "ai_confidence_score": "55%",
                    "ai_reasoning": "Fallback heuristic (LLM unavailable)",
                },
            ],
        )
        self.assertEqual(set(result[0]), {
            "description",
            "net_amount",
            "quantity",
            "unit_price",
            "ai_suggested_commodity_code",
            "ai_confidence_score",
            "ai_reasoning",
        })

    def test_pipeline_payload_contains_all_serialized_joule_rows(self) -> None:
        """The persisted result payload includes the complete structured Joule result."""

        enriched = pd.DataFrame(
            [{
                "description": "Brake pads",
                "netAmount": 100,
                "quantity": 2,
                "unitPrice": 50,
                "LLM_Suggestion_Desc": "RC0001",
                "LLM_Confidence_Desc": 0.91,
                "LLM_Reason_Desc": "Best semantic match.",
            }]
        )
        result = ExtractionResult(
            output_path=Path("result.xlsx"),
            headers_df=pd.DataFrame(),
            line_items_df=enriched,
            runtime_seconds=1.0,
            reference_data_version="v1",
        )

        with patch("app.services.extraction_service._run_embedding_pipeline", return_value=result):
            payload = run_extraction_for_paths([Path("sample.pdf")], ExtractionConfig(llm_verify=True))

        self.assertEqual(payload["joule_line_items"][0]["ai_confidence_score"], "91%")


if __name__ == "__main__":
    unittest.main()
