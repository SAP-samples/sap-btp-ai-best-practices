from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from app.models.metal_composition import (
    FinalMetalComposition,
    MetalCompositionResponse,
    TokenUsageSummary,
)
from app.services.metal_composition.config import MetalCompositionSettings
from app.services.metal_composition.classification_jobs import (
    SUPERSEDED_ERROR_MESSAGE,
    ClassificationJobStore,
)
from app.services.metal_composition.hts_catalog import (
    CODE_MAP_COLUMNS,
    CATALOG_COLUMNS,
    HanaHTSCatalogResolver,
)
from app.services.metal_composition.section_232_rulesets import Section232DraftRuleCandidate
from app.services.metal_composition.section_232_sources import InMemorySection232SourceStore
from app.services.metal_composition.service import (
    MetalCompositionPredictInput,
    MetalCompositionService,
    MissingDocumentsConfirmationRequiredError,
)
from app.services.metal_composition.ui_state import (
    InMemoryMetalCompositionUIStateStore,
    StoredDocumentReference,
)
from app.services.metal_composition.serving_store import (
    LocalSnapshotStore,
    WorkbookStore,
    build_serving_table_frame,
    load_serving_store,
    split_serving_table_frame,
)
from app.services.metal_composition.workflow import (
    DiagramPayload,
    MetalCompositionWorkflowRunner,
    TokenUsageRecorder,
    normalize_token_usage,
    validate_and_repair_final_composition,
)
from app.services.metal_composition.workflow.hts_fact_profile import (
    build_hts_fact_profile_input,
    synthesize_hts_fact_profile,
)
from app.services.metal_composition.workflow import hana_tree_search as hana_tree_search_module
from app.services.metal_composition.workflow.hana_tree_search import (
    build_hana_tree_search_context,
    run_hana_tree_search,
)
from app.services.metal_composition.workflow.original_data import (
    prompt_safe_source_row,
    prompt_safe_source_summary,
)
from app.services.metal_composition.workflow.trade_decision import (
    build_trade_decision_prompt,
    merge_candidate_entries,
    run_trade_decision,
)

ROOT = Path(__file__).resolve().parents[2]

class _ServiceTestWorkflowRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs["product_code"] == "FAIL":
            raise RuntimeError("workflow exploded")

        diagram_payloads = list(kwargs.get("diagram_payloads") or [])
        total_weight = float(kwargs.get("source_summary", {}).get("total_weight_gram", 0.0) or 0.0)
        final_composition = (
            kwargs.get("gcc_tracker_composition")
            or {
            "is_metal_item": True,
            "total_weight_grams": total_weight,
            "estimated_total_metal_grams": 10.0,
            "top_level_grams": {
                "steel": 10.0,
                "aluminum": 0.0,
                "copper": 0.0,
                "cast_iron": 0.0,
            },
            "steel_subtype_grams": {
                "electrical_steel": 0.0,
                "cold_rolled_coil_steel": 0.0,
                "hot_rolled_coil_steel": 0.0,
                "stainless_steel_304": 0.0,
                "stainless_steel_316": 0.0,
                "stainless_steel_bar": 0.0,
                "duplex_steel": 0.0,
                "cast_steel": 0.0,
            },
            "confidence": 0.8,
            "reasoning": "Mocked service test composition.",
            "provenance": {"dominant_source": "diagram"},
            }
        )
        result = {
            "final_composition": final_composition,
            "diagram_output": {
                "status": "completed" if diagram_payloads else "omitted",
                "diagram_count": len(diagram_payloads),
                "mode": kwargs.get("composition_mode", "diagram_manual"),
            },
            "hts_fact_profile": {"status": "completed"},
            "hana_tree_search_output": {"status": "completed"},
            "hts_resolution_output": {"status": "completed"},
            "timing": {"phases": {}, "summary": {}},
        }
        if kwargs.get("include_token_usage"):
            result["token_usage"] = TokenUsageSummary(
                entries=[
                    {
                        "phase": "diagram",
                        "task": "base_pass",
                        "model": "gpt-5",
                        "call_count": 2,
                        "input_tokens": 100,
                        "output_tokens": 25,
                        "total_tokens": 125,
                        "usage_available": True,
                    }
                ],
                input_tokens=100,
                output_tokens=25,
                total_tokens=125,
                missing_usage_entry_count=0,
            ).model_dump(mode="json")
        return result

def _sample_serving_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_df = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "normalized_product_code": "pc-9",
                "Product code": "PC-9",
                "PN Revised/ Standardized": "PN-9",
                "Part description": "Pump housing",
                "New Part Description": "Pump housing new",
                "Site": "Madrid",
                "Business Segment": "WI",
                "Priority": "P1",
                "Priority.1": "Pump spares",
                "Material Content Method": "ERP Extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 42.0,
                "Date Started": "2025-01-01",
                "Date Completed": "2025-01-02",
            }
        ]
    )
    prepared_df = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "row_id": 1,
                "Priority": "p1",
                "Business Segment": "wi",
                "Site": "madrid",
                "Product code": "PC-9",
                "PN Revised/ Standardized": "PN-9",
                "Part description": "Pump housing",
                "New Part Description": "Pump housing new",
                "Priority.1": "Pump spares",
                "Material Content Method": "erp extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 42.0,
                "date_started_days": 0.0,
                "date_completed_days": 1.0,
                "days_started_to_completed": 1.0,
                "started_month": 1.0,
                "completed_month": 1.0,
                "date_completed_missing": 0.0,
                "Steel_grams": 12.0,
                "Aluminum_grams": 30.0,
                "Copper_grams": 0.0,
                "Cast_Iron_grams": 0.0,
                "Electrical_Steel_grams": 12.0,
                "Cold_Rolled_Coil_Steel_grams": 0.0,
                "Hot_Rolled_Coil_Steel_grams": 0.0,
                "Stainless_Steel_304_grams": 0.0,
                "Stainless_Steel_316_grams": 0.0,
                "Stainless_Steel_Bar_grams": 0.0,
                "Duplex_Steel_grams": 0.0,
                "Cast_Steel_grams": 0.0,
                "Steel_present": True,
                "Aluminum_present": True,
                "Copper_present": False,
                "Cast_Iron_present": False,
                "Electrical_Steel_present": True,
                "Cold_Rolled_Coil_Steel_present": False,
                "Hot_Rolled_Coil_Steel_present": False,
                "Stainless_Steel_304_present": False,
                "Stainless_Steel_316_present": False,
                "Stainless_Steel_Bar_present": False,
                "Duplex_Steel_present": False,
                "Cast_Steel_present": False,
                "Electrical_Steel_share": 1.0,
                "Cold_Rolled_Coil_Steel_share": 0.0,
                "Hot_Rolled_Coil_Steel_share": 0.0,
                "Stainless_Steel_304_share": 0.0,
                "Stainless_Steel_316_share": 0.0,
                "Stainless_Steel_Bar_share": 0.0,
                "Duplex_Steel_share": 0.0,
                "Cast_Steel_share": 0.0,
                "Any_metal_present": True,
                "group_key": "PN-9",
                "presence_signature": "1100",
            }
        ]
    )
    return source_df, prepared_df

def _sample_settings(**overrides) -> MetalCompositionSettings:
    return MetalCompositionSettings(
        workbook_path=ROOT / "data" / "GCC Tracker.xlsb",
        api_env_path=ROOT / "api" / ".env",
        diagram_page_routing_enabled=False,
        **overrides,
    )

def _write_fake_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 fake")
    return path

def _upload_fake_pdf(service: MetalCompositionService, item_id: str, filename: str = "uploaded.pdf"):
    return service.upload_item_document(
        item_id,
        filename=filename,
        content=b"%PDF-1.4 uploaded test pdf",
    )

def _make_png_bytes(size: tuple[int, int] = (32, 32), color: str = "white") -> bytes:
    from io import BytesIO

    from PIL import Image

    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def _make_single_page_pdf_bytes(text: str = "TEST") -> bytes:
    import fitz

    document = fitz.open()
    page = document.new_page(width=200, height=120)
    page.insert_text((20, 50), text)
    return document.tobytes()

def _sample_hts_catalog_resolver() -> HanaHTSCatalogResolver:
    catalog_frame = pd.DataFrame(
        [
            {
                "code": "7208",
                "raw_code": "7208",
                "digits": 4,
                "chapter_number": 72,
                "heading_code": "7208",
                "family_6_code": "",
                "family_8_code": "",
                "indent": 0,
                "parent_code": "",
                "description": "Flat-rolled products of iron or nonalloy steel, of a width of 600 mm or more, hot-rolled, not clad, plated or coated",
                "path_description": "Flat-rolled products of iron or nonalloy steel, of a width of 600 mm or more, hot-rolled, not clad, plated or coated",
                "unit_of_quantity": "",
                "general_rate_of_duty": "",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "7208 flat rolled products hot rolled steel plate coil",
                "sort_order": 0,
            },
            {
                "code": "7208.52.0000",
                "raw_code": "7208.52.00.00",
                "digits": 10,
                "chapter_number": 72,
                "heading_code": "7208",
                "family_6_code": "7208.52",
                "family_8_code": "7208.52.00",
                "indent": 2,
                "parent_code": "7208",
                "description": "In coils, not further worked than hot-rolled",
                "path_description": "Flat-rolled products of iron or nonalloy steel, of a width of 600 mm or more, hot-rolled, not clad, plated or coated > In coils, not further worked than hot-rolled",
                "unit_of_quantity": "[\"kg\"]",
                "general_rate_of_duty": "Free",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "35%",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "7208.52.0000 hot rolled coil steel raw material",
                "sort_order": 1,
            },
            {
                "code": "7307",
                "raw_code": "7307",
                "digits": 4,
                "chapter_number": 73,
                "heading_code": "7307",
                "family_6_code": "",
                "family_8_code": "",
                "indent": 0,
                "parent_code": "",
                "description": "Tube or pipe fittings (for example, couplings, elbows, sleeves), of iron or steel",
                "path_description": "Tube or pipe fittings (for example, couplings, elbows, sleeves), of iron or steel",
                "unit_of_quantity": "",
                "general_rate_of_duty": "",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "7307 tube pipe fittings flange blind plate steel class 150 disc head",
                "sort_order": 2,
            },
            {
                "code": "7307.99",
                "raw_code": "7307.99",
                "digits": 6,
                "chapter_number": 73,
                "heading_code": "7307",
                "family_6_code": "7307.99",
                "family_8_code": "",
                "indent": 1,
                "parent_code": "7307",
                "description": "Other",
                "path_description": "Tube or pipe fittings (for example, couplings, elbows, sleeves), of iron or steel > Other",
                "unit_of_quantity": "",
                "general_rate_of_duty": "Free",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "45%",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "7307.99 flange blind disc head pipe fitting steel class 150",
                "sort_order": 3,
            },
            {
                "code": "7326",
                "raw_code": "7326",
                "digits": 4,
                "chapter_number": 73,
                "heading_code": "7326",
                "family_6_code": "",
                "family_8_code": "",
                "indent": 0,
                "parent_code": "",
                "description": "Other articles of iron or steel",
                "path_description": "Other articles of iron or steel",
                "unit_of_quantity": "",
                "general_rate_of_duty": "",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "7326 fabricated steel article disc coated diffuser",
                "sort_order": 4,
            },
            {
                "code": "7326.90",
                "raw_code": "7326.90",
                "digits": 6,
                "chapter_number": 73,
                "heading_code": "7326",
                "family_6_code": "7326.90",
                "family_8_code": "",
                "indent": 1,
                "parent_code": "7326",
                "description": "Other",
                "path_description": "Other articles of iron or steel > Other",
                "unit_of_quantity": "",
                "general_rate_of_duty": "Free",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "45%",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "7326.90 fabricated steel article diffuser disc coated component",
                "sort_order": 5,
            },
            {
                "code": "7318",
                "raw_code": "7318",
                "digits": 4,
                "chapter_number": 73,
                "heading_code": "7318",
                "family_6_code": "",
                "family_8_code": "",
                "indent": 0,
                "parent_code": "",
                "description": "Screws, bolts, nuts and similar articles, of iron or steel",
                "path_description": "Screws, bolts, nuts and similar articles, of iron or steel",
                "unit_of_quantity": "",
                "general_rate_of_duty": "",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "7318 screws bolts nuts and similar articles of iron or steel",
                "sort_order": 6,
            },
            {
                "code": "7318.24.0000",
                "raw_code": "7318.24.00.00",
                "digits": 10,
                "chapter_number": 73,
                "heading_code": "7318",
                "family_6_code": "7318.24",
                "family_8_code": "7318.24.00",
                "indent": 2,
                "parent_code": "7318",
                "description": "Cotters and cotter pins",
                "path_description": "Screws, bolts, nuts and similar articles, of iron or steel > Cotters and cotter pins",
                "unit_of_quantity": "[\"kg\"]",
                "general_rate_of_duty": "Free",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "8%",
                "quota_quantity": "",
                "additional_duties": "9903.80.01",
                "searchable_text": "7318.24.0000 cotters and cotter pins retaining ring steel",
                "sort_order": 7,
            },
            {
                "code": "7318.29.00",
                "raw_code": "7318.29.00",
                "digits": 8,
                "chapter_number": 73,
                "heading_code": "7318",
                "family_6_code": "7318.29",
                "family_8_code": "7318.29.00",
                "indent": 2,
                "parent_code": "7318",
                "description": "Nonthreaded articles, other",
                "path_description": "Screws, bolts, nuts and similar articles, of iron or steel > Nonthreaded articles, other",
                "unit_of_quantity": "[\"kg\"]",
                "general_rate_of_duty": "Free",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "8%",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "7318.29.00 nonthreaded other retaining ring steel",
                "sort_order": 8,
            },
            {
                "code": "8421",
                "raw_code": "8421",
                "digits": 4,
                "chapter_number": 84,
                "heading_code": "8421",
                "family_6_code": "",
                "family_8_code": "",
                "indent": 0,
                "parent_code": "",
                "description": "Centrifuges, including centrifugal dryers; filtering or purifying machinery and apparatus, for liquids or gases; parts thereof",
                "path_description": "Centrifuges, including centrifugal dryers; filtering or purifying machinery and apparatus, for liquids or gases; parts thereof",
                "unit_of_quantity": "",
                "general_rate_of_duty": "",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "8421 filtering purifying machinery apparatus liquids gases water parts",
                "sort_order": 9,
            },
            {
                "code": "8421.99.01",
                "raw_code": "8421.99.01",
                "digits": 8,
                "chapter_number": 84,
                "heading_code": "8421",
                "family_6_code": "8421.99",
                "family_8_code": "8421.99.01",
                "indent": 2,
                "parent_code": "8421",
                "description": "Other",
                "path_description": "Centrifuges and filtering machinery > Other",
                "unit_of_quantity": "",
                "general_rate_of_duty": "",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "8421.99.01 other parts of filtering or purifying machinery",
                "sort_order": 10,
            },
            {
                "code": "8421.99.0140",
                "raw_code": "8421.99.01.40",
                "digits": 10,
                "chapter_number": 84,
                "heading_code": "8421",
                "family_6_code": "8421.99",
                "family_8_code": "8421.99.01",
                "indent": 3,
                "parent_code": "8421.99.01",
                "description": "Parts of machinery and apparatus for filtering or purifying water",
                "path_description": "Centrifuges and filtering machinery > Other > Parts of machinery and apparatus for filtering or purifying water",
                "unit_of_quantity": "",
                "general_rate_of_duty": "Free",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "35%",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "8421.99.0140 parts of machinery and apparatus for filtering or purifying water",
                "sort_order": 11,
            },
            {
                "code": "8481",
                "raw_code": "8481",
                "digits": 4,
                "chapter_number": 84,
                "heading_code": "8481",
                "family_6_code": "",
                "family_8_code": "",
                "indent": 0,
                "parent_code": "",
                "description": "Taps, cocks, valves and similar appliances; parts thereof",
                "path_description": "Taps, cocks, valves and similar appliances; parts thereof",
                "unit_of_quantity": "",
                "general_rate_of_duty": "",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "8481 valve disc butterfly valve part steel nbr coated",
                "sort_order": 12,
            },
            {
                "code": "8481.90",
                "raw_code": "8481.90",
                "digits": 6,
                "chapter_number": 84,
                "heading_code": "8481",
                "family_6_code": "8481.90",
                "family_8_code": "",
                "indent": 1,
                "parent_code": "8481",
                "description": "Parts",
                "path_description": "Taps, cocks, valves and similar appliances; parts thereof > Parts",
                "unit_of_quantity": "",
                "general_rate_of_duty": "Free",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "35%",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "8481.90 valve disc part butterfly valve coated steel nbr",
                "sort_order": 13,
            },
            {
                "code": "8536",
                "raw_code": "8536",
                "digits": 4,
                "chapter_number": 85,
                "heading_code": "8536",
                "family_6_code": "",
                "family_8_code": "",
                "indent": 0,
                "parent_code": "",
                "description": "Electrical apparatus for switching or protecting electrical circuits, or for making connections to or in electrical circuits",
                "path_description": "Electrical apparatus for switching or protecting electrical circuits, or for making connections to or in electrical circuits",
                "unit_of_quantity": "",
                "general_rate_of_duty": "",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "8536 electrical apparatus diffuser connector circuit",
                "sort_order": 14,
            },
            {
                "code": "8536.90",
                "raw_code": "8536.90",
                "digits": 6,
                "chapter_number": 85,
                "heading_code": "8536",
                "family_6_code": "8536.90",
                "family_8_code": "",
                "indent": 1,
                "parent_code": "8536",
                "description": "Other apparatus",
                "path_description": "Electrical apparatus for switching or protecting electrical circuits, or for making connections to or in electrical circuits > Other apparatus",
                "unit_of_quantity": "",
                "general_rate_of_duty": "Free",
                "special_rate_of_duty": "",
                "column_2_rate_of_duty": "35%",
                "quota_quantity": "",
                "additional_duties": "",
                "searchable_text": "8536.90 electrical diffuser apparatus connector",
                "sort_order": 15,
            },
        ],
        columns=CATALOG_COLUMNS,
    )
    code_map_frame = pd.DataFrame(
        [
            {
                "source_code": "8421.99.0040",
                "target_code": "8421.99.0140",
                "mapping_type": "explicit_official_map",
                "source_basis": "Official HTS annotation",
                "effective_note": "Effective January 27, 2022.",
            }
        ],
        columns=CODE_MAP_COLUMNS,
    )
    return HanaHTSCatalogResolver(
        settings=_sample_settings(),
        catalog_frame=catalog_frame,
        code_map_frame=code_map_frame,
    )

def _sample_service(
    ui_state_store: InMemoryMetalCompositionUIStateStore | None = None,
    **setting_overrides,
) -> MetalCompositionService:
    source_df, prepared_df = _sample_serving_frames()
    return MetalCompositionService(
        serving_store=WorkbookStore(source_df=source_df, prepared_df=prepared_df),
        workflow_runner=_ServiceTestWorkflowRunner(),
        settings=_sample_settings(**setting_overrides),
        ui_state_store=ui_state_store or InMemoryMetalCompositionUIStateStore(),
        section_232_source_store=InMemorySection232SourceStore(),
    )

def _duplicate_product_service(tmp_path: Path) -> MetalCompositionService:
    source_df = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "normalized_product_code": "dup-9",
                "Product code": "DUP-9",
                "PN Revised/ Standardized": "PN-DUP-1",
                "Part description": "Bearing housing variant A",
                "New Part Description": "Bearing housing variant A",
                "Site": "Madrid",
                "Business Segment": "WI",
                "Priority": "P1",
                "Priority.1": "Duplicate test",
                "Material Content Method": "ERP Extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 21.0,
                "Date Started": "2025-01-01",
                "Date Completed": "2025-01-02",
            },
            {
                "source_row_id": 2,
                "normalized_product_code": "dup-9",
                "Product code": "DUP-9",
                "PN Revised/ Standardized": "PN-DUP-2",
                "Part description": "Bearing housing variant B",
                "New Part Description": "Bearing housing variant B",
                "Site": "Madrid",
                "Business Segment": "WI",
                "Priority": "P2",
                "Priority.1": "Duplicate test",
                "Material Content Method": "ERP Extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 34.0,
                "Date Started": "2025-01-03",
                "Date Completed": "2025-01-04",
            },
        ]
    )
    prepared_df = pd.DataFrame(
        [
            {"source_row_id": 1, "row_id": 1, "Total Weight (Gram)": 21.0},
            {"source_row_id": 2, "row_id": 2, "Total Weight (Gram)": 34.0},
        ]
    )
    return MetalCompositionService(
        serving_store=WorkbookStore(source_df=source_df, prepared_df=prepared_df),
        workflow_runner=_ServiceTestWorkflowRunner(),
        settings=_sample_settings(
            ui_state_db_path=tmp_path / "ui_state.sqlite3",
        ),
        ui_state_store=InMemoryMetalCompositionUIStateStore(),
        section_232_source_store=InMemorySection232SourceStore(),
    )

def test_batch_settings_defaults():
    settings = MetalCompositionSettings(
        workbook_path=ROOT / "data" / "GCC Tracker.xlsb",
        api_env_path=ROOT / "api" / ".env",
    )

    assert settings.batch_max_concurrency >= 1
    assert settings.batch_max_items >= 1
    assert settings.classification_job_worker_max_concurrency == 10
    assert settings.hts_k_candidates == 5

def test_build_predict_input_uses_gcc_tracker_mode_for_gcc_items_when_enabled(tmp_path):
    service = _duplicate_product_service(tmp_path)
    service.update_app_settings(use_gcc_tracker_metal_composition=True)

    context = service._resolve_item_context("gcc:2")  # noqa: SLF001
    predict_input = service._build_predict_input_for_context(context)  # noqa: SLF001

    assert predict_input.composition_mode == "gcc_tracker"
    assert predict_input.document_mode == "text_only"

def test_priority_detail_is_preserved_in_selected_source_payload(tmp_path):
    """Selected GCC source payloads should retain BY Priority as priority_detail."""

    service = _sample_service(
        ui_state_db_path=tmp_path / "ui_state.sqlite3",
    )

    submission = service.submit_predict_item_job("gcc:1", document_mode="text_only")
    service.drain_classification_jobs()
    detail = service.get_item_detail("gcc:1")

    assert detail.priority_detail == "Pump spares"
    assert detail.latest_classification is not None
    assert detail.latest_classification.selected_source is not None
    assert detail.latest_classification.selected_source.priority_detail == "Pump spares"

def test_priority_detail_is_exposed_to_llm_prompts_as_extra_item_context():
    """Prompt-safe item context should expose priority_detail without priority labels."""

    source_summary = {
        "source_row_id": 310,
        "source_kind": "gcc",
        "part_description": "DIFFUSER COAT.",
        "priority_detail": "PUMP SPARES",
    }
    source_row = {
        "Part description": "DIFFUSER COAT.",
        "Priority.1": "PUMP SPARES",
        "Priority": "EP",
        "Total Weight (Gram)": 84000.0,
    }

    safe_summary = prompt_safe_source_summary(source_summary)
    safe_row = prompt_safe_source_row(source_row)
    profile_input = build_hts_fact_profile_input(
        {
            "source_summary": source_summary,
            "source_row": source_row,
            "final_composition": {"top_level_grams": {"cast_iron": 78000.0}},
            "diagram_output": {"status": "omitted"},
        }
    )
    hana_context = build_hana_tree_search_context(
        {
            "product_code": "4006107",
            "source_summary": source_summary,
            "source_row": source_row,
            "hts_fact_profile": {
                "article_summary": "Cast iron diffuser coat.",
                "function_summary": "Pump spare diffuser coat.",
                "material_profile": {"top_level_grams": {"cast_iron": 78000.0}},
            },
            "final_composition": {"top_level_grams": {"cast_iron": 78000.0}},
        }
    )

    assert safe_summary["extra_item_context"] == "PUMP SPARES"
    assert "priority_detail" not in safe_summary
    assert safe_row["extra_item_context"] == "PUMP SPARES"
    assert "Priority.1" not in safe_row
    assert "Priority" not in safe_row
    assert profile_input["source_summary"]["extra_item_context"] == "PUMP SPARES"
    assert profile_input["feature_context"]["extra_item_context"] == "PUMP SPARES"
    assert "PUMP SPARES" in hana_context["phrases"]
    assert "pump" in hana_context["tokens"]
    assert "spares" in hana_context["tokens"]

def test_gcc_tracker_mode_defaults_on_for_text_only_predict(tmp_path):
    service = _sample_service(
        ui_state_db_path=tmp_path / "ui_state.sqlite3",
    )

    assert service.get_app_settings().use_gcc_tracker_metal_composition is True

def test_submit_predict_item_job_text_only_skips_pdf_requirement(tmp_path):
    service = _sample_service(
        ui_state_db_path=tmp_path / "ui_state.sqlite3",
    )

    submission = service.submit_predict_item_job("gcc:1", document_mode="text_only")
    processed_count = service.drain_classification_jobs()
    detail = service.get_item_detail("gcc:1")
    job_items = service.classification_job_store.get_job_items(submission.job_id)

    assert processed_count == 1
    assert job_items[0].document_mode == "text_only"
    assert detail.latest_classification is not None
    assert detail.latest_classification.status == "completed"
    assert detail.latest_classification.document_mode == "text_only"
    assert service.workflow_runner.calls[-1]["composition_mode"] == "gcc_tracker"
    assert service.workflow_runner.calls[-1]["document_mode"] == "text_only"
    assert service.workflow_runner.calls[-1]["diagram_payloads"] == []

def test_submit_predict_item_job_text_only_can_include_token_usage(tmp_path):
    service = _sample_service(
        ui_state_db_path=tmp_path / "ui_state.sqlite3",
    )

    submission = service.submit_predict_item_job(
        "gcc:1",
        document_mode="text_only",
        include_token_usage=True,
    )
    processed_count = service.drain_classification_jobs()
    detail = service.get_item_detail("gcc:1")
    job_items = service.classification_job_store.get_job_items(submission.job_id)

    assert processed_count == 1
    assert job_items[0].document_mode == "text_only"
    assert job_items[0].include_token_usage is True
    assert service.workflow_runner.calls[-1]["include_token_usage"] is True
    assert detail.latest_classification is not None
    assert detail.latest_classification.token_usage is not None
    assert detail.latest_classification.token_usage.total_tokens == 125

def test_submit_predict_item_job_with_documents_uses_assigned_pdf(tmp_path):
    service = _sample_service(
        ui_state_db_path=tmp_path / "ui_state.sqlite3",
    )
    _upload_fake_pdf(service, "gcc:1", "gcc-extra-evidence.pdf")

    submission = service.submit_predict_item_job("gcc:1", document_mode="with_documents")
    processed_count = service.drain_classification_jobs()
    detail = service.get_item_detail("gcc:1")
    job_items = service.classification_job_store.get_job_items(submission.job_id)

    assert processed_count == 1
    assert job_items[0].document_mode == "with_documents"
    assert detail.latest_classification is not None
    assert detail.latest_classification.status == "completed"
    assert detail.latest_classification.document_mode == "with_documents"
    assert len(service.workflow_runner.calls[-1]["diagram_payloads"]) == 1
    assert service.workflow_runner.calls[-1]["gcc_tracker_composition"]["top_level_grams"]["steel"] == 12.0

def test_submit_predict_item_job_legacy_switch_off_still_requires_pdf(tmp_path):
    service = _sample_service(
        ui_state_db_path=tmp_path / "ui_state.sqlite3",
    )
    service.update_app_settings(use_gcc_tracker_metal_composition=False)

    with pytest.raises(MissingDocumentsConfirmationRequiredError) as exc_info:
        service.submit_predict_item_job("gcc:1", document_mode="text_only")

    assert exc_info.value.detail.items[0].item_id == "gcc:1"
    assert exc_info.value.detail.items[0].docs_status == "No PDFs assigned"

def test_predict_returns_token_usage_when_requested():
    service = _sample_service()
    response = service.predict(
        MetalCompositionPredictInput(
            product_code="PC-9",
            diagram_payloads=[
                DiagramPayload(
                    filename="drawing.pdf",
                    content_type="application/pdf",
                    data=b"%PDF-1.4 fake",
                )
            ],
            include_token_usage=True,
        )
    )

    assert response.status == "completed"
    assert response.token_usage is not None
    assert response.token_usage.total_tokens == 125
    assert response.token_usage.entries[0].task == "base_pass"

def test_predict_unknown_product_without_manual_still_not_found():
    service = _sample_service()
    response = service.predict(
        MetalCompositionPredictInput(
            product_code="DOES-NOT-EXIST-999",
            diagram_payloads=[
                DiagramPayload(
                    filename="drawing.pdf",
                    content_type="application/pdf",
                    data=b"%PDF-1.4 fake",
                )
            ],
        )
    )

    assert response.status == "not_found"

def test_predict_preserves_diagram_zoom_diagnostics_in_agent_outputs():
    class _ZoomDiagnosticsWorkflowRunner:
        def run(self, **kwargs):
            diagram_payloads = list(kwargs.get("diagram_payloads") or [])
            return {
                "final_composition": {
                    "is_metal_item": True,
                    "total_weight_grams": 42.0,
                    "estimated_total_metal_grams": 10.0,
                    "top_level_grams": {
                        "steel": 10.0,
                        "aluminum": 0.0,
                        "copper": 0.0,
                        "cast_iron": 0.0,
                    },
                    "steel_subtype_grams": {
                        "electrical_steel": 0.0,
                        "cold_rolled_coil_steel": 0.0,
                        "hot_rolled_coil_steel": 0.0,
                        "stainless_steel_304": 0.0,
                        "stainless_steel_316": 0.0,
                        "stainless_steel_bar": 0.0,
                        "duplex_steel": 0.0,
                        "cast_steel": 0.0,
                    },
                    "confidence": 0.9,
                    "reasoning": "Mocked composition.",
                },
                "diagram_output": {
                    "status": "completed" if diagram_payloads else "omitted",
                    "diagram_count": len(diagram_payloads),
                    "zoom_diagnostics": {
                        "enabled": True,
                        "triggered": True,
                        "requested_regions": [
                            {
                                "page_ref": "P1",
                                "reason": "Unreadable material grade.",
                                "normalized_box": {"x0": 0.1, "y0": 0.1, "x1": 0.3, "y1": 0.2},
                            }
                        ],
                        "executed_regions": [
                            {
                                "crop_ref": "Z1",
                                "page_ref": "P1",
                                "source_filename": "drawing.pdf",
                                "page_number": 1,
                            }
                        ],
                        "skipped_regions": [],
                        "affected_pages": ["P1"],
                        "refinement_applied": True,
                        "refinement_changed_output": True,
                        "refinement_error": "",
                    },
                },
                "hts_fact_profile": {"status": "completed"},
                "hana_tree_search_output": {"status": "completed"},
                "hts_resolution_output": {"status": "completed"},
                "timing": {"phases": {}, "summary": {}},
            }

    service = _sample_service()
    service.workflow_runner = _ZoomDiagnosticsWorkflowRunner()

    response = service.predict(
        MetalCompositionPredictInput(
            product_code="PC-9",
            diagram_payloads=[
                DiagramPayload(
                    filename="drawing.pdf",
                    content_type="application/pdf",
                    data=b"%PDF-1.4 fake",
                )
            ],
        )
    )

    assert response.status == "completed"
    assert response.agent_outputs is not None
    assert response.agent_outputs.diagram["zoom_diagnostics"]["triggered"] is True
    assert response.agent_outputs.diagram["zoom_diagnostics"]["refinement_applied"] is True
    assert response.agent_outputs.diagram["zoom_diagnostics"]["affected_pages"] == ["P1"]

def test_validate_and_repair_final_composition_enforces_mass_balance():
    payload = {
        "is_metal_item": True,
        "confidence": 1.7,
        "top_level_grams": {
            "steel": 8.0,
            "aluminum": 5.0,
            "copper": -2.0,
            "cast_iron": 4.0,
        },
        "steel_subtype_grams": {
            "electrical_steel": 6.0,
            "cold_rolled_coil_steel": 5.0,
            "hot_rolled_coil_steel": 4.0,
            "stainless_steel_304": 0.0,
            "stainless_steel_316": 0.0,
            "stainless_steel_bar": 0.0,
            "duplex_steel": 0.0,
            "cast_steel": 0.0,
        },
        "reasoning": "Initial synthesis.",
    }

    result = validate_and_repair_final_composition(payload, total_weight_grams=10.0)
    top_level = result.top_level_grams.model_dump()
    subtypes = result.steel_subtype_grams.model_dump()

    assert sum(top_level.values()) <= 10.0 + 1e-9
    assert sum(subtypes.values()) <= top_level["steel"] + 1e-9
    assert result.confidence == 1.0
    assert "Validation repairs applied" in result.reasoning

def test_validate_and_repair_final_composition_normalizes_llm_confidence_labels():
    payload = {
        "is_metal_item": True,
        "confidence": "high",
        "top_level_grams": {
            "steel": "93.0",
            "aluminum": 0.0,
            "copper": 0.0,
            "cast_iron": 0.0,
        },
        "steel_subtype_grams": {
            "electrical_steel": 0.0,
            "cold_rolled_coil_steel": 0.0,
            "hot_rolled_coil_steel": 0.0,
            "stainless_steel_304": 0.0,
            "stainless_steel_316": 0.0,
            "stainless_steel_bar": 0.0,
            "duplex_steel": 0.0,
            "cast_steel": 0.0,
        },
        "reasoning": "LLM synthesis.",
    }

    result = validate_and_repair_final_composition(payload, total_weight_grams=120.0)

    assert result.confidence == 0.82
    assert result.top_level_grams.steel == 93.0

def test_workbook_store_parses_excel_serial_dates(monkeypatch):
    raw_df = pd.DataFrame(
        [
            {
                "Material Content Status": "Complete",
                "Product code": "PC-1",
                "PN Revised/ Standardized": "PN-1",
                "Part description": "Part one",
                "New Part Description": "Part one new",
                "Priority": "P1",
                "Priority.1": "Pump spares",
                "Site": "Madrid",
                "Business Segment": "WI",
                "Material Content Method": "ERP Extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 100.0,
                "Date Started": 45967,
                "Date Completed": 46022,
                "Electrical Steel - Grams": 0.0,
                "Cold-Rolled Coil Steel -Grams": 0.0,
                "Hot-Rolled Coil Steel - Grams": 0.0,
                "Stainless Steel 304 - Grams": 0.0,
                "Stainless Steel 316 - Grams": 0.0,
                "Stainless Steel Bar - Grams": 0.0,
                "Duplex Steel - Grams": 0.0,
                "Cast Steel - Gram": 0.0,
                "Aluminum  - Gram": 0.0,
                "Copper - Gram": 0.0,
                "Cast Iron - Gram": 0.0,
            }
        ]
    )

    monkeypatch.setattr("app.services.metal_composition.serving_store.pd.read_excel", lambda *args, **kwargs: raw_df)

    settings = _sample_settings()
    store = WorkbookStore.from_settings(settings)
    candidate = store.lookup_candidates("PC-1")[0]

    assert candidate.date_started == "2025-11-06"
    assert candidate.date_completed == "2025-12-31"


def test_workbook_store_uses_openpyxl_for_xlsx_workbooks(monkeypatch, tmp_path: Path) -> None:
    """Verify WorkbookStore chooses the standard Excel reader for XLSX files."""

    captured = {}
    source_df = pd.DataFrame({"source_row_id": [1], "normalized_product_code": ["pc-1"]})
    prepared_df = pd.DataFrame(
        {
            "source_row_id": [1],
            "Any_metal_present": [False],
            "Steel_grams": [0.0],
            "Aluminum_grams": [0.0],
            "Copper_grams": [0.0],
            "Cast_Iron_grams": [0.0],
        }
    )

    def fake_read_excel(path: Path, *, sheet_name: str, engine: str) -> pd.DataFrame:
        """Capture pandas reader arguments and return a placeholder raw frame."""

        captured["path"] = path
        captured["sheet_name"] = sheet_name
        captured["engine"] = engine
        return pd.DataFrame({"status": ["Complete"]})

    monkeypatch.setattr("app.services.metal_composition.serving_store.pd.read_excel", fake_read_excel)
    monkeypatch.setattr(
        "app.services.metal_composition.serving_store.build_serving_frames_from_raw_df",
        lambda _raw_df: (source_df, prepared_df),
    )

    workbook_path = tmp_path / "uploaded_tracker.xlsx"
    settings = MetalCompositionSettings(
        workbook_path=workbook_path,
        api_env_path=ROOT / "api" / ".env",
        diagram_page_routing_enabled=False,
    )
    store = WorkbookStore.from_settings(settings)

    assert captured == {
        "path": workbook_path,
        "sheet_name": "Material Master",
        "engine": "openpyxl",
    }
    assert len(store.source_df) == 1


def test_workbook_store_maps_gcc_tracker_gram_columns(monkeypatch):
    raw_df = pd.DataFrame(
        [
            {
                "Material Content Status": "Complete",
                "Product code": "3365710",
                "PN Revised/ Standardized": "3365710",
                "Part description": "Oil housing coated",
                "New Part Description": "Oil housing coated ECEU 00000",
                "Priority": "EP",
                "Priority.1": "Pumpset",
                "Site": None,
                "Business Segment": "WI",
                "Material Content Method": "Engineering Drawings",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 10300.0,
                "Date Started": 46017,
                "Date Completed": 46017,
                "Electrical Steel - Grams": 100.0,
                "Cold-Rolled Coil Steel -Grams": 10.0,
                "Hot-Rolled Coil Steel - Grams": None,
                "Stainless Steel 304 - Grams": 0.0,
                "Stainless Steel 316 - Grams": 0.0,
                "Stainless Steel Bar - Grams": 0.0,
                "Duplex Steel - Grams": "",
                "Cast Steel - Gram": 5.0,
                "Aluminum  - Gram": None,
                "Copper - Gram": "",
                "Cast Iron - Gram": 9100.0,
            }
        ]
    )

    monkeypatch.setattr("app.services.metal_composition.serving_store.pd.read_excel", lambda *args, **kwargs: raw_df)

    settings = _sample_settings()
    store = WorkbookStore.from_settings(settings)
    profile = store.get_gcc_metal_profile(0)

    assert profile.top_level_grams == {
        "steel": 115.0,
        "aluminum": 0.0,
        "copper": 0.0,
        "cast_iron": 9100.0,
    }
    assert profile.steel_subtype_grams["electrical_steel"] == 100.0
    assert profile.steel_subtype_grams["cold_rolled_coil_steel"] == 10.0
    assert profile.steel_subtype_grams["cast_steel"] == 5.0

def test_resolved_record_source_context_excludes_raw_target_columns():
    source_df = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "normalized_product_code": "pc-2",
                "Product code": "PC-2",
                "PN Revised/ Standardized": "PN-2",
                "Part description": "Diffuser disc",
                "New Part Description": "Diffuser disc coated",
                "Site": "Madrid",
                "Business Segment": "WI",
                "Priority": "P1",
                "Priority.1": "Pump spares",
                "Material Content Method": "ERP Extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 100.0,
                "Date Started": 45967,
                "Date Completed": 46022,
                "_parsed_total_weight_gram": 100.0,
                "_date_started": pd.Timestamp("2025-11-06"),
                "_date_completed": pd.Timestamp("2025-12-31"),
                "Electrical Steel - Grams": 90.0,
                "Copper - Gram": 0.0,
            }
        ]
    )
    prepared_df = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "row_id": 10,
                "Priority": "p1",
                "Business Segment": "wi",
                "Site": "madrid",
                "Product code": "PC-2",
                "PN Revised/ Standardized": "PN-2",
                "Part description": "Diffuser disc",
                "New Part Description": "Diffuser disc coated",
                "Priority.1": "Pump spares",
                "Material Content Method": "erp extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 100.0,
                "date_started_days": 0.0,
                "date_completed_days": 55.0,
                "days_started_to_completed": 55.0,
                "started_month": 11.0,
                "completed_month": 12.0,
                "date_completed_missing": 0.0,
            }
        ]
    )

    store = WorkbookStore(source_df=source_df, prepared_df=prepared_df)
    record, _ = store.resolve("PC-2")

    assert record is not None
    assert "Electrical Steel - Grams" not in record.source_row
    assert "Copper - Gram" not in record.source_row

def test_serving_table_round_trip_preserves_source_and_prepared_rows():
    source_df = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "normalized_product_code": "pc-2",
                "Product code": "PC-2",
                "PN Revised/ Standardized": "PN-2",
                "Part description": "Diffuser disc",
                "New Part Description": "Diffuser disc coated",
                "Site": "Madrid",
                "Business Segment": "WI",
                "Priority": "P1",
                "Priority.1": "Pump spares",
                "Material Content Method": "ERP Extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 100.0,
                "Date Started": "2025-11-06",
                "Date Completed": "2025-12-31",
            }
        ]
    )
    prepared_df = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "row_id": 10,
                "Priority": "p1",
                "Business Segment": "wi",
                "Site": "madrid",
                "Product code": "PC-2",
                "PN Revised/ Standardized": "PN-2",
                "Part description": "Diffuser disc",
                "New Part Description": "Diffuser disc coated",
                "Priority.1": "Pump spares",
                "Material Content Method": "erp extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 100.0,
                "date_started_days": 0.0,
                "date_completed_days": 55.0,
                "days_started_to_completed": 55.0,
                "started_month": 11.0,
                "completed_month": 12.0,
                "date_completed_missing": 0.0,
                "Steel_grams": 90.0,
                "Aluminum_grams": 10.0,
                "Copper_grams": 0.0,
                "Cast_Iron_grams": 0.0,
                "Electrical_Steel_grams": 90.0,
                "Cold_Rolled_Coil_Steel_grams": 0.0,
                "Hot_Rolled_Coil_Steel_grams": 0.0,
                "Stainless_Steel_304_grams": 0.0,
                "Stainless_Steel_316_grams": 0.0,
                "Stainless_Steel_Bar_grams": 0.0,
                "Duplex_Steel_grams": 0.0,
                "Cast_Steel_grams": 0.0,
                "Steel_present": True,
                "Aluminum_present": True,
                "Copper_present": False,
                "Cast_Iron_present": False,
                "Electrical_Steel_present": True,
                "Cold_Rolled_Coil_Steel_present": False,
                "Hot_Rolled_Coil_Steel_present": False,
                "Stainless_Steel_304_present": False,
                "Stainless_Steel_316_present": False,
                "Stainless_Steel_Bar_present": False,
                "Duplex_Steel_present": False,
                "Cast_Steel_present": False,
                "Electrical_Steel_share": 1.0,
                "Cold_Rolled_Coil_Steel_share": 0.0,
                "Hot_Rolled_Coil_Steel_share": 0.0,
                "Stainless_Steel_304_share": 0.0,
                "Stainless_Steel_316_share": 0.0,
                "Stainless_Steel_Bar_share": 0.0,
                "Duplex_Steel_share": 0.0,
                "Cast_Steel_share": 0.0,
                "Any_metal_present": True,
                "group_key": "PN-2",
                "presence_signature": "1100",
            }
        ]
    )

    serving_frame = build_serving_table_frame(source_df, prepared_df)
    restored_source, restored_prepared = split_serving_table_frame(serving_frame)

    assert restored_source.iloc[0]["Product code"] == "PC-2"
    assert restored_source.iloc[0]["Date Started"] == "2025-11-06"
    assert bool(restored_prepared.iloc[0]["Steel_present"]) is True
    assert float(restored_prepared.iloc[0]["Steel_grams"]) == 90.0

def test_serving_table_round_trip_preserves_minimal_gcc_gram_columns():
    source_df = pd.DataFrame(
        [
            {
                "source_row_id": 331,
                "normalized_product_code": "3365710",
                "Product code": "3365710",
                "PN Revised/ Standardized": "3365710",
                "Part description": "Oil housing coated",
                "New Part Description": "Oil housing coated ECEU 00000",
                "Site": None,
                "Business Segment": "WI",
                "Priority": "EP",
                "Priority.1": "Pumpset",
                "Material Content Method": "Engineering Drawings",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 10300.0,
                "Date Started": "2025-12-26",
                "Date Completed": "2025-12-26",
            }
        ]
    )
    prepared_df = pd.DataFrame(
        [
            {
                "source_row_id": 331,
                "Steel_grams": 115.0,
                "Aluminum_grams": 0.0,
                "Copper_grams": 0.0,
                "Cast_Iron_grams": 9100.0,
                "Electrical_Steel_grams": 100.0,
                "Cold_Rolled_Coil_Steel_grams": 10.0,
                "Hot_Rolled_Coil_Steel_grams": 0.0,
                "Stainless_Steel_304_grams": 0.0,
                "Stainless_Steel_316_grams": 0.0,
                "Stainless_Steel_Bar_grams": 0.0,
                "Duplex_Steel_grams": 0.0,
                "Cast_Steel_grams": 5.0,
            }
        ]
    )

    serving_frame = build_serving_table_frame(source_df, prepared_df)
    restored_source, restored_prepared = split_serving_table_frame(serving_frame)

    assert "prepared__Cast_Iron_grams" in serving_frame.columns
    assert restored_source.iloc[0]["Product code"] == "3365710"
    assert float(restored_prepared.iloc[0]["Steel_grams"]) == 115.0
    assert float(restored_prepared.iloc[0]["Cast_Iron_grams"]) == 9100.0
    assert float(restored_prepared.iloc[0]["Cast_Steel_grams"]) == 5.0

def test_load_serving_store_falls_back_to_snapshot(monkeypatch, tmp_path):
    settings = _sample_settings(
        data_source="hana",
        cache_dir=tmp_path,
        prewarm_on_startup=False,
    )
    source_df, prepared_df = _sample_serving_frames()

    snapshot_store = LocalSnapshotStore(settings)
    snapshot_store.save(WorkbookStore(source_df=source_df, prepared_df=prepared_df))
    monkeypatch.setattr(
        "app.services.metal_composition.serving_store._load_from_hana",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("hana down")),
    )

    result = load_serving_store(settings)

    assert result.loaded_from == "local_snapshot"
    assert result.store.lookup_candidates("PC-9")[0].source_row_id == 1

def test_snapshot_store_falls_back_to_pickle_when_parquet_engine_is_missing(monkeypatch, tmp_path):
    settings = _sample_settings(
        cache_dir=tmp_path,
        prewarm_on_startup=False,
    )
    source_df, prepared_df = _sample_serving_frames()
    snapshot_store = LocalSnapshotStore(settings)

    def fail_to_parquet(self, *_args, **_kwargs):
        raise ImportError("pyarrow is required for parquet support")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)

    snapshot_store.save(WorkbookStore(source_df=source_df, prepared_df=prepared_df))
    loaded_store = snapshot_store.load()

    assert (tmp_path / "source_lookup.pkl").exists()
    assert (tmp_path / "prepared_features.pkl").exists()
    assert loaded_store.metadata["snapshot_format"] == "pickle"
    assert loaded_store.lookup_candidates("PC-9")[0].source_row_id == 1

def test_load_serving_store_returns_hana_when_snapshot_save_fails(monkeypatch, tmp_path):
    settings = _sample_settings(
        data_source="hana",
        cache_dir=tmp_path,
        prewarm_on_startup=False,
    )
    source_df, prepared_df = _sample_serving_frames()
    hana_store = WorkbookStore(source_df=source_df, prepared_df=prepared_df)

    monkeypatch.setattr(
        "app.services.metal_composition.serving_store._load_from_hana",
        lambda *_args, **_kwargs: hana_store,
    )
    monkeypatch.setattr(
        LocalSnapshotStore,
        "save",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("disk full")),
    )

    result = load_serving_store(settings)

    assert result.loaded_from == "hana"
    assert result.store.lookup_candidates("PC-9")[0].source_row_id == 1
    assert result.startup_timing["details"]["snapshot_saved"] is False
    assert result.startup_timing["substeps"]["save_snapshot"]["status"] == "failed"

def test_native_chat_completion_omits_temperature_and_token_limit_for_gpt5(monkeypatch):
    calls: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {"ok": True}

    fake_module = types.ModuleType("gen_ai_hub.proxy.native.openai")
    fake_module.chat = types.SimpleNamespace(completions=FakeCompletions())
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.native.openai", fake_module)

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings())
    monkeypatch.setattr(runner.llm, "load_runtime_env", lambda: None)

    response = runner.llm.invoke_native_chat_completion(
        model_name="gpt-5",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        max_tokens=500,
    )

    assert response == {"ok": True}
    assert len(calls) == 1
    assert calls[0]["reasoning_effort"] == "low"
    assert "temperature" not in calls[0]
    assert "max_tokens" not in calls[0]
    assert "max_completion_tokens" not in calls[0]
    assert "max_tokens" not in calls[0]

def test_native_chat_completion_retries_with_max_completion_tokens(monkeypatch):
    calls: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "max_tokens" in kwargs:
                raise RuntimeError(
                    "Error code: 400 - {'error': {'code': 'unsupported_parameter', "
                    "\"message\": \"Unsupported parameter: 'max_tokens' is not supported with this model. "
                    "Use 'max_completion_tokens' instead.\"}}"
                )
            return {"ok": True}

    fake_module = types.ModuleType("gen_ai_hub.proxy.native.openai")
    fake_module.chat = types.SimpleNamespace(completions=FakeCompletions())
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.native.openai", fake_module)

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings())
    monkeypatch.setattr(runner.llm, "load_runtime_env", lambda: None)

    response = runner.llm.invoke_native_chat_completion(
        model_name="custom-model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=654,
    )

    assert response == {"ok": True}
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 654
    assert calls[1]["max_completion_tokens"] == 654
    assert "max_tokens" not in calls[1]

def test_native_chat_completion_routes_gemini_and_omits_temperature_and_token_limit(monkeypatch):
    calls: list[dict] = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(text='{"status":"completed"}')

    class FakeClient:
        def __init__(self, *, proxy_client):
            self.proxy_client = proxy_client
            self.models = FakeModels()

    fake_clients_module = types.ModuleType("gen_ai_hub.proxy.native.google_genai.clients")
    fake_clients_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.native.google_genai.clients", fake_clients_module)

    fake_proxy_module = types.ModuleType("gen_ai_hub.proxy.core.proxy_clients")
    fake_proxy_module.get_proxy_client = lambda name: {"proxy": name}
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.core.proxy_clients", fake_proxy_module)

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings())
    monkeypatch.setattr(runner.llm, "load_runtime_env", lambda: None)

    response = runner.llm.invoke_native_chat_completion(
        model_name="gemini-2.5-pro",
        messages=[
            {"role": "system", "content": "You are a classifier."},
            {"role": "user", "content": "Summarize the part."},
        ],
        temperature=0.4,
        max_tokens=321,
        reasoning_effort="low",
    )

    assert response.choices[0].message.content == '{"status":"completed"}'
    assert len(calls) == 1
    assert calls[0]["model"] == "gemini-2.5-pro"
    assert calls[0]["contents"] == [{"role": "user", "parts": [{"text": "Summarize the part."}]}]
    assert calls[0]["config"] == {"system_instruction": "You are a classifier."}
    assert "temperature" not in calls[0]
    assert "max_tokens" not in calls[0]
    assert "max_output_tokens" not in calls[0]

def test_native_chat_completion_converts_multimodal_messages_for_gemini(monkeypatch):
    calls: list[dict] = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(text='{"status":"completed"}')

    class FakeClient:
        def __init__(self, *, proxy_client):
            self.proxy_client = proxy_client
            self.models = FakeModels()

    fake_clients_module = types.ModuleType("gen_ai_hub.proxy.native.google_genai.clients")
    fake_clients_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.native.google_genai.clients", fake_clients_module)

    fake_proxy_module = types.ModuleType("gen_ai_hub.proxy.core.proxy_clients")
    fake_proxy_module.get_proxy_client = lambda name: {"proxy": name}
    monkeypatch.setitem(sys.modules, "gen_ai_hub.proxy.core.proxy_clients", fake_proxy_module)

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings())
    monkeypatch.setattr(runner.llm, "load_runtime_env", lambda: None)

    runner.llm.invoke_native_chat_completion(
        model_name="gemini-2.5-pro",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this diagram."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/diagram.png"}},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=111,
    )

    assert len(calls) == 1
    assert calls[0]["contents"] == [
        {
            "role": "user",
            "parts": [
                {"text": "Analyze this diagram."},
                {"inline_data": {"mime_type": "image/png", "data": "cG5n"}},
                {"text": "[image: https://example.com/diagram.png]"},
            ],
        }
    ]
    assert "config" not in calls[0]
    assert "temperature" not in calls[0]
    assert "max_tokens" not in calls[0]
    assert "max_output_tokens" not in calls[0]

def test_sample_settings_defaults_diagram_model_to_gpt5():
    assert _sample_settings().diagram_model_name == "gpt-5"

def test_analyze_diagrams_omits_explicit_token_cap(monkeypatch):
    captured: dict = {}

    def fake_preprocess(payload, settings):
        return payload, {"preprocessing": "skipped"}

    def fake_convert_pdf(payload, settings):
        return [payload]

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            '{"status":"completed","extracted_codes":[],"is_likely_metal_item":true,'
                            '"estimated_metal_share":0.5,"material_cues":[],"uncertainty_notes":[]}'
                        )
                    )
                )
            ]
        )

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(diagram_model_name="diagram-model"))
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.preprocess_diagram_payload", fake_preprocess
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.convert_pdf_to_images", fake_convert_pdf
    )
    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    from app.services.metal_composition.workflow.diagrams import analyze_diagrams

    result = analyze_diagrams(
        [DiagramPayload(filename="diagram.png", content_type="image/png", data=b"png")],
        runner.settings,
        runner.llm,
    )

    assert result["status"] == "completed"
    assert captured["model_name"] == "diagram-model"
    assert "max_tokens" not in captured
    prompt_text = captured["messages"][0]["content"][0]["text"]
    assert "Return 1.0 when the supplied item is shown as a single-piece metal part" in prompt_text
    assert "Do not reduce the share for speculative paint, labels, coatings, packaging" in prompt_text
    assert '"metal_share_basis": one of "explicit_full_metal", "explicit_partial_non_metal"' in prompt_text
    assert '"non_metal_evidence": array of short concrete strings' in prompt_text
    assert '"metal_share_reasoning": one short sentence' in prompt_text
    assert '"zoom_requests": optional array of objects with keys "page_ref", "reason", "normalized_box"' in prompt_text
    assert "you MUST request at least one targeted zoom region for the relevant page_ref" in prompt_text
    assert "Do not both claim that blocking text is too small or unreadable and return zoom_requests as []" in prompt_text
    assert "Any explanatory text fields you return must be user-facing." in prompt_text
    assert "Avoid phrases such as `explicit_full_metal` or `inferred_full_metal`" in prompt_text
    assert "A material standard, alloy sheet, or internal standard page does not count as weight evidence" in prompt_text
    assert "prefer that source over general standards or indirect references" in prompt_text
    assert "Do not use speculative company-practice language such as `typical company practice`" in prompt_text
    assert result["zoom_diagnostics"]["refinement_applied"] is False
    assert result["zoom_diagnostics"]["triggered"] is False

def test_preprocess_diagram_payload_uses_pdf_specific_limits_for_rendered_pages():
    from io import BytesIO

    from PIL import Image

    from app.services.metal_composition.workflow.diagrams import preprocess_diagram_payload

    image = Image.frombytes("RGB", (1400, 1000), os.urandom(1400 * 1000 * 3))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    settings = _sample_settings(
        image_max_dimension=1024,
        image_max_bytes=262144,
        pdf_image_max_dimension=3072,
        pdf_image_max_bytes=10 * 1024 * 1024,
    )
    processed, details = preprocess_diagram_payload(
        DiagramPayload(
            filename="00_3365100_0_13_page_1.png",
            content_type="image/png",
            data=buffer.getvalue(),
            source_filename="00_3365100_0_13.pdf",
            page_number=1,
        ),
        settings,
    )

    assert details["pdf_derived_page"] is True
    assert details["processed_size"] == [1400, 1000]
    assert processed.content_type == "image/png"
    assert len(processed.data) > settings.image_max_bytes

def test_analyze_diagrams_extracts_matched_weight_and_item_context(monkeypatch):
    captured: dict = {}

    def fake_preprocess(payload, settings):  # noqa: ARG001
        return payload, {"preprocessing": "skipped"}

    def fake_convert_pdf(payload, settings):  # noqa: ARG001
        return [payload]

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "completed",
                                "extracted_codes": [],
                                "is_likely_metal_item": True,
                                "estimated_metal_share": 1.0,
                                "metal_share_certainty": "exact",
                                "metal_share_basis": "explicit_full_metal",
                                "non_metal_evidence": [],
                                "metal_share_reasoning": "Single matched metal variant.",
                                "matched_identifiers": ["279 76 02"],
                                "weight_evidence": [
                                    {
                                        "identifier": "279 76 02",
                                        "applies_to_current_item": True,
                                        "raw_weight": "0.093kg",
                                        "numeric_value": 0.093,
                                        "unit": "kg",
                                        "source_excerpt": "Variant 279 76 02 0.093kg",
                                        "is_explicit": True,
                                        "match_confidence": 0.99,
                                    }
                                ],
                                "material_evidence": [
                                    {
                                        "identifier": "279 76 02",
                                        "applies_to_current_item": True,
                                        "raw_material": "steel",
                                        "normalized_metal": "steel",
                                        "source_excerpt": "Material column shows steel",
                                        "is_explicit": True,
                                        "match_confidence": 0.92,
                                    }
                                ],
                                "material_cues": ["steel"],
                                "material_properties": [],
                                "context_of_use": "Tube variant",
                                "hts_hints": [],
                                "uncertainty_notes": [],
                                "provenance_flags": ["pdf_explicit"],
                            }
                        )
                    )
                )
            ]
        )

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(diagram_model_name="diagram-model"))
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.preprocess_diagram_payload",
        fake_preprocess,
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.convert_pdf_to_images",
        fake_convert_pdf,
    )
    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    from app.services.metal_composition.workflow.diagrams import analyze_diagrams

    result = analyze_diagrams(
        [DiagramPayload(filename="diagram.png", content_type="image/png", data=b"png")],
        runner.settings,
        runner.llm,
        product_code="2797602",
        source_summary={
            "pn_revised_standardized": "PN-2797602",
            "part_description": "Tube",
            "new_part_description": "Tube variant",
            "business_segment": "WI",
            "total_weight_gram": 93.0,
        },
    )

    prompt_text = captured["messages"][0]["content"][0]["text"]
    assert '"product_code": "2797602"' in prompt_text
    assert '"part_description": "Tube"' in prompt_text
    assert "business_segment" not in prompt_text
    assert "WI" not in prompt_text
    assert "2797602" in prompt_text
    assert "`2797602` should be treated as the same identifier as `279 76 02`" in prompt_text
    assert result["matched_identifiers"] == ["279 76 02"]
    assert result["metal_share_certainty"] == "exact"
    # Composition is now produced directly by the diagram step
    composition = result["composition"]
    assert composition["is_metal_item"] is True
    assert composition["total_weight_grams"] == 93.0
    assert composition["top_level_grams"]["steel"] == 93.0

def test_analyze_diagrams_defaults_metal_share_certainty_to_estimated_when_missing(monkeypatch):
    def fake_preprocess(payload, settings):  # noqa: ARG001
        return payload, {"preprocessing": "skipped"}

    def fake_convert_pdf(payload, settings):  # noqa: ARG001
        return [payload]

    def fake_invoke(**_kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "completed",
                                "extracted_codes": [],
                                "is_likely_metal_item": True,
                                "estimated_metal_share": 0.5,
                                "metal_share_basis": "explicit_partial_non_metal",
                                "non_metal_evidence": ["Rubber seal shown on drawing."],
                                "metal_share_reasoning": "The body is metal but the assembly includes non-metal content.",
                                "matched_identifiers": ["279 76 02"],
                                "weight_evidence": [],
                                "material_evidence": [
                                    {
                                        "identifier": "279 76 02",
                                        "applies_to_current_item": True,
                                        "raw_material": "steel",
                                        "normalized_metal": "steel",
                                        "source_excerpt": "Material column shows steel",
                                        "is_explicit": True,
                                        "match_confidence": 0.92,
                                    }
                                ],
                                "material_cues": ["steel"],
                                "material_properties": [],
                                "context_of_use": "Valve insert",
                                "hts_hints": [],
                                "uncertainty_notes": [],
                                "provenance_flags": ["pdf_explicit"],
                            }
                        )
                    )
                )
            ]
        )

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(diagram_model_name="diagram-model"))
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.preprocess_diagram_payload",
        fake_preprocess,
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.convert_pdf_to_images",
        fake_convert_pdf,
    )
    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    from app.services.metal_composition.workflow.diagrams import analyze_diagrams

    result = analyze_diagrams(
        [DiagramPayload(filename="diagram.png", content_type="image/png", data=b"png")],
        runner.settings,
        runner.llm,
    )

    assert result["metal_share_certainty"] == "estimated"

def test_analyze_diagrams_preserves_source_filename_and_page_metadata(monkeypatch):
    captured: dict = {}

    def fake_preprocess(payload, settings):  # noqa: ARG001
        return payload, {"preprocessing": "skipped"}

    def fake_convert_pdf(payload, settings):  # noqa: ARG001
        return [
            DiagramPayload(
                filename="15_2805400_V_0_page_1.png",
                content_type="image/png",
                data=b"png",
                source_filename=payload.source_filename,
                page_number=1,
            )
        ]

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "completed",
                                "extracted_codes": ["EN 1706"],
                                "is_likely_metal_item": True,
                                "estimated_metal_share": 1.0,
                                "metal_share_basis": "explicit_full_metal",
                                "non_metal_evidence": [],
                                "metal_share_reasoning": "Explicit matched weight and material.",
                                "matched_identifiers": ["2805400"],
                                "weight_evidence": [
                                    {
                                        "identifier": "2805400",
                                        "applies_to_current_item": True,
                                        "raw_weight": "2.6kg",
                                        "numeric_value": 2.6,
                                        "unit": "kg",
                                        "normalized_grams": 2600,
                                        "source_excerpt": "WEIGHT 2.6 KG",
                                        "source_filename": "15_2805400_V_0.pdf",
                                        "page_number": 1,
                                        "is_explicit": True,
                                        "match_confidence": 0.99,
                                    }
                                ],
                                "material_evidence": [
                                    {
                                        "identifier": "2805400",
                                        "applies_to_current_item": True,
                                        "raw_material": "Aluminium casting",
                                        "normalized_metal": "aluminum",
                                        "normalized_steel_subtype": None,
                                        "source_excerpt": "ALUMINIUM CASTING",
                                        "source_filename": "M0404.4253.pdf",
                                        "page_number": 1,
                                        "is_explicit": True,
                                        "match_confidence": 0.97,
                                    }
                                ],
                                "material_cues": ["Aluminium casting"],
                                "material_properties": [],
                                "context_of_use": "Connection flange",
                                "hts_hints": [],
                                "uncertainty_notes": [],
                                "provenance_flags": ["pdf_explicit"],
                            }
                        )
                    )
                )
            ]
        )

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(diagram_model_name="diagram-model"))
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.preprocess_diagram_payload",
        fake_preprocess,
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.convert_pdf_to_images",
        fake_convert_pdf,
    )
    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    from app.services.metal_composition.workflow.diagrams import analyze_diagrams

    result = analyze_diagrams(
        [
            DiagramPayload(
                filename="15_2805400_V_0.pdf",
                content_type="application/pdf",
                data=b"%PDF",
                source_filename="15_2805400_V_0.pdf",
            )
        ],
        runner.settings,
        runner.llm,
        product_code="2805400",
        source_summary={
            "pn_revised_standardized": "2805400",
            "part_description": "CONNECTION FLANGE",
            "new_part_description": "CONNECTION FLANGE ECEU",
            "business_segment": "WI",
            "total_weight_gram": 2600.0,
        },
    )

    prompt_text = captured["messages"][0]["content"][0]["text"]
    assert '"source_filename", "page_number"' in prompt_text
    assert captured["messages"][0]["content"][1]["text"] == "Source label: P1 | 15_2805400_V_0.pdf page 1"
    assert result["matched_weight_evidence"]["source_filename"] == "15_2805400_V_0.pdf"
    assert result["matched_weight_evidence"]["page_number"] == 1
    assert result["matched_material_evidence"]["source_filename"] == "M0404.4253.pdf"
    assert result["matched_material_evidence"]["page_number"] == 1

def test_render_diagram_pages_assigns_stable_refs_for_pdf_and_image_inputs():
    from app.services.metal_composition.workflow.diagrams import render_diagram_pages

    pages, details = render_diagram_pages(
        [
            DiagramPayload(
                filename="drawing.pdf",
                content_type="application/pdf",
                data=_make_single_page_pdf_bytes("DRAWING"),
                source_filename="drawing.pdf",
            ),
            DiagramPayload(
                filename="photo.png",
                content_type="image/png",
                data=_make_png_bytes(),
            ),
        ],
        _sample_settings(),
    )

    assert [page.page_ref for page in pages] == ["P1", "P2"]
    assert pages[0].source_document_index == 0
    assert pages[0].source_filename == "drawing.pdf"
    assert pages[0].page_number == 1
    assert pages[0].rendered_width > 0
    assert pages[0].rendered_height > 0
    assert pages[1].source_document_index == 1
    assert pages[1].page_number is None
    assert pages[1].rendered_width > 0
    assert pages[1].rendered_height > 0
    assert details[0]["page_ref"] == "P1"
    assert details[1]["page_ref"] == "P2"

def test_render_zoom_crops_pdf_clip_clamps_and_preserves_source_mapping():
    from app.services.metal_composition.workflow.diagrams import render_diagram_pages, render_zoom_crops

    settings = _sample_settings(
        diagram_zoom_render_dpi=600,
        diagram_zoom_padding_ratio=0.03,
        diagram_zoom_image_max_dimension=4096,
        diagram_zoom_image_max_bytes=2 * 1024 * 1024,
    )
    rendered_pages, _details = render_diagram_pages(
        [
            DiagramPayload(
                filename="drawing.pdf",
                content_type="application/pdf",
                data=_make_single_page_pdf_bytes("SMALL TEXT"),
                source_filename="drawing.pdf",
            )
        ],
        settings,
    )

    crops, executed_regions, skipped_regions = render_zoom_crops(
        rendered_pages,
        zoom_requests=[
            {
                "page_ref": "P1",
                "reason": "Tiny title block text.",
                "normalized_box": {"x0": 0.01, "y0": 0.02, "x1": 0.08, "y1": 0.12},
            }
        ],
        settings=settings,
    )

    assert len(crops) == 1
    assert skipped_regions == []
    assert executed_regions[0]["page_ref"] == "P1"
    assert executed_regions[0]["render_details"]["crop_source"] == "pdf_clip"
    assert executed_regions[0]["normalized_box"]["x0"] == 0.0
    assert executed_regions[0]["normalized_box"]["y0"] == 0.0
    assert crops[0].source_filename == "drawing.pdf"
    assert crops[0].page_number == 1
    assert crops[0].rendered_width > 0
    assert crops[0].rendered_height > 0

def test_analyze_diagrams_runs_zoom_refinement_when_requested(monkeypatch):
    calls: list[dict] = []
    page_png = _make_png_bytes()

    def fake_render_pages(_diagram_payloads, _settings):
        from app.services.metal_composition.workflow.types import RenderedDiagramPage

        return (
            [
                RenderedDiagramPage(
                    page_ref="P1",
                    source_document_index=0,
                    filename="drawing_page_1.png",
                    content_type="image/png",
                    data=page_png,
                    source_filename="drawing.pdf",
                    page_number=1,
                    rendered_width=32,
                    rendered_height=32,
                    input_payload=DiagramPayload(
                        filename="drawing.pdf",
                        content_type="application/pdf",
                        data=b"%PDF-1.4 fake",
                        source_filename="drawing.pdf",
                    ),
                )
            ],
            [{"page_ref": "P1", "processed_size": [32, 32]}],
        )

    def fake_render_zoom_crops(_rendered_pages, *, zoom_requests, settings):  # noqa: ARG001
        from app.services.metal_composition.workflow.types import ZoomedDiagramCrop

        assert zoom_requests == [
            {
                "page_ref": "P1",
                "reason": "Unreadable material grade in title block.",
                "normalized_box": {"x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 0.3},
            }
        ]
        crop = ZoomedDiagramCrop(
            crop_ref="Z1",
            page_ref="P1",
            filename="drawing_zoom.png",
            content_type="image/png",
            data=page_png,
            normalized_box={"x0": 0.07, "y0": 0.07, "x1": 0.43, "y1": 0.33},
            rendered_width=64,
            rendered_height=64,
            source_filename="drawing.pdf",
            page_number=1,
        )
        return (
            [crop],
            [
                {
                    "crop_ref": "Z1",
                    "page_ref": "P1",
                    "reason": "Unreadable material grade in title block.",
                    "normalized_box": {"x0": 0.07, "y0": 0.07, "x1": 0.43, "y1": 0.33},
                    "source_filename": "drawing.pdf",
                    "page_number": 1,
                    "crop_size": [64, 64],
                    "render_details": {"crop_source": "pdf_clip"},
                    "preprocessing": {"processed_size": [64, 64]},
                }
            ],
            [],
        )

    def fake_invoke(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=json.dumps(
                                {
                                    "status": "completed",
                                    "extracted_codes": [],
                                    "is_likely_metal_item": True,
                                    "estimated_metal_share": 1.0,
                                    "metal_share_basis": "explicit_full_metal",
                                    "non_metal_evidence": [],
                                    "metal_share_reasoning": "Base pass found the part body.",
                                    "matched_identifiers": ["577 99 00"],
                                    "weight_evidence": [],
                                    "material_evidence": [],
                                    "material_cues": [],
                                    "material_properties": [],
                                    "context_of_use": "Pump component",
                                    "hts_hints": [],
                                    "uncertainty_notes": ["Material grade on drawing is unreadable."],
                                    "provenance_flags": ["pdf_inference"],
                                    "zoom_requests": [
                                        {
                                            "page_ref": "P1",
                                            "reason": "Unreadable material grade in title block.",
                                            "normalized_box": [0.1, 0.1, 0.4, 0.3],
                                        }
                                    ],
                                }
                            )
                        )
                    )
                ]
            )
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "completed",
                                "extracted_codes": ["EN-GJL-250"],
                                "is_likely_metal_item": True,
                                "estimated_metal_share": 1.0,
                                "metal_share_basis": "explicit_full_metal",
                                "non_metal_evidence": [],
                                "metal_share_reasoning": "Zoomed note identifies the cast grade explicitly.",
                                "matched_identifiers": ["577 99 00"],
                                "weight_evidence": [],
                                "material_evidence": [
                                    {
                                        "identifier": "577 99 00",
                                        "applies_to_current_item": True,
                                        "raw_material": "EN-GJL-250 cast iron",
                                        "normalized_metal": "cast_iron",
                                        "source_excerpt": "EN-GJL-250",
                                        "source_filename": "drawing.pdf",
                                        "page_number": 1,
                                        "is_explicit": True,
                                        "match_confidence": 0.97,
                                    }
                                ],
                                "material_cues": ["EN-GJL-250 cast iron"],
                                "material_properties": [],
                                "context_of_use": "Pump component",
                                "hts_hints": [],
                                "uncertainty_notes": [],
                                "provenance_flags": ["pdf_explicit"],
                                "zoom_requests": [],
                                "composition": {
                                    "is_metal_item": True,
                                    "total_weight_grams": 237000.0,
                                    "estimated_total_metal_grams": 237000.0,
                                    "top_level_grams": {
                                        "steel": 0.0,
                                        "aluminum": 0.0,
                                        "copper": 0.0,
                                        "cast_iron": 237000.0,
                                    },
                                    "steel_subtype_grams": {
                                        "electrical_steel": 0.0,
                                        "cold_rolled_coil_steel": 0.0,
                                        "hot_rolled_coil_steel": 0.0,
                                        "stainless_steel_304": 0.0,
                                        "stainless_steel_316": 0.0,
                                        "stainless_steel_bar": 0.0,
                                        "duplex_steel": 0.0,
                                        "cast_steel": 0.0,
                                    },
                                    "confidence": 0.92,
                                    "reasoning": "Zoomed evidence resolved the grade to cast iron.",
                                },
                            }
                        )
                    )
                )
            ]
        )

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(diagram_model_name="diagram-model"))
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.render_diagram_pages",
        fake_render_pages,
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.render_zoom_crops",
        fake_render_zoom_crops,
    )
    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    from app.services.metal_composition.workflow.diagrams import analyze_diagrams

    result = analyze_diagrams(
        [DiagramPayload(filename="drawing.pdf", content_type="application/pdf", data=b"%PDF-1.4 fake")],
        runner.settings,
        runner.llm,
        product_code="5779900",
        source_summary={"part_description": "Pump component", "total_weight_gram": 237000.0},
    )

    assert len(calls) == 2
    assert "Provisional base-pass JSON" in calls[1]["messages"][0]["content"][0]["text"]
    assert any(
        block["type"] == "text" and block["text"].startswith("Source label: Z1 | zoom from P1")
        for block in calls[1]["messages"][0]["content"]
    )
    assert result["uncertainty_notes"] == []
    assert result["matched_material_evidence"]["source_filename"] == "drawing.pdf"
    assert result["zoom_diagnostics"]["refinement_applied"] is True
    assert result["zoom_diagnostics"]["refinement_changed_output"] is True
    assert result["composition"]["top_level_grams"]["cast_iron"] == 237000.0

def test_analyze_diagrams_caps_and_dedupes_zoom_requests(monkeypatch):
    captured_zoom_requests: list[dict] = []

    def fake_render_pages(_diagram_payloads, _settings):
        from app.services.metal_composition.workflow.types import RenderedDiagramPage

        return (
            [
                RenderedDiagramPage(
                    page_ref="P1",
                    source_document_index=0,
                    filename="drawing_page_1.png",
                    content_type="image/png",
                    data=_make_png_bytes(),
                    source_filename="drawing.pdf",
                    page_number=1,
                    rendered_width=32,
                    rendered_height=32,
                    input_payload=DiagramPayload(
                        filename="drawing.pdf",
                        content_type="application/pdf",
                        data=b"%PDF-1.4 fake",
                        source_filename="drawing.pdf",
                    ),
                )
            ],
            [{"page_ref": "P1", "processed_size": [32, 32]}],
        )

    def fake_render_zoom_crops(_rendered_pages, *, zoom_requests, settings):  # noqa: ARG001
        captured_zoom_requests.extend(zoom_requests)
        return [], [], []

    def fake_invoke(**_kwargs):
        zoom_requests = [
            {"page_ref": "P1", "reason": "1", "normalized_box": [0.10, 0.10, 0.20, 0.20]},
            {"page_ref": "P1", "reason": "dup", "normalized_box": [0.101, 0.101, 0.201, 0.201]},
            {"page_ref": "P1", "reason": "2", "normalized_box": [0.20, 0.20, 0.30, 0.30]},
            {"page_ref": "P1", "reason": "3", "normalized_box": [0.30, 0.30, 0.40, 0.40]},
            {"page_ref": "P1", "reason": "4", "normalized_box": [0.40, 0.40, 0.50, 0.50]},
            {"page_ref": "P1", "reason": "5", "normalized_box": [0.50, 0.50, 0.60, 0.60]},
            {"page_ref": "P1", "reason": "6", "normalized_box": [0.60, 0.60, 0.70, 0.70]},
            {"page_ref": "P1", "reason": "7", "normalized_box": [0.70, 0.70, 0.80, 0.80]},
        ]
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "completed",
                                "extracted_codes": [],
                                "is_likely_metal_item": True,
                                "estimated_metal_share": 1.0,
                                "metal_share_basis": "explicit_full_metal",
                                "non_metal_evidence": [],
                                "metal_share_reasoning": "Base pass.",
                                "matched_identifiers": [],
                                "weight_evidence": [],
                                "material_evidence": [],
                                "material_cues": [],
                                "material_properties": [],
                                "context_of_use": "Pump component",
                                "hts_hints": [],
                                "uncertainty_notes": ["Tiny text needs zoom."],
                                "provenance_flags": ["pdf_inference"],
                                "zoom_requests": zoom_requests,
                            }
                        )
                    )
                )
            ]
        )

    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(diagram_model_name="diagram-model", diagram_zoom_max_requests=6)
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.render_diagram_pages",
        fake_render_pages,
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.render_zoom_crops",
        fake_render_zoom_crops,
    )
    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    from app.services.metal_composition.workflow.diagrams import analyze_diagrams

    result = analyze_diagrams(
        [DiagramPayload(filename="drawing.pdf", content_type="application/pdf", data=b"%PDF-1.4 fake")],
        runner.settings,
        runner.llm,
    )

    assert len(captured_zoom_requests) == 6
    assert captured_zoom_requests[0]["normalized_box"] == {"x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2}
    skipped_reasons = {item["reason"] for item in result["zoom_diagnostics"]["skipped_regions"]}
    assert "duplicate_box" in skipped_reasons
    assert "over_limit" in skipped_reasons
    assert result["zoom_diagnostics"]["refinement_applied"] is False

def test_analyze_diagrams_accepts_qualitative_match_confidence(monkeypatch):
    def fake_preprocess(payload, settings):  # noqa: ARG001
        return payload, {"preprocessing": "skipped"}

    def fake_convert_pdf(payload, settings):  # noqa: ARG001
        return [payload]

    def fake_invoke(**_kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "completed",
                                "extracted_codes": [],
                                "is_likely_metal_item": True,
                                "estimated_metal_share": 1.0,
                                "metal_share_basis": "explicit_full_metal",
                                "non_metal_evidence": [],
                                "metal_share_reasoning": "Single matched metal variant.",
                                "matched_identifiers": ["279 76 02"],
                                "weight_evidence": [
                                    {
                                        "identifier": "279 76 02",
                                        "applies_to_current_item": True,
                                        "raw_weight": "0.093kg",
                                        "numeric_value": 0.093,
                                        "unit": "kg",
                                        "source_excerpt": "Variant 279 76 02 0.093kg",
                                        "is_explicit": True,
                                        "match_confidence": "high",
                                    }
                                ],
                                "material_evidence": [
                                    {
                                        "identifier": "279 76 02",
                                        "applies_to_current_item": True,
                                        "raw_material": "steel",
                                        "normalized_metal": "steel",
                                        "source_excerpt": "Material column shows steel",
                                        "is_explicit": True,
                                        "match_confidence": "medium high",
                                    }
                                ],
                                "material_cues": ["steel"],
                                "material_properties": [],
                                "context_of_use": "Tube variant",
                                "hts_hints": [],
                                "uncertainty_notes": [],
                                "provenance_flags": ["pdf_explicit"],
                            }
                        )
                    )
                )
            ]
        )

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(diagram_model_name="diagram-model"))
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.preprocess_diagram_payload",
        fake_preprocess,
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.convert_pdf_to_images",
        fake_convert_pdf,
    )
    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    from app.services.metal_composition.workflow.diagrams import analyze_diagrams

    result = analyze_diagrams(
        [DiagramPayload(filename="diagram.png", content_type="image/png", data=b"png")],
        runner.settings,
        runner.llm,
        product_code="2797602",
        source_summary={
            "pn_revised_standardized": "PN-2797602",
            "part_description": "Tube",
            "new_part_description": "Tube variant",
            "business_segment": "WI",
            "total_weight_gram": 93.0,
        },
    )

    assert result["matched_weight_grams"] == 93.0
    assert result["matched_metal_type"] == "steel"

def test_metal_composition_response_accepts_saved_snapshots_without_new_ui_fields():
    payload = {
        "status": "completed",
        "product_code": "2805400",
        "final_composition": {
            "is_metal_item": True,
            "estimated_total_metal_grams": 2600.0,
            "top_level_grams": {
                "steel": 0.0,
                "aluminum": 2600.0,
                "copper": 0.0,
                "cast_iron": 0.0,
            },
            "steel_subtype_grams": {
                "electrical_steel": 0.0,
                "cold_rolled_coil_steel": 0.0,
                "hot_rolled_coil_steel": 0.0,
                "stainless_steel_304": 0.0,
                "stainless_steel_316": 0.0,
                "stainless_steel_bar": 0.0,
                "duplex_steel": 0.0,
                "cast_steel": 0.0,
            },
            "confidence": 0.9,
            "reasoning": "Legacy saved reasoning.",
            "user_reasoning": "Legacy simplified reasoning.",
        },
        "hts_classification": {
            "status": "completed",
            "confidence": 0.6,
            "reasoning": "Legacy HTS reasoning.",
            "needs_human_review": True,
            "best_candidate": {
                "code": "7609.00.0000",
                "description": "Aluminum tube or pipe fittings",
                "digits": 10,
                "confidence": 0.6,
                "reasoning": "Legacy candidate reasoning.",
            },
            "candidates": [],
        },
        "section_232_assessment": {
            "status": "completed",
            "decision": "subject",
            "confidence": 0.98,
            "basis_summary": "Legacy section 232 summary.",
        },
    }

    result = MetalCompositionResponse.model_validate(payload)

    assert result.final_composition is not None
    assert result.final_composition.reasoning == "Legacy saved reasoning."
    assert result.final_composition.metal_rows == []
    assert result.hts_classification is not None
    assert result.section_232_assessment is not None
    assert result.section_232_assessment.needs_human_review is False
    assert result.section_232_assessment.weight_rule_applied is False

def test_synthesize_hts_fact_profile_uses_configured_model(monkeypatch):
    captured: dict = {}

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            '{"status":"completed","article_summary":"pump housing",'
                            '"function_summary":"pump spare housing","material_profile":'
                            '{"is_metal_item":true,"estimated_total_metal_grams":12.0,'
                            '"top_level_grams":{"steel":12.0,"aluminum":0.0,"copper":0.0,"cast_iron":0.0},'
                            '"steel_subtype_grams":{"electrical_steel":12.0,"cold_rolled_coil_steel":0.0,'
                            '"hot_rolled_coil_steel":0.0,"stainless_steel_304":0.0,"stainless_steel_316":0.0,'
                            '"stainless_steel_bar":0.0,"duplex_steel":0.0,"cast_steel":0.0},'
                            '"confidence":0.8,"reasoning":"Profile reasoning"},'
                            '"diagram_clues":[],"heading_hypotheses":["8413"],'
                            '"discriminator_notes":[],"reasoning":"Configured HTS profile model used.",'
                            '"confidence":0.82}'
                        )
                    )
                )
            ]
        )

    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(hts_fact_profile_model_name="profile-model")
    )
    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    state = {
        "source_summary": {
            "part_description": "Pump housing",
            "priority": "P1",
            "business_segment": "WI",
            "site": "Emmaboda",
            "date_started": "2026-01-01",
            "date_completed": "2026-01-31",
        },
        "source_row": {
            "Part description": "Pump housing",
            "Priority": "P1",
            "Business Segment": "WI",
            "Site": "Emmaboda",
            "Date Started": "2026-01-01",
            "Date Completed": "2026-01-31",
        },
        "final_composition": {
            "is_metal_item": True,
            "estimated_total_metal_grams": 12.0,
            "top_level_grams": {"steel": 12.0, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
            "steel_subtype_grams": {"electrical_steel": 12.0},
            "confidence": 0.9,
            "reasoning": "Final composition reasoning",
        },
        "diagram_output": {"status": "omitted"},
    }
    profile, details = synthesize_hts_fact_profile(state, runner.settings, runner.llm)

    assert profile["heading_hypotheses"] == ["8413"]
    assert captured["model_name"] == "profile-model"
    prompt_text = captured["messages"][1]["content"]
    assert "priority" not in prompt_text.lower()
    assert "business_segment" not in prompt_text
    assert "Business Segment" not in prompt_text
    assert "site" not in prompt_text.lower()
    assert "Date Started" not in prompt_text
    assert "Date Completed" not in prompt_text
    assert "2026-01-01" not in prompt_text
    assert "2026-01-31" not in prompt_text
    assert details["model"] == "profile-model"

def test_synthesize_hts_fact_profile_normalizes_tree_search_directives(monkeypatch):
    """HTS hint synthesis should pass safe tree-search directives to HANA recall."""

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(hts_fact_profile_model_name="profile-model"))

    def fake_invoke(**kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "completed",
                                "article_summary": "Cast iron impeller",
                                "function_summary": "Possible pump component",
                                "material_profile": {
                                    "is_metal_item": True,
                                    "estimated_total_metal_grams": 7520.0,
                                    "top_level_grams": {"cast_iron": 7520.0},
                                    "steel_subtype_grams": {},
                                    "confidence": 0.95,
                                    "reasoning": "GCC tracker reports cast iron.",
                                },
                                "diagram_clues": [],
                                "heading_hypotheses": ["8413", "not-a-heading"],
                                "tree_search_directives": [
                                    {
                                        "heading": "8413",
                                        "branch_focus": ["parts", "residual_other", "nonsense"],
                                        "include_residual_children": True,
                                        "rationale": "Impeller appears to be a component; explore pump-parts branches.",
                                        "confidence": "72%",
                                    },
                                    {
                                        "heading": "7203",
                                        "branch_focus": ["material_articles"],
                                        "include_residual_children": False,
                                        "rationale": "Cast iron material fallback.",
                                        "confidence": 0.31,
                                    },
                                    {
                                        "heading": "84AB",
                                        "branch_focus": ["parts"],
                                        "include_residual_children": True,
                                        "rationale": "Invalid heading should be ignored.",
                                        "confidence": 0.9,
                                    },
                                ],
                                "discriminator_notes": [],
                                "reasoning": "Configured HTS profile model used.",
                                "confidence": 0.72,
                            }
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    profile, _details = synthesize_hts_fact_profile(
        {
            "source_summary": {
                "new_part_description": "IMPELLER ECEU 00000",
                "priority_detail": "CSTG IMPLR 14DX",
            },
            "source_row": {
                "New Part Description": "IMPELLER ECEU 00000",
                "Priority.1": "CSTG IMPLR 14DX",
            },
            "final_composition": {
                "is_metal_item": True,
                "estimated_total_metal_grams": 7520.0,
                "top_level_grams": {"cast_iron": 7520.0},
                "steel_subtype_grams": {},
                "confidence": 1.0,
            },
        },
        runner.settings,
        runner.llm,
    )

    assert profile["heading_hypotheses"] == ["8413"]
    assert profile["tree_search_directives"] == [
        {
            "heading": "8413",
            "branch_focus": ["parts", "residual_other"],
            "include_residual_children": True,
            "rationale": "Impeller appears to be a component; explore pump-parts branches.",
            "confidence": 0.72,
        },
        {
            "heading": "7203",
            "branch_focus": ["material_articles"],
            "include_residual_children": False,
            "rationale": "Cast iron material fallback.",
            "confidence": 0.31,
        },
    ]

def test_hana_tree_search_context_carries_profile_tree_search_directives():
    """HANA recall context should receive normalized profile directives."""

    context = build_hana_tree_search_context(
        {
            "source_summary": {"new_part_description": "IMPELLER ECEU 00000"},
            "source_row": {"New Part Description": "IMPELLER ECEU 00000"},
            "hts_fact_profile": {
                "article_summary": "Cast iron impeller",
                "function_summary": "Possible pump component",
                "heading_hypotheses": ["8413"],
                "tree_search_directives": [
                    {
                        "heading": "8413",
                        "branch_focus": ["parts"],
                        "include_residual_children": True,
                        "rationale": "Explore pump parts.",
                        "confidence": 0.72,
                    }
                ],
            },
            "final_composition": {"top_level_grams": {"cast_iron": 7520.0}},
        }
    )

    assert context["tree_search_directives"] == [
        {
            "heading": "8413",
            "branch_focus": ["parts"],
            "include_residual_children": True,
            "rationale": "Explore pump parts.",
            "confidence": 0.72,
        }
    ]

def test_trade_decision_prompt_excludes_non_identifying_original_fields():
    _system_prompt, user_prompt = build_trade_decision_prompt(
        state={
            "product_code": "ABC-123",
            "source_summary": {
                "pn_revised_standardized": "PN-123",
                "part_description": "Pump body",
                "new_part_description": "Cast pump body",
                "total_weight_gram": 42.0,
                "priority": "P1",
                "business_segment": "WI",
                "site": "Emmaboda",
                "date_started": "2026-01-01",
                "date_completed": "2026-01-31",
            },
            "source_row": {
                "PN Revised/ Standardized": "PN-123",
                "Part description": "Pump body",
                "New Part Description": "Cast pump body",
                "Total Weight (Gram)": 42.0,
                "Priority": "P1",
                "Business Segment": "WI",
                "Site": "Emmaboda",
                "Date Started": "2026-01-01",
                "Date Completed": "2026-01-31",
            },
            "final_composition": {},
            "hts_fact_profile": {},
            "hana_tree_search_output": {},
        },
        resolution_output={"validated_candidates": [], "rejected_candidates": []},
    )

    assert "PN-123" in user_prompt
    assert "Pump body" in user_prompt
    assert "priority" not in user_prompt.lower()
    assert "business_segment" not in user_prompt
    assert "Business Segment" not in user_prompt
    assert "site" not in user_prompt.lower()
    assert "Date Started" not in user_prompt
    assert "Date Completed" not in user_prompt
    assert "2026-01-01" not in user_prompt
    assert "2026-01-31" not in user_prompt


def test_trade_decision_prompt_forbids_internal_status_leaks():
    system_prompt, _user_prompt = build_trade_decision_prompt(
        state={
            "product_code": "ABC-123",
            "source_summary": {"part_description": "Pump impeller"},
            "source_row": {},
            "final_composition": {},
            "hts_fact_profile": {},
            "hana_tree_search_output": {},
        },
        resolution_output={
            "validated_candidates": [
                {
                    "code": "8413.91.9085",
                    "validation_status": "current_exact",
                    "description": "Pump parts",
                }
            ],
            "rejected_candidates": [],
        },
    )

    assert "Do not mention internal field names or raw enum values" in system_prompt
    assert "current_exact" in system_prompt
    assert "exact active entry in the current HTS catalog" in system_prompt


def test_synthesize_hts_fact_profile_accepts_qualitative_confidence(monkeypatch):
    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(hts_fact_profile_model_name="profile-model"))

    def fake_invoke(**kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            '{"status":"completed","article_summary":"retaining ring",'
                            '"function_summary":"Top 50 Priority PSC","material_profile":'
                            '{"is_metal_item":true,"estimated_total_metal_grams":18.0,'
                            '"top_level_grams":{"steel":18.0,"aluminum":0.0,"copper":0.0,"cast_iron":0.0},'
                            '"steel_subtype_grams":{"electrical_steel":0.0,"cold_rolled_coil_steel":18.0,'
                            '"hot_rolled_coil_steel":0.0,"stainless_steel_304":0.0,"stainless_steel_316":0.0,'
                            '"stainless_steel_bar":0.0,"duplex_steel":0.0,"cast_steel":0.0},'
                            '"confidence":"high","reasoning":"Profile reasoning"},'
                            '"diagram_clues":[],"heading_hypotheses":["7318"],'
                            '"discriminator_notes":[],"reasoning":"Configured HTS profile model used.",'
                            '"confidence":"very high"}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)
    state = {
        "source_summary": {
            "part_description": "Top 50 Priority PSC",
            "new_part_description": "RETAINING RING DIN 471-60X3-SPRING",
        },
        "source_row": {
            "Part description": "Top 50 Priority PSC",
            "New Part Description": "RETAINING RING DIN 471-60X3-SPRING",
        },
        "final_composition": {
            "is_metal_item": True,
            "estimated_total_metal_grams": 18.0,
            "top_level_grams": {"steel": 18.0, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
            "steel_subtype_grams": {"cold_rolled_coil_steel": 18.0},
            "confidence": 0.9,
            "reasoning": "Final composition reasoning",
        },
        "diagram_output": {"status": "omitted"},
    }
    profile, details = synthesize_hts_fact_profile(state, runner.settings, runner.llm)

    assert details["fallback_used"] is False
    assert profile["confidence"] == 0.94
    assert profile["material_profile"]["confidence"] == 0.82

def test_token_usage_recorder_aggregates_repeated_calls():
    recorder = TokenUsageRecorder()

    recorder.record(
        phase="diagram",
        task="base_pass",
        model="gpt-5",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    recorder.record(
        phase="diagram",
        task="base_pass",
        model="gpt-5",
        usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    )
    recorder.record(
        phase="diagram",
        task="refinement",
        model="gpt-5",
        usage=None,
    )

    summary = recorder.build_summary()

    assert summary.input_tokens == 17
    assert summary.output_tokens == 8
    assert summary.total_tokens == 25
    assert summary.missing_usage_entry_count == 1
    assert summary.entries[0].call_count == 2
    assert summary.entries[0].task == "base_pass"
    assert summary.entries[1].usage_available is False

def test_normalize_token_usage_accepts_openai_style_usage():
    normalized = normalize_token_usage(
        types.SimpleNamespace(prompt_tokens=12, completion_tokens=4, total_tokens=16)
    )

    assert normalized == {
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16,
    }

def test_normalize_token_usage_accepts_google_style_usage():
    normalized = normalize_token_usage(
        {
            "inputTokenCount": 21,
            "outputTokenCount": 9,
            "totalTokenCount": 30,
        }
    )

    assert normalized == {
        "input_tokens": 21,
        "output_tokens": 9,
        "total_tokens": 30,
    }

def test_normalize_token_usage_accepts_bedrock_style_usage():
    normalized = normalize_token_usage(
        {
            "inputTokens": 14,
            "outputTokens": 6,
            "totalTokens": 20,
        }
    )

    assert normalized == {
        "input_tokens": 14,
        "output_tokens": 6,
        "total_tokens": 20,
    }

def test_normalize_token_usage_handles_missing_usage():
    normalized = normalize_token_usage({})

    assert normalized == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }

def test_parallel_inputs_node_only_runs_pre_hts_phases(monkeypatch):
    runner = MetalCompositionWorkflowRunner(settings=_sample_settings())
    seen: list[str] = []

    def fake_run_timed_phase(phase_name, state):
        seen.append(phase_name)
        return {
            "output": {f"{phase_name}_output": {"status": "completed"}},
            "timing": {"status": "completed", "duration_ms": 1.0},
        }

    monkeypatch.setattr(runner, "_run_timed_phase", fake_run_timed_phase)

    result = runner._parallel_inputs_node({})

    assert seen == ["diagram"]
    assert "hts_output" not in result
    assert "parallel_agents" in result["timing"]["phases"]

def test_legal_evidence_node_runs_hana_tree_phase(monkeypatch):
    runner = MetalCompositionWorkflowRunner(settings=_sample_settings())
    seen: list[str] = []

    def fake_run_timed_legal_phase(phase_name, state):
        seen.append(phase_name)
        return {
            "output": {
                "hana_tree_search_output": {
                    "status": "completed",
                    "timing": {"status": "completed", "duration_ms": 1.0},
                },
            },
            "timing": {"status": "completed", "duration_ms": 1.0},
        }

    monkeypatch.setattr(runner, "_run_timed_legal_phase", fake_run_timed_legal_phase)

    result = runner._legal_evidence_node({})

    assert seen == ["hana_tree_search"]
    assert result["hana_tree_search_output"]["status"] == "completed"
    assert "legal_evidence" in result["timing"]["phases"]

def test_hana_tree_search_routes_diffuser_disc_to_relevant_families(monkeypatch):
    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(hana_tree_router_model_name="hana-router"),
        hts_catalog_resolver=_sample_hts_catalog_resolver(),
    )
    calls: list[dict] = []

    def fake_invoke(**kwargs):
        calls.append(kwargs)
        user_prompt = kwargs["messages"][1]["content"]
        if '"chapter_groups"' in user_prompt:
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=(
                                '{"selected_headings":['
                                '{"chapter_number":73,"heading_code":"7307","rationale":"ASME flange/fitting interface is plausible.","confidence":0.71,"missing_discriminators":["Need flange drawing."]},'
                                '{"chapter_number":73,"heading_code":"7326","rationale":"Catch-all fabricated steel article remains viable.","confidence":0.69,"missing_discriminators":[]},'
                                '{"chapter_number":84,"heading_code":"8481","rationale":"Valve-part branch stays plausible with NBR-coated disc geometry.","confidence":0.52,"missing_discriminators":["Need valve assembly context."]}'
                                '],'
                                '"excluded_chapters":[{"chapter_number":85,"rationale":"No electrical function evidence."}],'
                                '"reasoning":"Three headings remain defensible after merged chapter+heading routing.",'
                                '"confidence":0.67,'
                                '"family_only_recommended":true}'
                            )
                        )
                    )
                ]
            )
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            '{"candidate_suggestions":['
                            '{"code":"7307.99","rationale":"Most specific fitting family supported by the ASME Class 150 interface clues.","confidence":0.66,"specificity_supported":6,"missing_discriminators":["Need flange dimensions."]},'
                            '{"code":"7326.90","rationale":"Fallback fabricated-steel family if fitting use is not confirmed.","confidence":0.58,"specificity_supported":6,"missing_discriminators":[]},'
                            '{"code":"8481.90","rationale":"Valve-part family remains contingent on assembly evidence.","confidence":0.43,"specificity_supported":6,"missing_discriminators":["Need BOM to prove valve part."]}'
                            '],'
                            '"reasoning":"The HANA tree search preserves the three strongest branches.",'
                            '"confidence":0.64}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    result = run_hana_tree_search(
        {
            "product_code": "4092301",
            "source_summary": {"part_description": "DIFFUSER DISC COATED", "new_part_description": 'DISC HEAD F 10"-150# X 20"BD SQ'},
            "source_row": {"Part description": "DIFFUSER DISC COATED", "New Part Description": 'DISC HEAD F 10"-150# X 20"BD SQ'},
            "hts_fact_profile": {
                "article_summary": "Industrial diffuser disc, predominantly steel, coated with NBR.",
                "function_summary": "Acts as a flow-conditioning disc in Class 150 piping service.",
                "material_profile": {
                    "top_level_grams": {"steel": 12964.6, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                    "steel_subtype_grams": {"hot_rolled_coil_steel": 12964.6},
                },
                "heading_hypotheses": ["7307", "7326", "8481"],
                "discriminator_notes": ["Need assembly context to distinguish fitting from valve part."],
            },
            "final_composition": {
                "top_level_grams": {"steel": 12964.6, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                "steel_subtype_grams": {"hot_rolled_coil_steel": 12964.6},
            },
        },
        runner.settings,
        runner.llm,
        _sample_hts_catalog_resolver(),
    )

    assert result["status"] == "completed"
    assert [item["hts_code"] for item in result["candidate_suggestions"]] == ["7307.99", "7326.90", "8481.90"]
    assert "8536.90" not in [item["hts_code"] for item in result["candidate_suggestions"]]
    assert all(call["reasoning_effort"] == "low" for call in calls)
    assert all("temperature" not in call for call in calls)

def test_hana_tree_search_expands_prefiltered_headings_without_heading_router(monkeypatch):
    """HANA tree search should call only the family router after heading expansion."""

    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(hana_tree_router_model_name="hana-router"),
        hts_catalog_resolver=_sample_hts_catalog_resolver(),
    )
    calls: list[dict] = []

    def fake_invoke(**kwargs):
        """Return family-router output and fail if a heading-router call appears."""

        calls.append(kwargs)
        assert kwargs["task"] == "family_router"
        user_prompt = kwargs["messages"][1]["content"]
        assert '"selected_headings"' not in user_prompt
        assert '"headings":' not in user_prompt
        assert '"family_options"' in user_prompt
        assert '"child_options"' in user_prompt
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            '{"candidate_suggestions":['
                            '{"code":"7307.99","rationale":"Pipe fitting family remains catalog-backed.","confidence":0.66,"specificity_supported":6,"missing_discriminators":[]},'
                            '{"code":"7326.90","rationale":"Fabricated article fallback remains catalog-backed.","confidence":0.58,"specificity_supported":6,"missing_discriminators":[]},'
                            '{"code":"8481.90","rationale":"Valve-part family remains catalog-backed.","confidence":0.43,"specificity_supported":6,"missing_discriminators":[]}'
                            '],'
                            '"reasoning":"Expanded from all deterministic heading options in the considered chapters.",'
                            '"confidence":0.64}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    result = run_hana_tree_search(
        {
            "product_code": "4092301",
            "source_summary": {
                "part_description": "DIFFUSER DISC COATED",
                "new_part_description": 'DISC HEAD F 10"-150# X 20"BD SQ',
                "priority_detail": "PUMP SPARES",
            },
            "source_row": {
                "Part description": "DIFFUSER DISC COATED",
                "New Part Description": 'DISC HEAD F 10"-150# X 20"BD SQ',
                "Priority.1": "PUMP SPARES",
            },
            "hts_fact_profile": {
                "article_summary": "Industrial diffuser disc, predominantly steel, coated with NBR.",
                "function_summary": "Acts as a flow-conditioning disc in Class 150 piping service.",
                "material_profile": {
                    "top_level_grams": {"steel": 12964.6, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                    "steel_subtype_grams": {"hot_rolled_coil_steel": 12964.6},
                },
                "heading_hypotheses": ["7307", "7326", "8481"],
                "discriminator_notes": ["Need assembly context to distinguish fitting from valve part."],
            },
            "final_composition": {
                "top_level_grams": {"steel": 12964.6, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                "steel_subtype_grams": {"hot_rolled_coil_steel": 12964.6},
            },
        },
        runner.settings,
        runner.llm,
        _sample_hts_catalog_resolver(),
    )

    assert [call["task"] for call in calls] == ["family_router"]
    assert "heading_router" not in result
    assert result["routing_diagnostics"]["selected_chapters"]
    assert result["routing_diagnostics"]["heading_options_considered_count"] >= 3
    assert result["routing_diagnostics"]["family_option_count"] >= 3
    assert result["routing_diagnostics"]["child_family_count"] >= 3
    assert result["routing_diagnostics"]["max_families_per_considered_heading"] == 2
    assert result["routing_diagnostics"]["max_children_per_explored_family"] == 3
    assert result["routing_diagnostics"]["family_prompt_bytes"] > 0
    assert {item["heading_code"] for item in result["family_options_sent"]} >= {"7307", "7326", "8481"}
    assert [item["hts_code"] for item in result["candidate_suggestions"]] == ["7307.99", "7326.90", "8481.90"]

def test_hana_tree_search_backfills_candidate_suggestions_to_configured_floor(monkeypatch):
    """HANA tree search should send at least k recalled codes to final HTS selection."""

    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(hana_tree_router_model_name="hana-router", hts_k_candidates=5),
        hts_catalog_resolver=_sample_hts_catalog_resolver(),
    )
    captured_prompt: dict = {}

    def fake_invoke(**kwargs):
        """Return one router candidate so deterministic recall backfill must fill the floor."""

        captured_prompt.update(json.loads(kwargs["messages"][1]["content"].split("Decision input:\n", 1)[1]))
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            '{"candidate_suggestions":['
                            '{"code":"7307.99","rationale":"Pipe fitting family remains catalog-backed.","confidence":0.66,"specificity_supported":6,"missing_discriminators":[]}'
                            '],'
                            '"reasoning":"Only one candidate was selected by the router.",'
                            '"confidence":0.66}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    result = run_hana_tree_search(
        {
            "product_code": "4092301",
            "source_summary": {
                "part_description": "DIFFUSER DISC COATED",
                "new_part_description": 'DISC HEAD F 10"-150# X 20"BD SQ',
            },
            "source_row": {
                "Part description": "DIFFUSER DISC COATED",
                "New Part Description": 'DISC HEAD F 10"-150# X 20"BD SQ',
            },
            "hts_fact_profile": {
                "article_summary": "Industrial diffuser disc, predominantly steel, coated with NBR.",
                "function_summary": "Acts as a flow-conditioning disc in Class 150 piping service.",
                "heading_hypotheses": ["7307", "7326", "8481"],
            },
            "final_composition": {
                "top_level_grams": {"steel": 12964.6},
                "steel_subtype_grams": {"hot_rolled_coil_steel": 12964.6},
            },
        },
        runner.settings,
        runner.llm,
        _sample_hts_catalog_resolver(),
    )

    target_count = min(5, len(result["recall_candidates"]))
    suggestion_codes = [item["hts_code"] for item in result["candidate_suggestions"]]
    assert result["family_router"]["status"] == "completed"
    assert captured_prompt["candidate_suggestion_target"] == target_count
    assert len(suggestion_codes) == target_count
    assert suggestion_codes[0] == "7307.99"
    assert len(suggestion_codes) == len(set(suggestion_codes))
    assert result["routing_diagnostics"]["candidate_suggestion_target"] == target_count
    assert result["routing_diagnostics"]["candidate_suggestion_floor_added_count"] == target_count - 1
    assert all("backfill" not in item.get("rationale", "").lower() for item in result["candidate_suggestions"])
    assert all("comparison coverage" not in item.get("rationale", "").lower() for item in result["candidate_suggestions"])

def test_hana_tree_search_packets_family_router_when_prompt_exceeds_budget(monkeypatch):
    """Large same-level recall payloads should be routed through multiple family-router calls."""

    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(hana_tree_router_model_name="hana-router"),
        hts_catalog_resolver=_sample_hts_catalog_resolver(),
    )
    calls: list[dict] = []
    monkeypatch.setattr(hana_tree_search_module, "_MAX_FAMILY_ROUTER_PROMPT_BYTES", 1)

    def fake_invoke(**kwargs):
        """Return the first code from each family-router packet."""

        calls.append(kwargs)
        prompt_payload = json.loads(kwargs["messages"][1]["content"].split("Decision input:\n", 1)[1])
        packet_codes = [item["code"] for item in prompt_payload.get("family_options", [])]
        for child_options in prompt_payload.get("child_options", {}).values():
            packet_codes.extend(item["code"] for item in child_options)
        selected_code = packet_codes[0]
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(
                            {
                                "candidate_suggestions": [
                                    {
                                        "code": selected_code,
                                        "rationale": "Packet-local best catalog option.",
                                        "confidence": 0.61,
                                        "specificity_supported": 6,
                                        "missing_discriminators": [],
                                    }
                                ],
                                "reasoning": "Packet-local family routing.",
                                "confidence": 0.61,
                            }
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    result = run_hana_tree_search(
        {
            "product_code": "4092301",
            "source_summary": {
                "part_description": "DIFFUSER DISC COATED",
                "new_part_description": 'DISC HEAD F 10"-150# X 20"BD SQ',
                "priority_detail": "PUMP SPARES",
            },
            "source_row": {
                "Part description": "DIFFUSER DISC COATED",
                "New Part Description": 'DISC HEAD F 10"-150# X 20"BD SQ',
                "Priority.1": "PUMP SPARES",
            },
            "hts_fact_profile": {
                "article_summary": "Industrial diffuser disc, predominantly steel, coated with NBR.",
                "function_summary": "Acts as a flow-conditioning disc in Class 150 piping service.",
                "material_profile": {
                    "top_level_grams": {"steel": 12964.6, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                    "steel_subtype_grams": {"hot_rolled_coil_steel": 12964.6},
                },
                "heading_hypotheses": ["7307", "7326", "8481"],
                "discriminator_notes": ["Need assembly context to distinguish fitting from valve part."],
            },
            "final_composition": {
                "top_level_grams": {"steel": 12964.6, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                "steel_subtype_grams": {"hot_rolled_coil_steel": 12964.6},
            },
        },
        runner.settings,
        runner.llm,
        _sample_hts_catalog_resolver(),
    )

    assert len(calls) > 1
    assert all(call["task"] == "family_router" for call in calls)
    assert result["family_router"]["packeted"] is True
    assert result["family_router"]["packet_count"] == len(calls)
    assert result["routing_diagnostics"]["family_router_packet_count"] == len(calls)
    assert result["routing_diagnostics"]["family_router_max_prompt_bytes"] > 0
    assert result["candidate_suggestions"]

def test_hana_tree_search_directive_headings_select_chapters_without_zero_score_padding(monkeypatch):
    """Tree-search directives should select their chapters without adding zero-score filler chapters."""

    class _DirectiveHeadingResolver:
        def __init__(self) -> None:
            self.heading_chapter_calls: list[list[int]] = []

        def list_chapter_options(self, context):
            return [
                {"chapter_number": 84, "title": "Machinery", "summary": "Pump parts.", "prefilter_score": 3.0},
                {"chapter_number": 1, "title": "Live animals", "summary": "Animals.", "prefilter_score": 0.0},
                {"chapter_number": 2, "title": "Meat", "summary": "Meat.", "prefilter_score": 0.0},
                {"chapter_number": 39, "title": "Plastics", "summary": "Plastic articles.", "prefilter_score": 0.0},
            ]

        def list_heading_options(self, chapters, context, *, per_chapter=25):
            self.heading_chapter_calls.append(list(chapters))
            options = []
            if 84 in chapters:
                options.append(
                    {
                        "chapter_number": 84,
                        "heading_code": "8413",
                        "description": "Pumps for liquids; parts thereof",
                        "path_description": "Pumps for liquids; parts thereof",
                        "prefilter_score": 6.0,
                        "matched_terms": ["impeller"],
                        "matched_phrases": [],
                    }
                )
            if 39 in chapters:
                options.append(
                    {
                        "chapter_number": 39,
                        "heading_code": "3917",
                        "description": "Plastic tubes and fittings",
                        "path_description": "Plastic tubes and fittings",
                        "prefilter_score": 0.0,
                        "matched_terms": [],
                        "matched_phrases": [],
                    }
                )
            return options

        def list_family_options(self, headings, context, *, per_heading=5):
            rows = []
            if "8413" in headings:
                rows.append(
                    {
                        "code": "8413.91",
                        "digits": 6,
                        "chapter_number": 84,
                        "heading_code": "8413",
                        "description": "Parts",
                        "path_description": "Pumps for liquids; parts thereof > Parts > Of pumps",
                        "prefilter_score": 6.0,
                        "matched_terms": ["impeller"],
                        "matched_phrases": [],
                    }
                )
            if "3917" in headings:
                rows.append(
                    {
                        "code": "3917.40",
                        "digits": 6,
                        "chapter_number": 39,
                        "heading_code": "3917",
                        "description": "Fittings",
                        "path_description": "Plastic tubes and fittings > Fittings",
                        "prefilter_score": 0.0,
                        "matched_terms": [],
                        "matched_phrases": [],
                    }
                )
            return rows

        def expand_children(self, families, context, *, per_family=5):
            return {}

    resolver = _DirectiveHeadingResolver()
    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(hana_tree_router_model_name="hana-router"))

    def fake_invoke(**kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='{"candidate_suggestions":[{"code":"8413.91","rationale":"Directive-backed pump parts family.","confidence":0.7,"specificity_supported":6,"missing_discriminators":[]}],"reasoning":"ok","confidence":0.7}'
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    result = run_hana_tree_search(
        {
            "product_code": "impeller-no-heading-hypothesis",
            "source_summary": {"part_description": "IMPELLER"},
            "source_row": {"Part description": "IMPELLER"},
            "hts_fact_profile": {
                "article_summary": "Impeller.",
                "function_summary": "Pump component.",
                "heading_hypotheses": [],
                "tree_search_directives": [
                    {
                        "heading": "8413",
                        "branch_focus": ["parts"],
                        "include_residual_children": True,
                        "rationale": "Explore pump parts.",
                        "confidence": 0.85,
                    },
                    {
                        "heading": "3917",
                        "branch_focus": ["material_articles"],
                        "include_residual_children": False,
                        "rationale": "Explore plastic alternatives.",
                        "confidence": 0.35,
                    },
                ],
            },
            "final_composition": {"top_level_grams": {}, "steel_subtype_grams": {}},
        },
        runner.settings,
        runner.llm,
        resolver,
    )

    assert result["routing_diagnostics"]["selected_chapters"] == [84, 39]
    assert resolver.heading_chapter_calls == [[84, 39]]
    assert 1 not in result["routing_diagnostics"]["selected_chapters"]
    assert 2 not in result["routing_diagnostics"]["selected_chapters"]

def test_hana_tree_search_partial_packet_failure_uses_candidate_floor_without_fallback_marker(monkeypatch):
    """A partial packet failure should use the candidate floor without failed-packet fallback markers."""

    class _PartialPacketResolver:
        def list_chapter_options(self, context):
            return [
                {"chapter_number": 84, "title": "Machinery", "summary": "Pump parts.", "prefilter_score": 3.0},
                {"chapter_number": 4, "title": "Dairy", "summary": "Dairy products.", "prefilter_score": 0.0},
            ]

        def list_heading_options(self, chapters, context, *, per_chapter=25):
            options = []
            if 84 in chapters:
                options.append(
                    {
                        "chapter_number": 84,
                        "heading_code": "8413",
                        "description": "Pumps for liquids; parts thereof",
                        "path_description": "Pumps for liquids; parts thereof",
                        "prefilter_score": 6.0,
                        "matched_terms": ["impeller"],
                        "matched_phrases": [],
                    }
                )
            if 4 in chapters:
                options.append(
                    {
                        "chapter_number": 4,
                        "heading_code": "0404",
                        "description": "Whey",
                        "path_description": "Whey",
                        "prefilter_score": 1.35,
                        "matched_terms": ["not"],
                        "matched_phrases": [],
                    }
                )
            return options

        def list_family_options(self, headings, context, *, per_heading=5):
            rows = []
            if "8413" in headings:
                rows.append(
                    {
                        "code": "8413.91",
                        "digits": 6,
                        "chapter_number": 84,
                        "heading_code": "8413",
                        "description": "Parts",
                        "path_description": "Pumps for liquids; parts thereof > Parts > Of pumps",
                        "prefilter_score": 6.0,
                        "matched_terms": ["impeller"],
                        "matched_phrases": [],
                    }
                )
            if "0404" in headings:
                rows.append(
                    {
                        "code": "0404.90",
                        "digits": 6,
                        "chapter_number": 4,
                        "heading_code": "0404",
                        "description": "Other",
                        "path_description": "Whey > Other",
                        "prefilter_score": 1.35,
                        "matched_terms": ["not"],
                        "matched_phrases": [],
                    }
                )
            return rows

        def expand_children(self, families, context, *, per_family=5):
            if "8413.91" not in families:
                return {}
            return {
                "8413.91": [
                    {
                        "code": "8413.91.9096",
                        "digits": 10,
                        "chapter_number": 84,
                        "heading_code": "8413",
                        "family_code": "8413.91",
                        "description": "Other",
                        "path_description": "Pumps for liquids; parts thereof > Parts > Of pumps > Other > Other",
                        "prefilter_score": 5.7,
                        "matched_terms": ["pump"],
                        "matched_phrases": [],
                    }
                ]
            }

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(hana_tree_router_model_name="hana-router"))
    calls: list[int] = []
    monkeypatch.setattr(hana_tree_search_module, "_MAX_FAMILY_ROUTER_PROMPT_BYTES", 1)

    def fake_invoke(**kwargs):
        prompt_payload = json.loads(kwargs["messages"][1]["content"].split("Decision input:\n", 1)[1])
        chapter_number = prompt_payload["considered_chapters"][0]["chapter_number"]
        calls.append(chapter_number)
        if chapter_number == 4:
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='{"candidate_suggestions":[],"reasoning":"No dairy code applies.","confidence":0.0}'
                        )
                    )
                ]
            )
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='{"candidate_suggestions":[{"code":"8413.91.9096","rationale":"Pump parts residual.","confidence":0.92,"specificity_supported":10,"missing_discriminators":[]}],"reasoning":"ok","confidence":0.92}'
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    result = run_hana_tree_search(
        {
            "product_code": "impeller-partial-packet",
            "source_summary": {"part_description": "IMPELLER"},
            "source_row": {"Part description": "IMPELLER"},
            "hts_fact_profile": {
                "article_summary": "Impeller.",
                "function_summary": "Pump component.",
                "heading_hypotheses": ["8413", "0404"],
            },
            "final_composition": {"top_level_grams": {}, "steel_subtype_grams": {}},
        },
        runner.settings,
        runner.llm,
        _PartialPacketResolver(),
    )

    assert calls == [84, 4]
    assert result["family_router"]["status"] == "partial"
    assert result["family_router"]["fallback_used"] is False
    assert "family_router[chapter-4" in result["family_router"]["error"]
    assert [item["hts_code"] for item in result["candidate_suggestions"]] == [
        "8413.91.9096",
        "8413.91",
        "0404.90",
    ]
    assert result["routing_diagnostics"]["candidate_suggestion_floor_added_count"] == 2
    assert all(
        "fallback retained" not in item.get("rationale", "").lower()
        for item in result["candidate_suggestions"]
    )
    failed_call = [call for call in result["family_router"]["router_calls"] if call["chapter_number"] == 4][0]
    assert failed_call["candidate_count"] == 0
    assert failed_call["fallback_candidate_count"] == 1
    assert failed_call["fallback_used"] is False

def test_hana_tree_search_includes_all_families_for_small_heading(monkeypatch):
    """Small HTS headings should not be pruned to only the top lexical families."""

    class _SmallHeadingResolver:
        def list_chapter_options(self, context):
            return [{"chapter_number": 84, "title": "Machinery", "summary": "Pump headings.", "prefilter_score": 10.0}]

        def list_heading_options(self, chapters, context, *, per_chapter=25):
            return [
                {
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Pumps for liquids; parts thereof",
                    "path_description": "Pumps for liquids; parts thereof",
                    "prefilter_score": 10.0,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                }
            ]

        def list_family_options(self, headings, context, *, per_heading=5):
            codes = [
                "8413.92",
                "8413.91",
                "8413.82",
                "8413.81",
                "8413.70",
                "8413.60",
                "8413.50",
                "8413.40",
                "8413.30",
                "8413.20",
                "8413.19",
                "8413.11",
            ]
            options = [
                {
                    "code": code,
                    "digits": 6,
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Parts" if code == "8413.91" else "Pump family",
                    "path_description": (
                        "Pumps for liquids; parts thereof > Parts: > Of pumps:"
                        if code == "8413.91"
                        else f"Pumps for liquids; parts thereof > Family {code}"
                    ),
                    "prefilter_score": 10.0,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                }
                for code in codes
            ]
            return options[:per_heading]

        def expand_children(self, families, context, *, per_family=3):
            return {}

    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(hana_tree_router_model_name="hana-router"),
    )

    def fake_invoke(**kwargs):
        prompt_payload = json.loads(kwargs["messages"][1]["content"].split("Decision input:\n", 1)[1])
        code = prompt_payload["family_options"][0]["code"]
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(
                            {
                                "candidate_suggestions": [
                                    {
                                        "code": code,
                                        "rationale": "First recalled family retained for test.",
                                        "confidence": 0.5,
                                        "specificity_supported": 6,
                                        "missing_discriminators": [],
                                    }
                                ],
                                "reasoning": "Test router response.",
                                "confidence": 0.5,
                            }
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    result = run_hana_tree_search(
        {
            "product_code": "gcc:2253",
            "source_summary": {
                "part_description": "Broker Report Additions (CHR)",
                "new_part_description": "IMPELLER ECEU 00000",
                "priority_detail": "CSTG IMPLR 14DX",
            },
            "source_row": {
                "Part description": "Broker Report Additions (CHR)",
                "New Part Description": "IMPELLER ECEU 00000",
                "Priority.1": "CSTG IMPLR 14DX",
            },
            "hts_fact_profile": {
                "article_summary": "Cast iron impeller",
                "function_summary": "Possible pump component",
                "material_profile": {
                    "top_level_grams": {"cast_iron": 7520.0, "steel": 0.0, "aluminum": 0.0, "copper": 0.0},
                    "steel_subtype_grams": {},
                },
                "heading_hypotheses": ["8413"],
            },
            "final_composition": {
                "top_level_grams": {"cast_iron": 7520.0, "steel": 0.0, "aluminum": 0.0, "copper": 0.0},
                "steel_subtype_grams": {},
            },
        },
        runner.settings,
        runner.llm,
        _SmallHeadingResolver(),
    )

    recalled_8413_families = {
        item["code"]
        for item in result["family_options_sent"]
        if item.get("heading_code") == "8413"
    }

    assert "8413.91" in recalled_8413_families
    assert "8413.20" in recalled_8413_families
    assert len(recalled_8413_families) >= 12

def test_hana_tree_search_directive_keeps_pump_parts_residual_child(monkeypatch):
    """A parts directive should keep residual Other > Other children below explored parts branches."""

    class _ResidualChildResolver:
        def list_chapter_options(self, context):
            return [{"chapter_number": 84, "title": "Machinery", "summary": "Pump headings.", "prefilter_score": 10.0}]

        def list_heading_options(self, chapters, context, *, per_chapter=25):
            return [
                {
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Pumps for liquids; parts thereof",
                    "path_description": "Pumps for liquids; parts thereof",
                    "prefilter_score": 10.0,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                }
            ]

        def list_family_options(self, headings, context, *, per_heading=5):
            options = [
                {
                    "code": "8413.91",
                    "digits": 6,
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Parts",
                    "path_description": "Pumps for liquids; parts thereof > Parts: > Of pumps:",
                    "prefilter_score": 9.0,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                }
            ]
            return options[:per_heading]

        def expand_children(self, families, context, *, per_family=3):
            children = [
                {
                    "code": "8413.91.1000",
                    "digits": 10,
                    "family_code": "8413.91",
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Fuel-injection pump parts",
                    "path_description": "Pumps for liquids; parts thereof > Parts: > Of pumps: > Fuel-injection pump parts",
                    "prefilter_score": 9.5,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                },
                {
                    "code": "8413.91.2000",
                    "digits": 10,
                    "family_code": "8413.91",
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Stock pump parts",
                    "path_description": "Pumps for liquids; parts thereof > Parts: > Of pumps: > Stock pump parts",
                    "prefilter_score": 9.4,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                },
                {
                    "code": "8413.91.9060",
                    "digits": 10,
                    "family_code": "8413.91",
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Hydraulic fluid power pump parts",
                    "path_description": "Pumps for liquids; parts thereof > Parts: > Of pumps: > Other > Of hydraulic fluid power pumps",
                    "prefilter_score": 9.3,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                },
                {
                    "code": "8413.91.9096",
                    "digits": 10,
                    "family_code": "8413.91",
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Other",
                    "path_description": "Pumps for liquids; parts thereof > Parts: > Of pumps: > Other > Other",
                    "prefilter_score": 1.0,
                    "matched_terms": [],
                    "matched_phrases": [],
                },
            ]
            return {"8413.91": children[:per_family]}

    def fake_invoke(**kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='{"candidate_suggestions":[{"code":"8413.91","rationale":"Parts family recalled.","confidence":0.5,"specificity_supported":6,"missing_discriminators":[]}],"reasoning":"ok","confidence":0.5}'
                    )
                )
            ]
        )

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(hana_tree_router_model_name="hana-router"))
    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    result = run_hana_tree_search(
        {
            "product_code": "gcc:2253",
            "source_summary": {"new_part_description": "IMPELLER ECEU 00000", "priority_detail": "CSTG IMPLR 14DX"},
            "source_row": {"New Part Description": "IMPELLER ECEU 00000", "Priority.1": "CSTG IMPLR 14DX"},
            "hts_fact_profile": {
                "article_summary": "Cast iron impeller",
                "function_summary": "Possible pump component",
                "heading_hypotheses": ["8413"],
                "tree_search_directives": [
                    {
                        "heading": "8413",
                        "branch_focus": ["parts"],
                        "include_residual_children": True,
                        "rationale": "Explore pump parts.",
                        "confidence": 0.72,
                    }
                ],
            },
            "final_composition": {"top_level_grams": {"cast_iron": 7520.0}},
        },
        runner.settings,
        runner.llm,
        _ResidualChildResolver(),
    )

    child_codes = {item["code"] for item in result["child_options_sent"]["8413.91"]}
    recall_codes = {item["code"] for item in result["recall_candidates"]}

    assert "8413.91.9096" in child_codes
    assert "8413.91.9096" in recall_codes

def test_hana_tree_search_router_failure_falls_back_to_recall_order(monkeypatch):
    """Router failure should not let numeric HTS order outrank selected heading recall order."""

    class _FallbackOrderingResolver:
        def list_chapter_options(self, context):
            return [
                {"chapter_number": 84, "title": "Machinery", "summary": "Pump headings.", "prefilter_score": 10.0},
                {"chapter_number": 72, "title": "Iron and steel", "summary": "Primary material headings.", "prefilter_score": 8.0},
            ]

        def list_heading_options(self, chapters, context, *, per_chapter=25):
            return [
                {
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Pumps for liquids; parts thereof",
                    "path_description": "Pumps for liquids; parts thereof",
                    "prefilter_score": 10.0,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                },
                {
                    "chapter_number": 72,
                    "heading_code": "7203",
                    "description": "Ferrous products obtained by direct reduction",
                    "path_description": "Ferrous products obtained by direct reduction",
                    "prefilter_score": 8.0,
                    "matched_terms": ["iron"],
                    "matched_phrases": [],
                },
            ]

        def list_family_options(self, headings, context, *, per_heading=5):
            options = [
                {
                    "code": "7203.90",
                    "digits": 6,
                    "chapter_number": 72,
                    "heading_code": "7203",
                    "description": "Other",
                    "path_description": "Ferrous products obtained by direct reduction > Other",
                    "prefilter_score": 8.0,
                    "matched_terms": ["iron"],
                    "matched_phrases": [],
                },
                {
                    "code": "8413.91",
                    "digits": 6,
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Parts",
                    "path_description": "Pumps for liquids; parts thereof > Parts: > Of pumps:",
                    "prefilter_score": 10.0,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                },
            ]
            return options[:per_heading]

        def expand_children(self, families, context, *, per_family=3):
            return {}

    def failing_router(**kwargs):
        raise RuntimeError("family router exploded")

    runner = MetalCompositionWorkflowRunner(settings=_sample_settings(hana_tree_router_model_name="hana-router"))
    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", failing_router)

    result = run_hana_tree_search(
        {
            "product_code": "gcc:2253",
            "source_summary": {"new_part_description": "IMPELLER ECEU 00000", "priority_detail": "CSTG IMPLR 14DX"},
            "source_row": {"New Part Description": "IMPELLER ECEU 00000", "Priority.1": "CSTG IMPLR 14DX"},
            "hts_fact_profile": {
                "article_summary": "Cast iron impeller",
                "function_summary": "Possible pump component",
                "heading_hypotheses": ["8413"],
                "tree_search_directives": [
                    {
                        "heading": "8413",
                        "branch_focus": ["parts"],
                        "include_residual_children": True,
                        "rationale": "Explore pump parts.",
                        "confidence": 0.72,
                    }
                ],
            },
            "final_composition": {"top_level_grams": {"cast_iron": 7520.0}},
        },
        runner.settings,
        runner.llm,
        _FallbackOrderingResolver(),
    )

    assert result["family_router"]["status"] == "failed"
    assert result["candidate_suggestions"][0]["hts_code"].startswith("8413")
    assert result["candidate_suggestions"][0]["hts_code"] != "7203.90"
    assert all(item["confidence"] <= 0.4 for item in result["candidate_suggestions"])

def test_hana_tree_search_gcc_2253_like_sparse_fields_recalls_8413919096(monkeypatch):
    """Sparse impeller fields should make 8413.91.9096 available for selection."""

    class _ImpellerResolver:
        def list_chapter_options(self, context):
            return [{"chapter_number": 84, "title": "Machinery", "summary": "Pump headings.", "prefilter_score": 10.0}]

        def list_heading_options(self, chapters, context, *, per_chapter=25):
            return [
                {
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Pumps for liquids; parts thereof",
                    "path_description": "Pumps for liquids; parts thereof",
                    "prefilter_score": 10.0,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                }
            ]

        def list_family_options(self, headings, context, *, per_heading=5):
            options = [
                {
                    "code": "8413.70",
                    "digits": 6,
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Other centrifugal pumps",
                    "path_description": "Pumps for liquids; parts thereof > Other centrifugal pumps",
                    "prefilter_score": 10.0,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                },
                {
                    "code": "8413.91",
                    "digits": 6,
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Parts",
                    "path_description": "Pumps for liquids; parts thereof > Parts: > Of pumps:",
                    "prefilter_score": 8.0,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                },
            ]
            return options[:per_heading]

        def expand_children(self, families, context, *, per_family=3):
            if "8413.91" not in set(families):
                return {}
            children = [
                {
                    "code": "8413.91.9060",
                    "digits": 10,
                    "family_code": "8413.91",
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Hydraulic fluid power pump parts",
                    "path_description": "Pumps for liquids; parts thereof > Parts: > Of pumps: > Other > Of hydraulic fluid power pumps",
                    "prefilter_score": 9.0,
                    "matched_terms": ["pump"],
                    "matched_phrases": [],
                },
                {
                    "code": "8413.91.9096",
                    "digits": 10,
                    "family_code": "8413.91",
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Other",
                    "path_description": "Pumps for liquids; parts thereof > Parts: > Of pumps: > Other > Other",
                    "prefilter_score": 1.0,
                    "matched_terms": [],
                    "matched_phrases": [],
                },
            ]
            return {"8413.91": children[:per_family]}

    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(hana_tree_router_model_name="hana-router"),
    )

    def fake_invoke(**kwargs):
        prompt_payload = json.loads(kwargs["messages"][1]["content"].split("Decision input:\n", 1)[1])
        code = prompt_payload["family_options"][0]["code"]
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(
                            {
                                "candidate_suggestions": [
                                    {
                                        "code": code,
                                        "rationale": "Recall-only regression does not depend on router choice.",
                                        "confidence": 0.5,
                                        "specificity_supported": 6,
                                        "missing_discriminators": [],
                                    }
                                ],
                                "reasoning": "Test router response.",
                                "confidence": 0.5,
                            }
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    result = run_hana_tree_search(
        {
            "product_code": "gcc:2253",
            "source_summary": {
                "part_description": "Broker Report Additions (CHR)",
                "new_part_description": "IMPELLER ECEU 00000",
                "priority_detail": "CSTG IMPLR 14DX",
            },
            "source_row": {
                "Part description": "Broker Report Additions (CHR)",
                "New Part Description": "IMPELLER ECEU 00000",
                "Priority.1": "CSTG IMPLR 14DX",
            },
            "hts_fact_profile": {
                "article_summary": "Cast iron impeller",
                "function_summary": "Possible pump component",
                "material_profile": {
                    "top_level_grams": {"cast_iron": 7520.0, "steel": 0.0, "aluminum": 0.0, "copper": 0.0},
                    "steel_subtype_grams": {},
                },
                "heading_hypotheses": ["8413"],
                "tree_search_directives": [
                    {
                        "heading": "8413",
                        "branch_focus": ["parts"],
                        "include_residual_children": True,
                        "rationale": "Impeller appears to be a component; explore pump-parts branches.",
                        "confidence": 0.72,
                    }
                ],
            },
            "final_composition": {
                "top_level_grams": {"cast_iron": 7520.0, "steel": 0.0, "aluminum": 0.0, "copper": 0.0},
                "steel_subtype_grams": {},
            },
        },
        runner.settings,
        runner.llm,
        _ImpellerResolver(),
    )

    recall_codes = {item["code"] for item in result["recall_candidates"]}
    child_codes = {
        item["code"]
        for children in result["child_options_sent"].values()
        for item in children
    }

    assert "8413.91.9096" in recall_codes
    assert "8413.91.9096" in child_codes

def test_hana_tree_search_family_router_fallback_keeps_broad_families(monkeypatch):
    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(hana_tree_router_model_name="hana-router"),
        hts_catalog_resolver=_sample_hts_catalog_resolver(),
    )
    call_count = {"count": 0}

    def fake_invoke(**kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='{"selected_headings":[{"chapter_number":73,"heading_code":"7307","rationale":"fitting path","confidence":0.7,"missing_discriminators":[]},{"chapter_number":73,"heading_code":"7326","rationale":"fabricated article path","confidence":0.6,"missing_discriminators":[]},{"chapter_number":84,"heading_code":"8481","rationale":"valve-part path","confidence":0.5,"missing_discriminators":[]}],"excluded_chapters":[],"reasoning":"Three headings remain.","confidence":0.66,"family_only_recommended":true}'
                        )
                    )
                ]
            )
        raise RuntimeError("family router exploded")

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    result = run_hana_tree_search(
        {
            "source_summary": {"part_description": "DIFFUSER DISC COATED", "new_part_description": 'DISC HEAD F 10"-150# X 20"BD SQ'},
            "source_row": {"Part description": "DIFFUSER DISC COATED", "New Part Description": 'DISC HEAD F 10"-150# X 20"BD SQ'},
            "hts_fact_profile": {
                "article_summary": "Industrial diffuser disc, predominantly steel, coated with NBR.",
                "function_summary": "Acts as a flow-conditioning disc in Class 150 piping service.",
                "material_profile": {
                    "top_level_grams": {"steel": 12964.6},
                    "steel_subtype_grams": {"hot_rolled_coil_steel": 12964.6},
                },
                "heading_hypotheses": ["7307", "7326", "8481"],
            },
            "final_composition": {
                "top_level_grams": {"steel": 12964.6},
                "steel_subtype_grams": {"hot_rolled_coil_steel": 12964.6},
            },
        },
        runner.settings,
        runner.llm,
        _sample_hts_catalog_resolver(),
    )

    assert result["family_router"]["status"] == "failed"
    assert result["family_router"]["fallback_used"] is True
    assert {item["hts_code"] for item in result["candidate_suggestions"]} == {"7307.99", "7326.90", "8481.90"}
    assert all(len(item["hts_code"].replace(".", "")) == 6 for item in result["candidate_suggestions"])

def test_hana_tree_search_router_failure_constrains_oil_housing_fallback_to_likely_chapters(monkeypatch):
    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(hana_tree_router_model_name="hana-router"),
        hts_catalog_resolver=_sample_hts_catalog_resolver(),
    )

    class _ConstrainedFallbackResolver:
        def __init__(self) -> None:
            self.heading_calls: list[dict] = []
            self.family_headings: list[str] = []
            self.heading_catalog = [
                {
                    "chapter_number": 1,
                    "heading_code": "0101",
                    "description": "Live horses, asses, mules and hinnies",
                    "path_description": "Live horses, asses, mules and hinnies",
                    "prefilter_score": 99.0,
                    "matched_terms": ["chapter"],
                    "matched_phrases": [],
                },
                {
                    "chapter_number": 84,
                    "heading_code": "8483",
                    "description": "Bearing housings; gearing and transmission parts",
                    "path_description": "Bearing housings; gearing and transmission parts",
                    "prefilter_score": 7.4,
                    "matched_terms": ["housing"],
                    "matched_phrases": [],
                },
                {
                    "chapter_number": 84,
                    "heading_code": "8413",
                    "description": "Pumps for liquids; parts thereof",
                    "path_description": "Pumps for liquids; parts thereof",
                    "prefilter_score": 9.6,
                    "matched_terms": ["pump", "oil", "housing"],
                    "matched_phrases": ["oil housing"],
                },
                {
                    "chapter_number": 73,
                    "heading_code": "7325",
                    "description": "Other cast articles of iron or steel",
                    "path_description": "Other cast articles of iron or steel",
                    "prefilter_score": 8.1,
                    "matched_terms": ["cast", "iron", "housing"],
                    "matched_phrases": [],
                },
                {
                    "chapter_number": 40,
                    "heading_code": "4016",
                    "description": "Other articles of vulcanized rubber",
                    "path_description": "Other articles of vulcanized rubber",
                    "prefilter_score": 3.2,
                    "matched_terms": ["coated"],
                    "matched_phrases": [],
                },
            ]

        def list_chapter_options(self, context):
            return [
                {"chapter_number": 1, "title": "Live animals", "summary": "Animal headings.", "prefilter_score": 99.0},
                {"chapter_number": 84, "title": "Machinery", "summary": "Pump and bearing headings.", "prefilter_score": 7.0},
                {"chapter_number": 73, "title": "Articles of iron or steel", "summary": "Cast article headings.", "prefilter_score": 6.5},
                {"chapter_number": 40, "title": "Rubber", "summary": "Rubber article headings.", "prefilter_score": 2.0},
                {"chapter_number": 99, "title": "Temporary legislation", "summary": "Chapter 99 headings.", "prefilter_score": 1.0},
            ]

        def list_heading_options(self, chapters, context, *, per_chapter=25):
            chapter_numbers = [int(chapter) for chapter in chapters]
            self.heading_calls.append({"chapters": chapter_numbers, "per_chapter": per_chapter})
            return [
                item
                for item in self.heading_catalog
                if int(item["chapter_number"]) in set(chapter_numbers)
            ]

        def list_family_options(self, headings, context, *, per_heading=5):
            self.family_headings = [str(heading) for heading in headings]
            return []

        def expand_children(self, families, context, *, per_family=3):
            return {}

    resolver = _ConstrainedFallbackResolver()

    def fake_invoke(**kwargs):
        raise RuntimeError("context_length_exceeded")

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    hana_result = run_hana_tree_search(
        {
            "product_code": "3365710",
            "source_summary": {
                "part_description": "OIL HOUSING COATED",
                "new_part_description": "OIL HOUSING COATED ECEU 00000",
            },
            "source_row": {
                "Part description": "Oil housing coated",
                "New Part Description": "Oil housing coated ECEU 00000",
            },
            "hts_fact_profile": {
                "article_summary": "Cast-iron pump oil housing with coating.",
                "function_summary": "Housing used as a pump spare part.",
                "material_profile": {
                    "top_level_grams": {"cast_iron": 1400.0, "steel": 0.0, "aluminum": 0.0, "copper": 0.0},
                    "steel_subtype_grams": {},
                },
                "heading_hypotheses": ["8413", "8483", "7325", "4016"],
            },
            "final_composition": {
                "top_level_grams": {"cast_iron": 1400.0, "steel": 0.0, "aluminum": 0.0, "copper": 0.0},
                "steel_subtype_grams": {},
            },
        },
        runner.settings,
        runner.llm,
        resolver,
    )

    assert resolver.heading_calls == [{"chapters": [84, 73, 40], "per_chapter": 10}]
    assert set(resolver.family_headings) == {"8413", "7325", "8483", "4016"}
    assert hana_result["routing_diagnostics"]["selected_chapters"] == [84, 73, 40]
    assert hana_result["routing_diagnostics"]["chapter_options_sent_count"] == 3
    assert hana_result["routing_diagnostics"]["heading_options_sent_count"] == 4
    assert hana_result["routing_diagnostics"]["heading_options_considered_count"] == 4
    assert "heading_router" not in hana_result
    assert {item["heading_code"] for item in hana_result["family_options_sent"]} == set()
    assert "0101" not in resolver.family_headings
    assert hana_result["candidate_suggestions"] == []

    trade_result, details = run_trade_decision(
        {
            "product_code": "3365710",
            "source_summary": {"new_part_description": "OIL HOUSING COATED ECEU 00000"},
            "source_row": {"Part description": "Oil housing coated"},
            "hana_tree_search_output": hana_result,
        },
        runner.settings,
        runner.llm,
        _sample_hts_catalog_resolver(),
    )

    assert trade_result["hts_classification"]["status"] == "unavailable"
    assert trade_result["hts_classification"]["best_candidate"] is None
    assert trade_result["hts_resolution_output"]["selected_code"] == ""
    assert details["candidate_count"] == 0

def test_trade_decision_prompt_includes_hana_tree_search_and_hana_only_candidates(monkeypatch):
    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(trade_decision_model_name="trade-model"),
        hts_catalog_resolver=_sample_hts_catalog_resolver(),
    )
    captured: dict = {}

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            '{"selected_code":"7326.90",'
                            '"reasoning":"The HANA tree search keeps 7326.90 as the broadest defensible family when official search is absent.",'
                            '"confidence":0.62,'
                            '"needs_human_review":true,'
                            '"candidate_rankings":['
                            '{"code":"7326.90","rationale":"Broad fabricated-steel family preserved by HANA tree routing.","confidence":0.62,"missing_discriminators":["Need assembly context for narrower classification."]},'
                            '{"code":"7307.99","rationale":"Pipe-fitting path remains possible but unproven.","confidence":0.49,"missing_discriminators":["Need flange/fitting evidence."]}'
                            ']}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    trade_result, details = run_trade_decision(
        {
            "hts_fact_profile": {
                "material_profile": {
                    "top_level_grams": {"steel": 12964.6, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                    "confidence": 0.84,
                }
            },
            "hana_tree_search_output": {
                "status": "completed",
                "context": {"article_summary": "Diffuser disc", "function_summary": "Flow-conditioning disc"},
                "chapter_router": {"status": "completed", "selected_chapters": [{"chapter_number": 73}], "reasoning": "Steel article chapter."},
                "heading_router": {
                    "status": "completed",
                    "selected_headings": [{"heading_code": "7307"}, {"heading_code": "7326"}],
                    "reasoning": "Two chapter 73 branches remain.",
                },
                "family_router": {
                    "status": "completed",
                    "candidate_suggestions": [],
                    "reasoning": "Broad family-only evidence.",
                },
                "candidate_suggestions": [
                    {
                        "hts_code": "7326.90",
                        "confidence": 0.62,
                        "rationale": "Fabricated steel article family from HANA router.",
                        "specificity_supported": 6,
                        "hana_router_stage": "family",
                        "matched_path": "Other articles of iron or steel > Other",
                        "matched_terms": ["steel", "disc"],
                        "matched_phrases": ["diffuser disc"],
                        "retrieval_score": 7.2,
                        "missing_discriminators": ["Need machinery or fitting context for narrower branch."],
                    },
                    {
                        "hts_code": "7307.99",
                        "confidence": 0.49,
                        "rationale": "Pipe-fitting branch from HANA router.",
                        "specificity_supported": 6,
                        "hana_router_stage": "family",
                        "matched_path": "Tube or pipe fittings > Other",
                        "matched_terms": ["class", "150#", "steel"],
                        "matched_phrases": ['10"-150#'],
                        "retrieval_score": 6.8,
                        "missing_discriminators": ["Need flange evidence."],
                    },
                ],
            },
            "source_summary": {"new_part_description": "DIFFUSER DISCXNBR COATED"},
            "source_row": {"Part description": "Diffuser disc coated"},
        },
        runner.settings,
        runner.llm,
        _sample_hts_catalog_resolver(),
    )

    assert trade_result["hts_classification"]["best_candidate"]["code"] == "7326.90"
    assert trade_result["hts_classification"]["best_candidate"]["origins"] == ["hana_tree"]
    assert trade_result["hts_resolution_output"]["validated_candidates"][0]["origins"] == ["hana_tree"]
    assert '"hana_tree_search_findings"' in captured["messages"][1]["content"]
    assert details["fallback_used"] is False

def test_trade_decision_prefers_hana_supported_specific_code(monkeypatch):
    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(),
        hts_catalog_resolver=_sample_hts_catalog_resolver(),
        section_232_source_store=_StubSection232Store([_build_section_232_rule()]),
    )
    captured: dict = {}

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            '{"selected_code":"7318.24.0000",'
                            '"reasoning":"The validated candidate 7318.24.0000 is the best match because the product is a steel retaining ring and the HANA tree search preserves the retaining-hardware branch under heading 7318.",'
                            '"confidence":0.91,'
                            '"needs_human_review":false,'
                            '"candidate_rankings":['
                            '{"code":"7318.24.0000","rationale":"Specific validated 10-digit code supported by the HANA tree-search evidence.","confidence":0.91,"missing_discriminators":[]}'
                            ']}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    trade_result, details = run_trade_decision(
        {
            "final_composition": {
                "is_metal_item": True,
                "total_weight_grams": 18.0,
                "estimated_total_metal_grams": 18.0,
                "top_level_grams": {"steel": 18.0, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                "steel_subtype_grams": {},
                "confidence": 0.9,
                "reasoning": "Synthetic retaining ring composition.",
            },
            "hts_fact_profile": {
                "material_profile": {
                    "top_level_grams": {"steel": 18.0, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                    "confidence": 0.9,
                }
            },
            "hana_tree_search_output": {
                "status": "completed",
                "context": {"article_summary": "Retaining ring", "function_summary": "Retains shafts"},
                "chapter_router": {
                    "status": "completed",
                    "selected_chapters": [{"chapter_number": 73}],
                    "reasoning": "Steel hardware chapter remains strongest.",
                },
                "heading_router": {
                    "status": "completed",
                    "selected_headings": [{"heading_code": "7318"}],
                    "reasoning": "Fastener and retaining-hardware heading matches the product.",
                },
                "family_router": {
                    "status": "completed",
                    "candidate_suggestions": [],
                    "reasoning": "Specific retaining-ring branch is supported.",
                },
                "candidate_suggestions": [
                    {
                        "hts_code": "7318.24.0000",
                        "confidence": 0.91,
                        "rationale": "Specific retaining-ring branch.",
                        "specificity_supported": 10,
                        "hana_router_stage": "family",
                        "matched_path": "Cotters and cotter pins; other lock washers",
                        "matched_terms": ["retaining", "ring", "steel"],
                        "matched_phrases": ["retaining ring"],
                        "retrieval_score": 8.4,
                        "missing_discriminators": [],
                    },
                ],
            },
            "source_summary": {"new_part_description": "RETAINING RING DIN 471-60X3-SPRING"},
            "source_row": {"Part description": "Retaining ring"},
        },
        runner.settings,
        runner.llm,
        _sample_hts_catalog_resolver(),
        section_232_source_store=runner.section_232_source_store,
    )

    assert trade_result["hts_classification"]["best_candidate"]["code"] == "7318.24.0000"
    assert trade_result["hts_classification"]["best_candidate"]["validation_status"] == "current_exact"
    assert trade_result["hts_classification"]["confidence"] == 0.91
    assert trade_result["hts_classification"]["best_candidate"]["citations"] == []
    assert trade_result["hts_resolution_output"]["selected_code"] == "7318.24.0000"
    assert trade_result["hts_resolution_output"]["selector_status"] == "completed"
    assert trade_result["section_232_assessment"]["decision"] == "subject"
    assert details["model"] == runner.settings.trade_decision_model_name
    assert details["strategy"] == "llm_catalog_reasoning"
    assert captured["reasoning_effort"] == "medium"
    assert "temperature" not in captured
    assert "max_tokens" not in captured
    assert "max_completion_tokens" not in captured

class _StubPublishedRulesetStore:
    def __init__(self, rules: list[Section232DraftRuleCandidate], *, version: str = "section232-v0001") -> None:
        self._rules = list(rules)
        self._version = version

    def get_active_ruleset_version(self) -> str:
        return self._version

    def list_active_rules(self) -> list[Section232DraftRuleCandidate]:
        return list(self._rules)

class _StubSection232Store:
    def __init__(self, rules: list[Section232DraftRuleCandidate], *, version: str = "section232-v0001") -> None:
        self.ruleset_store = _StubPublishedRulesetStore(rules, version=version)

def _build_section_232_rule(
    *,
    hts_code: str = "7318.24.0000",
    rule_type: str = "include",
    metal_scope: str = "steel",
    effective_from: str = "2026-01-01",
    effective_to: str | None = None,
) -> Section232DraftRuleCandidate:
    return Section232DraftRuleCandidate(
        candidate_id=f"{rule_type}-{hts_code}",
        batch_id="batch-1",
        hts_code=hts_code,
        rule_type=rule_type,
        coverage_effect="remove" if rule_type == "remove" else "include",
        effective_from=effective_from,
        effective_to=effective_to,
        metal_scope=metal_scope,
        source_document_ids=["source-1"],
        source_pages=[1],
        source_excerpt="Synthetic published rule for service tests.",
        interpreter_confidence=0.95,
        catalog_match_found=True,
        review_decision="accepted",
    )

def _run_trade_decision_weight_rule_case(
    monkeypatch,
    *,
    final_composition: dict,
    diagram_output: dict | None = None,
    rule_type: str = "include",
):
    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(section_232_model_name="gpt-4.1"),
        hts_catalog_resolver=_sample_hts_catalog_resolver(),
        section_232_source_store=_StubSection232Store([_build_section_232_rule(rule_type=rule_type)]),
    )
    captured_calls: list[dict] = []

    def fake_invoke(**kwargs):
        captured_calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            '{"selected_code":"7318.24.0000",'
                            '"reasoning":"The validated candidate 7318.24.0000 is best supported by the HANA tree-search evidence.",'
                            '"confidence":0.91,'
                            '"needs_human_review":false,'
                            '"candidate_rankings":['
                            '{"code":"7318.24.0000","rationale":"Specific validated 10-digit code supported by the HANA tree-search evidence.","confidence":0.91,"missing_discriminators":[]}'
                            ']}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    top_level_grams = dict(final_composition.get("top_level_grams") or {})
    trade_result, details = run_trade_decision(
        {
            "final_composition": final_composition,
            "diagram_output": diagram_output or {},
            "hts_fact_profile": {
                "material_profile": {
                    "top_level_grams": {
                        "steel": top_level_grams.get("steel", 0.0),
                        "aluminum": top_level_grams.get("aluminum", 0.0),
                        "copper": top_level_grams.get("copper", 0.0),
                        "cast_iron": top_level_grams.get("cast_iron", 0.0),
                    },
                    "confidence": 0.9,
                }
            },
            "hana_tree_search_output": {
                "status": "completed",
                "context": {"article_summary": "Retaining ring", "function_summary": "Retains shafts"},
                "chapter_router": {
                    "status": "completed",
                    "selected_chapters": [{"chapter_number": 73}],
                    "reasoning": "Steel hardware chapter remains strongest.",
                },
                "heading_router": {
                    "status": "completed",
                    "selected_headings": [{"heading_code": "7318"}],
                    "reasoning": "Retaining hardware heading remains strongest.",
                },
                "family_router": {
                    "status": "completed",
                    "candidate_suggestions": [],
                    "reasoning": "Specific retaining-ring branch is supported.",
                },
                "candidate_suggestions": [
                    {
                        "hts_code": "7318.24.0000",
                        "rationale": "Specific retaining-ring branch.",
                        "confidence": 0.91,
                        "specificity_supported": 10,
                        "hana_router_stage": "family",
                        "matched_path": "Cotters and cotter pins; other lock washers",
                        "matched_terms": ["retaining", "ring", "steel"],
                        "matched_phrases": ["retaining ring"],
                        "retrieval_score": 8.4,
                        "missing_discriminators": [],
                    }
                ],
            },
            "source_summary": {"new_part_description": "RETAINING RING DIN 471-60X3-SPRING"},
            "source_row": {"Part description": "Retaining ring"},
        },
        runner.settings,
        runner.llm,
        _sample_hts_catalog_resolver(),
        section_232_source_store=runner.section_232_source_store,
    )
    return trade_result, details, captured_calls

def test_trade_decision_uses_published_ruleset_when_section_232_rules_match(monkeypatch):
    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(section_232_model_name="gpt-4.1"),
        hts_catalog_resolver=_sample_hts_catalog_resolver(),
        section_232_source_store=_StubSection232Store([_build_section_232_rule()]),
    )
    captured_calls: list[dict] = []

    def fake_invoke(**kwargs):
        captured_calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            '{"selected_code":"7318.24.0000",'
                            '"reasoning":"The validated candidate 7318.24.0000 is best supported by the HANA tree-search evidence.",'
                            '"confidence":0.91,'
                            '"needs_human_review":false,'
                            '"candidate_rankings":['
                            '{"code":"7318.24.0000","rationale":"Specific validated 10-digit code supported by the HANA tree-search evidence.","confidence":0.91,"missing_discriminators":[]}'
                            ']}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    trade_result, _details = run_trade_decision(
        {
            "final_composition": {
                "is_metal_item": True,
                "total_weight_grams": 18.0,
                "estimated_total_metal_grams": 18.0,
                "top_level_grams": {"steel": 18.0, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                "steel_subtype_grams": {},
                "confidence": 0.9,
                "reasoning": "Synthetic retaining ring composition.",
            },
            "hts_fact_profile": {
                "material_profile": {
                    "top_level_grams": {"steel": 18.0, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                    "confidence": 0.9,
                }
            },
            "hana_tree_search_output": {
                "status": "completed",
                "context": {"article_summary": "Retaining ring", "function_summary": "Retains shafts"},
                "chapter_router": {
                    "status": "completed",
                    "selected_chapters": [{"chapter_number": 73}],
                    "reasoning": "Steel hardware chapter remains strongest.",
                },
                "heading_router": {
                    "status": "completed",
                    "selected_headings": [{"heading_code": "7318"}],
                    "reasoning": "Retaining hardware heading remains strongest.",
                },
                "family_router": {
                    "status": "completed",
                    "candidate_suggestions": [],
                    "reasoning": "Specific retaining-ring branch is supported.",
                },
                "candidate_suggestions": [
                    {
                        "hts_code": "7318.24.0000",
                        "rationale": "Specific retaining-ring branch.",
                        "confidence": 0.91,
                        "specificity_supported": 10,
                        "hana_router_stage": "family",
                        "matched_path": "Cotters and cotter pins; other lock washers",
                        "matched_terms": ["retaining", "ring", "steel"],
                        "matched_phrases": ["retaining ring"],
                        "retrieval_score": 8.4,
                        "missing_discriminators": [],
                    }
                ],
            },
            "source_summary": {"new_part_description": "RETAINING RING DIN 471-60X3-SPRING"},
            "source_row": {"Part description": "Retaining ring"},
        },
        runner.settings,
        runner.llm,
        _sample_hts_catalog_resolver(),
        section_232_source_store=runner.section_232_source_store,
    )

    assert trade_result["section_232_assessment"]["decision"] == "subject"
    assert trade_result["section_232_assessment"]["chapter_99_candidates"] == []
    assert trade_result["section_232_reasoner_output"]["strategy"] == "published_ruleset"
    assert trade_result["section_232_reasoner_output"]["fallback_used"] is False
    assert trade_result["section_232_reasoner_output"]["reason"] == "matched_rule"
    assert trade_result["section_232_reasoner_output"]["matched_rule_type"] == "include"
    assert trade_result["section_232_reasoner_output"]["matched_hts_code"] == "7318.24.0000"
    assert trade_result["section_232_reasoner_output"]["active_ruleset_version"] == "section232-v0001"
    assert len(captured_calls) == 1

def test_trade_decision_marks_section_232_for_review_without_published_ruleset(monkeypatch):
    runner = MetalCompositionWorkflowRunner(
        settings=_sample_settings(section_232_model_name="gpt-4.1"),
        hts_catalog_resolver=_sample_hts_catalog_resolver(),
    )
    captured_calls: list[dict] = []

    def fake_invoke(**kwargs):
        captured_calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=(
                            '{"selected_code":"7318.24.0000",'
                            '"reasoning":"The validated candidate 7318.24.0000 is best supported by the HANA tree-search evidence.",'
                            '"confidence":0.91,'
                            '"needs_human_review":false,'
                            '"candidate_rankings":['
                            '{"code":"7318.24.0000","rationale":"Specific validated 10-digit code supported by the HANA tree-search evidence.","confidence":0.91,"missing_discriminators":[]}'
                            ']}'
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(runner.llm, "invoke_native_chat_completion", fake_invoke)

    trade_result, _details = run_trade_decision(
        {
            "hts_fact_profile": {
                "material_profile": {
                    "top_level_grams": {"steel": 18.0, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
                    "confidence": 0.9,
                }
            },
            "hana_tree_search_output": {
                "status": "completed",
                "context": {"article_summary": "Retaining ring", "function_summary": "Retains shafts"},
                "chapter_router": {
                    "status": "completed",
                    "selected_chapters": [{"chapter_number": 73}],
                    "reasoning": "Steel hardware chapter remains strongest.",
                },
                "heading_router": {
                    "status": "completed",
                    "selected_headings": [{"heading_code": "7318"}],
                    "reasoning": "Retaining hardware heading remains strongest.",
                },
                "family_router": {
                    "status": "completed",
                    "candidate_suggestions": [],
                    "reasoning": "Specific retaining-ring branch is supported.",
                },
                "candidate_suggestions": [
                    {
                        "hts_code": "7318.24.0000",
                        "rationale": "Specific retaining-ring branch.",
                        "confidence": 0.91,
                        "specificity_supported": 10,
                        "hana_router_stage": "family",
                        "matched_path": "Cotters and cotter pins; other lock washers",
                        "matched_terms": ["retaining", "ring", "steel"],
                        "matched_phrases": ["retaining ring"],
                        "retrieval_score": 8.4,
                        "missing_discriminators": [],
                    }
                ],
            },
            "source_summary": {"new_part_description": "RETAINING RING DIN 471-60X3-SPRING"},
            "source_row": {"Part description": "Retaining ring"},
        },
        runner.settings,
        runner.llm,
        _sample_hts_catalog_resolver(),
    )

    assert trade_result["section_232_assessment"]["decision"] == "needs_review"
    assert trade_result["section_232_reasoner_output"]["strategy"] == "published_ruleset"
    assert trade_result["section_232_reasoner_output"]["fallback_used"] is False
    assert trade_result["section_232_reasoner_output"]["reason"] == "no_ruleset_store"
    assert len(captured_calls) == 1

def test_trade_decision_weight_rule_exact_below_threshold_marks_not_subject(monkeypatch):
    trade_result, _details, _captured_calls = _run_trade_decision_weight_rule_case(
        monkeypatch,
        final_composition={
            "is_metal_item": True,
            "total_weight_grams": 100.0,
            "estimated_total_metal_grams": 14.0,
            "top_level_grams": {"steel": 8.0, "aluminum": 6.0, "copper": 0.0, "cast_iron": 10.0},
            "steel_subtype_grams": {},
            "confidence": 0.92,
            "reasoning": "Exact PDF composition.",
        },
        diagram_output={"metal_share_certainty": "exact"},
    )

    assert trade_result["section_232_assessment"]["decision"] == "not_subject"
    assert trade_result["section_232_assessment"]["needs_human_review"] is False
    assert trade_result["section_232_assessment"]["weight_rule_applied"] is True
    assert trade_result["section_232_assessment"]["confidence"] >= 0.95
    assert "below 15% of total item weight" in trade_result["section_232_assessment"]["basis_summary"]
    assert trade_result["section_232_reasoner_output"]["metal_weight_override"]["affected_metal_share"] == pytest.approx(0.14)
    assert trade_result["section_232_reasoner_output"]["metal_weight_override"]["reason"] == "exact_below_threshold"

def test_trade_decision_weight_rule_estimated_subject_above_threshold_appends_estimate_note(monkeypatch):
    trade_result, _details, _captured_calls = _run_trade_decision_weight_rule_case(
        monkeypatch,
        final_composition={
            "is_metal_item": True,
            "total_weight_grams": 100.0,
            "estimated_total_metal_grams": 16.0,
            "top_level_grams": {"steel": 16.0, "aluminum": 0.0, "copper": 0.0, "cast_iron": 0.0},
            "steel_subtype_grams": {},
            "confidence": 0.85,
            "reasoning": "Estimated PDF composition.",
        },
        diagram_output={"metal_share_certainty": "estimated"},
    )

    assert trade_result["section_232_assessment"]["decision"] == "subject"
    assert trade_result["section_232_assessment"]["needs_human_review"] is False
    assert trade_result["section_232_assessment"]["weight_rule_applied"] is False
    assert "estimated from the assigned PDFs" in trade_result["section_232_assessment"]["basis_summary"]
    assert trade_result["section_232_reasoner_output"]["metal_weight_override"]["reason"] == "threshold_not_met"

def test_trade_decision_weight_rule_estimated_below_threshold_marks_not_subject_with_review(monkeypatch):
    trade_result, _details, _captured_calls = _run_trade_decision_weight_rule_case(
        monkeypatch,
        final_composition={
            "is_metal_item": True,
            "total_weight_grams": 100.0,
            "estimated_total_metal_grams": 14.0,
            "top_level_grams": {"steel": 8.0, "aluminum": 6.0, "copper": 0.0, "cast_iron": 0.0},
            "steel_subtype_grams": {},
            "confidence": 0.7,
            "reasoning": "Estimated PDF composition.",
        },
        diagram_output={"metal_share_certainty": "estimated"},
    )

    assert trade_result["section_232_assessment"]["decision"] == "not_subject"
    assert trade_result["section_232_assessment"]["needs_human_review"] is True
    assert trade_result["section_232_assessment"]["weight_rule_applied"] is True
    assert trade_result["section_232_assessment"]["confidence"] == pytest.approx(0.55)
    assert "should be reviewed by a human" in trade_result["section_232_assessment"]["basis_summary"]
    assert trade_result["section_232_reasoner_output"]["metal_weight_override"]["reason"] == "estimated_below_threshold"

def test_trade_decision_weight_rule_skips_without_total_weight(monkeypatch):
    trade_result, _details, _captured_calls = _run_trade_decision_weight_rule_case(
        monkeypatch,
        final_composition={
            "is_metal_item": True,
            "total_weight_grams": None,
            "estimated_total_metal_grams": 14.0,
            "top_level_grams": {"steel": 8.0, "aluminum": 6.0, "copper": 0.0, "cast_iron": 0.0},
            "steel_subtype_grams": {},
            "confidence": 0.7,
            "reasoning": "Estimated PDF composition.",
        },
        diagram_output={"metal_share_certainty": "estimated"},
    )

    assert trade_result["section_232_assessment"]["decision"] == "subject"
    assert trade_result["section_232_assessment"]["weight_rule_applied"] is False
    assert trade_result["section_232_reasoner_output"]["metal_weight_override"]["reason"] == "missing_total_weight"

def test_trade_decision_weight_rule_does_not_promote_base_not_subject(monkeypatch):
    trade_result, _details, captured_calls = _run_trade_decision_weight_rule_case(
        monkeypatch,
        final_composition={
            "is_metal_item": True,
            "total_weight_grams": 100.0,
            "estimated_total_metal_grams": 40.0,
            "top_level_grams": {"steel": 25.0, "aluminum": 15.0, "copper": 0.0, "cast_iron": 0.0},
            "steel_subtype_grams": {},
            "confidence": 0.9,
            "reasoning": "Exact PDF composition.",
        },
        diagram_output={"metal_share_certainty": "exact"},
        rule_type="remove",
    )

    assert trade_result["section_232_assessment"]["decision"] == "not_subject"
    assert trade_result["section_232_assessment"]["weight_rule_applied"] is False
    assert trade_result["section_232_reasoner_output"]["metal_weight_override"]["reason"] == "base_not_subject"
    assert len(captured_calls) == 1

def test_merge_candidate_entries_dedupes_repeated_reasoning():
    runner = MetalCompositionWorkflowRunner(settings=_sample_settings())

    merged = merge_candidate_entries(
        {
            "code": "7318.29.00",
            "description": "Other",
            "digits": 8,
            "confidence": 0.7,
            "reasoning": "Retaining rings of steel are classified under 7318.29.0000.",
            "citations": [],
            "missing_discriminators": [],
        },
        {
            "code": "7318.29.00",
            "description": "Other",
            "digits": 8,
            "confidence": 0.72,
            "reasoning": "Retaining rings of steel are classified under 7318.29.0000.",
            "citations": [],
            "missing_discriminators": [],
        },
    )

    assert merged["reasoning"] == "Retaining rings of steel are classified under 7318.29.0000."

def test_classify_item_gcc_tracker_mode_normalizes_missing_grams_to_zero(tmp_path):
    source_df, _prepared_df = _sample_serving_frames()
    prepared_df = pd.DataFrame([{"source_row_id": 1}])
    service = MetalCompositionService(
        serving_store=WorkbookStore(source_df=source_df, prepared_df=prepared_df),
        workflow_runner=_ServiceTestWorkflowRunner(),
        settings=_sample_settings(
            ui_state_db_path=tmp_path / "ui_state.sqlite3",
        ),
        ui_state_store=InMemoryMetalCompositionUIStateStore(),
        section_232_source_store=InMemorySection232SourceStore(),
    )
    service.update_app_settings(use_gcc_tracker_metal_composition=True)
    _upload_fake_pdf(service, "gcc:1", "gcc-tracker-zero.pdf")

    result = service.classify_item("gcc:1")

    assert result.result.status == "completed"
    assert result.result.final_composition is not None
    assert result.result.final_composition.is_metal_item is False
    assert result.result.final_composition.estimated_total_metal_grams == 0.0
    assert result.result.final_composition.top_level_grams.steel == 0.0
    assert result.result.final_composition.top_level_grams.aluminum == 0.0

def test_submit_classify_item_job_supersedes_existing_active_job(tmp_path):
    service = _sample_service(
        ui_state_db_path=tmp_path / "ui_state.sqlite3",
    )
    _upload_fake_pdf(service, "gcc:1", "job-gcc.pdf")

    first = service.submit_classify_item_job("gcc:1")
    second = service.submit_classify_item_job("gcc:1")

    first_job = service.get_classification_job(first.job_id)
    first_items = service.classification_job_store.get_job_items(first.job_id)
    second_job = service.get_classification_job(second.job_id)

    assert first_job.status == "failed"
    assert first_job.failed_count == 1
    assert first_items[0].status == "failed"
    assert first_items[0].error_message == SUPERSEDED_ERROR_MESSAGE
    assert second_job.status == "queued"
    assert service.classification_job_store.get_active_item_ids({("gcc:1", service.dataset_signature)}) == ["gcc:1"]

def test_hana_supersede_active_items_does_not_compare_nclob_error_message():
    store = object.__new__(ClassificationJobStore)
    store.job_items_table = "JOB_ITEMS"
    store.ownership_table = "OWNERSHIP"
    store.schema = None
    fetched_sql: list[str] = []
    executed_sql: list[str] = []
    refreshed_jobs: list[str] = []

    def fake_fetch_rows(sql, params=None):
        """Return active rows while rejecting the HANA-invalid NCLOB equality."""
        sql_text = str(sql)
        assert 'WHERE "ERROR_MESSAGE" = ?' not in sql_text
        fetched_sql.append(sql_text)
        assert params == ["gcc:1", "scope-a"]
        return [{"job_id": "job-1", "item_count": 1}]

    def fake_execute(sql, params=None):
        """Capture update/delete SQL while rejecting the HANA-invalid NCLOB equality."""
        sql_text = str(sql)
        assert 'WHERE "ERROR_MESSAGE" = ?' not in sql_text
        executed_sql.append(sql_text)

    store._fetch_rows = fake_fetch_rows
    store.connection = types.SimpleNamespace(execute=fake_execute)
    store._refresh_job_counts = refreshed_jobs.append

    superseded = store.supersede_active_items({("gcc:1", "scope-a")})

    assert superseded == 1
    assert len(fetched_sql) == 1
    assert 'COUNT(*) AS "ITEM_COUNT"' in fetched_sql[0]
    assert len(executed_sql) == 2
    assert refreshed_jobs == ["job-1"]

def test_superseded_running_item_cannot_overwrite_newer_snapshot(tmp_path):
    service = _sample_service(
        ui_state_db_path=tmp_path / "ui_state.sqlite3",
    )
    _upload_fake_pdf(service, "gcc:1", "superseded-gcc.pdf")

    first_job = service.submit_classify_item_job("gcc:1")
    claimed_first = service.claim_next_classification_job("worker-1")
    assert claimed_first is not None
    assert claimed_first.job_id == first_job.job_id
    service.classification_job_store.mark_job_items_running(first_job.job_id)

    second_job = service.submit_classify_item_job("gcc:1")

    stale_result = service.classify_item("gcc:1", owner_job_id=first_job.job_id)
    assert stale_result.last_classified_at is None
    assert service.get_item_detail("gcc:1").latest_classification is None

    claimed_second = service.claim_next_classification_job("worker-2")
    assert claimed_second is not None
    assert claimed_second.job_id == second_job.job_id
    service.process_claimed_classification_job(second_job.job_id)
    fresh_detail = service.get_item_detail("gcc:1")
    assert fresh_detail.latest_classification is not None
    assert fresh_detail.latest_classification.status == "completed"

    service.process_claimed_classification_job(first_job.job_id)
    first_status = service.get_classification_job(first_job.job_id)
    first_items = service.classification_job_store.get_job_items(first_job.job_id)
    final_detail = service.get_item_detail("gcc:1")

    assert first_status.status == "failed"
    assert first_items[0].error_message == SUPERSEDED_ERROR_MESSAGE
    assert final_detail.latest_classification is not None
    assert final_detail.latest_classification.status == "completed"

def test_dataset_signature_invalidates_saved_gcc_snapshot(tmp_path):
    db_path = tmp_path / "ui_state.sqlite3"
    ui_state_store = InMemoryMetalCompositionUIStateStore()

    service_one = _sample_service(
        ui_state_store=ui_state_store,
        ui_state_db_path=db_path,
    )
    _upload_fake_pdf(service_one, "gcc:1", "signature-gcc.pdf")
    classify_result = service_one.classify_item("gcc:1")
    assert classify_result.result.status == "completed"
    assert service_one.get_item_detail("gcc:1").latest_classification is not None

    source_df, prepared_df = _sample_serving_frames()
    source_df.loc[0, "Part description"] = "Pump housing updated"
    service_two = MetalCompositionService(
        serving_store=WorkbookStore(source_df=source_df, prepared_df=prepared_df),
        workflow_runner=_ServiceTestWorkflowRunner(),
        settings=_sample_settings(
            ui_state_db_path=db_path,
        ),
        ui_state_store=ui_state_store,
        section_232_source_store=InMemorySection232SourceStore(),
    )

    detail = service_two.get_item_detail("gcc:1")

    assert detail.latest_classification is None

def test_dataset_signature_stays_stable_across_runtime_metadata_changes(tmp_path):
    db_path = tmp_path / "ui_state.sqlite3"
    ui_state_store = InMemoryMetalCompositionUIStateStore()
    source_df, prepared_df = _sample_serving_frames()

    service_one = MetalCompositionService(
        serving_store=WorkbookStore(
            source_df=source_df.copy(),
            prepared_df=prepared_df.copy(),
            metadata={
                "source": "hana",
                "hana_table": "METAL_COMPOSITION_SERVING",
                "row_count": 1,
            },
        ),
        workflow_runner=_ServiceTestWorkflowRunner(),
        settings=_sample_settings(
            ui_state_db_path=db_path,
        ),
        ui_state_store=ui_state_store,
        section_232_source_store=InMemorySection232SourceStore(),
        cache_source="hana",
    )

    _upload_fake_pdf(service_one, "gcc:1", "stable-signature-gcc.pdf")
    classify_result = service_one.classify_item("gcc:1")
    assert classify_result.result.status == "completed"
    assert service_one.get_item_detail("gcc:1").latest_classification is not None

    service_two = MetalCompositionService(
        serving_store=WorkbookStore(
            source_df=source_df.copy(),
            prepared_df=prepared_df.copy(),
            metadata={
                "source": "local_snapshot",
                "snapshot_format": "pickle",
                "saved_at": "2026-04-01T10:00:00Z",
                "row_count": 1,
            },
        ),
        workflow_runner=_ServiceTestWorkflowRunner(),
        settings=_sample_settings(
            ui_state_db_path=db_path,
        ),
        ui_state_store=ui_state_store,
        section_232_source_store=InMemorySection232SourceStore(),
        cache_source="local_snapshot",
    )

    detail = service_two.get_item_detail("gcc:1")

    assert detail.latest_classification is not None
