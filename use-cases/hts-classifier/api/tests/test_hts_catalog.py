from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.services.metal_composition.config import MetalCompositionSettings
from app.services.metal_composition.hts_catalog import (
    CATALOG_COLUMNS,
    CODE_MAP_COLUMNS,
    HanaHTSCatalogResolver,
    compile_hts_catalog_frame,
    compile_hts_code_map_frame,
    refresh_hts_catalog_tables,
)


ROOT = Path(__file__).resolve().parents[2]
HTS_CSV_DIR = ROOT / "data" / "hts_chapters"
HTS_CODE_MAP = HTS_CSV_DIR / "hts_code_map.csv"


def _sample_settings(**overrides) -> MetalCompositionSettings:
    return MetalCompositionSettings(
        workbook_path=ROOT / "data" / "GCC Tracker.xlsb",
        api_env_path=ROOT / "api" / ".env",
        hts_catalog_dir=HTS_CSV_DIR,
        hts_code_map_path=HTS_CODE_MAP,
        **overrides,
    )


def _synthetic_resolver_for_chapter_ranking() -> HanaHTSCatalogResolver:
    def row(code: str, chapter_number: int, description: str, searchable_text: str) -> dict:
        return {
            "code": code,
            "raw_code": code,
            "digits": 4,
            "chapter_number": chapter_number,
            "heading_code": code,
            "family_6_code": "",
            "family_8_code": "",
            "indent": 0,
            "parent_code": "",
            "description": description,
            "path_description": description,
            "unit_of_quantity": "",
            "general_rate_of_duty": "",
            "special_rate_of_duty": "",
            "column_2_rate_of_duty": "",
            "quota_quantity": "",
            "additional_duties": "",
            "searchable_text": searchable_text,
            "sort_order": int(code),
        }

    return HanaHTSCatalogResolver(
        settings=_sample_settings(),
        catalog_frame=pd.DataFrame(
            [
                row("0101", 1, "Live horses, asses, mules and hinnies", "0101 chapter live horses animals"),
                row("4016", 40, "Other articles of vulcanized rubber", "4016 vulcanized rubber gasket coated housing"),
                row("7325", 73, "Other cast articles of iron or steel", "7325 cast iron pump housing"),
                row("8413", 84, "Pumps for liquids; parts thereof", "8413 pump oil housing parts"),
                row("8483", 84, "Bearing housings and parts thereof", "8483 bearing housing machinery parts"),
            ],
            columns=CATALOG_COLUMNS,
        ),
        code_map_frame=pd.DataFrame([], columns=CODE_MAP_COLUMNS),
    )


def test_compile_hts_catalog_frame_reconstructs_current_10_digit_codes():
    catalog = compile_hts_catalog_frame(csv_dir=HTS_CSV_DIR)

    row_7308 = catalog.loc[catalog["code"] == "7308.90.9560"].iloc[0].to_dict()
    row_8421 = catalog.loc[catalog["code"] == "8421.99.0140"].iloc[0].to_dict()

    assert row_7308["raw_code"] == "7308.90.95.60"
    assert row_7308["digits"] == 10
    assert "Architectural and ornamental work" in row_7308["path_description"]
    assert row_8421["raw_code"] == "8421.99.01.40"
    assert row_8421["family_8_code"] == "8421.99.01"


def test_compile_hts_catalog_frame_preserves_uncoded_branch_labels_in_paths():
    catalog = compile_hts_catalog_frame(csv_dir=HTS_CSV_DIR)

    row_9060 = catalog.loc[catalog["code"] == "8413.91.9060"].iloc[0].to_dict()
    row_9085 = catalog.loc[catalog["code"] == "8413.91.9085"].iloc[0].to_dict()
    row_9096 = catalog.loc[catalog["code"] == "8413.91.9096"].iloc[0].to_dict()

    assert "Parts:" in row_9060["path_description"]
    assert "Of hydraulic fluid power pumps:" in row_9060["path_description"]
    assert "Of subheading 8413.50.00:" in row_9085["path_description"]
    assert "Of subheading 8413.50.00:" not in row_9096["path_description"]
    assert row_9096["path_description"].endswith("Of pumps: > Other > Other")


def test_compile_hts_code_map_frame_normalizes_official_mapping_file():
    code_map = compile_hts_code_map_frame(code_map_path=HTS_CODE_MAP)

    row = code_map.iloc[0].to_dict()

    assert row["source_code"] == "8421.99.0040"
    assert row["target_code"] == "8421.99.0140"
    assert row["mapping_type"] == "explicit_official_map"


def test_hana_hts_catalog_resolver_maps_legacy_and_rejects_invalid():
    catalog = compile_hts_catalog_frame(csv_dir=HTS_CSV_DIR)
    code_map = compile_hts_code_map_frame(code_map_path=HTS_CODE_MAP)
    resolver = HanaHTSCatalogResolver(
        settings=_sample_settings(),
        catalog_frame=catalog,
        code_map_frame=code_map,
    )

    mapped = resolver.resolve_code("8421.99.0040")
    current = resolver.resolve_code("7308.90.95.60")
    invalid = resolver.resolve_code("7326.90.99.85")

    assert mapped.validation_status == "legacy_mapped_exact"
    assert mapped.resolved_code == "8421.99.0140"
    assert current.validation_status == "current_exact"
    assert current.resolved_code == "7308.90.9560"
    assert invalid.validation_status == "invalid"
    assert invalid.resolved_code == ""


def test_hana_hts_catalog_resolver_lists_relevant_chapters_headings_and_families():
    catalog = compile_hts_catalog_frame(csv_dir=HTS_CSV_DIR)
    code_map = compile_hts_code_map_frame(code_map_path=HTS_CODE_MAP)
    resolver = HanaHTSCatalogResolver(
        settings=_sample_settings(),
        catalog_frame=catalog,
        code_map_frame=code_map,
    )
    context = {
        "article_summary": "Industrial diffuser disc, predominantly steel, coated with NBR.",
        "function_summary": "Acts as a flow-conditioning disc within Class 150 piping service.",
        "material_clues": ["steel", "hot rolled coil steel"],
        "standard_cues": ["ASME Class 150"],
        "heading_hypotheses": ["7307", "7326", "8481"],
        "phrases": ['DISC HEAD F 10"-150# X 20"BD SQ', "diffuser disc coated"],
    }

    chapter_options = resolver.list_chapter_options(context)
    heading_options = resolver.list_heading_options([73, 84, 85], context)
    family_options = resolver.list_family_options(["7307", "7326", "8481"], context)

    assert {73, 84}.issubset({item["chapter_number"] for item in chapter_options[:6]})
    assert {"7307", "7326", "8481"}.issubset({item["heading_code"] for item in heading_options})
    assert {"7307.99", "7326.90", "8481.90"}.issubset({item["code"] for item in family_options})


def test_hana_hts_catalog_resolver_ranks_pump_housing_likely_chapters_above_unrelated():
    resolver = _synthetic_resolver_for_chapter_ranking()

    chapter_options = resolver.list_chapter_options(
        {
            "article_summary": "Cast-iron oil housing for a pump assembly.",
            "function_summary": "Pump spare housing with rubber coating and bearing-housing cues.",
            "material_clues": ["cast iron", "rubber coating"],
            "heading_hypotheses": ["8413", "8483", "7325", "4016"],
        }
    )

    ranked_chapters = [item["chapter_number"] for item in chapter_options]
    assert set(ranked_chapters[:3]) == {84, 73, 40}
    assert ranked_chapters.index(1) > 2


def test_hana_hts_catalog_resolver_drops_stopword_only_raw_tokens_from_prefilter_matches():
    resolver = _synthetic_resolver_for_chapter_ranking()

    heading_options = resolver.list_heading_options(
        [1],
        {
            "tokens": ["chapter", "the", "and"],
            "phrases": ["the chapter"],
        },
    )

    chapter_one_heading = heading_options[0]
    assert chapter_one_heading["heading_code"] == "0101"
    assert chapter_one_heading["matched_terms"] == []
    assert chapter_one_heading["matched_phrases"] == []
    assert chapter_one_heading["prefilter_score"] == 0.0


def test_expand_children_only_returns_current_catalog_children():
    catalog = compile_hts_catalog_frame(csv_dir=HTS_CSV_DIR)
    code_map = compile_hts_code_map_frame(code_map_path=HTS_CODE_MAP)
    resolver = HanaHTSCatalogResolver(
        settings=_sample_settings(),
        catalog_frame=catalog,
        code_map_frame=code_map,
    )

    children = resolver.expand_children(
        ["8421.99"],
        {
            "article_summary": "Part for filtering or purifying water machinery",
            "function_summary": "Water purification apparatus part",
            "material_clues": ["steel"],
            "heading_hypotheses": ["8421"],
        },
    )

    child_codes = {item["code"] for item in children.get("8421.99", [])}
    assert "8421.99.0140" in child_codes
    assert "8421.99.0040" not in child_codes


def test_refresh_hts_catalog_tables_pushes_catalog_and_code_map(monkeypatch):
    captured: list[dict] = []

    class _FakeHANAConnection:
        def refresh_serving_table(self, *, frame, table, schema, primary_key, index_columns):
            captured.append(
                {
                    "table": table,
                    "schema": schema,
                    "primary_key": primary_key,
                    "index_columns": tuple(index_columns),
                    "row_count": len(frame),
                    "columns": list(frame.columns),
                }
            )
            return {"table": table, "schema": schema, "row_count": len(frame)}

    monkeypatch.setattr("app.services.metal_composition.hts_catalog.HANAConnection", _FakeHANAConnection)

    result = refresh_hts_catalog_tables(
        settings=_sample_settings(hts_hana_schema="TEST", hts_catalog_hana_table="CATALOG", hts_code_map_hana_table="CODE_MAP")
    )

    assert result["status"] == "completed"
    assert [entry["table"] for entry in captured] == ["CATALOG", "CODE_MAP"]
    assert captured[0]["primary_key"] == "code"
    assert captured[1]["primary_key"] == "source_code"
    assert captured[0]["row_count"] > 0
    assert captured[1]["row_count"] == 1
