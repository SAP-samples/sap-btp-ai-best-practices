from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.nbo.config import COL_RATE_PLAN
from app.nbo.hana_loader import load_seed_datasets


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DATA_DIR = REPO_ROOT / "api" / "demo_data"
DEMO_CUSTOMER_WORKBOOK = DEMO_DATA_DIR / "data_seed" / "customer_seed.xlsx"
DEMO_PROGRAM_WORKBOOK = DEMO_DATA_DIR / "data_seed" / "program_seed.xlsx"
EXPECTED_PROGRAM_IDS = {
    "prepay_advance",
    "income_qualified_discount",
    "battery_partner",
}


def _read_json(path: Path) -> list[dict]:
    """Load one generated catalog JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_values(value):
    """Yield scalar values from nested JSON-compatible data."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _source_documents(payloads: dict[str, list[dict]]) -> set[str]:
    """Return all source document names referenced by generated catalogs."""
    sources: set[str] = set()

    def walk(value) -> None:
        """Collect source/evidence document values from one JSON node."""
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"source_documents", "evidence_references"}:
                    sources.update(str(source) for source in item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for payload in payloads.values():
        walk(payload)
    return sources


def test_generator_outputs_catalogs_aligned_with_demo_workbooks(tmp_path: Path) -> None:
    """Catalog generation should stay small and match the public demo workbooks."""
    from api.scripts import generate_anonymized_catalogs

    assert DEMO_CUSTOMER_WORKBOOK.exists()
    assert DEMO_PROGRAM_WORKBOOK.exists()

    summary = generate_anonymized_catalogs.generate_catalogs(
        customer_workbook=DEMO_CUSTOMER_WORKBOOK,
        program_workbook=DEMO_PROGRAM_WORKBOOK,
        output_dir=tmp_path,
    )

    assert summary["program_count"] == len(EXPECTED_PROGRAM_IDS)
    payloads = {
        "program_catalog": _read_json(tmp_path / "program_catalog.json"),
        "program_rule_matrix": _read_json(tmp_path / "program_rule_matrix.json"),
        "tariff_catalog": _read_json(tmp_path / "tariff_catalog.json"),
        "commercial_taxonomy": _read_json(tmp_path / "commercial_taxonomy.json"),
    }

    program_ids = {entry["program_id"] for entry in payloads["program_catalog"]}
    matrix_ids = {entry["program_id"] for entry in payloads["program_rule_matrix"]}
    assert program_ids == EXPECTED_PROGRAM_IDS
    assert matrix_ids == EXPECTED_PROGRAM_IDS

    datasets = load_seed_datasets(
        customer_workbook=DEMO_CUSTOMER_WORKBOOK,
        program_codes_workbook=DEMO_PROGRAM_WORKBOOK,
    )
    active_rate_plans = set(
        datasets["active_offering"][COL_RATE_PLAN].dropna().astype(str).str.strip()
    )
    tariff_rate_plans = {entry["rate_plan"] for entry in payloads["tariff_catalog"]}
    assert tariff_rate_plans <= active_rate_plans
    assert {"E21", "E23", "E24", "E26", "E16"} <= tariff_rate_plans
    assert "E00" not in tariff_rate_plans

    assert payloads["commercial_taxonomy"][-1]["taxonomy"] == "GENERAL_BUSINESS"


def test_generated_catalogs_use_public_source_document_filenames(tmp_path: Path) -> None:
    """Generated source references should be filenames with matching placeholders."""
    from api.scripts import generate_anonymized_catalogs

    generate_anonymized_catalogs.generate_catalogs(
        customer_workbook=DEMO_CUSTOMER_WORKBOOK,
        program_workbook=DEMO_PROGRAM_WORKBOOK,
        output_dir=tmp_path,
    )
    payloads = {
        path.stem: _read_json(path)
        for path in tmp_path.glob("*.json")
    }

    for source in _source_documents(payloads):
        assert "/" not in source
        assert source in {
            "customer-assistance-guide.pdf",
            "battery-partner-guide.pdf",
            "rate-plan-guide.pdf",
            "offer-guidance-notes.docx",
        }
        assert generate_anonymized_catalogs.public_source_document_path(source).exists()


def test_public_catalog_validation_rejects_forbidden_terms() -> None:
    """Generator validation should fail fast if public catalogs leak private text."""
    from api.scripts import generate_anonymized_catalogs

    private_brand = "S" + "RP"
    payloads = {
        "program_catalog": [
            {
                "program_id": "private_demo",
                "display_name": f"{private_brand} private case",
                "evidence_references": ["customer-assistance-guide.pdf"],
            }
        ],
        "program_rule_matrix": [],
        "tariff_catalog": [],
        "commercial_taxonomy": [],
    }

    with pytest.raises(ValueError, match=private_brand):
        generate_anonymized_catalogs.validate_public_catalog_payloads(payloads)


def test_generated_catalog_values_do_not_contain_forbidden_terms(tmp_path: Path) -> None:
    """Generated catalog text should remain publishable."""
    from api.scripts import generate_anonymized_catalogs

    generate_anonymized_catalogs.generate_catalogs(
        customer_workbook=DEMO_CUSTOMER_WORKBOOK,
        program_workbook=DEMO_PROGRAM_WORKBOOK,
        output_dir=tmp_path,
    )

    joined_values = "\n".join(
        str(value)
        for path in tmp_path.glob("*.json")
        for value in _walk_values(_read_json(path))
    )
    for term in generate_anonymized_catalogs.FORBIDDEN_PUBLIC_TERMS:
        assert term not in joined_values
