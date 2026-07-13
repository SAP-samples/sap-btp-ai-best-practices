from __future__ import annotations

import json
from pathlib import Path

from app.nbo import config
from app.nbo.hana import DATASET_TABLES
from app.nbo.hana_loader import load_seed_datasets
from synthetic_runtime import synthetic_runtime_datasets


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = REPO_ROOT / "api" / "app" / "nbo" / "catalogs"
DEMO_DATA_DIR = REPO_ROOT / "api" / "demo_data"
DEMO_SOURCE_DOCUMENT_DIR = DEMO_DATA_DIR / "source_documents"
DEMO_DATA_SEED_DIR = DEMO_DATA_DIR / "data_seed"
UI_SOURCE_PATHS = [
    REPO_ROOT / "ui" / "index.html",
    REPO_ROOT / "ui" / "package.json",
    *sorted((REPO_ROOT / "ui" / "src").rglob("*.*")),
]
API_VISIBLE_PATHS = [
    REPO_ROOT / "api" / "app" / "main.py",
    REPO_ROOT / "api" / "app" / "chat" / "service.py",
]
BLOCKED_PUBLIC_TERMS = (
    "S" + "RP",
    "s" + "rp",
    "Next best " + "Offer",
    "M-" + "Power",
    "I" + "QD",
    "data/" + "Programs",
    "data/" + "Rate Plans",
)


def _walk_values(value):
    """Yield every scalar value nested inside a JSON-compatible structure."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _source_document_values(value):
    """Yield source/evidence document strings from active catalog payloads."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"source_documents", "evidence_references"}:
                for source in item:
                    yield source
            else:
                yield from _source_document_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _source_document_values(item)


def _catalog_source_documents() -> set[str]:
    """Return unique active catalog source-document filenames."""
    sources: set[str] = set()
    for path in sorted(CATALOG_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.update(_source_document_values(payload))
    return sources


def _expected_public_source_path(source_document: str) -> Path:
    """Map one catalog source-document name to its public anonymized file path."""
    return DEMO_SOURCE_DOCUMENT_DIR / source_document


def _assert_public_text_is_anonymized(text: str, label: str) -> None:
    """Assert public share text does not contain blocked internal source terms."""
    for term in BLOCKED_PUBLIC_TERMS:
        assert term not in text, f"{label} exposes {term!r}"


def test_active_catalogs_use_anonymized_source_document_names() -> None:
    """Active catalogs should expose only generic final filenames."""
    for path in sorted(CATALOG_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for source in _source_document_values(payload):
            assert "/" not in source, f"{path.name} contains a path-like source: {source}"
            assert BLOCKED_PUBLIC_TERMS[0] not in source.upper(), (
                f"{path.name} contains private source text: {source}"
            )
            assert BLOCKED_PUBLIC_TERMS[2] not in source, (
                f"{path.name} contains old offer source: {source}"
            )


def test_active_catalogs_do_not_expose_private_branding() -> None:
    """Active catalog values should not expose private labels."""
    blocked_terms = (
        BLOCKED_PUBLIC_TERMS[0],
        BLOCKED_PUBLIC_TERMS[2],
        BLOCKED_PUBLIC_TERMS[5],
        BLOCKED_PUBLIC_TERMS[6],
    )
    for path in sorted(CATALOG_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = [str(value) for value in _walk_values(payload)]
        joined = "\n".join(values)
        for term in blocked_terms:
            assert term not in joined, f"{path.name} exposes {term!r}"


def test_runtime_config_is_hana_first_and_generic() -> None:
    """Runtime config should avoid local file seeds and private-prefixed tables."""
    removed_names = {
        "DATA_DIR",
        "DATA_SEED_DIR",
        "CUSTOMER_SEED_WORKBOOK",
        "PROGRAM_CODES_SEED_WORKBOOK",
        "PROGRAMS_PDF_DIR",
        "RATE_PLAN_PDF_DIR",
        "NEXT_BEST_OFFER_DOCX",
    }
    for name in removed_names:
        assert not hasattr(config, name), f"config still exports {name}"

    table_values = [
        value
        for name, value in vars(config).items()
        if name.startswith("TABLE_") and isinstance(value, str)
    ]
    assert table_values
    assert all(value.startswith("COA_") for value in table_values)
    assert not any(value.startswith("S" + "RP_NBO_") for value in table_values)


def test_ui_and_api_visible_text_are_anonymized() -> None:
    """Visible application text should use Customer Offer Advisor branding."""
    visible_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [*UI_SOURCE_PATHS, *API_VISIBLE_PATHS]
        if path.is_file()
    )
    assert "Customer Offer Advisor" in visible_text
    assert BLOCKED_PUBLIC_TERMS[0] not in visible_text
    assert BLOCKED_PUBLIC_TERMS[1] not in visible_text
    assert "Next Best " + "Offer" not in visible_text


def test_public_demo_seed_workbooks_load_through_hana_loader() -> None:
    """Public seed workbooks should load into the same datasets as HANA runtime."""
    customer_workbook = DEMO_DATA_SEED_DIR / "customer_seed.xlsx"
    program_workbook = DEMO_DATA_SEED_DIR / "program_seed.xlsx"

    assert customer_workbook.exists()
    assert program_workbook.exists()

    datasets = load_seed_datasets(
        customer_workbook=customer_workbook,
        program_codes_workbook=program_workbook,
    )
    expected_datasets = synthetic_runtime_datasets()

    assert set(datasets) == set(DATASET_TABLES)
    assert {
        name: len(frame)
        for name, frame in datasets.items()
    } == {
        name: len(frame)
        for name, frame in expected_datasets.items()
    }
    assert len(datasets["residential"]) > 0
    assert len(datasets["commercial"]) > 0
    assert len(datasets["active_offering"]) > 0
    assert len(datasets["program_contract"]) > 0

    joined_values = "\n".join(
        str(value)
        for frame in datasets.values()
        for value in frame.astype(str).to_numpy().ravel()
    )
    _assert_public_text_is_anonymized(joined_values, "public seed workbooks")


def test_public_demo_source_documents_exist_for_catalog_references() -> None:
    """Every active catalog source-document filename should have a public placeholder."""
    sources = _catalog_source_documents()
    assert sources

    for source_document in sources:
        path = _expected_public_source_path(source_document)
        assert path.exists(), f"Missing public source placeholder for {source_document}"
        if path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF-")
        else:
            _assert_public_text_is_anonymized(
                path.read_text(encoding="utf-8", errors="ignore"),
                path.name,
            )
