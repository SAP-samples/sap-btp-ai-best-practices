from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.services.metal_composition import section_232_sources as section_232_sources_module
from app.services.metal_composition.config import MetalCompositionSettings
from app.services.metal_composition.section_232_sources import (
    ExtractedSection232Source,
    InMemorySection232SourceStore,
    extract_section_232_pdf,
    extract_hts_mentions,
)
from app.services.metal_composition.service import MetalCompositionService
from app.services.metal_composition.section_232_rulesets import InMemorySection232RulesetStore
from app.services.metal_composition.serving_store import WorkbookStore
from app.services.metal_composition.ui_state import InMemoryMetalCompositionUIStateStore

ROOT = Path(__file__).resolve().parents[2]

def _sample_settings(**overrides) -> MetalCompositionSettings:
    return MetalCompositionSettings(
        workbook_path=ROOT / "data" / "Material Master.xlsb",
        api_env_path=ROOT / "api" / ".env",
        **overrides,
    )

def _section_232_service_for_reset() -> MetalCompositionService:
    source_df = pd.DataFrame([{"source_row_id": 1, "normalized_product_code": "demo-1"}])
    prepared_df = pd.DataFrame([{"source_row_id": 1}])
    return MetalCompositionService(
        serving_store=WorkbookStore(source_df=source_df, prepared_df=prepared_df),
        workflow_runner=SimpleNamespace(),
        settings=_sample_settings(),
        ui_state_store=InMemoryMetalCompositionUIStateStore(),
        section_232_source_store=InMemorySection232SourceStore(),
        section_232_ruleset_store=InMemorySection232RulesetStore(),
    )

def test_extract_section_232_pdf_finds_expected_8483_mentions():
    fixture_path = ROOT / "data" / "section232_eligible" / "2025-15819.pdf"

    extracted = extract_section_232_pdf(fixture_path.read_bytes())

    assert extracted.page_count == 4
    assert extracted.extraction_status in {"completed", "partial"}
    assert "8483.30.80" in extracted.hts_mentions
    assert "8483.90.30" in extracted.hts_mentions

def test_extract_section_232_pdf_preserves_10_digit_leaf_mentions():
    fixture_path = (
        ROOT
        / "data"
        / "section232_eligible"
        / (
            "Implementation of Duties on Aluminum Pursuant to Proclamation 10895 "
            "Adjusting Imports of Aluminum Into the United States.pdf"
        )
    )

    extracted = extract_section_232_pdf(fixture_path.read_bytes())

    assert extracted.page_count >= 1
    assert extracted.extraction_status in {"completed", "partial"}
    assert "8536.90.8585" in extracted.hts_mentions

def test_extract_hts_mentions_accepts_contextual_four_digit_headings():
    text = (
        "Section 232 aluminum derivatives under HTS heading 7608 remain covered. "
        "The listed headings 7604, 7605, and 7608 are included."
    )

    assert extract_hts_mentions(text) == ["7608", "7604", "7605"]

def test_extract_hts_mentions_ignores_unlabeled_four_digit_numbers():
    text = "The notice was published in 2025 and references page 7608 of the docket."

    assert extract_hts_mentions(text) == []

def test_extract_section_232_pdf_page_payload_includes_layout_text_and_excerpt():
    fixture_path = ROOT / "data" / "section232_eligible" / "2025-15819.pdf"

    extracted = extract_section_232_pdf(fixture_path.read_bytes())

    populated_page = next(page for page in extracted.page_texts if page.get("text"))

    assert populated_page["page_number"] >= 1
    assert isinstance(populated_page["layout_aware_text"], str)
    assert populated_page["layout_aware_text"]
    assert isinstance(populated_page["page_excerpt"], str)
    assert populated_page["page_excerpt"]
    assert isinstance(populated_page["char_count"], int)
    assert populated_page["char_count"] >= len(populated_page["page_excerpt"])

def test_extract_section_232_pdf_page_payload_includes_occurrence_context_for_review():
    fixture_path = ROOT / "data" / "section232_eligible" / "2025-15819.pdf"

    extracted = extract_section_232_pdf(fixture_path.read_bytes())

    populated_page = next(page for page in extracted.page_texts if page.get("hts_occurrences"))
    docket_match = next(
        occurrence
        for occurrence in populated_page["hts_occurrences"]
        if occurrence.get("normalized_hts_code") == "2408.14"
    )
    footer_match = next(
        occurrence
        for occurrence in extracted.page_texts[0]["hts_occurrences"]
        if occurrence.get("normalized_hts_code") == "2650.01"
    )

    assert docket_match["matched_text"] == "240814"
    assert "Docket No." in docket_match["context_text"]
    assert docket_match["text_sources"]
    assert footer_match["matched_text"] == "265001"
    assert "Jkt 265001" in footer_match["context_text"]

def test_extract_section_232_pdf_preserves_plain_text_when_layout_extraction_fails(monkeypatch):
    class _FakePage:
        pass

    class _FakeReader:
        pages = [_FakePage()]

    def _fake_extract_page_text(page, *, layout_aware=False):
        if layout_aware:
            raise RuntimeError("layout extraction unavailable")
        return "Plain Section 232 notice text for HTS 7308.90.95."

    monkeypatch.setattr(section_232_sources_module, "PdfReader", lambda _: _FakeReader())
    monkeypatch.setattr(section_232_sources_module, "_extract_page_text", _fake_extract_page_text)

    extracted = extract_section_232_pdf(b"%PDF-1.4 test payload")

    assert extracted.extraction_status == "partial"
    assert extracted.full_text == "[Page 1]\nPlain Section 232 notice text for HTS 7308.90.95."
    assert extracted.page_texts[0]["text"] == "Plain Section 232 notice text for HTS 7308.90.95."
    assert extracted.page_texts[0]["layout_aware_text"] == "Plain Section 232 notice text for HTS 7308.90.95."
    assert extracted.warnings == ["Failed to extract layout-aware text for page 1: layout extraction unavailable"]

def test_section_232_retrieval_matches_10_digit_candidate_to_8_digit_family():
    fixture_path = ROOT / "data" / "section232_eligible" / "2025-15819.pdf"
    extracted = extract_section_232_pdf(fixture_path.read_bytes())
    store = InMemorySection232SourceStore()
    store.save_source(
        filename=fixture_path.name,
        size_bytes=fixture_path.stat().st_size,
        extracted=extracted,
    )

    snippets = store.retrieve_snippets(
        hts_codes=["8483.30.8020"],
        metal_keywords=["steel", "aluminum"],
        settings=_sample_settings(),
    )

    assert snippets
    assert any("8483.30.80" in snippet.matched_codes for snippet in snippets)
    assert any(snippet.filename == "2025-15819.pdf" for snippet in snippets)

def test_section_232_retrieval_matches_copper_keywords():
    store = InMemorySection232SourceStore()
    store.save_source(
        filename="copper-notice.pdf",
        size_bytes=123,
        extracted=ExtractedSection232Source(
            page_count=1,
            extraction_status="completed",
            full_text="[Page 1]\nSection 232 copper derivative coverage for HTS 7407.10.50.",
            page_texts=[
                {
                    "page_number": 1,
                    "text": "Section 232 copper derivative coverage for HTS 7407.10.50.",
                    "hts_mentions": ["7407.10.50"],
                }
            ],
            hts_mentions=["7407.10.50"],
            warnings=[],
        ),
    )

    snippets = store.retrieve_snippets(
        hts_codes=["7407.10.5050"],
        metal_keywords=["copper"],
        settings=_sample_settings(),
    )

    assert snippets
    assert any("7407.10.50" in snippet.matched_codes for snippet in snippets)
    assert snippets[0].filename == "copper-notice.pdf"

def test_section_232_retrieval_falls_back_to_layout_text_when_plain_text_is_blank():
    store = InMemorySection232SourceStore()
    store.save_source(
        filename="layout-only-notice.pdf",
        size_bytes=123,
        extracted=ExtractedSection232Source(
            page_count=1,
            extraction_status="completed",
            full_text="[Page 1]\nSection 232 steel derivatives HTS 7308.90.95 additional duties apply.",
            page_texts=[
                {
                    "page_number": 1,
                    "text": "",
                    "layout_aware_text": "Section 232 steel derivatives HTS 7308.90.95 additional duties apply.",
                    "page_excerpt": "Section 232 steel derivatives HTS 7308.90.95 additional duties apply.",
                    "char_count": 68,
                    "hts_mentions": ["7308.90.95"],
                }
            ],
            hts_mentions=["7308.90.95"],
            warnings=[],
        ),
    )

    snippets = store.retrieve_snippets(
        hts_codes=["7308.90.9500"],
        metal_keywords=["steel"],
        settings=_sample_settings(),
    )

    assert snippets
    assert snippets[0].text == "Section 232 steel derivatives HTS 7308.90.95 additional duties apply."
    assert snippets[0].matched_codes == ["7308.90.95"]

def test_in_memory_section_232_store_preserves_rich_page_payload_fields():
    store = InMemorySection232SourceStore()
    saved = store.save_source(
        filename="notice.pdf",
        size_bytes=123,
        extracted=ExtractedSection232Source(
            page_count=1,
            extraction_status="completed",
            full_text="[Page 1]\nSection 232 notice text.",
            page_texts=[
                {
                    "page_number": 1,
                    "text": "Section 232 notice text.",
                    "layout_aware_text": "Section 232 notice text.",
                    "page_excerpt": "Section 232 notice text.",
                    "char_count": 24,
                    "hts_mentions": ["7308.90.95"],
                }
            ],
            hts_mentions=["7308.90.95"],
            warnings=[],
        ),
    )

    assert saved.page_texts[0]["layout_aware_text"] == "Section 232 notice text."
    assert saved.page_texts[0]["page_excerpt"] == "Section 232 notice text."
    assert saved.page_texts[0]["char_count"] == 24

def test_in_memory_section_232_store_can_append_and_replace_eligible_hts_codes():
    store = InMemorySection232SourceStore()

    appended = store.append_eligible_hts_codes(["8483.30.80", "8483.90.30", "8483.30.80"])
    assert appended == ["8483.30.80", "8483.90.30"]

    replaced = store.replace_eligible_hts_codes(["7616.99.51.60", "7308.90.95"])
    assert replaced == ["7308.90.95", "7616.99.5160"]

def test_in_memory_section_232_store_can_clear_eligible_hts_codes():
    store = InMemorySection232SourceStore()
    store.append_eligible_hts_codes(["8483.30.80", "7308.90.95"])

    store.clear_eligible_hts_codes()

    assert store.list_eligible_hts_codes() == []

def test_reset_section_232_data_clears_legacy_eligible_hts_codes_too():
    service = _section_232_service_for_reset()
    service.section_232_source_store.save_source(
        filename="legacy-notice.pdf",
        size_bytes=123,
        extracted=ExtractedSection232Source(
            page_count=1,
            extraction_status="completed",
            full_text="[Page 1]\nLegacy Section 232 notice text.",
            page_texts=[
                {
                    "page_number": 1,
                    "text": "Legacy Section 232 notice text.",
                    "layout_aware_text": "Legacy Section 232 notice text.",
                    "page_excerpt": "Legacy Section 232 notice text.",
                    "char_count": 31,
                    "hts_mentions": ["8483.30.80"],
                }
            ],
            hts_mentions=["8483.30.80"],
            warnings=[],
        ),
    )
    service.section_232_source_store.append_eligible_hts_codes(["8483.30.80"])

    reset_response = service.reset_section_232_data()

    assert reset_response.cleared_source_count == 1
    assert service.section_232_source_store.list_sources() == []
    assert service.section_232_source_store.list_eligible_hts_codes() == []
    assert service.list_section_232_eligible_hts_codes().total == 0
