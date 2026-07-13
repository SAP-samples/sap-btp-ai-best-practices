from __future__ import annotations

import json
import threading
import time
import types
from pathlib import Path

import fitz

from app.services.metal_composition.config import MetalCompositionSettings
from app.services.metal_composition.workflow.diagrams import analyze_diagrams
from app.services.metal_composition.workflow.page_routing import PageRouteDecision
from app.services.metal_composition.workflow.page_routing import PdfNativeSignals
from app.services.metal_composition.workflow.page_routing import PreviewSignals
from app.services.metal_composition.workflow.page_routing import RouteScores
from app.services.metal_composition.workflow.page_routing import _build_page_routing_prompt
from app.services.metal_composition.workflow.page_routing import build_mixed_diagram_batches
from app.services.metal_composition.workflow.page_routing import materialize_diagram_sources
from app.services.metal_composition.workflow.token_usage import TokenUsageRecorder
from app.services.metal_composition.workflow.types import DiagramPayload
from app.services.metal_composition.workflow.types import RenderedDiagramPage
from app.services.metal_composition.workflow.types import RenderedDiagramTextPage


ROOT = Path(__file__).resolve().parents[2]


def _sample_settings(**overrides) -> MetalCompositionSettings:
    defaults = dict(
        diagram_model_name="diagram-model",
        diagram_page_routing_enabled=True,
        diagram_page_routing_fallback_model_name="gpt-4o-mini",
        diagram_page_routing_skip_enabled=True,
        max_diagram_text_chars_per_batch=600,
        max_diagram_text_chars_per_page_chunk=300,
    )
    defaults.update(overrides)
    return MetalCompositionSettings(
        workbook_path=ROOT / "data" / "Material Master.xlsb",
        api_env_path=ROOT / "api" / ".env",
        **defaults,
    )


def _load_repo_pdf_bytes(relative_path: str) -> bytes:
    return (ROOT / relative_path).read_bytes()


def _build_test_pdf_bytes(page_count: int) -> bytes:
    document = fitz.open()
    for index in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), f"Test page {index + 1}")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def _test_native_signals(page_number: int) -> PdfNativeSignals:
    return PdfNativeSignals(
        page_number=page_number,
        extracted_chars=10,
        non_whitespace_chars=10,
        word_count=2,
        preview=f"preview {page_number}",
        paragraph_block_count=0,
        paragraph_text_chars=0,
        text_block_coverage_ratio=0.0,
        sentence_density=0.0,
        uppercase_ratio=0.0,
        drawing_keyword_hits=0,
        standards_or_manual_keyword_hits=0,
        border_coordinate_grid_detected=False,
        title_block_detected=False,
        bom_or_revision_table_detected=False,
        drawing_path_count=0,
        embedded_image_count=0,
        embedded_image_area_ratio=0.0,
    )


class _RoutingOnlyLLM:
    def __init__(self, *, route_by_page: dict[int, str]) -> None:
        self.route_by_page = route_by_page
        self.calls: list[dict] = []

    def invoke_native_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        content_blocks = kwargs["messages"][0]["content"]
        decisions = []
        page_blocks = [
            block["text"]
            for block in content_blocks
            if block.get("type") == "text" and block["text"].startswith("Page ref:")
        ]
        if page_blocks:
            for block_text in page_blocks:
                lines = block_text.splitlines()
                page_ref = lines[0].split("Page ref: ", 1)[1].strip()
                page_number = int(lines[2].split("Page number: ", 1)[1].strip())
                decisions.append(
                    {
                        "page_ref": page_ref,
                        "final_route": self.route_by_page[page_number],
                        "confidence": 0.95,
                        "reason": f"resolved page {page_number}",
                    }
                )
            content = json.dumps({"decisions": decisions})
        else:
            source_text = next(
                block["text"]
                for block in content_blocks
                if block.get("type") == "text" and block["text"].startswith("Source file:")
            )
            page_number = int(source_text.split("Page number: ", 1)[1].splitlines()[0])
            final_route = self.route_by_page[page_number]
            content = json.dumps(
                {
                    "final_route": final_route,
                    "confidence": 0.95,
                    "reason": f"resolved page {page_number}",
                }
            )
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=content
                    )
                )
            ]
        )


class _BatchRoutingLLM:
    def __init__(self, *, route_by_page: dict[int, str]) -> None:
        self.route_by_page = route_by_page
        self.calls: list[dict] = []

    def invoke_native_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        decisions = []
        for block in kwargs["messages"][0]["content"]:
            if block.get("type") != "text":
                continue
            text = block.get("text", "")
            if not text.startswith("Page ref: "):
                continue
            lines = text.splitlines()
            page_ref = lines[0].split("Page ref: ", 1)[1].strip()
            page_number = int(lines[2].split("Page number: ", 1)[1].strip())
            decisions.append(
                {
                    "page_ref": page_ref,
                    "final_route": self.route_by_page[page_number],
                    "confidence": 0.95,
                    "reason": f"resolved page {page_number}",
                }
            )
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps({"decisions": decisions})
                    )
                )
            ]
        )


def test_materialize_diagram_sources_routes_m59104_cover_and_drawing():
    llm = _RoutingOnlyLLM(route_by_page={1: "skip", 2: "image_analysis"})
    payload = DiagramPayload(
        filename="m59104.pdf",
        content_type="application/pdf",
        data=_load_repo_pdf_bytes(
            "data/new_data/OneDrive_1_3-18-2026/AW Parts/103258LF/103258LF_2D/m59104.pdf"
        ),
        source_filename="m59104.pdf",
    )

    materialized = materialize_diagram_sources([payload], _sample_settings(), llm)

    assert len(materialized.image_pages) == 1
    assert len(materialized.text_pages) == 0
    assert materialized.routing_summary["skip_pages"] == 1
    assert materialized.routing_summary["image_analysis_pages"] == 1
    assert materialized.routing_summary["ambiguous_pages"] == 0
    assert materialized.routing_summary["llm_fallback_attempted_pages"] == 2
    assert materialized.routing_summary["llm_fallback_resolved_pages"] == 2
    assert materialized.image_pages[0].source_filename == "m59104.pdf"
    assert materialized.image_pages[0].page_number == 2


def test_materialize_diagram_sources_routes_p58671_manual_pages_to_text():
    llm = _RoutingOnlyLLM(route_by_page={6: "text_analysis", 7: "text_analysis", 12: "text_analysis", 13: "text_analysis", 18: "text_analysis"})
    payload = DiagramPayload(
        filename="P58671_rev_11.pdf",
        content_type="application/pdf",
        data=_load_repo_pdf_bytes(
            "data/new_data/OneDrive_1_3-18-2026/AW Parts/103258LF/103258LF_2D/P58671_rev_11.pdf"
        ),
        source_filename="P58671_rev_11.pdf",
    )

    materialized = materialize_diagram_sources([payload], _sample_settings(), llm)

    assert len(materialized.image_pages) == 0
    assert len(materialized.text_pages) == 18
    assert materialized.routing_summary["text_analysis_pages"] == 18
    assert materialized.routing_summary["ambiguous_pages"] == 0
    assert materialized.routing_summary["llm_fallback_attempted_pages"] == 5
    assert materialized.routing_summary["llm_fallback_resolved_pages"] == 5
    assert all(page.source_filename == "P58671_rev_11.pdf" for page in materialized.text_pages)


def test_materialize_diagram_sources_passes_usage_recorder_to_fallback_llm():
    llm = _RoutingOnlyLLM(route_by_page={1: "skip", 2: "image_analysis"})
    payload = DiagramPayload(
        filename="m59104.pdf",
        content_type="application/pdf",
        data=_load_repo_pdf_bytes(
            "data/new_data/OneDrive_1_3-18-2026/AW Parts/103258LF/103258LF_2D/m59104.pdf"
        ),
        source_filename="m59104.pdf",
    )
    usage_recorder = TokenUsageRecorder()

    materialize_diagram_sources(
        [payload],
        _sample_settings(),
        llm,
        usage_recorder=usage_recorder,
    )

    assert llm.calls
    assert all(call["phase"] == "diagram" for call in llm.calls)
    assert all(call["task"] == "page_routing_fallback" for call in llm.calls)
    assert all(call["usage_recorder"] is usage_recorder for call in llm.calls)


def test_materialize_diagram_sources_batches_fallback_requests_to_15_images(monkeypatch):
    llm = _BatchRoutingLLM(route_by_page={index: "image_analysis" for index in range(1, 17)})
    payload = DiagramPayload(
        filename="batched.pdf",
        content_type="application/pdf",
        data=_build_test_pdf_bytes(16),
        source_filename="batched.pdf",
    )

    monkeypatch.setattr(
        "app.services.metal_composition.workflow.page_routing.extract_pdf_native_signals",
        lambda page: _test_native_signals(page.number + 1),
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.page_routing.extract_preview_signals",
        lambda page, dpi: PreviewSignals(
            whitespace_ratio=0.5,
            ink_ratio=0.5,
            edge_density=0.1,
            line_art_density=0.1,
            lower_band_density=0.1,
            large_raster_coverage_ratio=0.1,
        ),
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.page_routing.route_page",
        lambda native, preview: PageRouteDecision(
            final_route="image_analysis",
            override_name=None,
            is_ambiguous=True,
            used_preview=preview is not None,
            scores=RouteScores(image_score=0.1, text_score=0.1, skip_score=0.0),
        ),
    )

    materialized = materialize_diagram_sources([payload], _sample_settings(), llm)

    assert len(llm.calls) == 2
    assert [sum(1 for block in call["messages"][0]["content"] if block.get("type") == "image_url") for call in llm.calls] == [15, 1]
    assert materialized.routing_summary["llm_fallback_attempted_pages"] == 16
    assert materialized.routing_summary["llm_fallback_resolved_pages"] == 16
    assert len(materialized.image_pages) == 16


def test_materialize_diagram_sources_maps_batched_fallback_routes_back_to_pages(monkeypatch):
    llm = _BatchRoutingLLM(route_by_page={1: "skip", 2: "text_analysis", 3: "image_analysis"})
    payload = DiagramPayload(
        filename="mixed.pdf",
        content_type="application/pdf",
        data=_build_test_pdf_bytes(3),
        source_filename="mixed.pdf",
    )

    monkeypatch.setattr(
        "app.services.metal_composition.workflow.page_routing.extract_pdf_native_signals",
        lambda page: _test_native_signals(page.number + 1),
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.page_routing.extract_preview_signals",
        lambda page, dpi: PreviewSignals(
            whitespace_ratio=0.5,
            ink_ratio=0.5,
            edge_density=0.1,
            line_art_density=0.1,
            lower_band_density=0.1,
            large_raster_coverage_ratio=0.1,
        ),
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.page_routing.route_page",
        lambda native, preview: PageRouteDecision(
            final_route="image_analysis",
            override_name=None,
            is_ambiguous=True,
            used_preview=preview is not None,
            scores=RouteScores(image_score=0.1, text_score=0.1, skip_score=0.0),
        ),
    )

    materialized = materialize_diagram_sources([payload], _sample_settings(), llm)

    assert materialized.routing_summary["skip_pages"] == 1
    assert materialized.routing_summary["text_analysis_pages"] == 1
    assert materialized.routing_summary["image_analysis_pages"] == 1
    assert [page["final_route"] for page in materialized.page_decisions] == [
        "skip",
        "text_analysis",
        "image_analysis",
    ]
    assert len(materialized.text_pages) == 1
    assert len(materialized.image_pages) == 1


def test_materialize_diagram_sources_uses_configured_fallback_render_dpi(monkeypatch):
    llm = _BatchRoutingLLM(route_by_page={1: "image_analysis"})
    payload = DiagramPayload(
        filename="dpi.pdf",
        content_type="application/pdf",
        data=_build_test_pdf_bytes(1),
        source_filename="dpi.pdf",
    )
    captured_dpis: list[int] = []

    monkeypatch.setattr(
        "app.services.metal_composition.workflow.page_routing.extract_pdf_native_signals",
        lambda page: _test_native_signals(page.number + 1),
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.page_routing.extract_preview_signals",
        lambda page, dpi: PreviewSignals(
            whitespace_ratio=0.5,
            ink_ratio=0.5,
            edge_density=0.1,
            line_art_density=0.1,
            lower_band_density=0.1,
            large_raster_coverage_ratio=0.1,
        ),
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.page_routing.route_page",
        lambda native, preview: PageRouteDecision(
            final_route="image_analysis",
            override_name=None,
            is_ambiguous=True,
            used_preview=preview is not None,
            scores=RouteScores(image_score=0.1, text_score=0.1, skip_score=0.0),
        ),
    )
    monkeypatch.setattr(
        "app.services.metal_composition.workflow.page_routing._render_page_png_bytes",
        lambda page, *, dpi: captured_dpis.append(dpi) or b"png-bytes",
    )

    materialize_diagram_sources(
        [payload],
        _sample_settings(diagram_page_routing_fallback_render_dpi=100),
        llm,
    )

    assert captured_dpis == [100]


def test_build_mixed_diagram_batches_chunks_text_when_needed():
    image_page = RenderedDiagramPage(
        page_ref="P1",
        source_document_index=0,
        filename="drawing.png",
        content_type="image/png",
        data=b"png",
        source_filename="drawing.pdf",
        page_number=1,
        rendered_width=32,
        rendered_height=32,
    )
    text_page = RenderedDiagramTextPage(
        page_ref="P2",
        source_document_index=0,
        source_filename="manual.pdf",
        page_number=2,
        text=("A" * 650),
        chunk_index=1,
        chunk_count=1,
        char_count=650,
    )

    batches = build_mixed_diagram_batches(
        image_pages=[image_page],
        text_pages=[text_page],
        settings=_sample_settings(max_diagram_text_chars_per_batch=300, max_diagram_text_chars_per_page_chunk=250),
    )

    assert len(batches) >= 3
    assert sum(1 for batch in batches for entry in batch if entry.kind == "text") == 3


def test_build_mixed_diagram_batches_caps_image_batches_to_provider_limit():
    image_pages = [
        RenderedDiagramPage(
            page_ref=f"P{index + 1}",
            sequence_index=index,
            source_document_index=0,
            filename=f"drawing-{index + 1}.png",
            content_type="image/png",
            data=b"png",
            source_filename=f"drawing-{index + 1}.pdf",
            page_number=index + 1,
            rendered_width=32,
            rendered_height=32,
        )
        for index in range(51)
    ]

    batches = build_mixed_diagram_batches(
        image_pages=image_pages,
        text_pages=[],
        settings=_sample_settings(max_diagram_images=500),
    )

    assert len(batches) == 2
    assert [sum(1 for entry in batch if entry.kind == "image") for batch in batches] == [50, 1]


def test_build_mixed_diagram_batches_respects_configured_image_cap_when_lower():
    image_pages = [
        RenderedDiagramPage(
            page_ref=f"P{index + 1}",
            sequence_index=index,
            source_document_index=0,
            filename=f"drawing-{index + 1}.png",
            content_type="image/png",
            data=b"png",
            source_filename=f"drawing-{index + 1}.pdf",
            page_number=index + 1,
            rendered_width=32,
            rendered_height=32,
        )
        for index in range(21)
    ]

    batches = build_mixed_diagram_batches(
        image_pages=image_pages,
        text_pages=[],
        settings=_sample_settings(max_diagram_images=10),
    )

    assert len(batches) == 3
    assert [sum(1 for entry in batch if entry.kind == "image") for batch in batches] == [10, 10, 1]


def test_page_routing_prompt_biases_covers_to_skip_and_uncertain_pages_to_image():
    prompt = _build_page_routing_prompt(
        source_filename="m59104.pdf",
        page_number=1,
        native_signals={"word_count": 3},
        preview_signals={"edge_density": 0.1},
    )

    assert "skip: only for explicit cover/separator pages" in prompt
    assert "If uncertain between skip and image_analysis, choose image_analysis." in prompt
    assert "text_analysis: useful text-heavy manual, standard, specification" in prompt


def test_analyze_diagrams_builds_mixed_prompt_and_omits_skipped_pages(monkeypatch):
    captured: dict = {}

    def fake_materialize(_payloads, _settings, _llm, usage_recorder=None):
        image_page = RenderedDiagramPage(
            page_ref="P2",
            source_document_index=0,
            filename="drawing.png",
            content_type="image/png",
            data=b"png",
            source_filename="drawing.pdf",
            page_number=2,
            rendered_width=32,
            rendered_height=32,
        )
        text_page = RenderedDiagramTextPage(
            page_ref="P3",
            source_document_index=0,
            source_filename="manual.pdf",
            page_number=3,
            text="Material Standard\nScope: stainless steel body",
            chunk_index=1,
            chunk_count=1,
            char_count=45,
        )
        return types.SimpleNamespace(
            image_pages=[image_page],
            text_pages=[text_page],
            preprocess_details_list=[],
            routing_summary={
                "page_count": 3,
                "image_analysis_pages": 1,
                "text_analysis_pages": 1,
                "skip_pages": 1,
                "ambiguous_pages": 0,
                "llm_fallback_attempted_pages": 0,
                "llm_fallback_resolved_pages": 0,
                "image_pages_rendered": 1,
                "text_chars_sent": 45,
            },
            page_decisions=[
                {"page_ref": "P1", "final_route": "skip", "page_number": 1},
                {"page_ref": "P2", "final_route": "image_analysis", "page_number": 2},
                {"page_ref": "P3", "final_route": "text_analysis", "page_number": 3},
            ],
        )

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='{"status":"completed","extracted_codes":[],"is_likely_metal_item":true,"estimated_metal_share":1.0,"material_cues":[],"uncertainty_notes":[]}'
                    )
                )
            ]
        )

    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.materialize_diagram_sources",
        fake_materialize,
    )

    llm = types.SimpleNamespace(invoke_native_chat_completion=fake_invoke)
    result = analyze_diagrams(
        [DiagramPayload(filename="drawing.pdf", content_type="application/pdf", data=b"%PDF-1.4 fake")],
        _sample_settings(),
        llm,
    )

    content = captured["messages"][0]["content"]
    text_blocks = [block["text"] for block in content if block.get("type") == "text"]
    assert any("Source label: P2 | drawing.pdf page 2" in text for text in text_blocks)
    assert any("Source text: P3 | manual.pdf page 3" in text for text in text_blocks)
    assert not any(
        text.startswith("Source label: P1 |") or text.startswith("Source text: P1 |")
        for text in text_blocks
    )
    assert result["timing"]["details"]["routing"]["skip_pages"] == 1


def test_analyze_diagrams_keeps_zoom_disabled_for_text_only_inputs(monkeypatch):
    def fake_materialize(_payloads, _settings, _llm, usage_recorder=None):
        text_page = RenderedDiagramTextPage(
            page_ref="P1",
            source_document_index=0,
            source_filename="manual.pdf",
            page_number=1,
            text="Material Standard\nRequirements: ASTM A351 stainless steel body.",
            chunk_index=1,
            chunk_count=1,
            char_count=63,
        )
        return types.SimpleNamespace(
            image_pages=[],
            text_pages=[text_page],
            preprocess_details_list=[],
            routing_summary={
                "page_count": 1,
                "image_analysis_pages": 0,
                "text_analysis_pages": 1,
                "skip_pages": 0,
                "ambiguous_pages": 0,
                "llm_fallback_attempted_pages": 0,
                "llm_fallback_resolved_pages": 0,
                "image_pages_rendered": 0,
                "text_chars_sent": 63,
            },
            page_decisions=[{"page_ref": "P1", "final_route": "text_analysis", "page_number": 1}],
        )

    calls: list[dict] = []

    def fake_invoke(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='{"status":"completed","extracted_codes":[],"is_likely_metal_item":true,"estimated_metal_share":1.0,"material_cues":[],"uncertainty_notes":[]}'
                    )
                )
            ]
        )

    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.materialize_diagram_sources",
        fake_materialize,
    )

    llm = types.SimpleNamespace(invoke_native_chat_completion=fake_invoke)
    result = analyze_diagrams(
        [DiagramPayload(filename="manual.pdf", content_type="application/pdf", data=b"%PDF-1.4 fake")],
        _sample_settings(),
        llm,
    )

    assert len(calls) == 1
    assert result["zoom_diagnostics"]["triggered"] is False
    assert result["zoom_diagnostics"]["refinement_applied"] is False


def test_analyze_diagrams_normalizes_list_shaped_composition(monkeypatch):
    def fake_materialize(_payloads, _settings, _llm, usage_recorder=None):
        image_page = RenderedDiagramPage(
            page_ref="P1",
            source_document_index=0,
            filename="drawing.png",
            content_type="image/png",
            data=b"png",
            source_filename="drawing.pdf",
            page_number=1,
            rendered_width=32,
            rendered_height=32,
        )
        return types.SimpleNamespace(
            image_pages=[image_page],
            text_pages=[],
            preprocess_details_list=[],
            routing_summary={
                "page_count": 1,
                "image_analysis_pages": 1,
                "text_analysis_pages": 0,
                "skip_pages": 0,
                "ambiguous_pages": 0,
                "llm_fallback_attempted_pages": 0,
                "llm_fallback_resolved_pages": 0,
                "image_pages_rendered": 1,
                "text_chars_sent": 0,
            },
            page_decisions=[{"page_ref": "P1", "final_route": "image_analysis", "page_number": 1}],
        )

    def fake_invoke(**kwargs):
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
                                "material_cues": [],
                                "uncertainty_notes": [],
                                "composition": [
                                    {
                                        "is_metal_item": True,
                                        "total_weight_grams": 93.0,
                                        "estimated_total_metal_grams": 93.0,
                                        "top_level_grams": [
                                            {"key": "steel", "value": 93.0},
                                            {"key": "aluminum", "value": 0.0},
                                        ],
                                        "steel_subtype_grams": [],
                                        "confidence": 0.98,
                                        "reasoning": "Returned as a list.",
                                    }
                                ],
                            }
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.materialize_diagram_sources",
        fake_materialize,
    )

    llm = types.SimpleNamespace(invoke_native_chat_completion=fake_invoke)
    result = analyze_diagrams(
        [DiagramPayload(filename="drawing.pdf", content_type="application/pdf", data=b"%PDF-1.4 fake")],
        _sample_settings(),
        llm,
        source_summary={"total_weight_gram": 93.0},
    )

    composition = result["composition"]
    assert composition["top_level_grams"]["steel"] == 93.0
    assert composition["top_level_grams"]["aluminum"] == 0.0
    assert composition["provenance"]["notes"] == [
        "Composition payload was normalized from a list-shaped model response."
    ]
    assert "normalized from a list-shaped model response" in composition["reasoning"]


def test_analyze_diagrams_runs_base_batches_in_parallel(monkeypatch):
    def fake_materialize(_payloads, _settings, _llm, usage_recorder=None):
        image_pages = [
            RenderedDiagramPage(
                page_ref=f"P{index + 1}",
                sequence_index=index,
                source_document_index=0,
                filename=f"drawing-{index + 1}.png",
                content_type="image/png",
                data=b"png",
                source_filename=f"drawing-{index + 1}.pdf",
                page_number=index + 1,
                rendered_width=32,
                rendered_height=32,
            )
            for index in range(2)
        ]
        return types.SimpleNamespace(
            image_pages=image_pages,
            text_pages=[],
            preprocess_details_list=[],
            routing_summary={
                "page_count": 2,
                "image_analysis_pages": 2,
                "text_analysis_pages": 0,
                "skip_pages": 0,
                "ambiguous_pages": 0,
                "llm_fallback_attempted_pages": 0,
                "llm_fallback_resolved_pages": 0,
                "image_pages_rendered": 2,
                "text_chars_sent": 0,
            },
            page_decisions=[
                {"page_ref": "P1", "final_route": "image_analysis", "page_number": 1},
                {"page_ref": "P2", "final_route": "image_analysis", "page_number": 2},
            ],
        )

    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def fake_invoke(**kwargs):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            time.sleep(0.05)
            source_label = next(
                block["text"]
                for block in kwargs["messages"][0]["content"]
                if block.get("type") == "text" and block["text"].startswith("Source label:")
            )
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
                                    "material_cues": [source_label],
                                    "uncertainty_notes": [],
                                }
                            )
                        )
                    )
                ]
            )
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(
        "app.services.metal_composition.workflow.diagrams.materialize_diagram_sources",
        fake_materialize,
    )

    llm = types.SimpleNamespace(invoke_native_chat_completion=fake_invoke)
    result = analyze_diagrams(
        [DiagramPayload(filename="drawing.pdf", content_type="application/pdf", data=b"%PDF-1.4 fake")],
        _sample_settings(max_diagram_images=1, diagram_batch_max_concurrency=2, diagram_zoom_enabled=False),
        llm,
    )

    assert state["max_active"] >= 2
    assert result["status"] == "completed"
    assert len(result["material_cues"]) == 2
