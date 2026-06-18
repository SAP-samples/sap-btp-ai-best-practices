from __future__ import annotations

from dataclasses import replace as dataclass_replace
from datetime import date
from io import BytesIO
import json
import os
import types
from pathlib import Path

import fitz
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.models.metal_composition import (
    FinalMetalComposition,
    MetalCompositionAppSettings,
    MetalCompositionResponse,
    TokenUsageSummary,
)
from app.routers.metal_composition import router as metal_composition_router
from app.services.metal_composition.classification_jobs import SUPERSEDED_ERROR_MESSAGE
from app.services.metal_composition.config import MetalCompositionSettings
from app.services.metal_composition.hts_catalog import (
    HanaHTSCatalogResolver,
    compile_hts_catalog_frame,
    compile_hts_code_map_frame,
)
from app.services.metal_composition.hts_catalog_sources import InMemoryHTSCatalogSourceStore
from app.services.metal_composition.section_232_rulesets import Section232DraftRuleCandidate
from app.services.metal_composition.section_232_sources import InMemorySection232SourceStore
from app.services.metal_composition.section_232_sources import ExtractedSection232Source
from app.services.metal_composition.service import MetalCompositionService
from app.services.metal_composition.workflow.trade_decision import finalize_trade_decision
from app.services.metal_composition.ui_state import (
    InMemoryMetalCompositionUIStateStore,
    StoredDocumentReference,
)
from app.services.metal_composition.workflow import DiagramPayload
from app.services.metal_composition.workbook import WorkbookStore
from app.services.metal_composition import get_metal_composition_service

ROOT = Path(__file__).resolve().parents[2]
HTS_CSV_DIR = ROOT / "data" / "hts_chapters"
HTS_CODE_MAP = HTS_CSV_DIR / "hts_code_map.csv"

def _build_test_catalog_resolver() -> HanaHTSCatalogResolver:
    return HanaHTSCatalogResolver(
        settings=MetalCompositionSettings(
            workbook_path=Path("/tmp/unused.xlsb"),
            api_env_path=Path("/tmp/api.env"),
            hts_catalog_dir=HTS_CSV_DIR,
            hts_code_map_path=HTS_CODE_MAP,
        ),
        catalog_frame=compile_hts_catalog_frame(csv_dir=HTS_CSV_DIR),
        code_map_frame=compile_hts_code_map_frame(code_map_path=HTS_CODE_MAP),
    )

class FakeWorkflowRunner:
    def __init__(self, resolver: HanaHTSCatalogResolver | None = None) -> None:
        self.hts_catalog_resolver = resolver or _build_test_catalog_resolver()
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs["product_code"] == "FAIL":
            raise RuntimeError("workflow exploded")

        diagram_payloads = list(kwargs.get("diagram_payloads") or [])
        total_weight = float(kwargs.get("source_summary", {}).get("total_weight_gram", 0.0) or 0.0)
        hana_tree_search_output = {
            "status": "completed",
            "reasoning": "Mocked HANA tree search found a matching steel valve path.",
            "chapter_candidates": ["84"],
            "heading_candidates": ["8481"],
            "validated_candidates": [
                {
                    "code": "8481.80.30.90",
                    "description": "Other hand operated valves of steel",
                    "confidence": 0.88,
                }
            ],
            "timing": {
                "status": "completed",
                "duration_ms": 98.0,
            },
        }
        final_composition = kwargs.get("material_master_composition") or {
            "is_metal_item": True,
            "total_weight_grams": total_weight,
            "estimated_total_metal_grams": 9.0,
            "top_level_grams": {
                "steel": 7.0,
                "aluminum": 1.0,
                "copper": 1.0,
                "cast_iron": 0.0,
            },
            "steel_subtype_grams": {
                "electrical_steel": 2.0,
                "cold_rolled_coil_steel": 2.0,
                "hot_rolled_coil_steel": 1.0,
                "stainless_steel_304": 1.0,
                "stainless_steel_316": 1.0,
                "stainless_steel_bar": 0.0,
                "duplex_steel": 0.0,
                "cast_steel": 0.0,
            },
            "confidence": 0.81,
            "reasoning": "Mocked final composition.",
        }

        result = {
            "final_composition": final_composition,
            "hts_classification": {
                "status": "completed",
                "best_candidate": {
                    "code": "8481.80.30.90",
                    "description": "Other hand operated valves of steel",
                    "digits": 10,
                    "confidence": 0.88,
                    "reasoning": "Mocked HTS reasoning.",
                    "citations": [
                        {
                            "page_number": 2861,
                            "page_label": "84-116",
                            "chapter_number": 84,
                            "heading_code": "8481",
                        }
                    ],
                    "missing_discriminators": [],
                },
                "candidates": [],
                "confidence": 0.88,
                "reasoning": "Mocked HTS reasoning.",
                "needs_human_review": False,
            },
            "section_232_assessment": {
                "status": "completed",
                "decision": "subject",
                "confidence": 0.72,
                "basis_summary": "Steel valve family likely requires Section 232 review.",
                "needs_human_review": False,
                "weight_rule_applied": False,
                "supporting_hts_candidates": ["8481.80.30.90"],
                "chapter_99_candidates": [],
                "evidence": [{"source": "hana_catalog", "summary": "Mocked HANA catalog support.", "citations": []}],
            },
            "section_232_reasoner_output": {
                "status": "omitted",
                "strategy": "heuristic_fallback",
                "fallback_used": True,
                "reason": "no_source_store",
                "metal_weight_override": {
                    "base_decision": "subject",
                    "final_decision": "subject",
                    "override_applied": False,
                    "threshold": 0.15,
                    "total_weight_grams": total_weight,
                    "affected_metal_weight_grams": 9.0,
                    "affected_metal_share": 0.5 if total_weight else None,
                    "metal_share_certainty": "estimated",
                    "reason": "threshold_not_met",
                    "needs_human_review": False,
                },
            },
            "diagram_output": (
                {
                    "status": "completed",
                    "diagram_count": len(diagram_payloads),
                    "filenames": [payload.filename for payload in diagram_payloads],
                    "mode": kwargs.get("composition_mode", "diagram_manual"),
                }
                if diagram_payloads
                else {"status": "omitted", "diagram_count": 0, "filenames": [], "mode": kwargs.get("composition_mode", "diagram_manual")}
            ),
            "hana_tree_search_output": hana_tree_search_output,
            "hts_resolution_output": {
                "status": "completed",
                "selected_code": "8481.80.30.90",
                "candidate_count": 1,
            },
            "hts_fact_profile": {
                "status": "completed",
                "article_summary": "Manual steel valve",
                "function_summary": "Hand operated valve assembly",
                "material_profile": {
                    "is_metal_item": True,
                    "estimated_total_metal_grams": 9.0,
                    "top_level_grams": {"steel": 7.0, "aluminum": 1.0, "copper": 1.0, "cast_iron": 0.0},
                    "steel_subtype_grams": {},
                    "confidence": 0.81,
                    "reasoning": "Mocked material profile.",
                },
                "diagram_clues": [],
                "heading_hypotheses": ["8481"],
                "discriminator_notes": [],
                "reasoning": "Mocked HTS fact profile.",
                "confidence": 0.83,
                "raw_model_response": "",
                "fallback_reason": "",
            },
            "timing": {
                "phases": {
                    "diagram": {
                        "status": "completed" if diagram_payloads else "omitted",
                        "duration_ms": 60.0 if diagram_payloads else 0.4,
                    },
                    "hana_tree_search": {
                        "status": hana_tree_search_output["status"],
                        "duration_ms": 98.0,
                    },
                    "legal_evidence": {
                        "status": "completed",
                        "duration_ms": 98.0,
                        "details": {
                            "bottleneck": "hana_tree_search",
                            "parallel_wall_clock_duration_ms": 98.0,
                            "estimated_parallel_time_saved_ms": 0.0,
                        },
                    },
                    "parallel_agents": {
                        "status": "completed",
                        "duration_ms": 121.0,
                        "details": {
                            "bottleneck": "diagram",
                            "parallel_wall_clock_duration_ms": 121.0,
                            "estimated_parallel_time_saved_ms": 0.4,
                        },
                    },
                    "hts_fact_profile": {"status": "completed", "duration_ms": 22.0},
                    "trade_decision": {"status": "completed", "duration_ms": 31.0},
                    "workflow": {"status": "completed", "duration_ms": 305.0},
                },
                "summary": {
                    "slowest_parallel_phase": "diagram",
                    "slowest_legal_phase": "hana_tree_search",
                    "critical_path_duration_ms": 361.0,
                },
            },
        }
        if kwargs.get("include_token_usage"):
            result["token_usage"] = TokenUsageSummary(
                entries=[
                    {
                        "phase": "diagram",
                        "task": "base_pass",
                        "model": "gemini-2.5-pro",
                        "call_count": 2,
                        "input_tokens": 120,
                        "output_tokens": 40,
                        "total_tokens": 160,
                        "usage_available": True,
                    },
                    {
                        "phase": "hana_tree_search",
                        "task": "heading_router",
                        "model": "gpt-5",
                        "call_count": 1,
                        "input_tokens": 80,
                        "output_tokens": 20,
                        "total_tokens": 100,
                        "usage_available": True,
                    },
                ],
                input_tokens=200,
                output_tokens=60,
                total_tokens=260,
                missing_usage_entry_count=0,
            ).model_dump(mode="json")
        return result

class PublishedRulesetWorkflowRunner:
    def __init__(self, service: MetalCompositionService, resolver: HanaHTSCatalogResolver | None = None) -> None:
        self.hts_catalog_resolver = resolver or _build_test_catalog_resolver()
        self.section_232_source_store = service._build_workflow_section_232_runtime_store()
        self.llm = types.SimpleNamespace()

    def run(self, **kwargs):
        total_weight = float(kwargs.get("source_summary", {}).get("total_weight_gram", 0.0) or 0.0)
        return finalize_trade_decision(
            state={
                "final_composition": {
                    "is_metal_item": True,
                    "total_weight_grams": total_weight,
                    "estimated_total_metal_grams": 6.0,
                    "top_level_grams": {
                        "steel": 0.0,
                        "aluminum": 0.0,
                        "copper": 6.0,
                        "cast_iron": 0.0,
                    },
                    "steel_subtype_grams": {},
                    "confidence": 1.0,
                    "reasoning": "Synthetic published ruleset workflow composition.",
                },
                "diagram_output": {"metal_share_certainty": "exact"},
                "hts_fact_profile": {"material_profile": {"top_level_grams": {"copper": 6.0}, "confidence": 1.0}},
            },
            candidate_pool={
                "7407.10.5050": {
                    "code": "7407.10.5050",
                    "description": "Copper bars and rods",
                    "digits": 10,
                    "confidence": 0.97,
                    "reasoning": "Synthetic workflow candidate.",
                    "validation_status": "current_exact",
                    "resolution_basis": "Synthetic workflow test candidate.",
                    "citations": [],
                    "origins": ["test"],
                    "missing_discriminators": [],
                }
            },
            resolution_output={"status": "completed", "validated_candidates": [], "rejected_candidates": []},
            ordered_codes=["7407.10.5050"],
            selected_code="7407.10.5050",
            selected_reasoning="Synthetic workflow selection.",
            selected_confidence=0.97,
            needs_human_review_hint=False,
            selector_status="completed",
            settings=MetalCompositionSettings(
                workbook_path=Path("/tmp/unused.xlsb"),
                api_env_path=Path("/tmp/api.env"),
                hts_catalog_dir=HTS_CSV_DIR,
                hts_code_map_path=HTS_CODE_MAP,
            ),
            llm=self.llm,
            resolver=self.hts_catalog_resolver,
            section_232_source_store=self.section_232_source_store,
        )

app = FastAPI()
app.include_router(
    metal_composition_router,
    prefix="/api/metal-composition",
    tags=["metal-composition"],
)

def _build_workbook_store() -> WorkbookStore:
    source_df = pd.DataFrame(
        [
            {
                "source_row_id": 101,
                "normalized_product_code": "dup",
                "Product code": "DUP",
                "PN Revised/ Standardized": "PN-DUP-1",
                "Part description": "Pump housing",
                "New Part Description": "Pump housing rev A",
                "Site": "Madrid",
                "Business Segment": "Water",
                "_parsed_total_weight_gram": 12.0,
                "_date_started": pd.Timestamp("2025-01-02"),
                "_date_completed": pd.Timestamp("2025-01-09"),
            },
            {
                "source_row_id": 102,
                "normalized_product_code": "dup",
                "Product code": "DUP",
                "PN Revised/ Standardized": "PN-DUP-2",
                "Part description": "Motor bracket",
                "New Part Description": "Motor bracket rev B",
                "Site": "Charlotte",
                "Business Segment": "Water",
                "_parsed_total_weight_gram": 11.0,
                "_date_started": pd.Timestamp("2025-02-02"),
                "_date_completed": pd.Timestamp("2025-02-08"),
            },
            {
                "source_row_id": 201,
                "normalized_product_code": "unique",
                "Product code": "UNIQUE",
                "PN Revised/ Standardized": "PN-UNIQUE",
                "Part description": "Valve body",
                "New Part Description": "Valve body rev A",
                "Site": "Milan",
                "Business Segment": "Test & Measurement",
                "_parsed_total_weight_gram": 18.0,
                "_date_started": pd.Timestamp("2025-03-01"),
                "_date_completed": pd.Timestamp("2025-03-03"),
            },
            {
                "source_row_id": 301,
                "normalized_product_code": "fail",
                "Product code": "FAIL",
                "PN Revised/ Standardized": "PN-FAIL",
                "Part description": "Failure row",
                "New Part Description": "Failure row",
                "Site": "Berlin",
                "Business Segment": "Water",
                "_parsed_total_weight_gram": 20.0,
                "_date_started": pd.Timestamp("2025-04-01"),
                "_date_completed": pd.Timestamp("2025-04-05"),
            },
        ]
    )
    prepared_df = pd.DataFrame(
        [
            {"source_row_id": 101, "row_id": 1, "Total Weight (Gram)": 12.0},
            {"source_row_id": 102, "row_id": 2, "Total Weight (Gram)": 11.0},
            {"source_row_id": 201, "row_id": 3, "Total Weight (Gram)": 18.0},
            {"source_row_id": 301, "row_id": 4, "Total Weight (Gram)": 20.0},
        ]
    )
    return WorkbookStore(source_df=source_df, prepared_df=prepared_df)

def _make_service() -> MetalCompositionService:
    settings = MetalCompositionSettings(
        workbook_path=Path("/tmp/unused.xlsb"),
        api_env_path=Path("/tmp/api.env"),
        hts_catalog_dir=HTS_CSV_DIR,
        hts_code_map_path=HTS_CODE_MAP,
    )
    return MetalCompositionService(
        serving_store=_build_workbook_store(),
        workflow_runner=FakeWorkflowRunner(),
        settings=settings,
        ui_state_store=InMemoryMetalCompositionUIStateStore(),
        section_232_source_store=InMemorySection232SourceStore(),
        hts_catalog_source_store=InMemoryHTSCatalogSourceStore(settings),
    )

def _client_with_service(
    service: MetalCompositionService,
    *,
    raise_server_exceptions: bool = True,
) -> TestClient:
    app.dependency_overrides[get_metal_composition_service] = lambda: service
    os.environ["API_KEY"] = "test-api-key"
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)

def _drain_job(client: TestClient, service: MetalCompositionService, job_id: str) -> dict:
    service.drain_classification_jobs()
    response = client.get(
        f"/api/metal-composition/classification-jobs/{job_id}",
        headers={"X-API-Key": "test-api-key"},
    )
    assert response.status_code == 200
    return response.json()

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

def _write_text_pdf(path: Path, *page_texts: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    document = fitz.open()
    try:
        for page_text in page_texts or ("",):
            page = document.new_page(width=595, height=842)
            page.insert_text((72, 72), page_text)
        document.save(path)
    finally:
        document.close()
    return path

def _extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return " ".join(raw_text.split())

def _attach_section_232_batch_interpreter_stub(
    service: MetalCompositionService,
    *,
    hts_code: str = "7308.90.95",
    rule_type: str = "rate_schedule",
    metal_scope: str = "aluminum",
    usage: dict[str, int] | None = None,
) -> None:
    def _fake_invoke(**kwargs):
        messages = kwargs.get("messages") or []
        prompt = str(messages[-1]["content"])
        if "Section 232 source batch:\n" in prompt:
            batch_payload = json.loads(prompt.split("Section 232 source batch:\n", maxsplit=1)[1])
        else:
            batch_payload = json.loads(prompt.split("Section 232 source interpretation input:\n", maxsplit=1)[1])
        source = (batch_payload.get("sources") or [batch_payload.get("source")])[0]
        candidates = list(batch_payload.get("candidates") or [])
        usage_recorder = kwargs.get("usage_recorder")
        if usage_recorder is not None and usage is not None:
            usage_recorder.record(
                phase=kwargs.get("phase") or "section_232_ruleset",
                task=kwargs.get("task") or "section_232_batch_interpreter",
                model=kwargs.get("model_name") or "test-section-232-model",
                usage=usage,
            )
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps(
                            {
                                "results": [
                                    {
                                        "candidate_key": candidate["candidate_key"],
                                        "is_real_hts_candidate": True,
                                        "rule_type": rule_type,
                                        "effective_from": "2026-04-01",
                                        "effective_to": "2026-12-31",
                                        "metal_scope": metal_scope,
                                        "source_pages": candidate.get("source_pages") or [1],
                                        "source_excerpt": "Temporary reduced rate remains covered during the published window.",
                                        "interpreter_confidence": 0.93,
                                    }
                                    for candidate in candidates
                                ]
                            }
                        )
                    )
                )
            ]
        )

    service.workflow_runner.llm = types.SimpleNamespace(invoke_native_chat_completion=_fake_invoke)

def _publish_section_232_rule(
    service: MetalCompositionService,
    *,
    hts_code: str,
    rule_type: str,
    metal_scope: str = "steel",
    effective_from: str = "2026-04-07",
    effective_to: str | None = None,
) -> None:
    store = service.section_232_ruleset_store
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    candidate = Section232DraftRuleCandidate(
        candidate_id=f"{rule_type}-{hts_code}",
        batch_id=batch.batch_id,
        hts_code=hts_code,
        rule_type=rule_type,
        coverage_effect="remove" if rule_type == "remove" else "include",
        effective_from=effective_from,
        effective_to=effective_to,
        metal_scope=metal_scope,
        source_document_ids=["source-1"],
        source_pages=[29],
        source_excerpt="Synthetic published rule for router tests.",
        interpreter_confidence=0.95,
        catalog_match_found=True,
        review_decision="accepted",
    )
    store.replace_batch_candidates(batch.batch_id, [candidate])
    service.publish_section_232_draft_batch(batch_id=batch.batch_id, published_by="pytest")

def _attach_section_232_batch_interpreter_failure_stub(
    service: MetalCompositionService,
    *,
    message: str = "section 232 batch interpreter failed",
) -> None:
    def _fake_invoke(**kwargs):
        raise RuntimeError(message)

    service.workflow_runner.llm = types.SimpleNamespace(invoke_native_chat_completion=_fake_invoke)

def _process_section_232_batch(
    client: TestClient,
    fixture_path: Path,
    *,
    endpoint: str = "/api/metal-composition/section-232/draft-batches/process",
) -> dict:
    response = client.post(
        endpoint,
        files=[("files", (fixture_path.name, fixture_path.read_bytes(), "application/pdf"))],
        headers={"X-API-Key": "test-api-key"},
    )
    assert response.status_code == 201
    return response.json()

def test_section_232_source_upload_rejects_non_pdf():
    client = _client_with_service(_make_service())
    try:
        response = client.post(
            "/api/metal-composition/section-232/sources",
            files=[("files", ("not-a-pdf.txt", b"plain text", "text/plain"))],
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 422
        assert "PDF" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()

def test_section_232_source_upload_aliases_draft_batch_process_without_publishing_live_codes(tmp_path):
    service = _make_service()
    _attach_section_232_batch_interpreter_stub(service, hts_code="8481.80.30.90", rule_type="include")
    client = _client_with_service(service)
    fixture_path = _write_text_pdf(
        tmp_path / "single-code.pdf",
        "HTS 8481.80.30.90 remains covered.",
    )
    try:
        response = client.post(
            "/api/metal-composition/section-232/sources",
            files=[
                ("files", (fixture_path.name, fixture_path.read_bytes(), "application/pdf")),
                ("files", ("copy.pdf", fixture_path.read_bytes(), "application/pdf")),
            ],
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 201
        payload = response.json()
        assert payload["batch"]["status"] == "pending_review"
        assert payload["batch"]["source_count"] == 2
        assert payload["batch"]["rule_candidate_count"] == 2
        assert payload["batch"]["source_filenames"] == [fixture_path.name, "copy.pdf"]
        assert payload["ruleset_summary"]["active_ruleset_version"] is None
        assert payload["ruleset_summary"]["eligible_hts_code_count"] == 0
        assert payload["ruleset_summary"]["pending_draft_batch_count"] == 1
        assert len(payload["ruleset_summary"]["pending_draft_batches"]) == 1
        assert payload["ruleset_summary"]["pending_draft_batches"][0]["batch_id"] == payload["batch"]["batch_id"]
        assert payload["ruleset_summary"]["pending_draft_batches"][0]["source_filenames"] == [fixture_path.name, "copy.pdf"]
        assert len(payload["items"]) == 2
        assert [item["hts_code"] for item in payload["items"]] == ["8481.80.3090", "8481.80.3090"]
        assert sorted(item["source_filenames"][0] for item in payload["items"]) == ["copy.pdf", fixture_path.name]

        summary_response = client.get(
            "/api/metal-composition/section-232/ruleset-summary",
            headers={"X-API-Key": "test-api-key"},
        )
        assert summary_response.status_code == 200
        assert summary_response.json()["eligible_hts_code_count"] == 0
        assert summary_response.json()["pending_draft_batches"][0]["source_filenames"] == [fixture_path.name, "copy.pdf"]

        eligible_response = client.get(
            "/api/metal-composition/section-232/eligible-hts-codes",
            headers={"X-API-Key": "test-api-key"},
        )
        assert eligible_response.status_code == 200
        assert eligible_response.json() == {"total": 0, "codes": []}
    finally:
        app.dependency_overrides.clear()

def test_section_232_source_upload_can_include_token_usage(tmp_path):
    """Verify API-only usage tracking aggregates Section 232 PDF ingestion calls."""

    service = _make_service()
    _attach_section_232_batch_interpreter_stub(
        service,
        hts_code="8481.80.30.90",
        rule_type="include",
        usage={"prompt_tokens": 1234, "completion_tokens": 321, "total_tokens": 1555},
    )
    client = _client_with_service(service)
    fixture_path = _write_text_pdf(
        tmp_path / "single-code.pdf",
        "HTS 8481.80.30.90 remains covered.",
    )
    try:
        response = client.post(
            "/api/metal-composition/section-232/sources",
            data={"include_token_usage": "true"},
            files=[("files", (fixture_path.name, fixture_path.read_bytes(), "application/pdf"))],
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 201
        payload = response.json()
        assert payload["token_usage"]["input_tokens"] == 1234
        assert payload["token_usage"]["output_tokens"] == 321
        assert payload["token_usage"]["total_tokens"] == 1555
        assert payload["token_usage"]["missing_usage_entry_count"] == 0
        assert payload["token_usage"]["entries"] == [
            {
                "phase": "section_232_ruleset",
                "task": "section_232_batch_interpreter",
                "model": "test-section-232-model",
                "call_count": 1,
                "input_tokens": 1234,
                "output_tokens": 321,
                "total_tokens": 1555,
                "usage_available": True,
            }
        ]
    finally:
        app.dependency_overrides.clear()

def test_section_232_draft_batch_process_omits_token_usage_by_default(tmp_path):
    """Verify Section 232 ingestion token usage stays opt-in for API callers."""

    service = _make_service()
    _attach_section_232_batch_interpreter_stub(
        service,
        hts_code="8481.80.30.90",
        rule_type="include",
        usage={"prompt_tokens": 1234, "completion_tokens": 321, "total_tokens": 1555},
    )
    client = _client_with_service(service)
    fixture_path = _write_text_pdf(
        tmp_path / "single-code.pdf",
        "HTS 8481.80.30.90 remains covered.",
    )
    try:
        response = client.post(
            "/api/metal-composition/section-232/draft-batches/process",
            files=[("files", (fixture_path.name, fixture_path.read_bytes(), "application/pdf"))],
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 201
        assert response.json()["token_usage"] is None
    finally:
        app.dependency_overrides.clear()

def test_cancel_section_232_draft_batch_removes_pending_upload_without_changing_published_rules(tmp_path):
    service = _make_service()
    _publish_section_232_rule(
        service,
        hts_code="8481.80.30.90",
        rule_type="include",
    )
    _attach_section_232_batch_interpreter_stub(
        service,
        hts_code="7308.90.95",
        rule_type="include",
    )
    client = _client_with_service(service)
    fixture_path = _write_text_pdf(
        tmp_path / "pending-draft.pdf",
        "HTS 7308.90.95 is included in the new pending notice.",
    )
    try:
        process_payload = _process_section_232_batch(client, fixture_path)
        batch_id = process_payload["batch"]["batch_id"]

        cancel_response = client.delete(
            f"/api/metal-composition/section-232/draft-batches/{batch_id}",
            headers={"X-API-Key": "test-api-key"},
        )

        assert cancel_response.status_code == 200
        cancel_payload = cancel_response.json()
        assert cancel_payload["batch_id"] == batch_id
        assert cancel_payload["deleted_source_count"] == 1
        assert cancel_payload["deleted_source_filenames"] == [fixture_path.name]
        assert cancel_payload["deleted_draft_rule_count"] == 1
        assert cancel_payload["ruleset_summary"]["active_ruleset_version"] == "section232-v0001"
        assert cancel_payload["ruleset_summary"]["eligible_hts_code_count"] == 1
        assert cancel_payload["ruleset_summary"]["pending_draft_batch_count"] == 0

        stale_review_response = client.get(
            f"/api/metal-composition/section-232/draft-batches/{batch_id}/rules",
            headers={"X-API-Key": "test-api-key"},
        )
        assert stale_review_response.status_code == 404

        source_listing = client.get(
            "/api/metal-composition/section-232/sources",
            headers={"X-API-Key": "test-api-key"},
        )
        assert source_listing.status_code == 200
        assert source_listing.json()["items"] == []

        eligible_response = client.get(
            "/api/metal-composition/section-232/eligible-hts-codes",
            headers={"X-API-Key": "test-api-key"},
        )
        assert eligible_response.status_code == 200
        assert eligible_response.json() == {
            "total": 1,
            "codes": ["8481.80.3090"],
        }
    finally:
        app.dependency_overrides.clear()

def test_cancel_section_232_draft_batch_returns_404_or_409_for_unavailable_batches():
    service = _make_service()
    _publish_section_232_rule(
        service,
        hts_code="8481.80.30.90",
        rule_type="include",
    )
    published_batch = service.section_232_ruleset_store.list_draft_batches(status="published")[0]
    client = _client_with_service(service)
    try:
        missing_response = client.delete(
            "/api/metal-composition/section-232/draft-batches/not-a-batch",
            headers={"X-API-Key": "test-api-key"},
        )
        assert missing_response.status_code == 404

        published_response = client.delete(
            f"/api/metal-composition/section-232/draft-batches/{published_batch.batch_id}",
            headers={"X-API-Key": "test-api-key"},
        )
        assert published_response.status_code == 409
        assert "already published" in published_response.json()["detail"]
    finally:
        app.dependency_overrides.clear()

def test_section_232_source_upload_processes_unrecognized_scope_phrasing_without_crashing(tmp_path):
    service = _make_service()
    _attach_section_232_batch_interpreter_stub(
        service,
        hts_code="8302.41.00",
        rule_type="include",
        metal_scope="applies to all covered products in the notice",
    )
    client = _client_with_service(service)
    fixture_path = _write_text_pdf(
        tmp_path / "scope-one-code.pdf",
        "HTS 8302.41.00 is included in the notice.",
    )
    try:
        response = client.post(
            "/api/metal-composition/section-232/sources",
            files=[("files", (fixture_path.name, fixture_path.read_bytes(), "application/pdf"))],
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 201
        payload = response.json()
        assert payload["batch"]["rule_candidate_count"] == 1
        assert payload["items"][0]["metal_scope"] == "unspecified"
    finally:
        app.dependency_overrides.clear()

def test_process_review_publish_section_232_batch_preserves_prior_active_rules_and_browsing_reads_published_rules(tmp_path):
    service = _make_service()
    client = _client_with_service(service)
    first_fixture_path = _write_text_pdf(
        tmp_path / "include-code.pdf",
        "HTS 8481.80.30.90 is included.",
    )
    second_fixture_path = _write_text_pdf(
        tmp_path / "rate-code.pdf",
        "HTS 7308.90.95 is covered at a temporary rate.",
    )
    try:
        _attach_section_232_batch_interpreter_stub(
            service,
            hts_code="8481.80.30.90",
            rule_type="include",
        )
        first_process_payload = _process_section_232_batch(client, first_fixture_path)
        first_batch_id = first_process_payload["batch"]["batch_id"]
        first_candidate_id = first_process_payload["items"][0]["candidate_id"]

        review_rows = client.get(
            f"/api/metal-composition/section-232/draft-batches/{first_batch_id}/rules",
            headers={"X-API-Key": "test-api-key"},
        )

        assert review_rows.status_code == 200
        review_rows_payload = review_rows.json()
        assert review_rows_payload["total"] == 1
        assert review_rows_payload["items"][0]["candidate_id"] == first_candidate_id
        assert review_rows_payload["items"][0]["description"] is not None

        review_response = client.patch(
            f"/api/metal-composition/section-232/draft-batches/{first_batch_id}/rules/{first_candidate_id}",
            json={"decision": "accepted"},
            headers={"X-API-Key": "test-api-key"},
        )

        assert review_response.status_code == 200
        assert review_response.json()["item"]["review_decision"] == "accepted"

        first_publish_response = client.post(
            f"/api/metal-composition/section-232/draft-batches/{first_batch_id}/publish",
            json={"published_by": "pytest"},
            headers={"X-API-Key": "test-api-key"},
        )

        assert first_publish_response.status_code == 200
        first_publish_payload = first_publish_response.json()
        assert first_publish_payload["published_version"] == "section232-v0001"
        assert first_publish_payload["accepted_rule_count"] == 1

        stale_review_rows = client.get(
            f"/api/metal-composition/section-232/draft-batches/{first_batch_id}/rules",
            headers={"X-API-Key": "test-api-key"},
        )
        assert stale_review_rows.status_code == 404

        _attach_section_232_batch_interpreter_stub(
            service,
            hts_code="7308.90.95",
            rule_type="rate_schedule",
        )
        second_process_payload = _process_section_232_batch(client, second_fixture_path)
        second_batch_id = second_process_payload["batch"]["batch_id"]
        second_candidate_id = second_process_payload["items"][0]["candidate_id"]

        second_review_response = client.patch(
            f"/api/metal-composition/section-232/draft-batches/{second_batch_id}/rules/{second_candidate_id}",
            json={"decision": "accepted"},
            headers={"X-API-Key": "test-api-key"},
        )
        assert second_review_response.status_code == 200

        second_publish_response = client.post(
            f"/api/metal-composition/section-232/draft-batches/{second_batch_id}/publish",
            json={"published_by": "pytest"},
            headers={"X-API-Key": "test-api-key"},
        )
        assert second_publish_response.status_code == 200
        publish_payload = second_publish_response.json()
        assert publish_payload["published_version"] == "section232-v0002"
        assert publish_payload["accepted_rule_count"] == 2
        assert publish_payload["ruleset_summary"]["active_ruleset_version"] == "section232-v0002"
        assert publish_payload["ruleset_summary"]["eligible_hts_code_count"] == 2
        assert publish_payload["ruleset_summary"]["pending_draft_batch_count"] == 0
        assert publish_payload["ruleset_summary"]["last_published_at"] is not None

        summary_response = client.get(
            "/api/metal-composition/section-232/ruleset-summary",
            headers={"X-API-Key": "test-api-key"},
        )

        assert summary_response.status_code == 200
        assert summary_response.json() == publish_payload["ruleset_summary"]

        source_listing = client.get(
            "/api/metal-composition/section-232/sources",
            headers={"X-API-Key": "test-api-key"},
        )
        assert source_listing.status_code == 200
        assert source_listing.json()["eligible_hts_code_count"] == 2

        eligible_response = client.get(
            "/api/metal-composition/section-232/eligible-hts-codes",
            headers={"X-API-Key": "test-api-key"},
        )
        assert eligible_response.status_code == 200
        assert eligible_response.json() == {
            "total": 2,
            "codes": ["7308.90.95", "8481.80.3090"],
        }

        details_response = client.get(
            "/api/metal-composition/section-232/eligible-hts-codes/details",
            params={"query": "valve"},
            headers={"X-API-Key": "test-api-key"},
        )
        assert details_response.status_code == 200
        details_payload = details_response.json()
        assert details_payload["total"] == 1
        assert details_payload["items"][0]["code"] == "8481.80.3090"
    finally:
        app.dependency_overrides.clear()

def test_process_section_232_draft_batch_keeps_suspect_rows_when_interpretation_payload_is_invalid(tmp_path):
    service = _make_service()

    def _fake_invoke(**_kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='{"templates": [{"rule_type": "rate_schedule", "effective_from": "2025-03-12T00:01:00-04:00"'
                    )
                )
            ]
        )

    service.workflow_runner.llm = types.SimpleNamespace(invoke_native_chat_completion=_fake_invoke)
    client = _client_with_service(service)
    fixture_path = _write_text_pdf(
        tmp_path / "broken-section232.pdf",
        "HTS 7308.90.95 remains covered.",
    )
    try:
        response = client.post(
            "/api/metal-composition/section-232/draft-batches/process",
            files=[("files", (fixture_path.name, fixture_path.read_bytes(), "application/pdf"))],
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 201
        payload = response.json()
        assert payload["batch"]["rule_candidate_count"] == 1
        assert payload["items"][0]["candidate_quality"] == "suspect"
        assert "llm_result_missing" in payload["items"][0]["candidate_flags"]
        assert "rule_type_undefined" in payload["items"][0]["candidate_flags"]

        sources_response = client.get(
            "/api/metal-composition/section-232/sources",
            headers={"X-API-Key": "test-api-key"},
        )
        assert sources_response.status_code == 200
        assert len(sources_response.json()["items"]) == 1

        summary_response = client.get(
            "/api/metal-composition/section-232/ruleset-summary",
            headers={"X-API-Key": "test-api-key"},
        )
        assert summary_response.status_code == 200
        assert summary_response.json()["pending_draft_batch_count"] == 1
    finally:
        app.dependency_overrides.clear()

def test_bulk_review_section_232_draft_rules_updates_multiple_candidates():
    service = _make_service()
    client = _client_with_service(service)
    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["annex.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            Section232DraftRuleCandidate(
                candidate_id="candidate-include",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
                effective_from="2026-04-07",
                effective_to=None,
                metal_scope="steel",
                source_document_ids=["source-1"],
                source_pages=[1],
                source_excerpt="Synthetic include rule for router tests.",
                interpreter_confidence=0.95,
                catalog_match_found=True,
                review_decision="pending",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-rate",
                batch_id=batch.batch_id,
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_from="2026-04-07",
                effective_to="2026-12-31",
                metal_scope="steel",
                source_document_ids=["source-1"],
                source_pages=[2],
                source_excerpt="Synthetic rate rule for router tests.",
                interpreter_confidence=0.93,
                catalog_match_found=True,
                review_decision="pending",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-remove",
                batch_id=batch.batch_id,
                hts_code="2710.19.30.50",
                rule_type="remove",
                coverage_effect="remove",
                effective_from="2026-04-07",
                effective_to=None,
                metal_scope="steel",
                source_document_ids=["source-1"],
                source_pages=[3],
                source_excerpt="Synthetic remove rule for router tests.",
                interpreter_confidence=0.91,
                catalog_match_found=True,
                review_decision="pending",
            ),
        ],
    )

    try:
        response = client.patch(
            f"/api/metal-composition/section-232/draft-batches/{batch.batch_id}/rules",
            json={
                "candidate_ids": ["candidate-include", "candidate-remove"],
                "decision": "accepted",
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "updated_count": 2,
            "candidate_ids": ["candidate-include", "candidate-remove"],
            "decision": "accepted",
        }

        refreshed_candidates = {
            candidate.candidate_id: candidate.review_decision
            for candidate in service.section_232_ruleset_store.list_draft_candidates(batch_id=batch.batch_id)
        }
        assert refreshed_candidates == {
            "candidate-include": "accepted",
            "candidate-rate": "pending",
            "candidate-remove": "accepted",
        }
    finally:
        app.dependency_overrides.clear()

def test_publish_section_232_batch_replaces_same_code_rows_across_scopes(tmp_path):
    service = _make_service()
    client = _client_with_service(service)
    fixture_path = _write_text_pdf(
        tmp_path / "same-code-two-scopes.pdf",
        "HTS 7308.90.95 remains covered.",
    )
    try:
        _attach_section_232_batch_interpreter_stub(
            service,
            hts_code="7308.90.95",
            rule_type="include",
            metal_scope="steel",
        )
        first_process_payload = _process_section_232_batch(client, fixture_path)
        first_batch_id = first_process_payload["batch"]["batch_id"]
        first_candidate_id = first_process_payload["items"][0]["candidate_id"]

        review_response = client.patch(
            f"/api/metal-composition/section-232/draft-batches/{first_batch_id}/rules/{first_candidate_id}",
            json={"decision": "accepted"},
            headers={"X-API-Key": "test-api-key"},
        )
        assert review_response.status_code == 200

        first_publish_response = client.post(
            f"/api/metal-composition/section-232/draft-batches/{first_batch_id}/publish",
            json={"published_by": "pytest"},
            headers={"X-API-Key": "test-api-key"},
        )
        assert first_publish_response.status_code == 200

        _attach_section_232_batch_interpreter_stub(
            service,
            hts_code="7308.90.95",
            rule_type="remove",
            metal_scope="aluminum",
        )
        second_process_payload = _process_section_232_batch(client, fixture_path)
        second_batch_id = second_process_payload["batch"]["batch_id"]
        second_candidate_id = second_process_payload["items"][0]["candidate_id"]

        second_review_response = client.patch(
            f"/api/metal-composition/section-232/draft-batches/{second_batch_id}/rules/{second_candidate_id}",
            json={"decision": "accepted"},
            headers={"X-API-Key": "test-api-key"},
        )
        assert second_review_response.status_code == 200

        second_publish_response = client.post(
            f"/api/metal-composition/section-232/draft-batches/{second_batch_id}/publish",
            json={"published_by": "pytest"},
            headers={"X-API-Key": "test-api-key"},
        )
        assert second_publish_response.status_code == 200
        publish_payload = second_publish_response.json()
        assert publish_payload["published_version"] == "section232-v0002"
        assert publish_payload["accepted_rule_count"] == 1
        assert publish_payload["ruleset_summary"]["eligible_hts_code_count"] == 0

        review_response = client.get(
            "/api/metal-composition/section-232/review?version=section232-v0002",
            headers={"X-API-Key": "test-api-key"},
        )
        assert review_response.status_code == 200
        review_payload = review_response.json()
        assert review_payload["total"] == 1
        remove_row = next(row for row in review_payload["rows"] if row["candidate_id"] == second_candidate_id)
        assert remove_row["legal_hts_code"] == "7308.90.95"
        assert remove_row["rule_type"] == "remove"
        assert remove_row["coverage_effect"] == "remove"
        assert remove_row["metal_scope"] == "aluminum"
        assert remove_row["source_documents"] == [
            {
                "source_id": second_process_payload["items"][0]["source_document_ids"][0],
                "filename": fixture_path.name,
                "uploaded_at": second_process_payload["items"][0]["source_uploaded_at"],
            }
        ]
        assert remove_row["source_uploaded_at"] == second_process_payload["items"][0]["source_uploaded_at"]
        assert remove_row["processed_at"] == second_process_payload["batch"]["created_at"]
        assert [(item["candidate_id"], item["coverage_effect"]) for item in remove_row["history"]] == [
            (first_candidate_id, "include")
        ]
    finally:
        app.dependency_overrides.clear()

def test_section_232_source_listing_returns_reverse_chronological_order():
    service = _make_service()
    _attach_section_232_batch_interpreter_stub(service)
    source_store = service.section_232_source_store
    assert isinstance(source_store, InMemorySection232SourceStore)
    fixture_path = (
        Path(__file__).resolve().parents[2] / "data" / "section232_eligible" / "2025-15819.pdf"
    )
    extracted = fixture_path.read_bytes()
    service.upload_section_232_source(filename="older.pdf", content=extracted)
    service.upload_section_232_source(filename="newer.pdf", content=extracted)
    client = _client_with_service(service)
    try:
        response = client.get(
            "/api/metal-composition/section-232/sources",
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 2
        assert payload["eligible_hts_code_count"] == 0
        assert [item["filename"] for item in payload["items"]] == ["newer.pdf", "older.pdf"]
    finally:
        app.dependency_overrides.clear()

def test_section_232_process_failure_still_creates_reviewable_suspect_batch(tmp_path):
    service = _make_service()
    _attach_section_232_batch_interpreter_failure_stub(service)
    client = _client_with_service(service, raise_server_exceptions=False)
    fixture_path = _write_text_pdf(
        tmp_path / "process-failure.pdf",
        "HTS 7308.90.95 remains covered.",
    )
    try:
        response = client.post(
            "/api/metal-composition/section-232/draft-batches/process",
            files=[("files", (fixture_path.name, fixture_path.read_bytes(), "application/pdf"))],
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 201
        payload = response.json()
        assert payload["batch"]["rule_candidate_count"] == 1
        assert payload["items"][0]["candidate_quality"] == "suspect"

        summary_response = client.get(
            "/api/metal-composition/section-232/ruleset-summary",
            headers={"X-API-Key": "test-api-key"},
        )
        assert summary_response.status_code == 200
        assert summary_response.json()["pending_draft_batch_count"] == 1
    finally:
        app.dependency_overrides.clear()

def test_section_232_direct_eligible_hts_code_update_route_is_removed():
    client = _client_with_service(_make_service())
    try:
        update_response = client.put(
            "/api/metal-composition/section-232/eligible-hts-codes",
            json={
                "codes": ["7616.99.51.60", "7308.90.95", "7616.99.51.60"],
                "update_mode": "append",
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert update_response.status_code == 405
    finally:
        app.dependency_overrides.clear()

def test_bulk_review_section_232_draft_rules_supports_workspace_wide_select_all():
    service = _make_service()
    client = _client_with_service(service)
    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["annex.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            Section232DraftRuleCandidate(
                candidate_id="candidate-include",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
                effective_from="2026-04-07",
                effective_to=None,
                metal_scope="steel",
                source_document_ids=["source-1"],
                source_pages=[1],
                source_excerpt="Synthetic include rule for router tests.",
                interpreter_confidence=0.95,
                catalog_match_found=True,
                review_decision="pending",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-rate",
                batch_id=batch.batch_id,
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_from="2026-04-07",
                effective_to="2026-12-31",
                metal_scope="steel",
                source_document_ids=["source-1"],
                source_pages=[2],
                source_excerpt="Synthetic rate rule for router tests.",
                interpreter_confidence=0.93,
                catalog_match_found=True,
                review_decision="pending",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-remove",
                batch_id=batch.batch_id,
                hts_code="2710.19.30.50",
                rule_type="remove",
                coverage_effect="remove",
                effective_from="2026-04-07",
                effective_to=None,
                metal_scope="steel",
                source_document_ids=["source-1"],
                source_pages=[3],
                source_excerpt="Synthetic remove rule for router tests.",
                interpreter_confidence=0.91,
                catalog_match_found=True,
                review_decision="pending",
            ),
        ],
    )

    try:
        response = client.patch(
            f"/api/metal-composition/section-232/draft-batches/{batch.batch_id}/rules",
            json={
                "selection_mode": "all",
                "excluded_candidate_ids": ["candidate-rate"],
                "decision": "rejected",
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "updated_count": 2,
            "candidate_ids": [],
            "decision": "rejected",
        }

        refreshed_candidates = {
            candidate.candidate_id: candidate.review_decision
            for candidate in service.section_232_ruleset_store.list_draft_candidates(batch_id=batch.batch_id)
        }
        assert refreshed_candidates == {
            "candidate-include": "rejected",
            "candidate-rate": "pending",
            "candidate-remove": "rejected",
        }
    finally:
        app.dependency_overrides.clear()

def test_section_232_review_rows_use_managed_catalog_lookup_for_catalog_warning(tmp_path):
    service = _make_service()
    _attach_section_232_batch_interpreter_stub(service, hts_code="9999.99.99", rule_type="include")
    client = _client_with_service(service)
    fixture_path = _write_text_pdf(
        tmp_path / "missing-catalog.pdf",
        "HTS 9999.99.99 remains covered.",
    )
    try:
        process_payload = _process_section_232_batch(client, fixture_path)
        assert process_payload["items"][0]["catalog_match_found"] is False
        assert process_payload["items"][0]["catalog_warning"] == "HTS code not found in managed catalog"

        review_rows = client.get(
            f"/api/metal-composition/section-232/draft-batches/{process_payload['batch']['batch_id']}/rules",
            headers={"X-API-Key": "test-api-key"},
        )
        assert review_rows.status_code == 200
        review_payload = review_rows.json()
        assert review_payload["items"][0]["catalog_match_found"] is False
        assert review_payload["items"][0]["catalog_warning"] == "HTS code not found in managed catalog"
    finally:
        app.dependency_overrides.clear()

def test_get_section_232_review_workspace_for_draft_batch():
    service = _make_service()
    store = service.section_232_ruleset_store
    source = service.section_232_source_store.save_source(
        filename="copper-notice.pdf",
        size_bytes=321,
        extracted=ExtractedSection232Source(
            page_count=1,
            extraction_status="completed",
            full_text="[Page 6]\n[Docket No. 240814-0099]",
            page_texts=[
                {
                    "page_number": 6,
                    "text": "[Docket No. 240814-0099]",
                    "layout_aware_text": "[Docket No. 240814-0099]",
                    "page_excerpt": "[Docket No. 240814-0099]",
                    "char_count": 24,
                    "hts_mentions": ["2408.14"],
                    "hts_occurrences": [
                        {
                            "page_number": 6,
                            "matched_text": "240814",
                            "normalized_hts_code": "2408.14",
                            "context_text": "[Docket No. 240814-0099]",
                            "text_sources": ["plain", "layout"],
                        }
                    ],
                }
            ],
            hts_mentions=["2408.14"],
            warnings=[],
        ),
    )
    batch = store.create_draft_batch(source_ids=[source.source_id], source_filenames=[source.filename])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            Section232DraftRuleCandidate(
                candidate_id="candidate-1",
                batch_id=batch.batch_id,
                hts_code="2408.14",
                rule_type="include",
                coverage_effect="include",
                effective_from="2025-08-01",
                effective_to=None,
                metal_scope="unspecified",
                source_document_ids=[source.source_id],
                source_pages=[6],
                source_excerpt="Synthetic draft rule for review workspace tests.",
                interpreter_confidence=0.94,
                catalog_match_found=False,
                review_decision="pending",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-2",
                batch_id=batch.batch_id,
                hts_code="2650.01",
                rule_type="include",
                coverage_effect="include",
                effective_from="2025-08-01",
                effective_to=None,
                metal_scope="unspecified",
                source_document_ids=[source.source_id],
                source_pages=[2],
                source_excerpt="Another missing HTS code for ordering tests.",
                interpreter_confidence=0.93,
                catalog_match_found=False,
                review_decision="pending",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-3",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_from="2025-08-01",
                effective_to=None,
                metal_scope="copper",
                source_document_ids=[source.source_id],
                source_pages=[3],
                source_excerpt="Catalog-matched HTS code for ordering tests.",
                interpreter_confidence=0.95,
                catalog_match_found=True,
                review_decision="pending",
            ),
        ],
    )
    service._set_hts_catalog_resolver_for_admin(
        catalog_frame=pd.DataFrame(
            [
                {
                    "code": "7407.10.30.00",
                    "raw_code": "7407103000",
                    "digits": 10,
                    "chapter_number": 74,
                    "heading_code": "7407",
                    "family_6_code": "7407.10",
                    "family_8_code": "7407.10.30",
                    "indent": 0,
                    "parent_code": "7407.10.30",
                    "description": "Copper bars rods and profiles of refined copper",
                    "path_description": "Copper bars rods and profiles of refined copper",
                    "unit_of_quantity": "",
                    "general_rate_of_duty": "",
                    "special_rate_of_duty": "",
                    "column_2_rate_of_duty": "",
                    "quota_quantity": "",
                    "additional_duties": "",
                    "searchable_text": "copper bars rods and profiles of refined copper",
                    "sort_order": 1,
                }
            ]
        ),
        code_map_frame=pd.DataFrame(columns=["source_code", "target_code", "mapping_type", "source_basis", "effective_note"]),
    )
    client = _client_with_service(service)
    try:
        response = client.get(
            f"/api/metal-composition/section-232/review?batch_id={batch.batch_id}",
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "draft"
        assert payload["batch"]["batch_id"] == batch.batch_id
        assert [row["legal_hts_code"] for row in payload["rows"]] == ["2408.14", "2650.01", "7407.10.30"]
        assert [row["catalog_match_type"] for row in payload["rows"]] == ["missing", "missing", "family"]
        assert payload["rows"][0]["match_evidence"] == [
            {
                "source_id": source.source_id,
                "source_filename": "copper-notice.pdf",
                "page_number": 6,
                "matched_text": "240814",
                "normalized_hts_code": "2408.14",
                "context_text": "[Docket No. 240814-0099]",
                "text_sources": ["plain", "layout"],
            }
        ]
        assert payload["rows"][0]["source_documents"] == [
            {
                "source_id": source.source_id,
                "filename": "copper-notice.pdf",
                "uploaded_at": source.uploaded_at,
            }
        ]
        assert payload["rows"][0]["source_uploaded_at"] == source.uploaded_at
        assert payload["rows"][0]["processed_at"] == batch.created_at
    finally:
        app.dependency_overrides.clear()

def test_get_section_232_review_workspace_for_draft_batch_returns_only_requested_page():
    service = _make_service()
    store = service.section_232_ruleset_store
    source = service.section_232_source_store.save_source(
        filename="copper-notice.pdf",
        size_bytes=321,
        extracted=ExtractedSection232Source(
            page_count=1,
            extraction_status="completed",
            full_text="[Page 6]\n[Docket No. 240814-0099]",
            page_texts=[
                {
                    "page_number": 6,
                    "text": "[Docket No. 240814-0099]",
                    "layout_aware_text": "[Docket No. 240814-0099]",
                    "page_excerpt": "[Docket No. 240814-0099]",
                    "char_count": 24,
                    "hts_mentions": ["2408.14"],
                    "hts_occurrences": [
                        {
                            "page_number": 6,
                            "matched_text": "240814",
                            "normalized_hts_code": "2408.14",
                            "context_text": "[Docket No. 240814-0099]",
                            "text_sources": ["plain", "layout"],
                        }
                    ],
                }
            ],
            hts_mentions=["2408.14"],
            warnings=[],
        ),
    )
    batch = store.create_draft_batch(source_ids=[source.source_id], source_filenames=[source.filename])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            Section232DraftRuleCandidate(
                candidate_id="candidate-1",
                batch_id=batch.batch_id,
                hts_code="2408.14",
                rule_type="include",
                coverage_effect="include",
                effective_from="2025-08-01",
                effective_to=None,
                metal_scope="unspecified",
                source_document_ids=[source.source_id],
                source_pages=[6],
                source_excerpt="Synthetic draft rule for review workspace tests.",
                interpreter_confidence=0.94,
                catalog_match_found=False,
                review_decision="pending",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-2",
                batch_id=batch.batch_id,
                hts_code="2650.01",
                rule_type="include",
                coverage_effect="include",
                effective_from="2025-08-01",
                effective_to=None,
                metal_scope="unspecified",
                source_document_ids=[source.source_id],
                source_pages=[2],
                source_excerpt="Another missing HTS code for ordering tests.",
                interpreter_confidence=0.93,
                catalog_match_found=False,
                review_decision="pending",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-3",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_from="2025-08-01",
                effective_to=None,
                metal_scope="copper",
                source_document_ids=[source.source_id],
                source_pages=[3],
                source_excerpt="Catalog-matched HTS code for ordering tests.",
                interpreter_confidence=0.95,
                catalog_match_found=True,
                review_decision="pending",
            ),
        ],
    )
    service._set_hts_catalog_resolver_for_admin(
        catalog_frame=pd.DataFrame(
            [
                {
                    "code": "7407.10.30.00",
                    "raw_code": "7407103000",
                    "digits": 10,
                    "chapter_number": 74,
                    "heading_code": "7407",
                    "family_6_code": "7407.10",
                    "family_8_code": "7407.10.30",
                    "indent": 0,
                    "parent_code": "7407.10.30",
                    "description": "Copper bars rods and profiles of refined copper",
                    "path_description": "Copper bars rods and profiles of refined copper",
                    "unit_of_quantity": "",
                    "general_rate_of_duty": "",
                    "special_rate_of_duty": "",
                    "column_2_rate_of_duty": "",
                    "quota_quantity": "",
                    "additional_duties": "",
                    "searchable_text": "copper bars rods and profiles of refined copper",
                    "sort_order": 1,
                }
            ]
        ),
        code_map_frame=pd.DataFrame(columns=["source_code", "target_code", "mapping_type", "source_basis", "effective_note"]),
    )
    client = _client_with_service(service)
    try:
        response = client.get(
            f"/api/metal-composition/section-232/review?batch_id={batch.batch_id}&limit=1&offset=1",
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 3
        assert payload["limit"] == 1
        assert payload["offset"] == 1
        assert [row["legal_hts_code"] for row in payload["rows"]] == ["2650.01"]
    finally:
        app.dependency_overrides.clear()

def test_publish_section_232_draft_batch_reports_duplicate_hts_codes_in_conflict():
    service = _make_service()
    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["duplicate-slot.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            Section232DraftRuleCandidate(
                candidate_id="candidate-1",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_from="2026-01-01",
                effective_to=None,
                metal_scope="steel and copper",
                source_document_ids=["source-1"],
                source_pages=[1],
                source_excerpt="Synthetic duplicate slot include row.",
                interpreter_confidence=0.95,
                catalog_match_found=True,
                review_decision="accepted",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-2",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_from="2026-01-01",
                effective_to=None,
                metal_scope="copper + steel",
                source_document_ids=["source-1"],
                source_pages=[1],
                source_excerpt="Synthetic duplicate slot include row.",
                interpreter_confidence=0.94,
                catalog_match_found=True,
                review_decision="accepted",
            ),
        ],
    )
    client = _client_with_service(service)
    try:
        response = client.post(
            f"/api/metal-composition/section-232/draft-batches/{batch.batch_id}/publish",
            json={"published_by": "pytest"},
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 409
        assert response.json()["detail"] == (
            f"draft batch {batch.batch_id} has duplicate normalized publication slots for HTS codes "
            "7407.10.30 and cannot be published"
        )
    finally:
        app.dependency_overrides.clear()

def test_get_section_232_review_workspace_rejects_missing_or_mixed_query_args():
    client = _client_with_service(_make_service())
    try:
        missing = client.get(
            "/api/metal-composition/section-232/review",
            headers={"X-API-Key": "test-api-key"},
        )
        mixed = client.get(
            "/api/metal-composition/section-232/review?batch_id=batch-1&version=section232-v0001",
            headers={"X-API-Key": "test-api-key"},
        )

        assert missing.status_code == 422
        assert mixed.status_code == 422
    finally:
        app.dependency_overrides.clear()

def test_get_section_232_review_workspace_for_published_ruleset():
    service = _make_service()
    _publish_section_232_rule(
        service,
        hts_code="7407.10.30",
        rule_type="rate_schedule",
        metal_scope="copper",
        effective_from="2025-08-01",
    )
    client = _client_with_service(service)
    try:
        response = client.get(
            "/api/metal-composition/section-232/review?version=section232-v0001",
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "published"
        assert payload["version"] == "section232-v0001"
        assert payload["rows"][0]["legal_hts_code"] == "7407.10.30"
        assert payload["source_filenames"] == ["annex.pdf"]
        assert payload["rows"][0]["source_documents"] == [
            {
                "source_id": "source-1",
                "filename": "annex.pdf",
                "uploaded_at": None,
            }
        ]
    finally:
        app.dependency_overrides.clear()

def test_get_section_232_review_workspace_filters_hts_query_before_pagination():
    service = _make_service()
    _publish_section_232_rule(
        service,
        hts_code="3403.99.00",
        rule_type="include",
        metal_scope="steel",
        effective_from="2026-04-01",
    )
    _publish_section_232_rule(
        service,
        hts_code="3405.10.00",
        rule_type="include",
        metal_scope="steel",
        effective_from="2026-04-01",
    )
    client = _client_with_service(service)
    try:
        dotted_response = client.get(
            "/api/metal-composition/section-232/review?version=section232-v0002&hts_query=3403.99&limit=1&offset=0",
            headers={"X-API-Key": "test-api-key"},
        )
        digits_response = client.get(
            "/api/metal-composition/section-232/review?version=section232-v0002&hts_query=340399&limit=1&offset=0",
            headers={"X-API-Key": "test-api-key"},
        )

        assert dotted_response.status_code == 200
        assert digits_response.status_code == 200
        assert dotted_response.json()["total"] == 1
        assert digits_response.json()["total"] == 1
        assert dotted_response.json()["rows"][0]["legal_hts_code"] == "3403.99.00"
        assert digits_response.json()["rows"][0]["legal_hts_code"] == "3403.99.00"
    finally:
        app.dependency_overrides.clear()

def test_get_section_232_review_workspace_collapses_published_rows_and_returns_history():
    service = _make_service()
    store = service.section_232_ruleset_store

    older_batch = store.create_draft_batch(source_ids=["source-older"], source_filenames=["2025-15819.pdf"])
    older_candidate = dataclass_replace(
        Section232DraftRuleCandidate(
            candidate_id="candidate-older-include",
            batch_id=older_batch.batch_id,
            hts_code="3403.99.00",
            rule_type="include",
            coverage_effect="include",
            effective_from="2026-04-01",
            effective_to=None,
            metal_scope="steel",
            source_document_ids=["source-older"],
            source_pages=[1],
            source_excerpt="Older include rule.",
            interpreter_confidence=0.95,
            catalog_match_found=True,
            review_decision="accepted",
        ),
        processed_at="2026-04-23T12:11:22Z",
    )
    store.replace_batch_candidates(older_batch.batch_id, [older_candidate])
    store.publish_batch(older_batch.batch_id, published_by="pytest")

    newer_batch = store.create_draft_batch(source_ids=["source-newer"], source_filenames=["ANNEXES-I-A-I-B-II-III-IV.pdf"])
    newer_candidate = dataclass_replace(
        Section232DraftRuleCandidate(
            candidate_id="candidate-newer-remove",
            batch_id=newer_batch.batch_id,
            hts_code="3403.99.00",
            rule_type="remove",
            coverage_effect="remove",
            effective_from="2026-04-01",
            effective_to=None,
            metal_scope="unspecified",
            source_document_ids=["source-newer"],
            source_pages=[2],
            source_excerpt="Newer remove rule.",
            interpreter_confidence=0.95,
            catalog_match_found=True,
            review_decision="accepted",
        ),
        processed_at="2026-04-23T12:27:05Z",
    )
    store.replace_batch_candidates(newer_batch.batch_id, [newer_candidate])
    store.publish_batch(newer_batch.batch_id, published_by="pytest")

    client = _client_with_service(service)
    try:
        response = client.get(
            "/api/metal-composition/section-232/review?version=section232-v0002&hts_query=3403.99",
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert [(row["candidate_id"], row["coverage_effect"]) for row in payload["rows"]] == [
            ("candidate-newer-remove", "remove")
        ]
        history = payload["rows"][0]["history"]
        assert [(item["candidate_id"], item["coverage_effect"], item["source_filenames"]) for item in history] == [
            ("candidate-older-include", "include", ["2025-15819.pdf"])
        ]
        assert history[0]["version"] == "section232-v0001"
    finally:
        app.dependency_overrides.clear()

def test_get_section_232_review_workspace_preserves_published_row_source_filenames_after_source_store_clear():
    service = _make_service()
    _publish_section_232_rule(
        service,
        hts_code="7407.10.30",
        rule_type="rate_schedule",
        metal_scope="copper",
        effective_from="2025-08-01",
    )
    service.section_232_source_store.clear_sources()
    client = _client_with_service(service)
    try:
        response = client.get(
            "/api/metal-composition/section-232/review?version=section232-v0001",
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["mode"] == "published"
        assert payload["rows"][0]["source_filenames"] == ["annex.pdf"]
    finally:
        app.dependency_overrides.clear()

def test_delete_section_232_draft_hts_code_endpoint_removes_matching_candidates():
    service = _make_service()
    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["draft-delete.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            Section232DraftRuleCandidate(
                candidate_id="candidate-1",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="include",
                coverage_effect="include",
                effective_from="2026-04-07",
                effective_to=None,
                metal_scope="copper",
                source_document_ids=["source-1"],
                source_pages=[1],
                source_excerpt="Delete me 1.",
                interpreter_confidence=0.94,
                catalog_match_found=True,
                review_decision="pending",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-2",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_from="2026-05-01",
                effective_to=None,
                metal_scope="steel",
                source_document_ids=["source-1"],
                source_pages=[1],
                source_excerpt="Delete me 2.",
                interpreter_confidence=0.95,
                catalog_match_found=True,
                review_decision="pending",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-keep",
                batch_id=batch.batch_id,
                hts_code="7407.10.50",
                rule_type="include",
                coverage_effect="include",
                effective_from="2026-04-07",
                effective_to=None,
                metal_scope="copper",
                source_document_ids=["source-1"],
                source_pages=[1],
                source_excerpt="Keep me.",
                interpreter_confidence=0.93,
                catalog_match_found=True,
                review_decision="pending",
            ),
        ],
    )
    client = _client_with_service(service)
    try:
        response = client.delete(
            f"/api/metal-composition/section-232/draft-batches/{batch.batch_id}/hts-codes/74071030",
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "batch_id": batch.batch_id,
            "deleted_hts_code": "7407.10.30",
            "deleted_count": 2,
        }
        remaining_candidates = service.section_232_ruleset_store.list_draft_candidates(batch_id=batch.batch_id)
        assert [candidate.candidate_id for candidate in remaining_candidates] == ["candidate-keep"]
    finally:
        app.dependency_overrides.clear()

def test_delete_section_232_published_hts_code_endpoint_creates_new_version():
    service = _make_service()
    _publish_section_232_rule(
        service,
        hts_code="7407.10.30",
        rule_type="rate_schedule",
        metal_scope="copper",
        effective_from="2026-01-01",
    )
    client = _client_with_service(service)
    try:
        response = client.post(
            "/api/metal-composition/section-232/published/hts-codes/7407.10.30/delete",
            json={"published_by": "auditor"},
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "deleted_hts_code": "7407.10.30",
            "removed_rule_count": 1,
            "published_version": "section232-v0002",
            "ruleset_summary": {
                "active_ruleset_version": "section232-v0002",
                "eligible_hts_code_count": 0,
                "pending_draft_batch_count": 0,
                "pending_draft_batches": [],
                "last_published_at": service.section_232_ruleset_store.get_last_published_at(),
            },
        }
        review_response = client.get(
            "/api/metal-composition/section-232/review?version=section232-v0002",
            headers={"X-API-Key": "test-api-key"},
        )
        assert review_response.status_code == 200
        assert review_response.json()["rows"] == []
    finally:
        app.dependency_overrides.clear()

def test_get_section_232_review_workspace_returns_404_for_missing_batch_id():
    client = _client_with_service(_make_service())
    try:
        response = client.get(
            "/api/metal-composition/section-232/review?batch_id=missing",
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()

def test_get_section_232_review_workspace_returns_404_for_missing_version():
    client = _client_with_service(_make_service())
    try:
        response = client.get(
            "/api/metal-composition/section-232/review?version=missing",
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()

def test_section_232_direct_classification_requires_active_published_ruleset():
    client = _client_with_service(_make_service())
    try:
        response = client.post(
            "/api/metal-composition/section-232/classify",
            json={
                "hts_code": "7407.10.5000",
                "top_level_grams": {
                    "steel": 0.0,
                    "aluminum": 0.0,
                    "copper": 10.0,
                },
                "total_weight_grams": 100.0,
                "metal_share_certainty": "exact",
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["section_232_assessment"]["decision"] == "needs_review"
        assert payload["section_232_reasoner_output"]["strategy"] == "published_ruleset"
        assert payload["section_232_reasoner_output"]["fallback_used"] is False
        assert payload["section_232_reasoner_output"]["reason"] == "no_active_ruleset"
        assert payload["section_232_assessment"]["weight_rule_applied"] is False
        assert payload["section_232_reasoner_output"]["metal_weight_override"]["reason"] == "legal_coverage_not_established"
    finally:
        app.dependency_overrides.clear()

def test_section_232_direct_classification_uses_published_ruleset_metadata():
    service = _make_service()
    _publish_section_232_rule(
        service,
        hts_code="7407.10.50",
        rule_type="rate_schedule",
        metal_scope="copper",
        effective_to="2027-12-31",
    )
    client = _client_with_service(service)
    try:
        response = client.post(
            "/api/metal-composition/section-232/classify",
            json={
                "hts_code": "7407.10.5050",
                "top_level_grams": {
                    "steel": 0.0,
                    "aluminum": 0.0,
                    "copper": 20.0,
                },
                "total_weight_grams": 100.0,
                "metal_share_certainty": "exact",
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["section_232_assessment"]["decision"] == "subject"
        assert payload["section_232_reasoner_output"]["strategy"] == "published_ruleset"
        assert payload["section_232_reasoner_output"]["fallback_used"] is False
        assert payload["section_232_reasoner_output"]["active_ruleset_version"] == "section232-v0001"
        assert payload["section_232_reasoner_output"]["matched_rule_type"] == "rate_schedule"
        assert payload["section_232_reasoner_output"]["matched_hts_code"] == "7407.10.50"
        assert "tariff_rate" not in payload["section_232_reasoner_output"]
        assert payload["section_232_reasoner_output"]["effective_window"] == {
            "evaluation_date": date.today().isoformat(),
            "effective_from": "2026-04-07",
            "effective_to": "2027-12-31",
            "is_active": True,
        }
    finally:
        app.dependency_overrides.clear()

def test_section_232_source_snippets_do_not_promote_sibling_codes_to_parent_family_matches():
    service = _make_service()
    service.section_232_source_store.save_source(
        filename="section-232-notice.pdf",
        size_bytes=123,
        extracted=ExtractedSection232Source(
            page_count=1,
            extraction_status="completed",
            full_text=(
                "[Page 1]\n"
                "Section 232 derivative list includes HTS 8413.81.00 for covered aluminum products."
            ),
            page_texts=[
                {
                    "page_number": 1,
                    "text": "Section 232 derivative list includes HTS 8413.81.00 for covered aluminum products.",
                    "hts_mentions": ["8413.81.00"],
                }
            ],
            hts_mentions=["8413.81.00"],
            warnings=[],
        ),
    )

    snippets = service.section_232_source_store.retrieve_snippets(
        hts_codes=["8413.70.2090"],
        metal_keywords=["aluminum"],
        settings=service.settings,
    )

    assert snippets == []

def test_section_232_source_snippets_match_ancestor_codes_for_selected_item():
    service = _make_service()
    service.section_232_source_store.save_source(
        filename="section-232-notice.pdf",
        size_bytes=123,
        extracted=ExtractedSection232Source(
            page_count=1,
            extraction_status="completed",
            full_text=(
                "[Page 1]\n"
                "Section 232 derivative list includes HTS 8413.70 for covered aluminum products."
            ),
            page_texts=[
                {
                    "page_number": 1,
                    "text": "Section 232 derivative list includes HTS 8413.70 for covered aluminum products.",
                    "hts_mentions": ["8413.70"],
                }
            ],
            hts_mentions=["8413.70"],
            warnings=[],
        ),
    )

    snippets = service.section_232_source_store.retrieve_snippets(
        hts_codes=["8413.70.2090"],
        metal_keywords=["aluminum"],
        settings=service.settings,
    )

    assert len(snippets) == 1
    assert snippets[0].matched_codes == ["8413.70"]

def test_section_232_direct_classification_applies_weight_override_for_exact_and_estimated_inputs():
    service = _make_service()
    _publish_section_232_rule(
        service,
        hts_code="7407.10.50",
        rule_type="rate_schedule",
        metal_scope="copper",
    )
    client = _client_with_service(service)
    try:
        exact_response = client.post(
            "/api/metal-composition/section-232/classify",
            json={
                "hts_code": "7407.10.5000",
                "top_level_grams": {
                    "steel": 0.0,
                    "aluminum": 0.0,
                    "copper": 10.0,
                },
                "total_weight_grams": 100.0,
                "metal_share_certainty": "exact",
            },
            headers={"X-API-Key": "test-api-key"},
        )
        estimated_response = client.post(
            "/api/metal-composition/section-232/classify",
            json={
                "hts_code": "7407.10.5000",
                "top_level_grams": {
                    "steel": 0.0,
                    "aluminum": 0.0,
                    "copper": 10.0,
                },
                "total_weight_grams": 100.0,
                "metal_share_certainty": "estimated",
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert exact_response.status_code == 200
        exact_payload = exact_response.json()
        assert exact_payload["section_232_assessment"]["decision"] == "not_subject"
        assert exact_payload["section_232_assessment"]["weight_rule_applied"] is True
        assert exact_payload["section_232_assessment"]["needs_human_review"] is False
        assert exact_payload["section_232_reasoner_output"]["metal_weight_override"]["reason"] == "exact_below_threshold"

        assert estimated_response.status_code == 200
        estimated_payload = estimated_response.json()
        assert estimated_payload["section_232_assessment"]["decision"] == "not_subject"
        assert estimated_payload["section_232_assessment"]["weight_rule_applied"] is True
        assert estimated_payload["section_232_assessment"]["needs_human_review"] is True
        assert estimated_payload["section_232_reasoner_output"]["metal_weight_override"]["reason"] == "estimated_below_threshold"
    finally:
        app.dependency_overrides.clear()

def test_section_232_direct_classification_uses_family_rule_then_weight_exemption_for_plastic_item():
    service = _make_service()
    _publish_section_232_rule(
        service,
        hts_code="7616.99.51",
        rule_type="include",
    )
    client = _client_with_service(service)
    try:
        response = client.post(
            "/api/metal-composition/section-232/classify",
            json={
                "hts_code": "7616.99.5190",
                "top_level_grams": {
                    "steel": 0.0,
                    "aluminum": 0.0,
                    "copper": 0.0,
                },
                "total_weight_grams": 100.0,
                "metal_share_certainty": "exact",
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["section_232_assessment"]["decision"] == "not_subject"
        assert payload["section_232_assessment"]["confidence"] == 0.95
        assert payload["section_232_assessment"]["needs_human_review"] is False
        assert payload["section_232_assessment"]["weight_rule_applied"] is True
        assert payload["section_232_reasoner_output"]["matched_rule_type"] == "include"
        assert payload["section_232_reasoner_output"]["reason"] == "matched_rule"
        assert payload["section_232_reasoner_output"]["metal_weight_override"]["reason"] == "exact_below_threshold"
    finally:
        app.dependency_overrides.clear()

def test_section_232_direct_classification_uses_published_parent_family_match():
    service = _make_service()
    _publish_section_232_rule(
        service,
        hts_code="8483.30.80",
        rule_type="include",
    )
    client = _client_with_service(service)
    try:
        response = client.post(
            "/api/metal-composition/section-232/classify",
            json={
                "hts_code": "8483.30.8040",
                "top_level_grams": {
                    "steel": 20.0,
                    "aluminum": 0.0,
                    "copper": 0.0,
                },
                "total_weight_grams": 100.0,
                "metal_share_certainty": "exact",
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["section_232_assessment"]["decision"] == "subject"
        assert payload["section_232_reasoner_output"]["matched_rule_type"] == "include"
        assert payload["section_232_reasoner_output"]["matched_hts_code"] == "8483.30.80"
        assert payload["section_232_reasoner_output"]["metal_weight_override"]["reason"] == "threshold_not_met"
    finally:
        app.dependency_overrides.clear()

def test_section_232_direct_classification_validates_request_body():
    client = _client_with_service(_make_service())
    try:
        missing_top_level = client.post(
            "/api/metal-composition/section-232/classify",
            json={
                "hts_code": "7407.10.5000",
                "total_weight_grams": 100.0,
                "metal_share_certainty": "exact",
            },
            headers={"X-API-Key": "test-api-key"},
        )
        invalid_certainty = client.post(
            "/api/metal-composition/section-232/classify",
            json={
                "hts_code": "7407.10.5000",
                "top_level_grams": {
                    "steel": 0.0,
                    "aluminum": 0.0,
                    "copper": 10.0,
                },
                "total_weight_grams": 100.0,
                "metal_share_certainty": "approximate",
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert missing_top_level.status_code == 422
        assert invalid_certainty.status_code == 422
    finally:
        app.dependency_overrides.clear()

def test_section_232_direct_classification_defaults_certainty_to_exact():
    service = _make_service()
    _publish_section_232_rule(
        service,
        hts_code="7407.10.50",
        rule_type="rate_schedule",
        metal_scope="copper",
    )
    client = _client_with_service(service)
    try:
        response = client.post(
            "/api/metal-composition/section-232/classify",
            json={
                "hts_code": "7407.10.5000",
                "top_level_grams": {
                    "steel": 0.0,
                    "aluminum": 0.0,
                    "copper": 10.0,
                },
                "total_weight_grams": 100.0
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["section_232_assessment"]["decision"] == "not_subject"
        assert payload["section_232_assessment"]["needs_human_review"] is False
        assert payload["section_232_reasoner_output"]["metal_weight_override"]["metal_share_certainty"] == "exact"
        assert payload["section_232_reasoner_output"]["metal_weight_override"]["reason"] == "exact_below_threshold"
    finally:
        app.dependency_overrides.clear()

def test_section_232_reset_clears_sources_pending_drafts_and_published_rulesets():
    service = _make_service()
    extracted = ExtractedSection232Source(
        page_count=1,
        extraction_status="completed",
        full_text="[Page 1]\nHTS 7308.90.95 remains covered.",
        page_texts=[
            {
                "page_number": 1,
                "text": "HTS 7308.90.95 remains covered.",
                "layout_aware_text": "HTS 7308.90.95 remains covered.",
                "page_excerpt": "HTS 7308.90.95 remains covered.",
                "char_count": 32,
                "hts_mentions": ["7308.90.95"],
            }
        ],
        hts_mentions=["7308.90.95"],
        warnings=[],
    )
    source = service.section_232_source_store.save_source(
        filename="annex-reset.pdf",
        size_bytes=123,
        extracted=extracted,
    )

    pending_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=[source.source_id],
        source_filenames=[source.filename],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        pending_batch.batch_id,
        [
            Section232DraftRuleCandidate(
                candidate_id="pending-include",
                batch_id=pending_batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
                effective_from="2026-04-07",
                effective_to=None,
                metal_scope="steel",
                source_document_ids=[source.source_id],
                source_pages=[1],
                source_excerpt="Pending draft rule.",
                interpreter_confidence=0.9,
                catalog_match_found=True,
                review_decision="pending",
            )
        ],
    )

    published_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=[source.source_id],
        source_filenames=[source.filename],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        published_batch.batch_id,
        [
            Section232DraftRuleCandidate(
                candidate_id="published-include",
                batch_id=published_batch.batch_id,
                hts_code="8481.80.30.90",
                rule_type="include",
                coverage_effect="include",
                effective_from="2026-04-07",
                effective_to=None,
                metal_scope="steel",
                source_document_ids=[source.source_id],
                source_pages=[1],
                source_excerpt="Published rule.",
                interpreter_confidence=0.95,
                catalog_match_found=True,
                review_decision="accepted",
            )
        ],
    )
    service.section_232_ruleset_store.publish_batch(published_batch.batch_id, published_by="pytest")

    client = _client_with_service(service)
    try:
        reset_response = client.post(
            "/api/metal-composition/section-232/reset",
            headers={"X-API-Key": "test-api-key"},
        )

        assert reset_response.status_code == 200
        assert reset_response.json() == {
            "cleared_source_count": 1,
            "cleared_draft_batch_count": 2,
            "cleared_draft_rule_count": 2,
            "cleared_delete_override_count": 0,
            "cleared_published_ruleset_count": 1,
            "cleared_published_rule_count": 1,
        }

        source_listing = client.get(
            "/api/metal-composition/section-232/sources",
            headers={"X-API-Key": "test-api-key"},
        )
        assert source_listing.status_code == 200
        assert source_listing.json() == {
            "total": 0,
            "eligible_hts_code_count": 0,
            "items": [],
        }

        summary_response = client.get(
            "/api/metal-composition/section-232/ruleset-summary",
            headers={"X-API-Key": "test-api-key"},
        )
        assert summary_response.status_code == 200
        assert summary_response.json() == {
            "active_ruleset_version": None,
            "eligible_hts_code_count": 0,
            "pending_draft_batch_count": 0,
            "pending_draft_batches": [],
            "last_published_at": None,
        }

        eligible_response = client.get(
            "/api/metal-composition/section-232/eligible-hts-codes",
            headers={"X-API-Key": "test-api-key"},
        )
        assert eligible_response.status_code == 200
        assert eligible_response.json() == {"total": 0, "codes": []}
    finally:
        app.dependency_overrides.clear()

def _make_chapter_csv(*, code: str, description: str) -> bytes:
    frame = pd.DataFrame(
        [
            {
                "HTS Number": code,
                "Indent": 0,
                "Description": description,
                "Unit of Quantity": "No.",
                "General Rate of Duty": "Free",
                "Special Rate of Duty": "",
                "Column 2 Rate of Duty": "",
                "Quota Quantity": "",
                "Additional Duties": "",
            }
        ]
    )
    return frame.to_csv(index=False).encode("utf-8")

def test_hts_catalog_source_endpoints_upload_refresh_and_replace_by_filename(monkeypatch, tmp_path):
    service = _make_ui_service(tmp_path)
    refresh_calls: list[dict] = []

    def _fake_refresh_hts_catalog_tables(*, settings, csv_dir, code_map_path, **_kwargs):
        refresh_calls.append(
            {
                "csv_dir": Path(csv_dir),
                "code_map_path": Path(code_map_path),
            }
        )
        return {
            "status": "completed",
            "catalog_row_count": len(compile_hts_catalog_frame(csv_dir=Path(csv_dir))),
            "code_map_row_count": len(compile_hts_code_map_frame(code_map_path=Path(code_map_path))),
        }

    monkeypatch.setattr(
        "app.services.metal_composition.service.refresh_hts_catalog_tables",
        _fake_refresh_hts_catalog_tables,
    )

    client = _client_with_service(service)
    try:
        initial_listing = client.get(
            "/api/metal-composition/hts-catalog/sources",
            headers={"X-API-Key": "test-api-key"},
        )
        assert initial_listing.status_code == 200
        initial_payload = initial_listing.json()
        assert 86 not in initial_payload["summary"]["loaded_chapters"]

        upload_response = client.post(
            "/api/metal-composition/hts-catalog/sources",
            files=[
                (
                    "files",
                    (
                        "chapter86.csv",
                        _make_chapter_csv(code="8607.19.06.00", description="Axles and wheels for rolling stock"),
                        "text/csv",
                    ),
                )
            ],
            headers={"X-API-Key": "test-api-key"},
        )
        assert upload_response.status_code == 201
        upload_payload = upload_response.json()
        assert upload_payload["uploaded_file_count"] == 1
        assert upload_payload["overwritten_file_count"] == 0
        assert 86 in upload_payload["summary"]["loaded_chapters"]
        assert upload_payload["summary"]["last_refresh_status"] == "completed"
        assert refresh_calls
        assert service.workflow_runner.hts_catalog_resolver is not None
        assert 86 in {
            int(value)
            for value in service.workflow_runner.hts_catalog_resolver.catalog_frame["chapter_number"].dropna().unique()
        }

        replace_response = client.post(
            "/api/metal-composition/hts-catalog/sources",
            files=[
                (
                    "files",
                    (
                        "chapter86.csv",
                        _make_chapter_csv(code="8607.19.09.00", description="Other railway rolling-stock parts"),
                        "text/csv",
                    ),
                )
            ],
            headers={"X-API-Key": "test-api-key"},
        )
        assert replace_response.status_code == 201
        replace_payload = replace_response.json()
        assert replace_payload["overwritten_file_count"] == 1
    finally:
        app.dependency_overrides.clear()

def test_hts_catalog_source_upload_infers_chapter_from_official_hts_csv(monkeypatch, tmp_path):
    service = _make_ui_service(tmp_path)
    refresh_calls: list[dict] = []

    def _fake_refresh_hts_catalog_tables(*, settings, csv_dir, code_map_path, **_kwargs):
        refresh_calls.append(
            {
                "csv_dir": Path(csv_dir),
                "code_map_path": Path(code_map_path),
            }
        )
        return {
            "status": "completed",
            "catalog_row_count": len(compile_hts_catalog_frame(csv_dir=Path(csv_dir))),
            "code_map_row_count": len(compile_hts_code_map_frame(code_map_path=Path(code_map_path))),
        }

    monkeypatch.setattr(
        "app.services.metal_composition.service.refresh_hts_catalog_tables",
        _fake_refresh_hts_catalog_tables,
    )

    client = _client_with_service(service)
    try:
        upload_response = client.post(
            "/api/metal-composition/hts-catalog/sources",
            files=[
                (
                    "files",
                    (
                        "htsdata.csv",
                        _make_chapter_csv(code="8607.19.06.00", description="Axles and wheels for rolling stock"),
                        "text/csv",
                    ),
                )
            ],
            headers={"X-API-Key": "test-api-key"},
        )
        assert upload_response.status_code == 201
        upload_payload = upload_response.json()
        assert upload_payload["uploaded_file_count"] == 1
        assert upload_payload["overwritten_file_count"] == 0
        assert upload_payload["items"][0]["filename"] == "chapter86.csv"
        assert upload_payload["items"][0]["chapter_number"] == 86
        assert 86 in upload_payload["summary"]["loaded_chapters"]
        assert refresh_calls
        assert service.workflow_runner.hts_catalog_resolver is not None
        assert 86 in {
            int(value)
            for value in service.workflow_runner.hts_catalog_resolver.catalog_frame["chapter_number"].dropna().unique()
        }

        replace_response = client.post(
            "/api/metal-composition/hts-catalog/sources",
            files=[
                (
                    "files",
                    (
                        "htsdata.csv",
                        _make_chapter_csv(code="8607.19.09.00", description="Other railway rolling-stock parts"),
                        "text/csv",
                    ),
                )
            ],
            headers={"X-API-Key": "test-api-key"},
        )
        assert replace_response.status_code == 201
        replace_payload = replace_response.json()
        assert replace_payload["uploaded_file_count"] == 1
        assert replace_payload["overwritten_file_count"] == 1
        assert replace_payload["items"][0]["filename"] == "chapter86.csv"
    finally:
        app.dependency_overrides.clear()

def test_hts_catalog_source_upload_rejects_invalid_files_without_mutating_state(monkeypatch, tmp_path):
    service = _make_ui_service(tmp_path)
    client = _client_with_service(service)
    try:
        initial_listing = client.get(
            "/api/metal-composition/hts-catalog/sources",
            headers={"X-API-Key": "test-api-key"},
        )
        initial_payload = initial_listing.json()

        monkeypatch.setattr(
            "app.services.metal_composition.service.refresh_hts_catalog_tables",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("refresh should not run on invalid uploads")),
        )

        invalid_response = client.post(
            "/api/metal-composition/hts-catalog/sources",
            files=[("files", ("bad.csv", b"wrong,columns\n1,2\n", "text/csv"))],
            headers={"X-API-Key": "test-api-key"},
        )
        assert invalid_response.status_code == 422
        assert "chapterNN.csv or hts_code_map.csv" in invalid_response.json()["detail"]

        invalid_schema_response = client.post(
            "/api/metal-composition/hts-catalog/sources",
            files=[("files", ("chapter86.csv", b"wrong,columns\n1,2\n", "text/csv"))],
            headers={"X-API-Key": "test-api-key"},
        )
        assert invalid_schema_response.status_code == 422
        assert "missing required columns" in invalid_schema_response.json()["detail"]

        post_listing = client.get(
            "/api/metal-composition/hts-catalog/sources",
            headers={"X-API-Key": "test-api-key"},
        )
        assert post_listing.status_code == 200
        assert post_listing.json()["summary"]["loaded_chapters"] == initial_payload["summary"]["loaded_chapters"]
    finally:
        app.dependency_overrides.clear()

def test_hts_catalog_source_delete_removes_chapter_and_refreshes_catalog(monkeypatch, tmp_path):
    service = _make_ui_service(tmp_path)
    refresh_calls: list[dict] = []

    def _fake_refresh_hts_catalog_tables(*, settings, csv_dir, code_map_path, **_kwargs):
        refresh_calls.append(
            {
                "csv_dir": Path(csv_dir),
                "code_map_path": Path(code_map_path),
            }
        )
        return {
            "status": "completed",
            "catalog_row_count": len(compile_hts_catalog_frame(csv_dir=Path(csv_dir))),
            "code_map_row_count": len(compile_hts_code_map_frame(code_map_path=Path(code_map_path))),
        }

    monkeypatch.setattr(
        "app.services.metal_composition.service.refresh_hts_catalog_tables",
        _fake_refresh_hts_catalog_tables,
    )

    client = _client_with_service(service)
    try:
        initial_listing = client.get(
            "/api/metal-composition/hts-catalog/sources",
            headers={"X-API-Key": "test-api-key"},
        )
        assert initial_listing.status_code == 200
        assert 72 in initial_listing.json()["summary"]["loaded_chapters"]

        delete_response = client.delete(
            "/api/metal-composition/hts-catalog/sources/chapter72.csv",
            headers={"X-API-Key": "test-api-key"},
        )
        assert delete_response.status_code == 200
        delete_payload = delete_response.json()
        assert delete_payload["deleted_filename"] == "chapter72.csv"
        assert 72 not in delete_payload["summary"]["loaded_chapters"]
        assert refresh_calls
        assert service.workflow_runner.hts_catalog_resolver is not None
        assert 72 not in {
            int(value)
            for value in service.workflow_runner.hts_catalog_resolver.catalog_frame["chapter_number"].dropna().unique()
        }

        post_listing = client.get(
            "/api/metal-composition/hts-catalog/sources",
            headers={"X-API-Key": "test-api-key"},
        )
        assert post_listing.status_code == 200
        assert all(item["filename"] != "chapter72.csv" for item in post_listing.json()["items"])
    finally:
        app.dependency_overrides.clear()

def test_hts_catalog_source_delete_rejects_removing_last_remaining_chapter(monkeypatch, tmp_path):
    hts_catalog_dir = tmp_path / "hts"
    hts_catalog_dir.mkdir()
    (hts_catalog_dir / "chapter86.csv").write_bytes(
        _make_chapter_csv(code="8607.19.06.00", description="Axles and wheels for rolling stock")
    )

    service = _make_ui_service(
        tmp_path,
        hts_catalog_dir=hts_catalog_dir,
        hts_code_map_path=hts_catalog_dir / "hts_code_map.csv",
    )
    client = _client_with_service(service)
    try:
        monkeypatch.setattr(
            "app.services.metal_composition.service.refresh_hts_catalog_tables",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("refresh should not run when delete is rejected")),
        )

        delete_response = client.delete(
            "/api/metal-composition/hts-catalog/sources/chapter86.csv",
            headers={"X-API-Key": "test-api-key"},
        )
        assert delete_response.status_code == 422
        assert "At least one HTS chapter CSV must remain" in delete_response.json()["detail"]
    finally:
        app.dependency_overrides.clear()

def _build_ui_workbook_store() -> WorkbookStore:
    source_df = pd.DataFrame(
        [
            {
                "source_row_id": 1001,
                "normalized_product_code": "wi-100",
                "Product code": "WI-100",
                "PN Revised/ Standardized": "PN-WI-100",
                "Part description": "Pump casing",
                "New Part Description": "Pump casing assembly",
                "Site": "Madrid",
                "Business Segment": "WI",
                "Priority": "P1",
                "Priority.1": "Pump spares",
                "Material Content Method": "ERP Extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 18.0,
                "Date Started": "2025-01-01",
                "Date Completed": "2025-01-02",
            },
            {
                "source_row_id": 1002,
                "normalized_product_code": "aws-200",
                "Product code": "AWS-200",
                "PN Revised/ Standardized": "PN-AWS-200",
                "Part description": "Unsupported segment part",
                "New Part Description": "Unsupported segment part",
                "Site": "Berlin",
                "Business Segment": "AWS",
                "Priority": "P2",
                "Priority.1": "Special projects",
                "Material Content Method": "ERP Extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 10.0,
                "Date Started": "2025-02-01",
                "Date Completed": "2025-02-05",
            },
        ]
    )
    prepared_df = pd.DataFrame(
        [
            {"source_row_id": 1001, "row_id": 1, "Total Weight (Gram)": 18.0},
            {"source_row_id": 1002, "row_id": 2, "Total Weight (Gram)": 10.0},
        ]
    )
    return WorkbookStore(source_df=source_df, prepared_df=prepared_df)

def _make_ui_service(
    tmp_path: Path,
    *,
    ui_state_store: InMemoryMetalCompositionUIStateStore | None = None,
    hts_catalog_dir: Path | None = None,
    hts_code_map_path: Path | None = None,
) -> MetalCompositionService:
    uploaded_document_root = tmp_path / "uploaded-docs"

    settings = MetalCompositionSettings(
        workbook_path=Path("/tmp/unused.xlsb"),
        api_env_path=Path("/tmp/api.env"),
        uploaded_document_root=uploaded_document_root,
        ui_state_db_path=tmp_path / "ui_state.sqlite3",
        batch_max_items=10,
        hts_catalog_dir=hts_catalog_dir or HTS_CSV_DIR,
        hts_code_map_path=hts_code_map_path or HTS_CODE_MAP,
    )
    return MetalCompositionService(
        serving_store=_build_ui_workbook_store(),
        workflow_runner=FakeWorkflowRunner(),
        settings=settings,
        ui_state_store=ui_state_store or InMemoryMetalCompositionUIStateStore(),
        section_232_source_store=InMemorySection232SourceStore(),
        hts_catalog_source_store=InMemoryHTSCatalogSourceStore(settings),
    )

def _build_duplicate_ui_workbook_store() -> WorkbookStore:
    source_df = pd.DataFrame(
        [
            {
                "source_row_id": 2001,
                "normalized_product_code": "dup-232",
                "Product code": "DUP-232",
                "PN Revised/ Standardized": "PN-DUP-232-A",
                "Part description": "Bearing housing upper variant A",
                "New Part Description": "Bearing housing upper variant A",
                "Site": "Madrid",
                "Business Segment": "WI",
                "Priority": "P1",
                "Priority.1": "Duplicate row A",
                "Material Content Method": "ERP Extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 21.0,
                "Date Started": "2025-03-01",
                "Date Completed": "2025-03-02",
            },
            {
                "source_row_id": 2002,
                "normalized_product_code": "dup-232",
                "Product code": "DUP-232",
                "PN Revised/ Standardized": "PN-DUP-232-B",
                "Part description": "Bearing housing upper variant B",
                "New Part Description": "Bearing housing upper variant B",
                "Site": "Madrid",
                "Business Segment": "WI",
                "Priority": "P2",
                "Priority.1": "Duplicate row B",
                "Material Content Method": "ERP Extract",
                "MaterialIdentified": "1",
                "Total Weight (Gram)": 34.0,
                "Date Started": "2025-03-03",
                "Date Completed": "2025-03-04",
            },
        ]
    )
    prepared_df = pd.DataFrame(
        [
            {"source_row_id": 2001, "row_id": 1, "Total Weight (Gram)": 21.0},
            {"source_row_id": 2002, "row_id": 2, "Total Weight (Gram)": 34.0},
        ]
    )
    return WorkbookStore(source_df=source_df, prepared_df=prepared_df)

def _make_duplicate_ui_service(tmp_path: Path) -> MetalCompositionService:
    uploaded_document_root = tmp_path / "uploaded-docs"

    settings = MetalCompositionSettings(
        workbook_path=Path("/tmp/unused.xlsb"),
        api_env_path=Path("/tmp/api.env"),
        uploaded_document_root=uploaded_document_root,
        ui_state_db_path=tmp_path / "ui_state.sqlite3",
        batch_max_items=10,
    )
    return MetalCompositionService(
        serving_store=_build_duplicate_ui_workbook_store(),
        workflow_runner=FakeWorkflowRunner(),
        settings=settings,
        ui_state_store=InMemoryMetalCompositionUIStateStore(),
        section_232_source_store=InMemorySection232SourceStore(),
        hts_catalog_source_store=InMemoryHTSCatalogSourceStore(settings),
    )

def test_items_endpoint_supports_boolean_filters(tmp_path):
    service = _make_ui_service(tmp_path)
    service.upload_item_document(
        "mm:1001",
        filename="uploaded-mm.pdf",
        content=b"%PDF-1.4 uploaded mm",
    )
    service.classify_item("mm:1001")
    client = _client_with_service(service)
    try:
        response = client.get(
            "/api/metal-composition/items",
            params={"has_documents": "true", "is_classified": "true"},
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["items"][0]["item_id"] == "mm:1001"
        assert body["items"][0]["has_documents"] is True
        assert body["items"][0]["is_classified"] is True
    finally:
        app.dependency_overrides.clear()

def test_document_upload_endpoint_returns_not_found_for_unknown_item(tmp_path):
    client = _client_with_service(_make_ui_service(tmp_path))
    try:
        upload_response = client.post(
            "/api/metal-composition/items/manual:missing/documents/upload",
            files={"file": ("manual.pdf", b"%PDF-1.4 uploaded", "application/pdf")},
            headers={"X-API-Key": "test-api-key"},
        )

        assert upload_response.status_code == 404
    finally:
        app.dependency_overrides.clear()

def test_item_classification_endpoints_persist_results(tmp_path):
    service = _make_ui_service(tmp_path)
    _upload_fake_pdf(service, "mm:1001", "persist-mm-1001.pdf")
    _upload_fake_pdf(service, "mm:1002", "persist-mm-1002.pdf")
    client = _client_with_service(service)
    try:
        classify_response = client.post(
            "/api/metal-composition/items/mm:1001/classify",
            headers={"X-API-Key": "test-api-key"},
        )
        assert classify_response.status_code == 202
        classify_body = classify_response.json()
        assert classify_body["job_type"] == "single"
        assert classify_body["status"] == "queued"
        job_status = _drain_job(client, service, classify_body["job_id"])
        assert job_status["status"] == "completed"
        assert job_status["completed_count"] == 1
        assert job_status["failed_count"] == 0

        detail_response = client.get(
            "/api/metal-composition/items/mm:1001",
            headers={"X-API-Key": "test-api-key"},
        )
        assert detail_response.status_code == 200
        detail_body = detail_response.json()
        assert detail_body["latest_classification"]["status"] == "completed"
        assert detail_body["last_classified_at"] is not None

        batch_response = client.post(
            "/api/metal-composition/items/classify-batch",
            json={"item_ids": ["mm:1001", "mm:1002"]},
            headers={"X-API-Key": "test-api-key"},
        )
        assert batch_response.status_code == 202
        body = batch_response.json()
        assert body["job_type"] == "batch"
        assert body["total_count"] == 2
        batch_job_status = _drain_job(client, service, body["job_id"])
        assert batch_job_status["status"] == "completed"
        assert batch_job_status["completed_count"] == 2
        assert batch_job_status["failed_count"] == 0
    finally:
        app.dependency_overrides.clear()

def test_item_predict_endpoint_text_only_submits_without_pdfs(tmp_path):
    service = _make_ui_service(tmp_path)
    client = _client_with_service(service)
    try:
        response = client.post(
            "/api/metal-composition/items/predict",
            json={"item_id": "mm:1002"},
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 202
        body = response.json()
        assert body["job_type"] == "single"
        _drain_job(client, service, body["job_id"])
        detail_response = client.get(
            "/api/metal-composition/items/mm:1002",
            headers={"X-API-Key": "test-api-key"},
        )
        assert detail_response.status_code == 200
        assert detail_response.json()["latest_classification"]["document_mode"] == "text_only"
    finally:
        app.dependency_overrides.clear()

def test_item_predict_endpoint_accepts_token_usage_diagnostic_flag(tmp_path):
    service = _make_ui_service(tmp_path)
    client = _client_with_service(service)
    try:
        response = client.post(
            "/api/metal-composition/items/predict",
            json={
                "item_id": "mm:1002",
                "document_mode": "text_only",
                "include_token_usage": True,
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 202
        body = response.json()
        job_items = service.classification_job_store.get_job_items(body["job_id"])
        assert job_items[0].include_token_usage is True
        _drain_job(client, service, body["job_id"])
        detail_response = client.get(
            "/api/metal-composition/items/mm:1002",
            headers={"X-API-Key": "test-api-key"},
        )
        assert detail_response.status_code == 200
        token_usage = detail_response.json()["latest_classification"]["token_usage"]
        assert token_usage["total_tokens"] == 260
        assert [entry["task"] for entry in token_usage["entries"]] == ["base_pass", "heading_router"]
    finally:
        app.dependency_overrides.clear()

def test_item_predict_batch_endpoint_persists_document_mode(tmp_path):
    service = _make_ui_service(tmp_path)
    _upload_fake_pdf(service, "mm:1001", "predict-batch-evidence.pdf")
    client = _client_with_service(service)
    try:
        response = client.post(
            "/api/metal-composition/items/predict-batch",
            json={
                "item_ids": ["mm:1001"],
                "document_mode": "with_documents",
                "include_token_usage": True,
            },
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 202
        body = response.json()
        job_items = service.classification_job_store.get_job_items(body["job_id"])
        assert job_items[0].document_mode == "with_documents"
        assert job_items[0].include_token_usage is True
        _drain_job(client, service, body["job_id"])
        detail_response = client.get(
            "/api/metal-composition/items/mm:1001",
            headers={"X-API-Key": "test-api-key"},
        )
        assert detail_response.json()["latest_classification"]["document_mode"] == "with_documents"
        assert detail_response.json()["latest_classification"]["token_usage"]["total_tokens"] == 260
    finally:
        app.dependency_overrides.clear()

def test_deprecated_product_code_predict_routes_are_removed(tmp_path):
    client = _client_with_service(_make_ui_service(tmp_path))
    try:
        single_response = client.post(
            "/api/metal-composition" + "/predict",
            headers={"X-API-Key": "test-api-key"},
        )
        batch_response = client.post(
            "/api/metal-composition" + "/predict" + "/batch",
            headers={"X-API-Key": "test-api-key"},
        )

        assert single_response.status_code == 404
        assert batch_response.status_code == 404
    finally:
        app.dependency_overrides.clear()

def test_item_classification_endpoint_reruns_overlapping_job(tmp_path):
    service = _make_ui_service(tmp_path)
    _upload_fake_pdf(service, "mm:1001", "rerun-mm.pdf")
    client = _client_with_service(service)
    try:
        first_response = client.post(
            "/api/metal-composition/items/mm:1001/classify",
            headers={"X-API-Key": "test-api-key"},
        )
        second_response = client.post(
            "/api/metal-composition/items/mm:1001/classify",
            headers={"X-API-Key": "test-api-key"},
        )

        assert first_response.status_code == 202
        assert second_response.status_code == 202

        first_job_id = first_response.json()["job_id"]
        second_job_id = second_response.json()["job_id"]
        first_job = service.get_classification_job(first_job_id)
        first_items = service.classification_job_store.get_job_items(first_job_id)

        assert second_job_id != first_job_id
        assert first_job.status == "failed"
        assert first_items[0].error_message == SUPERSEDED_ERROR_MESSAGE
    finally:
        app.dependency_overrides.clear()

def test_batch_classification_endpoint_reruns_overlapping_job(tmp_path):
    service = _make_ui_service(tmp_path)
    _upload_fake_pdf(service, "mm:1001", "batch-rerun-mm.pdf")
    client = _client_with_service(service)
    try:
        first_response = client.post(
            "/api/metal-composition/items/classify-batch",
            json={"item_ids": ["mm:1001"]},
            headers={"X-API-Key": "test-api-key"},
        )
        second_response = client.post(
            "/api/metal-composition/items/classify-batch",
            json={"item_ids": ["mm:1001"]},
            headers={"X-API-Key": "test-api-key"},
        )

        assert first_response.status_code == 202
        assert second_response.status_code == 202

        first_job_id = first_response.json()["job_id"]
        second_job_id = second_response.json()["job_id"]
        first_job = service.get_classification_job(first_job_id)
        first_items = service.classification_job_store.get_job_items(first_job_id)

        assert second_job_id != first_job_id
        assert first_job.status == "failed"
        assert first_items[0].error_message == SUPERSEDED_ERROR_MESSAGE
    finally:
        app.dependency_overrides.clear()

def test_export_report_endpoint_rejects_missing_snapshot(tmp_path):
    client = _client_with_service(_make_ui_service(tmp_path))
    try:
        response = client.get(
            "/api/metal-composition/items/mm:1001/export-report",
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 409
        assert response.json()["detail"] == "A completed classification is required before exporting a report."
    finally:
        app.dependency_overrides.clear()

def test_export_report_endpoint_rejects_failed_snapshot(tmp_path):
    service = _make_ui_service(tmp_path)
    service.ui_state_store.save_classification_snapshot(
        "mm:1001",
        dataset_scope=service.dataset_signature,
        result=MetalCompositionResponse(status="failed", product_code="WI-100", error="workflow exploded"),
        last_classified_at="2026-04-12T00:20:00Z",
    )
    client = _client_with_service(service)
    try:
        response = client.get(
            "/api/metal-composition/items/mm:1001/export-report",
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 409
        assert response.json()["detail"] == "Only completed classifications can be exported as a report."
    finally:
        app.dependency_overrides.clear()

def test_item_classification_uses_source_row_id_for_duplicate_product_codes(tmp_path):
    service = _make_duplicate_ui_service(tmp_path)
    _upload_fake_pdf(service, "mm:2001", "duplicate-2001.pdf")
    _upload_fake_pdf(service, "mm:2002", "duplicate-2002.pdf")
    client = _client_with_service(service)
    try:
        first_response = client.post(
            "/api/metal-composition/items/mm:2001/classify",
            headers={"X-API-Key": "test-api-key"},
        )
        second_response = client.post(
            "/api/metal-composition/items/mm:2002/classify",
            headers={"X-API-Key": "test-api-key"},
        )

        assert first_response.status_code == 202
        assert second_response.status_code == 202
        first_body = first_response.json()
        second_body = second_response.json()
        _drain_job(client, service, first_body["job_id"])
        _drain_job(client, service, second_body["job_id"])

        first_detail = client.get(
            "/api/metal-composition/items/mm:2001",
            headers={"X-API-Key": "test-api-key"},
        )
        second_detail = client.get(
            "/api/metal-composition/items/mm:2002",
            headers={"X-API-Key": "test-api-key"},
        )
        assert first_detail.status_code == 200
        assert second_detail.status_code == 200
        assert first_detail.json()["latest_classification"]["selected_source"]["source_row_id"] == 2001
        assert second_detail.json()["latest_classification"]["selected_source"]["source_row_id"] == 2002
        assert first_detail.json()["latest_classification"]["selected_source"]["source_kind"] == "mm"
        assert second_detail.json()["latest_classification"]["selected_source"]["source_kind"] == "mm"
    finally:
        app.dependency_overrides.clear()

def test_batch_item_classification_uses_source_row_id_for_duplicate_product_codes(tmp_path):
    service = _make_duplicate_ui_service(tmp_path)
    _upload_fake_pdf(service, "mm:2001", "batch-duplicate-2001.pdf")
    _upload_fake_pdf(service, "mm:2002", "batch-duplicate-2002.pdf")
    client = _client_with_service(service)
    try:
        response = client.post(
            "/api/metal-composition/items/classify-batch",
            json={"item_ids": ["mm:2001", "mm:2002"]},
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 202
        body = response.json()
        assert body["total_count"] == 2
        job_status = _drain_job(client, service, body["job_id"])
        assert job_status["status"] == "completed"
        detail_one = client.get("/api/metal-composition/items/mm:2001", headers={"X-API-Key": "test-api-key"})
        detail_two = client.get("/api/metal-composition/items/mm:2002", headers={"X-API-Key": "test-api-key"})
        assert detail_one.json()["latest_classification"]["selected_source"]["source_row_id"] == 2001
        assert detail_two.json()["latest_classification"]["selected_source"]["source_row_id"] == 2002
    finally:
        app.dependency_overrides.clear()
