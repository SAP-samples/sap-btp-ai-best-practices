from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace as dataclass_replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from app.models.metal_composition import (
    Section232DirectClassificationRequest,
    Section232TopLevelComposition,
)
from app.services.metal_composition.config import MetalCompositionSettings
from app.services.metal_composition.section_232_rule_interpreter import interpret_section_232_batch
from app.services.metal_composition.section_232_rulesets import (
    InMemorySection232RulesetStore,
    PersistedSection232RulesetStore,
    Section232DraftRuleCandidate,
)
from app.services.metal_composition.section_232_sources import (
    ExtractedSection232Source,
    InMemorySection232SourceStore,
    PersistedSection232Source,
    Section232SourceStore,
)
from app.services.metal_composition.service import MetalCompositionService
from app.services.metal_composition.serving_store import WorkbookStore
from app.services.metal_composition.ui_state import InMemoryMetalCompositionUIStateStore


class _SQLiteHANAConnection:
    def __init__(self, *, fail_on_sql_substrings: tuple[str, ...] = ()) -> None:
        self._connection = sqlite3.connect(":memory:")
        self._transaction_depth = 0
        self._rollback_only = False
        self._fail_on_sql_substrings = [self._normalize_sql_for_match(item) for item in fail_on_sql_substrings]

    @contextmanager
    def transaction(self):
        is_outermost = self._transaction_depth == 0
        if is_outermost:
            self._rollback_only = False
        self._transaction_depth += 1
        try:
            yield self
        except Exception:
            if self._transaction_depth > 0:
                self._rollback_only = True
            if is_outermost:
                self._connection.rollback()
                self._rollback_only = False
            raise
        else:
            if is_outermost:
                if self._rollback_only:
                    self._connection.rollback()
                else:
                    self._connection.commit()
                self._rollback_only = False
        finally:
            self._transaction_depth = max(0, self._transaction_depth - 1)

    @contextmanager
    def cursor(self):
        cursor = self._connection.cursor()
        try:
            yield cursor
            if self._transaction_depth == 0:
                self._connection.commit()
        except Exception:
            if self._transaction_depth > 0:
                self._rollback_only = True
            else:
                self._connection.rollback()
            raise
        finally:
            cursor.close()

    def execute(self, sql: str, params=None) -> None:
        normalized_sql = self._normalize_sql(sql)
        self._maybe_fail(normalized_sql)
        with self.cursor() as cursor:
            cursor.execute(normalized_sql, list(params or []))

    def executemany(self, sql: str, rows) -> None:
        normalized_sql = self._normalize_sql(sql)
        self._maybe_fail(normalized_sql)
        with self.cursor() as cursor:
            cursor.executemany(normalized_sql, list(rows))

    def table_exists(self, table: str, *, schema: str | None = None) -> bool:
        if schema:
            return False
        with self.cursor() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
                [table],
            )
            return cursor.fetchone() is not None

    def _ensure_index(self, table: str, *, schema: str | None, columns) -> None:  # noqa: ARG002
        return None

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        return str(sql).replace("CREATE COLUMN TABLE", "CREATE TABLE")

    @staticmethod
    def _normalize_sql_for_match(sql: str) -> str:
        return " ".join(str(sql).upper().split())

    def _maybe_fail(self, sql: str) -> None:
        normalized_sql = self._normalize_sql_for_match(sql)
        for index, token in enumerate(self._fail_on_sql_substrings):
            if token in normalized_sql:
                self._fail_on_sql_substrings.pop(index)
                raise sqlite3.OperationalError(f"simulated sqlite hana failure for {token}")


def _settings() -> MetalCompositionSettings:
    return MetalCompositionSettings(
        workbook_path=Path("/tmp/unused.xlsb"),
        api_env_path=Path("/tmp/api.env"),
        section_232_hana_schema="",
        section_232_draft_batches_table="TEST_SECTION232_DRAFT_BATCHES",
        section_232_draft_rules_table="TEST_SECTION232_DRAFT_RULES",
        section_232_rulesets_table="TEST_SECTION232_RULESETS",
        section_232_ruleset_rules_table="TEST_SECTION232_RULESET_RULES",
    )


def _source(source_id: str, filename: str) -> PersistedSection232Source:
    return PersistedSection232Source(
        source_id=source_id,
        filename=filename,
        size_bytes=100,
        page_count=1,
        extraction_status="completed",
        full_text="[Page 1]\nSynthetic notice text.",
        page_texts=[
            {
                "page_number": 1,
                "text": "Synthetic notice text.",
                "layout_aware_text": "Synthetic notice text.",
                "page_excerpt": "Synthetic notice text.",
                "char_count": 22,
                "hts_mentions": [],
            }
        ],
        hts_mentions=[],
        warnings=[],
        content_sha256=f"sha-{source_id}",
        uploaded_at="2026-04-19T10:00:00Z",
    )


def _candidate(
    candidate_id: str,
    *,
    batch_id: str,
    hts_code: str,
    rule_type: str,
    metal_scope: str,
    effective_from: str = "2026-04-07",
    effective_to: str | None = None,
) -> Section232DraftRuleCandidate:
    return Section232DraftRuleCandidate(
        candidate_id=candidate_id,
        batch_id=batch_id,
        hts_code=hts_code,
        rule_type=rule_type,
        coverage_effect="remove" if rule_type == "remove" else "include",
        effective_from=effective_from,
        effective_to=effective_to,
        metal_scope=metal_scope,
        source_document_ids=["source-1"],
        source_pages=[1],
        source_excerpt="Synthetic rule excerpt.",
        interpreter_confidence=0.95,
        catalog_match_found=True,
        review_decision="pending",
    )


class _ServiceTestWorkflowRunner:
    def run(self, **kwargs):
        return {
            "final_composition": {
                "is_metal_item": True,
                "total_weight_grams": kwargs.get("source_summary", {}).get("total_weight_gram"),
                "estimated_total_metal_grams": 80.0,
                "top_level_grams": {
                    "steel": 0.0,
                    "aluminum": 0.0,
                    "copper": 80.0,
                    "cast_iron": 0.0,
                },
                "steel_subtype_grams": {},
                "confidence": 1.0,
                "reasoning": "Synthetic test composition.",
            },
        }


def _section_232_service() -> MetalCompositionService:
    source_df = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "normalized_product_code": "demo-1",
            }
        ]
    )
    prepared_df = pd.DataFrame([{"source_row_id": 1}])
    return MetalCompositionService(
        serving_store=WorkbookStore(source_df=source_df, prepared_df=prepared_df),
        workflow_runner=_ServiceTestWorkflowRunner(),
        settings=_settings(),
        ui_state_store=InMemoryMetalCompositionUIStateStore(),
        section_232_source_store=InMemorySection232SourceStore(),
        section_232_ruleset_store=InMemorySection232RulesetStore(),
    )


def _persisted_section_232_service(connection: _SQLiteHANAConnection) -> MetalCompositionService:
    source_df = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "normalized_product_code": "demo-1",
            }
        ]
    )
    prepared_df = pd.DataFrame([{"source_row_id": 1}])
    return MetalCompositionService(
        serving_store=WorkbookStore(source_df=source_df, prepared_df=prepared_df),
        workflow_runner=_ServiceTestWorkflowRunner(),
        settings=_settings(),
        ui_state_store=InMemoryMetalCompositionUIStateStore(),
        section_232_source_store=Section232SourceStore(_settings(), connection=connection),
        section_232_ruleset_store=PersistedSection232RulesetStore(_settings(), connection=connection),
    )


def _catalog_row(*, code: str, description: str) -> dict[str, object]:
    return {
        "code": code,
        "raw_code": code.replace(".", ""),
        "digits": sum(1 for char in code if char.isdigit()),
        "chapter_number": int(code[:2]),
        "heading_code": code.split(".")[0],
        "family_6_code": ".".join(code.split(".")[:2]) if len(code.split(".")) >= 2 else "",
        "family_8_code": ".".join(code.split(".")[:3]) if len(code.split(".")) >= 3 else "",
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
        "searchable_text": f"{code} {description}",
        "sort_order": 0,
    }


def _set_catalog(service: MetalCompositionService, rows: list[dict[str, object]]) -> None:
    service._set_hts_catalog_resolver_for_admin(  # noqa: SLF001
        catalog_frame=pd.DataFrame(rows),
        code_map_frame=pd.DataFrame(),
    )


class _FailingPublishRulesetStore(InMemorySection232RulesetStore):
    def publish_batch(
        self,
        batch_id: str,
        *,
        published_by: str,
        accepted_rules_snapshot=None,
    ):
        raise RuntimeError("publish exploded")


def test_interpret_section_232_batch_candidate_ids_change_when_metal_scope_changes():
    source = _source("source-1", "notice.pdf")
    source.page_texts[0]["text"] = "HTS 7308.90.95 remains covered."
    source.page_texts[0]["layout_aware_text"] = "HTS 7308.90.95 remains covered."
    source.page_texts[0]["page_excerpt"] = "HTS 7308.90.95 remains covered."

    steel_llm = SimpleNamespace(
        invoke_native_chat_completion=lambda **_: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "results": [
                                    {
                                        "candidate_key": "source-1:7308.90.95",
                                        "is_real_hts_candidate": True,
                                        "rule_type": "include",
                                        "effective_from": "2026-04-07",
                                        "metal_scope": "steel",
                                        "source_pages": [1],
                                    }
                                ]
                            }
                        )
                    )
                )
            ]
        )
    )

    aluminum_llm = SimpleNamespace(
        invoke_native_chat_completion=lambda **_: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "results": [
                                    {
                                        "candidate_key": "source-1:7308.90.95",
                                        "is_real_hts_candidate": True,
                                        "rule_type": "include",
                                        "effective_from": "2026-04-07",
                                        "metal_scope": "aluminum",
                                        "source_pages": [1],
                                    }
                                ]
                            }
                        )
                    )
                )
            ]
        )
    )

    steel_candidates = interpret_section_232_batch(
        batch_id="batch-ids",
        extracted_sources=[source],
        llm=steel_llm,
    )
    aluminum_candidates = interpret_section_232_batch(
        batch_id="batch-ids",
        extracted_sources=[source],
        llm=aluminum_llm,
    )

    assert len(steel_candidates) == 1
    assert len(aluminum_candidates) == 1
    assert steel_candidates[0].candidate_id != aluminum_candidates[0].candidate_id


def test_section_232_review_rows_mark_midlevel_legal_codes_as_family_matches():
    service = _section_232_service()
    _set_catalog(
        service,
        [
            _catalog_row(
                code="7407.10.30.00",
                description="Copper bars rods and profiles of refined copper",
            )
        ],
    )

    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["copper-notice.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-7407",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
            )
        ],
    )

    row = service.list_section_232_draft_rules(batch_id=batch.batch_id).items[0]

    assert row.legal_hts_code == "7407.10.30"
    assert row.hts_code == "7407.10.30"
    assert row.catalog_match_type == "family"
    assert row.catalog_representative_code == "7407.10.30.00"
    assert row.catalog_family_match_count == 1
    assert row.catalog_warning is None


def test_section_232_review_rows_mark_exact_legal_codes_as_exact_matches():
    service = _section_232_service()
    _set_catalog(
        service,
        [
            _catalog_row(
                code="7407.10.30",
                description="Copper bars rods and profiles of refined copper",
            )
        ],
    )

    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["copper-notice.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-exact",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
            )
        ],
    )

    row = service.list_section_232_draft_rules(batch_id=batch.batch_id).items[0]

    assert row.legal_hts_code == "7407.10.30"
    assert row.catalog_match_type == "exact"
    assert row.catalog_representative_code == "7407.10.30"
    assert row.catalog_family_match_count == 1
    assert row.catalog_warning is None


def test_section_232_review_rows_mark_missing_legal_codes_as_missing_matches():
    service = _section_232_service()
    _set_catalog(
        service,
        [
            _catalog_row(
                code="7308.90.95.00",
                description="Steel structures and parts",
            )
        ],
    )

    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["copper-notice.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-missing",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
            )
        ],
    )

    row = service.list_section_232_draft_rules(batch_id=batch.batch_id).items[0]

    assert row.legal_hts_code == "7407.10.30"
    assert row.catalog_match_type == "missing"
    assert row.catalog_representative_code is None
    assert row.catalog_family_match_count == 0
    assert row.catalog_warning == "HTS code not found in managed catalog"


def test_section_232_process_draft_batch_builds_source_filename_map_once(monkeypatch):
    service = _section_232_service()
    source_filename_map_calls = 0
    original_map = service._build_section_232_source_filename_map

    def spy(batch_ids):
        nonlocal source_filename_map_calls
        source_filename_map_calls += 1
        return original_map(batch_ids)

    def fake_interpret_section_232_batch(*, batch_id, extracted_sources, llm):
        assert llm is not None
        assert len(extracted_sources) == 2
        return [
            Section232DraftRuleCandidate(
                candidate_id="candidate-1",
                batch_id=batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_from="2026-04-07",
                effective_to=None,
                metal_scope="copper",
                source_document_ids=[extracted_sources[0].source_id],
                source_pages=[1],
                source_excerpt="Synthetic rule excerpt 1.",
                interpreter_confidence=0.95,
                catalog_match_found=True,
                review_decision="pending",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-2",
                batch_id=batch_id,
                hts_code="7407.10.31",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_from="2026-04-07",
                effective_to=None,
                metal_scope="copper",
                source_document_ids=[extracted_sources[1].source_id],
                source_pages=[2],
                source_excerpt="Synthetic rule excerpt 2.",
                interpreter_confidence=0.95,
                catalog_match_found=True,
                review_decision="pending",
            ),
        ]

    monkeypatch.setattr(service, "_build_section_232_source_filename_map", spy)
    monkeypatch.setattr(service, "_get_workflow_llm", lambda: object())
    monkeypatch.setattr(
        "app.services.metal_composition.service.interpret_section_232_batch",
        fake_interpret_section_232_batch,
    )

    response = service.process_section_232_draft_batch(
        uploads=[
            ("copper-a.pdf", b"%PDF-1.4 first"),
            ("copper-b.pdf", b"%PDF-1.4 second"),
        ]
    )

    assert source_filename_map_calls == 1
    assert response.batch.rule_candidate_count == 2
    assert [item.source_filenames for item in response.items] == [["copper-a.pdf"], ["copper-b.pdf"]]


def test_section_232_process_draft_batch_cleans_up_stored_sources_when_interpretation_fails(monkeypatch):
    service = _section_232_service()

    monkeypatch.setattr(service, "_get_workflow_llm", lambda: object())

    def _raise_interpretation_error(**_kwargs):
        raise ValueError("Could not parse JSON object from model response")

    monkeypatch.setattr(
        "app.services.metal_composition.service.interpret_section_232_batch",
        _raise_interpretation_error,
    )

    with pytest.raises(ValueError, match="Could not parse JSON object from model response"):
        service.process_section_232_draft_batch(
            uploads=[("broken.pdf", b"%PDF-1.4 broken-source")]
        )

    assert service.section_232_source_store.list_sources() == []
    assert service.section_232_ruleset_store.list_draft_batches() == []


def test_section_232_process_draft_batch_rolls_back_created_batch_when_candidate_persistence_fails(monkeypatch):
    connection = _SQLiteHANAConnection()
    service = _persisted_section_232_service(connection)

    monkeypatch.setattr(service, "_get_workflow_llm", lambda: object())

    def _fake_interpret_section_232_batch(**_kwargs):
        return [
            Section232DraftRuleCandidate(
                candidate_id="candidate-open-ended",
                batch_id="placeholder-batch",
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
                effective_from=None,
                effective_to=None,
                metal_scope="steel",
                source_document_ids=["source-1"],
                source_pages=[1],
                source_excerpt="Synthetic rule excerpt.",
                interpreter_confidence=0.95,
                catalog_match_found=True,
                review_decision="pending",
            )
        ]

    def _raise_persistence_error(_batch_id, _candidates):
        raise ValueError("candidate persistence failed")

    monkeypatch.setattr(
        "app.services.metal_composition.service.interpret_section_232_batch",
        _fake_interpret_section_232_batch,
    )
    monkeypatch.setattr(
        service.section_232_ruleset_store,
        "replace_batch_candidates",
        _raise_persistence_error,
    )

    with pytest.raises(ValueError, match="candidate persistence failed"):
        service.process_section_232_draft_batch(
            uploads=[("broken.pdf", b"%PDF-1.4 broken-source")]
        )

    assert service.section_232_source_store.list_sources() == []
    assert service.section_232_ruleset_store.list_draft_batches() == []


def test_section_232_list_draft_rules_builds_source_filename_map_once(monkeypatch):
    service = _section_232_service()
    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["copper-notice.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-list-1",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
            ),
            _candidate(
                "candidate-list-2",
                batch_id=batch.batch_id,
                hts_code="7407.10.31",
                rule_type="rate_schedule",
                metal_scope="copper",
            ),
        ],
    )

    source_filename_map_calls = 0
    original_map = service._build_section_232_source_filename_map

    def spy(batch_ids):
        nonlocal source_filename_map_calls
        source_filename_map_calls += 1
        return original_map(batch_ids)

    monkeypatch.setattr(service, "_build_section_232_source_filename_map", spy)

    response = service.list_section_232_draft_rules(batch_id=batch.batch_id)

    assert source_filename_map_calls == 1
    assert response.total == 2
    assert [item.source_filenames for item in response.items] == [
        ["copper-notice.pdf"],
        ["copper-notice.pdf"],
    ]


def test_section_232_list_draft_rules_rejects_batches_after_publish():
    service = _section_232_service()

    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["published-batch.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-published",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        batch.batch_id,
        "candidate-published",
        decision="accepted",
    )
    service.publish_section_232_draft_batch(
        batch_id=batch.batch_id,
        published_by="pytest",
    )

    with pytest.raises(KeyError, match=f"draft batch {batch.batch_id} not found"):
        service.list_section_232_draft_rules(batch_id=batch.batch_id)


def test_section_232_publish_section_232_draft_batch_replaces_older_code_group_when_newer_batch_mentions_same_hts():
    service = _section_232_service()

    first_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["copper-current.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        first_batch.batch_id,
        [
            _candidate(
                "candidate-current",
                batch_id=first_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="steel and copper",
                effective_from="2026-01-01",
                effective_to="2026-06-30",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        first_batch.batch_id,
        "candidate-current",
        decision="accepted",
    )
    first_publish = service.publish_section_232_draft_batch(
        batch_id=first_batch.batch_id,
        published_by="pytest",
    )

    second_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-2"],
        source_filenames=["copper-future.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        second_batch.batch_id,
        [
            _candidate(
                "candidate-future",
                batch_id=second_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper and steel",
                effective_from="2026-07-01",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        second_batch.batch_id,
        "candidate-future",
        decision="accepted",
    )
    second_publish = service.publish_section_232_draft_batch(
        batch_id=second_batch.batch_id,
        published_by="pytest",
    )

    matching_active_rules = [
        rule
        for rule in service.section_232_ruleset_store.list_active_rules()
        if rule.hts_code == "7407.10.30"
    ]

    assert first_publish.accepted_rule_count == 1
    assert second_publish.accepted_rule_count == 1
    assert [(rule.effective_from, rule.effective_to) for rule in matching_active_rules] == [("2026-07-01", None)]
    assert [rule.metal_scope for rule in matching_active_rules] == ["copper and steel"]


def test_section_232_publish_section_232_draft_batch_auto_rejects_pending_candidates():
    service = _section_232_service()

    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["mixed-review.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-accepted",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
            ),
            _candidate(
                "candidate-pending",
                batch_id=batch.batch_id,
                hts_code="7407.10.50",
                rule_type="include",
                metal_scope="copper",
            ),
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        batch.batch_id,
        "candidate-accepted",
        decision="accepted",
    )

    published = service.publish_section_232_draft_batch(
        batch_id=batch.batch_id,
        published_by="pytest",
    )

    updated_candidates = {
        candidate.candidate_id: candidate.review_decision
        for candidate in service.section_232_ruleset_store.list_draft_candidates(batch_id=batch.batch_id)
    }

    assert published.accepted_rule_count == 1
    assert updated_candidates == {
        "candidate-accepted": "accepted",
        "candidate-pending": "rejected",
    }


def test_section_232_publish_section_232_draft_batch_rejects_duplicate_normalized_publication_slots():
    service = _section_232_service()

    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["copper-duplicate.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-first",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="steel and copper",
                effective_from="2026-01-01",
            ),
            _candidate(
                "candidate-second",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper + steel",
                effective_from="2026-01-01",
            ),
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        batch.batch_id,
        "candidate-first",
        decision="accepted",
    )
    service.section_232_ruleset_store.review_candidate(
        batch.batch_id,
        "candidate-second",
        decision="accepted",
    )

    with pytest.raises(
        ValueError,
        match=r"duplicate normalized publication slots for HTS codes 7407\.10\.30",
    ):
        service.publish_section_232_draft_batch(
            batch_id=batch.batch_id,
            published_by="pytest",
        )


def test_section_232_review_workspace_sorts_catalog_missing_rows_first_then_numeric_hts():
    service = _section_232_service()
    _set_catalog(
        service,
        [
            _catalog_row(
                code="7407.10.30.00",
                description="Copper bars rods and profiles of refined copper",
            )
        ],
    )

    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["mixed-order.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-high-match",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
            ),
            _candidate(
                "candidate-low-missing",
                batch_id=batch.batch_id,
                hts_code="2408.14",
                rule_type="include",
                metal_scope="unspecified",
            ),
            _candidate(
                "candidate-mid-missing",
                batch_id=batch.batch_id,
                hts_code="2650.01",
                rule_type="include",
                metal_scope="unspecified",
            ),
        ],
    )

    workspace = service.get_section_232_review_workspace(batch_id=batch.batch_id)

    assert [row.legal_hts_code for row in workspace.rows] == [
        "2408.14",
        "2650.01",
        "7407.10.30",
    ]
    assert [row.catalog_match_type for row in workspace.rows] == [
        "missing",
        "missing",
        "family",
    ]


def test_section_232_review_workspace_sort_handles_mixed_length_hts_codes():
    service = _section_232_service()
    _set_catalog(
        service,
        [
            _catalog_row(
                code="9401",
                description="Seats and parts",
            ),
            _catalog_row(
                code="9401.99",
                description="Parts of seats",
            ),
        ],
    )

    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["mixed-length.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-10-digit",
                batch_id=batch.batch_id,
                hts_code="9401.99.9081",
                rule_type="include",
                metal_scope="unspecified",
            ),
            _candidate(
                "candidate-4-digit",
                batch_id=batch.batch_id,
                hts_code="9401",
                rule_type="include",
                metal_scope="unspecified",
            ),
            _candidate(
                "candidate-6-digit",
                batch_id=batch.batch_id,
                hts_code="9401.99",
                rule_type="include",
                metal_scope="unspecified",
            ),
        ],
    )

    workspace = service.get_section_232_review_workspace(batch_id=batch.batch_id)

    assert [row.legal_hts_code for row in workspace.rows] == [
        "9401.99.9081",
        "9401",
        "9401.99",
    ]
    assert [row.catalog_match_type for row in workspace.rows] == [
        "missing",
        "exact",
        "exact",
    ]


def test_section_232_publish_section_232_draft_batch_replaces_same_window_composite_scope_updates():
    service = _section_232_service()

    first_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["copper-first.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        first_batch.batch_id,
        [
            _candidate(
                "candidate-first",
                batch_id=first_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="steel and copper",
                effective_from="2026-01-01",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        first_batch.batch_id,
        "candidate-first",
        decision="accepted",
    )
    first_publish = service.publish_section_232_draft_batch(
        batch_id=first_batch.batch_id,
        published_by="pytest",
    )

    second_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-2"],
        source_filenames=["copper-second.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        second_batch.batch_id,
        [
            _candidate(
                "candidate-second",
                batch_id=second_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper and steel",
                effective_from="2026-01-01",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        second_batch.batch_id,
        "candidate-second",
        decision="accepted",
    )
    second_publish = service.publish_section_232_draft_batch(
        batch_id=second_batch.batch_id,
        published_by="pytest",
    )

    matching_active_rules = [
        rule
        for rule in service.section_232_ruleset_store.list_active_rules()
        if rule.hts_code == "7407.10.30"
    ]

    assert first_publish.accepted_rule_count == 1
    assert second_publish.accepted_rule_count == 1
    assert len(matching_active_rules) == 1
    assert matching_active_rules[0].metal_scope == "copper and steel"
    assert matching_active_rules[0].effective_from == "2026-01-01"


def test_section_232_publish_section_232_draft_batch_replaces_overlapping_active_rules_for_same_material_rule():
    service = _section_232_service()

    first_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["copper-first.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        first_batch.batch_id,
        [
            _candidate(
                "candidate-first",
                batch_id=first_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="steel and copper",
                effective_from="2026-01-01",
                effective_to="2026-06-30",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        first_batch.batch_id,
        "candidate-first",
        decision="accepted",
    )
    service.publish_section_232_draft_batch(
        batch_id=first_batch.batch_id,
        published_by="pytest",
    )

    second_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-2"],
        source_filenames=["copper-second.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        second_batch.batch_id,
        [
            _candidate(
                "candidate-second",
                batch_id=second_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper + steel",
                effective_from="2026-06-01",
                effective_to=None,
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        second_batch.batch_id,
        "candidate-second",
        decision="accepted",
    )

    second_publish = service.publish_section_232_draft_batch(
        batch_id=second_batch.batch_id,
        published_by="pytest",
    )

    matching_active_rules = [
        rule
        for rule in service.section_232_ruleset_store.list_active_rules()
        if rule.hts_code == "7407.10.30"
    ]
    assert second_publish.accepted_rule_count == 1
    assert [(rule.effective_from, rule.effective_to) for rule in matching_active_rules] == [("2026-06-01", None)]
    assert [rule.metal_scope for rule in matching_active_rules] == ["copper + steel"]


def test_section_232_publish_section_232_draft_batch_rejects_overlapping_accepted_rules_inside_the_same_batch():
    service = _section_232_service()

    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["aluminum-steel.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-wide",
                batch_id=batch.batch_id,
                hts_code="8207.30.6062",
                rule_type="rate_schedule",
                metal_scope="steel and aluminum",
                effective_from="2024-04-01",
                effective_to="2027-12-31",
            ),
            _candidate(
                "candidate-narrow",
                batch_id=batch.batch_id,
                hts_code="8207.30.6062",
                rule_type="rate_schedule",
                metal_scope="aluminum + steel",
                effective_from="2026-04-07",
                effective_to="2027-12-31",
            ),
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        batch.batch_id,
        "candidate-wide",
        decision="accepted",
    )
    service.section_232_ruleset_store.review_candidate(
        batch.batch_id,
        "candidate-narrow",
        decision="accepted",
    )

    with pytest.raises(
        ValueError,
        match=r"duplicate normalized publication slots for HTS codes 8207\.30\.6062",
    ):
        service.publish_section_232_draft_batch(
            batch_id=batch.batch_id,
            published_by="pytest",
        )


def test_section_232_rejects_invalid_metal_scope_before_publish():
    service = _section_232_service()
    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["copper-invalid.pdf"],
    )

    with pytest.raises(ValueError, match="invalid metal_scope"):
        service.section_232_ruleset_store.replace_batch_candidates(
            batch.batch_id,
            [
                _candidate(
                    "candidate-invalid",
                    batch_id=batch.batch_id,
                    hts_code="7407.10.30",
                    rule_type="rate_schedule",
                    metal_scope="steeel",
                )
            ],
        )


def test_section_232_runtime_uses_published_legal_code_as_prefix_scope():
    service = _section_232_service()
    first_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["copper-notice.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        first_batch.batch_id,
        [
            _candidate(
                "candidate-prefix-first",
                batch_id=first_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
                effective_from="2026-01-01",
                effective_to="2026-02-28",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        first_batch.batch_id,
        "candidate-prefix-first",
        decision="accepted",
    )
    service.publish_section_232_draft_batch(
        batch_id=first_batch.batch_id,
        published_by="pytest",
    )

    second_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-2"],
        source_filenames=["copper-notice-v2.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        second_batch.batch_id,
        [
            _candidate(
                "candidate-prefix-second",
                batch_id=second_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
                effective_from="2026-03-01",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        second_batch.batch_id,
        "candidate-prefix-second",
        decision="accepted",
    )
    service.publish_section_232_draft_batch(
        batch_id=second_batch.batch_id,
        published_by="pytest",
    )

    response = service.classify_section_232(
        Section232DirectClassificationRequest(
            hts_code="7407.10.30.00",
            supporting_hts_candidates=[],
            total_weight_grams=100.0,
            top_level_grams=Section232TopLevelComposition(
                steel=0.0,
                aluminum=0.0,
                copper=80.0,
            ),
            metal_share_certainty="exact",
        )
    )

    assert response.section_232_assessment.decision == "subject"
    assert response.section_232_reasoner_output["matched_hts_code"] == "7407.10.30"
    assert response.section_232_reasoner_output["reason"] == "matched_rule"
    assert response.section_232_reasoner_output["matched_rule_scope"] == "family"
    assert any(
        evidence.get("matched_rule_code") == "7407.10.30"
        and "prefix-family scope" in str(evidence.get("summary") or "")
        for evidence in response.section_232_assessment.evidence
    )


def test_section_232_publish_section_232_draft_batch_keeps_newer_uploaded_group_when_older_batch_is_published_later():
    service = _section_232_service()

    older_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-older"],
        source_filenames=["older-annex.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        older_batch.batch_id,
        [
            _candidate(
                "candidate-older",
                batch_id=older_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="steel and copper",
                effective_from="2026-01-01",
                effective_to="2026-03-31",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        older_batch.batch_id,
        "candidate-older",
        decision="accepted",
    )

    newer_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-newer"],
        source_filenames=["newer-annex.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        newer_batch.batch_id,
        [
            _candidate(
                "candidate-newer",
                batch_id=newer_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
                effective_from="2026-04-01",
                effective_to=None,
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        newer_batch.batch_id,
        "candidate-newer",
        decision="accepted",
    )

    first_publish = service.publish_section_232_draft_batch(
        batch_id=newer_batch.batch_id,
        published_by="pytest",
    )
    second_publish = service.publish_section_232_draft_batch(
        batch_id=older_batch.batch_id,
        published_by="pytest",
    )

    active_rules = [
        rule
        for rule in service.section_232_ruleset_store.list_active_rules()
        if rule.hts_code == "7407.10.30"
    ]

    assert first_publish.published_version == "section232-v0001"
    assert second_publish.published_version == "section232-v0002"
    assert second_publish.accepted_rule_count == 1
    assert [
        (rule.batch_id, rule.metal_scope, rule.effective_from, rule.effective_to)
        for rule in active_rules
    ] == [
        (newer_batch.batch_id, "copper", "2026-04-01", None),
    ]


def test_section_232_active_rules_project_newest_rule_per_hts_code_across_scopes():
    service = _section_232_service()

    older_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-older"],
        source_filenames=["older-include.pdf"],
    )
    older_candidate = dataclass_replace(
        _candidate(
            "candidate-older-include",
            batch_id=older_batch.batch_id,
            hts_code="3403.99.00",
            rule_type="include",
            metal_scope="steel",
        ),
        processed_at="2026-04-23T12:11:22Z",
    )
    service.section_232_ruleset_store.replace_batch_candidates(older_batch.batch_id, [older_candidate])
    service.section_232_ruleset_store.review_candidate(
        older_batch.batch_id,
        "candidate-older-include",
        decision="accepted",
    )
    service.publish_section_232_draft_batch(batch_id=older_batch.batch_id, published_by="pytest")

    newer_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-newer"],
        source_filenames=["annex-remove.pdf"],
    )
    newer_candidate = dataclass_replace(
        _candidate(
            "candidate-newer-remove",
            batch_id=newer_batch.batch_id,
            hts_code="3403.99.00",
            rule_type="remove",
            metal_scope="unspecified",
        ),
        processed_at="2026-04-23T12:27:05Z",
    )
    service.section_232_ruleset_store.replace_batch_candidates(newer_batch.batch_id, [newer_candidate])
    service.section_232_ruleset_store.review_candidate(
        newer_batch.batch_id,
        "candidate-newer-remove",
        decision="accepted",
    )
    service.publish_section_232_draft_batch(batch_id=newer_batch.batch_id, published_by="pytest")

    active_rules = [
        rule
        for rule in service.section_232_ruleset_store.list_active_rules()
        if rule.hts_code == "3403.99.00"
    ]

    assert [(rule.candidate_id, rule.coverage_effect, rule.metal_scope) for rule in active_rules] == [
        ("candidate-newer-remove", "remove", "unspecified")
    ]


def test_section_232_active_rules_project_newer_include_over_older_remove():
    service = _section_232_service()

    older_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-older"],
        source_filenames=["older-remove.pdf"],
    )
    older_candidate = dataclass_replace(
        _candidate(
            "candidate-older-remove",
            batch_id=older_batch.batch_id,
            hts_code="3403.99.00",
            rule_type="remove",
            metal_scope="unspecified",
        ),
        processed_at="2026-04-23T12:11:22Z",
    )
    service.section_232_ruleset_store.replace_batch_candidates(older_batch.batch_id, [older_candidate])
    service.section_232_ruleset_store.review_candidate(
        older_batch.batch_id,
        "candidate-older-remove",
        decision="accepted",
    )
    service.publish_section_232_draft_batch(batch_id=older_batch.batch_id, published_by="pytest")

    newer_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-newer"],
        source_filenames=["newer-include.pdf"],
    )
    newer_candidate = dataclass_replace(
        _candidate(
            "candidate-newer-include",
            batch_id=newer_batch.batch_id,
            hts_code="3403.99.00",
            rule_type="include",
            metal_scope="steel",
        ),
        processed_at="2026-04-23T12:27:05Z",
    )
    service.section_232_ruleset_store.replace_batch_candidates(newer_batch.batch_id, [newer_candidate])
    service.section_232_ruleset_store.review_candidate(
        newer_batch.batch_id,
        "candidate-newer-include",
        decision="accepted",
    )
    service.publish_section_232_draft_batch(batch_id=newer_batch.batch_id, published_by="pytest")

    active_rules = [
        rule
        for rule in service.section_232_ruleset_store.list_active_rules()
        if rule.hts_code == "3403.99.00"
    ]

    assert [(rule.candidate_id, rule.coverage_effect, rule.metal_scope) for rule in active_rules] == [
        ("candidate-newer-include", "include", "steel")
    ]


def test_delete_section_232_draft_hts_code_removes_all_matching_candidates():
    service = _section_232_service()
    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["draft-delete.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-delete-1",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
            ),
            _candidate(
                "candidate-delete-2",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="include",
                metal_scope="steel",
            ),
            _candidate(
                "candidate-keep",
                batch_id=batch.batch_id,
                hts_code="7407.10.50",
                rule_type="include",
                metal_scope="copper",
            ),
        ],
    )

    deleted = service.delete_section_232_draft_hts_code(
        batch_id=batch.batch_id,
        hts_code="74071030",
    )
    remaining_candidates = service.section_232_ruleset_store.list_draft_candidates(batch_id=batch.batch_id)

    assert deleted.batch_id == batch.batch_id
    assert deleted.deleted_hts_code == "7407.10.30"
    assert deleted.deleted_count == 2
    assert [candidate.candidate_id for candidate in remaining_candidates] == ["candidate-keep"]


def test_delete_section_232_published_hts_code_creates_new_version_and_removes_runtime_match():
    service = _section_232_service()
    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["published-delete.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-delete-runtime",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
                effective_from="2026-01-01",
                effective_to=None,
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        batch.batch_id,
        "candidate-delete-runtime",
        decision="accepted",
    )
    service.publish_section_232_draft_batch(
        batch_id=batch.batch_id,
        published_by="pytest",
    )

    deleted = service.delete_section_232_published_hts_code(
        hts_code="7407.10.30",
        published_by="auditor",
    )
    classification = service.classify_section_232(
        Section232DirectClassificationRequest(
            hts_code="7407.10.30.00",
            supporting_hts_candidates=[],
            total_weight_grams=100.0,
            top_level_grams=Section232TopLevelComposition(
                steel=0.0,
                aluminum=0.0,
                copper=80.0,
            ),
            metal_share_certainty="exact",
        )
    )

    assert deleted.deleted_hts_code == "7407.10.30"
    assert deleted.removed_rule_count == 1
    assert deleted.published_version == "section232-v0002"
    assert service.section_232_ruleset_store.list_active_rules() == []
    assert service.section_232_ruleset_store.list_active_eligible_codes(on_date=date(2026, 4, 22)) == []
    assert service.section_232_ruleset_store.list_delete_overrides()[0].deleted_by == "auditor"
    assert classification.section_232_assessment.decision == "needs_review"


def test_section_232_publish_section_232_draft_batch_does_not_reintroduce_deleted_code_from_older_batch():
    service = _section_232_service()

    older_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-older"],
        source_filenames=["older-delete.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        older_batch.batch_id,
        [
            _candidate(
                "candidate-older-delete",
                batch_id=older_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
                effective_from="2026-01-01",
                effective_to="2026-03-31",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        older_batch.batch_id,
        "candidate-older-delete",
        decision="accepted",
    )

    newer_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-newer"],
        source_filenames=["newer-delete.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        newer_batch.batch_id,
        [
            _candidate(
                "candidate-newer-delete",
                batch_id=newer_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
                effective_from="2026-04-01",
                effective_to=None,
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        newer_batch.batch_id,
        "candidate-newer-delete",
        decision="accepted",
    )

    service.publish_section_232_draft_batch(
        batch_id=newer_batch.batch_id,
        published_by="pytest",
    )
    service.delete_section_232_published_hts_code(
        hts_code="7407.10.30",
        published_by="auditor",
    )

    republish = service.publish_section_232_draft_batch(
        batch_id=older_batch.batch_id,
        published_by="pytest",
    )

    assert republish.accepted_rule_count == 0
    assert republish.published_version == "section232-v0003"
    assert service.section_232_ruleset_store.list_active_rules() == []


def test_section_232_review_workspace_includes_source_timing_metadata():
    service = _section_232_service()
    source = service.section_232_source_store.save_source(
        filename="timing-metadata.pdf",
        size_bytes=123,
        extracted=ExtractedSection232Source(
            page_count=1,
            extraction_status="completed",
            full_text="[Page 1]\n7407.10.30",
            page_texts=[
                {
                    "page_number": 1,
                    "text": "7407.10.30",
                    "layout_aware_text": "7407.10.30",
                    "page_excerpt": "7407.10.30",
                    "char_count": 10,
                    "hts_mentions": ["7407.10.30"],
                    "hts_occurrences": [],
                }
            ],
            hts_mentions=["7407.10.30"],
            warnings=[],
        ),
    )
    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=[source.source_id],
        source_filenames=[source.filename],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            Section232DraftRuleCandidate(
                candidate_id="candidate-timing",
                batch_id=batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_from="2026-04-01",
                effective_to=None,
                metal_scope="copper",
                source_document_ids=[source.source_id],
                source_pages=[1],
                source_excerpt="Synthetic source timing metadata test.",
                interpreter_confidence=0.95,
                catalog_match_found=True,
                review_decision="pending",
            )
        ],
    )

    workspace = service.get_section_232_review_workspace(batch_id=batch.batch_id)
    row = workspace.rows[0]

    assert row.processed_at == batch.created_at
    assert row.source_uploaded_at == source.uploaded_at
    assert [document.model_dump(mode="json") for document in row.source_documents] == [
        {
            "source_id": source.source_id,
            "filename": "timing-metadata.pdf",
            "uploaded_at": source.uploaded_at,
        }
    ]


def test_section_232_runtime_uses_exact_rule_scope_wording_for_exact_matches():
    service = _section_232_service()
    first_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["copper-notice.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        first_batch.batch_id,
        [
            _candidate(
                "candidate-exact-scope",
                batch_id=first_batch.batch_id,
                hts_code="7407.10.30",
                rule_type="rate_schedule",
                metal_scope="copper",
                effective_from="2026-01-01",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        first_batch.batch_id,
        "candidate-exact-scope",
        decision="accepted",
    )
    service.publish_section_232_draft_batch(
        batch_id=first_batch.batch_id,
        published_by="pytest",
    )

    response = service.classify_section_232(
        Section232DirectClassificationRequest(
            hts_code="7407.10.30",
            supporting_hts_candidates=[],
            total_weight_grams=100.0,
            top_level_grams=Section232TopLevelComposition(
                steel=0.0,
                aluminum=0.0,
                copper=80.0,
            ),
            metal_share_certainty="exact",
        )
    )

    assert response.section_232_assessment.decision == "subject"
    assert response.section_232_reasoner_output["matched_hts_code"] == "7407.10.30"
    assert response.section_232_reasoner_output["matched_rule_scope"] == "exact"
    assert any(
        evidence.get("matched_rule_code") == "7407.10.30"
        and "exact-match scope" in str(evidence.get("summary") or "")
        for evidence in response.section_232_assessment.evidence
    )


def test_persisted_ruleset_store_rejects_duplicate_hts_codes_in_one_batch():
    store = PersistedSection232RulesetStore(
        _settings(),
        connection=_SQLiteHANAConnection(),
    )
    batch = store.create_draft_batch(
        source_ids=["source-1", "source-2"],
        source_filenames=["annex-a.pdf", "annex-b.pdf"],
    )

    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "steel-include",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                metal_scope="steel",
            ),
            _candidate(
                "aluminum-remove",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="remove",
                metal_scope="aluminum",
            ),
        ],
    )

    pending_batches = store.list_draft_batches(status="pending_review")
    assert [item.batch_id for item in pending_batches] == [batch.batch_id]
    assert store.count_pending_batches() == 1

    store.review_candidate(batch.batch_id, "steel-include", decision="accepted")
    store.review_candidate(batch.batch_id, "aluminum-remove", decision="accepted")
    with pytest.raises(ValueError, match="duplicate normalized HTS code"):
        store.publish_batch(batch.batch_id, published_by="pytest")

    assert store.get_active_ruleset_version() is None
    assert store.count_pending_batches() == 1


def test_persisted_ruleset_store_initialization_avoids_nullif_backfill_for_nclob_columns():
    store = PersistedSection232RulesetStore(
        _settings(),
        connection=_SQLiteHANAConnection(
            fail_on_sql_substrings=(
                'NULLIF("CANDIDATE_QUALITY"',
                'NULLIF("CANDIDATE_FLAGS_JSON"',
            )
        ),
    )

    assert store.list_draft_batches() == []
    assert store.list_active_rules() == []


def test_persisted_ruleset_store_clear_all_wipes_drafts_and_published_history():
    store = PersistedSection232RulesetStore(
        _settings(),
        connection=_SQLiteHANAConnection(),
    )
    pending_batch = store.create_draft_batch(
        source_ids=["source-pending"],
        source_filenames=["pending-annex.pdf"],
    )
    store.replace_batch_candidates(
        pending_batch.batch_id,
        [
            _candidate(
                "pending-include",
                batch_id=pending_batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                metal_scope="steel",
            )
        ],
    )

    published_batch = store.create_draft_batch(
        source_ids=["source-published"],
        source_filenames=["published-annex.pdf"],
    )
    store.replace_batch_candidates(
        published_batch.batch_id,
        [
            _candidate(
                "published-include",
                batch_id=published_batch.batch_id,
                hts_code="8481.80.30.90",
                rule_type="include",
                metal_scope="steel",
            )
        ],
    )
    store.review_candidate(published_batch.batch_id, "published-include", decision="accepted")
    store.publish_batch(published_batch.batch_id, published_by="pytest")

    reset_counts = store.clear_all()

    assert reset_counts == {
        "cleared_draft_batch_count": 2,
        "cleared_draft_rule_count": 2,
        "cleared_delete_override_count": 0,
        "cleared_published_ruleset_count": 1,
        "cleared_published_rule_count": 1,
    }
    assert store.list_draft_batches() == []
    assert store.count_pending_batches() == 0
    assert store.get_active_ruleset_version() is None
    assert store.get_last_published_at() is None
    assert store.list_active_rules() == []
    assert store.list_active_eligible_codes(on_date=date(2026, 4, 19)) == []
    assert store.get_published_ruleset("section232-v0001") is None


def test_persisted_ruleset_publish_rolls_back_when_ruleset_rule_insert_fails():
    settings = _settings()
    store = PersistedSection232RulesetStore(
        settings,
        connection=_SQLiteHANAConnection(
            fail_on_sql_substrings=(f'INSERT INTO "{settings.section_232_ruleset_rules_table}"',)
        ),
    )
    batch = store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["annex-a.pdf"],
    )
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "steel-include",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                metal_scope="steel",
            )
        ],
    )
    store.review_candidate(batch.batch_id, "steel-include", decision="accepted")

    with pytest.raises(sqlite3.OperationalError, match="simulated sqlite hana failure"):
        store.publish_batch(batch.batch_id, published_by="pytest")

    assert store.get_active_ruleset_version() is None
    assert store.count_pending_batches() == 1
    assert [item.batch_id for item in store.list_draft_batches(status="pending_review")] == [batch.batch_id]
    assert store.get_published_ruleset("section232-v0001") is None


def test_reset_section_232_data_rolls_back_shared_connection_failure():
    settings = _settings()
    connection = _SQLiteHANAConnection(
        fail_on_sql_substrings=(f'DELETE FROM "{settings.section_232_rulesets_table}"',)
    )
    service = _persisted_section_232_service(connection)
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

    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["pending-annex.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "pending-include",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                metal_scope="steel",
            )
        ],
    )

    with pytest.raises(sqlite3.OperationalError, match="simulated sqlite hana failure"):
        service.reset_section_232_data()

    assert len(service.section_232_source_store.list_sources()) == 1
    assert service.section_232_source_store.list_eligible_hts_codes() == ["8483.30.80"]
    assert service.section_232_ruleset_store.count_pending_batches() == 1
    assert [item.batch_id for item in service.section_232_ruleset_store.list_draft_batches()] == [batch.batch_id]


def test_cancel_section_232_draft_batch_preserves_sources_still_referenced_by_other_batches():
    service = _section_232_service()
    source = service.section_232_source_store.save_source(
        filename="shared-notice.pdf",
        size_bytes=123,
        extracted=ExtractedSection232Source(
            page_count=1,
            extraction_status="completed",
            full_text="[Page 1]\nShared Section 232 notice text.",
            page_texts=[
                {
                    "page_number": 1,
                    "text": "Shared Section 232 notice text.",
                    "layout_aware_text": "Shared Section 232 notice text.",
                    "page_excerpt": "Shared Section 232 notice text.",
                    "char_count": 31,
                    "hts_mentions": ["7308.90.95"],
                }
            ],
            hts_mentions=["7308.90.95"],
            warnings=[],
        ),
    )
    first_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=[source.source_id],
        source_filenames=[source.filename],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        first_batch.batch_id,
        [
            _candidate(
                "first-include",
                batch_id=first_batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                metal_scope="steel",
            )
        ],
    )
    second_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=[source.source_id],
        source_filenames=[source.filename],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        second_batch.batch_id,
        [
            _candidate(
                "second-include",
                batch_id=second_batch.batch_id,
                hts_code="7604.29.10",
                rule_type="include",
                metal_scope="aluminum",
            )
        ],
    )

    response = service.cancel_section_232_draft_batch(first_batch.batch_id)

    assert response.batch_id == first_batch.batch_id
    assert response.deleted_source_count == 0
    assert response.deleted_source_filenames == []
    assert [source.source_id for source in service.section_232_source_store.list_sources()] == [source.source_id]
    assert [batch.batch_id for batch in service.section_232_ruleset_store.list_draft_batches(status="pending_review")] == [
        second_batch.batch_id
    ]


def test_cancel_section_232_draft_batch_rolls_back_shared_connection_failure():
    connection = _SQLiteHANAConnection()
    service = _persisted_section_232_service(connection)
    source = service.section_232_source_store.save_source(
        filename="pending-notice.pdf",
        size_bytes=123,
        extracted=ExtractedSection232Source(
            page_count=1,
            extraction_status="completed",
            full_text="[Page 1]\nPending Section 232 notice text.",
            page_texts=[
                {
                    "page_number": 1,
                    "text": "Pending Section 232 notice text.",
                    "layout_aware_text": "Pending Section 232 notice text.",
                    "page_excerpt": "Pending Section 232 notice text.",
                    "char_count": 32,
                    "hts_mentions": ["7308.90.95"],
                }
            ],
            hts_mentions=["7308.90.95"],
            warnings=[],
        ),
    )
    batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=[source.source_id],
        source_filenames=[source.filename],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "pending-include",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                metal_scope="steel",
            )
        ],
    )

    def _fail_delete_sources(source_ids):  # noqa: ARG001
        raise sqlite3.OperationalError("simulated source delete failure")

    service.section_232_source_store.delete_sources = _fail_delete_sources

    with pytest.raises(sqlite3.OperationalError, match="simulated source delete failure"):
        service.cancel_section_232_draft_batch(batch.batch_id)

    assert [source.source_id for source in service.section_232_source_store.list_sources()] == [source.source_id]
    assert service.section_232_ruleset_store.count_pending_batches() == 1
    assert [item.batch_id for item in service.section_232_ruleset_store.list_draft_batches()] == [batch.batch_id]
    assert [
        candidate.candidate_id
        for candidate in service.section_232_ruleset_store.list_draft_candidates(batch_id=batch.batch_id)
    ] == ["pending-include"]


def test_section_232_publish_failure_leaves_pending_batch_review_rows_untouched():
    source_df = pd.DataFrame(
        [
            {
                "source_row_id": 1,
                "normalized_product_code": "demo-1",
            }
        ]
    )
    prepared_df = pd.DataFrame([{"source_row_id": 1}])
    store = _FailingPublishRulesetStore()
    service = MetalCompositionService(
        serving_store=WorkbookStore(source_df=source_df, prepared_df=prepared_df),
        workflow_runner=_ServiceTestWorkflowRunner(),
        settings=_settings(),
        ui_state_store=InMemoryMetalCompositionUIStateStore(),
        section_232_source_store=InMemorySection232SourceStore(),
        section_232_ruleset_store=store,
    )

    published_batch = store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["carry-forward.pdf"],
    )
    store.replace_batch_candidates(
        published_batch.batch_id,
        [
            _candidate(
                "carry-forward",
                batch_id=published_batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                metal_scope="steel",
            )
        ],
    )
    store.review_candidate(published_batch.batch_id, "carry-forward", decision="accepted")
    store._active_rules = store.list_draft_candidates(batch_id=published_batch.batch_id)  # noqa: SLF001

    pending_batch = store.create_draft_batch(
        source_ids=["source-2"],
        source_filenames=["pending.pdf"],
    )
    store.replace_batch_candidates(
        pending_batch.batch_id,
        [
            _candidate(
                "pending-update",
                batch_id=pending_batch.batch_id,
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                metal_scope="copper",
            )
        ],
    )
    store.review_candidate(pending_batch.batch_id, "pending-update", decision="accepted")

    original_pending_candidates = store.list_draft_candidates(batch_id=pending_batch.batch_id)

    with pytest.raises(RuntimeError, match="publish exploded"):
        service.publish_section_232_draft_batch(
            batch_id=pending_batch.batch_id,
            published_by="pytest",
        )

    pending_candidates_after_failure = store.list_draft_candidates(batch_id=pending_batch.batch_id)

    assert [candidate.candidate_id for candidate in pending_candidates_after_failure] == ["pending-update"]
    assert pending_candidates_after_failure == original_pending_candidates
    assert pending_candidates_after_failure[0].batch_id == pending_batch.batch_id


def test_section_232_later_published_review_preserves_source_filenames_for_carried_forward_rules():
    service = _section_232_service()

    first_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-1"],
        source_filenames=["original-notice.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        first_batch.batch_id,
        [
            _candidate(
                "carry-forward",
                batch_id=first_batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                metal_scope="steel",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        first_batch.batch_id,
        "carry-forward",
        decision="accepted",
    )
    service.publish_section_232_draft_batch(
        batch_id=first_batch.batch_id,
        published_by="pytest",
    )

    second_batch = service.section_232_ruleset_store.create_draft_batch(
        source_ids=["source-2"],
        source_filenames=["updated-notice.pdf"],
    )
    service.section_232_ruleset_store.replace_batch_candidates(
        second_batch.batch_id,
        [
            _candidate(
                "new-rule",
                batch_id=second_batch.batch_id,
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                metal_scope="copper",
            )
        ],
    )
    service.section_232_ruleset_store.review_candidate(
        second_batch.batch_id,
        "new-rule",
        decision="accepted",
    )
    second_publish = service.publish_section_232_draft_batch(
        batch_id=second_batch.batch_id,
        published_by="pytest",
    )

    workspace = service.get_section_232_review_workspace(version=second_publish.published_version)
    rows_by_candidate_id = {row.candidate_id: row for row in workspace.rows}

    assert rows_by_candidate_id["carry-forward"].source_filenames == ["original-notice.pdf"]
