from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest

from app.services.metal_composition.config import MetalCompositionSettings
from app.services.metal_composition.section_232_rules_engine import (
    build_section_232_ruleset_assessment,
    evaluate_section_232_ruleset,
)
from app.services.metal_composition.section_232_rulesets import (
    InMemorySection232RulesetStore,
    PersistedSection232RulesetStore,
    Section232DraftRuleCandidate,
)


def _candidate(
    candidate_id: str,
    *,
    batch_id: str = "batch-1",
    hts_code: str,
    rule_type: str,
    coverage_effect: str,
    effective_from: str = "2026-04-07",
    effective_to: str | None = None,
    metal_scope: str = "steel",
) -> Section232DraftRuleCandidate:
    return Section232DraftRuleCandidate(
        candidate_id=candidate_id,
        batch_id=batch_id,
        hts_code=hts_code,
        rule_type=rule_type,
        coverage_effect=coverage_effect,
        effective_from=effective_from,
        effective_to=effective_to,
        metal_scope=metal_scope,
        source_document_ids=["source-1"],
        source_pages=[29],
        source_excerpt="Synthetic rule for tests.",
        interpreter_confidence=0.91,
        catalog_match_found=True,
        review_decision="pending",
    )


class _SQLiteHANAConnection:
    def __init__(self) -> None:
        self._connection = sqlite3.connect(":memory:")
        self._transaction_depth = 0
        self._rollback_only = False

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
        with self.cursor() as cursor:
            cursor.execute(self._normalize_sql(sql), list(params or []))

    def executemany(self, sql: str, rows) -> None:
        with self.cursor() as cursor:
            cursor.executemany(self._normalize_sql(sql), list(rows))

    def table_exists(self, table: str, *, schema: str | None = None) -> bool:
        if schema:
            return False
        with self.cursor() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
                [table],
            )
            return cursor.fetchone() is not None

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        return str(sql).replace("CREATE COLUMN TABLE", "CREATE TABLE")


def _persisted_settings() -> MetalCompositionSettings:
    return MetalCompositionSettings(
        workbook_path=Path("/tmp/unused.xlsb"),
        api_env_path=Path("/tmp/api.env"),
        section_232_hana_schema="",
        section_232_draft_batches_table="TEST_SECTION232_DRAFT_BATCHES",
        section_232_draft_rules_table="TEST_SECTION232_DRAFT_RULES",
        section_232_rulesets_table="TEST_SECTION232_RULESETS",
        section_232_ruleset_rules_table="TEST_SECTION232_RULESET_RULES",
    )


@pytest.mark.parametrize(
    ("store_factory"),
    [
        pytest.param(InMemorySection232RulesetStore, id="in-memory"),
        pytest.param(
            lambda: PersistedSection232RulesetStore(
                _persisted_settings(),
                connection=_SQLiteHANAConnection(),
            ),
            id="persisted",
        ),
    ],
)
def test_review_candidates_updates_multiple_rows(store_factory):
    store = store_factory()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-include",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
            ),
            _candidate(
                "candidate-rate",
                batch_id=batch.batch_id,
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                coverage_effect="include",
            ),
            _candidate(
                "candidate-remove",
                batch_id=batch.batch_id,
                hts_code="2710.19.30.50",
                rule_type="remove",
                coverage_effect="remove",
            ),
        ],
    )

    updated_candidates = store.review_candidates(
        batch.batch_id,
        ["candidate-include", "candidate-remove"],
        decision="accepted",
    )

    assert [candidate.candidate_id for candidate in updated_candidates] == [
        "candidate-include",
        "candidate-remove",
    ]
    assert all(candidate.review_decision == "accepted" for candidate in updated_candidates)

    refreshed_candidates = {
        candidate.candidate_id: candidate.review_decision
        for candidate in store.list_draft_candidates(batch_id=batch.batch_id)
    }
    assert refreshed_candidates == {
        "candidate-include": "accepted",
        "candidate-rate": "pending",
        "candidate-remove": "accepted",
    }


@pytest.mark.parametrize(
    ("store_factory"),
    [
        pytest.param(InMemorySection232RulesetStore, id="in-memory"),
        pytest.param(
            lambda: PersistedSection232RulesetStore(
                _persisted_settings(),
                connection=_SQLiteHANAConnection(),
            ),
            id="persisted",
        ),
    ],
)
def test_delete_pending_draft_batch_removes_only_selected_pending_batch(store_factory):
    store = store_factory()
    cancelled_batch = store.create_draft_batch(
        source_ids=["source-cancelled"],
        source_filenames=["cancelled-notice.pdf"],
    )
    store.replace_batch_candidates(
        cancelled_batch.batch_id,
        [
            _candidate(
                "cancelled-include",
                batch_id=cancelled_batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
            ),
            _candidate(
                "cancelled-rate",
                batch_id=cancelled_batch.batch_id,
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                coverage_effect="include",
            ),
        ],
    )
    remaining_batch = store.create_draft_batch(
        source_ids=["source-remaining"],
        source_filenames=["remaining-notice.pdf"],
    )
    store.replace_batch_candidates(
        remaining_batch.batch_id,
        [
            _candidate(
                "remaining-include",
                batch_id=remaining_batch.batch_id,
                hts_code="7604.29.10",
                rule_type="include",
                coverage_effect="include",
            )
        ],
    )
    published_batch = store.create_draft_batch(
        source_ids=["source-published"],
        source_filenames=["published-notice.pdf"],
    )
    store.replace_batch_candidates(
        published_batch.batch_id,
        [
            _candidate(
                "published-include",
                batch_id=published_batch.batch_id,
                hts_code="8481.80.30.90",
                rule_type="include",
                coverage_effect="include",
            )
        ],
    )
    store.review_candidate(published_batch.batch_id, "published-include", decision="accepted")
    store.publish_batch(published_batch.batch_id, published_by="pytest")

    result = store.delete_pending_draft_batch(cancelled_batch.batch_id)

    assert result.batch_id == cancelled_batch.batch_id
    assert result.source_ids == ["source-cancelled"]
    assert result.source_filenames == ["cancelled-notice.pdf"]
    assert result.deleted_rule_count == 2
    with pytest.raises(KeyError, match=f"draft batch {cancelled_batch.batch_id} not found"):
        store.get_draft_batch(cancelled_batch.batch_id)
    assert [batch.batch_id for batch in store.list_draft_batches(status="pending_review")] == [
        remaining_batch.batch_id
    ]
    assert [candidate.candidate_id for candidate in store.list_draft_candidates(batch_id=remaining_batch.batch_id)] == [
        "remaining-include"
    ]
    assert store.get_active_ruleset_version() == "section232-v0001"
    assert [rule.candidate_id for rule in store.list_active_rules()] == ["published-include"]


@pytest.mark.parametrize(
    ("store_factory"),
    [
        pytest.param(InMemorySection232RulesetStore, id="in-memory"),
        pytest.param(
            lambda: PersistedSection232RulesetStore(
                _persisted_settings(),
                connection=_SQLiteHANAConnection(),
            ),
            id="persisted",
        ),
    ],
)
def test_delete_pending_draft_batch_rejects_published_batches(store_factory):
    store = store_factory()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["published-notice.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "published-include",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
            )
        ],
    )
    store.review_candidate(batch.batch_id, "published-include", decision="accepted")
    store.publish_batch(batch.batch_id, published_by="pytest")

    with pytest.raises(ValueError, match=f"draft batch {batch.batch_id} is already published"):
        store.delete_pending_draft_batch(batch.batch_id)

    assert store.get_draft_batch(batch.batch_id).status == "published"
    assert [rule.candidate_id for rule in store.list_active_rules()] == ["published-include"]


@pytest.mark.parametrize(
    ("store_factory"),
    [
        pytest.param(InMemorySection232RulesetStore, id="in-memory"),
        pytest.param(
            lambda: PersistedSection232RulesetStore(
                _persisted_settings(),
                connection=_SQLiteHANAConnection(),
            ),
            id="persisted",
        ),
    ],
)
def test_get_draft_batch_stats_counts_suspect_and_flagged_rows_as_warnings(store_factory):
    store = store_factory()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-clean",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
            ),
            Section232DraftRuleCandidate(
                candidate_id="candidate-suspect",
                batch_id=batch.batch_id,
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_from="2026-04-07",
                effective_to=None,
                metal_scope="steel",
                source_document_ids=["source-1"],
                source_pages=[2],
                source_excerpt="Synthetic suspect row.",
                interpreter_confidence=0.9,
                catalog_match_found=True,
                review_decision="pending",
                candidate_quality="suspect",
                candidate_flags=["llm_result_missing"],
            ),
        ],
    )

    stats = store.get_draft_batch_stats(batch_id=batch.batch_id)

    assert stats.total == 2
    assert stats.pending_count == 2
    assert stats.warning_count == 1


def test_publish_draft_batch_projects_current_eligible_codes():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])

    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-include",
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
            ),
            _candidate(
                "candidate-reduced",
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_to="2027-12-31",
            ),
            _candidate(
                "candidate-remove",
                hts_code="2710.19.30.50",
                rule_type="remove",
                coverage_effect="remove",
            ),
        ],
    )

    store.review_candidate(batch.batch_id, "candidate-include", decision="accepted")
    store.review_candidate(batch.batch_id, "candidate-reduced", decision="accepted")
    store.review_candidate(batch.batch_id, "candidate-remove", decision="accepted")

    published = store.publish_batch(batch.batch_id, published_by="codex")
    eligible_codes = store.list_active_eligible_codes(on_date=date(2026, 4, 19))

    assert published.version == "section232-v0001"
    assert eligible_codes == ["7308.90.95", "8421.29.00"]
    assert "2710.19.30.50" not in eligible_codes


def test_published_versions_remain_retrievable_after_later_publish():
    store = InMemorySection232RulesetStore()

    first_batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["first.pdf"])
    store.replace_batch_candidates(
        first_batch.batch_id,
        [
            _candidate(
                "first-include",
                batch_id=first_batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
            ),
        ],
    )
    store.review_candidate(first_batch.batch_id, "first-include", decision="accepted")
    first_published = store.publish_batch(first_batch.batch_id, published_by="codex")

    second_batch = store.create_draft_batch(source_ids=["source-2"], source_filenames=["second.pdf"])
    store.replace_batch_candidates(
        second_batch.batch_id,
        [
            _candidate(
                "second-include",
                batch_id=second_batch.batch_id,
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                coverage_effect="include",
            ),
        ],
    )
    store.review_candidate(second_batch.batch_id, "second-include", decision="accepted")
    second_published = store.publish_batch(second_batch.batch_id, published_by="codex")

    first_snapshot = store.get_published_ruleset(first_published.version)
    second_snapshot = store.get_published_ruleset(second_published.version)

    assert first_snapshot is not None
    assert second_snapshot is not None
    assert first_snapshot.version == first_published.version
    assert second_snapshot.version == second_published.version
    assert [item.candidate_id for item in first_snapshot.accepted_rules] == ["first-include"]
    assert [item.candidate_id for item in second_snapshot.accepted_rules] == ["second-include"]
    assert store.get_active_ruleset_version() == second_published.version
    assert store.list_active_eligible_codes(on_date=date(2026, 4, 19)) == ["8421.29.00"]


def test_returned_published_ruleset_does_not_mutate_persisted_snapshot():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["first.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "first-include",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
            ),
        ],
    )
    store.review_candidate(batch.batch_id, "first-include", decision="accepted")

    published = store.publish_batch(batch.batch_id, published_by="codex")
    published.accepted_rules.clear()

    stored_snapshot = store.get_published_ruleset(published.version)

    assert stored_snapshot is not None
    assert [item.candidate_id for item in stored_snapshot.accepted_rules] == ["first-include"]

    published_again = store.get_published_ruleset(published.version)
    assert published_again is not None
    published_again.accepted_rules[0].source_pages.append(99)
    published_again.accepted_rules[0].source_document_ids.append("mutated-source")

    stored_snapshot = store.get_published_ruleset(published.version)

    assert stored_snapshot is not None
    assert stored_snapshot.accepted_rules[0].source_pages == [29]
    assert stored_snapshot.accepted_rules[0].source_document_ids == ["source-1"]


def test_publish_batch_uses_explicit_snapshot_without_rewriting_pending_draft_batch():
    store = InMemorySection232RulesetStore()

    published_batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["first.pdf"])
    store.replace_batch_candidates(
        published_batch.batch_id,
        [
            _candidate(
                "first-include",
                batch_id=published_batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
            ),
        ],
    )
    store.review_candidate(published_batch.batch_id, "first-include", decision="accepted")
    first_publish = store.publish_batch(published_batch.batch_id, published_by="codex")

    pending_batch = store.create_draft_batch(source_ids=["source-2"], source_filenames=["second.pdf"])
    store.replace_batch_candidates(
        pending_batch.batch_id,
        [
            _candidate(
                "second-include",
                batch_id=pending_batch.batch_id,
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                coverage_effect="include",
            ),
        ],
    )
    store.review_candidate(pending_batch.batch_id, "second-include", decision="accepted")

    original_pending_candidates = store.list_draft_candidates(batch_id=pending_batch.batch_id)
    merged_candidates = [
        *first_publish.accepted_rules,
        *original_pending_candidates,
    ]

    published = store.publish_batch(
        pending_batch.batch_id,
        published_by="codex",
        accepted_rules_snapshot=merged_candidates,
    )

    pending_candidates_after_publish = store.list_draft_candidates(batch_id=pending_batch.batch_id)

    assert [candidate.candidate_id for candidate in pending_candidates_after_publish] == ["second-include"]
    assert pending_candidates_after_publish[0].batch_id == pending_batch.batch_id
    assert [candidate.candidate_id for candidate in published.accepted_rules] == ["first-include", "second-include"]
    assert published.accepted_rules[0].batch_id == published_batch.batch_id
    assert published.accepted_rules[1].batch_id == pending_batch.batch_id


def test_persisted_ruleset_store_degrades_legacy_free_form_metal_scope_to_unspecified():
    store = PersistedSection232RulesetStore(
        _persisted_settings(),
        connection=_SQLiteHANAConnection(),
    )
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["legacy.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "legacy-scope",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
                metal_scope="steel",
            ),
        ],
    )
    store.review_candidate(batch.batch_id, "legacy-scope", decision="accepted")
    published = store.publish_batch(batch.batch_id, published_by="codex")

    store.connection.execute(
        f'UPDATE "{store.ruleset_rules_table}" SET "METAL_SCOPE" = ? WHERE "VERSION" = ? AND "CANDIDATE_ID" = ?',
        ["applies to all covered products in the notice", published.version, "legacy-scope"],
    )

    snapshot = store.get_published_ruleset(published.version)

    assert snapshot is not None
    assert snapshot.accepted_rules[0].metal_scope == "unspecified"


def test_persisted_ruleset_store_degrades_legacy_exclusionary_metal_scope_to_unspecified():
    store = PersistedSection232RulesetStore(
        _persisted_settings(),
        connection=_SQLiteHANAConnection(),
    )
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["legacy.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "legacy-exclusion",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
                metal_scope="steel",
            ),
        ],
    )
    store.review_candidate(batch.batch_id, "legacy-exclusion", decision="accepted")
    published = store.publish_batch(batch.batch_id, published_by="codex")

    store.connection.execute(
        f'UPDATE "{store.ruleset_rules_table}" SET "METAL_SCOPE" = ? WHERE "VERSION" = ? AND "CANDIDATE_ID" = ?',
        ["aluminum except steel fasteners", published.version, "legacy-exclusion"],
    )

    snapshot = store.get_published_ruleset(published.version)

    assert snapshot is not None
    assert snapshot.accepted_rules[0].metal_scope == "unspecified"


def test_store_rejects_include_and_remove_for_same_hts_code_in_one_batch():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])

    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-include",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
            ),
            _candidate(
                "candidate-remove",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="remove",
                coverage_effect="remove",
            ),
        ],
    )

    store.review_candidate(batch.batch_id, "candidate-include", decision="accepted")
    store.review_candidate(batch.batch_id, "candidate-remove", decision="accepted")

    with pytest.raises(ValueError, match="duplicate normalized HTS code"):
        store.publish_batch(batch.batch_id, published_by="codex")


def test_store_rejects_multiple_accepted_rows_for_same_hts_code_in_one_batch():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])

    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-steel-include",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
                metal_scope="steel",
            ),
            _candidate(
                "candidate-aluminum-remove",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="remove",
                coverage_effect="remove",
                metal_scope="aluminum",
            ),
        ],
    )

    store.review_candidate(batch.batch_id, "candidate-steel-include", decision="accepted")
    store.review_candidate(batch.batch_id, "candidate-aluminum-remove", decision="accepted")

    with pytest.raises(ValueError, match="duplicate normalized HTS code"):
        store.publish_batch(batch.batch_id, published_by="codex")


def test_evaluate_section_232_ruleset_prefers_explicit_remove_over_broader_include():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "include-parent",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
            ),
            _candidate(
                "remove-leaf",
                batch_id=batch.batch_id,
                hts_code="7308.90.9540",
                rule_type="remove",
                coverage_effect="remove",
            ),
        ],
    )

    store.review_candidate(batch.batch_id, "include-parent", decision="accepted")
    store.review_candidate(batch.batch_id, "remove-leaf", decision="accepted")
    store.publish_batch(batch.batch_id, published_by="pytest")

    outcome = evaluate_section_232_ruleset(
        selected_code="7308.90.9540",
        on_date=date(2026, 4, 19),
        ruleset_store=store,
        top_level_grams={
            "steel": 18.0,
            "aluminum": 0.0,
            "copper": 0.0,
        },
    )

    assert outcome["decision"] == "not_subject"
    assert outcome["matched_rule_type"] == "remove"
    assert outcome["matched_hts_code"] == "7308.90.9540"
    assert outcome["effective_window"] == {
        "evaluation_date": "2026-04-19",
        "effective_from": "2026-04-07",
        "effective_to": None,
        "is_active": True,
    }


def test_evaluate_section_232_ruleset_prefers_more_specific_rate_schedule_and_respects_dates():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "include-parent",
                batch_id=batch.batch_id,
                hts_code="8421.29",
                rule_type="include",
                coverage_effect="include",
            ),
            _candidate(
                "rate-current",
                batch_id=batch.batch_id,
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                coverage_effect="include",
                effective_to="2027-12-31",
            ),
            _candidate(
                "remove-future",
                batch_id=batch.batch_id,
                hts_code="8421.29.0000",
                rule_type="remove",
                coverage_effect="remove",
                effective_from="2028-01-01",
            ),
        ],
    )

    store.review_candidate(batch.batch_id, "include-parent", decision="accepted")
    store.review_candidate(batch.batch_id, "rate-current", decision="accepted")
    store.review_candidate(batch.batch_id, "remove-future", decision="accepted")
    store.publish_batch(batch.batch_id, published_by="pytest")

    current_outcome = evaluate_section_232_ruleset(
        selected_code="8421.29.0000",
        on_date=date(2026, 4, 19),
        ruleset_store=store,
        top_level_grams={
            "steel": 14.0,
            "aluminum": 0.0,
            "copper": 0.0,
        },
    )
    future_outcome = evaluate_section_232_ruleset(
        selected_code="8421.29.0000",
        on_date=date(2028, 1, 2),
        ruleset_store=store,
        top_level_grams={
            "steel": 14.0,
            "aluminum": 0.0,
            "copper": 0.0,
        },
    )

    assert current_outcome["decision"] == "subject"
    assert current_outcome["matched_rule_type"] == "rate_schedule"
    assert current_outcome["matched_hts_code"] == "8421.29.00"
    assert "tariff_rate" not in current_outcome
    assert current_outcome["effective_window"]["evaluation_date"] == "2026-04-19"

    assert future_outcome["decision"] == "not_subject"
    assert future_outcome["matched_rule_type"] == "remove"
    assert future_outcome["matched_hts_code"] == "8421.29.0000"
    assert "tariff_rate" not in future_outcome
    assert future_outcome["effective_window"]["evaluation_date"] == "2028-01-02"


def test_evaluate_section_232_ruleset_ignores_material_scope_for_family_matches():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "steel-only",
                batch_id=batch.batch_id,
                hts_code="7616.99.51",
                rule_type="include",
                coverage_effect="include",
            ),
        ],
    )
    store.review_candidate(batch.batch_id, "steel-only", decision="accepted")
    store.publish_batch(batch.batch_id, published_by="pytest")

    aluminum_only_outcome = evaluate_section_232_ruleset(
        selected_code="7616.99.5190",
        on_date=date(2026, 4, 19),
        ruleset_store=store,
        top_level_grams={
            "steel": 0.0,
            "aluminum": 12.0,
            "copper": 0.0,
        },
    )

    assert aluminum_only_outcome["decision"] == "subject"
    assert aluminum_only_outcome["matched_rule_type"] == "include"
    assert aluminum_only_outcome["matched_hts_code"] == "7616.99.51"
    assert aluminum_only_outcome["reason"] == "matched_rule"


@pytest.mark.parametrize(
    ("legal_result", "expected_fragment", "expected_confidence"),
    [
        (
            {
                "decision": "needs_review",
                "reason": "no_ruleset_store",
                "matched_rule_type": None,
                "matched_hts_code": None,
                "effective_window": {"evaluation_date": "2026-04-19", "effective_from": None, "effective_to": None, "is_active": False},
                "active_ruleset_version": None,
            },
            "No published Section 232 ruleset store is available",
            0.4,
        ),
        (
            {
                "decision": "needs_review",
                "reason": "no_active_ruleset",
                "matched_rule_type": None,
                "matched_hts_code": None,
                "effective_window": {"evaluation_date": "2026-04-19", "effective_from": None, "effective_to": None, "is_active": False},
                "active_ruleset_version": None,
            },
            "No published Section 232 ruleset is active",
            0.4,
        ),
        (
            {
                "decision": "needs_review",
                "reason": "no_selected_code",
                "matched_rule_type": None,
                "matched_hts_code": None,
                "effective_window": {"evaluation_date": "2026-04-19", "effective_from": None, "effective_to": None, "is_active": False},
                "active_ruleset_version": None,
            },
            "No stable HTS code was available",
            0.4,
        ),
        (
            {
                "decision": "needs_review",
                "reason": "no_matching_rule",
                "matched_rule_type": None,
                "matched_hts_code": None,
                "effective_window": {"evaluation_date": "2026-04-19", "effective_from": None, "effective_to": None, "is_active": False},
                "active_ruleset_version": "section232-v0001",
            },
            "does not contain a current match for HTS code",
            0.7,
        ),
    ],
)
def test_build_section_232_ruleset_assessment_uses_reason_specific_needs_review_messages(
    legal_result,
    expected_fragment,
    expected_confidence,
):
    assessment = build_section_232_ruleset_assessment(
        selected_code="" if legal_result["reason"] == "no_selected_code" else "7407.10.5000",
        candidate_codes=[] if legal_result["reason"] == "no_selected_code" else ["7407.10.5000"],
        legal_result=legal_result,
    )

    assert assessment["decision"] == "needs_review"
    assert assessment["confidence"] == expected_confidence
    assert assessment["needs_human_review"] is True
    assert expected_fragment in assessment["basis_summary"]
    assert "marks HTS code" not in assessment["basis_summary"]


def test_publish_draft_batch_fails_when_any_candidate_remains_pending():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-accepted",
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
            ),
            _candidate(
                "candidate-pending",
                hts_code="8421.29.00",
                rule_type="rate_schedule",
                coverage_effect="include",
            ),
        ],
    )

    store.review_candidate(batch.batch_id, "candidate-accepted", decision="accepted")

    with pytest.raises(ValueError, match="pending"):
        store.publish_batch(batch.batch_id, published_by="codex")


def test_publish_draft_batch_fails_when_nothing_was_accepted():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            _candidate(
                "candidate-rejected",
                hts_code="2710.19.30.50",
                rule_type="remove",
                coverage_effect="remove",
            ),
        ],
    )
    store.review_candidate(batch.batch_id, "candidate-rejected", decision="rejected")

    with pytest.raises(ValueError, match="accepted"):
        store.publish_batch(batch.batch_id, published_by="codex")


def test_store_rejects_invalid_status_review_decision_and_coverage_effect_values():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])

    object.__setattr__(store._batches[batch.batch_id], "status", "bogus")
    with pytest.raises(ValueError, match="status"):
        store.publish_batch(batch.batch_id, published_by="codex")

    batch = store.create_draft_batch(source_ids=["source-2"], source_filenames=["annex-2.pdf"])
    candidate = _candidate(
        "candidate-invalid",
        hts_code="7308.90.95",
        rule_type="include",
        coverage_effect="include",
    )
    object.__setattr__(candidate, "coverage_effect", "bogus")
    with pytest.raises(ValueError, match="coverage_effect"):
        store.replace_batch_candidates(batch.batch_id, [candidate])

    candidate = _candidate(
        "candidate-invalid-rule-type",
        hts_code="7308.90.95",
        rule_type="include",
        coverage_effect="include",
    )
    object.__setattr__(candidate, "rule_type", "bogus")
    with pytest.raises(ValueError, match="rule_type"):
        store.replace_batch_candidates(batch.batch_id, [candidate])

    candidate = _candidate(
        "candidate-invalid-decision",
        batch_id=batch.batch_id,
        hts_code="8421.29.00",
        rule_type="rate_schedule",
        coverage_effect="include",
    )
    store.replace_batch_candidates(batch.batch_id, [candidate])
    object.__setattr__(store._candidates[batch.batch_id][candidate.candidate_id], "review_decision", "bogus")
    with pytest.raises(ValueError, match="review_decision"):
        store.review_candidate(batch.batch_id, candidate.candidate_id, decision="accepted")


def test_contradictory_remove_include_combination_is_rejected():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    candidate = _candidate(
        "candidate-contradiction",
        batch_id=batch.batch_id,
        hts_code="7308.90.95",
        rule_type="remove",
        coverage_effect="include",
    )

    with pytest.raises(ValueError, match="rule combination"):
        store.replace_batch_candidates(batch.batch_id, [candidate])


def test_duplicate_candidate_ids_in_replace_batch_candidates_are_rejected():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    candidates = [
        _candidate(
            "candidate-dup",
            batch_id=batch.batch_id,
            hts_code="7308.90.95",
            rule_type="include",
            coverage_effect="include",
        ),
        _candidate(
            "candidate-dup",
            batch_id=batch.batch_id,
            hts_code="8421.29.00",
            rule_type="rate_schedule",
            coverage_effect="include",
        ),
    ]

    with pytest.raises(ValueError, match="duplicate candidate_id"):
        store.replace_batch_candidates(batch.batch_id, candidates)


def test_uncanonicalizable_hts_code_is_rejected():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    candidate = _candidate(
        "candidate-bad-hts",
        batch_id=batch.batch_id,
        hts_code="not-a-code",
        rule_type="include",
        coverage_effect="include",
    )

    with pytest.raises(ValueError, match="hts_code"):
        store.replace_batch_candidates(batch.batch_id, [candidate])


def test_post_publish_edits_are_rejected():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    candidate = _candidate(
        "candidate-accepted",
        batch_id=batch.batch_id,
        hts_code="7308.90.95",
        rule_type="include",
        coverage_effect="include",
    )
    store.replace_batch_candidates(batch.batch_id, [candidate])
    store.review_candidate(batch.batch_id, candidate.candidate_id, decision="accepted")
    store.publish_batch(batch.batch_id, published_by="codex")

    with pytest.raises(ValueError, match="published"):
        store.replace_batch_candidates(
            batch.batch_id,
            [
                _candidate(
                    "candidate-second",
                    batch_id=batch.batch_id,
                    hts_code="8421.29.00",
                    rule_type="rate_schedule",
                    coverage_effect="include",
                )
            ],
        )

    with pytest.raises(ValueError, match="published"):
        store.review_candidate(batch.batch_id, candidate.candidate_id, decision="rejected")

    with pytest.raises(ValueError, match="published"):
        store.publish_batch(batch.batch_id, published_by="codex")


def test_malformed_effective_from_is_rejected():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    candidate = _candidate(
        "candidate-malformed-from",
        batch_id=batch.batch_id,
        hts_code="7308.90.95",
        rule_type="include",
        coverage_effect="include",
        effective_from="not-a-date",
    )

    with pytest.raises(ValueError, match="effective_from"):
        store.replace_batch_candidates(batch.batch_id, [candidate])


def test_malformed_effective_to_is_rejected():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    candidate = _candidate(
        "candidate-malformed-to",
        batch_id=batch.batch_id,
        hts_code="7308.90.95",
        rule_type="include",
        coverage_effect="include",
        effective_to="not-a-date",
    )

    with pytest.raises(ValueError, match="effective_to"):
        store.replace_batch_candidates(batch.batch_id, [candidate])


def test_persisted_ruleset_store_round_trips_optional_dates_as_none():
    store = PersistedSection232RulesetStore(
        _persisted_settings(),
        connection=_SQLiteHANAConnection(),
    )
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    store.replace_batch_candidates(
        batch.batch_id,
        [
            Section232DraftRuleCandidate(
                candidate_id="candidate-open-ended",
                batch_id=batch.batch_id,
                hts_code="7308.90.95",
                rule_type="include",
                coverage_effect="include",
                effective_from=None,
                effective_to=None,
                metal_scope="steel",
                source_document_ids=["source-1"],
                source_pages=[29],
                source_excerpt="Synthetic rule for tests.",
                interpreter_confidence=0.91,
                catalog_match_found=True,
                review_decision="pending",
            )
        ],
    )

    stored_candidate = store.list_draft_candidates(batch_id=batch.batch_id)[0]

    assert stored_candidate.effective_from is None
    assert stored_candidate.effective_to is None


def test_inverted_effective_range_is_rejected():
    store = InMemorySection232RulesetStore()
    batch = store.create_draft_batch(source_ids=["source-1"], source_filenames=["annex.pdf"])
    candidate = _candidate(
        "candidate-inverted-range",
        batch_id=batch.batch_id,
        hts_code="7308.90.95",
        rule_type="include",
        coverage_effect="include",
        effective_from="2026-12-31",
        effective_to="2026-04-07",
    )

    with pytest.raises(ValueError, match="invalid effective range"):
        store.replace_batch_candidates(batch.batch_id, [candidate])
