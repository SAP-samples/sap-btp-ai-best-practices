"""Verify bounded analytical queries and source-grounded solution explanations."""

import pytest
from app.workspace.models import QuerySpec
from app.workspace.analytics import execute_query, compare_solutions, explain_assignment


class MemoryRepository:
    """Provide deterministic immutable rows and run metadata for semantic tests."""

    def __init__(self, tables, records=None):
        """Store supplied table fixtures and optional run dictionaries."""
        self.tables, self.records = tables, records or {}
        self.reads = []

    def rows(self, owner_id, view, filters=None):
        """Return point-filtered fixtures and record predicates sent to storage."""
        self.reads.append((owner_id, view, filters or {}))
        return [
            dict(row)
            for row in self.tables.get((owner_id, view), [])
            if all(
                float(row.get(key, -1)) == value
                for key, value in (filters or {}).items()
            )
        ]

    def get(self, kind, identifier):
        """Return the requested fixture metadata record."""
        return self.records.get(identifier, {"metadata": {}, "dataset_id": "d1"})


def test_aggregate_uses_whole_scope_before_pagination():
    """A page-size-one request still aggregates every filtered source row."""
    repo = MemoryRepository(
        {
            ("r", "groups"): [
                {
                    "point_index": 1,
                    "plant": "A",
                    "coverage_days": i,
                    "group_demand_litres": i,
                }
                for i in range(1, 11)
            ]
        }
    )
    result = execute_query(
        repo,
        QuerySpec(
            view="groups",
            run_id="r",
            point_index=1,
            metrics=[
                {"field": "coverage_days", "operation": "mean"},
                {"field": "coverage_days", "operation": "p90"},
                {
                    "field": "coverage_days",
                    "operation": "weighted_mean",
                    "weight_field": "group_demand_litres",
                },
            ],
            limit=1,
        ),
    )
    assert result["rows"][0] == {
        "mean_coverage_days": 5.5,
        "p90_coverage_days": 9.0,
        "weighted_mean_coverage_days": 7.0,
    }
    assert result["scope"]["filtered_rows"] == 10


def test_empty_aggregate_and_invalid_fields():
    """Empty sum is null; count is zero; unknown fields are rejected."""
    repo = MemoryRepository({})
    result = execute_query(
        repo,
        QuerySpec(
            view="groups",
            run_id="r",
            metrics=[
                {"field": "coverage_days", "operation": "sum"},
                {"operation": "count"},
            ],
        ),
    )
    assert result["rows"] == [{"sum_coverage_days": None, "count": 0}]
    with pytest.raises(ValueError, match="field"):
        execute_query(
            repo, QuerySpec(view="groups", run_id="r", fields=["sql_expression"])
        )


def test_numeric_filter_and_stable_null_last_sort():
    """Historical numeric strings sort numerically and nulls remain last descending."""
    repo = MemoryRepository(
        {
            ("r", "groups"): [
                {"coverage_days": "2", "plant": "B"},
                {"coverage_days": "10", "plant": "A"},
                {"coverage_days": "", "plant": "C"},
            ]
        }
    )
    query = QuerySpec(
        view="groups",
        run_id="r",
        fields=["coverage_days", "plant"],
        sort=[{"field": "coverage_days", "direction": "desc"}],
    )
    assert [r["plant"] for r in execute_query(repo, query)["rows"]] == ["A", "B", "C"]
    query = QuerySpec.model_validate(
        {
            **query.model_dump(),
            "filters": [{"field": "coverage_days", "op": "gt", "value": 3}],
        }
    )
    assert execute_query(repo, query)["total"] == 1


def test_compare_uses_membership_not_local_labels():
    """Renumbered identical groups do not count as changed membership across datasets."""
    tables = {}
    for run, group in (("left", "G001"), ("right", "G009")):
        tables[(run, "solutions")] = [
            {"point_index": 1, "j_ch": 2.0, "proof_scope": "restricted-library"}
        ]
        tables[(run, "members")] = [
            {
                "point_index": 1,
                "plant": "P1",
                "material": m,
                "proposed_subgroup": group,
                "proposed_members": "F1|F2",
                "selected_pv": "PV1",
            }
            for m in ("F1", "F2")
        ]
    tables[("right", "members")].append(
        {"point_index": 1, "plant": "P1", "material": "F3"}
    )
    repo = MemoryRepository(
        tables, {"left": {"dataset_id": "d1"}, "right": {"dataset_id": "d2"}}
    )
    result = compare_solutions(
        repo,
        {"run_id": "left", "point_index": 1},
        {"run_id": "right", "point_index": 1},
    )
    assert result["population"]["intersection_count"] == 2
    assert result["membership_changes"] == []
    assert result["population"]["right_only"] == [{"plant": "P1", "material": "F3"}]
    assert result["caveats"]


def test_explain_recomputes_effective_batch_formulas_and_eligibility():
    """Explanation uses effective batch and original demand/calendar operands."""
    repo = MemoryRepository(
        {
            ("r", "solutions"): [{"point_index": 1, "proof_scope": "runtime-limited"}],
            ("r", "groups"): [
                {
                    "point_index": 1,
                    "group_id": "g",
                    "plant": "P",
                    "sefi": "S",
                    "members": "F1|F2",
                    "group_size": 2,
                    "effective_batch_litres": 100.0,
                    "nominal_lot_litres": 80.0,
                    "group_demand_litres": 1000.0,
                    "common_lines": "1|2",
                    "selected_pv": "PV1",
                }
            ],
            ("r", "members"): [
                {"point_index": 1, "material": "F1", "plant": "P", "group_id": "g"}
            ],
        },
        {"r": {"metadata": {"config": {"demand_days": 250, "productive_weeks": 50}}}},
    )
    result = explain_assignment(repo, "r", 1, material="F1")
    assert result["derivations"]["base_group_coverage_days"]["calculated"] == 25.0
    assert result["derivations"]["frequency_per_week"]["calculated"] == 0.2
    assert result["derivations"]["j_ch_contribution"]["calculated"] == 0.2
    assert result["line_eligibility"]["scheduled_assignment"] is False


def test_weighted_mean_requires_explicit_valid_weights():
    """Missing weight fields and negative weights cannot silently produce metrics."""
    repo = MemoryRepository(
        {("r", "groups"): [{"coverage_days": 3, "group_demand_litres": -1}]}
    )
    with pytest.raises(ValueError, match="weight_field"):
        execute_query(
            repo,
            QuerySpec(
                view="groups",
                run_id="r",
                metrics=[{"field": "coverage_days", "operation": "weighted_mean"}],
            ),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        execute_query(
            repo,
            QuerySpec(
                view="groups",
                run_id="r",
                metrics=[
                    {
                        "field": "coverage_days",
                        "operation": "weighted_mean",
                        "weight_field": "group_demand_litres",
                    }
                ],
            ),
        )
    with pytest.raises(ValueError, match="numeric"):
        execute_query(
            repo,
            QuerySpec(
                view="groups",
                run_id="r",
                filters=[{"field": "coverage_days", "value": "not a number"}],
            ),
        )


def test_grouped_pagination_total_counts_aggregate_groups():
    """Totals report all grouped rows even when the response contains one row."""
    repo = MemoryRepository(
        {
            ("r", "groups"): [
                {"plant": "A", "coverage_days": 2},
                {"plant": "A", "coverage_days": 4},
                {"plant": "B", "coverage_days": 10},
            ]
        }
    )
    result = execute_query(
        repo,
        QuerySpec(
            view="groups",
            run_id="r",
            group_by=["plant"],
            metrics=[
                {"operation": "sum", "field": "coverage_days", "alias": "total_days"}
            ],
            sort=[{"field": "total_days", "direction": "desc"}],
            limit=1,
        ),
    )
    assert result["rows"] == [{"plant": "B", "total_days": 10}]
    assert result["total"] == 2 and result["truncated"] is True


def test_explain_out_of_scope_preserves_source_admission_evidence():
    """Run-specific exclusion is explained without rewriting original source status."""
    member = {
        "point_index": 1,
        "material": "F1",
        "plant": "P",
        "group_id": "",
        "model_status": "modeled",
        "exclusion_reason": "",
        "run_model_status": "out_of_scope",
        "run_exclusion_reason": "outside_run_scope_or_disposition",
    }
    repo = MemoryRepository(
        {("r", "solutions"): [{"point_index": 1}], ("r", "members"): [member]}
    )
    result = explain_assignment(repo, "r", 1, material="F1")
    assert result["exclusion_reason"] == "outside_run_scope_or_disposition"
    assert result["evidence"]["model_status"] == "out_of_scope"
    assert result["evidence"]["source_model_status"] == "modeled"
    assert result["evidence"]["source_exclusion_reason"] == ""
    assert result["members"][0] == member


def test_query_and_compare_push_point_scope_to_repository():
    """Queries and comparisons must not fetch unrelated point membership records."""
    repo = MemoryRepository(
        {
            ("r", "solutions"): [{"point_index": 1}, {"point_index": 2}],
            ("r", "members"): [
                {"point_index": 1, "plant": "P", "material": "A"},
                {"point_index": 2, "plant": "P", "material": "B"},
            ],
        }
    )
    result = execute_query(
        repo, QuerySpec(view="members", run_id="r", point_index=1, fields=["material"])
    )
    assert result["rows"] == [{"material": "A"}]
    assert result["scope"]["source_rows"] == 1
    assert result["scope"]["storage_filters"] == {"point_index": 1}
    compare_solutions(
        repo, {"run_id": "r", "point_index": 1}, {"run_id": "r", "point_index": 2}
    )
    assert all(filters.get("point_index") in (1, 2) for _, _, filters in repo.reads)
    assert ("r", "members", {"point_index": 2}) in repo.reads


def test_comparison_distinguishes_modeled_population_from_preserved_source_rows():
    """Identical source keys do not imply identical optimization populations."""
    tables = {}
    for run in ("full", "scoped"):
        tables[(run, "solutions")] = [
            {
                "point_index": 1,
                "result_class": "greenfield-runtime-limited",
                "candidate_pool_completeness": "restricted",
                "primary_relative_gap": 0.12,
                "relative_gap": 0.04,
                "proof_scope": "runtime-limited",
                "validation_status": "valid",
            }
        ]
        tables[(run, "members")] = []
        for material, demand, coverage in [("A", 100.0, 10.0), ("B", 300.0, 20.0)]:
            assigned = run == "full" or material == "A"
            tables[(run, "members")].append(
                {
                    "point_index": 1,
                    "plant": "P",
                    "sefi": "S",
                    "material": material,
                    "model_status": "modeled",
                    "run_model_status": "modeled" if assigned else "out_of_scope",
                    "proposed_subgroup": "G1" if assigned else "",
                    "proposed_members": "A|B"
                    if run == "full"
                    else "A"
                    if assigned
                    else "",
                    "fini_adjusted_coverage_days": coverage if assigned else "",
                    "forecast_litres_12m": demand,
                }
            )
    repo = MemoryRepository(
        tables,
        {
            "full": {
                "dataset_id": "d",
                "metadata": {"business_acceptance_assessed": False},
            },
            "scoped": {
                "dataset_id": "d",
                "metadata": {"business_acceptance_assessed": False},
            },
        },
    )
    result = compare_solutions(
        repo,
        {"run_id": "full", "point_index": 1},
        {"run_id": "scoped", "point_index": 1},
    )
    assert result["population"]["intersection_count"] == 2
    assert result["modeled_population"]["left_count"] == 2
    assert result["modeled_population"]["right_count"] == 1
    assert result["modeled_population"]["intersection_count"] == 1
    assert result["assigned_population"]["left_only"] == [
        {"plant": "P", "material": "B"}
    ]
    assert len(result["membership_changes"]) == 2
    assert len(result["modeled_membership_changes"]) == 1
    assert (
        result["change_scopes"]["membership_changes"]
        == "source_population_intersection"
    )
    assert result["common_member_coverage"]["population_count"] == 1
    assert result["common_member_coverage"]["left"]["demand_weighted_mean_days"] == 10.0
    assert result["proof"]["left"]["result_class"] == "greenfield-runtime-limited"
    assert result["proof"]["left"]["candidate_pool_completeness"] == "restricted"
    assert result["proof"]["left"]["primary_relative_gap"] == 0.12
    assert result["proof"]["left"]["business_acceptance_assessed"] is False
    assert any(
        "modeled populations differ" in caveat.lower() for caveat in result["caveats"]
    )


def test_common_member_coverage_counts_each_fini_once_and_requires_evidence():
    """Coverage aggregation uses material demand, never duplicated group demand."""
    tables = {}
    for run in ("left", "right"):
        tables[(run, "solutions")] = [{"point_index": 1}]
        tables[(run, "members")] = [
            {
                "point_index": 1,
                "plant": "P",
                "material": material,
                "proposed_subgroup": "G1",
                "proposed_members": "A|B",
                "model_status": "modeled",
                "forecast_litres_12m": demand,
                "group_demand_litres": 400.0,
                "fini_adjusted_coverage_days": coverage,
            }
            for material, demand, coverage in [("A", 100.0, 10.0), ("B", 300.0, 20.0)]
        ]
    repo = MemoryRepository(tables)
    result = compare_solutions(
        repo,
        {"run_id": "left", "point_index": 1},
        {"run_id": "right", "point_index": 1},
    )
    assert result["common_member_coverage"]["left"]["demand_weighted_mean_days"] == 17.5
    assert result["proof"]["left"]["business_acceptance_assessed"] is None
    tables[("right", "members")][1]["fini_adjusted_coverage_days"] = ""
    result = compare_solutions(
        repo,
        {"run_id": "left", "point_index": 1},
        {"run_id": "right", "point_index": 1},
    )
    assert result["common_member_coverage"]["coverage_evidence_complete"] is False
    assert (
        result["common_member_coverage"]["right"]["demand_weighted_mean_days"] is None
    )
