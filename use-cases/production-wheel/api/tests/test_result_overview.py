"""Deterministic frontier comparison and compact agent evidence regressions."""

from app.agent.tools.result_overview import result_overview


def test_comparison_counts_each_direction_and_omits_repeated_solver_evidence():
    """Cover dominance, opposite trade-offs, equality and duplicate metadata."""
    points = [dict(point_index=i, demand_weighted_mean_coverage_days=c, j_ch=j,
                   baseline_demand_weighted_mean_coverage_days=21, baseline_j_ch=34)
              for i, c, j in [(1, 46, 0), (2, 20, 27), (3, 18, 40), (4, 22, 35), (5, 21, 34)]]
    result = result_overview({'run': {'metadata': {'points': 'large'}},
                             'metadata': {'points': 'large', 'integrity_valid': True}, 'points': points})
    comparison = result['baseline_comparison']
    assert comparison['below_baseline_coverage'] == {'count': 2, 'point_indices': [2, 3]}
    assert comparison['below_baseline_j_ch']['point_indices'] == [1, 2]
    assert comparison['dominates_baseline']['point_indices'] == [2]
    assert comparison['dominated_by_baseline']['point_indices'] == [4]
    assert comparison['tradeoff_count'] == 2
    assert result['run'] == {}
    assert result['metadata'] == {'integrity_valid': True}


def test_missing_or_inconsistent_baseline_does_not_invent_comparison():
    """Absent evidence must not be replaced with a user supplied benchmark."""
    assert 'baseline_comparison' not in result_overview({'run': {}, 'metadata': {}, 'points': []})
    points = [dict(point_index=i, baseline_demand_weighted_mean_coverage_days=i, baseline_j_ch=34)
              for i in (1, 2)]
    assert 'baseline_comparison' not in result_overview({'run': {}, 'metadata': {}, 'points': points})


def test_result_overview_excludes_frozen_replay_and_raw_metadata_payloads():
    """Agent result evidence must not reintroduce matrix-heavy run snapshots."""
    overview = result_overview(
        {
            "run": {
                "run_id": "run-1",
                "status": "completed",
                "plant_profile": {"matrix_rows": [{"volume_a": "A"}]},
                "request": {"config": {"matrix_pairs": [{"volume_a": "A"}]}},
                "source_request": {"config": {"matrix_pairs": [{"volume_a": "A"}]}},
                "metadata": {"opaque": "hidden"},
            },
            "metadata": {
                "integrity_valid": True,
                "complete": True,
                "matrix_pairs": [{"volume_a": "A"}],
                "source_provenance": {"private": "hidden"},
            },
            "points": [],
        }
    )

    assert overview["run"]["run_id"] == "run-1"
    assert set(overview["run"]).isdisjoint(
        {"plant_profile", "request", "source_request", "metadata"}
    )
    assert overview["metadata"] == {"integrity_valid": True, "complete": True}
