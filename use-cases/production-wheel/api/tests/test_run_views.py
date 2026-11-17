"""Contracts for compact, safe optimizer run read models."""


def test_run_summary_excludes_replay_payload_and_reports_matrix_summary():
    """List/status output keeps lifecycle evidence without frozen replay objects."""
    from app.workspace.run_views import run_status

    record = {
        "run_id": "run-1",
        "revision": 3,
        "dataset_id": "dataset-1",
        "draft_id": "draft-1",
        "title": "Scenario",
        "plant_profile_id": "profile-1",
        "plant_profile_revision": 4,
        "plant_profile": {
            "profile_id": "profile-1",
            "name": "Profile",
            "plant": "P1",
            "revision": 4,
            "settings": {
                "horizon_days": 80,
                "demand_days_per_week": 5,
                "high_runner_threshold_days": 15,
                "runner_basis": "reference",
                "matrix_mode": "FLEXIBLE",
                "maximum_group_size": 7,
            },
            "rules": [{"constraint_id": "runner-mix"}],
            "matrix_rows": [
                {"volume_a": "A", "volume_b": "B", "status": "allowed"}
            ],
        },
        "request": {"config": {"matrix_pairs": [{"volume_a": "A"}]}},
        "source_request": {"config": {"matrix_pairs": [{"volume_a": "A"}]}},
        "validation": {"effective_request": {"config": {"matrix_pairs": [{"volume_a": "A"}]}}},
        "budget": {"frontier_points": 17},
        "parameter_sources": {"request.config.matrix_pairs": {"origin": "plant_profile"}},
        "metadata": {"internal": "opaque"},
        "status": "failed",
        "stage": "publication_failed",
        "error": "Solve process exited with code 1",
        "created_at": "2026-09-16T10:00:00+00:00",
        "started_at": "2026-09-16T10:01:00+00:00",
        "finished_at": "2026-09-16T10:02:00+00:00",
        "progress": {"current": 1, "total": 2},
        "results_ready": False,
    }

    summary = run_status(record)

    assert summary["run_id"] == "run-1"
    assert summary["configuration"]["matrix_pair_count"] == 1
    assert summary["configuration"]["horizon_days"] == 80
    assert summary["configuration"]["fixed_rule_ids"] == ["runner-mix"]
    assert set(summary).isdisjoint(
        {"plant_profile", "request", "source_request", "validation", "budget", "metadata"}
    )


def test_failure_diagnostics_summarizes_checkpoint_block_failures():
    """A failed run exposes bounded causal evidence instead of its raw checkpoint."""
    import json

    from app.workspace.run_views import failure_diagnostics

    diagnostics = failure_diagnostics(
        {
            "run_id": "run-1",
            "status": "failed",
            "stage": "publication_failed",
            "error": "Solve process exited with code 1",
        },
        json.dumps(
            {
                "metadata": {"complete": False, "integrity_valid": False},
                "tables": {
                    "block_failures": [
                        {
                            "plant": "P1",
                            "sefi": "S1",
                            "candidate_count": 12,
                            "error_type": "uncovered_precheck",
                            "error_message": "No eligible candidate covers this block",
                        },
                        {
                            "plant": "P2",
                            "sefi": "S2",
                            "candidate_count": 5,
                            "error_type": "uncovered_precheck",
                            "error_message": "No eligible candidate covers this block",
                        },
                    ]
                },
            }
        ).encode(),
    )

    assert diagnostics["checkpoint_available"] is True
    assert diagnostics["complete"] is False
    assert diagnostics["integrity_valid"] is False
    assert diagnostics["failure_types"] == ["uncovered_precheck"]
    assert diagnostics["affected_block_count"] == 2
    assert diagnostics["candidate_count"] == 17
    assert diagnostics["affected_blocks"] == [
        {
            "plant": "P1",
            "sefi": "S1",
            "candidate_count": 12,
            "error_type": "uncovered_precheck",
            "error_message": "No eligible candidate covers this block",
        },
        {
            "plant": "P2",
            "sefi": "S2",
            "candidate_count": 5,
            "error_type": "uncovered_precheck",
            "error_message": "No eligible candidate covers this block",
        },
    ]


def test_matrix_page_reads_frozen_profile_with_filters_and_pagination():
    """Matrix evidence remains available for failed runs without leaking every pair."""
    from app.workspace.run_views import matrix_page

    page = matrix_page(
        {
            "run_id": "run-1",
            "status": "failed",
            "plant_profile": {
                "matrix_rows": [
                    {"volume_a": "A", "volume_b": "B", "status": "allowed"},
                    {"volume_a": "A", "volume_b": "C", "status": "blocked"},
                    {"volume_a": "B", "volume_b": "C", "status": "allowed"},
                ]
            },
        },
        offset=0,
        limit=1,
        status="allowed",
    )

    assert page == {
        "run_id": "run-1",
        "total": 2,
        "offset": 0,
        "limit": 1,
        "truncated": True,
        "rows": [{"volume_a": "A", "volume_b": "B", "status": "allowed"}],
    }


def test_failure_diagnostics_bounds_sampled_error_messages():
    """One pathological checkpoint message cannot turn a diagnostic into a large payload."""
    import json

    from app.workspace.run_views import failure_diagnostics

    diagnostics = failure_diagnostics(
        {"run_id": "run-1", "status": "failed"},
        json.dumps(
            {
                "tables": {
                    "block_failures": [
                        {"error_type": "solver_error", "error_message": "x" * 10_000}
                    ]
                }
            }
        ).encode(),
    )

    assert len(diagnostics["affected_blocks"][0]["error_message"]) <= 4_000


def test_failure_diagnostics_tolerates_non_numeric_checkpoint_candidate_counts():
    """Malformed diagnostic evidence remains readable instead of failing the status view."""
    import json

    from app.workspace.run_views import failure_diagnostics

    diagnostics = failure_diagnostics(
        {"run_id": "run-1", "status": "failed"},
        json.dumps(
            {
                "tables": {
                    "block_failures": [
                        {"candidate_count": "unknown", "error_type": "solver_error"}
                    ]
                }
            }
        ).encode(),
    )

    assert diagnostics["candidate_count"] == 0
