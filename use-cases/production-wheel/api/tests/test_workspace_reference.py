"""Regression contracts for factual optimizer interpretation guidance."""

from app.workspace.reference import reference


def test_reference_always_exposes_minimization_and_separate_evidence_scopes():
    """Keep proof and acceptance distinctions present even in a metric-only lookup."""
    result = reference("coverage")
    rules = result["interpretation_rules"]
    assert rules["objective_directions"] == {
        "coverage_days": "minimize",
        "J_CH": "minimize",
    }
    assert rules["structural_validity_is_business_acceptance"] is False
    assert rules["subproblem_optimality_proves_full_run"] is False
    assert rules["runtime_limited_run_can_be_called_optimal"] is False
    assert rules["historical_comparison_requires_common_population"] is True


def test_exact_proof_and_comparison_topics_are_not_displaced_by_keyword_matches():
    """Retrieve requested proof and comparison sections despite broad word overlap."""
    result = reference("coverage changeovers proof comparison")
    assert set(result["sections"]) == {"coverage", "changeovers", "proof", "comparison"}
    assert "46.36" in result["sections"]["coverage"]
    assert "25.36" in result["sections"]["coverage"]
    assert "VALID_PARETO_POINT" in result["sections"]["proof"]
    assert "not business acceptance" in result["sections"]["proof"]


def test_application_help_covers_every_configuration_field_and_skill():
    """Configuration additions must ship definitions usable by both the UI and agent."""
    from pathlib import Path

    from app.workspace.reference import CONFIGURATION_HELP
    from production_wheel.schemas import RunConfig

    assert set(RunConfig.model_fields) == set(
        CONFIGURATION_HELP["advanced-config"]["fields"]
    )
    skill = (
        Path(__file__).parents[1] / "app/agent/agent_skills/workspace-guide/SKILL.md"
    ).read_text()
    for entry in CONFIGURATION_HELP.values():
        assert entry["text"] in skill
    assert (
        "groups: one proposed group"
        in reference("view groups")["configuration_help"]["views"]["text"]
    )


def test_execution_budget_rejects_exponent_unsupported_by_solver():
    """Reject invalid epsilon bias while editing, before launching an inevitably failed run."""
    import pytest
    from app.workspace.models import ExecutionBudget

    with pytest.raises(ValueError):
        ExecutionBudget(global_epsilon_exponent=0.5)
