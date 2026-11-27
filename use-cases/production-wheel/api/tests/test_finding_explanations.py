"""Explanations of dataset review findings for the Workspace Assistant.

Run with:
    cd api && ../.venv/bin/python -m pytest -q tests/test_finding_explanations.py
"""

import re
from pathlib import Path

import pytest

from app.workspace.finding_explanations import FINDING_HELP, MAX_ROWS, explain_finding

UI_FINDINGS = Path(__file__).parents[2] / "ui/src/workspace/findings.js"


def _row(key, sefi, *, fixed="excluded", optimized="excluded", reason="missing_fixed_pv", frequency="08W. Every two months"):
    """Build one fini_master row with the admission fields the explanation reads."""
    return {
        "fini_plant_key": key, "material": key.split("/")[0], "sefi": sefi, "fixed_pv": None,
        "production_frequency": frequency, "model_status": fixed, "exclusion_reason": reason,
        "optimized_pv_model_status": optimized,
    }


def _dataset(issues, settings=None):
    """Build a dataset record with issues and settings metadata."""
    return {"dataset_id": "d1", "issues": issues, "metadata": {"settings": settings or {}}}


def test_optimized_only_finding_lists_rows_modes_and_sefi_versions():
    """The explanation shows the product data, its mode outcome and the SEFI's PV lots."""
    rows = [_row("1/P1", "S1", optimized="modeled"), _row("2/P1", "S2"), _row("3/P1", "S1", fixed="modeled", optimized="modeled", reason=None)]
    versions = [
        {"sefi": "S1", "production_version": "PV - 1", "lot_size_litres": 0},
        {"sefi": "S1", "production_version": "PV - 2", "lot_size_litres": 974},
    ]
    issues = [{"rule_id": "FINI_OPTIMIZED_PV_ONLY", "severity": "warning", "entity_key": "1/P1", "reason": "missing_fixed_pv"},
              {"rule_id": "FINI_DATA_QUALITY", "severity": "warning", "entity_key": "2/P1", "reason": "missing_fixed_pv"}]
    result = explain_finding(_dataset(issues), rows, versions, "FINI_OPTIMIZED_PV_ONLY", "missing_fixed_pv")
    assert result["affected_total"] == 1
    assert result["affected_rows"][0]["fini_plant_key"] == "1/P1"
    assert result["pv_mode_outcome"]["modeled_only_with_optimized"] == 1
    assert result["production_versions_by_sefi"] == {"S1": [
        {"production_version": "PV - 1", "lot_size_litres": 0}, {"production_version": "PV - 2", "lot_size_litres": 974}]}
    assert "OPTIMIZED" in result["help"]["effect"]
    assert result["help"]["reason"]["code"] == "missing_fixed_pv"

    quality = explain_finding(_dataset(issues), rows, versions, "FINI_DATA_QUALITY", "missing_fixed_pv")
    # A SEFI without catalog rows is still listed, with no versions.
    assert quality["production_versions_by_sefi"] == {"S2": []}
    assert quality["pv_mode_outcome"]["excluded_in_both"] == 1


def test_frequency_scope_notes_select_rows_by_out_of_scope_reason():
    """Scope notes read the products from fini_master, not from the finding keys."""
    rows = [
        _row("1/P1", "S1", fixed="out_of_scope", optimized="out_of_scope",
             reason="production_frequency_unrecognized_out_of_scope", frequency="999.Not possible to determine"),
        _row("2/P1", "S1", fixed="out_of_scope", optimized="out_of_scope",
             reason="production_frequency_16w_out_of_scope", frequency="16W. Every four months"),
    ]
    issues = [{"rule_id": "PRODUCTION_FREQUENCY_UNDETERMINED", "severity": "warning",
               "entity_key": "999.Not possible to determine", "observed": "1 FINI rows"}]
    dataset = _dataset(issues, {"modeled_frequencies": ["01W", "02W"]})
    undetermined = explain_finding(dataset, rows, [], "PRODUCTION_FREQUENCY_UNDETERMINED")
    assert [row["fini_plant_key"] for row in undetermined["affected_rows"]] == ["1/P1"]
    assert undetermined["pv_mode_outcome"]["out_of_scope"] == 1
    not_selected = explain_finding(dataset, rows, [], "FREQUENCY_NOT_SELECTED", "16W")
    assert [row["fini_plant_key"] for row in not_selected["affected_rows"]] == ["2/P1"]
    assert not_selected["modeled_frequencies"] == ["01W", "02W"]


def test_rows_are_capped_but_totals_cover_everything():
    """Large findings return the first MAX_ROWS rows and the full total."""
    rows = [_row(f"{i}/P1", "S1", reason="missing_demand") for i in range(MAX_ROWS + 5)]
    issues = [{"rule_id": "FINI_DATA_QUALITY", "severity": "warning", "entity_key": row["fini_plant_key"], "reason": "missing_demand"} for row in rows]
    result = explain_finding(_dataset(issues), rows, [], "FINI_DATA_QUALITY", "missing_demand")
    assert result["affected_total"] == MAX_ROWS + 5
    assert len(result["affected_rows"]) == MAX_ROWS
    assert result["truncated"] is True


def test_unknown_check_is_rejected_with_the_checks_present():
    """An invented check name fails and tells the model which checks exist."""
    with pytest.raises(ValueError, match="FINI_DATA_QUALITY"):
        explain_finding(_dataset([{"rule_id": "FINI_DATA_QUALITY", "entity_key": "1/P1"}]), [], [], "NOT_A_CHECK")


def test_every_ui_check_and_reason_has_assistant_help():
    """The assistant can explain every check and exclusion reason the review shows."""
    source = UI_FINDINGS.read_text()
    checks_block = source[source.index("const CHECKS = {"):source.index("/** Split one finding")]
    ui_checks = set(re.findall(r"^  ([A-Z_]+): \{", checks_block, re.MULTILINE))
    ui_checks |= {"PRODUCTION_FREQUENCY_UNDETERMINED", "FREQUENCY_NOT_SELECTED"}
    assert ui_checks <= set(FINDING_HELP)
    reasons_block = source[source.index("const EXCLUSION_REASONS = {"):source.index("const reasonText")]
    ui_reasons = set(re.findall(r"^  ([a-z_]+): \{", reasons_block, re.MULTILINE))
    assert ui_reasons == set(FINDING_HELP["FINI_DATA_QUALITY"]["reasons"])
    for entry in FINDING_HELP.values():
        assert {"title", "severity", "cause", "effect", "fix"} <= set(entry)
