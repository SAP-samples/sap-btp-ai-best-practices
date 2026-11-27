"""End-to-end SAP settings: selection, generation, persistence, query and XLSX export.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_sap_settings_api.py -q
"""

from __future__ import annotations

import io

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app.routers.decisions import router as decisions_router
from app.routers.sap_settings import router as sap_router
from app.routers.workspace import router as workspace_router
from app.workspace.dependencies import get_service
from app.workspace.repository import MemoryRepository
from app.workspace.sap_settings_template import TEMPLATE_SHEETS
from app.workspace.service import WorkspaceService


def _member(material, subgroup, pv="PV - 2", frequency="02W"):
    """One wheel FINI of SEFI S1 at point 1 (base coverage 10 days, one pallet per batch)."""

    return {"point_index": 1, "plant": "P1", "sefi": "S1", "material": material,
            "proposed_subgroup_code": subgroup, "selected_pv": pv, "base_group_coverage_days": 10.2,
            "legacy_nominal_pallet_count": 1.0, "forecast_litres_12m": 1000, "sefi_forecast_litres": 2000,
            "production_frequency_code": frequency, "pck_code": "K1", "package_volume": 5,
            "common_lines": "L1", "reference_coverage_source_days": 10}


def _client():
    """Published dataset with SAP lookups and one completed single-point run."""

    repo = MemoryRepository()
    service = WorkspaceService(repo)
    lookups = {
        "dc_network": [{"material": m, "plant": "P1", "dc": "D1", "has_forecast": "YES"} for m in ("100", "200", "900")],
        "dc_abc_classes": [{"material": "900", "dc": "D1", "abc_class": "A"}],
        "dc_pallet_quantities": [{"material": m, "pallet_qty": 40, "layer_qty": 10, "package_volume": 5}
                                 for m in ("100", "200", "900")],
        "production_version_masses": [{"sefi": "S1", "production_version": "PV - 2", "lot_size_kg": 1300}],
        "recipe_characters": [{"list_of_characters_allowed_in_bw": c} for c in "A10"],
    }
    repo.insert("datasets", "d", {"dataset_id": "d", "revision": 1, "status": "published",
                                  "metadata": {"plant": "P1", "counts": {view: len(rows) for view, rows in lookups.items()}}})
    repo.replace_tables("d", lookups)
    profile = service.create_plant_profile({"name": "Plant plan", "plant": "P1", **service.plant_profile_defaults()})
    repo.insert("runs", "r", {"run_id": "r", "revision": 1, "dataset_id": "d", "status": "succeeded",
                              "results_ready": True, "plant_profile_id": profile["profile_id"],
                              "plant_profile": profile, "plant_profile_revision": profile["revision"],
                              "metadata": {"config": {"demand_days": 250, "productive_weeks": 50}}})
    repo.replace_tables("r", {
        "solutions": [{"point_index": 1, "j_ch": 0.5, "demand_weighted_mean_coverage_days": 10.2,
                       "group_count": 2, "validation_status": "valid"}],
        "members": [_member("100", "G1"), _member("200", "G1"), _member("300", "", frequency="01W")],
        "groups": [{"point_index": 1, "frequency_per_week": 0.5}],
    })
    app = FastAPI()
    for router in (decisions_router, sap_router, workspace_router):
        app.include_router(router)
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app), service, repo


def _set_calendar_codes(service, odd="ODD-W1", even="EVEN-W2"):
    """Store generic odd/even lot-size profiles in the run's plant profile."""

    profile_id = service.get_run("r")["plant_profile_id"]
    revision = service.get_plant_profile(profile_id)["revision"]
    return service.apply_plant_profile_change(profile_id, revision, {"action": "update_settings", "settings": {
        "sap_settings": {"calendar_odd_profile": odd, "calendar_even_profile": even}}})


def test_generation_requires_a_selected_scenario() -> None:
    """Without a decision there is nothing to translate into SAP settings."""

    client, _, _ = _client()
    response = client.post("/api/runs/r/sap-settings", json={"period_label": "26Q4"})
    assert response.status_code == 422
    assert "select a decision scenario" in response.json()["detail"]
    assert client.post("/api/runs/r/sap-settings", json={"period_label": "bad;label"}).status_code == 422


def test_generation_persists_rows_and_summary() -> None:
    """Tabs are written to HANA views and the summary to the run payload."""

    client, service, repo = _client()
    profile = _set_calendar_codes(service)
    assert client.put("/api/runs/r/decision-selection", json={"point_index": 1}).status_code == 200
    summary = client.post("/api/runs/r/sap-settings", json={"period_label": "26Q4"}).json()

    assert {name: tab["status"] for name, tab in summary["tabs"].items()} == {
        "pv_check": "ready", "kpis": "ready", "cycle_stock": "ready", "planning_calendar": "ready",
        "recipe_combination": "ready", "apo_lot_size": "ready", "central_dc_lot_size": "not_applicable",
        "runner_audit": "ready"}
    assert summary["kpi_changeovers_per_week"] == 0.5 == summary["point_j_ch"]
    assert summary["sap_settings_source"]["origin"] == "live"
    assert summary["profile_revision"] == profile["revision"]
    assert summary["policy_sources"]["calendar_odd_profile"] == "profile"
    assert summary["policy_sources"]["recipe_characters"] == "dataset:recipe_characters"
    lots = {row["material"]: row for row in repo.rows("r", "sap_dc_lot_sizes", {"point_index": 1})}
    assert (lots["100"]["lot_size_code"], lots["900"]["lot_size_code"]) == ("10", "05")
    assert repo.rows("r", "sap_recipe_combinations", {"point_index": 1})[0]["recipe_combination"] == "0"
    assert client.get("/api/runs/r/sap-settings").json()["sap_settings"]["period_label"] == "26Q4"
    assert service.get_run("r")["sap_settings"]["generated_at"]
    queried = client.post("/api/query", json={"view": "sap_apo_lot_sizes", "run_id": "r", "point_index": 1})
    assert queried.status_code == 200


def test_export_follows_the_template_and_a_new_selection_clears_outputs() -> None:
    """XLSX reproduces the template layout; choosing again invalidates generated settings."""

    client, service, repo = _client()
    client.put("/api/runs/r/decision-selection", json={"point_index": 1})
    client.post("/api/runs/r/sap-settings", json={"period_label": "26Q4"})
    response = client.get("/api/runs/r/sap-settings/export")
    assert response.status_code == 200
    book = load_workbook(io.BytesIO(response.content))
    assert book.sheetnames == [
        "Check - Several Lot size in Sub", "RoC and Change overs", "Cycle stock", "FINI Planning calendar",
        "Recipe combination", "APO Lot size", "Central DC lot size", "Comment for APO", "Generation summary",
        "Audit - PV check", "Audit - RoC by subgroup", "Audit - Cycle stock", "Audit - Planning calendar",
        "Audit - APO lot size", "Runner RoC audit"]
    # Every template table sits at its template anchor with its template headers.
    for spec in TEMPLATE_SHEETS:
        for table in spec.tables:
            assert book[spec.name].tables[table.name].ref.split(":")[0] == table.anchor
    cycle = book["Cycle stock"]
    assert [cell.value for cell in cycle["A6:H6"][0]][:5] == [
        "Plant", "Material", "Lot size", "Min lot size at DC", "Rounding value"]
    assert (cycle["A7"].value, cycle["B7"].value, cycle["C7"].value, cycle["D7"].value) == ("D1", "100", "10", 40)
    assert cycle["E6"].fill.fill_type is None and cycle["F6"].fill.fgColor.rgb == "FFA5A5A5"  # blue upload vs grey info
    roc = book["RoC and Change overs"]
    assert (roc["A2"].value, roc["C2"].value) == (10, "P1")  # template averages the rounded RoC
    assert (roc["B6"].value, roc["E6"].value) == ("Site total", 0.5)
    calendar = book["FINI Planning calendar"]
    assert calendar["A3"].value == "BLOCKED: Set the odd- and even-week lot-size profiles in the plant profile (Settings > SAP settings)."
    assert calendar.tables["FINI_Planning_calendar_at_P_site"].ref == "A4:E5" and calendar["A5"].value is None
    assert book["APO Lot size"]["D6"].value == "Production Wheel 26Q4"
    assert book["Comment for APO"].sheet_state == "hidden" and book["Comment for APO"]["A2"].value == "26Q4"
    audit = list(book["Audit - Cycle stock"].values)
    assert audit[0][9:13] == ("RoC selected PV", "Pallets as per subgroup", "Pallet-adjusted days", "Cycle days")

    client.put("/api/runs/r/decision-selection", json={"point_index": 1})
    assert service.get_run("r")["sap_settings"] is None
    assert repo.rows("r", "sap_dc_lot_sizes") == []
    assert client.get("/api/runs/r/sap-settings/export").status_code == 409


def test_runner_classes_use_the_run_profile_thresholds() -> None:
    """SAP runner classes follow the cut-offs frozen with the run's plant profile.

    FINI 200 covers 90 days alone, 9 times the SEFI's fastest (10 days). With the
    template defaults (ratio 3 / 10) it is Medium; a profile whose low ratio is
    5 makes it Low. Runs without frozen cut-offs fall back to the defaults.
    """

    client, service, repo = _client()
    slow = {**_member("200", "G1"), "reference_coverage_source_days": 90}
    repo.replace_tables("r", {**{v: repo.rows("r", v) for v in ("solutions", "groups")},
                              "members": [_member("100", "G1"), slow]})
    client.put("/api/runs/r/decision-selection", json={"point_index": 1})
    summary = client.post("/api/runs/r/sap-settings", json={"period_label": "26Q4"}).json()
    classes = {row["material"]: row["runner_class"] for row in repo.rows("r", "sap_runner_adjustments")}
    assert classes == {"100": "High", "200": "Medium"}
    assert summary["runner_thresholds_source"].startswith("defaults")

    run = service.get_run("r")
    thresholds = {"high_max_days": 10, "low_min_days": 80, "high_max_ratio": 3, "low_min_ratio": 5}
    repo.cas("runs", "r", run["revision"], {"metadata": {"config": {
        "demand_days": 250, "productive_weeks": 50, "runner_thresholds": thresholds}}})
    summary = client.post("/api/runs/r/sap-settings", json={"period_label": "26Q4"}).json()
    classes = {row["material"]: row["runner_class"] for row in repo.rows("r", "sap_runner_adjustments")}
    assert classes == {"100": "High", "200": "Low"}
    assert summary["runner_thresholds"] == thresholds
    assert summary["runner_thresholds_source"] == "run plant profile"
    assert summary["cycle_stock_policy_source"].startswith("defaults")
    assert summary["cycle_stock_policy"]["max_days"] == 200

    # The run's frozen cycle stock policy feeds the SAP summary and the cycle-stock cap.
    run = service.get_run("r")
    cycle = {"mode": "single", "high_min_days": 0, "medium_min_days": 20, "low_min_days": 40,
             "medium_batch_factor": 0.5, "max_days": 150, "mixed_groups_only": True}
    repo.cas("runs", "r", run["revision"], {"metadata": {"config": {
        "demand_days": 250, "productive_weeks": 50, "runner_thresholds": thresholds, "cycle_stock_policy": cycle}}})
    summary = client.post("/api/runs/r/sap-settings", json={"period_label": "26Q4"}).json()
    assert summary["cycle_stock_policy"] == cycle
    assert summary["cycle_stock_policy_source"] == "run plant profile"


def test_profile_codes_fixed_after_the_run_apply_without_a_rerun() -> None:
    """Generation reads the live profile: a code added after the run unblocks the calendar."""

    client, service, _ = _client()
    client.put("/api/runs/r/decision-selection", json={"point_index": 1})
    first = client.post("/api/runs/r/sap-settings", json={"period_label": "26Q4"}).json()
    assert first["tabs"]["planning_calendar"]["status"] == "blocked"
    readiness = client.get("/api/runs/r/sap-settings/readiness").json()
    assert readiness["last_summary"]["stale"] is False
    gap = readiness["tabs"]["planning_calendar"]["gaps"][0]
    assert gap["code"] == "profile_setting" and "calendar_odd_profile" in gap["setting"]
    assert readiness["policy"]["calendar_odd_profile"] == {"value": None, "source": "missing"}

    profile = _set_calendar_codes(service)
    readiness = client.get("/api/runs/r/sap-settings/readiness").json()
    assert readiness["last_summary"]["stale"] is True and "regenerate" in readiness["last_summary"]["stale_reason"]
    assert readiness["tabs"]["planning_calendar"]["status"] == "ready"
    second = client.post("/api/runs/r/sap-settings", json={"period_label": "26Q4"}).json()
    assert second["tabs"]["planning_calendar"]["status"] == "ready"
    assert second["profile_revision"] == profile["revision"]


def test_readiness_matches_a_real_generation_and_explains_missing_inputs() -> None:
    """Dry-run tab statuses equal the real ones; without a point only input gaps are known."""

    client, service, repo = _client()
    before = client.get("/api/runs/r/sap-settings/readiness").json()
    assert before["point_source"] == "none" and before["can_generate"] is False
    assert before["tabs"]["planning_calendar"]["status"] == "blocked"
    assert before["tabs"]["pv_check"]["status"] == "unknown"
    assert before["tabs"]["central_dc_lot_size"]["status"] == "not_applicable"
    requested = client.get("/api/runs/r/sap-settings/readiness?point_index=1").json()
    assert requested["point_source"] == "requested"
    assert client.get("/api/runs/r/sap-settings/readiness?point_index=7").status_code == 422

    client.put("/api/runs/r/decision-selection", json={"point_index": 1})
    readiness = client.get("/api/runs/r/sap-settings/readiness").json()
    summary = client.post("/api/runs/r/sap-settings", json={"period_label": "26Q4"}).json()
    assert {k: v["status"] for k, v in readiness["tabs"].items()} == {k: v["status"] for k, v in summary["tabs"].items()}
    assert readiness["blocking_tabs"] == ["planning_calendar"]

    # Without a RecipeCharacters table the recipe tab and APO are blocked, with the fix.
    dataset = repo.get("datasets", "d")
    counts = {k: v for k, v in dataset["metadata"]["counts"].items() if k != "recipe_characters"}
    repo.cas("datasets", "d", dataset["revision"], {"metadata": {**dataset["metadata"], "counts": counts}})
    readiness = client.get("/api/runs/r/sap-settings/readiness").json()
    assert readiness["tabs"]["recipe_combination"]["gaps"][0]["view"] == "recipe_characters"
    assert readiness["tabs"]["apo_lot_size"]["status"] == "blocked"


def test_removed_profile_and_runs_without_profile_fall_back_explicitly() -> None:
    """A removed profile uses the frozen copy; a run without profile uses defaults; both say so."""

    client, service, repo = _client()
    _set_calendar_codes(service)
    profile_id = service.get_run("r")["plant_profile_id"]
    service.remove_plant_profile(profile_id, service.repo.get("profiles", profile_id)["revision"], True)
    readiness = client.get("/api/runs/r/sap-settings/readiness").json()
    assert readiness["profile"]["origin"] == "frozen" and "removed" in readiness["notes"][0]
    # The frozen copy predates the codes, so the calendar is still blocked.
    assert readiness["tabs"]["planning_calendar"]["status"] == "blocked"

    run = service.get_run("r")
    repo.cas("runs", "r", run["revision"], {"plant_profile_id": None, "plant_profile": None})
    readiness = client.get("/api/runs/r/sap-settings/readiness").json()
    assert readiness["profile"]["origin"] == "none" and "no plant profile" in readiness["notes"][0]


def test_readiness_waits_for_published_results() -> None:
    """Before results are published there is nothing to check yet."""

    client, service, repo = _client()
    run = service.get_run("r")
    repo.cas("runs", "r", run["revision"], {"results_ready": False, "status": "running"})
    payload = client.get("/api/runs/r/sap-settings/readiness").json()
    assert payload["available"] is False and "running" in payload["reason"]


def _add_calendar_dataset(repo, dataset_id, plant="P1", created_at="2026-01-01"):
    """Older dataset of a plant whose workbook carried the previous wheel's calendar table."""

    calendar = [{"material": "100", "lot_size": "PK", "lot_size_profile": "ODD-W1"},
                {"material": "200", "lot_size": "PK", "lot_size_profile": "EVEN-W2"}]
    finis = [{"material": "100", "sefi": "S1", "sefi_forecast_litres": "2000"},
             {"material": "200", "sefi": "S2", "sefi_forecast_litres": "1000"}]
    repo.insert("datasets", dataset_id, {"dataset_id": dataset_id, "name": f"Older {dataset_id}", "revision": 1,
                                         "status": "published", "created_at": created_at,
                                         "metadata": {"plant": plant, "counts": {"planning_calendar_inform": 2, "fini_master": 2}}})
    repo.replace_tables(dataset_id, {"planning_calendar_inform": calendar, "fini_master": finis})


def test_calendar_codes_are_suggested_from_an_older_dataset_of_the_plant() -> None:
    """The run's dataset has no calendar table; an older dataset of the same plant does."""

    client, service, repo = _client()
    _add_calendar_dataset(repo, "old")
    _add_calendar_dataset(repo, "other-plant", plant="P9", created_at="2026-02-01")
    profile_id = service.get_run("r")["plant_profile_id"]
    payload = client.get(f"/api/plant-profiles/{profile_id}/sap-settings/suggestions").json()
    [suggestion] = payload["suggestions"]
    assert suggestion["status"] == "suggested" and suggestion["source"] == "dataset:old:planning_calendar_inform"
    assert suggestion["patch"] == {"sap_settings": {"calendar_odd_profile": "ODD-W1", "calendar_even_profile": "EVEN-W2"}}
    assert suggestion["differs"] is True
    assert [item["dataset_id"] for item in payload["searched"]] == ["old"]

    readiness = client.get("/api/runs/r/sap-settings/readiness").json()
    assert readiness["suggestions"][0]["values"]["calendar_odd_profile"] == "ODD-W1"
    # Nothing was applied: the calendar stays blocked until the profile is saved.
    assert readiness["tabs"]["planning_calendar"]["status"] == "blocked"
    revision = service.get_plant_profile(profile_id)["revision"]
    service.apply_plant_profile_change(profile_id, revision, {"action": "update_settings", "settings": suggestion["patch"]})
    readiness = client.get("/api/runs/r/sap-settings/readiness").json()
    assert readiness["suggestions"] == [] and readiness["tabs"]["planning_calendar"]["status"] == "unknown"
