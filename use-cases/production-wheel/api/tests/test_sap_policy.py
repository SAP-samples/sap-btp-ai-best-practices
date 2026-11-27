"""Plant policy rules: validation and resolution order.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_sap_policy.py -q
"""

from __future__ import annotations

import pytest
from production_wheel.sap_settings.policy import class_value, resolve_policy, validate_rows


def test_defaults_then_all_plants_then_own_plant() -> None:
    """A plant row beats a '*' row, which beats the code default."""

    rows = validate_rows([
        {"plant": "*", "setting": "dc_rounding", "value": "pallet"},
        {"plant": "P1", "setting": "dc_rounding", "abc_class": "A", "value": "layer"},
        {"plant": "P1", "setting": "dc_min_lot", "abc_class": "c", "value": "litres", "number": 300},
        {"plant": "P2", "setting": "abc_cycle_days", "abc_class": "B", "number": 15},
    ])
    first, second = resolve_policy(rows, "P1"), resolve_policy(rows, "P2")

    assert class_value(first, "dc_rounding", "A") == ("layer", None)
    assert class_value(first, "dc_rounding", "B") == ("pallet", None)
    assert class_value(first, "dc_min_lot", "C") == ("litres", 300)
    assert class_value(first, "dc_min_lot", None) == ("pallet", None)
    assert class_value(first, "abc_cycle_days", "B") == 10
    assert class_value(second, "abc_cycle_days", "B") == 15
    assert class_value(second, "abc_cycle_days", "A") == 5
    assert first["calendar_odd_profile"] is None


def test_every_problem_is_reported_at_once() -> None:
    """Unknown settings, misplaced classes, bad rules and duplicates fail together."""

    with pytest.raises(ValueError) as error:
        validate_rows([
            {"plant": "P1", "setting": "unknown"},
            {"plant": "P1", "setting": "calendar_lot_size", "abc_class": "A", "value": "PK"},
            {"plant": "P1", "setting": "dc_min_lot", "value": "litres"},
            {"plant": "P1", "setting": "dc_rounding", "value": "crate"},
            {"plant": "P1", "setting": "calendar_odd_profile"},
            {"plant": "P1", "setting": "abc_cycle_days", "abc_class": "A", "number": 5},
            {"plant": "P1", "setting": "abc_cycle_days", "abc_class": "A", "number": 6},
        ])
    message = str(error.value)
    for fragment in ("unknown setting", "ABC class not allowed", "litres rule needs", "rule must be one of",
                     "text value is required", "duplicate row"):
        assert fragment in message


def test_runner_cut_offs_moved_to_the_plant_profile() -> None:
    """Former runner class rows are rejected with a pointer, but still readable for migration."""

    from production_wheel.sap_settings.policy import legacy_runner_thresholds

    with pytest.raises(ValueError, match="now set in the plant profile"):
        validate_rows([{"plant": "P1", "setting": "runner_high_days", "number": 9}])
    rows = [{"plant": "*", "setting": "runner_high_days", "number": 9},
            {"plant": "P1", "setting": "runner_high_days", "number": 7},
            {"plant": "P1", "setting": "runner_ratio_low", "number": 12}]
    assert legacy_runner_thresholds(rows, "P1") == {"high_max_days": 7, "low_min_ratio": 12}
    assert legacy_runner_thresholds(rows, "P2") == {"high_max_days": 9}


def test_runner_multiple_settings_moved_to_the_plant_profile() -> None:
    """Former runner RoC and cap rows are rejected, but still readable for migration."""

    from production_wheel.sap_settings.policy import legacy_cycle_stock_policy

    with pytest.raises(ValueError, match="cycle stock strategy"):
        validate_rows([{"plant": "P1", "setting": "max_cycle_days", "number": 150}])
    rows = [{"plant": "*", "setting": "runner_low_min_days", "number": 50},
            {"plant": "P1", "setting": "max_cycle_days", "number": 150},
            {"plant": "P1", "setting": "runner_medium_batch_factor", "number": 0.4}]
    assert legacy_cycle_stock_policy(rows, "P1") == {"low_min_days": 50, "max_days": 150, "medium_batch_factor": 0.4}
    assert legacy_cycle_stock_policy(rows, "P2") == {"low_min_days": 50}


def test_policy_api_is_read_only_after_the_move_to_plant_profiles() -> None:
    """GET still lists leftover rows and the catalogue; PUT is refused with a pointer to the profile."""

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routers.sap_settings import router
    from app.workspace.dependencies import get_service
    from app.workspace.repository import MemoryRepository
    from app.workspace.service import WorkspaceService

    app = FastAPI()
    app.include_router(router)
    service = WorkspaceService(MemoryRepository())
    service.repo.replace_tables("sap-policy", {"sap_policy_rules": [
        {"plant": "Z", "setting": "calendar_odd_profile", "abc_class": "*", "value": "ODD-W9", "number": None}]})
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)

    payload = client.get("/api/sap-policy").json()
    assert payload["settings"]["abc_cycle_days"]["default"] == {"A": 5, "B": 10, "C": 20}
    assert payload["deprecated"] is True and payload["rows"][0]["value"] == "ODD-W9"
    refused = client.put("/api/sap-policy", json={"rows": []})
    assert refused.status_code == 410 and "plant profile" in refused.json()["detail"]
    assert len(service.repo.rows("sap-policy", "sap_policy_rules")) == 1
