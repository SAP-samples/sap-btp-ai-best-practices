"""Test in-memory result serialization and read-only historical imports."""

from dataclasses import replace
import json

import pytest

from production_wheel.candidates import (
    CandidateMember,
    ProductionVersion,
    generate_candidate_pools,
)
from production_wheel.greenfield_reporting import write_greenfield_frontier_artifacts
from production_wheel.pareto import build_greenfield_frontier
from production_wheel.result_bundle import build_result_bundle, import_result_bundle
from production_wheel.scenarios import CanonicalInputs
from production_wheel.schemas import CoverageMode, MatrixMode, RunConfig


@pytest.fixture
def result_inputs(tmp_path):
    """Return a small solved frontier and complete FINI input including exclusion."""
    members = tuple(
        CandidateMember(
            fini_id=f"F{i}",
            plant="P1",
            sefi="S1",
            eligible_lines=frozenset({"1"}),
            demand_litres=1000.0,
            pallet_litres=25.0,
            package_volume=0.25,
            fixed_pv="PV1",
        )
        for i in (1, 2)
    )
    versions = (ProductionVersion("P1", "S1", "PV1", 100.0),)
    config = RunConfig(
        scenario_id="bundle-test",
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    result = build_greenfield_frontier(
        generate_candidate_pools(members, versions, config),
        members,
        config,
        block_worker_count=1,
    )
    result = replace(
        result,
        points=tuple(replace(p, point_index=p.point_index + 7) for p in result.points),
    )
    rows = tuple(
        {
            "plant": "P1",
            "sefi": "S1",
            "material": f"F{i}",
            "model_status": "modeled" if i < 3 else "excluded",
            "source_row": str(i),
            "fixed_pv": "PV1",
        }
        for i in (1, 2, 3)
    )
    return result, CanonicalInputs(tmp_path, rows, members, versions)


def test_in_memory_bundle_keeps_rows_and_retained_option_links(result_inputs):
    """All source rows survive; each selected block links to retained definitions."""
    result, inputs = result_inputs
    bundle = build_result_bundle(result, inputs, {"solve_request": {"points": 17}})
    assert len(bundle.tables["members"]) == 3 * len(result.points)
    assert {r["point_index"] for r in bundle.tables["solutions"]} == {
        p.point_index for p in result.points
    }
    assert all(
        r["group_id"] == "" for r in bundle.tables["members"] if r["material"] == "F3"
    )
    groups = {r["group_id"] for r in bundle.tables["groups"]}
    assert all(
        r["group_id"] in groups
        for r in bundle.tables["members"]
        if r["material"] != "F3"
    )
    assert len(bundle.tables["point_options"]) == len(result.points)
    options = {r["option_id"] for r in bundle.tables["block_options"]}
    assert all(r["option_id"] in options for r in bundle.tables["point_options"])
    assert len(bundle.tables["option_members"]) == 2 * len(result.block_options)
    assert isinstance(bundle.tables["option_groups"][0]["coverage_days"], float)
    for option in bundle.tables["option_groups"]:
        assert option["j_ch_contribution"] == option["coefficients"]["j_ch"]
        assert (
            option["matrix_exception_pair_count"]
            == option["coefficients"]["matrix_exceptions"]
        )
    assert bundle.tables["option_members"][0]["demand_litres"] == 1000.0
    assert bundle.metadata["solve_request"] == {"points": 17}
    json.dumps(bundle.metadata, allow_nan=False)
    json.dumps(bundle.tables, allow_nan=False)
    assert not list(inputs.extracted_directory.iterdir())


def test_import_matches_legacy_and_rejects_tampering(result_inputs, tmp_path):
    """Historical import preserves indices and refuses altered point evidence."""
    result, inputs = result_inputs
    path = tmp_path / "historical"
    write_greenfield_frontier_artifacts(
        path, inputs.fini_rows, inputs.production_versions, result
    )
    imported = import_result_bundle(path)
    direct = build_result_bundle(result, inputs)
    for name in (
        "solutions",
        "groups",
        "members",
        "pools",
        "constraints",
        "validation",
        "matrix_pairs",
    ):
        assert len(imported.tables[name]) == len(direct.tables[name])
    assert imported.metadata["option_definitions_available"] is False
    assert imported.metadata["point_option_mapping_available"] is False
    assert imported.tables["point_options"] == []
    assert [r["point_index"] for r in imported.tables["solutions"]] == [
        p.point_index for p in result.points
    ]
    artifact = next((path / "points").glob("*/solution_groups.csv"))
    artifact.write_text(artifact.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="invalid|integrity"):
        import_result_bundle(path)


def test_import_rejects_missing_root(tmp_path):
    """A directory without manifests is never accepted as a result bundle."""
    with pytest.raises(ValueError):
        import_result_bundle(tmp_path)


def test_import_checks_all_manifest_row_counts(result_inputs, tmp_path):
    """Even untouched files are rejected when a declared group count is wrong."""
    result, inputs = result_inputs
    path = tmp_path / "wrong-count"
    write_greenfield_frontier_artifacts(
        path, inputs.fini_rows, inputs.production_versions, result
    )
    manifest_path = next((path / "points").glob("*/run_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["row_counts"]["solution_groups"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="count|integrity"):
        import_result_bundle(path)


def test_import_rejects_duplicate_point_indices(result_inputs, tmp_path):
    """Manifest points must map one-to-one to original frontier indices."""
    result, inputs = result_inputs
    path = tmp_path / "duplicate-point"
    write_greenfield_frontier_artifacts(
        path, inputs.fini_rows, inputs.production_versions, result
    )
    manifest_path = path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["points"].append(manifest["points"][0])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        import_result_bundle(path)


def test_import_rejects_path_escape(tmp_path):
    """Manifest paths outside the supplied root cannot be read as artifacts."""
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"outputs": {"../outside": {}}})
    )
    with pytest.raises(ValueError, match="escapes"):
        import_result_bundle(tmp_path)


def test_scoped_bundle_preserves_original_rows_and_separates_run_scope(result_inputs):
    """A limited run keeps other source records and their original admission status."""
    result, inputs = result_inputs
    extra = {
        "plant": "P2",
        "sefi": "S2",
        "material": "OUTSIDE",
        "model_status": "modeled",
        "source_row": "4",
        "fixed_pv": "PV2",
        "exclusion_reason": "",
    }
    full_inputs = replace(inputs, fini_rows=(*inputs.fini_rows, extra))
    bundle = build_result_bundle(result, full_inputs)
    for row in bundle.tables["members"]:
        if row["material"] == "OUTSIDE":
            assert row["model_status"] == "modeled"
            assert row["run_model_status"] == "out_of_scope"
            assert row["run_exclusion_reason"] == "outside_run_scope_or_disposition"
            assert row["exclusion_reason"] == ""
            assert row["group_id"] == ""
        elif row["material"] == "F3":
            assert row["model_status"] == "excluded"
            assert row["run_model_status"] == "excluded"
        else:
            assert row["run_model_status"] == "modeled"
            assert row["run_exclusion_reason"] == ""
    assert len(bundle.tables["members"]) == len(full_inputs.fini_rows) * len(
        result.points
    )
