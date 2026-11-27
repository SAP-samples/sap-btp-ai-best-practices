"""Nearest-class volume matching: the customer's query, class matrix expansion and profile runs."""

import pytest
from app.workspace.matrix_description import expand_matrix_description
from app.workspace.profile_matrix import validate_matrix_rows
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService
from production_wheel.volume_classes import (
    DEFAULT_CLASS_LABELS, check_labels, classify_volume, default_class_matrix,
    expand_class_matrix, needs_default_classes, with_default_classes)
from test_workspace import dataset


@pytest.mark.parametrize("volume,label", [
    (0.25, "<1"), (0.75, "<1"), (0.999, "<1"), (1, "1"), (1.1, "1"), (1.75, "1"),
    (1.76, "2.5"), (2.2, "2.5"), (3.75, "2.5"), (4, "5"), (7.5, "5"), (7.6, "10"),
    (12.5, "10"), (13, "15"), (15, "15"), (15.01, ">15"), (20, ">15"), (1000, ">15"),
])
def test_classification_matches_the_customer_query(volume, label):
    """Below 1 -> <1, above 15 -> >15, otherwise the nearest stop; ties keep the smaller stop."""
    assert classify_volume(volume, DEFAULT_CLASS_LABELS) == label


def test_default_class_matrix_is_the_customer_matrix_with_no_as_avoid():
    """Y on the diagonal and the six confirmed pairs; every customer NO is AVOID."""
    rows = default_class_matrix()
    validate_matrix_rows(rows, "nearest_class")
    yes = {frozenset((r["volume_a"], r["volume_b"])) for r in rows
           if r["status"] == "Y" and r["volume_a"] != r["volume_b"]}
    assert yes == {frozenset(p) for p in (("<1", "1"), ("1", "2.5"), ("1", "5"), ("2.5", "5"), ("5", "10"), ("10", "15"))}
    assert {r["status"] for r in rows} == {"Y", "AVOID"}


def test_expansion_gives_concrete_pairs_and_names_volumes_without_a_class():
    """Concrete pairs take their class statuses; a volume in a missing tail stays out."""
    rows, classes, missing = expand_class_matrix(default_class_matrix(), [2.2, "4", 12.5, 20])
    status = {(r["volume_a"], r["volume_b"]): r["status"] for r in rows}
    assert classes == [["2.2", "2.5"], ["4", "5"], ["12.5", "10"], ["20", ">15"]]
    assert status["2.2", "4"] == "Y" and status["4", "12.5"] == "Y" and status["12.5", "20"] == "AVOID"
    no_upper = [r for r in default_class_matrix() if ">15" not in (r["volume_a"], r["volume_b"])]
    _, _, missing = expand_class_matrix(no_upper, [20, 2])
    assert missing == ["20"]


@pytest.mark.parametrize("labels,message", [
    (["<1", ">15"], "at least one reference"),
    (["<2", "1", "5"], "smallest"),
    (["1", "5", ">10"], "largest"),
])
def test_tails_must_name_the_extreme_stops(labels, message):
    """'<n' names the smallest stop and '>n' the largest; otherwise the axis is rejected."""
    with pytest.raises(ValueError, match=message):
        check_labels(labels)


def test_exact_profiles_reject_class_labels():
    """Tails need nearest_class volume matching."""
    with pytest.raises(ValueError, match="nearest_class"):
        validate_matrix_rows(default_class_matrix(), "exact")


def test_profile_transformation_for_migration_is_idempotent():
    """with_default_classes switches mode and matrix and leaves every other key."""
    profile = {"name": "P", "plant": "P", "rules": [{"x": 1}],
               "settings": {"matrix_mode": "HARD"}, "matrix_rows": [{"volume_a": "1", "volume_b": "1", "status": "Y"}]}
    migrated = with_default_classes(profile)
    assert needs_default_classes(profile) and not needs_default_classes(migrated)
    assert migrated["settings"] == {"matrix_mode": "HARD", "volume_matching": "nearest_class"}
    assert migrated["rules"] == profile["rules"] and migrated["matrix_rows"] == default_class_matrix()
    assert with_default_classes(migrated) == migrated
    assert profile["settings"] == {"matrix_mode": "HARD"}


def stop_profile_matrix():
    """Default class matrix used as the current matrix in description tests."""
    return default_class_matrix()


def test_add_a_stop_that_inherits_a_neighbour():
    """A 2 L stop copies the 2.5 row, is Y with 2.5 and leaves the other pairs unchanged."""
    rows = expand_matrix_description({"volumes": ["2"], "keep_existing_pairs": True, "default_status": "AVOID",
                                      "inherit_from": {"2": "2.5"}}, stop_profile_matrix(), volume_matching="nearest_class")
    status = {(r["volume_a"], r["volume_b"]): r["status"] for r in rows}
    assert status["2", "2.5"] == "Y" and status["2", "1"] == "Y" and status["2", "5"] == "Y"
    assert status["2", "10"] == "AVOID" and status["2", "<1"] == "AVOID"
    old = {(r["volume_a"], r["volume_b"]): r["status"] for r in stop_profile_matrix()}
    assert all(status[key] == value for key, value in old.items())
    assert classify_volume(2.2, {r["volume_a"] for r in rows}) == "2"


def test_remove_a_stop_and_move_the_upper_tail():
    """remove_volumes drops a stop; moving >15 to >20 renames the tail and keeps its row."""
    rows = expand_matrix_description({"keep_existing_pairs": True, "remove_volumes": ["2.5"], "default_status": "N"},
                                     stop_profile_matrix(), volume_matching="nearest_class")
    assert "2.5" not in {r["volume_a"] for r in rows}
    moved = expand_matrix_description({"volumes": ["20", ">20"], "keep_existing_pairs": True,
                                       "remove_volumes": [">15"], "default_status": "N",
                                       "inherit_from": {"20": "15", ">20": ">15"}},
                                      stop_profile_matrix(), volume_matching="nearest_class")
    status = {(r["volume_a"], r["volume_b"]): r["status"] for r in moved}
    assert ">15" not in {r["volume_a"] for r in moved}
    assert status["20", "15"] == "Y" and status["20", "10"] == "Y" and status[">20", "15"] == "AVOID"
    with pytest.raises(ValueError, match="largest"):
        expand_matrix_description({"volumes": ["20"], "keep_existing_pairs": True, "default_status": "N"},
                                  stop_profile_matrix(), volume_matching="nearest_class")


def test_rules_leave_tail_pairs_and_dataset_rows_are_refused_in_class_mode():
    """Numeric rules never touch '<n'/'>n'; a class matrix does not take dataset volumes as rows."""
    rows = expand_matrix_description({"volumes": ["<1", "1", "5", ">5"], "default_status": "AVOID",
                                      "rules": [{"kind": "max_difference_litres", "litres": 100,
                                                 "within_status": "Y", "outside_status": "N"}]},
                                     volume_matching="nearest_class")
    status = {(r["volume_a"], r["volume_b"]): r["status"] for r in rows}
    assert status["1", "5"] == "Y" and status["<1", "1"] == "AVOID" and status["5", ">5"] == "AVOID"
    with pytest.raises(ValueError, match="classified"):
        expand_matrix_description({"dataset_id": "d", "default_status": "N"}, volume_matching="nearest_class")


def class_service(volumes):
    """Published fixture dataset whose products have the given package volumes, plus a class profile."""
    service = WorkspaceService(MemoryRepository())
    source = dataset(service)
    rows = service.repo.rows(source["dataset_id"], "fini_master")
    template = rows[0]
    rows = [{**template, "material": f"{index:03d}", "package_volume": volume} for index, volume in enumerate(volumes, 1)]
    service.repo.replace_tables(source["dataset_id"], {"fini_master": rows})
    service.publish_dataset(source["dataset_id"])
    profile = service.create_plant_profile({"name": "Classes", "plant": "P", **service.plant_profile_defaults()})
    return service, source, profile


def test_new_profile_runs_volumes_missing_from_any_exact_matrix():
    """12.5 L and 20 L are placed in the 10 and >15 classes, frozen with the draft, and validate."""
    service, source, profile = class_service([1, 12.5, 20])
    assert profile["settings"]["volume_matching"] == "nearest_class"
    draft = service.create_draft(source["dataset_id"], profile["profile_id"])
    config = draft["request"]["config"]
    assert config["volume_matching"] == "nearest_class"
    assert config["matrix_volume_classes"] == [["1", "1"], ["12.5", "10"], ["20", ">15"]]
    status = {(r["volume_a"], r["volume_b"]): r["status"] for r in config["matrix_pairs"]}
    assert status["1", "12.5"] == "AVOID" and status["12.5", "20"] == "AVOID" and status["20", "20"] == "Y"
    validation = service.validate_draft(draft["draft_id"])
    assert validation["valid"] and validation["matrix_coverage"]["complete"], validation
    compiled = service.compiled_draft(service.get_draft(draft["draft_id"])).config
    assert compiled.matrix_volume_classes == (("1", "1"), ("12.5", "10"), ("20", ">15"))
    coverage = service.dataset_volume_coverage(source["dataset_id"], profile["profile_id"], ["<1", "1", "2.5", "5", "10", "12.5", "15", ">15"])
    assert coverage["class_product_counts"] == {"1": 1, "10": 1, ">15": 1}
    assert coverage["reclassified_volumes"] == [{"volume": "12.5", "product_count": 1, "from": "10", "to": "12.5"}]


def test_class_profile_without_upper_tail_names_the_unplaced_volume():
    """A 20 L product without a '>15' class blocks validation with a class-specific message."""
    service, source, profile = class_service([1, 20])
    rows = [r for r in default_class_matrix() if ">15" not in (r["volume_a"], r["volume_b"])]
    profile = service.apply_plant_profile_change(profile["profile_id"], profile["revision"],
                                                 {"action": "update_settings", "matrix_rows": rows})
    draft = service.create_draft(source["dataset_id"], profile["profile_id"])
    result = service.validate_draft(draft["draft_id"])
    assert not result["valid"] and "outside every class" in result["errors"][0]
    missing = result["matrix_coverage"]["missing_volumes"]
    assert [(float(item["volume"]), item["member_count"]) for item in missing] == [(20.0, 1)]
