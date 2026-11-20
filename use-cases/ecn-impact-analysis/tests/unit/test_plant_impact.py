"""Plant table rows from rule-derived BOM change plants and plant-matched master data.

Run with::

    PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_plant_impact.py
"""

from app.ecn.evidence.plant_impact import build_plant_impact

PLANT_A = {"plant_code": "PL10", "plant_name": "Plant A"}
PLANT_B = {"plant_code": "PK01", "plant_name": "Plant B"}


def line(change, *, plants=(PLANT_A,), vc="Q900VV0080402", outgoing=None, incoming=None, reason=None, **dates):
    """Build one BOM change line with rule-derived plants."""
    return {"change": change, "plants": list(plants), "vc_groups": [vc], "plant_reason": reason, "unmapped_brands": [],
            "manufacturing_date": dates.get("manufacturing_date"), "end_date": dates.get("end_date"),
            "assembly": dates.get("assembly"), "assembly_description": dates.get("assembly_description"),
            "model_range": dates.get("model_range"), "levels": dates.get("levels", []),
            "qualifiers": dates.get("qualifiers", []),
            "outgoing": {"number": outgoing, "description": f"{outgoing} text"} if outgoing else None,
            "incoming": {"number": incoming, "description": f"{incoming} text"} if incoming else None}


def changes():
    """ECN346863-like changes: a Plant A removal and addition, plus one VC group built in both plants."""
    return {
        "lines": [line("REMOVED", outgoing="QKX2833570", end_date="2026-05-18"),
                  line("ADDED", incoming="QKX7702260", manufacturing_date="2026-05-19"),
                  line("ADDED", incoming="QKX0187890", plants=(PLANT_A, PLANT_B), vc="Q900VV0080301",
                       manufacturing_date="2026-05-19")],
        "materials": [{"number": "QKX2833570", "role": "OUTGOING"}, {"number": "QKX7702260", "role": "INCOMING"},
                      {"number": "QKX0187890", "role": "INCOMING"}],
    }


def test_rule_plants_count_lines_and_only_plant_matched_master_data():
    """Master data counts only where the rules place the material; other plants go to the footer."""
    refs = [
        {"dataset_type": "STOCK", "material_number": "QKX2833570", "plant_code": "PL10", "unrestricted_quantity": "3"},
        {"dataset_type": "STOCK", "material_number": "QKX2833570", "plant_code": "PL10", "unrestricted_quantity": "2"},
        {"dataset_type": "STOCK", "material_number": "QKX2833570", "plant_code": "PK01", "unrestricted_quantity": "9"},
        {"dataset_type": "MATERIAL_MASTER", "material_number": "QKX7702260", "plant_code": "PK01"},
        {"dataset_type": "MATERIAL_MASTER", "material_number": "QKX0187890", "plant_code": "PK01"},
        {"dataset_type": "PURCHASE_INFO_RECORDS", "material_number": "QKX0187890", "plant_code": "PL10"},
        {"dataset_type": "MATERIAL_MASTER", "material_number": "QKX0187890", "plant_code": "PL10"},
        {"dataset_type": "STOCK", "material_number": "UNRELATED", "plant_code": "XX01", "unrestricted_quantity": "1"},
        {"dataset_type": "STOCK", "material_number": "QKX2833570", "plant_code": "PL10", "unrestricted_quantity": "7", "comparison": True},
    ]
    source = {"CNAffectLinks": [{"InventoryDisposition": {"Value": "USEEXISTING", "Display": "Use Existing"}}, {}]}
    result = build_plant_impact(changes(), refs, source)
    pk01, pl10 = result["plants"]
    assert (pl10["plant_code"], pl10["status"], pl10["plant_name"]) == ("PL10", "CONFIRMED", "Plant A")
    assert pl10["changes"] == {"REMOVED": 1, "ADDED": 2} and pl10["cutover_from"] == pl10["cutover_to"] == "2026-05-19"
    assert pl10["old_part_stock"] == [{"number": "QKX2833570", "quantity": "5"}]
    assert pl10["new_parts_with_purchasing"] == ["QKX0187890"]
    assert (pk01["plant_name"], pk01["changes"]) == ("Plant B", {"ADDED": 1})
    assert pk01["new_parts_without_purchasing"] == ["QKX0187890"] and pk01["old_part_stock"] == []
    assert [m["number"] for m in pl10["materials"]] == ["QKX2833570", "QKX0187890", "QKX7702260"]
    assert [(m["number"], m["vc_groups"]) for m in pk01["materials"]] == [("QKX0187890", ["Q900VV0080301"])]
    assert result["records_outside_rule_plants"] == [{"plant_code": "PK01", "materials": ["QKX2833570", "QKX7702260"]}]
    assert result["inventory_disposition"] == ["Use Existing"]


def test_lines_and_materials_without_plant_form_an_undetermined_row():
    """No plant is guessed: unresolved lines and materials are listed last with their reason."""
    data = {"lines": [line("ADDED", plants=(), incoming="NEW7", reason="BRAND_NOT_MAPPED", manufacturing_date="2026-05-19")],
            "materials": [{"number": "NEW7", "role": "INCOMING"}]}
    [undetermined] = build_plant_impact(data, [], {})["plants"]
    assert (undetermined["plant_code"], undetermined["status"], undetermined["status_label"]) == (None, "CHECK", "Plant not determined")
    assert undetermined["materials"][0]["reason_label"] == "The brand of its VC group has no plant in the Plant Code Mapping"


def test_materials_without_lines_fall_back_to_their_bom_locations():
    """An ECN object with no change line is placed through its own rule-derived locations."""
    objects = [{"number": "Q900VV0080301", "technical": True, "description": "VC group",
                "locations": [{"comparison": False, "plant_scope": {"vc_groups": ["Q900VV0080301"], "plants": [PLANT_B],
                                                                   "reason": None, "unmapped_brands": []}}]}]
    [plant_b] = build_plant_impact({"lines": [], "materials": []}, [], {}, objects)["plants"]
    assert (plant_b["plant_code"], plant_b["status"], plant_b["materials"][0]["role"]) == ("PK01", "CHECK", "OTHER")


def test_missing_dates_and_missing_mbom_need_a_plant_check():
    """A plant cannot be green just because a change line exists."""
    data = changes()
    for item in data["lines"]:
        item.update(manufacturing_date=None, end_date=None, qualifiers=["MBOM_NOT_UPDATED"])
    assert {plant["status"] for plant in build_plant_impact(data, [], {})["plants"]} == {"CHECK"}


def test_plant_summary_names_the_assembly_and_alignment_result():
    """Plant cards can state scope and BOM status without relying on their color."""
    data = {"lines": [
        line("ADDED", incoming="NEW1", manufacturing_date="2026-05-19", assembly="Q900VV0080402",
             assembly_description="M1V decal assembly", model_range="Q900V", levels=["EBOM", "MBOM"]),
        line("ADDED", incoming="NEW2", manufacturing_date=None, assembly="Q900VV0080301",
             assembly_description="M1V safety decal assembly", model_range="Q900V", levels=["EBOM"],
             qualifiers=["MBOM_NOT_UPDATED"]),
    ], "materials": []}
    [plant] = build_plant_impact(data, [], {})["plants"]
    assert plant["assemblies"] == [
        {"assembly": "Q900VV0080402", "assembly_description": "M1V decal assembly", "model_range": "Q900V",
         "change_count": 1, "effective_from": "2026-05-19", "effective_to": "2026-05-19",
         "bom_alignment": "ALIGNED", "bom_alignment_label": "EBOM and MBOM aligned"},
        {"assembly": "Q900VV0080301", "assembly_description": "M1V safety decal assembly", "model_range": "Q900V",
         "change_count": 1, "effective_from": None, "effective_to": None,
         "bom_alignment": "MBOM_PENDING", "bom_alignment_label": "Manufacturing BOM update pending"},
    ]


def test_paired_change_requires_dates_on_both_sides():
    """One known endpoint cannot make an incomplete replacement green."""
    for change in ("REPLACED", "QUANTITY_CHANGED", "VALIDITY_CHANGED"):
        for missing in ("manufacturing_date", "end_date"):
            item = line(change, manufacturing_date="2026-05-19", end_date="2026-05-18")
            item[missing] = None
            [plant] = build_plant_impact({"lines": [item], "materials": []}, [], {})["plants"]
            assert plant["status"] == "CHECK"


def test_a_purchase_info_record_in_any_plant_prices_a_new_part():
    """Priced new parts ignore the rule plants, old parts and comparison rows."""
    refs = [
        {"dataset_type": "PURCHASE_INFO_RECORDS", "material_number": "QKX7702260", "plant_code": "PK01"},
        {"dataset_type": "PURCHASE_INFO_RECORDS", "material_number": "QKX2833570", "plant_code": "PL10"},
        {"dataset_type": "PURCHASE_INFO_RECORDS", "material_number": "QKX0187890", "plant_code": "PL10", "comparison": True},
        {"dataset_type": "MATERIAL_MASTER", "material_number": "QKX0187890", "plant_code": "PL10"},
    ]
    assert build_plant_impact(changes(), refs, {})["priced_new_parts"] == ["QKX7702260"]


def test_material_master_procurement_type_is_exposed_on_the_affected_plant_material():
    """The UI can identify a make/buy change without exposing raw SAP code values."""
    refs = [{
        "dataset_type": "MATERIAL_MASTER", "material_number": "QKX7702260", "plant_code": "PL10",
        "procurement_type_label": "External procurement",
    }]

    [plant] = build_plant_impact({
        "lines": [line("ADDED", incoming="QKX7702260", manufacturing_date="2026-05-19")],
        "materials": [{"number": "QKX7702260", "role": "INCOMING"}],
    }, refs, {})["plants"]

    assert plant["materials"][0]["procurement_types"] == ["External procurement"]
