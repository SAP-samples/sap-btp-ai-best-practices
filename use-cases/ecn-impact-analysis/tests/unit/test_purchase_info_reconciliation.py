"""Unit tests for deterministic targeted Purchase Info reconciliation."""

from app.ecn.reference_data.purchase_info_reconciliation import (
    reconcile_targeted_purchase_info,
)


def test_reconciliation_preserves_all_baseline_rows_and_adds_only_targeted_price_rows() -> None:
    baseline = [
        {
            "info_record": "OLD-1",
            "vendor": "V1",
            "supplier_name": "Historic supplier",
            "material_number": "MAT-OLD",
            "plant_code": "PL10",
            "purchasing_group": "P1",
            "category": "A",
            "tax_code": "T",
            "fixed_firm_date": "2026-01-01",
        },
        {
            "info_record": "OLD-2",
            "vendor": "V2",
            "supplier_name": "Other supplier",
            "material_number": "MAT-EXISTING",
            "plant_code": "PK01",
            "purchasing_group": "P2",
        },
    ]
    prices = [
        {
            "info_record": "NEW-1",
            "vendor": "V3",
            "supplier_name": "New supplier",
            "material_number": " mat-new ",
            "plant_code": " pl10 ",
            "purchasing_group": "P3",
            "net_price": "10.00",
        },
        {
            "info_record": "NEW-1B",
            "vendor": "V3",
            "supplier_name": "New supplier",
            "material_number": "MAT-NEW",
            "plant_code": "PL10",
            "purchasing_group": "P3",
            "net_price": "12.00",
        },
        {
            "info_record": "UNRELATED",
            "vendor": "V4",
            "supplier_name": "Unrelated supplier",
            "material_number": "MAT-UNRELATED",
            "plant_code": "PL10",
            "purchasing_group": "P4",
            "net_price": "5.00",
        },
    ]

    result = reconcile_targeted_purchase_info(
        baseline,
        prices,
        {
            ("MAT-EXISTING", "PK01"),
            ("MAT-NEW", "PL10"),
            ("MAT-MISSING", "PL10"),
        },
    )

    assert result.carried_forward_count == 2
    assert result.targeted_price_csv_count == 2
    assert result.baseline_pair_count == 2
    assert result.targeted_pair_count == 1
    assert result.uncovered_target_pairs == (("mat-missing", "pl10"),)
    assert [row.origin for row in result.rows] == [
        "HANA_CARRIED_FORWARD",
        "HANA_CARRIED_FORWARD",
        "PRICE_CSV_TARGETED",
        "PRICE_CSV_TARGETED",
    ]
    assert result.rows[0].origin == "HANA_CARRIED_FORWARD"
    assert result.rows[0].values == baseline[0]
    assert result.rows[0].source_sheet is None
    assert result.rows[0].source_row_number is None
    assert result.rows[2].values == {
        "info_record": "NEW-1",
        "vendor": "V3",
        "supplier_name": "New supplier",
        "material_number": " mat-new ",
        "plant_code": " pl10 ",
        "purchasing_group": "P3",
        "category": None,
        "tax_code": None,
        "fixed_firm_date": None,
    }
