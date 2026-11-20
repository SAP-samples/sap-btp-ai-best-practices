"""Tests for controlled Material + Plant price-overlay coverage classification."""

from app.ecn.reference_data.purchase_info_price_coverage import (
    assess_price_overlay_coverage,
)


def test_price_overlay_coverage_does_not_choose_among_ambiguous_source_prices() -> None:
    """Surface safe display states without treating a source price as a decision."""
    coverage = assess_price_overlay_coverage(
        [
            {"material_number": "MAT-1", "plant_code": "PK01", "net_price": "12.50"},
            {"material_number": "MAT-2", "plant_code": "PK01", "net_price": "0"},
            {"material_number": "MAT-3", "plant_code": "PL10", "net_price": "7.25"},
            {"material_number": "MAT-3", "plant_code": "PL10", "net_price": "7.50"},
        ],
        [
            ("MAT-1", "PK01"),
            ("MAT-2", "PK01"),
            ("MAT-3", "PL10"),
            ("MAT-4", "PL10"),
        ],
    )

    assert [(entry.material_number, entry.plant_code, entry.status) for entry in coverage.entries] == [
        ("MAT-1", "PK01", "SOURCE_PRICE_AVAILABLE"),
        ("MAT-2", "PK01", "ZERO_PRICE_REVIEW"),
        ("MAT-3", "PL10", "MULTIPLE_SOURCE_PRICES_REVIEW"),
        ("MAT-4", "PL10", "NOT_SUPPLIED"),
    ]
    assert coverage.count("SOURCE_PRICE_AVAILABLE") == 1
    assert coverage.count("ZERO_PRICE_REVIEW") == 1
    assert coverage.count("MULTIPLE_SOURCE_PRICES_REVIEW") == 1
    assert coverage.count("NOT_SUPPLIED") == 1
