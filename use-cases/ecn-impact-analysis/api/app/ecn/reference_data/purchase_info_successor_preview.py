"""Create a reviewable Purchase Info Record successor workbook without publishing it."""

from __future__ import annotations

from io import BytesIO
from typing import Final

from openpyxl import Workbook

from app.ecn.reference_data.purchase_info_reconciliation import PurchaseInfoReconciliation

PURCHASE_INFO_SUCCESSOR_HEADERS: Final[tuple[str, ...]] = (
    "Info Rec.",
    "Vendor",
    "Name 1",
    "Material",
    "Plnt",
    "PGr",
)

_VALUE_BY_HEADER: Final[dict[str, str]] = {
    "Info Rec.": "info_record",
    "Vendor": "vendor",
    "Name 1": "supplier_name",
    "Material": "material_number",
    "Plnt": "plant_code",
    "PGr": "purchasing_group",
}

_STANDARD_ONLY_FIELDS: Final[tuple[str, ...]] = (
    "category",
    "tax_code",
    "fixed_firm_date",
)


def _non_empty(value: object) -> bool:
    """Return whether a source value would be lost in the legacy successor layout."""
    return value is not None and str(value).strip() != ""


def build_purchase_info_successor_workbook(
    reconciliation: PurchaseInfoReconciliation,
    *,
    baseline_dataset_id: str,
    price_source_sha256: str,
) -> bytes:
    """Return an XLSX successor preview plus a lineage manifest worksheet.

    The established Purchase Info Record parser supports a supplier-name layout
    containing the six columns in :data:`PURCHASE_INFO_SUCCESSOR_HEADERS`. This
    function refuses to generate a preview if an active baseline has populated
    standard-only fields that this layout cannot represent, preventing silent
    data loss. It never publishes or changes HANA data.
    """
    for row in reconciliation.rows:
        discarded = [
            field for field in _STANDARD_ONLY_FIELDS if _non_empty(row.values.get(field))
        ]
        if discarded:
            raise ValueError(
                "Cannot create a supplier-name-layout successor because source fields "
                f"would be lost: {', '.join(discarded)}."
            )

    workbook = Workbook()
    records = workbook.active
    records.title = "Purchase Info Records"
    records.append(PURCHASE_INFO_SUCCESSOR_HEADERS)
    for row in reconciliation.rows:
        records.append([row.values.get(_VALUE_BY_HEADER[header]) for header in PURCHASE_INFO_SUCCESSOR_HEADERS])
    records.freeze_panes = "A2"
    records.auto_filter.ref = f"A1:F{records.max_row}"

    manifest = workbook.create_sheet("Reconciliation Manifest")
    manifest.append(("Property", "Value"))
    manifest.append(("Preview status", "NOT PUBLISHED"))
    manifest.append(("Baseline dataset ID", baseline_dataset_id))
    manifest.append(("Price source SHA-256", price_source_sha256))
    manifest.append(("Baseline Material + Plant pairs", reconciliation.baseline_pair_count))
    manifest.append(("Targeted Material + Plant pairs added", reconciliation.targeted_pair_count))
    manifest.append(("Baseline rows carried forward", reconciliation.carried_forward_count))
    manifest.append(("Price-CSV rows added", reconciliation.targeted_price_csv_count))
    manifest.append(("Successor rows", len(reconciliation.rows)))
    manifest.append(("Uncovered targeted pairs", len(reconciliation.uncovered_target_pairs)))
    manifest.append(())
    manifest.append(("Candidate row", "Origin", "Source sheet", "Source row"))
    for row_number, row in enumerate(reconciliation.rows, start=2):
        manifest.append((row_number, row.origin, row.source_sheet, row.source_row_number))
    manifest.freeze_panes = "A2"

    output = BytesIO()
    workbook.save(output)
    return output.getvalue()
