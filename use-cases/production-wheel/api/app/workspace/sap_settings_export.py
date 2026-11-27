"""XLSX export of generated SAP settings, in the customer's upload-template layout.

The workbook has three parts:
1. The template sheets (``sap_settings_template.TEMPLATE_SHEETS``): same order,
   Excel tables, positions, notes and blue upload / grey information headers
   as the template. Planners upload the blue columns, as they do today.
2. ``Generation summary``: scenario, period, tab statuses and KPIs.
3. Audit sheets: every generated column, including the evidence the template
   does not show (cycle days and their source, calendar ranks, runner RoC).

A tab that is not ready gets a red status line directly above its tables;
a blocked tab keeps its table headers with one empty row, so it never looks
like a complete empty result.

The workbook is built only from HANA: the ``sap_*`` result views of the run
and the ``sap_settings`` summary in the run payload.
"""

from __future__ import annotations

import io
import json
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.styles.colors import Color
from openpyxl.utils.cell import coordinate_to_tuple, get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from .sap_settings_template import TEMPLATE_SHEETS, Source, TemplateSheet, TemplateTable, flag
from .wheel_export import write_sheet

GREY = PatternFill("solid", fgColor="FFA5A5A5")  # Template header fill of information-only columns.

# (sheet, summary tab, view, [(column header, field)]): every generated column.
AUDIT_SHEETS = [
    ("Audit - PV check", "pv_check", "sap_pv_conflicts",
     [("SEFI", "sefi"), ("Subgroup", "subgroup"), ("Selected PVs", "selected_pvs")]),
    ("Audit - RoC by subgroup", "kpis", "sap_changeover_kpis",
     [("Site", "plant"), ("Level", "level"), ("Filling line", "filling_line"), ("SEFI", "sefi"),
      ("Subgroup", "subgroup"), ("Cycle stock strategy", "cycle_stock_strategy"), ("RoC days", "coverage_days"),
      ("Base RoC days", "base_coverage_days"), ("FINIs", "fini_count"),
      ("Average batches per week", "batches_per_week"), ("Average changeovers per week", "changeovers_per_week"),
      ("Changeovers per week with single RoC", "single_changeovers_per_week"),
      ("Average pack-change changeovers per week", "pack_changeovers_per_week"),
      ("Average same-pack changeovers per week", "same_pack_changeovers_per_week"),
      ("Pack changes estimated by Volume per week", "volume_estimated_pack_changeovers_per_week"),
      ("Unclassified changeovers per week", "unclassified_changeovers_per_week"), ("Pack basis", "pack_basis"),
      ("Common lines", "common_lines")]),
    ("Audit - Cycle stock", "cycle_stock", "sap_dc_lot_sizes",
     [("Plant", "plant"), ("Material", "material"), ("Lot size", "lot_size_code"),
      ("Min lot size at DC", "min_lot_size_at_dc"), ("Rounding value", "rounding_value"),
      ("Defined in wheel", "defined_in_wheel"), ("Sales scenario", "sales_scenario"),
      ("Sales network scenario", "sales_network_scenario"), ("Pallet conversion identified", "min_lot_identified"),
      ("RoC selected PV", "coverage_days"), ("Pallets as per subgroup", "pallets_per_subgroup"),
      ("Pallet-adjusted days", "coverage_days_pallet_adjusted"), ("Cycle days", "cycle_days"),
      ("Days source", "cycle_source"), ("ABC at DC", "abc_class"), ("Status", "status")]),
    ("Audit - Planning calendar", "planning_calendar", "sap_planning_calendar",
     [("Action", "action"), ("Plant", "plant"), ("Material", "material"), ("SEFI", "sefi"),
      ("SEFI forecast (litres)", "sefi_forecast_litres"), ("SEFI order", "sefi_order"),
      ("Cumulative forecast share", "cumulative_share"), ("Lot size", "lot_size_procedure"),
      ("Lot size Profile", "lot_size_profile"), ("Planning calendar", "planning_calendar")]),
    ("Audit - APO lot size", "apo_lot_size", "sap_apo_lot_sizes",
     [("SEFI", "sefi"), ("Recipe combination", "recipe_combination"), ("Lot size KG", "lot_size_kg"),
      ("Comment", "comment"), ("Selected PV", "selected_pv"), ("Status", "status")]),
    ("Runner RoC audit", "runner_audit", "sap_runner_adjustments",
     [("SEFI", "sefi"), ("Subgroup", "subgroup"), ("Material", "material"),
      ("Cycle stock strategy", "cycle_stock_strategy"),
      ("RoC 1 full SEFI batch", "full_batch_coverage_days"), ("Ratio within SEFI", "sefi_ratio"),
      ("Runner class", "runner_class"), ("Ratio within subgroup", "subgroup_ratio"),
      ("Base RoC", "base_coverage_days"), ("Adjusted RoC", "adjusted_coverage_days"),
      ("Applied", "applied"), ("Runner multiple", "runner_multiple"),
      ("Optimizer runner multiple", "optimizer_runner_multiple"), ("Used RoC", "coverage_days"), ("Status", "status")]),
]


def _clean(value: Any) -> Any:
    """Show whole-number floats as integers (pieces, days); keep text such as '05' intact."""

    return int(value) if isinstance(value, float) and value.is_integer() else value


def _status_line(tab: dict[str, Any] | None) -> str | None:
    """Return the red status text of a tab that is not ready, else ``None``."""

    if not tab or tab["status"] == "ready":
        return None
    label = tab["status"].replace("_", " ").upper()
    return f"{label}: {tab['message']}" if tab.get("message") else label


def write_template_table(sheet, table: TemplateTable, source: Source) -> None:
    """Write one Excel table at its template anchor, with its status line above it."""

    top, left = coordinate_to_tuple(table.anchor)
    tab = source.summary["tabs"].get(table.tab) if table.tab else None
    status = _status_line(tab)
    if status:
        sheet.cell(top - 1, left, status).font = Font(bold=True, color="FFFF0000")
    produced = not tab or tab["status"] not in ("blocked", "not_applicable")
    rows = table.rows(source) if produced else []
    upload = len(table.columns) if table.upload is None else table.upload
    for offset, (header, _) in enumerate(table.columns):
        cell = sheet.cell(top, left + offset, header)
        if offset >= upload:
            cell.fill, cell.font = GREY, Font(bold=True)
    # An Excel table needs at least one data row; an empty table keeps one blank row, as in the template.
    for number, row in enumerate(rows or [{}], start=1):
        for offset, (_, name) in enumerate(table.columns):
            cell = sheet.cell(top + number, left + offset, _clean(row.get(name)))
            if isinstance(cell.value, str):
                cell.data_type = "s"  # Codes such as '05' or '=' text must never become numbers or formulas.
    bottom = f"{get_column_letter(left + len(table.columns) - 1)}{top + max(len(rows), 1)}"
    excel_table = Table(displayName=table.name, ref=f"{table.anchor}:{bottom}")
    excel_table.tableStyleInfo = TableStyleInfo(name=table.style, showRowStripes=True)
    sheet.add_table(excel_table)


def write_template_sheet(book, spec: TemplateSheet, source: Source) -> None:
    """Write one template sheet: tab colour, widths, notes and tables."""

    sheet = book.create_sheet(spec.name)
    if spec.tab_color:
        sheet.sheet_properties.tabColor = Color(**spec.tab_color)
    for letter, width in spec.widths.items():
        sheet.column_dimensions[letter].width = width
    for cell, text, font in spec.notes:
        sheet[cell] = text
        sheet[cell].font = Font(**font)
    for table in spec.tables:
        write_template_table(sheet, table, source)
    if spec.hidden:
        sheet.sheet_state = "hidden"


def _overview(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of the generation summary sheet."""

    point = int(summary["point_index"])
    return [
        {"Item": "Scenario", "Value": f"{summary.get('decision_label') or 'Point'} (point {point})"},
        {"Item": "Period", "Value": summary.get("period_label")},
        {"Item": "Generated at", "Value": summary.get("generated_at")},
        {"Item": "Wheel RoC (days)", "Value": summary.get("wheel_roc_days")},
        {"Item": "Changeovers per week (KPI)", "Value": summary.get("kpi_changeovers_per_week")},
        {"Item": "Changeovers per week (optimizer J_CH)", "Value": summary.get("point_j_ch")},
        {"Item": "KPI changeovers equal optimizer J_CH", "Value": summary.get("kpi_matches_point_j_ch")},
        {"Item": "Changeovers per week with single RoC (KPI)", "Value": summary.get("kpi_single_changeovers_per_week")},
        {"Item": "Subgroups with runner multiples", "Value": summary.get("runner_multiple_subgroups")},
        {"Item": "FINIs with runner-adjusted RoC", "Value": summary.get("runner_adjusted_fini")},
        {"Item": "Cycle stock policy", "Value": (
            f"{json.dumps(summary['cycle_stock_policy'], sort_keys=True)} ({summary.get('cycle_stock_policy_source')})"
            if summary.get("cycle_stock_policy") else "defaults (generated before the cycle stock strategy)")},
    ] + [
        {"Item": f"Tab: {name}", "Value": f"{tab['status']} - {tab['rows']} rows" + (f" - {tab['message']}" if tab["message"] else "")}
        for name, tab in summary["tabs"].items()
    ] + [{"Item": "Upload", "Value": "Upload the blue columns of each table, as with the customer template: "
                                     "the ECC tabs through the ECC upload, 'APO Lot size' to APO by hand. "
                                     "Grey columns and the audit sheets are information only."}]


def export_sap_settings(service, run_id: str) -> bytes:
    """Return the SAP settings workbook of a run's last generation.

    Raises:
        ValueError: SAP settings were never generated for the current selection.
    """

    summary = service.sap_settings(run_id)
    if not summary:
        raise ValueError("SAP settings are not available; select a scenario and generate them first")
    point = int(summary["point_index"])
    source = Source(summary, lambda view: service.repo.rows(run_id, view, {"point_index": point}))
    book = Workbook()
    book.remove(book.active)
    for spec in TEMPLATE_SHEETS:
        write_template_sheet(book, spec, source)
    write_sheet(book, "Generation summary", ["Item", "Value"], _overview(summary))
    for sheet_name, tab_name, view, columns in AUDIT_SHEETS:
        tab = summary["tabs"].get(tab_name, {"status": "not_applicable", "message": ""})
        if tab["status"] in ("blocked", "not_applicable"):
            write_sheet(book, sheet_name, ["Status", "Message"],
                        [{"Status": tab["status"], "Message": tab.get("message", "")}])
            continue
        write_sheet(book, sheet_name, [header for header, _ in columns],
                    [{header: _clean(flag(row.get(name)) if name == "min_lot_identified" else row.get(name))
                      for header, name in columns} for row in source.view(view)])
    book.active = 0
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()
