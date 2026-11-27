"""Layout of the customer's "Wheel settings for SAP" template, described as data.

The SAP settings XLSX reproduces the template sheet by sheet, so planners keep
their routine:
- the same sheet order;
- the same Excel table names and positions, and the same notes;
- blue headers for the columns uploaded to SAP, grey headers for information
  columns.

These sheets carry only the template columns. The generator's evidence columns
go to the audit sheets written by ``sap_settings_export``.

Where the template names a site (its central-DC sheet), a generic name is used.
The template's hidden planner-group sheet is not reproduced: the planner split
of recipe characters is not supported (see ``docs/sap-settings-generation.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from production_wheel.extraction.common import as_text

UPLOAD_STYLE = "TableStyleMedium16"  # Blue table style of the template's upload tables.
UPLOAD_NOTE = "Upload to SAP columns in blue ONLY"
BLUE_TAB = {"rgb": "FF0070C0"}
TITLE = {"bold": True, "size": 16}
RED = {"color": "FFFF0000"}


@dataclass
class Source:
    """Generated settings of the selected point, read from HANA once per view.

    Attributes:
        summary: The run's ``sap_settings`` payload (tab statuses, wheel RoC, period).
        read: Returns the rows of one ``sap_*`` view for the selected point.
    """

    summary: dict[str, Any]
    read: Callable[[str], list[dict[str, Any]]]
    _cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def view(self, name: str) -> list[dict[str, Any]]:
        """Return (and cache) the rows of one result view."""

        if name not in self._cache:
            self._cache[name] = self.read(name)
        return self._cache[name]

    @property
    def plant(self) -> str | None:
        """Producing plant, taken from the site-total KPI row."""

        return next((row.get("plant") for row in self.view("sap_changeover_kpis")
                     if row.get("level") == "site_total"), None)


@dataclass(frozen=True)
class TemplateTable:
    """One Excel table of the template.

    Attributes:
        name: Excel table name, identical to the template.
        anchor: Cell of the first header, e.g. ``"A6"``.
        columns: ``(header, field)`` pairs in template order.
        rows: Returns the table rows (dicts keyed by field) from a ``Source``.
        upload: Number of leading columns uploaded to SAP; later columns get a grey header.
        tab: Summary tab whose status applies; ``None`` for parameter tables.
        style: Excel table style.
    """

    name: str
    anchor: str
    columns: tuple[tuple[str, str], ...]
    rows: Callable[[Source], list[dict[str, Any]]]
    upload: int | None = None
    tab: str | None = None
    style: str = UPLOAD_STYLE


@dataclass(frozen=True)
class TemplateSheet:
    """One sheet of the template.

    Attributes:
        name: Sheet name.
        tables: Excel tables on the sheet.
        notes: ``(cell, text, font options)`` written above the tables.
        widths: Column widths by letter.
        tab_color: openpyxl ``Color`` arguments for the sheet tab.
        hidden: Whether the sheet is hidden, as in the template.
    """

    name: str
    tables: tuple[TemplateTable, ...]
    notes: tuple[tuple[str, str, dict[str, Any]], ...] = ()
    widths: dict[str, float] = field(default_factory=dict)
    tab_color: dict[str, Any] | None = None
    hidden: bool = False


def from_view(view: str, order: tuple[str, ...], keep: Callable[[dict], bool] | None = None):
    """Return a row source reading one view, filtered, in a stable order."""

    def rows(source: Source) -> list[dict[str, Any]]:
        kept = [row for row in source.view(view) if keep is None or keep(row)]
        return sorted(kept, key=lambda row: tuple(as_text(row.get(name)) for name in order))

    return rows


def flag(value: Any) -> bool:
    """Read a boolean persisted as text in HANA."""

    return value is True or str(value).lower() == "true"


_LEVELS = {"site_total": "Site total", "filling_line": "Filling line"}
_WEEKLY = (
    "batches_per_week",
    "changeovers_per_week",
    "pack_changeovers_per_week",
    "same_pack_changeovers_per_week",
    "volume_estimated_pack_changeovers_per_week",
    "unclassified_changeovers_per_week",
)


def changeover_estimation(source: Source) -> list[dict[str, Any]]:
    """Site total first, then filling lines; weekly values rounded to 1 decimal.

    Python ``round`` rounds half to even, like Power Query ``Number.Round``.
    Subgroup rows stay on the audit sheet.
    """

    rows = [row for row in source.view("sap_changeover_kpis") if row.get("level") in _LEVELS]
    rows.sort(key=lambda row: (row["level"] != "site_total", as_text(row.get("filling_line"))))
    return [{**row, "level": _LEVELS[row["level"]],
             **{name: round(float(row.get(name) or 0), 1) for name in _WEEKLY}} for row in rows]


_DC_COLUMNS = (
    ("Plant", "plant"), ("Material", "material"), ("Lot size", "lot_size_code"),
    ("Min lot size at DC", "min_lot_size_at_dc"), ("Rounding value", "rounding_value"),
    ("Defined in wheel", "defined_in_wheel"), ("Sales scenario", "sales_scenario"),
    ("Sales network scenario", "sales_network_scenario"),
)
_DC_WIDTHS = {"A": 7.9, "B": 10.9, "C": 9.9, "D": 18.7, "E": 17.1, "F": 21.9, "G": 19.9, "H": 40.9}

TEMPLATE_SHEETS: tuple[TemplateSheet, ...] = (
    TemplateSheet(
        "Check - Several Lot size in Sub",
        notes=(("A3", "To fix in FINI grouping by selecting same PV in the subgroup", {}),),
        widths={"A": 6.9, "B": 11.7},
        tables=(TemplateTable("Check___Several_Lot_size_in_Subgroup", "A5",
                              (("SEFI", "sefi"), ("Subgroup", "subgroup")),
                              from_view("sap_pv_conflicts", ("sefi", "subgroup")), tab="pv_check"),),
    ),
    TemplateSheet(
        "RoC and Change overs",
        tab_color={"theme": 7, "tint": 0.5999938962981048},
        widths={"A": 19.3, "B": 9.1, "C": 12.6, "D": 27.0, "E": 31.4, "F": 43.3, "G": 41.7, "H": 44.0, "I": 35.1},
        tables=(
            TemplateTable("Wheel_RoC", "A1", (("Wheel RoC", "wheel_roc"),),
                          lambda source: [{"wheel_roc": source.summary.get("wheel_roc_days")}]),
            TemplateTable("Site", "C1", (("Site", "site"),),
                          lambda source: [{"site": source.plant}], style="TableStyleLight9"),
            # ponytail: Scope is left empty; the app keeps no wheel scope, and only the unsupported planner split reads it.
            TemplateTable("Scope", "D1", (("Scope", "scope"),), lambda source: [], style="TableStyleLight9"),
            TemplateTable("Change_over_estimation", "A5", (
                ("Site", "plant"), ("Level", "level"), ("Filling line", "filling_line"),
                ("Average batches per week", "batches_per_week"),
                ("Average changeovers per week", "changeovers_per_week"),
                ("Average pack-change changeovers per week", "pack_changeovers_per_week"),
                ("Average same-pack changeovers per week", "same_pack_changeovers_per_week"),
                ("Pack changes estimated by Volume per week", "volume_estimated_pack_changeovers_per_week"),
                ("Unclassified changeovers per week", "unclassified_changeovers_per_week"),
            ), changeover_estimation, tab="kpis"),
        ),
    ),
    TemplateSheet(
        "Cycle stock",
        tab_color=BLUE_TAB,
        notes=(
            ("A1", "Note: For those cases where pallets as per subgroups are less than 0,9 pallets, "
                   "the Lot size uses correspond to the coverage of 1 full pallet", {}),
            ("A3", "With pallet conversion identified", TITLE), ("J3", "Without pallet conversion identified", TITLE),
            ("A4", UPLOAD_NOTE, {}), ("J4", UPLOAD_NOTE, {}),
        ),
        widths={**_DC_WIDTHS, "I": 10.0, "J": 7.9, "K": 10.9, "L": 9.9, "M": 21.9, "N": 19.9, "O": 29.6},
        tables=(
            TemplateTable("DC_Lot_size_with_pallet", "A6", _DC_COLUMNS,
                          from_view("sap_dc_lot_sizes", ("plant", "material"),
                                    lambda row: flag(row.get("min_lot_identified"))),
                          upload=5, tab="cycle_stock"),
            TemplateTable("DC_Lot_size_without_pallet", "J6", _DC_COLUMNS[:3] + _DC_COLUMNS[5:],
                          from_view("sap_dc_lot_sizes", ("plant", "material"),
                                    lambda row: not flag(row.get("min_lot_identified"))),
                          upload=3, tab="cycle_stock"),
        ),
    ),
    TemplateSheet(
        "FINI Planning calendar",
        tab_color=BLUE_TAB,
        notes=(("H2", "Pre-existing planning calendar, not relevant for 02W, have to be reset for materials "
                      "not in wheel", RED),),
        widths={"A": 10.9, "B": 7.9, "C": 9.9, "D": 19.1, "E": 16.4, "H": 7.9, "I": 10.9, "J": 9.9, "K": 26.6},
        tables=(
            TemplateTable("FINI_Planning_calendar_at_P_site", "A4", (
                ("Material", "material"), ("Plant", "plant"), ("Lot size", "lot_size_procedure"),
                ("Planning calendar", "planning_calendar"), ("Lot size Profile", "lot_size_profile"),
            ), from_view("sap_planning_calendar", ("material",), lambda row: row.get("action") == "assign"),
                tab="planning_calendar"),
            TemplateTable("FINI_Planning_calendar_at_P_site_RESET", "H4", (
                ("Plant", "plant"), ("Material", "material"), ("Lot size", "lot_size_procedure"),
                ("Lot size Profile", "lot_size_profile"), ("Planning calendar", "planning_calendar"),
            ), from_view("sap_planning_calendar", ("plant", "material"), lambda row: row.get("action") == "reset"),
                tab="planning_calendar"),
        ),
    ),
    TemplateSheet(
        "Recipe combination",
        tab_color=BLUE_TAB,
        notes=(("A2", UPLOAD_NOTE, {}),),
        widths={"A": 7.9, "B": 10.9, "C": 21.1, "D": 8.7, "E": 14.6},
        tables=(TemplateTable("Recipe_combination", "A4", (
            ("Plant", "plant"), ("Material", "material"), ("Recipe combination", "recipe_combination"),
            ("SEFI", "sefi"), ("Subgroup", "subgroup"),
        ), from_view("sap_recipe_combinations", ("plant", "material")), upload=3, tab="recipe_combination"),),
    ),
    TemplateSheet(
        "APO Lot size",
        tab_color={"theme": 4, "tint": 0.3999755851924192},
        notes=(
            ("A2", "In general, before the upload, delete the entries that refers to production wheel in comments", {}),
            ("A3", "In case upload fails for certain SEFIs and recipe combination, delete those entries in APO "
                   "and then upload again.", {}),
        ),
        widths={"A": 8.0, "B": 21.1, "C": 12.9, "D": 22.3},
        tables=(TemplateTable("APO_lot_size", "A5", (
            ("SEFI", "sefi"), ("Recipe combination", "recipe_combination"), ("Lot size KG", "lot_size_kg"),
            ("Comment", "comment"),
        ), from_view("sap_apo_lot_sizes", ("sefi", "recipe_combination")), tab="apo_lot_size",
            style="TableStyleMedium20"),),
    ),
    TemplateSheet(
        "Central DC lot size",
        tab_color={"theme": 2, "tint": -0.749992370372631},
        notes=(("A2", "Only applicable for sites with a central DC", TITLE), ("A3", UPLOAD_NOTE, {})),
        widths=_DC_WIDTHS,
        tables=(TemplateTable("Central_DC_Lot_size", "A5", _DC_COLUMNS, lambda source: [], upload=5,
                              tab="central_dc_lot_size"),),
    ),
    TemplateSheet(
        "Comment for APO",
        hidden=True,
        tab_color={"theme": 5},
        widths={"A": 9.4},
        tables=(TemplateTable("Comment4APO", "A1", (("Quarter", "quarter"),),
                              lambda source: [{"quarter": source.summary.get("period_label")}],
                              style="TableStyleMedium2"),),
    ),
)
