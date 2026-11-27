/**
 * Decision scenarios: the frontier knee ("Balanced") and its neighbours.
 *
 * The server labels points (GET /api/runs/{id}/decision-scenarios); this module
 * only turns that payload into chart datasets and a selectable table. By
 * default only the key points (knee plus three per side) are shown; "show all"
 * reveals every usable frontier point.
 */
import { el, display } from "./dom.js";

/** Business wording for each decision class returned by the API. */
export const CLASS_LABELS = {
  changeover_oriented: "Site / changeover-oriented",
  balanced: "Balanced",
  inventory_oriented: "Inventory-reduction-oriented",
};

const COLORS = {
  changeover_oriented: "#007d8a",
  balanced: "#e9730c",
  inventory_oriented: "#8b47d7",
  other: "#a9b4be",
};

/** Footnote for the changeover column: J_CH counts within-group changes only. */
export const CHANGEOVER_FOOTNOTE =
  "* Changeovers/week count recurring FINI changes within a group; initial and between-group setups are excluded.";

/** Return the labelled key points (knee plus neighbours) in changeover order. */
export function keyPoints(scenarios) {
  return (scenarios?.points || []).filter((point) => point.decision_label);
}

/** Return the points on display: key points by default, every usable point on request. */
export function visiblePoints(scenarios, showAll) {
  return showAll ? scenarios?.points || [] : keyPoints(scenarios);
}

/** Build one Chart.js scatter dataset per decision class, plus unlabelled points when shown. */
export function frontierDatasets(scenarios, showAll) {
  const kinds = [...Object.keys(CLASS_LABELS), ...(showAll ? ["other"] : [])];
  return kinds
    .map((kind) => ({
      label: CLASS_LABELS[kind] || "Other frontier points",
      data: visiblePoints(scenarios, showAll)
        .filter((point) => (point.decision_class || "other") === kind)
        .map((point) => ({
          x: Number(point.j_ch),
          y: Number(point.coverage_days),
          point_index: point.point_index,
          label: point.decision_label,
        })),
      backgroundColor: COLORS[kind],
      pointStyle: kind === "balanced" ? "rectRot" : "circle",
      pointRadius: kind === "balanced" ? 9 : 6,
      pointHoverRadius: 10,
    }))
    .filter((dataset) => dataset.data.length);
}

/** Map a clicked chart element to its frontier point; indexes are per dataset. */
export function pointIndexAt(datasets, element) {
  return datasets[element.datasetIndex]?.data[element.index]?.point_index;
}

/** Round for display while keeping missing values missing. */
function rounded(value, digits = 1) {
  return value == null || value === "" ? null : Number(Number(value).toFixed(digits));
}

/** Return decision-table rows with business labels and the stored selection flag. */
export function scenarioRows(scenarios, showAll) {
  const selected = scenarios?.selection?.point_index;
  return visiblePoints(scenarios, showAll).map((point) => ({
    point_index: point.point_index,
    label: point.decision_label || `Point ${point.point_index}`,
    orientation: CLASS_LABELS[point.decision_class] || "Other frontier point",
    decision_class: point.decision_class || "other",
    coverage_days: rounded(point.coverage_days),
    changeovers_per_week: rounded(point.j_ch),
    sefi_batches_per_week: rounded(point.sefi_batches_per_week),
    groups: point.group_count,
    // "single / multiple" group counts; only shown when a point uses runner multiples.
    cycle_stock: Number(point.runner_multiple_group_count) > 0
      ? `${point.single_roc_group_count ?? "?"} single · ${point.runner_multiple_group_count} multiple`
      : null,
    selected: selected != null && Number(selected) === Number(point.point_index),
  }));
}

/** Explain a frontier with fewer distinct trade-offs than requested; empty otherwise. */
export function frontierNote(scenarios) {
  if (!scenarios?.frontier_shortfall) return "";
  return `This scope has only ${scenarios.frontier_points_retained} distinct trade-offs, fewer than the requested minimum.`;
}

const COLUMNS = [
  ["label", "Scenario"],
  ["orientation", "Orientation"],
  ["coverage_days", "Inventory coverage · days"],
  ["changeovers_per_week", "Changeovers/week *"],
  ["sefi_batches_per_week", "SEFI batches/week"],
  ["groups", "Groups"],
];

/**
 * Render the decision table into `container`.
 * Clicking a row inspects that point's evidence; the Select button stores the
 * point as the run's decision for SAP settings generation.
 */
export function renderDecisionTable(container, scenarios, showAll, { onInspect, onSelect, fail }) {
  const rows = scenarioRows(scenarios, showAll);
  const columns = rows.some((row) => row.cycle_stock) ? [...COLUMNS, ["cycle_stock", "Cycle stock groups"]] : COLUMNS;
  const grid = el("table");
  const head = el("tr");
  for (const [, title] of [...columns, ["select", ""]]) head.append(el("th", title));
  const thead = el("thead");
  thead.append(head);
  const body = el("tbody");
  for (const row of rows) {
    const tr = el("tr", null, `decision-row decision-${row.decision_class}${row.selected ? " selected" : ""}`);
    tr.tabIndex = 0;
    for (const [key] of columns) tr.append(el("td", display(row[key])));
    const cell = el("td");
    const button = el("ui5-button", row.selected ? "Selected" : "Select");
    button.setAttribute("design", row.selected ? "Positive" : "Emphasized");
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      onSelect(row.point_index).catch(fail);
    });
    cell.append(button);
    tr.append(cell);
    tr.addEventListener("click", () => onInspect(row.point_index).catch(fail));
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter") onInspect(row.point_index).catch(fail);
    });
    body.append(tr);
  }
  grid.append(thead, body);
  container.replaceChildren(grid, el("p", CHANGEOVER_FOOTNOTE, "muted"));
}
