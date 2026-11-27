/** Shared display labels and definitions. Keys remain the original API identifiers. */
const fields = {
  filling_line: ["Line being evaluated", "Candidate filling line for this row. Read with Allowed on this line?; this is not a scheduled assignment."],
  eligible: ["Allowed on this line?", "Yes (1): allowed. No (0): not allowed. Unknown: source eligibility is missing. Search uses the stored values 1 or 0."],
  eligible_lines: ["Allowed filling lines", "All filling-line alternatives supported by the effective source evidence."],
  common_lines: ["Lines allowed for every group member", "Intersection of members' allowed lines. These are alternatives, not scheduled assignments."],
  line_data_status: ["Line evidence availability", "Known means the source specifies eligibility; it does not mean this particular line is allowed. Missing means unknown."],
  source_pattern: ["Original line pattern", "Original source text. For example, - - - - 5 - allows line 5. A source correction, when present, can override this pattern."],
  filling_line_raw: ["Original line pattern", "Filling-line text as read from the source workbook, before any line corrections."],
  material: ["Finished product (FINI)", "Identifier of the finished product."],
  sefi: ["Semi-finished product (SEFI)", "Identifier of the semi-finished product associated with the finished product or group."],
  pck_code: ["Packaging code (PCK)", "Source packaging code. Matching codes alone do not establish group feasibility."],
  pck_codes: ["Packaging codes (PCK)", "Packaging codes represented in this group or population."],
  plant: ["Production site", "Plant identifier that scopes materials and production data."],
  source_sheet: ["Source worksheet", "Worksheet containing the original evidence."],
  source_row: ["Source Excel row", "Original worksheet row number, not the row number on this page."],
  source_range: ["Source cell range", "Original workbook cell range containing the evidence."],
  production_version: ["Production version (PV)", "Source production-version option associated with a lot size."],
  selected_pv: ["Selected production version", "Production-version option selected for the proposed group."],
  fixed_pv: ["Baseline production version", "Production-version option recorded for the baseline grouping."],
  model_status: ["Optimization inclusion status", "Whether the source record is modeled, excluded, or outside the configured scope."],
  optimized_pv_model_status: ["Inclusion with selectable production versions", "Admission status when production-version alternatives are considered."],
  exclusion_reason: ["Reason for exclusion", "Source or validation reason preventing inclusion."],
  forecast_litres_12m: ["12-month demand forecast (L)", "Forecast volume over the twelve-month source horizon."],
  avg_daily_demand: ["Average daily demand (L/day)", "Forecast volume divided by the configured demand days."],
  package_volume: ["Volume per package (L)", "Volume of one package, in litres."],
  lot_size_litres: ["Production-version lot size (L)", "Lot volume associated with a source production version."],
  nominal_lot_litres: ["Nominal lot volume (L)", "Lot volume before the configured batch factor is applied."],
  effective_batch_litres: ["Effective batch volume (L)", "Batch volume after the configured factor is applied."],
  canonical_factor: ["Batch calculation factor", "Multiplier applied to nominal lot volume in the application's calculations."],
  pallet_litres_resolved: ["Resolved volume per pallet (L)", "Pallet volume selected through source and enrichment resolution."],
  resolved_pallet_litres: ["Resolved volume per pallet (L)", "Pallet volume selected through source and enrichment resolution."],
  resolution_status: ["Source resolution status", "Outcome of resolving source evidence, including missing or conflicting values."],
  resolution_source: ["Chosen evidence source", "Evidence used to resolve the value."],
  coverage_days: ["Demand coverage (days)", "Calculated demand coverage under the active coverage basis; not a dated production schedule."],
  demand_weighted_mean_coverage_days: ["Demand-weighted average coverage (days)", "Average coverage weighted by demand, rather than giving each group equal weight."],
  worst_fini_coverage_days: ["Highest finished-product coverage (days)", "Largest finished-product coverage within the group."],
  active_coverage_basis: ["Coverage calculation basis", "Selected method used to calculate coverage."],
  j_ch: ["Changeover proxy", "Within-group changeover measure. Excludes singleton, between-group, sequence and cleaning setup costs; not total changeover time."],
  j_ch_contribution: ["Group changeover proxy contribution", "This group's contribution to the within-group changeover proxy. Repeated member-row values must not be summed."],
  cycle_stock_mode: ["Cycle stock strategy setting", "Plant profile setting of the run: Single RoC, Runner multiples, or Optimizer chooses per group."],
  cycle_stock_strategy: ["Cycle stock strategy", "Single RoC: every product of the group is filled in every batch and gets the group's coverage. Runner multiple: only the fastest product keeps it; slower products are filled every 2nd, 3rd... batch and get a whole multiple of it."],
  strategy_coverage_days: ["Coverage under the strategy (days)", "Group row: coverage of the group under its cycle stock strategy and the run's coverage basis. Product row: wheel RoC x the product's runner multiple, without pallets."],
  base_group_coverage_days: ["Batch coverage before rounding (days)", "Demand days x effective batch / group demand. The wheel RoC rounds it to whole days."],
  wheel_roc_days: ["Wheel RoC (days)", "Batch coverage rounded to whole days like the workbook (at least 1 day). Batches per week, changeovers and runner multiples use it; pallets never change it."],
  wheel_coverage_days: ["Wheel coverage without pallets (days)", "Demand-weighted mean of the products' wheel RoC x runner multiple."],
  fini_wheel_roc_days: ["Product wheel RoC (days)", "Wheel RoC x the product's runner multiple. Always a whole multiple of the group's wheel RoC."],
  pallets_per_batch: ["Pallets per batch", "Pallets one batch fills for the product: nominal lot x demand share / litres per pallet, rounded to 1 decimal."],
  fini_pallets_per_batch: ["Pallets per batch by product", "Pallets one batch fills for each product of the group."],
  fini_inventory_roc_days: ["Product inventory RoC with pallets (days)", "Product wheel RoC raised to cover at least one full pallet (below 0.9 pallets per batch: floor(RoC / pallets)). Used for inventory and DC lot sizes only."],
  fini_adjusted_coverage_days: ["Product inventory RoC with pallets (days)", "Product coverage after the one-pallet adjustment. Used for inventory only, not for batches or changeovers."],
  adjusted_group_coverage_days: ["Inventory coverage with pallets (days)", "Demand-weighted mean of the products' inventory RoC with pallets."],
  runner_class: ["Runner class", "High, medium or low runner from the plant profile's runner cut-offs."],
  runner_multiple: ["Runner multiple", "Batches one fill of this product covers: 1 = filled every batch, 2 = every 2nd batch. Always 1 with Single RoC."],
  runner_multiples: ["Runner multiples", "Runner multiple of each product of the group."],
  max_runner_multiple: ["Largest runner multiple", "Largest runner multiple in the group; 1 means multiples change nothing."],
  single_roc_group_count: ["Single RoC groups", "Groups where every product gets the group's coverage."],
  runner_multiple_group_count: ["Runner multiple groups", "Groups where slower products are filled only every few batches."],
  point_index: ["Solution point", "Identifier of a solution on the trade-off frontier."],
  group_count: ["Number of groups", "Total groups in the solution."],
  singleton_group_count: ["Single-product groups", "Groups containing exactly one finished product."],
  group_size: ["Finished products in group", "Number of finished products belonging to the group."],
  member_count: ["Number of group members", "Count of finished products in the group."],
  members: ["Group member identifiers", "Finished-product identifiers belonging to this group."],
  frequency_per_week: ["Production runs per week", "Days per week / wheel RoC (whole days). Recurring modeled frequency, not dated production orders."],
  validation_status: ["Structural validation status", "Checks of solution structure and constraints. This does not establish business acceptance or global optimality."],
  acceptance_status: ["Business acceptance status", "Assessment against the configured business acceptance criteria."],
  result_class: ["Solution evidence classification", "Classification of the result according to its validation and optimization evidence."],
  candidate_pool_completeness: ["Candidate search coverage", "Whether candidate generation is complete within its stated scope or restricted."],
  proof_scope: ["Scope of optimality evidence", "Domain covered by solver evidence. A restricted candidate pool does not prove full-space optimality."],
  relative_gap: ["Relative optimality gap", "Relative difference between the objective and solver bound within the stated proof scope."],
  rule_id: ["Validation rule", "Identifier of the check that produced this finding."],
  entity_key: ["Affected record identifier", "Identifier of the record or group associated with this finding."],
};

const views = {
  fini_master: "Finished products and source data",
  fini_line_eligibility: "Finished-product filling-line permissions",
  pck_line_eligibility: "Packaging filling-line permissions",
  line_corrections: "Source filling-line corrections",
  pallet_resolution: "Pallet volume evidence",
  production_versions: "Production versions and lot sizes",
  validation_issues: "Extraction quality findings",
  groups: "Proposed production groups",
  members: "Finished-product assignments",
  solutions: "Solution trade-offs",
  validation: "Solution validation findings",
  block_audits: "Optimization evidence by block",
  global_audits: "Whole-solution optimization evidence",
  matrix_pairs: "Package compatibility evidence",
  sap_dc_lot_sizes: "SAP cycle stock per DC",
  sap_planning_calendar: "SAP planning calendar",
  sap_recipe_combinations: "SAP recipe combinations",
  sap_apo_lot_sizes: "SAP APO lot sizes",
  sap_changeover_kpis: "SAP RoC and changeover KPI",
  sap_pv_conflicts: "SAP production-version check",
  sap_runner_adjustments: "SAP runner-adjusted coverage audit",
};

/**
 * Readable names for compatibility modes. The API, solver and stored records keep the
 * original values (HARD, FLEXIBLE, DIAGNOSTIC, OFF); only people see these labels.
 * Listed in the order shown in selects, starting with the profile default.
 */
export const MATRIX_MODE_LABELS = {
  FLEXIBLE: "Avoid allowed",
  DIAGNOSTIC: "Prefer compatible",
  HARD: "Strict",
  OFF: "Ignore table",
};

/** Return the readable compatibility-mode name, or the raw value when it is unknown. */
export function matrixModeLabel(mode) {
  return MATRIX_MODE_LABELS[mode] || String(mode ?? "—");
}

/** Return <option> markup for a compatibility-mode select; values stay API identifiers. */
export function matrixModeOptions() {
  return Object.entries(MATRIX_MODE_LABELS)
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");
}

/**
 * Readable names for a plant profile's volume matching (production_wheel.volume_classes).
 * exact: the matrix lists every package volume. nearest_class: the matrix lists
 * reference volumes (stops) plus optional "<n" / ">n" classes.
 */
export const VOLUME_MATCHING_LABELS = {
  nearest_class: "Nearest reference volume",
  exact: "Exact volumes",
};

/** Return the readable volume-matching name; stored profiles without the setting are exact. */
export function volumeMatchingLabel(mode) {
  return VOLUME_MATCHING_LABELS[mode || "exact"] || String(mode);
}

/** Return <option> markup for the volume-matching select. */
export function volumeMatchingOptions() {
  return Object.entries(VOLUME_MATCHING_LABELS)
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");
}

/**
 * Sort key of a matrix axis label: "<1" before 1, ">15" after 15.
 * @param {string|number} label Volume in litres or a "<n" / ">n" class.
 * @returns {number[]} [value, kind] with kind -1 below, 0 stop, 1 above.
 */
export function volumeLabelKey(label) {
  const text = String(label).replaceAll(" ", "");
  if (text.startsWith("<")) return [Number(text.slice(1)), -1];
  if (text.startsWith(">")) return [Number(text.slice(1)), 1];
  return [Number(text), 0];
}

/** Compare two matrix axis labels for Array.sort. */
export function compareVolumeLabels(a, b) {
  const [left, leftKind] = volumeLabelKey(a), [right, rightKind] = volumeLabelKey(b);
  return left - right || leftKind - rightKind;
}

/** Return a readable label for an API field, preserving unknown terms with expanded acronyms. */
export function fieldLabel(value = "") {
  if (fields[value]) return fields[value][0];
  const text = String(value).replace(/j_ch/gi, "changeover proxy").replaceAll("_", " ")
    .replace(/\b(fini|sefi|pck|pv|sap|mrp|dc|id)\b/gi, (term) => term.toUpperCase());
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Return the verified explanation for a field, or an empty string for undocumented fields. */
export function fieldDescription(field) { return fields[field]?.[1] || ""; }

/** Return a readable view name while retaining its API identifier as the select value. */
export function viewLabel(view) { return views[view] || fieldLabel(view); }

/** Readable names of a group's cycle stock strategy; stored values stay API identifiers. */
export const CYCLE_STOCK_STRATEGY_LABELS = { single: "Single RoC", runner_multiple: "Runner multiple" };

/** Format tri-state eligibility and cycle stock strategies without losing unknowns. */
export function fieldValue(field, value) {
  if (field === "cycle_stock_strategy" && CYCLE_STOCK_STRATEGY_LABELS[value]) return CYCLE_STOCK_STRATEGY_LABELS[value];
  if (field === "eligible") {
    if (value == null || value === "") return "Unknown";
    if ([true, 1, "1"].includes(value)) return "Yes";
    if ([false, 0, "0"].includes(value)) return "No";
  }
  return value == null ? "—" : typeof value === "object" ? JSON.stringify(value) : String(value);
}

/** Explain the record grain and source notation for the selected table. */
export function viewDescription(view) {
  if (["fini_line_eligibility", "pck_line_eligibility"].includes(view))
    return "One row per product or packaging code and candidate filling line. Read the line together with Allowed on this line?. For example, - - - - 5 - allows only line 5; - 2 3 - - - allows lines 2 and 3. Search by finished product to see its alternatives. These are permissions, not scheduled assignments.";
  if (view === "members") return "One row per finished product and solution point, including unassigned products. Group metrics repeat across members; do not sum them as group totals.";
  return "Search applies to this view. The page shows only part of the full result. Open the glossary for column meanings.";
}
