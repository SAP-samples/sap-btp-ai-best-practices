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
  point_index: ["Solution point", "Identifier of a solution on the trade-off frontier."],
  group_count: ["Number of groups", "Total groups in the solution."],
  singleton_group_count: ["Single-product groups", "Groups containing exactly one finished product."],
  group_size: ["Finished products in group", "Number of finished products belonging to the group."],
  member_count: ["Number of group members", "Count of finished products in the group."],
  members: ["Group member identifiers", "Finished-product identifiers belonging to this group."],
  frequency_per_week: ["Production runs per week", "Recurring modeled production frequency, not dated production orders."],
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
};

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

/** Format tri-state eligibility from HANA numeric or extractor boolean values without losing unknowns. */
export function fieldValue(field, value) {
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
