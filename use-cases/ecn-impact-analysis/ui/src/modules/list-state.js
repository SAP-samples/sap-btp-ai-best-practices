export const LIST_SORTS = new Set([
  "uploaded_at", "resolution_date", "affected_plants", "ecn_number", "description", "affected_item_count",
  "ebom_match_count", "mbom_match_count", "warning_count", "analysis_state",
]);
export const JOB_STATES = new Set([
  "READY", "QUEUED", "BLOCKED", "RUNNING", "COMPLETED", "COMPLETED_WITH_GAPS", "FAILED",
]);
export const ANALYSIS_FILTERS = new Set([...JOB_STATES, "SUMMARIES_AVAILABLE", "IN_PROGRESS"]);
export const DEFAULT_LIST_STATE = Object.freeze({
  query: "", ecn_number: "", plant: "", analysis_state: "", has_warnings: "",
  sort: "uploaded_at", order: "desc", page: 1, page_size: 25,
});

/** Parse a bounded integer or return the supplied safe fallback. */
function boundedInteger(value, fallback, allowed) {
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && allowed(parsed) ? parsed : fallback;
}

/** Parse only values accepted by GET /api/ecns. */
export function parseListState(input) {
  const params = input instanceof URLSearchParams ? input : new URLSearchParams(input || "");
  const analysisState = params.get("analysis_state") || "";
  const warnings = params.get("has_warnings") || "";
  const sort = params.get("sort") || DEFAULT_LIST_STATE.sort;
  const order = params.get("order") || DEFAULT_LIST_STATE.order;
  return {
    query: params.get("query") || "",
    ecn_number: params.get("ecn_number") || "",
    plant: params.get("plant") || "",
    analysis_state: ANALYSIS_FILTERS.has(analysisState) ? analysisState : "",
    has_warnings: ["true", "false"].includes(warnings) ? warnings : "",
    sort: LIST_SORTS.has(sort) ? sort : DEFAULT_LIST_STATE.sort,
    order: ["asc", "desc"].includes(order) ? order : DEFAULT_LIST_STATE.order,
    page: boundedInteger(params.get("page"), 1, (candidate) => candidate >= 1),
    page_size: boundedInteger(
      params.get("page_size"),
      DEFAULT_LIST_STATE.page_size,
      (candidate) => candidate >= 1 && candidate <= 100,
    ),
  };
}

/** Serialize non-empty list state using API snake-case names. */
export function serializeListState(state) {
  const params = new URLSearchParams();
  for (const key of Object.keys(DEFAULT_LIST_STATE)) {
    const value = state[key];
    if (value !== "" && value !== null && value !== undefined) params.set(key, String(value));
  }
  return params;
}

/** Build a row-identity detail route while preserving the originating list state. */
export function detailHref(ecnRowId, state) {
  return `/ecns/${encodeURIComponent(ecnRowId)}?${serializeListState(state).toString()}`;
}
