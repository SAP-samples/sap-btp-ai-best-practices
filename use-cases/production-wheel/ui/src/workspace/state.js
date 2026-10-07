/** Whitelist only navigation identifiers for browser persistence; never data rows. */
export const PERSISTED_KEYS = [
  "plant_profile_id",
  "dataset_id",
  "draft_id",
  "run_id",
  "point_index",
];

/** Restore a sanitized selection from storage; malformed state becomes empty. */
export function restoreSelection(storage) {
  try {
    const saved = JSON.parse(
      storage.getItem("production-wheel.selection") || "{}",
    );
    return Object.fromEntries(
      PERSISTED_KEYS.filter((key) =>
        ["string", "number"].includes(typeof saved[key]),
      ).map((key) => [key, saved[key]]),
    );
  } catch {
    return {};
  }
}

/** Save identifier selections, deliberately excluding request bodies and tables. */
export function persistSelection(storage, state) {
  storage.setItem(
    "production-wheel.selection",
    JSON.stringify(
      Object.fromEntries(
        PERSISTED_KEYS.filter((key) => state[key] != null).map((key) => [
          key,
          state[key],
        ]),
      ),
    ),
  );
}

/**
 * Return the plant codes a dataset covers, as strings.
 * Input: a dataset record from /api/datasets (plant lives in metadata.plant today;
 * list-valued and top-level variants are accepted for older/newer records).
 * Output: array of non-empty plant strings, possibly empty.
 */
export function datasetPlants(dataset) {
  const value = dataset?.plants || dataset?.metadata?.plants || dataset?.plant || dataset?.metadata?.plant;
  return [value].flat().filter((plant) => plant != null && plant !== "").map(String);
}

/** Recognize states that no longer require polling. */
export function isTerminal(status = "") {
  return /^(completed|succeeded|failed|cancelled|canceled|error|worker_lost)$/i.test(
    status,
  );
}

/** Copy a server draft while updating explicit UI values and preserving other settings. */
export function mergeDraft(draft, fields) {
  const request = structuredClone(draft.request || {});
  request.config = {
    ...request.config,
    pv_mode: fields.pv,
    coverage_mode: fields.coverage,
    coverage_basis:
      fields.basis || request.config?.coverage_basis || "BASE_GROUP",
    matrix_mode: draft.plant_profile_id ? request.config?.matrix_mode : fields.matrix,
  };
  const cap = Number(fields.cap);
  if (!draft.plant_profile_id && (!Number.isInteger(cap) || cap < 7))
    throw new Error(
      "Global group cap must be an integer of at least 7. Use per-block overrides for other caps.",
    );
  if (!draft.plant_profile_id) request.config.group_size = {
    base_limit: 7,
    mode: cap === 7 ? "HARD" : "BOUNDED_RELAXATION",
    max_excess: cap - 7,
  };
  request.scope = JSON.parse(fields.scope);
  request.constraints = JSON.parse(fields.constraints);
  return {
    revision: draft.revision,
    ...(fields.title !== undefined ? { title: fields.title.trim() } : {}),
    request,
    budget: JSON.parse(fields.budget),
  };
}
