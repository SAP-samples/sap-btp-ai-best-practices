/**
 * Plain-language extraction findings for the dataset review.
 *
 * The import stores its quality findings on the dataset record (`dataset.issues`):
 * one object per finding with `rule_id`, `severity`, `entity_key` (one key, or
 * several joined with "|"), optional `reason`, `observed`, `expected` and
 * `action`. Planners should not read that JSON, so findings are grouped by check
 * (and by reason, e.g. why a FINI was excluded) and described in plain words.
 *
 * Two kinds of entries are produced:
 * - Findings (`findingGroups`): data problems found in the workbook, as warnings
 *   or errors. Errors block publishing.
 * - Scope notes (`scopeNotes`): products left out because of the extraction
 *   settings the user chose (frequency codes) or because their frequency has no
 *   weekly code. The user already saw these in the extraction settings, so they
 *   are not repeated as warnings; they stay visible for whoever reviews later.
 *
 * Every check the application import can raise has a catalogue entry. Any other
 * check (e.g. from an imported historical bundle) still appears, with its rule
 * ID and its own reason/action text, so nothing is hidden.
 */
import { el } from "./dom.js";

/** Items listed per finding before "and N more". */
export const LISTED_ITEMS = 10;

// Why the import excluded a finished product (FINI_DATA_QUALITY and
// FINI_OPTIMIZED_PV_ONLY reason codes): short label for the title, what is
// missing in the production workbook, and how to fix it.
const EXCLUSION_REASONS = {
  missing_material: {
    label: "no material code",
    meaning: "The \"Material\" cell of these rows is empty in the production workbook.",
    fix: "Fill in the material code and import the workbook again.",
  },
  missing_sefi: {
    label: "no SEFI",
    meaning: "The \"SEFI\" cell of these products is empty, so they cannot be placed in a SEFI block.",
    fix: "Fill in the SEFI and import the workbook again.",
  },
  missing_demand: {
    label: "no forecast",
    meaning: "The \"Total FINI forecast (litres)\" cell of these products is empty.",
    fix: "Fill in the forecast and import the workbook again.",
  },
  nonpositive_demand: {
    label: "a zero or negative forecast",
    meaning: "The \"Total FINI forecast (litres)\" of these products is zero or negative, so they have no demand to plan.",
    fix: "Correct the forecast if the products are still sold, then import the workbook again.",
  },
  missing_line_evidence: {
    label: "no filling-line data",
    meaning: "The \"Filling line\" cell of these products is empty or cannot be read, and no line correction covers them.",
    fix: "Fill in the filling lines (for example \"23 - 24\") and import the workbook again.",
  },
  no_eligible_line: {
    label: "no eligible filling line",
    meaning: "After applying the line corrections, these products have no filling line they can use.",
    fix: "Review the \"Filling line\" cell and the line corrections, then import the workbook again.",
  },
  missing_package_volume: {
    label: "no package volume",
    meaning: "The \"Volume\" cell (package size) of these products is empty.",
    fix: "Fill in the package volume and import the workbook again.",
  },
  nonpositive_package_volume: {
    label: "a zero or negative package volume",
    meaning: "The \"Volume\" (package size) of these products is zero or negative.",
    fix: "Correct the package volume and import the workbook again.",
  },
  missing_pallet_data: {
    label: "no pallet size",
    meaning: "Neither the production workbook (\"Pallet in litres\") nor the enrichment workbook gives a pallet size for these products.",
    fix: "Fill in the pallet size in either workbook and import again.",
  },
  nonpositive_pallet_data: {
    label: "a zero or negative pallet size",
    meaning: "The pallet size of these products is zero or negative.",
    fix: "Correct the pallet size and import again.",
  },
  missing_fixed_pv: {
    label: "no selected production version",
    meaning: "The \"Selected PV\" cell of these products is empty, and their SEFI has no production version with one positive lot size in the production versions list.",
    fix: "Fill in \"Selected PV\", or the lot size of the SEFI's production versions, and import the workbook again.",
  },
  fixed_pv_not_unique_positive_lot: {
    label: "a selected production version without one positive lot size",
    meaning: "The \"Selected PV\" of these products has no lot size, or several different lot sizes, in the production versions list.",
    fix: "Give the selected production version one positive \"Lot size (Litres)\" and import the workbook again.",
  },
};

/** Reasons where OPTIMIZED may still include a product that FIXED excludes. */
const PV_REASONS = new Set(["missing_fixed_pv", "fixed_pv_not_unique_positive_lot"]);

/**
 * Whether a production-version finding comes from an import made before the
 * FINI_OPTIMIZED_PV_ONLY split. Such findings have no `observed` text and may
 * still contain products OPTIMIZED includes, so their text must not say
 * "excluded with both settings".
 */
const legacyPvFinding = (reason, raw = []) =>
  PV_REASONS.has(reason) && raw.some((issue) => issue.observed == null);

/** Catalogue text of one exclusion reason; unknown reasons keep their code. */
const reasonText = (reason) =>
  EXCLUSION_REASONS[reason] || { label: reason || "data quality", meaning: "", fix: "" };

const products = (count) => `${count} finished product${count === 1 ? "" : "s"}`;

/**
 * Catalogue of checks: title(count, reason), meaning, effect and the table view
 * that shows the affected rows (`link`: view, optional search field/value, sort).
 * Findings of one check form one group; `byReason` splits a check by its reason.
 */
const CHECKS = {
  FINI_DATA_QUALITY: {
    byReason: true, // One group per exclusion reason.
    title: (count, reason) => `${products(count)} excluded: ${reasonText(reason).label}`,
    meaning: (reason, raw) =>
      legacyPvFinding(reason, raw)
        ? "The \"Selected PV\" of these products is empty or has no single positive lot size in the production versions list."
        : reasonText(reason).meaning || "The production workbook does not give enough information to model these products.",
    effect: (reason, raw) =>
      legacyPvFinding(reason, raw)
        ? "Production version FIXED (the default) leaves them out. OPTIMIZED can include those whose SEFI has a production version with one positive lot size; ask the Assistant which ones, or import the workbook again to see them as a separate finding."
        : `They are left out of every optimization run, with both production version settings (FIXED and OPTIMIZED). ${reasonText(reason).fix || "Fix the workbook and import it again to include them."}`,
    link: (reason) => ({ view: "fini_master", field: "exclusion_reason", value: reason || "" }),
  },
  FINI_OPTIMIZED_PV_ONLY: {
    byReason: true,
    title: (count) => `${products(count)} only usable with the OPTIMIZED production version`,
    meaning: (reason) =>
      reason === "fixed_pv_not_unique_positive_lot"
        ? "Their \"Selected PV\" has no single positive lot size, but their SEFI has another production version with one positive lot size."
        : "Their \"Selected PV\" cell is empty, but their SEFI has a production version with one positive lot size in the production versions list.",
    effect: () =>
      "You choose per run in the Optimizer workspace. With Production version FIXED (the default) these products are left out. With OPTIMIZED they are included and the optimizer chooses their production version. To use them with FIXED, fill in \"Selected PV\" and import the workbook again.",
    link: (reason) => ({ view: "fini_master", field: "exclusion_reason", value: reason || "" }),
  },
  PALLET_NONBLANK_SOURCE_DISCREPANCY: {
    title: (count) => `${products(count)} with a different pallet size in the enrichment workbook`,
    meaning: "The enrichment workbook gives one or more pallet sizes that differ from the production workbook.",
    effect: "The production workbook value is kept; the optimization does not change.",
    link: () => ({ view: "pallet_resolution", sort: [{ field: "conflict", direction: "desc" }] }),
  },
  SAP_LOOKUP_MISSING: {
    title: (count) => `${count} SAP lookup table${count === 1 ? "" : "s"} missing from the enrichment workbook`,
    meaning: "The optimizer does not need it; only SAP settings generation does.",
    effect: "The SAP settings tabs that use it stay blocked until the table is uploaded or the value is set in the plant policy.",
  },
  FINI_KEY_UNIQUE: {
    title: (count) => `${products(count)} appear more than once`,
    meaning: "The same material and plant occur in several rows of the production workbook.",
    effect: "The snapshot cannot be published until the duplicates are removed from the workbook.",
  },
  CONFLICTING_LINE_CORRECTIONS: {
    title: (count) => `${products(count)} with conflicting filling-line corrections`,
    meaning: "Several correction rows give different filling lines for the same product.",
    effect: "Their eligible lines are unknown and the snapshot cannot be published until the corrections agree.",
    link: () => ({ view: "line_corrections" }),
  },
  INVALID_XYZ_COUNTS: {
    title: (count) => `${products(count)} with an invalid XYZ DC count`,
    meaning: "The XYZ column must hold three whole numbers (X-Y-Z).",
    effect: "The source text is kept and the counts are treated as unknown.",
    link: () => ({ view: "fini_master", field: "xyz_status", value: "invalid" }),
  },
  INVALID_DC_COUNT: {
    title: (count) => `${products(count)} with an invalid DC count`,
    meaning: "Primary and secondary DC counts must be whole numbers of zero or more.",
    effect: "The source text is kept and the count is treated as unknown.",
  },
  INCONSISTENT_SOURCE_HORIZON: {
    title: (count) => `${products(count)} imply a different demand horizon`,
    meaning: "Their total forecast divided by daily demand does not match the other rows.",
    effect: "No demand horizon is derived from the source; review the forecast and daily-demand columns.",
  },
};

/** Split one finding's entity key into the affected items. */
function keysOf(issue) {
  return String(issue.entity_key ?? "")
    .split("|")
    .map((key) => key.trim())
    .filter(Boolean);
}

/** Checks that describe the chosen import scope; shown as scope notes, not findings. */
const SCOPE_CHECKS = new Set(["PRODUCTION_FREQUENCY_UNDETERMINED"]);

/** Resolve a catalogue text that may depend on the group's reason and raw findings. */
const textOf = (value, reason, raw) => (typeof value === "function" ? value(reason, raw) : value);

/**
 * Group raw findings by check and reason, in plain language.
 *
 * @param {object[]} issues `dataset.issues` as returned by the API.
 * @returns {object[]} Groups `{key, rule_id, reason, severity, title, meaning, effect,
 *   items, total, link}`, errors first, then by number of affected items. `items`
 *   holds every listed key; `total` also counts items the finding only counted
 *   (`observed`) without listing them.
 */
export function findingGroups(issues = []) {
  const groups = new Map();
  for (const issue of issues || []) {
    const rule = issue.rule_id || "UNKNOWN_CHECK";
    // Scope decisions were made in the extraction settings; scopeNotes lists them.
    if (SCOPE_CHECKS.has(rule)) continue;
    // Catalogued checks group by rule (by reason only when flagged); unknown checks by rule and reason.
    const reason = !CHECKS[rule] || CHECKS[rule].byReason ? issue.reason || "" : "";
    const key = `${rule}|${reason}`;
    if (!groups.has(key))
      groups.set(key, { key, rule_id: rule, reason, severity: "warning", items: [], total: 0, raw: [] });
    const group = groups.get(key);
    const items = keysOf(issue);
    const observed = Number(issue.observed);
    group.items.push(...items);
    // A finding may count more items than it lists (older imports listed at most 10).
    group.total += Number.isInteger(observed) && observed > items.length ? observed : Math.max(items.length, 1);
    if (issue.severity === "error") group.severity = "error";
    group.raw.push(issue);
  }
  return [...groups.values()]
    .map(({ raw, ...group }) => {
      const check = CHECKS[group.rule_id];
      if (check)
        return {
          ...group,
          title: check.title(group.total, group.reason),
          meaning: textOf(check.meaning, group.reason, raw),
          effect: textOf(check.effect, group.reason, raw),
          link: check.link ? check.link(group.reason) : null,
        };
      const first = raw[0];
      return {
        ...group,
        title: `${group.rule_id}${group.reason ? `: ${group.reason}` : ""} (${group.total})`,
        meaning: [first.observed != null && `Observed: ${first.observed}`, first.expected != null && `Expected: ${first.expected}`]
          .filter(Boolean)
          .join(". "),
        effect: first.action || "",
        link: null,
      };
    })
    .sort((left, right) => (left.severity === "error" ? 0 : 1) - (right.severity === "error" ? 0 : 1) || right.total - left.total);
}

/** Text for the affected items: the first LISTED_ITEMS, then "and N more". */
export function itemsLine(group) {
  const listed = group.items.slice(0, LISTED_ITEMS);
  const more = group.total - listed.length;
  if (!listed.length) return "";
  return `${listed.join(", ")}${more > 0 ? ` and ${more} more` : ""}`;
}

/** Leading whole number of a text such as "12 FINI rows"; null when absent. */
function leadingCount(value) {
  const match = /^\s*(\d+)/.exec(String(value ?? ""));
  return match ? Number(match[1]) : null;
}

/**
 * Describe the products the extraction settings left out of scope.
 *
 * @param {object[]} issues `dataset.issues`; PRODUCTION_FREQUENCY_UNDETERMINED
 *   findings carry the source value in `entity_key` and "N FINI rows" in `observed`.
 * @param {object} metadata `dataset.metadata` with `frequency_inventory`
 *   (`codes` with `code`/`rows`) and `settings.modeled_frequencies` (list of codes).
 * @returns {object[]} Notes `{key, rule_id, reason, severity: "info", title, meaning, effect,
 *   items, total, products, link}` in the shape of finding groups, so they share the
 *   renderer; `products` is the number of finished products the note puts out of scope.
 */
export function scopeNotes(issues = [], metadata = {}) {
  const notes = [];
  // Frequency codes present in the workbook but not selected in the extraction settings.
  const inventory = metadata?.frequency_inventory;
  const modeled = metadata?.settings?.modeled_frequencies;
  if (inventory?.codes && Array.isArray(modeled)) {
    const skipped = inventory.codes.filter((item) => !modeled.includes(item.code));
    const total = skipped.reduce((sum, item) => sum + (item.rows || 0), 0);
    if (skipped.length)
      notes.push({
        key: "FREQUENCY_NOT_SELECTED|",
        rule_id: "FREQUENCY_NOT_SELECTED",
        reason: "",
        severity: "info",
        title: `${products(total)} with a production frequency not selected in the extraction settings`,
        meaning: `The extraction settings modeled ${modeled.join(", ")}; these products have ${skipped.map((item) => item.code).join(", ")}.`,
        effect: "They are out of scope for this snapshot. Import the workbook again with these codes selected to include them.",
        items: skipped.map((item) => `${item.code} (${item.rows})`),
        total: skipped.length,
        products: total,
        link: {
          view: "fini_master",
          field: "exclusion_reason",
          value: skipped.length === 1 ? `production_frequency_${skipped[0].code.toLowerCase()}_out_of_scope` : "out_of_scope",
        },
      });
  }
  // Frequency values without a weekly code ("999. Not possible to determine").
  for (const issue of issues || []) {
    if (issue.rule_id !== "PRODUCTION_FREQUENCY_UNDETERMINED") continue;
    const count = leadingCount(issue.observed);
    notes.push({
      key: `PRODUCTION_FREQUENCY_UNDETERMINED|${issue.entity_key}`,
      rule_id: "PRODUCTION_FREQUENCY_UNDETERMINED",
      reason: "",
      severity: "info",
      title: count == null
        ? `Products with the production frequency "${issue.entity_key}"`
        : `${products(count)} with the production frequency "${issue.entity_key}"`,
      meaning: "This value is not a weekly code, so no frequency setting can include these products.",
      effect: "They are out of scope. Give them a weekly frequency (for example \"08W\") in the workbook and import it again to include them.",
      items: [],
      total: count ?? 0,
      products: count ?? 0,
      link: { view: "fini_master", field: "exclusion_reason", value: "production_frequency_unrecognized_out_of_scope" },
    });
  }
  return notes;
}

/**
 * Question sent to the Workspace Assistant when the user asks about one entry.
 *
 * @param {object} group A finding group or scope note.
 * @returns {string} A self-contained question naming the check and reason, so
 *   the assistant can call `explain_dataset_finding` for the selected dataset.
 */
export function assistantQuestion(group) {
  const check = `check ${group.rule_id}${group.reason ? `, reason ${group.reason}` : ""}`;
  return `Explain the extraction finding "${group.title}" of the selected dataset (${check}): why it happens with the data of the affected products, how it affects optimization runs, and how to fix it.`;
}

/** Build one finding or scope-note card with its optional table and assistant buttons. */
function findingCard(group, onShow, onAsk) {
  const item = el("div", null, `finding finding-${group.severity}`);
  const head = el("div", null, "finding-head");
  const badge = { error: "Error", warning: "Warning", info: "Scope" }[group.severity] || "Warning";
  head.append(el("span", badge, "finding-severity"), el("strong", group.title));
  item.append(head);
  if (group.meaning) item.append(el("p", group.meaning));
  if (group.effect) item.append(el("p", group.effect));
  const affected = itemsLine(group);
  if (affected) item.append(el("p", `Affected: ${affected}`, "muted finding-items"));
  const actions = el("div", null, "finding-actions");
  if (group.link) {
    const button = el("ui5-button", "Show in table");
    button.setAttribute("design", "Transparent");
    button.addEventListener("click", () => onShow(group.link));
    actions.append(button);
  }
  if (onAsk) {
    const ask = el("ui5-button", "Ask the Assistant");
    ask.setAttribute("design", "Transparent");
    ask.setAttribute("icon", "discussion");
    ask.addEventListener("click", () => onAsk(assistantQuestion(group)));
    actions.append(ask);
  }
  if (actions.childElementCount) item.append(actions);
  return item;
}

/**
 * Render the grouped findings and, collapsed below them, the scope notes.
 *
 * @param {HTMLElement} target Container to replace.
 * @param {object[]} groups Output of `findingGroups`.
 * @param {(link: object) => void} onShow Opens a link in the evidence table.
 * @param {object} [options]
 * @param {object[]} [options.notes] Output of `scopeNotes`.
 * @param {(question: string) => void} [options.onAsk] Sends a question to the
 *   Workspace Assistant; without it no "Ask the Assistant" button is shown.
 */
export function renderFindings(target, groups, onShow, { notes = [], onAsk = null } = {}) {
  const heading = el("div", null, "findings-heading");
  heading.append(
    el("h3", "Extraction findings"),
    el("span", groups.length ? "Errors block publishing; warnings do not." : "No extraction findings.", "muted"),
  );
  /** @type {HTMLElement[]} */
  const children = [heading, ...groups.map((group) => findingCard(group, onShow, onAsk))];
  if (notes.length) {
    // Scope notes repeat choices made before extraction, so they start collapsed.
    const scope = el("details", null, "scope-notes");
    const out = notes.reduce((sum, note) => sum + (note.products || 0), 0);
    scope.append(el("summary", `Scope set by the extraction settings (${products(out)} out of scope by production frequency)`));
    scope.append(...notes.map((note) => findingCard(note, onShow, onAsk)));
    children.push(scope);
  }
  target.replaceChildren(...children);
}
