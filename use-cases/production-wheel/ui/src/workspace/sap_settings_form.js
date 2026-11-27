/**
 * SAP settings section of the plant profile form (Settings page).
 *
 * These rules shape the SAP upload generated after a run: DC minimum lot and
 * rounding per ABC class, cycle days of materials outside the wheel, planning
 * calendar codes, recipe characters and two switches. They are stored in the
 * profile as `settings.sap_settings` and read live when SAP settings are
 * generated, so changing them needs no re-run.
 *
 * The form is plain markup placed inside `.configuration` before the help
 * buttons are attached (help.js decorates controls by id). Reading and writing
 * go through an accessor `field(id)` returning the control, so the logic is
 * testable without a DOM.
 */

/** Allowed minimum-lot rules at the DC (stored value -> label). */
export const MIN_LOT_RULES = {
  pallet: "One pallet",
  half_pallet: "Half a pallet (whole layers)",
  one_piece: "One piece",
  litres: "Litres (enter amount)",
  none: "No minimum",
};

/** Allowed rounding rules at the DC (stored value -> label). */
export const ROUNDING_RULES = {
  layer_at_max_days: "Layer at maximum days, else pallet",
  pallet: "One pallet",
  layer: "One layer",
  half_pallet: "Half a pallet",
  one_piece: "One piece",
};

/** ABC scopes of the per-class table: profile key -> row label. */
const CLASS_SCOPES = { all_classes: "All classes", A: "Class A", B: "Class B", C: "Class C" };

/** Text and switch fields: setting -> {id, label, kind}. */
const SIMPLE_FIELDS = {
  calendar_odd_profile: { id: "sap_calendar_odd_profile", label: "Odd-week lot-size profile", kind: "optional" },
  calendar_even_profile: { id: "sap_calendar_even_profile", label: "Even-week lot-size profile", kind: "optional" },
  calendar_frequency_code: { id: "sap_calendar_frequency_code", label: "Two-weekly frequency code", kind: "text" },
  calendar_lot_size: { id: "sap_calendar_lot_size", label: "Calendar lot-size procedure", kind: "text" },
  calendar_reset_lot_size: { id: "sap_calendar_reset_lot_size", label: "Reset lot-size procedure", kind: "text" },
  calendar_reset_text: { id: "sap_calendar_reset_text", label: "Reset text", kind: "text" },
  recipe_characters: { id: "sap_recipe_characters", label: "Fallback recipe characters", kind: "optional" },
  apo_default_pv: { id: "sap_apo_default_pv", label: "APO default production version", kind: "text" },
  kpi_line_breakdown: { id: "sap_kpi_line_breakdown", label: "Changeover KPI per filling line", kind: "switch" },
  multi_tier_dc: { id: "sap_multi_tier_dc", label: "Plant ships through an intermediate DC", kind: "switch" },
};

/** People-facing names of every SAP setting, shared with approval cards. */
export const SAP_FIELD_LABELS = {
  ...Object.fromEntries(Object.entries(SIMPLE_FIELDS).map(([key, spec]) => [key, spec.label])),
  dc_min_lot: "Minimum lot at DC",
  dc_rounding: "Rounding value at DC",
  abc_cycle_days: "Cycle days outside the wheel",
};

/** Escape text placed in markup attributes or content. */
function esc(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

/** Build `<option>`s; `sameAsAll` adds the "same as all classes" choice (empty value). */
function ruleOptions(rules, sameAsAll) {
  const first = sameAsAll ? `<option value="">Same as all classes</option>` : "";
  return first + Object.entries(rules).map(([value, label]) => `<option value="${value}">${esc(label)}</option>`).join("");
}

/**
 * Markup of the SAP settings section, placed inside the profile `.configuration` card.
 * @returns {string} HTML with stable ids for help buttons and the form accessors.
 */
export function sapFormMarkup() {
  const simple = (key) => {
    const spec = SIMPLE_FIELDS[key];
    return spec.kind === "switch"
      ? `<label><input id="${spec.id}" type="checkbox"> ${esc(spec.label)}</label>`
      : `<label>${esc(spec.label)}<input id="${spec.id}" maxlength="${key === "calendar_reset_text" || key === "recipe_characters" ? 128 : 64}"${key.startsWith("calendar_") && spec.kind === "optional" ? ' placeholder="e.g. ODD-W1"' : ""}></label>`;
  };
  const rows = Object.entries(CLASS_SCOPES).map(([scope, label]) => {
    const all = scope === "all_classes";
    const days = all ? `<input id="sap_cycle_days_unclassified" type="number" min="0.000001" step="any" placeholder="skip" aria-label="Cycle days for materials without ABC class">`
      : `<input id="sap_cycle_days_${scope}" type="number" min="0.000001" step="any" aria-label="Cycle days, ${label}">`;
    return `<tr><th scope="row">${label}</th>`
      + `<td><select id="sap_min_lot_${scope}" aria-label="Minimum lot at DC, ${label}">${ruleOptions(MIN_LOT_RULES, !all)}</select></td>`
      + `<td><input id="sap_min_lot_litres_${scope}" type="number" min="0.000001" step="any" aria-label="Litres, ${label}"></td>`
      + `<td><select id="sap_rounding_${scope}" aria-label="Rounding value at DC, ${label}">${ruleOptions(ROUNDING_RULES, !all)}</select></td>`
      + `<td>${days}</td></tr>`;
  }).join("");
  return `<div class="sap-settings-form">`
    + `<p id="sap-settings" class="form-section">SAP settings</p>`
    + `<p class="muted">Used when SAP settings are generated for a selected scenario. They are read when you generate, so changes apply without running the optimizer again.</p>`
    + `<div class="form-grid">${["calendar_odd_profile", "calendar_even_profile"].map(simple).join("")}`
    + `<div class="actions"><ui5-button id="sap-detect">Detect from dataset</ui5-button></div><p id="sap-detect-status" class="muted" role="status"></p>`
    + ["calendar_frequency_code", "calendar_lot_size", "calendar_reset_lot_size", "calendar_reset_text", "recipe_characters", "apo_default_pv", "kpi_line_breakdown", "multi_tier_dc"].map(simple).join("")
    + `</div><p id="sap-dc-rules" class="form-section">DC lot sizes by ABC class</p>`
    + `<div class="table-scroll"><table class="sap-class-table"><thead><tr><th>ABC class</th><th>Minimum lot at DC</th><th>Litres</th><th>Rounding value</th><th>Cycle days outside the wheel</th></tr></thead><tbody>${rows}</tbody></table></div>`
    + `</div>`;
}

/** Number from an input value; blank or invalid becomes null. */
function number(value) {
  if (value === "" || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * Read the form into a `sap_settings` object.
 * @param {(id: string) => {value?: string, checked?: boolean}} field Control accessor by id.
 * @returns {object} SAP settings in the stored profile shape.
 */
export function readSapForm(field) {
  const policy = {};
  for (const [key, spec] of Object.entries(SIMPLE_FIELDS)) {
    if (spec.kind === "switch") policy[key] = Boolean(field(spec.id).checked);
    else {
      const text = String(field(spec.id).value ?? "").trim();
      policy[key] = spec.kind === "optional" && !text ? null : text;
    }
  }
  const rule = (prefix, scope, withNumber) => {
    const name = field(`${prefix}_${scope}`).value;
    if (!name) return null;
    return { rule: name, number: withNumber && name === "litres" ? number(field(`sap_min_lot_litres_${scope}`).value) : null };
  };
  policy.dc_min_lot = Object.fromEntries(Object.keys(CLASS_SCOPES).map((scope) => [scope, rule("sap_min_lot", scope, true)]));
  policy.dc_rounding = Object.fromEntries(Object.keys(CLASS_SCOPES).map((scope) => [scope, rule("sap_rounding", scope, false)]));
  policy.abc_cycle_days = {
    A: number(field("sap_cycle_days_A").value),
    B: number(field("sap_cycle_days_B").value),
    C: number(field("sap_cycle_days_C").value),
    unclassified: number(field("sap_cycle_days_unclassified").value),
  };
  return policy;
}

/**
 * Fill the form from a stored `sap_settings` object (missing parts use their empty state).
 * @param {(id: string) => object} field Control accessor by id.
 * @param {object} policy Stored SAP settings (or defaults).
 */
export function writeSapForm(field, policy = {}) {
  for (const [key, spec] of Object.entries(SIMPLE_FIELDS)) {
    if (spec.kind === "switch") field(spec.id).checked = Boolean(policy[key]);
    else field(spec.id).value = policy[key] ?? "";
  }
  for (const scope of Object.keys(CLASS_SCOPES)) {
    const min = policy.dc_min_lot?.[scope], round = policy.dc_rounding?.[scope];
    field(`sap_min_lot_${scope}`).value = min?.rule ?? (scope === "all_classes" ? "pallet" : "");
    field(`sap_min_lot_litres_${scope}`).value = min?.rule === "litres" ? (min.number ?? "") : "";
    field(`sap_rounding_${scope}`).value = round?.rule ?? (scope === "all_classes" ? "layer_at_max_days" : "");
  }
  const days = policy.abc_cycle_days || {};
  for (const key of ["A", "B", "C", "unclassified"]) field(`sap_cycle_days_${key}`).value = days[key] ?? "";
}

/**
 * Check the form values before saving, with messages planners understand.
 * @param {object} policy Output of readSapForm.
 * @returns {string[]} Problems found (empty when the form can be saved).
 */
export function sapFormProblems(policy) {
  const problems = [];
  for (const key of ["calendar_odd_profile", "calendar_even_profile"]) {
    const code = policy[key];
    if (code && !/^[^-]*\S[^-]*-.*\S/.test(code)) problems.push(`${SAP_FIELD_LABELS[key]} must look like <profile>-<calendar>, e.g. ODD-W1.`);
  }
  if (policy.calendar_odd_profile && policy.calendar_odd_profile === policy.calendar_even_profile)
    problems.push("The odd- and even-week lot-size profiles must differ.");
  for (const [key, spec] of Object.entries(SIMPLE_FIELDS))
    if (spec.kind === "text" && !policy[key]) problems.push(`${spec.label} cannot be empty.`);
  for (const [scope, label] of Object.entries(CLASS_SCOPES)) {
    const min = policy.dc_min_lot?.[scope];
    if (min?.rule === "litres" && !(min.number > 0)) problems.push(`Minimum lot at DC (${label}): enter the litres.`);
  }
  for (const key of ["A", "B", "C"])
    if (!(policy.abc_cycle_days?.[key] > 0)) problems.push(`Cycle days outside the wheel (class ${key}) must be positive.`);
  return problems;
}

/** Readable text of one per-class rule, or "same as all classes". */
function ruleText(rule, rules) {
  if (rule === null) return "same as all classes";
  if (!rule || typeof rule !== "object") return String(rule);
  const label = rules[rule.rule] || rule.rule;
  return rule.rule === "litres" && rule.number ? `${label.replace(" (enter amount)", "")}: ${rule.number}` : label;
}

/**
 * Describe a (partial) `sap_settings` patch for an approval card.
 * @param {object} patch Partial SAP settings, e.g. {calendar_odd_profile: "ODD-W1", dc_min_lot: {C: null}}.
 * @returns {string[]} One readable line per changed value.
 */
export function describeSapPatch(patch = {}) {
  const lines = [];
  for (const [key, value] of Object.entries(patch)) {
    if (key === "dc_min_lot" || key === "dc_rounding") {
      const rules = key === "dc_min_lot" ? MIN_LOT_RULES : ROUNDING_RULES;
      for (const [scope, rule] of Object.entries(value || {}))
        lines.push(`SAP settings · ${SAP_FIELD_LABELS[key]} (${CLASS_SCOPES[scope] || scope}): ${ruleText(rule, rules)}`);
    } else if (key === "abc_cycle_days") {
      for (const [scope, days] of Object.entries(value || {}))
        lines.push(`SAP settings · Cycle days outside the wheel (${scope === "unclassified" ? "no ABC class" : `class ${scope}`}): ${days ?? "skip"}`);
    } else {
      const shown = value === null ? "not set" : typeof value === "boolean" ? (value ? "yes" : "no") : value;
      lines.push(`SAP settings · ${SAP_FIELD_LABELS[key] || key}: ${shown}`);
    }
  }
  return lines;
}
