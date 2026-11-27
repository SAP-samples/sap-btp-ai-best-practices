import { request, downloadExport } from "../services/api.js";
import { action, el, options } from "./dom.js";
import { mountConfigurationHelp } from "./help.js";
import { changeRequest, ruleSummary } from "./proposals.js";
import { readSapForm, sapFormMarkup, sapFormProblems, writeSapForm } from "./sap_settings_form.js";
import { compareVolumeLabels, matrixModeLabel, matrixModeOptions, volumeMatchingLabel, volumeMatchingOptions } from "./vocabulary.js";

// Form input id -> RunnerThresholds field of the plant profile settings.
const RUNNER_FIELDS = {
  runner_high_max_days: "high_max_days",
  runner_low_min_days: "low_min_days",
  runner_high_max_ratio: "high_max_ratio",
  runner_low_min_ratio: "low_min_ratio",
};

// Form input id -> CycleStockPolicy numeric field of the plant profile settings.
const CYCLE_STOCK_FIELDS = {
  cycle_high_min_days: "high_min_days",
  cycle_medium_min_days: "medium_min_days",
  cycle_low_min_days: "low_min_days",
  cycle_medium_batch_factor: "medium_batch_factor",
  cycle_max_days: "max_days",
};

/** Planner labels of the cycle stock strategy modes (stored value -> label). */
export const CYCLE_STOCK_MODES = {
  single: "Single RoC",
  runner_multiple: "Runner multiples",
  optimizer_choice: "Optimizer chooses",
};

/**
 * Explain the cycle stock strategy in plain words for the given policy.
 * @param {{mode?:string, high_min_days?:number|string, medium_min_days?:number|string, low_min_days?:number|string, medium_batch_factor?:number|string, max_days?:number|string, mixed_groups_only?:boolean}} p
 *   Values may be raw form strings; missing values are shown as "?".
 * @returns {string} What the mode does and, when multiples apply, the numbers used; "?" stands in for a missing value.
 */
export function cycleStockDefinition(p = {}) {
  const v = (key) => (p[key] ?? "") === "" ? "?" : p[key];
  const mode = p.mode || "single";
  if (mode === "single") return `Cycle stock: Single RoC. Every product of a subgroup is filled in every batch and gets the subgroup's RoC. Cycle stock is capped at ${v("max_days")} days.`;
  const rule = `the fastest product of a subgroup keeps the subgroup's RoC; every other product is filled only every 2nd, 3rd... batch, so its RoC is a whole multiple of the subgroup's, at least ${v("high_min_days")} days for a high runner, ${v("medium_min_days")} for a medium runner and ${v("low_min_days")} for a low runner. A medium runner covers at most ${v("medium_batch_factor")} of one full batch. Cycle stock is capped at ${v("max_days")} days.`;
  if (mode === "runner_multiple") return `Cycle stock: Runner multiples. In every subgroup, ${rule}`;
  const scope = p.mixed_groups_only === false ? "every subgroup" : "subgroups that mix runner classes";
  return `Cycle stock: Optimizer chooses. For ${scope}, the optimizer compares Single RoC with runner multiples and keeps the better trade-off. With multiples, ${rule}`;
}

/**
 * Explain the three runner classes in plain words for the given cut-offs.
 * @param {{high_max_days?:number|string, low_min_days?:number|string, high_max_ratio?:number|string, low_min_ratio?:number|string}} t
 *   Values may be raw form strings; missing values are shown as "?".
 * @returns {string} One sentence per class; "?" stands in for a missing value.
 */
export function runnerDefinition(t = {}) {
  const v = (key) => (t[key] ?? "") === "" ? "?" : t[key];
  return `High runner: one full batch lasts ${v("high_max_days")} days or less, or it lasts at most ${v("high_max_ratio")} times as long as for the SEFI's fastest product. `
    + `Low runner: one batch lasts ${v("low_min_days")} days or more and at least ${v("low_min_ratio")} times as long as for the fastest product. `
    + "Medium runner: everything in between. The fastest product of each SEFI is always a high runner.";
}

/** Describe frozen profile settings and rules without treating server text as markup. */
export function profileSummary(profile) {
  if (!profile) return "Legacy draft: no plant profile attached.";
  const s = profile.settings || {};
  return `${profile.name} · ${profile.plant} · revision ${profile.revision}. Horizon ${s.horizon_days} demand days, ${s.demand_days_per_week} days/week. ${runnerDefinition(s.runner_thresholds)} ${cycleStockDefinition(s.cycle_stock_policy)} Runner basis: ${s.runner_basis}. Compatibility mode: ${matrixModeLabel(s.matrix_mode)}; volume matching: ${volumeMatchingLabel(s.volume_matching)}; maximum group size: ${s.maximum_group_size}. Fixed constraints: ${profile.rules?.length ? profile.rules.map((rule) => ruleSummary(rule).title).join("; ") : "none"}. Scenario constraints add to these fixed constraints.`;
}

/** Mount editable HANA-backed plant profiles, the approved constraint list, and confirmed removal. */
export function mountSettings(target, app) {
  target.innerHTML = `<section class="page-heading"><div><span class="eyebrow">03 / PLANT SETTINGS</span><h1>Plant profiles</h1><p>Save fixed plant settings and compatibility. Constraints are added with the Workspace Assistant. Submitted runs retain their profile revision.</p></div></section><section class="card configuration"><label>Saved profile<select id="profile-list"></select></label><ui5-button id="profile-new">New profile</ui5-button><div class="form-grid"><label>Profile name<input id="profile-name" required></label><label>Plant<input id="profile-plant" required></label><label>Horizon (demand days)<input id="horizon_days" type="number" min="1"></label><label>Demand days per week<input id="demand_days_per_week" type="number" min="1" max="7"></label><p id="runner-classes" class="form-section">Runner classes</p><label>High runner: batch lasts at most (days)<input id="runner_high_max_days" type="number" min="0.000001" step="any"></label><label>Low runner: batch lasts at least (days)<input id="runner_low_min_days" type="number" min="0.000001" step="any"></label><label>High runner: at most (× fastest product)<input id="runner_high_max_ratio" type="number" min="1" step="any"></label><label>Low runner: at least (× fastest product)<input id="runner_low_min_ratio" type="number" min="1" step="any"></label><p id="cycle-stock" class="form-section">Cycle stock strategy</p><label>Strategy<select id="cycle_mode">${Object.entries(CYCLE_STOCK_MODES).map(([value, label]) => `<option value="${value}">${label}</option>`).join("")}</select></label><label>High runner: minimum RoC (days)<input id="cycle_high_min_days" type="number" min="0" step="any"></label><label>Medium runner: minimum RoC (days)<input id="cycle_medium_min_days" type="number" min="0.000001" step="any"></label><label>Low runner: minimum RoC (days)<input id="cycle_low_min_days" type="number" min="0.000001" step="any"></label><label>Medium runner: share of one batch<input id="cycle_medium_batch_factor" type="number" min="0.000001" max="1" step="any"></label><label>Maximum cycle stock (days)<input id="cycle_max_days" type="number" min="0.000001" step="any"></label><label><input id="cycle_mixed_groups_only" type="checkbox"> Optimizer chooses only for subgroups mixing runner classes</label><label>Runner basis<select id="runner_basis"><option value="reference">Fixed source reference</option><option value="candidate_pv">Candidate PV (sensitivity)</option></select></label><label>Compatibility mode<select id="matrix_mode">${matrixModeOptions()}</select></label><label>Volume matching<select id="volume_matching">${volumeMatchingOptions()}</select></label><label>Maximum group size<input id="maximum_group_size" type="number" min="1"></label></div><p id="runner-definition" class="muted"></p><p id="cycle-stock-definition" class="muted"></p>${sapFormMarkup()}<div class="plant-constraints"><div class="section-heading"><h2 id="plant-constraints">Plant constraints</h2><ui5-button id="constraints-assistant">Ask the Assistant</ui5-button></div><div id="constraint-list" class="constraint-list"></div><p id="constraint-status" role="status"></p></div><h2>Volume compatibility</h2><p>Download the current matrix, edit it in Excel, then upload it for review. Add or remove volumes on both axes. Alternatively, describe compatible volume families to the Workspace Assistant and approve its proposal.</p><p>Nearest reference volume: rows are reference volumes plus optional &lt;n (below the smallest) and &gt;n (above the largest) classes; each product uses the row of its nearest reference volume (1.1 L uses 1, 2.2 L uses 2.5, 20 L uses &gt;15). Exact volumes: the matrix lists every package volume and a missing one stops the run.</p><p>Y: Compatible · AVOID: Discouraged · N: Forbidden. Accept replaces the entire matrix in this form. Save profile persists your accepted changes.</p><div class="actions"><ui5-button id="matrix-download">Download Excel template</ui5-button><ui5-button id="matrix-upload">Upload Excel</ui5-button><input id="matrix-file" type="file" accept=".xlsx" hidden aria-label="Compatibility Excel file"><ui5-button id="matrix-accept" disabled>Accept</ui5-button><ui5-button id="matrix-default">Use default class matrix</ui5-button></div><p id="matrix-status" role="status"></p><details id="matrix-review"><summary>Review volume compatibility matrix</summary><div id="profile-matrix" class="table-scroll"></div></details><p id="profile-status" role="status"></p><div class="actions"><ui5-button id="profile-save" design="Emphasized">Save profile</ui5-button><ui5-button id="profile-delete" design="Negative" disabled>Remove profile</ui5-button></div></section><section class="card ai-model-settings"><div class="section-heading"><div><span class="eyebrow">AI MODEL SETTINGS</span><h2>AI model settings</h2></div></div><p class="muted">Choose the AI Core model for the workspace assistant, including its constraint proposals. Changes apply to new LLM requests.</p><label>Model<select id="ai-model" disabled></select></label><p id="ai-model-status" role="status">Loading AI model settings…</p><ui5-button id="ai-model-save" design="Emphasized" disabled>Save AI model</ui5-button></section>`;
  mountConfigurationHelp(target, "/api/plant-profile-help").catch(app.fail);
  const input = (id) => target.querySelector(`#${id}`);
  const loadingControls = [...target.querySelectorAll(".configuration input, .configuration select, .configuration textarea, .configuration ui5-button")];
  for (const control of loadingControls) control.disabled = true;
  input("profile-status").textContent = "Loading plant profiles…";
  const keys = ["horizon_days", "demand_days_per_week", "runner_basis", "matrix_mode", "volume_matching", "maximum_group_size"];
  const TEXT_KEYS = ["runner_basis", "matrix_mode", "volume_matching"];
  // Profiles saved before volume classes have no volume_matching: they are exact.
  const KEY_FALLBACKS = {volume_matching: "exact"};
  let profiles = [], defaults, current = null, rows = [], savedKey = null, live = true;
  let pendingMatrix = null;
  let modelRevision = null, savedModel = null;
  /** Load the server-owned catalogue and current global model selection. */
  async function loadAIModelSettings() {
    const result = await request("/api/ai-model-settings");
    if (!live) return;
    options(input("ai-model"), result.models.map((model) => [model.name, model.label]), result.model);
    savedModel = result.model;
    modelRevision = result.revision;
    input("ai-model").disabled = false;
    input("ai-model-save").disabled = true;
    input("ai-model-status").textContent = `Current model: ${result.models.find((model) => model.name === result.model)?.label || result.model}.`;
  }
  input("ai-model").addEventListener("change", () => {
    input("ai-model-save").disabled = input("ai-model").value === savedModel;
    input("ai-model-status").textContent = input("ai-model-save").disabled ? "Model unchanged." : "Save to apply this model to new requests.";
  });
  action(input("ai-model-save"), async () => {
    input("ai-model-save").disabled = true;
    input("ai-model-status").textContent = "Saving AI model…";
    try {
      const result = await request("/api/ai-model-settings", "PUT", {model: input("ai-model").value, revision: modelRevision});
      if (live) {
        savedModel = result.model;
        modelRevision = result.revision;
        input("ai-model-status").textContent = `Saved. New LLM requests use ${result.models.find((model) => model.name === result.model)?.label || result.model}.`;
      }
    } catch (error) {
      if (live) {
        input("ai-model-save").disabled = false;
        input("ai-model-status").textContent = error.message;
      }
    }
  }, app.fail);
  loadAIModelSettings().catch((error) => { if (live) input("ai-model-status").textContent = error.message; });
  /** Read typed settings for the profile request. */
  function settings() {
    const values = Object.fromEntries(keys.map((key) => [key, TEXT_KEYS.includes(key) ? input(key).value : Number(input(key).value)]));
    values.runner_thresholds = Object.fromEntries(Object.entries(RUNNER_FIELDS).map(([id, field]) => [field, Number(input(id).value)]));
    values.cycle_stock_policy = {
      mode: input("cycle_mode").value,
      ...Object.fromEntries(Object.entries(CYCLE_STOCK_FIELDS).map(([id, field]) => [field, Number(input(id).value)])),
      mixed_groups_only: input("cycle_mixed_groups_only").checked,
    };
    // SAP settings section (DC lot sizes, calendar codes, ...), see sap_settings_form.js.
    values.sap_settings = readSapForm(input);
    return values;
  }
  /** Identify every editable form value; differs from savedKey when the form has unsaved edits. */
  function signature() { return JSON.stringify([current?.profile_id || "", input("profile-name").value.trim(), input("profile-plant").value.trim(), settings(), rows]); }
  /** True when the form differs from the saved profile or a matrix awaits Accept. */
  function dirty() { return Boolean(pendingMatrix) || signature() !== savedKey; }
  /** Refresh derived help text after any form edit. */
  function changed() {
    app.clearNotice();
    input("runner-definition").textContent = runnerDefinition(Object.fromEntries(Object.entries(RUNNER_FIELDS).map(([id, field]) => [field, input(id).value])));
    input("cycle-stock-definition").textContent = cycleStockDefinition({mode: input("cycle_mode").value,
      ...Object.fromEntries(Object.entries(CYCLE_STOCK_FIELDS).map(([id, field]) => [field, input(id).value])),
      mixed_groups_only: input("cycle_mixed_groups_only").checked});
  }
  /** List the approved constraints of the saved profile, each removable after confirmation. */
  function renderConstraints() {
    const list = input("constraint-list");
    const rules = current?.rules || [];
    list.replaceChildren();
    if (!current) {
      list.append(el("p", "Save the profile first, then ask the Assistant to add constraints.", "muted"));
      return;
    }
    if (!rules.length) list.append(el("p", "No constraints yet. Ask the Assistant to add one for this profile.", "muted"));
    for (const rule of rules) {
      const summary = ruleSummary(rule);
      const item = el("div", null, "constraint-item");
      const text = el("div");
      text.append(el("strong", summary.title), el("p", summary.meta, "muted"));
      if (summary.source) text.append(el("p", `Planner wording: ${summary.source}`, "muted"));
      const details = el("details");
      details.append(el("summary", "Typed rule"), el("pre", JSON.stringify(rule, null, 2), "evidence"));
      text.append(details);
      const remove = el("ui5-button", "Remove");
      remove.setAttribute("design", "Negative");
      action(remove, () => removeConstraint(rule), app.fail);
      item.append(text, remove);
      list.append(item);
    }
  }
  /** Remove one approved constraint against the displayed revision, after confirmation. */
  async function removeConstraint(rule) {
    if (dirty()) throw new Error("Save or discard your profile edits before removing a constraint.");
    if (!window.confirm(`Remove constraint "${ruleSummary(rule).title}"? New drafts will no longer apply it.`)) return;
    const { path, body } = changeRequest(current.profile_id, current.revision, {action: "remove_constraint", constraint_id: rule.constraint_id});
    await request(path, "POST", body);
    await refresh(current.profile_id);
  }
  /** Show a read-only matrix; all edits come from reviewed Excel or natural language. */
  function renderMatrix() {
    const shown = pendingMatrix?.rows || rows;
    const volumes = [...new Set(shown.map((r) => String(r.volume_a)))].sort(compareVolumeLabels);
    const pairs = new Map(shown.map((r) => [JSON.stringify([r.volume_a, r.volume_b]), r.status]));
    const table = el("table"), head = el("tr");
    head.append(el("th", "Volume (L)"), ...volumes.map((v) => el("th", v)));
    table.append(head);
    for (const a of volumes) {
      const tr = el("tr"); tr.append(el("th", a));
      for (const b of volumes) tr.append(el("td", pairs.get(JSON.stringify([a,b]))));
      table.append(tr);
    }
    input("profile-matrix").replaceChildren(table);
    input("matrix-status").textContent = pendingMatrix
      ? `${pendingMatrix.source}: ${volumes.length} volumes, ${shown.length} pairs. Review and Accept before saving.`
      : `Current matrix: ${volumes.length} volumes, ${shown.length} pairs. Upload Excel or ask the Assistant to propose a replacement.`;
    input("matrix-accept").disabled = !pendingMatrix;
  }
  /** Load an existing profile or server defaults without overwriting any saved profile. */
  function edit(profile) {
    current = profile;
    pendingMatrix = null;
    input("profile-list").value = profile?.profile_id || "";
    input("profile-name").value = profile?.name || "";
    input("profile-plant").value = profile?.plant || "";
    for (const key of keys) input(key).value = (profile || defaults).settings[key] ?? KEY_FALLBACKS[key];
    const thresholds = (profile || defaults).settings.runner_thresholds || {};
    for (const [id, field] of Object.entries(RUNNER_FIELDS)) input(id).value = thresholds[field] ?? "";
    const cycle = (profile || defaults).settings.cycle_stock_policy || {};
    input("cycle_mode").value = cycle.mode || "single";
    for (const [id, field] of Object.entries(CYCLE_STOCK_FIELDS)) input(id).value = cycle[field] ?? "";
    input("cycle_mixed_groups_only").checked = cycle.mixed_groups_only !== false;
    writeSapForm(input, (profile || defaults).settings.sap_settings || {});
    input("sap-detect-status").textContent = "";
    rows = structuredClone((profile || defaults).matrix_rows || []);
    renderMatrix();
    renderConstraints();
    savedKey = signature();
    input("profile-delete").disabled = !profile;
    input("profile-status").textContent = profile ? `Saved revision ${profile.revision}` : "New profile";
    changed();
  }
  /** Reload active profiles after a save or removal. */
  async function refresh(selected = "") {
    const result = await request("/api/plant-profiles");
    if (!live) return;
    profiles = result.profiles || [];
    options(input("profile-list"), [["", "New profile"], ...profiles.map((p) => [p.profile_id, `${p.name} · ${p.plant} · r${p.revision}`])], selected);
    edit(profiles.find((p) => p.profile_id === selected));
  }
  input("profile-list").closest(".configuration").addEventListener("input", changed);
  input("profile-list").closest(".configuration").addEventListener("change", changed);
  input("profile-list").addEventListener("change", () => {
    edit(profiles.find((p) => p.profile_id === input("profile-list").value));
    // The assistant's profile tools act on the profile selected here.
    if (current) app.select({plant_profile_id: current.profile_id});
  });
  action(input("constraints-assistant"), () => {
    if (current) app.select({plant_profile_id: current.profile_id});
    app.openAssistant();
  }, app.fail);
  /** Reload after an approved assistant change unless that would discard unsaved edits. */
  function profileChanged(event) {
    if (!live || event.detail?.profile_id !== current?.profile_id) return;
    if (!dirty()) refresh(current.profile_id).catch(app.fail);
    else input("constraint-status").textContent = "This profile was updated from the Assistant. Save or discard your edits, then reselect it to see the change.";
  }
  app.events.addEventListener("profile-changed", profileChanged);
  action(input("profile-new"), () => edit(null), app.fail);
  // Learn the odd/even calendar lot-size profiles from an earlier wheel of this plant.
  // The values only fill the form; Save profile keeps them.
  action(input("sap-detect"), async () => {
    const status = input("sap-detect-status");
    if (!current) {
      status.textContent = "Save the profile first: detection reads the datasets of its plant.";
      return;
    }
    status.textContent = "Looking for an earlier planning calendar of this plant…";
    const query = app.state.dataset_id ? `?dataset_id=${encodeURIComponent(app.state.dataset_id)}` : "";
    const result = await request(`/api/plant-profiles/${encodeURIComponent(current.profile_id)}/sap-settings/suggestions${query}`);
    if (!live) return;
    const found = (result.suggestions || [])[0];
    if (!found) {
      status.textContent = `No dataset of plant ${current.plant} has an earlier planning-calendar table. Enter the codes from the plant's SAP customizing, or ask the Assistant.`;
      return;
    }
    if (found.status !== "suggested") {
      status.textContent = `${found.message} Enter the codes from the plant's SAP customizing.`;
      return;
    }
    input("sap_calendar_odd_profile").value = found.values.calendar_odd_profile;
    input("sap_calendar_even_profile").value = found.values.calendar_even_profile;
    changed();
    status.textContent = `${found.message} Source: dataset "${found.dataset_name}". Review the codes and Save profile to keep them.`;
  }, app.fail);
  action(input("matrix-download"), () => downloadExport("/api/plant-profiles/matrix-template", "volume-compatibility.xlsx", {matrix_rows: rows}), app.fail);
  action(input("matrix-upload"), () => input("matrix-file").click(), app.fail);
  input("matrix-file").addEventListener("change", async () => {
    const file = input("matrix-file").files[0];
    if (!file) return;
    const key = signature();
    pendingMatrix = null; renderMatrix();
    input("matrix-status").textContent = "Validating uploaded Excel…";
    try {
      const body = new FormData(); body.append("file", file);
      const result = await request("/api/plant-profiles/matrix-preview", "POST", body);
      if (!live || key !== signature()) return;
      pendingMatrix = {rows: result.matrix_rows, source: "Excel preview", key};
      renderMatrix(); input("matrix-review").open = true;
    } catch (error) { renderMatrix(); app.fail(error); }
    finally { input("matrix-file").value = ""; }
  });
  // The customer's default class matrix (<1, 1, 2.5, 5, 10, 15, >15) from the server
  // defaults; it is reviewed like an upload and needs nearest reference volume matching.
  action(input("matrix-default"), () => {
    pendingMatrix = {rows: structuredClone(defaults.matrix_rows), source: "Default class matrix", key: signature()};
    input("volume_matching").value = "nearest_class";
    renderMatrix(); input("matrix-review").open = true;
    changed();
  }, app.fail);
  action(input("matrix-accept"), () => {
    if (!pendingMatrix) return;
    if (pendingMatrix.key !== signature()) throw new Error("Inputs changed. Upload or preview the matrix again before accepting.");
    rows = pendingMatrix.rows;
    pendingMatrix = null; renderMatrix(); changed();
    input("matrix-status").textContent = `Accepted ${new Set(rows.map((r) => r.volume_a)).size} volumes. Save profile to persist this matrix.`;
  }, app.fail);
  action(input("profile-save"), async () => {
    if (pendingMatrix) throw new Error("Accept the proposed volume compatibility matrix before saving.");
    const t = settings().runner_thresholds;
    if (!(t.high_max_days > 0 && t.high_max_days < t.low_min_days)) throw new Error("Runner days: the high-runner value must be positive and below the low-runner value.");
    if (!(t.high_max_ratio >= 1 && t.high_max_ratio < t.low_min_ratio)) throw new Error("Runner ratios: the high-runner value must be at least 1 and below the low-runner value.");
    const c = settings().cycle_stock_policy;
    if (!(c.high_min_days >= 0 && c.medium_min_days > 0 && c.low_min_days > 0)) throw new Error("Cycle stock: runner minimums must be positive (the high-runner minimum may be 0).");
    if (!(c.medium_batch_factor > 0 && c.medium_batch_factor <= 1)) throw new Error("Cycle stock: the medium-runner share of one batch must be above 0 and at most 1.");
    if (!(c.max_days > 0)) throw new Error("Cycle stock: the maximum cycle stock must be positive.");
    const sapProblems = sapFormProblems(settings().sap_settings);
    if (sapProblems.length) throw new Error(`SAP settings: ${sapProblems.join(" ")}`);
    if (!input("profile-name").value.trim() || !input("profile-plant").value.trim()) throw new Error("Profile name and plant are required.");
    // Constraints are not sent: a PATCH keeps the stored ones; they change only through approved changes.
    const body = {name: input("profile-name").value.trim(), plant: input("profile-plant").value.trim(), settings: settings(), matrix_rows: rows};
    const saved = await request(current ? `/api/plant-profiles/${encodeURIComponent(current.profile_id)}` : "/api/plant-profiles", current ? "PATCH" : "POST", {...body, ...(current ? {revision: current.revision} : {})});
    await refresh(saved.profile_id);
    // The assistant follows the saved profile and refreshes its name and revision label.
    app.select({plant_profile_id: saved.profile_id});
    app.events.dispatchEvent(new CustomEvent("profile-saved", {detail: {profile_id: saved.profile_id}}));
  }, app.fail);
  action(input("profile-delete"), async () => {
    if (!current || !window.confirm(`Remove profile "${current.name}"? Historical runs keep their saved profile revision.`)) return;
    await request(`/api/plant-profiles/${encodeURIComponent(current.profile_id)}`, "DELETE", {revision: current.revision, confirmed: true});
    if (app.state.plant_profile_id === current.profile_id) app.select({plant_profile_id: null});
    await refresh();
  }, app.fail);
  // Blocked behind the loading indicator until defaults and saved profiles fill the form.
  // The AI model card has its own status line and loads independently.
  app.loading(request("/api/plant-profile-defaults").then(async (result) => {
    // Open the profile the workspace has selected (e.g. "Open Settings" from the SAP card).
    defaults = result; await refresh(app.state.plant_profile_id || "");
    // Arrived from Datasets "Open workspace" for a plant with no profile: prefill a new one.
    const pendingPlant = app.state.pending_profile_plant;
    if (live && pendingPlant) {
      app.select({pending_profile_plant: null});
      input("profile-plant").value = pendingPlant;
      input("profile-status").textContent = `No profile exists for plant ${pendingPlant}. Complete and save this new profile, then open the dataset in the workspace again.`;
      changed();
    }
    if (live) for (const control of loadingControls) if (!["profile-delete", "matrix-accept"].includes(control.id)) control.disabled = false;
  }), "Loading plant profiles…").catch(app.fail);
  return () => {live = false; app.events.removeEventListener("profile-changed", profileChanged);};
}
