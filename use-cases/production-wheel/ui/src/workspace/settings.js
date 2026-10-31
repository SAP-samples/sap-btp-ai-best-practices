import { request, downloadExport } from "../services/api.js";
import { action, el, options } from "./dom.js";
import { formatChatMarkdown } from "./chat.js";
import { mountConfigurationHelp } from "./help.js";

/** Describe frozen profile settings and rules without treating server text as markup. */
export function profileSummary(profile) {
  if (!profile) return "Legacy draft: no plant profile attached.";
  const s = profile.settings || {};
  return `${profile.name} · ${profile.plant} · revision ${profile.revision}. Horizon ${s.horizon_days} demand days, ${s.demand_days_per_week} days/week. High ≤ ${s.high_runner_threshold_days} days; low > ${s.high_runner_threshold_days} days. Runner basis: ${s.runner_basis}. Matrix: ${s.matrix_mode}; maximum group size: ${s.maximum_group_size}. Fixed rules: ${profile.rules_text || `${profile.rules?.length || 0} rules`}. Scenario constraints add to these fixed rules.`;
}

/** Mount editable HANA-backed plant profiles, explicit rule preview, and confirmed removal. */
export function mountSettings(target, app) {
  target.innerHTML = `<section class="page-heading"><div><span class="eyebrow">03 / PLANT SETTINGS</span><h1>Plant profiles</h1><p>Save fixed plant rules and compatibility settings. Submitted runs retain their profile revision.</p></div></section><section class="card configuration"><label>Saved profile<select id="profile-list"></select></label><ui5-button id="profile-new">New profile</ui5-button><div class="form-grid"><label>Profile name<input id="profile-name" required></label><label>Plant<input id="profile-plant" required></label><label>Horizon (demand days)<input id="horizon_days" type="number" min="1"></label><label>Demand days per week<input id="demand_days_per_week" type="number" min="1" max="7"></label><label>High-runner threshold (days)<input id="high_runner_threshold_days" type="number" min="0.000001" step="any"></label><label>Runner basis<select id="runner_basis"><option value="reference">Fixed source reference</option><option value="candidate_pv">Candidate PV (sensitivity)</option></select></label><label>Compatibility mode<select id="matrix_mode"><option>FLEXIBLE</option><option>HARD</option><option>DIAGNOSTIC</option><option>OFF</option></select></label><label>Maximum group size<input id="maximum_group_size" type="number" min="1"></label></div><p id="runner-definition" class="muted"></p><label>Fixed plant rules<textarea id="rules-text" rows="5" placeholder="Describe the rules that every run for this profile must follow."></textarea></label><ui5-button id="interpret-rules">Preview interpretation</ui5-button><p id="interpret-state" role="status"></p><div id="interpretation"></div><h2>Volume compatibility</h2><p>Download the current matrix, edit it in Excel, then upload it for review. Add or remove volumes on both axes. Alternatively, describe compatible volume families and the status of all other pairs in Fixed plant rules, then preview the interpretation.</p><p>Y: Compatible · AVOID: Discouraged · N: Forbidden. Accept replaces the entire matrix in this form. Save profile persists your accepted changes.</p><div class="actions"><ui5-button id="matrix-download">Download Excel template</ui5-button><ui5-button id="matrix-upload">Upload Excel</ui5-button><input id="matrix-file" type="file" accept=".xlsx" hidden aria-label="Compatibility Excel file"><ui5-button id="matrix-accept" disabled>Accept</ui5-button></div><p id="matrix-status" role="status"></p><details id="matrix-review"><summary>Review volume compatibility matrix</summary><div id="profile-matrix" class="table-scroll"></div></details><p id="profile-status" role="status"></p><div class="actions"><ui5-button id="profile-save" design="Emphasized">Save profile</ui5-button><ui5-button id="profile-delete" design="Negative" disabled>Remove profile</ui5-button></div></section><section class="card ai-model-settings"><div class="section-heading"><div><span class="eyebrow">AI MODEL SETTINGS</span><h2>AI model settings</h2></div></div><p class="muted">Choose the AI Core model for the workspace assistant and plant-rule interpretation. Changes apply to new LLM requests.</p><label>Model<select id="ai-model" disabled></select></label><p id="ai-model-status" role="status">Loading AI model settings…</p><ui5-button id="ai-model-save" design="Emphasized" disabled>Save AI model</ui5-button></section>`;
  mountConfigurationHelp(target, "/api/plant-profile-help").catch(app.fail);
  const input = (id) => target.querySelector(`#${id}`);
  const loadingControls = [...target.querySelectorAll(".configuration input, .configuration select, .configuration textarea, .configuration ui5-button")];
  for (const control of loadingControls) control.disabled = true;
  input("profile-status").textContent = "Loading plant profiles…";
  const keys = ["horizon_days", "demand_days_per_week", "high_runner_threshold_days", "runner_basis", "matrix_mode", "maximum_group_size"];
  let profiles = [], defaults, current = null, rows = [], interpreted = null, previewKey = null, live = true;
  let pendingMatrix = null, matrixRulesText = "";
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
    return Object.fromEntries(keys.map((key) => [key, ["runner_basis", "matrix_mode"].includes(key) ? input(key).value : Number(input(key).value)]));
  }
  /** Identify every input that can change rule interpretation. */
  function signature() { return JSON.stringify([current?.profile_id || "", input("rules-text").value.trim(), input("profile-plant").value.trim(), settings(), rows]); }
  /** Invalidate stale previews when source rules or fixed settings change. */
  function changed() {
    const notice = document.querySelector("#notice");
    if (notice) notice.hidden = true;
    input("runner-definition").textContent = `High: coverage ≤ ${input("high_runner_threshold_days").value} days. Low: coverage > ${input("high_runner_threshold_days").value} days.`;
    input("interpret-state").textContent = input("rules-text").value.trim() && signature() !== previewKey ? "Preview the updated rules before saving." : "";
  }
  /** Show a read-only matrix; all edits come from reviewed Excel or natural language. */
  function renderMatrix() {
    const shown = pendingMatrix?.rows || rows;
    const volumes = [...new Set(shown.map((r) => String(r.volume_a)))].sort((a,b) => Number(a)-Number(b));
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
      : `Current matrix: ${volumes.length} volumes, ${shown.length} pairs. Upload Excel or preview a natural-language policy to replace it.`;
    input("matrix-accept").disabled = !pendingMatrix;
  }
  /** Load an existing profile or server defaults without overwriting any saved profile. */
  function edit(profile) {
    current = profile;
    pendingMatrix = null; matrixRulesText = profile?.matrix_rules_text || "";
    input("profile-list").value = profile?.profile_id || "";
    input("profile-name").value = profile?.name || "";
    input("profile-plant").value = profile?.plant || "";
    for (const key of keys) input(key).value = (profile || defaults).settings[key];
    input("rules-text").value = profile?.rules_text || "";
    rows = structuredClone((profile || defaults).matrix_rows || []);
    renderMatrix();
    interpreted = profile?.rules || []; previewKey = signature();
    input("interpretation").replaceChildren();
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
  input("profile-list").addEventListener("change", () => edit(profiles.find((p) => p.profile_id === input("profile-list").value)));
  action(input("profile-new"), () => edit(null), app.fail);
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
  action(input("matrix-accept"), () => {
    if (!pendingMatrix) return;
    if (pendingMatrix.key !== signature()) throw new Error("Inputs changed. Upload or preview the matrix again before accepting.");
    rows = pendingMatrix.rows;
    matrixRulesText = pendingMatrix.source === "Natural-language preview" ? input("rules-text").value.trim() : "";
    previewKey = matrixRulesText ? signature() : null;
    pendingMatrix = null; renderMatrix(); changed();
    input("matrix-status").textContent = `Accepted ${new Set(rows.map((r) => r.volume_a)).size} volumes. Save profile to persist this matrix.`;
  }, app.fail);
  action(input("interpret-rules"), async () => {
    const key = signature();
    previewKey = null; interpreted = []; pendingMatrix = null; renderMatrix();
    input("interpret-state").textContent = "Interpreting plant rules…";
    let result;
    try {
      result = await request("/api/plant-profiles/interpret", "POST", {text: input("rules-text").value.trim(), plant: input("profile-plant").value.trim(), settings: settings(), volumes: [...new Set(rows.map((r) => r.volume_a))]});
    } catch (error) {
      // Replace the in-progress label; the global notice shows the error itself.
      if (live) input("interpret-state").textContent = "Interpretation failed. See the error above, then preview again.";
      throw error;
    }
    if (!live || key !== signature()) { changed(); return; }
    interpreted = result.clarification_required ? [] : result.rules;
    previewKey = result.clarification_required || result.matrix_rows ? null : key;
    pendingMatrix = result.matrix_rows && !result.clarification_required ? {rows: result.matrix_rows, source: "Natural-language preview", key} : null;
    renderMatrix();
    if (pendingMatrix) input("matrix-review").open = true;
    const preview = input("interpretation");
    const explanation = el("div", null, "rule-markdown");
    explanation.innerHTML = formatChatMarkdown(result.interpretation || "No additional rules interpreted.");
    preview.replaceChildren(el("h3", "Interpreted plant rules"), explanation);
    for (const warning of result.warnings || []) {
      const note = el("div", null, "rule-markdown muted");
      note.innerHTML = formatChatMarkdown(`**AI advisory:** ${warning}`);
      preview.append(note);
    }
    const runtime = el("details"), notes = el("ul");
    runtime.append(el("summary", "Optimizer definitions"), notes);
    for (const note of result.runtime_notes || []) notes.append(el("li", note));
    preview.append(runtime);
    const details = el("details"), ruleEvidence = el("div");
    details.append(el("summary", "Typed rule evidence"), ruleEvidence);
    ruleEvidence.append(el("pre", JSON.stringify(interpreted || [], null, 2), "evidence"));
    preview.append(details);
    input("interpret-state").textContent = result.clarification_required ? "Clarification required: revise the rules and preview again." : "Review the interpretation above, then save the profile.";
  }, app.fail);
  action(input("profile-save"), async () => {
    const text = input("rules-text").value.trim();
    if (pendingMatrix) throw new Error("Accept the proposed volume compatibility matrix before saving.");
    if (!(settings().high_runner_threshold_days > 0)) throw new Error("High-runner threshold must be a positive number.");
    if (!input("profile-name").value.trim() || !input("profile-plant").value.trim()) throw new Error("Profile name and plant are required.");
    if (text && signature() !== previewKey) throw new Error("Preview and review the updated interpretation before saving.");
    const body = {name: input("profile-name").value.trim(), plant: input("profile-plant").value.trim(), settings: settings(), rules_text: text, matrix_rules_text: matrixRulesText === text ? matrixRulesText : "", rules: text ? interpreted : (!current?.rules_text ? current?.rules || [] : []), matrix_rows: rows};
    const saved = await request(current ? `/api/plant-profiles/${encodeURIComponent(current.profile_id)}` : "/api/plant-profiles", current ? "PATCH" : "POST", {...body, ...(current ? {revision: current.revision} : {})});
    await refresh(saved.profile_id);
  }, app.fail);
  action(input("profile-delete"), async () => {
    if (!current || !window.confirm(`Remove profile "${current.name}"? Historical runs keep their saved profile revision.`)) return;
    await request(`/api/plant-profiles/${encodeURIComponent(current.profile_id)}`, "DELETE", {revision: current.revision, confirmed: true});
    if (app.state.plant_profile_id === current.profile_id) app.select({plant_profile_id: null});
    await refresh();
  }, app.fail);
  request("/api/plant-profile-defaults").then(async (result) => {
    defaults = result; await refresh();
    // Arrived from Datasets "Open workspace" for a plant with no profile: prefill a new one.
    const pendingPlant = app.state.pending_profile_plant;
    if (live && pendingPlant) {
      app.select({pending_profile_plant: null});
      input("profile-plant").value = pendingPlant;
      input("profile-status").textContent = `No profile exists for plant ${pendingPlant}. Complete and save this new profile, then open the dataset in the workspace again.`;
      changed();
    }
    if (live) for (const control of loadingControls) if (!["profile-delete", "matrix-accept"].includes(control.id)) control.disabled = false;
  }).catch(app.fail);
  return () => {live = false;};
}
