/**
 * SAP settings panel for the run's selected decision scenario.
 *
 * Flow: choose SEFIs for runner-adjusted coverage (optional), enter the
 * planning period, generate, then review the per-tab status and the stored
 * rows, and download the XLSX laid out like the SAP upload template. The rules
 * (DC lot sizes, calendar profile codes, ...) come from the run's plant profile
 * (Settings > SAP settings) and are read when generating.
 * All data comes from the API; nothing is stored in the browser.
 */
import { request, downloadExport } from "../services/api.js";
import { action, el } from "./dom.js";
import { QueryTable } from "./table.js";
import { currentQuarterLabel, readinessLines, sourceLine } from "./sap_readiness.js";

/** Run-owned result views written by SAP settings generation. */
export const SAP_VIEWS = [
  "sap_dc_lot_sizes",
  "sap_planning_calendar",
  "sap_recipe_combinations",
  "sap_apo_lot_sizes",
  "sap_changeover_kpis",
  "sap_pv_conflicts",
  "sap_runner_adjustments",
];

const TAB_LABELS = {
  pv_check: "Check PV in subgroup",
  kpis: "RoC and change overs",
  cycle_stock: "Cycle stock",
  planning_calendar: "Planning calendar",
  recipe_combination: "Recipe combination",
  apo_lot_size: "APO lot size",
  central_dc_lot_size: "Central DC lot size",
  runner_audit: "Runner RoC audit",
};

const STATUS_LABELS = {
  ready: "Ready",
  partial: "Partial",
  blocked: "Blocked",
  not_applicable: "Not applicable",
};

/** Return one display row per generated tab, in template order. */
export function tabStatusRows(summary) {
  return Object.entries(summary?.tabs || {}).map(([name, tab]) => ({
    name,
    label: TAB_LABELS[name] || name,
    status: tab.status,
    statusLabel: STATUS_LABELS[tab.status] || tab.status,
    rows: tab.rows,
    message: tab.message || "",
  }));
}

/** Format the KPI cross-check line: template changeovers next to the optimizer's J_CH. */
export function kpiLine(summary) {
  if (!summary) return "";
  const value = (number) => (number == null ? "—" : Number(number).toFixed(2));
  const multiples = summary.runner_multiple_subgroups
    ? ` · ${summary.runner_multiple_subgroups} subgroups with runner multiples (single RoC would give ${value(summary.kpi_single_changeovers_per_week)} changeovers/week)`
    : "";
  // Both sides use the whole-day wheel RoC; a mismatch flags a run made before that rule.
  const check = summary.kpi_matches_point_j_ch === false ? ", differs: run made before the whole-day RoC rule" : "";
  return `Wheel RoC ${value(summary.wheel_roc_days)} days · ${value(summary.kpi_changeovers_per_week)} changeovers/week (optimizer: ${value(summary.point_j_ch)}${check}) · ${summary.runner_adjusted_fini ?? 0} FINIs with runner-adjusted RoC${multiples}`;
}

/** Describe the stored selection in one line. */
export function selectionLine(selection) {
  if (!selection) return "Select a decision scenario above to generate SAP settings.";
  const label = selection.decision_label || `Point ${selection.point_index}`;
  return `Selected scenario: ${label} (point ${selection.point_index}).`;
}

/**
 * Mount the panel into `container`.
 * The cycle stock strategy of every subgroup (single RoC or runner multiples)
 * comes from the selected point; the plant profile decides how it is chosen.
 * Before generating, the card shows which tabs would be blocked and how to fix
 * them (GET /api/runs/{id}/sap-settings/readiness), and reloads that check when
 * the plant profile changes (Settings save or an approved Assistant card).
 * Returns `{ load(runId, selection), clear(), generate(), readiness(), dispose() }`.
 */
export function mountSapSettings(container, app) {
  container.innerHTML = `<div class="section-heading"><div><span class="eyebrow">SAP SETTINGS</span><h2>Settings for the selected scenario</h2></div><ui5-button data-download disabled>Download SAP settings XLSX</ui5-button></div><p data-selection class="muted"></p><p class="muted">Runner-adjusted coverage follows the cycle stock strategy of each subgroup in the selected scenario (plant profile setting): subgroups with runner multiples get them here, the rest keep the coverage of the selected production version.</p><div class="form-grid"><label>Planning period<input data-period maxlength="32" placeholder="e.g. 26Q4"></label><div class="actions"><ui5-button data-generate design="Emphasized" disabled>Generate SAP settings</ui5-button></div></div><p data-source class="muted"></p><div data-inputs class="sap-inputs" role="status"></div><p data-kpi class="muted"></p><div data-tabs class="table-scroll"></div><div data-evidence></div><p class="muted">The rules (DC lot sizes, calendar profile codes, recipe characters) come from the run's plant profile: Settings &gt; SAP settings, or ask the Assistant.</p>`;
  const part = (name) => container.querySelector(`[data-${name}]`);
  let runId = null;
  let selection = null;
  let summary = null;
  let readiness = null;
  // Incremented by every load() and generate(); a slower earlier response is then ignored.
  let generation = 0;
  const evidence = new QueryTable(part("evidence"), {
    context: () => ({ run_id: runId, point_index: selection?.point_index }),
    views: SAP_VIEWS,
    fail: app.fail,
  });

  /** Show the generation summary: status per tab, KPI line, download state. */
  function renderSummary() {
    part("kpi").textContent = kpiLine(summary);
    part("download").disabled = !summary;
    const grid = el("table");
    const head = el("tr");
    for (const title of ["Tab", "Status", "Rows", "Message"]) head.append(el("th", title));
    const thead = el("thead");
    thead.append(head);
    const body = el("tbody");
    for (const row of tabStatusRows(summary)) {
      const tr = el("tr", null, `sap-status-${row.status}`);
      for (const value of [row.label, row.statusLabel, row.rows, row.message]) tr.append(el("td", value));
      body.append(tr);
    }
    grid.append(thead, body);
    part("tabs").replaceChildren(...(summary ? [grid] : []));
  }

  /** List the tabs generation would block, with the fix and shortcuts to fix it. */
  function renderInputs() {
    part("source").textContent = sourceLine(readiness);
    const lines = readinessLines(readiness);
    const box = part("inputs");
    box.replaceChildren();
    if (!lines.length) return;
    const list = el("ul");
    for (const line of lines) list.append(el("li", `${line.label}: ${line.status}. ${line.text} ${line.fix}`.trim()));
    const suggestion = (readiness.suggestions || []).find((item) => item.status === "suggested");
    if (suggestion) list.append(el("li", `Calendar codes found in an earlier wheel ("${suggestion.dataset_name}"): odd ${suggestion.values.calendar_odd_profile}, even ${suggestion.values.calendar_even_profile}. Save them in Settings > SAP settings ("Detect from dataset") or ask the Assistant.`));
    const controls = el("div", null, "actions");
    if (lines.some((line) => line.code === "profile_setting") && readiness.profile?.origin === "live") {
      const open = el("ui5-button", "Open Settings");
      action(open, () => {
        app.select({ plant_profile_id: readiness.profile.profile_id });
        location.hash = "#settings";
      }, app.fail);
      controls.append(open);
    }
    const ask = el("ui5-button", "Ask the Assistant");
    action(ask, () => app.askAssistant(`How do I fix the SAP settings tabs that are blocked for this run (${lines.map((line) => line.label).join(", ")})?`), app.fail);
    controls.append(ask);
    box.append(el("p", "Inputs to complete before generating:", "sap-inputs-title"), list, controls);
  }

  /** Re-check what generation would produce (selection point, else inputs only). */
  async function refreshReadiness() {
    if (!runId) return null;
    const id = runId;
    const result = await request(`/api/runs/${encodeURIComponent(id)}/sap-settings/readiness`);
    if (id !== runId) return readiness;
    readiness = result;
    renderInputs();
    return readiness;
  }

  /** Generate for the stored selection, then show the new summary and evidence. */
  async function generate() {
    if (!selection) throw new Error("Select a decision scenario before generating SAP settings.");
    if (!part("period").value.trim()) part("period").value = currentQuarterLabel(new Date());
    generation += 1;
    summary = await request(`/api/runs/${encodeURIComponent(runId)}/sap-settings`, "POST", {
      period_label: part("period").value.trim(),
    });
    renderSummary();
    await Promise.all([evidence.refresh(true), refreshReadiness()]);
    return summary;
  }

  /** Load the run's selection and stored summary; reset when nothing is selected. */
  async function load(nextRunId, nextSelection) {
    runId = nextRunId;
    selection = nextSelection || null;
    const token = ++generation;
    part("selection").textContent = selectionLine(selection);
    part("generate").disabled = !selection;
    const stored = selection
      ? (await request(`/api/runs/${encodeURIComponent(runId)}/sap-settings`)).sap_settings
      : null;
    if (token !== generation) return;
    summary = stored;
    if (summary?.period_label) part("period").value = summary.period_label;
    else if (!part("period").value.trim()) part("period").value = currentQuarterLabel(new Date());
    renderSummary();
    await refreshReadiness();
    if (summary) await evidence.refresh(true);
    else evidence.clear();
  }

  action(part("generate"), generate, app.fail);
  // Settings saves and approved Assistant cards change the live profile: re-check the inputs.
  const onProfile = () => refreshReadiness().catch(app.fail);
  app.events.addEventListener("profile-changed", onProfile);
  app.events.addEventListener("profile-saved", onProfile);
  action(
    part("download"),
    () => downloadExport(`/api/runs/${encodeURIComponent(runId)}/sap-settings/export`, `sap-settings-point-${selection.point_index}.xlsx`),
    app.fail,
  );
  return {
    load,
    generate,
    readiness: () => readiness,
    refreshReadiness,
    dispose() {
      app.events.removeEventListener("profile-changed", onProfile);
      app.events.removeEventListener("profile-saved", onProfile);
    },
    clear() {
      runId = null;
      selection = null;
      summary = null;
      readiness = null;
      renderInputs();
      part("selection").textContent = selectionLine(null);
      part("generate").disabled = true;
      renderSummary();
      evidence.clear();
    },
  };
}
