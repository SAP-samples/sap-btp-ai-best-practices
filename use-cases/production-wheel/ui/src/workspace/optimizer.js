import { profileSummary } from "./settings.js";
import { fieldDescription } from "./vocabulary.js";
import Chart from "chart.js/auto";
import { runProgress } from "./progress.js";
import { mountConfigurationHelp } from "./help.js";
import { request, downloadExport } from "../services/api.js";
import { action, el, evidence, options, display, fieldLabel } from "./dom.js";
import { mergeDraft, isTerminal, datasetPlants } from "./state.js";
import { QueryTable } from "./table.js";
import { renderComparison } from "./comparison.js";

/** Mount the versioned manual optimizer draft, live runs, frontier and audit panels. */
export function mountOptimizer(target, app) {
  target.innerHTML = `<section class="page-heading"><div><span class="eyebrow">02 / OPTIMIZATION & REVIEW</span><h1>Optimizer workspace</h1><p>Configure a run. Explore trade-offs. Inspect the evidence behind every point.</p></div><span id="draft-revision" class="badge">No draft selected</span></section><div class="workspace-grid"><section class="card configuration"><div class="section-heading"><h2>Run configuration</h2><span class="step">1</span></div><label>Plant profile<select id="workspace-profile"><option value="">Select plant profile</option></select></label><p id="inherited-profile" class="muted"></p><label>Published dataset<select id="workspace-dataset"><option value="">Select dataset</option></select></label><label>Run title<input id="title" placeholder="Name this scenario"></label><div class="form-grid"><label>Production version<select id="pv"><option>FIXED</option><option>OPTIMIZED</option></select></label><label>Coverage objective<select id="coverage"><option>PARETO</option></select></label><label>Coverage basis<select id="basis"><option>BASE_GROUP</option><option>ADJUSTED_GROUP</option><option>WORST_FINI</option></select></label><label>Compatibility matrix<select id="matrix"><option>FLEXIBLE</option><option>HARD</option><option>DIAGNOSTIC</option><option>OFF</option></select></label><label>Global member cap<input type="number" min="7" step="1" id="cap" value="7"></label></div><label>Plant / SEFI scope (JSON)<textarea id="scope" rows="3" placeholder='[{"plant":"...","sefi":"..."}]'>[]</textarea></label><label>Additional scenario constraints (JSON)<textarea id="constraints" rows="3">[]</textarea></label><details><summary>Budget & advanced configuration</summary><label>Execution budget (JSON)<textarea id="budget" rows="8"></textarea></label><label>Complete optimizer config (JSON)<textarea id="advanced-config" rows="9"></textarea></label><p class="muted">Apply advanced JSON to the draft before making further changes in the quick controls.</p><ui5-button id="apply-advanced">Apply advanced config</ui5-button></details><p id="draft-state" class="muted" role="status">Select a published dataset to create a draft.</p><div class="actions"><ui5-button id="save-draft">Save draft</ui5-button><ui5-button id="validate-draft">Validate</ui5-button><ui5-button id="launch" design="Emphasized" disabled>Launch run</ui5-button></div><div id="validation"></div></section><section class="card run-panel"><div class="section-heading"><h2>Run history</h2><ui5-button id="refresh-runs" design="Transparent">Refresh</ui5-button></div><p class="muted">Click a selected run again to deselect it.</p><ui5-button id="deselect-run" disabled>Deselect run</ui5-button><div id="run-list" class="run-list"></div><div id="run-status" class="run-status"><p class="muted">Select or launch a run to see live progress.</p></div><div id="run-failure-diagnostics" class="run-diagnostics" hidden></div><ui5-button id="cancel-run" design="Negative" disabled>Cancel selected run</ui5-button><details id="run-matrix-evidence" hidden><summary>Frozen compatibility evidence</summary><p class="muted">Compatibility is frozen with the run. Load a deliberate, paginated page rather than sending the full matrix to chat.</p><div class="actions"><label>Pair status<select id="run-matrix-status"><option value="">All</option><option value="allowed">Allowed</option><option value="blocked">Blocked</option></select></label><ui5-button id="load-run-matrix">Load matrix evidence</ui5-button></div><p id="run-matrix-summary" class="muted"></p><div id="run-matrix-page" class="table-scroll"></div><div class="actions"><ui5-button id="run-matrix-previous" disabled>Previous</ui5-button><ui5-button id="run-matrix-next" disabled>Next</ui5-button></div></details><details><summary>Run metadata</summary><div id="run-metadata"></div></details></section></div><section class="card"><div class="section-heading"><div><span class="eyebrow">FRONTIER</span><h2>Coverage & changeover trade-offs</h2></div><ui5-button id="refresh-results">Refresh results</ui5-button></div><p class="muted">Each point is a candidate decision. Structural validation does not imply business acceptance. Restricted-pool and time-limited results are not global optimality proofs. Changeover is a within-group proxy: singletons score zero; initial and between-group setups are excluded.</p><div id="frontier-empty" class="empty">Run the optimizer to explore available frontier points.</div><div class="chart-wrap"><canvas id="frontier" aria-label="Frontier coverage versus changeover" role="img"></canvas></div><div id="point-table" class="table-scroll"></div></section><section class="card"><div class="section-heading"><div><span class="eyebrow">POINT EVIDENCE</span><h2 id="point-title">Select a frontier point</h2></div><div class="actions"><ui5-button id="export-wheel" disabled>Export production wheel XLSX</ui5-button><ui5-button id="export" disabled>Export current view CSV</ui5-button></div></div><p class="muted">Common eligible lines show shared production eligibility, not a scheduled line. Use the audit view to inspect validation and proof boundaries.</p><div id="point-evidence"></div></section><section class="card"><div class="section-heading"><h2>Compare decisions</h2><ui5-button id="compare">Compare points</ui5-button></div><div class="form-grid"><label>Left run<select id="left-run"></select></label><label>Left point<input id="left-point" type="number" min="1" value="1"></label><label>Right run<select id="right-run"></select></label><label>Right point<input id="right-point" type="number" min="1" value="1"></label></div><p class="muted">Comparison retains population changes, membership changes, PV changes and comparability caveats.</p><div id="comparison"></div></section>`;
  mountConfigurationHelp(target).catch(app.fail);
  let draft = null;
  let profiles = [];
  let datasets = [];
  let live = true;
  let timer = null;
  let chart = null;
  let points = [];
  let pointSort = { field: "point_index", direction: 1 };
  let validatedRevision = null;
  let draftSequence = 0;
  let advancedPending = false;
  let failureDiagnosticsRunId = null;
  let matrixOffset = 0;
  let matrixPage = null;
  const launchKeys = new Map();
  const input = (id) => target.querySelector(`#${id}`);
  const table = new QueryTable(input("point-evidence"), {
    context: (view) => ({
      run_id: app.state.run_id,
      ...(["block_audits", "global_audits"].includes(view)
        ? {}
        : { point_index: app.state.point_index }),
    }),
    views: [
      "groups",
      "members",
      "validation",
      "solutions",
      "block_audits",
      "global_audits",
      "matrix_pairs",
    ],
    fail: app.fail,
  });

  /** Reflect a server-authoritative draft and reset validation after any revision change. */
  function renderDraft(next) {
    if (draft?.draft_id === next.draft_id && draft.revision > next.revision)
      return;
    draft = next;
    validatedRevision = null;
    advancedPending = false;
    app.select({ draft_id: draft.draft_id, dataset_id: draft.dataset_id, plant_profile_id: draft.plant_profile_id || null });
    input("title").value = draft.title || "";
    input("workspace-profile").value = draft.plant_profile_id || "";
    input("inherited-profile").textContent = profileSummary(draft.plant_profile || draft.plant_profile_snapshot);
    for (const key of ["matrix", "cap"]) input(key).disabled = Boolean(draft.plant_profile_id);
    renderDatasets(true);
    const config = draft.request?.config || {};
    input("pv").value = config.pv_mode || "FIXED";
    input("coverage").value = config.coverage_mode || "PARETO";
    input("matrix").value = config.matrix_mode || "FLEXIBLE";
    input("basis").value = config.coverage_basis || "BASE_GROUP";
    input("cap").value =
      (config.group_size?.base_limit || 7) +
      (config.group_size?.max_excess || 0);
    input("scope").value = JSON.stringify(draft.request?.scope || [], null, 2);
    input("constraints").value = JSON.stringify(
      draft.request?.constraints || [],
      null,
      2,
    );
    input("budget").value = JSON.stringify(draft.budget || {}, null, 2);
    input("advanced-config").value = JSON.stringify(config, null, 2);
    input("workspace-dataset").value = draft.dataset_id;
    input("draft-revision").textContent = `Draft · revision ${draft.revision}`;
    input("draft-state").textContent =
      "Saved. Validate this revision before launching.";
    input("launch").disabled = true;
  }

  /** Read editable fields while preserving complete server-provided configuration. */
  function draftPatch() {
    if (!draft) throw new Error("Select a published dataset first.");
    return mergeDraft(
      draft,
      Object.fromEntries(
        [
          "title",
          "pv",
          "coverage",
          "basis",
          "matrix",
          "cap",
          "scope",
          "constraints",
          "budget",
        ].map((key) => [key, input(key).value]),
      ),
    );
  }

  /** Save against the current revision; conflicts refetch rather than overwrite a newer draft. */
  async function save() {
    if (advancedPending)
      throw new Error(
        "Apply the advanced configuration JSON before saving or validating.",
      );
    const patch = draftPatch();
    try {
      renderDraft(
        await request(
          `/api/run-drafts/${encodeURIComponent(draft.draft_id)}`,
          "PATCH",
          patch,
        ),
      );
    } catch (error) {
      if (error.message.startsWith("409")) {
        await reloadDraft();
        throw new Error(
          "The draft changed elsewhere. The latest revision is now loaded; review it before editing again.",
        );
      }
      throw error;
    }
  }

  /** Refetch draft changes made by the shared-context assistant, with stale-response protection. */
  async function reloadDraft() {
    if (!app.state.draft_id) return;
    const sequence = ++draftSequence;
    const id = app.state.draft_id;
    const next = await request(`/api/run-drafts/${encodeURIComponent(id)}`);
    if (live && sequence === draftSequence && id === app.state.draft_id)
      renderDraft(next);
  }

  /** Create a fresh draft when selecting a different published dataset. */
  async function chooseDataset() {
    const id = input("workspace-dataset").value;
    const profileId = input("workspace-profile").value;
    ++draftSequence;
    app.select({
      dataset_id: id || null,
      draft_id: null,
      run_id: null,
      point_index: null,
    });
    clearRunView();
    draft = null;
    validatedRevision = null;
    input("draft-revision").textContent = "No draft selected";
    input("draft-state").textContent = "Select a published dataset to create a draft.";
    input("validation").replaceChildren();
    input("launch").disabled = true;
    input("run-list").replaceChildren();
    options(input("left-run"), [], "");
    options(input("right-run"), [], "");
    if (!id) return;
    if (!profileId) {
      input("draft-state").textContent = "Select a plant profile to create a new draft. Existing runs remain available.";
      await refreshRuns();
      return;
    }
    const next = await request("/api/run-drafts", "POST", { dataset_id: id, plant_profile_id: profileId, title: input("title").value.trim() });
    if (!live || id !== app.state.dataset_id || profileId !== app.state.plant_profile_id) return;
    renderDraft(next);
    await refreshRuns();
  }

  /** Populate run history and comparison selectors from server state. */
  async function refreshRuns() {
    if (!app.state.dataset_id) return;
    const datasetId = app.state.dataset_id;
    const result = await request(
      `/api/runs?dataset_id=${encodeURIComponent(datasetId)}&include_removed=true`,
    );
    if (!live || datasetId !== app.state.dataset_id) return;
    if (app.state.run_id && !(result.items || []).some((item) => item.run_id === app.state.run_id)) await selectRun(null);
    input("run-list").replaceChildren();
    if (!result.items?.length)
      input("run-list").append(
        el("p", "No runs for this snapshot yet.", "empty"),
      );
    for (const run of result.items || []) {
      const button = el(
        "button",
        null,
        `dataset-choice${run.run_id === app.state.run_id ? " selected" : ""}`,
      );
      button.dataset.id = run.run_id;
      button.setAttribute("aria-pressed", String(run.run_id === app.state.run_id));
      button.append(
        el("strong", `${run.removed_at ? "Removed · " : ""}${run.title || run.status} · ${run.stage || "run"}`),
        el("span", run.run_id, "muted"),
      );
      button.addEventListener("click", () =>
        selectRun(app.state.run_id === run.run_id ? null : run.run_id).catch(app.fail),
      );
      const row = el("div", null, "history-entry");
      const remove = el("button", "Delete permanently", "history-action");
      remove.disabled = !isTerminal(run.status);
      remove.title = remove.disabled ? "Cancel or finish this run before deleting it." : "Permanently delete this run and all its results and artifacts.";
      action(remove, async () => {
        if (!window.confirm(`Permanently delete run "${run.run_id}" and all its results and artifacts? This cannot be undone.`)) return;
        await request(`/api/runs/${encodeURIComponent(run.run_id)}`, "DELETE");
        if (app.state.run_id === run.run_id) await selectRun(null);
        input("comparison").replaceChildren();
        await refreshRuns();
      }, app.fail);
      row.append(button, remove);
      input("run-list").append(row);
    }
    const entries = (result.items || []).map((run) => [
      run.run_id,
      `${run.run_id} · ${run.status}`,
    ]);
    options(
      input("left-run"),
      entries,
      input("left-run").value || app.state.run_id,
    );
    options(
      input("right-run"),
      entries,
      input("right-run").value || app.state.run_id,
    );
  }

  /** Render bounded checkpoint evidence as fields and sampled affected blocks. */
  function renderFailureDiagnostics(diagnostics) {
    const panel = input("run-failure-diagnostics");
    panel.hidden = false;
    const fields = [
      ["Failure type", (diagnostics.failure_types || []).join(", ") || "Not recorded"],
      ["Affected blocks", diagnostics.affected_block_count],
      ["Candidates in affected blocks", diagnostics.candidate_count],
      ["Checkpoint available", diagnostics.checkpoint_available ? "Yes" : "No"],
      ["Completion", diagnostics.complete == null ? "Not recorded" : diagnostics.complete ? "Complete" : "Incomplete"],
      ["Integrity", diagnostics.integrity_valid == null ? "Not recorded" : diagnostics.integrity_valid ? "Valid" : "Invalid"],
    ];
    const fieldList = el("dl", null, "diagnostic-fields");
    for (const [label, value] of fields) {
      const row = el("div");
      row.append(el("dt", label), el("dd", display(value)));
      fieldList.append(row);
    }
    panel.replaceChildren(
      el("h3", "Failure diagnostics"),
      diagnostics.public_error ? el("p", diagnostics.public_error) : el("p", "The saved checkpoint provides the evidence below."),
      fieldList,
    );
    if (!(diagnostics.affected_blocks || []).length) return;
    const table = el("table");
    const head = el("tr");
    for (const label of ["Plant", "SEFI", "Candidates", "Failure type", "Message"])
      head.append(el("th", label));
    table.append(el("thead")).append(head);
    const body = el("tbody");
    for (const block of diagnostics.affected_blocks) {
      const row = el("tr");
      for (const key of ["plant", "sefi", "candidate_count", "error_type", "error_message"])
        row.append(el("td", display(block[key])));
      body.append(row);
    }
    table.append(body);
    panel.append(table);
    if (diagnostics.affected_blocks_truncated)
      panel.append(el("p", "Only the first affected blocks are shown. Use scoped diagnostics or result queries for more evidence.", "muted"));
  }

  /** Fetch the bounded checkpoint failure view once for the selected failed run. */
  async function loadFailureDiagnostics(runId) {
    if (failureDiagnosticsRunId === runId) return;
    const diagnostics = await request(`/api/runs/${encodeURIComponent(runId)}/failure-diagnostics`);
    if (!live || app.state.run_id !== runId) return;
    failureDiagnosticsRunId = runId;
    renderFailureDiagnostics(diagnostics);
  }

  /** Render a deliberately loaded page of frozen compatibility pairs. */
  function renderMatrixPage(page) {
    matrixPage = page;
    const first = page.total ? page.offset + 1 : 0;
    const last = Math.min(page.offset + page.rows.length, page.total);
    input("run-matrix-summary").textContent = page.total
      ? `Showing ${first}–${last} of ${page.total} frozen compatibility pairs.`
      : "No frozen compatibility pairs match this filter.";
    const table = el("table");
    const head = el("tr");
    for (const label of ["Volume A", "Volume B", "Status"])
      head.append(el("th", label));
    table.append(el("thead")).append(head);
    const body = el("tbody");
    for (const rowValue of page.rows) {
      const row = el("tr");
      for (const key of ["volume_a", "volume_b", "status"])
        row.append(el("td", display(rowValue[key])));
      body.append(row);
    }
    table.append(body);
    input("run-matrix-page").replaceChildren(table);
    input("run-matrix-previous").disabled = page.offset === 0;
    input("run-matrix-next").disabled = page.offset + page.rows.length >= page.total;
  }

  /** Request one explicit, filtered compatibility evidence page for the active run. */
  async function loadMatrixPage(offset = 0) {
    const runId = app.state.run_id;
    if (!runId) return;
    const status = input("run-matrix-status").value;
    const query = new URLSearchParams({ offset: String(offset), limit: "50" });
    if (status) query.set("status", status);
    const page = await request(`/api/runs/${encodeURIComponent(runId)}/matrix?${query}`);
    if (!live || app.state.run_id !== runId) return;
    matrixOffset = page.offset;
    renderMatrixPage(page);
  }

  /** Load live progress and schedule bounded polling while the current run is active. */
  async function pollRun() {
    clearTimeout(timer);
    const id = app.state.run_id;
    if (!id || !live) return;
    const run = await request(`/api/runs/${encodeURIComponent(id)}`);
    if (!live || id !== app.state.run_id) return;
    input("deselect-run").disabled = false;
    const status = input("run-status");
    status.replaceChildren(
      el("span", run.status, "badge"),
      el("h3", run.stage || "Waiting for execution"),
    );
    const phase = runProgress(run);
    if (run.status === "queued") status.append(el("p", phase.text, "muted"));
    if (!isTerminal(run.status) && run.status !== "queued") {
      const progress = el("progress");
      progress.setAttribute("aria-label", "Optimizer run progress");
      if (phase.fraction != null) {
        progress.max = 1;
        progress.value = phase.fraction;
      }
      status.append(progress, el("p", phase.text, "muted"));
    }
    if (run.message || run.error)
      status.append(el("p", display(run.message || run.error)));
    if (run.configuration?.fixed_rules_text)
      status.append(el("p", `Fixed profile rules: ${run.configuration.fixed_rules_text}`));
    else if (run.configuration?.fixed_rule_ids?.length)
      status.append(el("p", `Fixed profile rules: ${run.configuration.fixed_rule_ids.join(", ")}`));
    evidence(input("run-metadata"), run);
    input("run-matrix-evidence").hidden = false;
    input("cancel-run").disabled = isTerminal(run.status);
    if (isTerminal(run.status)) {
      await refreshRuns();
      if (/completed|succeeded/i.test(run.status)) await loadResults();
      if (/failed/i.test(run.status)) await loadFailureDiagnostics(id);
    } else
      timer = setTimeout(
        () =>
          pollRun().catch((error) => {
            app.fail(error);
            if (live)
              timer = setTimeout(() => pollRun().catch(app.fail), 10000);
          }),
        4000,
      );
  }

  /** Select a run, invalidating previous point evidence and polling ownership. */
  function clearRunView() {
    clearTimeout(timer);
    points = [];
    failureDiagnosticsRunId = null;
    matrixOffset = 0;
    matrixPage = null;
    input("run-status").replaceChildren(el("p", "Select or launch a run to see live progress.", "muted"));
    input("run-metadata").replaceChildren();
    input("run-failure-diagnostics").replaceChildren();
    input("run-failure-diagnostics").hidden = true;
    input("run-matrix-evidence").hidden = true;
    input("run-matrix-summary").textContent = "";
    input("run-matrix-page").replaceChildren();
    input("run-matrix-previous").disabled = true;
    input("run-matrix-next").disabled = true;
    input("cancel-run").disabled = true;
    input("deselect-run").disabled = !app.state.run_id;
    input("comparison").replaceChildren();
    chart?.destroy();
    chart = null;
    input("point-table").replaceChildren();
    input("point-title").textContent = "Select a frontier point";
    input("export").disabled = true;
    input("export-wheel").disabled = true;
    input("frontier-empty").hidden = false;
    table.clear();
  }

  /** Switch run ownership before loading progress and results. */
  async function selectRun(id) {
    app.select({ run_id: id, point_index: null });
    clearRunView();
    for (const button of input("run-list").querySelectorAll(".dataset-choice")) {
      const selected = button.dataset.id === id;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    }
    if (id) await pollRun();
  }

  /** Select a frontier point and load its server-paginated group/member/audit evidence. */
  async function selectPoint(index) {
    app.select({ point_index: Number(index) });
    input("point-title").textContent =
      `Point ${index} · groups, members & audit`;
    input("export").disabled = false;
    input("export-wheel").disabled = false;
    await table.refresh(true);
  }

  /** Render sortable metrics without interpreting any server-provided markup. */
  function renderPoints() {
    const preferred = [
      "point_index",
      "demand_weighted_mean_coverage_days",
      "j_ch",
      "group_count",
      "singleton_group_count",
      "validation_status",
      "result_class",
      "candidate_pool_completeness",
      "status",
    ];
    const columns = preferred.filter((key) =>
      points.some((point) => point[key] != null),
    );
    const grid = el("table");
    const head = el("tr");
    for (const column of columns) {
      const th = el("th");
      th.title = fieldDescription(column);
      const button = el("button", fieldLabel(column), "sort-button");
      button.addEventListener("click", () => {
        pointSort = {
          field: column,
          direction: pointSort.field === column ? -pointSort.direction : 1,
        };
        renderPoints();
      });
      th.append(button);
      head.append(th);
    }
    const thead = el("thead");
    thead.append(head);
    grid.append(thead);
    const body = el("tbody");
    for (const point of [...points].sort(
      (a, b) =>
        (a[pointSort.field] !== "" &&
        a[pointSort.field] != null &&
        b[pointSort.field] != null &&
        Number.isFinite(Number(a[pointSort.field])) &&
        Number.isFinite(Number(b[pointSort.field]))
          ? Number(a[pointSort.field]) - Number(b[pointSort.field])
          : display(a[pointSort.field]).localeCompare(
              display(b[pointSort.field]),
            )) * pointSort.direction,
    )) {
      const row = el("tr");
      row.tabIndex = 0;
      for (const column of columns) {
        const cell = el("td");
        if (column === "point_index") {
          const button = el(
            "button",
            `Point ${point.point_index}`,
            "point-button",
          );
          button.addEventListener("click", (event) => {
            event.stopPropagation();
            selectPoint(point.point_index).catch(app.fail);
          });
          cell.append(button);
        } else cell.textContent = display(point[column]);
        row.append(cell);
      }
      row.addEventListener("click", () =>
        selectPoint(point.point_index).catch(app.fail),
      );
      row.addEventListener("keydown", (event) => {
        if (event.key === "Enter")
          selectPoint(point.point_index).catch(app.fail);
      });
      body.append(row);
    }
    grid.append(body);
    input("point-table").replaceChildren(grid);
  }

  /** Fetch materialized points and display available metric axes with explicit labels. */
  async function loadResults() {
    const id = app.state.run_id;
    if (!id) return;
    const result = await request(`/api/runs/${encodeURIComponent(id)}/results`);
    if (!live || id !== app.state.run_id) return;
    points = (result.points || []).map((point, index) => ({
      ...point,
      ...(point.metrics || {}),
      point_index: point.point_index ?? index + 1,
    }));
    input("frontier-empty").hidden = points.length > 0;
    renderPoints();
    chart?.destroy();
    chart = null;
    if (!points.length) return;
    const numeric = Object.keys(points[0]).filter(
      (key) =>
        points[0][key] != null &&
        points[0][key] !== "" &&
        Number.isFinite(Number(points[0][key])) &&
        key !== "point_index",
    );
    const x =
      ["j_ch", "changeover_hours", "group_count"].find((key) =>
        numeric.includes(key),
      ) || numeric[0];
    const y =
      [
        "demand_weighted_mean_coverage_days",
        "mean_coverage_days",
        "coverage_days",
        "j_cov",
      ].find((key) => numeric.includes(key)) ||
      numeric.find((key) => key !== x);
    if (x && y)
      chart = new Chart(input("frontier"), {
        type: "scatter",
        data: {
          datasets: [
            {
              label: "Frontier candidates",
              data: points.map((point) => ({
                x: Number(point[x]),
                y: Number(point[y]),
                point_index: point.point_index,
              })),
              backgroundColor: "#007d8a",
              pointRadius: 6,
              pointHoverRadius: 9,
            },
          ],
        },
        options: {
          maintainAspectRatio: false,
          scales: {
            x: {
              title: {
                display: true,
                text:
                  x === "j_ch"
                    ? "Changeover · recurring within-group FINI changes/week"
                    : x.replaceAll("_", " "),
              },
            },
            y: {
              title: {
                display: true,
                text:
                  y === "demand_weighted_mean_coverage_days"
                    ? "Demand-weighted mean coverage · days"
                    : y.replaceAll("_", " "),
              },
            },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (context) =>
                  `Point ${context.raw.point_index}: ${context.raw.x}, ${context.raw.y}`,
              },
            },
          },
          onClick: (_event, elements) => {
            if (elements.length)
              selectPoint(points[elements[0].index].point_index).catch(
                app.fail,
              );
          },
        },
      });
    const selected = points.find(
      (point) => Number(point.point_index) === Number(app.state.point_index),
    );
    await selectPoint(selected?.point_index ?? points[0].point_index);
  }

  /** Invalidate launch authority when the user changes any draft field. */
  function dirty() {
    validatedRevision = null;
    input("launch").disabled = true;
    input("draft-state").textContent =
      "Unsaved changes. Save and validate before launching.";
  }
  for (const key of [
    "title",
    "pv",
    "coverage",
    "basis",
    "matrix",
    "cap",
    "scope",
    "constraints",
    "budget",
    "advanced-config",
  ])
    input(key).addEventListener("input", dirty);
  input("advanced-config").addEventListener("input", () => {
    advancedPending = true;
  });
  input("workspace-dataset").addEventListener("change", () =>
    chooseDataset().catch(app.fail),
  );
  action(input("save-draft"), save, app.fail);
  action(
    input("apply-advanced"),
    async () => {
      const patch = draftPatch();
      patch.request.config = JSON.parse(input("advanced-config").value);
      renderDraft(
        await request(
          `/api/run-drafts/${encodeURIComponent(draft.draft_id)}`,
          "PATCH",
          patch,
        ),
      );
    },
    app.fail,
  );
  action(
    input("validate-draft"),
    async () => {
      await save();
      const revision = draft.revision;
      const id = draft.draft_id;
      input("draft-state").textContent =
        "Loading snapshot and validating the scoped request…";
      const validation = await request(
        `/api/run-drafts/${encodeURIComponent(id)}/validate`,
        "POST",
        {},
      );
      if (draft.draft_id !== id || draft.revision !== revision)
        throw new Error(
          "Draft changed during validation. Validate the latest revision.",
        );
      evidence(input("validation"), validation);
      validatedRevision = validation.valid ? draft.revision : null;
      input("launch").disabled = !validation.valid;
      input("draft-state").textContent = validation.valid
        ? "Structurally valid request. Business acceptance and proof remain separate."
        : "Resolve the validation errors before launching.";
    },
    app.fail,
  );
  action(
    input("launch"),
    async () => {
      if (validatedRevision !== draft?.revision)
        throw new Error("Validate the latest draft revision first.");
      const launchIdentity = `${draft.draft_id}:${draft.revision}`;
      if (!launchKeys.has(launchIdentity))
        launchKeys.set(launchIdentity, crypto.randomUUID());
      const run = await request("/api/runs", "POST", {
        draft_id: draft.draft_id,
        revision: draft.revision,
        idempotency_key: launchKeys.get(launchIdentity),
      });
      validatedRevision = null;
      await refreshRuns();
      await selectRun(run.run_id);
    },
    app.fail,
  );
  action(
    input("cancel-run"),
    async () => {
      if (app.state.run_id) {
        await request(
          `/api/runs/${encodeURIComponent(app.state.run_id)}/cancel`,
          "POST",
          {},
        );
        await pollRun();
      }
    },
    app.fail,
  );
  action(input("deselect-run"), () => selectRun(null), app.fail);
  action(input("refresh-runs"), refreshRuns, app.fail);
  action(input("refresh-results"), loadResults, app.fail);
  action(input("load-run-matrix"), () => loadMatrixPage(0), app.fail);
  action(
    input("run-matrix-previous"),
    () => loadMatrixPage(Math.max(0, matrixOffset - 50)),
    app.fail,
  );
  action(
    input("run-matrix-next"),
    () => loadMatrixPage(matrixOffset + 50),
    app.fail,
  );
  input("run-matrix-status").addEventListener("change", () => {
    matrixOffset = 0;
    matrixPage = null;
    input("run-matrix-page").replaceChildren();
    input("run-matrix-summary").textContent = "Choose Load matrix evidence to apply this filter.";
    input("run-matrix-previous").disabled = true;
    input("run-matrix-next").disabled = true;
  });
  input("run-matrix-evidence").addEventListener("toggle", () => {
    if (input("run-matrix-evidence").open && !matrixPage)
      loadMatrixPage().catch(app.fail);
  });
  action(input("export-wheel"), async () => {
    if (app.state.run_id && app.state.point_index != null)
      await downloadExport(`/api/runs/${encodeURIComponent(app.state.run_id)}/wheel-export?point_index=${app.state.point_index}`, `production-wheel-point-${app.state.point_index}.xlsx`);
  }, app.fail);
  action(
    input("export"),
    async () => {
      if (app.state.run_id && app.state.point_index != null)
        await downloadExport(
          `/api/runs/${encodeURIComponent(app.state.run_id)}/export?view=${encodeURIComponent(table.view.value)}${["block_audits", "global_audits"].includes(table.view.value) ? "" : `&point_index=${app.state.point_index}`}`,
        );
    },
    app.fail,
  );
  action(
    input("compare"),
    async () => {
      if (!input("left-run").value || !input("right-run").value)
        throw new Error("Select two runs to compare.");
      renderComparison(
        input("comparison"),
        await request("/api/compare", "POST", {
          left: {
            run_id: input("left-run").value,
            point_index: Number(input("left-point").value),
          },
          right: {
            run_id: input("right-run").value,
            point_index: Number(input("right-point").value),
          },
        }),
      );
    },
    app.fail,
  );

  /** Limit new draft choices to datasets containing the selected profile plant. */
  function renderDatasets(includeCurrent = false) {
    const profile = profiles.find((p) => p.profile_id === app.state.plant_profile_id);
    options(input("workspace-dataset"), [["", "Select a published dataset"], ...datasets
      .filter((dataset) => /published/i.test(dataset.status) && (!profile || datasetPlants(dataset).includes(String(profile.plant)) || (includeCurrent && dataset.dataset_id === app.state.dataset_id)))
      .map((dataset) => [dataset.dataset_id, dataset.name || dataset.dataset_id])], app.state.dataset_id);
  }
  input("workspace-profile").addEventListener("change", async () => {
    app.select({plant_profile_id: input("workspace-profile").value || null});
    input("inherited-profile").textContent = profileSummary(profiles.find((p) => p.profile_id === app.state.plant_profile_id));
    renderDatasets();
    chooseDataset().catch(app.fail);
  });

  /** Restore historical drafts and runs, requiring a profile only for new drafts. */
  async function initialize() {
    const [result, profileResult] = await Promise.all([request("/api/datasets"), request("/api/plant-profiles")]);
    if (!live) return;
    datasets = result.items || [];
    profiles = profileResult.profiles || [];
    options(input("workspace-profile"), [["", "Select a plant profile"], ...profiles.map((p) => [p.profile_id, `${p.name} · ${p.plant}`])], app.state.plant_profile_id);
    if (!input("workspace-profile").value && !app.state.draft_id) app.select({plant_profile_id: null});
    // A profile restored from localStorage may belong to another plant than the selected
    // dataset; clear it instead of posting a draft the API would reject with 422.
    const profile = profiles.find((p) => p.profile_id === app.state.plant_profile_id);
    const dataset = datasets.find((d) => d.dataset_id === app.state.dataset_id);
    if (!app.state.draft_id && profile && dataset && !datasetPlants(dataset).includes(String(profile.plant))) {
      app.select({plant_profile_id: null});
      input("workspace-profile").value = "";
      input("draft-state").textContent = `Plant profile "${profile.name}" is for plant ${profile.plant}, not this dataset. Select a matching plant profile.`;
    }
    renderDatasets(true);
    if (app.state.draft_id) await reloadDraft();
    else if (input("workspace-dataset").value && input("workspace-profile").value) {
      renderDraft(await request("/api/run-drafts", "POST", {dataset_id: app.state.dataset_id, plant_profile_id: input("workspace-profile").value, title: input("title").value.trim()}));
    }
    await refreshRuns();
    if (app.state.run_id) await pollRun();
  }
  const onDraft = () => reloadDraft().catch(app.fail);
  app.events.addEventListener("draft-changed", onDraft);
  /** Attach chat-created runs to the shared workspace without relaunching them. */
  const onRun = async () => {
    try {
      const id = app.state.run_id;
      await selectRun(id);
      await refreshRuns();
    } catch (error) { app.fail(error); }
  };
  app.events.addEventListener("run-created", onRun);
  initialize().catch(app.fail);
  return () => {
    live = false;
    clearTimeout(timer);
    chart?.destroy();
    app.events.removeEventListener("draft-changed", onDraft);
    app.events.removeEventListener("run-created", onRun);
  };
}
