import { deleteEntity, request } from "../services/api.js";
import { action, el, evidence } from "./dom.js";
import { viewLabel } from "./vocabulary.js";
import { QueryTable } from "./table.js";
import { datasetPlants } from "./state.js";
import { findingGroups, renderFindings, scopeNotes } from "./findings.js";
import { modeledCodes, modeledRowCount, settingLines } from "./import_settings.js";

/** Render dataset ingestion, quality review and explicit publish workflow. */
export function mountDatasets(target, app) {
  target.innerHTML = `<section class="page-heading"><div><span class="eyebrow">01 / DATA FOUNDATION</span><h1>Datasets</h1><p>Turn source workbooks into a traceable optimizer snapshot.</p></div><span class="badge">HANA-backed</span></section><div class="dataset-grid"><section class="card"><div class="section-heading"><h2>Import a snapshot</h2><span class="step">1</span></div><p class="muted">Upload a production workbook and an optional enrichment workbook. Review extraction evidence before publishing.</p><form id="upload-form"><label>Snapshot name<input name="name" placeholder="September production snapshot" required></label><label>Production workbook<input name="primary" type="file" accept=".xlsx,.xlsm" required></label><label>Enrichment workbook <span class="optional">Optional</span><input name="enrichment" type="file" accept=".xlsx,.xlsm"></label><details id="extraction-settings"><summary>Extraction settings</summary><p class="muted">The production frequencies are read from the chosen workbook. To change a setting, ask the Workspace Assistant: it proposes values for this workbook and you apply them on its card. Defaults keep default provenance; applied values are recorded as user input.</p><input type="hidden" name="settings" value="{}"><div id="frequency-inventory"><p class="muted">Choose a production workbook to see its production frequencies.</p></div><ul id="applied-settings"></ul><div class="actions"><ui5-button id="settings-assistant">Ask the Assistant</ui5-button><ui5-button id="revert-overrides">Revert to default</ui5-button></div><p id="settings-status" role="status" class="muted"></p></details><ui5-button id="upload" design="Emphasized">Upload and extract</ui5-button><progress id="upload-progress" hidden aria-label="Extracting workbook"></progress><p id="upload-status" role="status" class="muted"></p></form></section><section class="card"><div class="section-heading"><h2>Available snapshots</h2><ui5-button id="refresh-datasets" design="Transparent">Refresh</ui5-button></div><p class="muted">Load a snapshot to review it. Click it again or use Deselect to clear the review.</p><ui5-button id="deselect-dataset" disabled>Deselect snapshot</ui5-button><div id="dataset-list" class="dataset-list"></div></section></div><section class="card" id="dataset-review"><div class="section-heading"><div><span class="eyebrow">REVIEW & PUBLISH</span><h2 id="review-title">Select a snapshot</h2></div><div class="actions"><ui5-button id="publish" disabled design="Emphasized">Publish dataset</ui5-button><ui5-button id="use-dataset" disabled>Open workspace</ui5-button></div></div><p class="muted">Model eligibility, structural validity, business acceptance and proof are separate assessments. Common eligible lines are compatibility evidence; they are not a scheduled line assignment.</p><div id="dataset-summary"></div><div id="dataset-findings" class="findings"></div><details><summary>Technical metadata (JSON)</summary><div id="dataset-evidence"></div></details><div id="dataset-table"></div></section>`;
  const table = new QueryTable(target.querySelector("#dataset-table"), {
    context: () => ({ dataset_id: app.state.dataset_id }),
    views: [
      "fini_master",
      "fini_line_eligibility",
      "line_corrections",
      "validation_issues",
      "pallet_resolution",
      "production_versions",
    ],
    fail: app.fail,
  });
  // Count boxes double as view shortcuts: keep the pressed box in step with the table view.
  const markMetric = (view) => {
    for (const box of target.querySelectorAll("#dataset-summary .metric-button"))
      box.setAttribute("aria-pressed", String(box.dataset.view === view));
  };
  /** Open a view in the evidence table (from a count box or a finding) and mark its box. */
  const showView = (view, preset) => {
    markMetric(view);
    table.show(view, preset).catch(app.fail);
  };
  table.view.addEventListener("change", () => markMetric(table.view.value));
  let live = true;
  let inventorySequence = 0;
  // Last dataset record loaded into the review panel; its plant drives profile routing.
  let loaded = null;

  /** Clear snapshot ownership and all review content; no server data is changed. */
  function deselect() {
    app.select({ dataset_id: null, draft_id: null, run_id: null, point_index: null });
    loaded = null;
    table.clear();
    target.querySelector("#review-title").textContent = "Select a snapshot";
    target.querySelector("#dataset-summary").replaceChildren();
    target.querySelector("#dataset-evidence").replaceChildren();
    target.querySelector("#dataset-findings").replaceChildren();
    target.querySelector("#publish").disabled = true;
    target.querySelector("#publish").textContent = "Publish dataset";
    target.querySelector("#use-dataset").disabled = true;
    target.querySelector("#deselect-dataset").disabled = true;
    markSelection();
  }

  /** Reflect the selected snapshot immediately, without a catalog round trip. */
  function markSelection() {
    for (const button of target.querySelectorAll("#dataset-list .dataset-choice")) {
      const selected = button.dataset.id === app.state.dataset_id;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    }
  }

  /** Load a selected dataset and refresh evidence only while this page is mounted. */
  async function select(id) {
    // Callers run after a request (initial load, upload, publish). If the user has left
    // this page meanwhile, #main holds another page: change neither the DOM nor the
    // selection, which would also clear the draft and run the user is working on.
    if (!live) return;
    if (app.state.dataset_id !== id)
      app.select({
        dataset_id: id,
        draft_id: null,
        run_id: null,
        point_index: null,
      });
    table.clear();
    target.querySelector("#review-title").textContent = "Loading snapshot…";
    target.querySelector("#dataset-summary").replaceChildren();
    target.querySelector("#dataset-evidence").replaceChildren();
    target.querySelector("#dataset-findings").replaceChildren();
    target.querySelector("#publish").disabled = true;
    target.querySelector("#use-dataset").disabled = true;
    target.querySelector("#deselect-dataset").disabled = false;
    markSelection();
    const dataset = await request(`/api/datasets/${encodeURIComponent(id)}`);
    if (!live || app.state.dataset_id !== id) return;
    loaded = dataset;
    target.querySelector("#review-title").textContent = dataset.name || id;
    // Findings are shown in plain words; the JSON keeps provenance (settings, hashes, counts) for audit.
    // Scope notes list products left out by the extraction settings; each entry can be
    // explained by the Workspace Assistant, which receives this dataset as page context.
    renderFindings(
      target.querySelector("#dataset-findings"),
      findingGroups(dataset.issues),
      (link) => showView(link.view, link),
      { notes: scopeNotes(dataset.issues, dataset.metadata), onAsk: (question) => app.askAssistant?.(question) },
    );
    evidence(target.querySelector("#dataset-evidence"), {
      metadata: dataset.metadata,
      summary: dataset.summary,
    });
    const counts = dataset.metadata?.counts || dataset.summary || {};
    const metrics = target.querySelector("#dataset-summary");
    metrics.replaceChildren();
    metrics.className = "metric-grid";
    for (const [key, value] of [
      "fini_master",
      "production_versions",
      "fini_line_eligibility",
      "validation_issues",
      "line_corrections",
      "pallet_resolution",
    ]
      .filter((key) => typeof counts[key] === "number")
      .map((key) => [key, counts[key]])) {
      const metric = el("button", null, "metric metric-button");
      metric.type = "button";
      metric.dataset.view = key;
      metric.title = `Show ${viewLabel(key).toLowerCase()} in the table below`;
      metric.append(
        el("strong", value.toLocaleString()),
        el("span", viewLabel(key)),
      );
      metric.addEventListener("click", () => showView(key));
      metrics.append(metric);
    }
    markMetric(table.view.value);
    const published = /published/i.test(dataset.status) && !dataset.removed_at;
    target.querySelector("#publish").disabled = dataset.status !== "review" || Boolean(dataset.removed_at);
    target.querySelector("#publish").textContent = published
      ? "Published"
      : "Publish dataset";
    target.querySelector("#use-dataset").disabled = !published;
    await table.refresh(true);
  }

  /** Fetch the dataset inventory and render safe buttons with source status. */
  async function refresh() {
    const sequence = ++inventorySequence;
    const result = await request("/api/datasets?include_removed=true");
    if (!live || sequence !== inventorySequence) return;
    if (app.state.dataset_id && !(result.items || []).some((item) => item.dataset_id === app.state.dataset_id)) deselect();
    const list = target.querySelector("#dataset-list");
    list.replaceChildren();
    if (!result.items?.length)
      list.append(
        el("p", "No snapshots yet. Import a workbook to begin.", "empty"),
      );
    for (const dataset of result.items || []) {
      const button = el(
        "button",
        null,
        `dataset-choice${dataset.dataset_id === app.state.dataset_id ? " selected" : ""}`,
      );
      button.append(
        el("strong", dataset.name || dataset.dataset_id),
        el("span", `${dataset.removed_at ? "Removed · " : ""}${dataset.status} · ${dataset.dataset_id}`, "muted"),
      );
      button.dataset.id = dataset.dataset_id;
      button.setAttribute("aria-pressed", String(dataset.dataset_id === app.state.dataset_id));
      button.addEventListener("click", () => {
        if (app.state.dataset_id === dataset.dataset_id) deselect();
        else select(dataset.dataset_id).catch(app.fail);
      });
      const row = el("div", null, "history-entry");
      const remove = el("button", "Delete permanently", "history-action");
      remove.title = "Permanently delete this snapshot, source files, drafts and all associated runs/results.";
      action(remove, async () => {
        if (!window.confirm(`Permanently delete snapshot "${dataset.name || dataset.dataset_id}"? This deletes its source files, extracted data, drafts and ALL associated runs and results. This cannot be undone.`)) return;
        // The cascading HANA delete can take a while; show it is running and block repeat clicks.
        remove.disabled = true;
        remove.textContent = "Deleting…";
        try {
          await deleteEntity(`/api/datasets/${encodeURIComponent(dataset.dataset_id)}`);
        } finally {
          remove.disabled = false;
          remove.textContent = "Delete permanently";
        }
        if (app.state.dataset_id === dataset.dataset_id) deselect();
        await refresh();
      }, app.fail);
      row.append(button, remove);
      list.append(row);
    }
  }
  action(target.querySelector("#deselect-dataset"), async () => deselect(), app.fail);
  action(target.querySelector("#refresh-datasets"), refresh, app.fail);

  // Extraction settings: the workbook preview and the overrides applied from the
  // assistant's cards. The hidden "settings" field keeps the upload's JSON contract.
  const form = target.querySelector("#upload-form");
  const settingsField = target.querySelector('input[name="settings"]');
  const settingsStatus = target.querySelector("#settings-status");
  // Settings applied on this page visit; kept when another workbook is chosen.
  let overrides = {};
  // Identifies the latest workbook choice, so late previews and old cards are ignored.
  let previewId = null;
  app.pendingImport = null;

  /** Publish the current pending import to the chat and redraw the settings panel. */
  function publishPending(pending) {
    app.pendingImport = pending;
    app.events.dispatchEvent(new Event("pending-import"));
    renderSettings();
  }

  /** Show applied settings and the workbook's frequency codes, marking modeled ones. */
  function renderSettings() {
    settingsField.value = JSON.stringify(overrides);
    const inventory = app.pendingImport?.frequency_inventory || null;
    target.querySelector("#applied-settings").replaceChildren(
      ...settingLines(overrides, inventory).map((line) => el("li", line)),
    );
    const box = target.querySelector("#frequency-inventory");
    if (!inventory) return;
    const modeled = new Set(modeledCodes(overrides, inventory));
    const grid = el("table", null, "inventory-table");
    const head = el("tr");
    for (const title of ["Code", "Source label", "FINI rows", "In scope"]) head.append(el("th", title));
    grid.append(head);
    for (const item of inventory.codes) {
      const row = el("tr");
      row.append(el("td", item.code), el("td", item.label || ""), el("td", item.rows.toLocaleString()),
        el("td", modeled.has(item.code) ? "Yes" : "No"));
      grid.append(row);
    }
    for (const item of inventory.unrecognized) {
      const row = el("tr", null, "muted");
      row.append(el("td", "—"), el("td", item.value), el("td", item.rows.toLocaleString()),
        el("td", "No (no weekly code)"));
      grid.append(row);
    }
    const count = modeledRowCount(overrides, inventory);
    box.replaceChildren(
      el("p", `Production frequencies in ${app.pendingImport.primary_filename}: ${count.modeled.toLocaleString()} of ${count.total.toLocaleString()} FINI rows are in scope by frequency with the settings below. Data checks during extraction can still exclude some of them.`),
      grid,
    );
  }

  /** Read the chosen production workbook's frequencies; nothing is saved. */
  async function previewPrimary() {
    const file = form.elements.primary.files[0];
    const id = (previewId = crypto.randomUUID());
    const box = target.querySelector("#frequency-inventory");
    publishPending(null);
    if (!file) {
      box.replaceChildren(el("p", "Choose a production workbook to see its production frequencies.", "muted"));
      return;
    }
    box.replaceChildren(el("p", "Reading production frequencies…", "muted"));
    const data = new FormData();
    data.append("primary", file);
    let preview;
    try {
      preview = await request("/api/datasets/preview", "POST", data);
    } catch (error) {
      if (live && id === previewId)
        box.replaceChildren(el("p", `Could not read the workbook: ${error.message}`, "muted"));
      return;
    }
    if (!live || id !== previewId) return;
    publishPending({
      preview_id: id,
      primary_filename: preview.filename,
      enrichment_filename: form.elements.enrichment.files[0]?.name || null,
      plant: preview.plant ?? null,
      frequency_inventory: preview.frequency_inventory,
      overrides,
    });
  }
  form.elements.primary.addEventListener("change", () => previewPrimary().catch(app.fail));
  form.elements.enrichment.addEventListener("change", () => {
    if (app.pendingImport)
      publishPending({ ...app.pendingImport, enrichment_filename: form.elements.enrichment.files[0]?.name || null });
  });

  /** Replace the applied overrides and tell the chat which values are now active. */
  function setOverrides(next) {
    overrides = next;
    if (app.pendingImport) publishPending({ ...app.pendingImport, overrides });
    else renderSettings();
  }

  // Apply handler for the assistant's cards while this page is mounted.
  app.importSettings = {
    /** Apply a proposal made for the workbook that is still chosen; reject stale ones. */
    apply(proposal) {
      if (!live || !app.pendingImport || proposal.preview_id !== app.pendingImport.preview_id)
        throw new Error("This proposal was made for another workbook choice. Ask the Assistant again.");
      setOverrides(proposal.overrides);
      target.querySelector("#extraction-settings").open = true;
      settingsStatus.textContent = "Settings applied from the Workspace Assistant.";
    },
  };
  action(target.querySelector("#settings-assistant"), async () => app.openAssistant(), app.fail);
  action(target.querySelector("#revert-overrides"), async () => {
    setOverrides({});
    settingsStatus.textContent = "Defaults restored.";
  }, app.fail);
  renderSettings();

  action(
    target.querySelector("#upload"),
    async () => {
      if (!form.reportValidity()) return;
      const data = new FormData(form);
      if (!form.elements.enrichment.files.length) data.delete("enrichment");
      const progress = target.querySelector("#upload-progress");
      progress.hidden = false;
      target.querySelector("#upload-status").textContent =
        "Uploading and extracting workbook. This may take several minutes…";
      try {
        const dataset = await request("/api/datasets", "POST", data);
        if (!live) return;
        target.querySelector("#upload-status").textContent =
          "Extraction complete. Review quality evidence below.";
        await select(dataset.dataset_id);
        await refresh();
      } catch (error) {
        target.querySelector("#upload-status").textContent =
          `Import did not finish: ${error.message}`;
        throw error;
      } finally {
        progress.hidden = true;
      }
    },
    app.fail,
  );
  target
    .querySelector("#upload-form")
    .addEventListener("submit", (event) => event.preventDefault());
  action(
    target.querySelector("#publish"),
    async () => {
      const id = app.state.dataset_id;
      if (!id) return;
      await request(
        `/api/datasets/${encodeURIComponent(id)}/publish`,
        "POST",
        {},
      );
      if (!live) return;
      if (app.state.dataset_id === id) await select(id);
      await refresh();
    },
    app.fail,
  );
  action(
    target.querySelector("#use-dataset"),
    async () => {
      const plants = datasetPlants(loaded);
      const result = await request("/api/plant-profiles");
      if (!live) return;
      const matching = (result.profiles || []).filter((profile) => plants.includes(String(profile.plant)));
      if (!matching.length) {
        // No profile can accept this dataset: route to Settings with the plant prefilled.
        const plant = plants[0] || "";
        window.alert(`No plant profile exists for plant ${plant || "(unknown)"}. You will be taken to Settings to create one before this dataset can be used in the workspace.`);
        app.select({ pending_profile_plant: plant, plant_profile_id: null });
        location.hash = "settings";
        return;
      }
      // Keep a compatible profile; otherwise auto-pick the only match or let the user choose.
      if (!matching.some((profile) => profile.profile_id === app.state.plant_profile_id))
        app.select({ plant_profile_id: matching.length === 1 ? matching[0].profile_id : null });
      location.hash = "workspace";
    },
    app.fail,
  );
  // The page stays blocked behind the loading indicator until the snapshot list
  // and, when one was selected before, its review are on screen.
  app
    .loading(
      refresh().then(() => {
        if (app.state.dataset_id) return select(app.state.dataset_id);
      }),
      "Loading datasets…",
    )
    .catch(app.fail);
  return () => {
    live = false;
    // The file inputs disappear with the page, so the assistant must forget them too.
    delete app.importSettings;
    app.pendingImport = null;
    app.events.dispatchEvent(new Event("pending-import"));
  };
}
