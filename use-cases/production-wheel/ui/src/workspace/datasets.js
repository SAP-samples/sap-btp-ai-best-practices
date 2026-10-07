import { deleteEntity, request } from "../services/api.js";
import { action, el, evidence } from "./dom.js";
import { viewLabel } from "./vocabulary.js";
import { QueryTable } from "./table.js";
import { datasetPlants } from "./state.js";

/** Render dataset ingestion, quality review and explicit publish workflow. */
export function mountDatasets(target, app) {
  target.innerHTML = `<section class="page-heading"><div><span class="eyebrow">01 / DATA FOUNDATION</span><h1>Datasets</h1><p>Turn source workbooks into a traceable optimizer snapshot.</p></div><span class="badge">HANA-backed</span></section><div class="dataset-grid"><section class="card"><div class="section-heading"><h2>Import a snapshot</h2><span class="step">1</span></div><p class="muted">Upload a production workbook and an optional enrichment workbook. Review extraction evidence before publishing.</p><form id="upload-form"><label>Snapshot name<input name="name" placeholder="September production snapshot" required></label><label>Production workbook<input name="primary" type="file" accept=".xlsx,.xlsm" required></label><label>Enrichment workbook <span class="optional">Optional</span><input name="enrichment" type="file" accept=".xlsx,.xlsm"></label><details><summary>Extraction settings</summary><p class="muted">Defaults: 01W / 02W, 250 demand days, 50 productive weeks, canonical factor 0.90. Leave the overrides empty to preserve default provenance; explicitly supplied values are recorded as user input.</p><input type="hidden" name="settings" value="{}"><label>Override settings<textarea id="override-text" rows="4" placeholder="e.g. Use 240 demand days and also model 04W"></textarea></label><div class="actions"><ui5-button id="process-overrides" design="Emphasized">Process new rules</ui5-button><ui5-button id="revert-overrides">Revert to default</ui5-button></div><div id="override-result" role="status"></div></details><ui5-button id="upload" design="Emphasized">Upload and extract</ui5-button><progress id="upload-progress" hidden aria-label="Extracting workbook"></progress><p id="upload-status" role="status" class="muted"></p></form></section><section class="card"><div class="section-heading"><h2>Available snapshots</h2><ui5-button id="refresh-datasets" design="Transparent">Refresh</ui5-button></div><p class="muted">Load a snapshot to review it. Click it again or use Deselect to clear the review.</p><ui5-button id="deselect-dataset" disabled>Deselect snapshot</ui5-button><div id="dataset-list" class="dataset-list"></div></section></div><section class="card" id="dataset-review"><div class="section-heading"><div><span class="eyebrow">REVIEW & PUBLISH</span><h2 id="review-title">Select a snapshot</h2></div><div class="actions"><ui5-button id="publish" disabled design="Emphasized">Publish dataset</ui5-button><ui5-button id="use-dataset" disabled>Open workspace</ui5-button></div></div><p class="muted">Model eligibility, structural validity, business acceptance and proof are separate assessments. Common eligible lines are compatibility evidence; they are not a scheduled line assignment.</p><div id="dataset-summary"></div><details><summary>Extraction metadata & quality issues</summary><div id="dataset-evidence"></div></details><div id="dataset-table"></div></section>`;
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
    target.querySelector("#publish").disabled = true;
    target.querySelector("#use-dataset").disabled = true;
    target.querySelector("#deselect-dataset").disabled = false;
    markSelection();
    const dataset = await request(`/api/datasets/${encodeURIComponent(id)}`);
    if (!live || app.state.dataset_id !== id) return;
    loaded = dataset;
    target.querySelector("#review-title").textContent = dataset.name || id;
    evidence(target.querySelector("#dataset-evidence"), {
      metadata: dataset.metadata,
      issues: dataset.issues,
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
      const metric = el("div", null, "metric");
      metric.append(
        el("strong", value.toLocaleString()),
        el("span", viewLabel(key)),
      );
      metrics.append(metric);
    }
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

  // Free-text overrides: the LLM translation is stored in the hidden "settings"
  // field, so the upload request keeps sending the same JSON contract.
  const overrideText = target.querySelector("#override-text");
  const overrideResult = target.querySelector("#override-result");
  const settingsField = target.querySelector('input[name="settings"]');
  let processedText = "";
  /** Reset overrides to defaults and clear the interpretation shown to the planner. */
  const revertOverrides = () => {
    overrideText.value = "";
    processedText = "";
    settingsField.value = "{}";
    overrideResult.replaceChildren();
  };
  action(
    target.querySelector("#process-overrides"),
    async () => {
      const text = overrideText.value.trim();
      if (!text) return revertOverrides();
      overrideResult.replaceChildren(el("p", "Interpreting…", "muted"));
      let result;
      try {
        result = await request("/api/datasets/settings/interpret", "POST", { text });
      } catch (error) {
        overrideResult.replaceChildren();
        throw error;
      }
      if (!live) return;
      settingsField.value = JSON.stringify(result.overrides);
      processedText = text;
      const changes = el("ul");
      for (const [key, value] of Object.entries(result.overrides))
        changes.append(el("li", `${key}: ${Array.isArray(value) ? value.join(" / ") : value}`));
      if (!changes.childElementCount) changes.append(el("li", "No settings changed. Defaults kept."));
      overrideResult.replaceChildren(
        el("strong", result.clarification_required ? "Clarification needed" : "LLM interpretation"),
        el("p", result.interpretation),
        changes,
        ...result.warnings.map((warning) => el("p", warning, "muted")),
      );
    },
    app.fail,
  );
  action(target.querySelector("#revert-overrides"), async () => revertOverrides(), app.fail);

  action(
    target.querySelector("#upload"),
    async () => {
      const form = target.querySelector("#upload-form");
      if (!form.reportValidity()) return;
      if (overrideText.value.trim() !== processedText) {
        target.querySelector("#upload-status").textContent =
          "Override settings changed. Click Process new rules (or Revert to default) before uploading.";
        return;
      }
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
  refresh()
    .then(() => {
      if (app.state.dataset_id) return select(app.state.dataset_id);
    })
    .catch(app.fail);
  return () => {
    live = false;
  };
}
