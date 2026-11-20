import {
  checkProviderStatus,
  deleteReferenceDataset,
  getAnalysisProgress,
  getPlantSettings,
  getProviderStatus,
  listReferenceDatasets,
  publishReferenceDataset,
  resetHistory,
  savePlantSettings,
} from "../../services/ecn-api.js";
import { initPlantMapping } from "./plant-mapping.js";
import { startAnalysisPolling } from "../../modules/analysis-polling.js";
import { createLatestRequestGate } from "../../modules/request-generation.js";

const defaultApi = {
  checkProviderStatus,
  deleteReferenceDataset,
  getAnalysisProgress,
  getPlantSettings,
  getProviderStatus,
  listReferenceDatasets,
  publishReferenceDataset,
  resetHistory,
  savePlantSettings,
};

const sourceDefinitions = Object.freeze([
  ["BOM", "EBOM and MBOM", "Engineering and manufacturing BOM workbook used for exact impact matching."],
  ["CONDITIONAL_RULES", "Conditional Configuration Rules", "Configuration applicability rules used to establish model and option scope."],
  ["STOCK", "Stock Information", "Reported unrestricted stock by material, plant and storage location; units and snapshot date may be unavailable."],
  ["MATERIAL_MASTER", "Material Master", "Material descriptions, groups, plant assignments, and ProcType procurement codes. Make/Buy status is read directly from this source."],
  ["PURCHASE_INFO_RECORDS", "Purchase Info Records", "Supplier and purchasing context retained as a versioned source."],
  ["PURCHASE_INFO_PRICES", "Purchase Info Prices", "Source-backed price overlay retained separately from purchasing setup."],
]);

const capabilityLabels = Object.freeze({
  oauth: "OAuth Authentication",
  text_generation: "Text Generation",
});

/** Convert a provider code into a concise user-facing status label. */
function statusLabel(value) {
  const labels = {
    NOT_CHECKED: "Not checked",
    AVAILABLE: "Available",
    CONFIGURATION_MISSING: "Configuration missing",
    AUTH_FAILED: "Authentication failed",
    FORBIDDEN: "Access forbidden",
    NOT_AVAILABLE: "Deployment unavailable",
    RATE_LIMITED: "Rate limited",
    TIMEOUT: "Timed out",
    PROVIDER_ERROR: "Provider error",
  };
  return labels[value] || "Unknown status";
}

/** Return a locale-aware timestamp or a clear missing value. */
function formatDate(value) {
  if (!value) return "Not checked yet";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "Not available";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(parsed);
}

/** Append one definition-list pair without interpreting raw source codes. */
function appendDefinition(list, term, value) {
  const termNode = document.createElement("dt");
  const valueNode = document.createElement("dd");
  termNode.textContent = term;
  valueNode.textContent = value;
  list.append(termNode, valueNode);
}

/** Render one OAuth or model availability card using safe diagnostic metadata only. */
function providerItem(title, model, status) {
  const card = document.createElement("article");
  card.className = "provider-status-item";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const modelName = document.createElement("p");
  modelName.textContent = model || "Credential check";
  const pill = document.createElement("span");
  pill.className = `status-pill status-${String(status.status).toLowerCase()}`;
  pill.textContent = statusLabel(status.status);
  card.append(heading, modelName, pill);
  const support = [
    status.http_status ? `HTTP ${status.http_status}` : null,
    status.provider_code ? `Code ${status.provider_code}` : null,
    status.request_id ? `Request ${status.request_id}` : null,
  ].filter(Boolean);
  if (support.length) {
    const metadata = document.createElement("small");
    metadata.textContent = support.join(" · ");
    card.append(metadata);
  }
  return card;
}

/** Render the cached or refreshed provider snapshot without endpoint or credential fields. */
function renderProvider(root, snapshot) {
  root.querySelector("#provider-checked-at").textContent = `Last checked: ${formatDate(snapshot.checked_at)}`;
  const host = root.querySelector("#provider-status");
  host.replaceChildren(providerItem(capabilityLabels.oauth, null, snapshot.oauth));
  for (const model of snapshot.models) {
    host.append(providerItem(
      capabilityLabels[model.capability] || model.capability,
      model.model,
      model,
    ));
  }
}

/** Return whether a selected file has the only source format accepted for this dataset type. */
function isAcceptedSourceFile(datasetType, sourceFile) {
  if (!(sourceFile instanceof File)) return false;
  const suffix = datasetType === "PURCHASE_INFO_PRICES" ? ".csv" : ".xlsx";
  return sourceFile.name.toLowerCase().endsWith(suffix);
}

/** Describe the source-file format without exposing source contents. */
function sourceFileLabel(datasetType) {
  return datasetType === "PURCHASE_INFO_PRICES" ? "CSV price overlay" : "XLSX workbook";
}

/** Create one publication row with its own accessible delete control. */
function publicationSummary(dataset, onDelete) {
  const summary = document.createElement("div");
  summary.className = `publication${dataset.active ? "" : " publication-inactive"}`;
  const heading = document.createElement("div");
  heading.className = "publication-heading";
  const file = document.createElement("strong");
  file.textContent = dataset.file_name;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "publication-delete";
  remove.textContent = "×";
  remove.setAttribute("aria-label", `Delete ${dataset.file_name}`);
  remove.title = `Delete ${dataset.file_name}`;
  remove.addEventListener("click", () => onDelete(dataset, remove));
  heading.append(file, remove);
  const metadata = document.createElement("span");
  metadata.textContent = `${dataset.row_count} rows · ${dataset.warning_count} validation warnings`;
  const provenance = document.createElement("span");
  provenance.textContent = `${formatDate(dataset.uploaded_at)} · Traceability hash ${dataset.sha256.slice(0, 12)}…`;
  provenance.title = dataset.sha256;
  const context = document.createElement("span");
  context.textContent = `Scope: ${dataset.context?.data_group || "customer"} · ${dataset.context?.model_range || "Model not supplied"} · ${dataset.context?.environment || "UNKNOWN"} · ${dataset.context?.snapshot_date || "Undated"}`;
  const activation = document.createElement("span");
  activation.textContent = dataset.active ? "Active source in this scope" : "Earlier publication";
  summary.append(heading, activation, metadata, context, provenance);
  return summary;
}

/** Build one source replacement form and its publication summaries. */
function sourceCard(definition, publications, onPublish, onDelete) {
  const [datasetType, title, description] = definition;
  const card = document.createElement("article");
  card.className = "source-card";
  card.dataset.datasetType = datasetType;
  const heading = document.createElement("h3");
  heading.textContent = title;
  const help = document.createElement("p");
  help.textContent = description;
  const active = document.createElement("div");
  active.className = "active-publications";
  const activeHeading = document.createElement("h4");
  activeHeading.textContent = "Published versions and scope";
  active.append(activeHeading);
  if (publications.length) publications.forEach((dataset) => active.append(publicationSummary(dataset, onDelete)));
  else {
    const missing = document.createElement("p");
    missing.className = "no-publication";
    missing.textContent = "No active publication.";
    active.append(missing);
  }

  const form = document.createElement("form");
  form.className = "source-upload-form";
  form.dataset.datasetType = datasetType;
  const chooseStep = document.createElement("fieldset");
  chooseStep.className = "source-upload-step";
  const chooseLegend = document.createElement("legend");
  chooseLegend.textContent = "1. Choose replacement";
  chooseStep.append(chooseLegend);
  const fileLabel = document.createElement("label");
  fileLabel.textContent = `Replacement ${sourceFileLabel(datasetType)}`;
  const file = document.createElement("input");
  file.type = "file";
  file.name = "source_file";
  file.accept = datasetType === "PURCHASE_INFO_PRICES"
    ? ".csv,text/csv"
    : ".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
  file.required = true;
  const fileHelp = document.createElement("small");
  fileHelp.id = `${datasetType.toLowerCase()}-file-help`;
  fileHelp.textContent = `Only a ${sourceFileLabel(datasetType)} can be validated for this source.`;
  file.setAttribute("aria-describedby", fileHelp.id);
  fileLabel.append(file);
  chooseStep.append(fileLabel, fileHelp);

  const scopeStep = document.createElement("fieldset");
  scopeStep.className = "source-upload-step";
  const scopeLegend = document.createElement("legend");
  scopeLegend.textContent = "2. Describe publication scope";
  scopeStep.append(scopeLegend);
  const scopeHelp = document.createElement("p");
  scopeHelp.className = "source-step-help";
  scopeHelp.textContent = "The active source is determined by this exact model range, data group, and environment.";
  scopeStep.append(scopeHelp);
  for (const [name, text, type, value] of [["model_range", "Model range", "text", ""], ["data_group", "Data group", "text", "customer"], ["snapshot_date", "Extraction date", "date", ""], ["comparison_dataset_id", "Comparison dataset ID (optional)", "text", ""]]) {
    const label = document.createElement("label");
    label.textContent = text;
    const input = document.createElement("input");
    input.name = name;
    input.type = type;
    input.value = value;
    label.append(input);
    scopeStep.append(label);
  }
  const envLabel = document.createElement("label");
  envLabel.textContent = "Source environment";
  const environment = document.createElement("select");
  environment.name = "environment";
  for (const value of ["UNKNOWN", "QA", "PRODUCTION", "SYNTHETIC"]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    environment.append(option);
  }
  envLabel.append(environment);
  scopeStep.append(envLabel);
  const coverageLabel = document.createElement("label");
  coverageLabel.className = "coverage-complete";
  coverageLabel.textContent = "Complete coverage for this model range";
  const coverage = document.createElement("input");
  coverage.type = "checkbox";
  coverage.name = "coverage_complete";
  coverageLabel.prepend(coverage);
  scopeStep.append(coverageLabel);

  const activationStep = document.createElement("fieldset");
  activationStep.className = "source-upload-step source-activation-step";
  const activationLegend = document.createElement("legend");
  activationLegend.textContent = "3. Validate and activate";
  const activationNote = document.createElement("p");
  activationNote.className = "source-activation-note";
  activationNote.textContent = "Validation runs before atomic activation. If validation fails, the current active source stays in place. A successful activation reports its warning count and is used only by future ECN imports in this scope.";
  activationStep.append(activationLegend, activationNote);
  const actions = document.createElement("div");
  actions.className = "source-form-actions";
  const submit = document.createElement("button");
  submit.type = "submit";
  submit.textContent = "Validate and activate";
  const status = document.createElement("p");
  status.className = "source-form-status";
  status.setAttribute("role", "status");
  actions.append(submit, status);
  activationStep.append(actions);
  form.append(chooseStep, scopeStep, activationStep);
  form.addEventListener("submit", (event) => onPublish(event, status, submit));
  card.append(heading, help, active, form);
  return card;
}

/** Build the Settings initializer with injectable IO for deterministic tests. */
export function createSettingsPage({ api = defaultApi, startPolling = startAnalysisPolling } = {}) {
  /** Initialize provider status, Settings datasets, and protected history reset. */
  return async function init(context) {
    const root = context.container;
    const activityReads = createLatestRequestGate(context.signal);
    const providerReads = createLatestRequestGate(context.signal);
    const datasetReads = createLatestRequestGate(context.signal);
    const publicationMutations = createLatestRequestGate(context.signal);
    const resetMutations = createLatestRequestGate(context.signal);
    const input = root.querySelector("#reset-confirmation");
    const resetButton = root.querySelector("#reset-history");
    const retryStatus = root.querySelector("#retry-status");
    const listeners = [];
    let idleKnown = false;
    let idle = false;
    let stopPolling = () => {};

    /** Register a page-owned event listener and retain its cleanup. */
    const on = (node, type, handler) => {
      node.addEventListener(type, handler);
      listeners.push(() => node.removeEventListener(type, handler));
    };

    /** Enable reset only for a known idle state and the exact phrase. */
    const updateEnabled = () => {
      resetButton.disabled = !idleKnown || !idle || input.value !== "RESET HISTORY";
    };

    /** Read current global activity through the status-only request lane. */
    const readStatus = async () => {
      const request = activityReads.next();
      const value = await api.getAnalysisProgress({ signal: request.signal });
      if (!request.isCurrent()) throw new DOMException("Superseded", "AbortError");
      return value;
    };

    /** Apply one known progress response to the reset gate. */
    const applyStatus = (value) => {
      idleKnown = true;
      idle = value.unfinished_count === 0;
      const blocked = value.blocked_count || 0;
      root.querySelector("#active-status").textContent = idle
        ? "No analyses are unfinished."
        : `${value.queued_count} waiting, ${blocked} waiting for AI, and ${value.running_count} running; reset is unavailable.`;
      retryStatus.hidden = true;
      root.querySelector("#reset-error").textContent = "";
      updateEnabled();
    };

    /** Disable reset whenever current activity cannot be proven. */
    const failClosed = (error) => {
      idleKnown = false;
      idle = false;
      resetButton.disabled = true;
      retryStatus.hidden = false;
      root.querySelector("#reset-error").textContent = error.message;
    };

    /** Poll serially until no unfinished analysis remains. */
    const beginPolling = () => {
      stopPolling();
      stopPolling = startPolling({
        load: readStatus,
        onUpdate: applyStatus,
        onError: failClosed,
        isFinished: (value) => value.unfinished_count === 0,
        signal: context.signal,
      });
    };

    /** Load and render immutable source metadata plus the raw-code legend. */
    const loadDatasets = async () => {
      const request = datasetReads.next();
      const response = await api.listReferenceDatasets({ signal: request.signal });
      if (!request.isCurrent()) throw new DOMException("Superseded", "AbortError");
      const host = root.querySelector("#source-cards");
      host.replaceChildren();
      for (const definition of sourceDefinitions) {
        const publications = response.items.filter((item) => item.dataset_type === definition[0]);
        host.append(sourceCard(definition, publications, async (event, status, submit) => {
          event.preventDefault();
          const form = event.currentTarget;
          const data = new FormData(form);
          const sourceFile = form.elements.source_file.files[0];
          if (!isAcceptedSourceFile(definition[0], sourceFile)) {
            form.elements.source_file.setAttribute("aria-invalid", "true");
            status.textContent = `Select a ${sourceFileLabel(definition[0])}.`;
            return;
          }
          form.elements.source_file.removeAttribute("aria-invalid");
          const request = publicationMutations.next();
          submit.disabled = true;
          form.setAttribute("aria-busy", "true");
          status.textContent = "Validating the selected source before activation…";
          try {
            const published = await api.publishReferenceDataset({
              datasetType: definition[0],
              sourceFile,
              context: { model_range: data.get("model_range") || null, data_group: data.get("data_group") || "customer", environment: data.get("environment"), snapshot_date: data.get("snapshot_date") || null, coverage_complete: data.get("coverage_complete") === "on", comparison_dataset_id: data.get("comparison_dataset_id") || null },
              signal: request.signal,
            });
            if (!request.isCurrent()) return;
            await loadDatasets();
            root.querySelector("#sources-error").textContent = `Validation completed: ${published.file_name} is now active for its stated scope with ${published.warning_count} warnings. Existing ECNs retain their pinned source versions.`;
          } catch (error) {
            if (error.name !== "AbortError" && request.isCurrent()) status.textContent = error.message;
          } finally {
            if (request.isCurrent()) {
              submit.disabled = false;
              form.removeAttribute("aria-busy");
            }
          }
        }, async (dataset, button) => {
          if (!window.confirm(`Delete ${dataset.file_name}? This removes its workbook and source rows. A previous version may become active.`)) return;
          const request = publicationMutations.next();
          button.disabled = true;
          root.querySelector("#sources-error").textContent = "";
          try {
            await api.deleteReferenceDataset(dataset.dataset_id, { signal: request.signal });
            if (!request.isCurrent()) return;
            await loadDatasets();
            root.querySelector("#sources-error").textContent = `${dataset.file_name} was deleted.`;
          } catch (error) {
            if (error.name !== "AbortError" && request.isCurrent()) {
              root.querySelector("#sources-error").textContent = error.message;
            }
          } finally {
            if (request.isCurrent()) button.disabled = false;
          }
        }));
      }
      root.querySelector("#legend-version").textContent = `(${response.status_legend_version})`;
      const legend = root.querySelector("#procurement-legend");
      legend.replaceChildren();
      for (const [code, label] of Object.entries(response.procurement_status_legend)) {
        appendDefinition(legend, code, label);
      }
    };

    /** Load either cached or freshly checked provider status through one request lane. */
    const loadProvider = async (refresh = false) => {
      const request = providerReads.next();
      const button = root.querySelector("#check-provider");
      button.disabled = true;
      root.querySelector("#provider-error").textContent = "";
      try {
        const snapshot = refresh
          ? await api.checkProviderStatus({ signal: request.signal })
          : await api.getProviderStatus({ signal: request.signal });
        if (request.isCurrent()) renderProvider(root, snapshot);
      } catch (error) {
        if (error.name !== "AbortError" && request.isCurrent()) {
          root.querySelector("#provider-error").textContent = error.message;
        }
      } finally {
        if (request.isCurrent()) button.disabled = false;
      }
    };

    on(root.querySelector("#check-provider"), "click", () => void loadProvider(true));
    on(input, "input", updateEnabled);
    on(retryStatus, "click", async () => {
      try {
        applyStatus(await readStatus());
      } catch (error) {
        if (error.name !== "AbortError") failClosed(error);
      }
    });
    on(resetButton, "click", async () => {
      const request = resetMutations.next();
      resetButton.disabled = true;
      try {
        const counts = await api.resetHistory(input.value, { signal: request.signal });
        if (!request.isCurrent()) return;
        sessionStorage.setItem("ecn-history-reset-summary", JSON.stringify(counts));
        context.navigate("/ecns");
      } catch (error) {
        if (!request.isCurrent() || error.name === "AbortError") return;
        root.querySelector("#reset-error").textContent = error.message;
        if (error.status === 409) {
          idleKnown = false;
          idle = false;
          beginPolling();
        }
        updateEnabled();
      }
    });

    await Promise.all([
      loadProvider(),
      loadDatasets().catch((error) => {
        if (error.name !== "AbortError") root.querySelector("#sources-error").textContent = error.message;
      }),
      initPlantMapping(root, { api, signal: context.signal }).catch((error) => {
        if (error.name !== "AbortError") root.querySelector("#plant-mapping-status").textContent = error.message;
      }),
    ]);
    const source = context.query?.get("source");
    if (!context.signal.aborted && sourceDefinitions.some(([type]) => type === source)) {
      const card = root.querySelector(`[data-dataset-type="${source}"]`);
      if (card) {
        card.tabIndex = -1;
        card.scrollIntoView?.({ block: "start" });
        card.focus({ preventScroll: true });
      }
    }
    try {
      const initial = await readStatus();
      applyStatus(initial);
      if (!idle) beginPolling();
    } catch (error) {
      if (error.name !== "AbortError") failClosed(error);
    }

    /** Stop all request lanes, polling, and top-level event listeners. */
    return () => {
      stopPolling();
      activityReads.dispose();
      providerReads.dispose();
      datasetReads.dispose();
      publicationMutations.dispose();
      resetMutations.dispose();
      listeners.forEach((remove) => remove());
    };
  };
}

export const init = createSettingsPage();
