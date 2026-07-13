/* UI5 Web Components for Optimizer Page */
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/FileUploader.js";
import "@ui5/webcomponents/dist/DatePicker.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Table.js";
import "@ui5/webcomponents/dist/TableRow.js";
import "@ui5/webcomponents/dist/TableCell.js";
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/SegmentedButton.js";
import "@ui5/webcomponents/dist/SegmentedButtonItem.js";
import "@ui5/webcomponents/dist/Switch.js";
import "@ui5/webcomponents/dist/Popover.js";
import "@ui5/webcomponents/dist/Icon.js";

import { request, uploadFile, API_BASE_URL, API_KEY } from "../../services/api.js";
import Chart from "chart.js/auto";

let appState = {
  currentProcessId: null,
  processes: [],
  results: null,
  limits: null,
  rules: null,
  params: null,
  charts: {},
  pollingInterval: null,
  availableCohorts: [],
  limitsEditMode: "automatic",
  planningMode: "single_week",
  selectedPage: 0,
  excludedPage: 0,
  pageSize: 50,
  selectedFile: null,
  weeklyExposureEntityTypeFilter: "all",
  optimizerExcludedCache: {},
  lastProgress: null,
  processStartedAtMs: null,
  lastElapsedSeconds: 0,
};

const AUTOMATIC_LIMIT_DEFAULTS = {
  alpha: 0.85,
  beta: 0.15,
  gamma: 0.30,
};

const MANUAL_LIMIT_DEFAULTS = {
  facility_limits_by_company_code: {
    "1000": 300000,
  },
  customer_limits: {
    "100456043": 50000,
  },
  group_limits: {
    "GROUP_A": 90000,
  },
  customer_to_group: {
    "CUSTAI02": "GRP_2410",
    "CUSTAI01": "GRP_2410",
  },
  base_exposure: {
    facility: {
      "2410": 72500,
    },
    customer: {
      "CUSTAI02": 49500,
      "CUSTAI01": 23000,
    },
    group: {
      "GRP_2410": 72500,
    },
  },
};

const WEEK_IN_MS = 7 * 24 * 60 * 60 * 1000;
const EUR_CURRENCY_FORMATTER = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "EUR",
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});

function formatRuleLabel(value) {
  return String(value || "").replace(/_/g, " ");
}

function normalizeCohortDate(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";

  const isoDatePrefix = raw.match(/^(\d{4}-\d{2}-\d{2})/);
  if (isoDatePrefix) return isoDatePrefix[1];

  return raw;
}

function formatManualLimitEntries(value) {
  const entries = Object.entries(value || {});
  if (!entries.length) return "";
  return entries
    .map(([key, amount]) => `"${key}": ${JSON.stringify(amount)}`)
    .join(",\n");
}

function parseManualLimitEntries(rawValue, label) {
  const text = String(rawValue || "").trim();
  if (!text) return {};

  const parseCandidates = [];
  parseCandidates.push(text);

  const hasObjectBraces = text.startsWith("{") && text.endsWith("}");
  if (!hasObjectBraces) {
    parseCandidates.push(`{${text}}`);

    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    if (lines.length > 1) {
      const commaJoinedLines = lines
        .map((line, index) => {
          const withoutTrailingComma = line.replace(/,\s*$/, "");
          return index < lines.length - 1
            ? `${withoutTrailingComma},`
            : withoutTrailingComma;
        })
        .join("\n");
      parseCandidates.push(`{${commaJoinedLines}}`);
    }
  }

  for (const candidate of parseCandidates) {
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed;
      }
    } catch (e) {
      // Try next candidate.
    }
  }

  throw new Error(`Invalid ${label}. Use JSON object content like "100821711": 75000.00`);
}

function parseYamlKey(rawKey) {
  const key = String(rawKey || "").trim();
  if (!key) {
    throw new Error("Invalid YAML key.");
  }

  if (key.startsWith("\"") && key.endsWith("\"")) {
    return JSON.parse(key);
  }

  if (key.startsWith("'") && key.endsWith("'")) {
    return key.slice(1, -1).replace(/\\'/g, "'");
  }

  return key;
}

function parseYamlScalar(rawValue) {
  const value = String(rawValue || "").trim();
  if (value === "{}") return {};
  if (value === "[]") return [];
  if (value.startsWith("\"") && value.endsWith("\"")) return JSON.parse(value);
  if (value.startsWith("'") && value.endsWith("'")) return value.slice(1, -1).replace(/\\'/g, "'");
  if (/^(true|false)$/i.test(value)) return value.toLowerCase() === "true";
  if (/^null$/i.test(value)) return null;
  if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(value)) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric;
  }
  return value;
}

function parseSimpleYamlObject(rawText) {
  const root = {};
  const stack = [{ indent: -1, value: root }];
  const lines = String(rawText || "").replace(/\r/g, "").split("\n");

  lines.forEach((rawLine, index) => {
    const lineNumber = index + 1;
    if (!rawLine.trim() || rawLine.trim().startsWith("#")) return;
    if (rawLine.includes("\t")) {
      throw new Error(`Invalid YAML line ${lineNumber}: tabs are not supported.`);
    }

    const indent = rawLine.match(/^ */)?.[0].length || 0;
    const trimmed = rawLine.trim();
    const separatorIndex = trimmed.indexOf(":");
    if (separatorIndex < 1) {
      throw new Error(`Invalid YAML line ${lineNumber}: expected key:value entry.`);
    }

    const rawKey = trimmed.slice(0, separatorIndex);
    const rawValue = trimmed.slice(separatorIndex + 1);
    const key = parseYamlKey(rawKey);

    while (stack.length > 1 && indent <= stack[stack.length - 1].indent) {
      stack.pop();
    }

    const parent = stack[stack.length - 1]?.value;
    if (!parent || typeof parent !== "object" || Array.isArray(parent)) {
      throw new Error(`Invalid YAML structure near line ${lineNumber}.`);
    }

    if (rawValue.trim() === "") {
      const nestedObject = {};
      parent[key] = nestedObject;
      stack.push({ indent, value: nestedObject });
      return;
    }

    parent[key] = parseYamlScalar(rawValue);
  });

  return root;
}

function parseManualLimitsConfigText(rawText) {
  const text = String(rawText || "").trim();
  if (!text) {
    throw new Error("Uploaded file is empty.");
  }

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (jsonError) {
    parsed = parseSimpleYamlObject(text);
  }

  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("File must contain a top-level object.");
  }

  return parsed;
}

function coerceObjectSection(value, label) {
  if (value == null) return {};
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`"${label}" must be an object.`);
  }
  return value;
}

function applyManualLimitsConfig(limitsConfig) {
  const facilityLimits = coerceObjectSection(
    limitsConfig.facility_limits_by_company_code,
    "facility_limits_by_company_code"
  );
  const customerLimits = coerceObjectSection(limitsConfig.customer_limits, "customer_limits");
  const groupLimits = coerceObjectSection(limitsConfig.group_limits, "group_limits");
  const customerToGroup = coerceObjectSection(limitsConfig.customer_to_group, "customer_to_group");
  const baseExposure = coerceObjectSection(limitsConfig.base_exposure, "base_exposure");
  const baseFacility = coerceObjectSection(baseExposure.facility, "base_exposure.facility");
  const baseCustomer = coerceObjectSection(baseExposure.customer, "base_exposure.customer");
  const baseGroup = coerceObjectSection(baseExposure.group, "base_exposure.group");

  const facilityEl = document.getElementById("opt-facility-limits");
  const customerEl = document.getElementById("opt-customer-limits");
  const groupEl = document.getElementById("opt-group-limits");
  const customerGroupMapEl = document.getElementById("opt-customer-group-map");
  const baseFacilityEl = document.getElementById("opt-base-facility");
  const baseCustomerEl = document.getElementById("opt-base-customer");
  const baseGroupEl = document.getElementById("opt-base-group");

  if (facilityEl) facilityEl.value = formatManualLimitEntries(facilityLimits);
  if (customerEl) customerEl.value = formatManualLimitEntries(customerLimits);
  if (groupEl) groupEl.value = formatManualLimitEntries(groupLimits);
  if (customerGroupMapEl) customerGroupMapEl.value = formatManualLimitEntries(customerToGroup);
  if (baseFacilityEl) baseFacilityEl.value = formatManualLimitEntries(baseFacility);
  if (baseCustomerEl) baseCustomerEl.value = formatManualLimitEntries(baseCustomer);
  if (baseGroupEl) baseGroupEl.value = formatManualLimitEntries(baseGroup);

  appState.limits = {
    ...(appState.limits || {}),
    facility_limits_by_company_code: facilityLimits,
    customer_limits: customerLimits,
    group_limits: groupLimits,
    customer_to_group: customerToGroup,
    base_exposure: {
      facility: baseFacility,
      customer: baseCustomer,
      group: baseGroup,
    },
    synthetic_generation: {
      ...(appState.limits?.synthetic_generation || {}),
      enabled: false,
    },
  };
}

export default function initOptimizerPage() {
  console.log("Optimizer page initialized");
  setupEventListeners();
  togglePlanningMode(appState.planningMode);
  loadProcessList();
}

function setupEventListeners() {
  const fileUploader = document.getElementById("opt-file-uploader");
  if (fileUploader) {
    fileUploader.addEventListener("change", handleFileUpload);
  }

  const createBtn = document.getElementById("opt-create-btn");
  if (createBtn) {
    createBtn.addEventListener("click", createProcess);
  }

  const saveConfigBtn = document.getElementById("save-config-btn");
  if (saveConfigBtn) {
    saveConfigBtn.addEventListener("click", saveConfiguration);
  }

  const runBtn = document.getElementById("run-optimization-btn");
  if (runBtn) {
    runBtn.addEventListener("click", startOptimization);
  }

  const limitsModeToggle = document.getElementById("limits-mode-toggle");
  if (limitsModeToggle) {
    const applyLimitsModeFromItem = (item) => {
      if (!item) return;
      const mode = item.textContent.trim().toLowerCase();
      toggleLimitsMode(mode);
    };

    limitsModeToggle.addEventListener("selection-change", (e) => {
      const selectedItem = e.detail?.selectedItem
        || e.detail?.targetItem
        || e.detail?.selectedItems?.[0];
      if (selectedItem) {
        applyLimitsModeFromItem(selectedItem);
        return;
      }
      const pressedByProp = Array.from(limitsModeToggle.querySelectorAll("ui5-segmented-button-item"))
        .find((item) => item.pressed === true);
      applyLimitsModeFromItem(pressedByProp);
    });

    limitsModeToggle.querySelectorAll("ui5-segmented-button-item").forEach((item) => {
      item.addEventListener("click", () => applyLimitsModeFromItem(item));
    });
  }

  const manualYamlUploadBtn = document.getElementById("opt-manual-yaml-upload-btn");
  const manualYamlUploadInput = document.getElementById("opt-manual-yaml-upload-input");
  if (manualYamlUploadBtn && manualYamlUploadInput) {
    manualYamlUploadBtn.addEventListener("click", () => manualYamlUploadInput.click());
  }
  if (manualYamlUploadInput) {
    manualYamlUploadInput.addEventListener("change", handleManualLimitsYamlUpload);
  }

  const planningModeToggle = document.getElementById("planning-mode-toggle");
  if (planningModeToggle) {
    const applyPlanningModeFromItem = (item) => {
      const mode = item?.dataset?.mode || item?.getAttribute("data-mode");
      if (mode === "single_week" || mode === "multi_week") {
        togglePlanningMode(mode);
      }
    };

    planningModeToggle.addEventListener("selection-change", (e) => {
      const selectedItem = e.detail?.selectedItem
        || e.detail?.targetItem
        || e.detail?.selectedItems?.[0];
      if (selectedItem) {
        applyPlanningModeFromItem(selectedItem);
        return;
      }
      const pressedByProp = Array.from(planningModeToggle.querySelectorAll("ui5-segmented-button-item"))
        .find((item) => item.pressed === true);
      applyPlanningModeFromItem(pressedByProp);
    });

    planningModeToggle.querySelectorAll("ui5-segmented-button-item").forEach((item) => {
      item.addEventListener("click", () => applyPlanningModeFromItem(item));
    });
  }

  const weeklyExposureEntityTypeToggle = document.getElementById("weekly-exposure-entity-type-toggle");
  if (weeklyExposureEntityTypeToggle) {
    const applyEntityTypeFilterFromItem = (item) => {
      const entityType = String(item?.dataset?.entityType || item?.getAttribute("data-entity-type") || "all")
        .trim()
        .toLowerCase();
      appState.weeklyExposureEntityTypeFilter = entityType || "all";
      if (appState.results) renderWeeklyTables(appState.results);
    };

    weeklyExposureEntityTypeToggle.addEventListener("selection-change", (e) => {
      const selectedItem = e.detail?.selectedItem
        || e.detail?.targetItem
        || e.detail?.selectedItems?.[0];
      if (selectedItem) {
        applyEntityTypeFilterFromItem(selectedItem);
        return;
      }
      const pressedByProp = Array.from(weeklyExposureEntityTypeToggle.querySelectorAll("ui5-segmented-button-item"))
        .find((item) => item.pressed === true);
      applyEntityTypeFilterFromItem(pressedByProp);
    });

    weeklyExposureEntityTypeToggle.querySelectorAll("ui5-segmented-button-item").forEach((item) => {
      item.addEventListener("click", () => applyEntityTypeFilterFromItem(item));
    });
  }

  // Pagination
  document.getElementById("selected-prev-btn")?.addEventListener("click", () => {
    if (appState.selectedPage > 0) {
      appState.selectedPage--;
      loadSelectedInvoices();
    }
  });
  document.getElementById("selected-next-btn")?.addEventListener("click", () => {
    appState.selectedPage++;
    loadSelectedInvoices();
  });
  document.getElementById("excluded-prev-btn")?.addEventListener("click", () => {
    if (appState.excludedPage > 0) {
      appState.excludedPage--;
      loadExcludedInvoices();
    }
  });
  document.getElementById("excluded-next-btn")?.addEventListener("click", () => {
    appState.excludedPage++;
    loadExcludedInvoices();
  });

  // Excluded filters
  document.getElementById("excluded-stage-filter")?.addEventListener("change", () => {
    appState.excludedPage = 0;
    loadExcludedInvoices();
  });
  document.getElementById("excluded-reason-filter")?.addEventListener("change", () => {
    appState.excludedPage = 0;
    loadExcludedInvoices();
  });

  // Downloads
  document.getElementById("download-selected-btn")?.addEventListener("click", () => downloadFile("selected"));
  document.getElementById("download-excluded-btn")?.addEventListener("click", () => downloadFile("excluded"));
  document.getElementById("download-weekly-plan-btn")?.addEventListener("click", () => downloadFile("weekly-plan"));
  document.getElementById("download-weekly-exposure-btn")?.addEventListener("click", () => downloadFile("weekly-exposure"));
  document.getElementById("download-pdf-btn")?.addEventListener("click", () => downloadFile("pdf"));
  document.getElementById("download-report-btn")?.addEventListener("click", () => downloadFile("report"));

  // Help icon popovers
  setupHelpPopover("help-limits-mode", "popover-limits-mode");
  setupHelpPopover("help-facility-ceiling", "popover-facility-ceiling");
  setupHelpPopover("help-customer-concentration", "popover-customer-concentration");
  setupHelpPopover("help-group-concentration", "popover-group-concentration");
  setupHelpPopover("help-manual-facility", "popover-manual-facility");
  setupHelpPopover("help-manual-yaml-upload", "popover-manual-yaml-upload");
  setupHelpPopover("help-manual-customer", "popover-manual-customer");
  setupHelpPopover("help-manual-group", "popover-manual-group");
  setupHelpPopover("help-manual-customer-group-map", "popover-manual-customer-group-map");
  setupHelpPopover("help-base-facility", "popover-base-facility");
  setupHelpPopover("help-base-customer", "popover-base-customer");
  setupHelpPopover("help-base-group", "popover-base-group");
  setupHelpPopover("help-solver-status", "popover-solver-status");
  setupHelpPopover("help-credit-bottleneck", "popover-credit-bottleneck");
  setupHelpPopover("help-deferral-drivers", "popover-deferral-drivers");
  setupHelpPopover("help-weekly-exposure", "popover-weekly-exposure");
  setupHelpPopover("help-weekly-exposure-evolution", "popover-weekly-exposure-evolution");
  setupHelpPopover("help-invoice-evolution", "popover-invoice-evolution");
}

async function handleManualLimitsYamlUpload(event) {
  const input = event?.target;
  const file = input?.files?.[0];
  if (!file) return;

  try {
    const lowerName = String(file.name || "").toLowerCase();
    const isExcel = lowerName.endsWith(".xlsx") || lowerName.endsWith(".xls");
    let parsedConfig;
    let importSummary = null;

    if (isExcel) {
      if (!appState.currentProcessId) {
        throw new Error("Select or create a process before importing an Excel limits file.");
      }
      const importResponse = await uploadFile(
        `/api/optimizer/processes/${appState.currentProcessId}/limits/import`,
        file
      );
      parsedConfig = importResponse?.limits_payload;
      importSummary = importResponse?.import_summary || null;
      if (!parsedConfig || typeof parsedConfig !== "object") {
        throw new Error("Limits import response did not include a valid limits payload.");
      }
    } else {
      const rawText = await file.text();
      parsedConfig = parseManualLimitsConfigText(rawText);
    }

    toggleLimitsMode("manual");
    applyManualLimitsConfig(parsedConfig);

    const filenameEl = document.getElementById("opt-manual-yaml-upload-filename");
    if (filenameEl) filenameEl.textContent = file.name;
    if (importSummary) {
      const rows = importSummary.rows_processed ?? 0;
      const totalRows = importSummary.rows_total ?? 0;
      const conversions = importSummary.gbp_conversions ?? 0;
      const warnings = importSummary.warnings_count ?? 0;
      showSuccess(
        `Loaded manual limits from ${file.name} (${rows}/${totalRows} rows, ${conversions} GBP->EUR conversions, ${warnings} warnings).`
      );
    } else {
      showSuccess(`Loaded manual limits from ${file.name}.`);
    }
  } catch (err) {
    showError(`Failed to load manual limits file: ${err.message}`);
  } finally {
    if (input) input.value = "";
  }
}

async function handleFileUpload(event) {
  const files = event.target.files;
  if (!files || files.length === 0) return;

  appState.selectedFile = files[0];
  const filenameSpan = document.getElementById("opt-selected-filename");
  if (filenameSpan) filenameSpan.textContent = appState.selectedFile.name;

  const createBtn = document.getElementById("opt-create-btn");
  if (createBtn) createBtn.disabled = false;
}

async function createProcess() {
  if (!appState.selectedFile) {
    showError("Please select an extraction or summary file first.");
    return;
  }

  const cohortInput = document.getElementById("opt-cohort-input");
  const cohort = normalizeCohortDate(cohortInput?.value) || null;

  try {
    showError("");
    const result = await uploadFile("/api/optimizer/processes", appState.selectedFile, {
      cohort: cohort || undefined,
      sheet_name: "SAPUI5 Export",
    });

    appState.currentProcessId = result.process_id;
    appState.availableCohorts = result.available_cohorts || [];

    // Show detected cohorts as placeholder hint
    if (appState.availableCohorts.length > 0 && cohortInput && !cohortInput.value) {
      cohortInput.placeholder = appState.availableCohorts[0].date.substring(0, 10);
    }

    await loadProcessList();
    await selectProcess(result.process_id);

    showSuccess("Process created successfully.");
  } catch (err) {
    showError(`Failed to create process: ${err.message}`);
  }
}

function populateCohortHint(cohorts) {
  const input = document.getElementById("opt-cohort-input");
  if (!input || !cohorts || cohorts.length === 0) return;

  // Show the first detected date as a placeholder hint
  if (!input.value) {
    input.placeholder = cohorts[0].date.substring(0, 10);
  }
}

async function loadProcessList() {
  try {
    const processes = await request("/api/optimizer/processes");
    appState.processes = processes;
    renderProcessTable(processes);
  } catch (err) {
    console.error("Failed to load processes:", err);
  }
}

function renderProcessTable(processes) {
  const table = document.getElementById("process-table");
  if (!table) return;

  const existing = table.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
  existing.forEach((r) => r.remove());

  if (processes.length === 0) {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `<ui5-table-cell colspan="6"><div class="table-empty">No processes yet. Upload a file to get started.</div></ui5-table-cell>`;
    table.appendChild(row);
    return;
  }

  processes.forEach((p) => {
    const row = document.createElement("ui5-table-row");
    if (p.process_id === appState.currentProcessId) {
      row.classList.add("selected-process");
    }

    const selAmt = p.selected_amount != null ? formatAmount(p.selected_amount) : "-";
    const selCount = p.selected_count != null ? p.selected_count : "-";
    const candCount = p.candidate_count != null ? p.candidate_count : "-";

    row.innerHTML = `
      <ui5-table-cell>${p.process_id || "-"}</ui5-table-cell>
      <ui5-table-cell><span class="status-badge ${p.status}">${p.status}</span></ui5-table-cell>
      <ui5-table-cell><div class="process-file-cell">${p.extraction_filename || "-"}</div></ui5-table-cell>
      <ui5-table-cell>${p.cohort || "-"}</ui5-table-cell>
      <ui5-table-cell>${selCount} / ${candCount}</ui5-table-cell>
      <ui5-table-cell>${selAmt}</ui5-table-cell>
      <ui5-table-cell></ui5-table-cell>
    `;

    // Add delete button to last cell
    const actionCell = row.querySelector("ui5-table-cell:last-child");
    const deleteBtn = document.createElement("ui5-button");
    deleteBtn.setAttribute("icon", "delete");
    deleteBtn.setAttribute("design", "Transparent");
    deleteBtn.setAttribute("tooltip", "Delete process");
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteProcess(p.process_id);
    });
    actionCell.appendChild(deleteBtn);

    row.addEventListener("click", () => selectProcess(p.process_id));
    table.appendChild(row);
  });
}

async function selectProcess(processId) {
  appState.currentProcessId = processId;
  appState.results = null;
  appState.selectedPage = 0;
  appState.excludedPage = 0;

  const runBtn = document.getElementById("run-optimization-btn");
  if (runBtn) runBtn.disabled = false;

  try {
    const detail = await request(`/api/optimizer/processes/${processId}`);

    // Load cohorts if not already loaded
    if (appState.availableCohorts.length === 0) {
      try {
        const cohorts = await request(`/api/optimizer/processes/${processId}/cohorts`);
        appState.availableCohorts = cohorts;
        populateCohortHint(cohorts);
      } catch (e) {
        console.warn("Could not load cohorts:", e);
      }
    }

    // Set cohort in input
    if (detail.cohort) {
      const cohortInput = document.getElementById("opt-cohort-input");
      if (cohortInput) cohortInput.value = normalizeCohortDate(detail.cohort);
    }

    // Load config
    await loadConfig(processId);

    // Highlight selected row
    const table = document.getElementById("process-table");
    if (table) {
      table.querySelectorAll("ui5-table-row").forEach((r) => r.classList.remove("selected-process"));
      // Re-render to show selection
      renderProcessTable(appState.processes);
    }

    // If completed, load results
    if (detail.status === "completed") {
      await loadResults(processId);
    } else if (detail.status === "running") {
      showProgress();
      pollStatus(processId);
    } else {
      hideProgress();
      hideDashboard();
    }
  } catch (err) {
    showError(`Failed to load process: ${err.message}`);
  }
}

async function deleteProcess(processId) {
  try {
    await request(`/api/optimizer/processes/${processId}`, "DELETE");

    // If the deleted process was selected, clear selection
    if (appState.currentProcessId === processId) {
      appState.currentProcessId = null;
      hideDashboard();
      hideProgress();
      const runBtn = document.getElementById("run-optimization-btn");
      if (runBtn) runBtn.disabled = true;
    }

    await loadProcessList();
    showSuccess("Process deleted.");
  } catch (err) {
    showError(`Failed to delete process: ${err.message}`);
  }
}

async function loadConfig(processId) {
  try {
    const [limits, rules, params] = await Promise.all([
      request(`/api/optimizer/processes/${processId}/limits`),
      request(`/api/optimizer/processes/${processId}/rules`),
      request(`/api/optimizer/processes/${processId}/params`),
    ]);

    appState.limits = limits;
    appState.rules = rules;
    appState.params = params;

    // Update sidebar with config values
    updateConfigUI(limits, rules, params);
  } catch (err) {
    console.error("Failed to load config:", err);
  }
}

function updateConfigUI(limits, rules, params) {
  if (params) {
    const planningMode = params.planning_mode || "single_week";
    togglePlanningMode(planningMode);
    const horizonEl = document.getElementById("opt-horizon-weeks");
    const attemptEl = document.getElementById("opt-attempt-cap");
    if (horizonEl) horizonEl.value = String(params.horizon_weeks ?? 8);
    if (attemptEl) attemptEl.value = String(params.attempt_cap ?? 1);
  }

  // Limits
  if (limits) {
    const synGen = limits.synthetic_generation || {};
    const alpha = document.getElementById("opt-alpha");
    const beta = document.getElementById("opt-beta");
    const gamma = document.getElementById("opt-gamma");
    if (alpha) alpha.value = String(synGen.alpha ?? AUTOMATIC_LIMIT_DEFAULTS.alpha);
    if (beta) beta.value = String(synGen.beta ?? AUTOMATIC_LIMIT_DEFAULTS.beta);
    if (gamma) gamma.value = String(synGen.gamma ?? AUTOMATIC_LIMIT_DEFAULTS.gamma);

    const facilityEl = document.getElementById("opt-facility-limits");
    const customerEl = document.getElementById("opt-customer-limits");
    const groupEl = document.getElementById("opt-group-limits");
    const customerGroupMapEl = document.getElementById("opt-customer-group-map");
    const baseFacilityEl = document.getElementById("opt-base-facility");
    const baseCustomerEl = document.getElementById("opt-base-customer");
    const baseGroupEl = document.getElementById("opt-base-group");
    if (facilityEl) facilityEl.value = formatManualLimitEntries(limits.facility_limits_by_company_code || {});
    if (customerEl) customerEl.value = formatManualLimitEntries(limits.customer_limits || {});
    if (groupEl) groupEl.value = formatManualLimitEntries(limits.group_limits || {});
    if (customerGroupMapEl) customerGroupMapEl.value = formatManualLimitEntries(limits.customer_to_group || {});
    if (baseFacilityEl) baseFacilityEl.value = formatManualLimitEntries(limits.base_exposure?.facility || {});
    if (baseCustomerEl) baseCustomerEl.value = formatManualLimitEntries(limits.base_exposure?.customer || {});
    if (baseGroupEl) baseGroupEl.value = formatManualLimitEntries(limits.base_exposure?.group || {});

    const isAutomatic = synGen.enabled !== false;
    toggleLimitsMode(isAutomatic ? "automatic" : "manual");
  }

  // Rules
  if (rules && rules.rules) {
    renderRulesList(rules.rules);
  }
}

function renderRulesList(rules) {
  const container = document.getElementById("rules-list");
  if (!container) return;
  container.innerHTML = "";

  rules.forEach((rule, idx) => {
    const ruleName = formatRuleLabel(rule.name);
    const ruleType = formatRuleLabel(rule.type);
    const item = document.createElement("div");
    item.classList.add("rule-item");
    item.innerHTML = `
      <div>
        <div class="rule-name">${ruleName}</div>
        <div class="rule-type">${ruleType}</div>
      </div>
      <ui5-switch id="rule-switch-${idx}" ${rule.enabled ? "checked" : ""}></ui5-switch>
    `;
    container.appendChild(item);
  });
}

function toggleLimitsMode(mode) {
  appState.limitsEditMode = mode;
  const autoPanel = document.getElementById("automatic-limits-panel");
  const manualPanel = document.getElementById("manual-limits-panel");
  const modeToggle = document.getElementById("limits-mode-toggle");
  if (modeToggle) {
    const items = modeToggle.querySelectorAll("ui5-segmented-button-item");
    items.forEach((item) => {
      const itemMode = item.textContent.trim().toLowerCase();
      item.pressed = itemMode === mode;
    });
  }
  if (autoPanel) autoPanel.style.display = mode === "automatic" ? "block" : "none";
  if (manualPanel) manualPanel.style.display = mode === "manual" ? "block" : "none";

  if (mode === "manual") {
    const facilityEl = document.getElementById("opt-facility-limits");
    const customerEl = document.getElementById("opt-customer-limits");
    const groupEl = document.getElementById("opt-group-limits");
    const customerGroupMapEl = document.getElementById("opt-customer-group-map");
    const baseFacilityEl = document.getElementById("opt-base-facility");
    const baseCustomerEl = document.getElementById("opt-base-customer");
    const baseGroupEl = document.getElementById("opt-base-group");
    const hasFacilityLimits = Object.keys(appState.limits?.facility_limits_by_company_code || {}).length > 0;
    const hasCustomerLimits = Object.keys(appState.limits?.customer_limits || {}).length > 0;
    const hasGroupLimits = Object.keys(appState.limits?.group_limits || {}).length > 0;
    const hasCustomerGroupMap = Object.keys(appState.limits?.customer_to_group || {}).length > 0;
    const hasBaseFacility = Object.keys(appState.limits?.base_exposure?.facility || {}).length > 0;
    const hasBaseCustomer = Object.keys(appState.limits?.base_exposure?.customer || {}).length > 0;
    const hasBaseGroup = Object.keys(appState.limits?.base_exposure?.group || {}).length > 0;

    const facilityText = (facilityEl?.value || "").trim();
    const customerText = (customerEl?.value || "").trim();
    const groupText = (groupEl?.value || "").trim();
    const customerGroupMapText = (customerGroupMapEl?.value || "").trim();
    const baseFacilityText = (baseFacilityEl?.value || "").trim();
    const baseCustomerText = (baseCustomerEl?.value || "").trim();
    const baseGroupText = (baseGroupEl?.value || "").trim();

    if (facilityEl && !hasFacilityLimits && (facilityText === "" || facilityText === "{}")) {
      facilityEl.value = formatManualLimitEntries(MANUAL_LIMIT_DEFAULTS.facility_limits_by_company_code);
    }
    if (customerEl && !hasCustomerLimits && (customerText === "" || customerText === "{}")) {
      customerEl.value = formatManualLimitEntries(MANUAL_LIMIT_DEFAULTS.customer_limits);
    }
    if (groupEl && !hasGroupLimits && (groupText === "" || groupText === "{}")) {
      groupEl.value = formatManualLimitEntries(MANUAL_LIMIT_DEFAULTS.group_limits);
    }
    if (customerGroupMapEl && !hasCustomerGroupMap && (customerGroupMapText === "" || customerGroupMapText === "{}")) {
      customerGroupMapEl.value = formatManualLimitEntries(MANUAL_LIMIT_DEFAULTS.customer_to_group);
    }
    if (baseFacilityEl && !hasBaseFacility && (baseFacilityText === "" || baseFacilityText === "{}")) {
      baseFacilityEl.value = formatManualLimitEntries(MANUAL_LIMIT_DEFAULTS.base_exposure.facility);
    }
    if (baseCustomerEl && !hasBaseCustomer && (baseCustomerText === "" || baseCustomerText === "{}")) {
      baseCustomerEl.value = formatManualLimitEntries(MANUAL_LIMIT_DEFAULTS.base_exposure.customer);
    }
    if (baseGroupEl && !hasBaseGroup && (baseGroupText === "" || baseGroupText === "{}")) {
      baseGroupEl.value = formatManualLimitEntries(MANUAL_LIMIT_DEFAULTS.base_exposure.group);
    }
  }
}

function togglePlanningMode(mode) {
  appState.planningMode = mode;
  const modeToggle = document.getElementById("planning-mode-toggle");
  if (modeToggle) {
    const items = modeToggle.querySelectorAll("ui5-segmented-button-item");
    items.forEach((item) => {
      const itemMode = item.dataset?.mode || item.getAttribute("data-mode");
      item.pressed = itemMode === mode;
    });
  }
  const mwControls = document.getElementById("multi-week-controls");
  if (mwControls) mwControls.style.display = mode === "multi_week" ? "block" : "none";
  updateWeeklyDownloadButtonVisibility();
}

function getEffectivePlanningModeForDownloads() {
  const resultsMode = appState.results?.planning_mode || appState.results?.metrics?.planning_mode;
  if (resultsMode === "single_week" || resultsMode === "multi_week") {
    return resultsMode;
  }
  return appState.planningMode || "single_week";
}

function updateWeeklyDownloadButtonVisibility() {
  const showWeeklyDownloads = getEffectivePlanningModeForDownloads() === "multi_week";
  const weeklyButtons = [
    document.getElementById("download-weekly-plan-btn"),
    document.getElementById("download-weekly-exposure-btn"),
  ];
  weeklyButtons.forEach((btn) => {
    if (btn) btn.style.display = showWeeklyDownloads ? "" : "none";
  });
}

function getPlanningModeFromUI() {
  const modeToggle = document.getElementById("planning-mode-toggle");
  if (!modeToggle) return appState.planningMode || "single_week";

  const items = Array.from(modeToggle.querySelectorAll("ui5-segmented-button-item"));
  const selectedItem = items.find((item) => item.pressed === true || item.getAttribute("aria-pressed") === "true");
  const mode = selectedItem?.dataset?.mode || selectedItem?.getAttribute("data-mode");
  if (mode === "single_week" || mode === "multi_week") {
    return mode;
  }
  return appState.planningMode || "single_week";
}

async function saveConfiguration() {
  if (!appState.currentProcessId) {
    showError("No process selected.");
    throw new Error("No process selected.");
  }

  const processId = appState.currentProcessId;

  try {
    // Save params (solver settings use server defaults)
    const cohortInput = document.getElementById("opt-cohort-input");
    const cohortValue = normalizeCohortDate(cohortInput?.value);
    const planningMode = getPlanningModeFromUI();
    togglePlanningMode(planningMode);
    const horizonWeeks = parseInt(document.getElementById("opt-horizon-weeks")?.value || "8", 10);
    const attemptCap = parseInt(document.getElementById("opt-attempt-cap")?.value || "1", 10);
    const params = {
      cohort: cohortValue || null,
      planning_mode: planningMode,
      planning_start_date: cohortValue || null,
      horizon_weeks: Number.isFinite(horizonWeeks) ? horizonWeeks : 8,
      attempt_cap: Number.isFinite(attemptCap) ? attemptCap : 1,
      release_event_mode: "reconciliation_file_date",
      release_event: "reconciliation_file_date",
      source_profile: appState.params?.source_profile || "extraction_file",
      lifecycle_input_path: appState.params?.lifecycle_input_path || null,
      solver_max_time_seconds: 60,
      solver_random_seed: appState.params?.solver_random_seed ?? 0,
      solver_num_search_workers: appState.params?.solver_num_search_workers ?? 1,
    };
    await request(`/api/optimizer/processes/${processId}/params`, "PUT", params);
    const persistedParams = await request(`/api/optimizer/processes/${processId}/params`);
    const persistedPlanningMode = persistedParams?.planning_mode || "single_week";
    if (persistedPlanningMode !== planningMode) {
      throw new Error(
        `Planning mode save mismatch (expected ${planningMode}, got ${persistedPlanningMode}).`
      );
    }

    // Save limits
    let baseExposureFacility = {};
    let baseExposureCustomer = {};
    let baseExposureGroup = {};
    let customerToGroup = {};
    try {
      baseExposureFacility = parseManualLimitEntries(
        document.getElementById("opt-base-facility")?.value,
        "Facility Base Exposure"
      );
    } catch (e) {
      throw new Error(e.message || "Invalid Facility Base Exposure.");
    }
    try {
      baseExposureCustomer = parseManualLimitEntries(
        document.getElementById("opt-base-customer")?.value,
        "Customer Base Exposure"
      );
    } catch (e) {
      throw new Error(e.message || "Invalid Customer Base Exposure.");
    }
    try {
      baseExposureGroup = parseManualLimitEntries(
        document.getElementById("opt-base-group")?.value,
        "Group Base Exposure"
      );
    } catch (e) {
      throw new Error(e.message || "Invalid Group Base Exposure.");
    }
    try {
      customerToGroup = parseManualLimitEntries(
        document.getElementById("opt-customer-group-map")?.value,
        "Customer to Group Mapping"
      );
    } catch (e) {
      throw new Error(e.message || "Invalid Customer to Group Mapping.");
    }

    let limitsPayload;
    if (appState.limitsEditMode === "automatic") {
      limitsPayload = {
        facility_limits_by_company_code: {},
        customer_limits: {},
        group_limits: {},
        customer_to_group: customerToGroup,
        base_exposure: {
          facility: baseExposureFacility,
          customer: baseExposureCustomer,
          group: baseExposureGroup,
        },
        defaults: appState.limits?.defaults || {
          customer_limit_fraction_of_facility: 0.15,
          group_limit_fraction_of_facility: 0.30,
        },
        synthetic_generation: {
          enabled: true,
          alpha: parseFloat(document.getElementById("opt-alpha")?.value) || AUTOMATIC_LIMIT_DEFAULTS.alpha,
          beta: parseFloat(document.getElementById("opt-beta")?.value) || AUTOMATIC_LIMIT_DEFAULTS.beta,
          gamma: parseFloat(document.getElementById("opt-gamma")?.value) || AUTOMATIC_LIMIT_DEFAULTS.gamma,
        },
      };
    } else {
      let facilityLimits = {};
      let customerLimits = {};
      let groupLimits = {};
      try {
        facilityLimits = parseManualLimitEntries(
          document.getElementById("opt-facility-limits")?.value,
          "Facility Limits"
        );
      } catch (e) {
        throw new Error(e.message || "Invalid Facility Limits.");
      }
      try {
        customerLimits = parseManualLimitEntries(
          document.getElementById("opt-customer-limits")?.value,
          "Customer Limits"
        );
      } catch (e) {
        throw new Error(e.message || "Invalid Customer Limits.");
      }
      try {
        groupLimits = parseManualLimitEntries(
          document.getElementById("opt-group-limits")?.value,
          "Group Limits"
        );
      } catch (e) {
        throw new Error(e.message || "Invalid Group Limits.");
      }

      limitsPayload = {
        facility_limits_by_company_code: facilityLimits,
        customer_limits: customerLimits,
        group_limits: groupLimits,
        customer_to_group: customerToGroup,
        base_exposure: {
          facility: baseExposureFacility,
          customer: baseExposureCustomer,
          group: baseExposureGroup,
        },
        defaults: appState.limits?.defaults || {
          customer_limit_fraction_of_facility: 0.15,
          group_limit_fraction_of_facility: 0.30,
        },
        synthetic_generation: { enabled: false },
      };
    }
    await request(`/api/optimizer/processes/${processId}/limits`, "PUT", limitsPayload);

    // Save rules
    const rulesListEl = document.getElementById("rules-list");
    if (rulesListEl && appState.rules?.rules) {
      const updatedRules = appState.rules.rules.map((rule, idx) => {
        const sw = document.getElementById(`rule-switch-${idx}`);
        return { ...rule, enabled: sw ? sw.checked : rule.enabled };
      });
      await request(`/api/optimizer/processes/${processId}/rules`, "PUT", { rules: updatedRules });
    }

    showSuccess("Configuration saved.");
  } catch (err) {
    showError(`Failed to save configuration: ${err.message}`);
    throw err;
  }
}

async function startOptimization() {
  if (!appState.currentProcessId) {
    showError("No process selected.");
    return;
  }

  const planningMode = getPlanningModeFromUI();
  togglePlanningMode(planningMode);

  const cohortInput = document.getElementById("opt-cohort-input");
  const cohortValue = normalizeCohortDate(cohortInput?.value);
  if (planningMode === "single_week" && !cohortValue) {
    showError("Please select a cohort date before running the optimization.");
    if (cohortInput) cohortInput.focus();
    return;
  }

  // Save config first
  try {
    await saveConfiguration();
  } catch (err) {
    // saveConfiguration already shows a user-facing error.
    return;
  }

  try {
    const horizonWeeks = parseInt(document.getElementById("opt-horizon-weeks")?.value || "8", 10);
    const attemptCap = parseInt(document.getElementById("opt-attempt-cap")?.value || "1", 10);
    await request(`/api/optimizer/processes/${appState.currentProcessId}/run`, "POST", {
      planning_mode: planningMode,
      planning_start_date: cohortValue || null,
      horizon_weeks: Number.isFinite(horizonWeeks) ? horizonWeeks : 8,
      attempt_cap: Number.isFinite(attemptCap) ? attemptCap : 1,
    });
    showProgress();
    pollStatus(appState.currentProcessId);
  } catch (err) {
    showError(`Failed to start optimization: ${err.message}`);
  }
}

function pollStatus(processId) {
  if (appState.pollingInterval) {
    clearInterval(appState.pollingInterval);
  }

  appState.pollingInterval = setInterval(async () => {
    try {
      const status = await request(`/api/optimizer/processes/${processId}/status`);
      renderOptimizationProgress(status.progress, status.status, status.started_at);

      if (status.status === "completed") {
        clearInterval(appState.pollingInterval);
        appState.pollingInterval = null;
        hideProgress();
        await loadProcessList();
        await loadResults(processId);
        showSuccess("Optimization completed successfully.");
      } else if (status.status === "failed") {
        clearInterval(appState.pollingInterval);
        appState.pollingInterval = null;
        hideProgress();
        await loadProcessList();
        showError(`Optimization failed: ${status.error_message || "Unknown error"}`);
      }
    } catch (err) {
      console.error("Polling error:", err);
    }
  }, 2000);
}

async function loadResults(processId) {
  try {
    const results = await request(`/api/optimizer/processes/${processId}/results`);
    appState.results = results;

    const metrics = results.metrics || {};

    // Update stat boxes
    document.getElementById("opt-candidates-count").textContent = (metrics.baseline_submitted_count || 0).toLocaleString();
    document.getElementById("opt-selected-count").textContent = (metrics.optimized_submitted_count || 0).toLocaleString();
    document.getElementById("opt-excluded-count").textContent = (
      (metrics.rule_excluded_count || 0) + (metrics.not_selected_count || 0)
    ).toLocaleString();
    updateSelectedAmountDisplay(metrics.selected_total_amount || 0);
    document.getElementById("opt-amount-ratio").textContent = `${(metrics.selected_amount_ratio_pct || 0).toFixed(1)}%`;
    document.getElementById("opt-solver-status").textContent = metrics.optimizer_status || "-";

    const planningMode = results.planning_mode || metrics.planning_mode || appState.planningMode || "single_week";
    togglePlanningMode(planningMode);

    await renderCharts(results);
    await loadSelectedInvoices();
    await loadExcludedInvoices();
    renderWeeklyTables(results);

    showDashboard();
  } catch (err) {
    showError(`Failed to load results: ${err.message}`);
  }
}

function renderWeeklyTables(results) {
  const section = document.getElementById("opt-weekly-section");
  const weeklyPlan = results.weekly_plan || [];
  const weeklyExposure = results.weekly_exposure || [];
  const planningMode = results.planning_mode || results.metrics?.planning_mode || "single_week";
  const lifetimeStatusEl = document.getElementById("weekly-lifetime-status");
  const selectedEntityType = String(appState.weeklyExposureEntityTypeFilter || "all").toLowerCase();

  if (!section) return;
  if (planningMode !== "multi_week") {
    section.style.display = "none";
    if (lifetimeStatusEl) lifetimeStatusEl.style.display = "none";
    hideWeeklyExposureEvolutionSection();
    return;
  }
  section.style.display = "block";

  const planTable = document.getElementById("weekly-plan-table");
  const exposureTable = document.getElementById("weekly-exposure-table");

  if (lifetimeStatusEl) {
    const rpt = results.lifetime_estimation || {};
    const status = String(rpt.status || "").toLowerCase();
    const predicted = Number(rpt.predicted_candidates || 0);
    const defaultWeeks = Number(
      weeklyPlan.find((row) => Number.isFinite(Number(row.expected_lifetime_weeks)))
        ?.expected_lifetime_weeks || 4
    );

    if (!status) {
      lifetimeStatusEl.style.display = "none";
    } else if (status === "completed" && predicted > 0) {
      lifetimeStatusEl.design = "Positive";
      lifetimeStatusEl.textContent = `RPT-1 lifetime estimation active: ${predicted} candidates predicted.`;
      lifetimeStatusEl.style.display = "block";
    } else {
      const firstError = Array.isArray(rpt.errors) && rpt.errors.length ? ` Error: ${rpt.errors[0]}` : "";
      lifetimeStatusEl.design = "Warning";
      lifetimeStatusEl.textContent = (
        `RPT-1 lifetime estimation was not applied (status=${status || "unknown"}). `
        + `Using fallback lifetime=${defaultWeeks} weeks.${firstError}`
      );
      lifetimeStatusEl.style.display = "block";
    }
  }

  if (planTable) {
    planTable.querySelectorAll("ui5-table-row:not([slot='headerRow'])").forEach((r) => r.remove());
    const sample = weeklyPlan.slice(0, 200);
    sample.forEach((row) => {
      const lifetimeWeeks = Number.isFinite(Number(row.expected_lifetime_weeks))
        ? Number(row.expected_lifetime_weeks)
        : "-";
      const lifetimeSource = row.expected_lifetime_source || (lifetimeWeeks !== "-" ? "fallback_default_weeks" : "-");
      const tr = document.createElement("ui5-table-row");
      tr.innerHTML = `
        <ui5-table-cell>${row["Invoice Reference"] || row.invoice_reference || "-"}</ui5-table-cell>
        <ui5-table-cell>${row.Customer || row.debtor_id || "-"}</ui5-table-cell>
        <ui5-table-cell>${row["Company Code"] || row.seller_id_external || "-"}</ui5-table-cell>
        <ui5-table-cell>${formatAmount(row["Purchase Price"] ?? row.candidate_amount ?? 0)}</ui5-table-cell>
        <ui5-table-cell>${row.planned_week_start_iso || row.planned_week_start || "-"}</ui5-table-cell>
        <ui5-table-cell>${lifetimeWeeks}</ui5-table-cell>
        <ui5-table-cell>${lifetimeSource}</ui5-table-cell>
      `;
      planTable.appendChild(tr);
    });
  }

  const filteredExposure = selectedEntityType === "all"
    ? weeklyExposure
    : weeklyExposure.filter((row) => String(row.entity_type || "").toLowerCase() === selectedEntityType);
  if (exposureTable) {
    exposureTable.querySelectorAll("ui5-table-row:not([slot='headerRow'])").forEach((r) => r.remove());
    const sample = filteredExposure.slice(0, 300);
    sample.forEach((row) => {
      const util = Number(row.utilization_pct ?? 0);
      const tr = document.createElement("ui5-table-row");
      tr.innerHTML = `
        <ui5-table-cell>${row.week_start || "-"}</ui5-table-cell>
        <ui5-table-cell>${row.entity_type || "-"}</ui5-table-cell>
        <ui5-table-cell>${row.entity_id || "-"}</ui5-table-cell>
        <ui5-table-cell>${formatAmount(row.used_total ?? row.used ?? 0)}</ui5-table-cell>
        <ui5-table-cell>${formatAmount(row.limit ?? 0)}</ui5-table-cell>
        <ui5-table-cell>${util.toFixed(1)}%</ui5-table-cell>
      `;
      exposureTable.appendChild(tr);
    });
  }

  renderWeeklyExposureEvolutionChart(filteredExposure, selectedEntityType);
}

function parseWeekStartToUtcMs(value) {
  const raw = String(value || "").trim();
  if (!raw) return null;

  const match = raw.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!match) return null;

  const [, y, m, d] = match;
  return Date.UTC(Number(y), Number(m) - 1, Number(d));
}

function formatUtcMsAsIsoDate(ms) {
  const date = new Date(ms);
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function toNumericAmount(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : 0;
}

function formatEntityTypeLabel(entityType) {
  const value = String(entityType || "").trim().toLowerCase();
  if (!value) return "Entity";
  return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}

function hideWeeklyExposureEvolutionSection() {
  const section = document.getElementById("weekly-exposure-evolution-section");
  if (section) section.style.display = "none";
  updateWeeklyExposureEvolutionLegend([]);

  const chart = appState.charts.weeklyExposureEvolution;
  if (chart && typeof chart.destroy === "function") {
    chart.destroy();
  }
  delete appState.charts.weeklyExposureEvolution;
}

function updateWeeklyExposureEvolutionLegend(datasets) {
  const legend = document.getElementById("weekly-exposure-evolution-legend");
  if (!legend) return;

  legend.replaceChildren();
  if (!Array.isArray(datasets) || datasets.length === 0) {
    legend.style.display = "none";
    return;
  }

  legend.style.display = "grid";
  datasets.forEach((dataset) => {
    const item = document.createElement("div");
    item.className = "weekly-exposure-evolution-legend-item";

    const swatch = document.createElement("span");
    swatch.className = "weekly-exposure-evolution-legend-swatch";
    swatch.style.backgroundColor = String(dataset.borderColor || dataset.backgroundColor || "#666");

    const label = document.createElement("span");
    label.className = "weekly-exposure-evolution-legend-label";
    label.textContent = String(dataset.label || "");

    item.appendChild(swatch);
    item.appendChild(label);
    legend.appendChild(item);
  });
}

function buildWeeklyExposureEvolutionSeries(filteredExposureRows) {
  if (!Array.isArray(filteredExposureRows) || filteredExposureRows.length === 0) return null;

  const rows = [];
  const weekSet = new Set();
  const entitySet = new Set();

  filteredExposureRows.forEach((row) => {
    const weekMs = parseWeekStartToUtcMs(row.week_start);
    if (!Number.isFinite(weekMs)) return;

    const entityId = String(row.entity_id || "").trim() || "(unknown)";
    const utilizationFromPayload = Number(row.utilization_pct);
    const limit = toNumericAmount(row.limit);
    const used = toNumericAmount(row.used_total ?? row.used);
    const utilizationPct = Number.isFinite(utilizationFromPayload)
      ? utilizationFromPayload
      : (limit > 0 ? (used / limit) * 100 : 0);

    rows.push({ weekMs, entityId, utilizationPct });
    weekSet.add(weekMs);
    entitySet.add(entityId);
  });

  if (!rows.length || !weekSet.size || !entitySet.size) return null;

  const weeks = Array.from(weekSet).sort((a, b) => a - b);
  const labels = weeks.map((weekMs) => formatUtcMsAsIsoDate(weekMs));
  const entities = Array.from(entitySet).sort((a, b) => a.localeCompare(b));
  const byEntity = new Map();

  entities.forEach((entityId) => byEntity.set(entityId, new Map()));
  rows.forEach((row) => {
    const weekMap = byEntity.get(row.entityId);
    if (weekMap) weekMap.set(row.weekMs, row.utilizationPct);
  });

  const datasets = entities.map((entityId, index) => {
    const hue = (index * 137.508) % 360;
    const borderColor = `hsl(${hue}, 70%, 45%)`;
    const backgroundColor = `hsla(${hue}, 70%, 45%, 0.18)`;
    const weekMap = byEntity.get(entityId);
    return {
      label: entityId,
      data: weeks.map((weekMs) => (weekMap?.has(weekMs) ? weekMap.get(weekMs) : null)),
      borderColor,
      backgroundColor,
      pointRadius: 1.5,
      pointHoverRadius: 3,
      tension: 0,
      spanGaps: false,
    };
  });

  return { labels, datasets };
}

function renderWeeklyExposureEvolutionChart(filteredExposureRows, selectedEntityType) {
  if (selectedEntityType === "all") {
    hideWeeklyExposureEvolutionSection();
    return;
  }

  const section = document.getElementById("weekly-exposure-evolution-section");
  const title = document.getElementById("weekly-exposure-evolution-title");
  if (!section) return;

  const series = buildWeeklyExposureEvolutionSeries(filteredExposureRows);
  if (!series || !series.labels.length || !series.datasets.length) {
    hideWeeklyExposureEvolutionSection();
    return;
  }

  const chart = appState.charts.weeklyExposureEvolution;
  if (chart && typeof chart.destroy === "function") {
    chart.destroy();
  }

  const ctx = document.getElementById("weekly-exposure-evolution-chart")?.getContext("2d");
  if (!ctx) {
    hideWeeklyExposureEvolutionSection();
    return;
  }

  section.style.display = "block";
  if (title) title.textContent = `Exposure Utilization Evolution (${formatEntityTypeLabel(selectedEntityType)})`;

  appState.charts.weeklyExposureEvolution = new Chart(ctx, {
    type: "line",
    data: {
      labels: series.labels,
      datasets: series.datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        title: {
          display: true,
          text: `Utilization by ${formatEntityTypeLabel(selectedEntityType)} and Week`,
        },
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${(Number(context.parsed.y) || 0).toFixed(1)}%`,
          },
        },
      },
      scales: {
        y: {
          min: 0,
          ticks: {
            callback: (value) => `${(Number(value) || 0).toFixed(0)}%`,
          },
        },
      },
    },
  });

  updateWeeklyExposureEvolutionLegend(series.datasets);
}

function buildWeeklyInvoiceEvolutionSeries(weeklyPlanRows) {
  if (!Array.isArray(weeklyPlanRows) || weeklyPlanRows.length === 0) return null;

  const rows = [];
  let firstWeekStartMs = null;
  let maxEndExclusiveMs = null;

  weeklyPlanRows.forEach((row) => {
    const startMs = parseWeekStartToUtcMs(row.planned_week_start_iso || row.planned_week_start);
    if (!Number.isFinite(startMs)) return;

    const parsedLifetime = Number(row.expected_lifetime_weeks);
    const lifetimeWeeks = Number.isFinite(parsedLifetime)
      ? Math.max(1, Math.trunc(parsedLifetime))
      : 4;
    const amount = toNumericAmount(row["Purchase Price"] ?? row.candidate_amount ?? row.purchase_price ?? 0);
    const endExclusiveMs = startMs + (lifetimeWeeks * WEEK_IN_MS);

    rows.push({ startMs, endExclusiveMs, amount });

    if (firstWeekStartMs == null || startMs < firstWeekStartMs) {
      firstWeekStartMs = startMs;
    }
    if (maxEndExclusiveMs == null || endExclusiveMs > maxEndExclusiveMs) {
      maxEndExclusiveMs = endExclusiveMs;
    }
  });

  if (!rows.length || firstWeekStartMs == null || maxEndExclusiveMs == null) return null;

  const labels = [];
  const inSystemSeries = [];
  const processingSeries = [];

  for (let weekMs = firstWeekStartMs; weekMs <= maxEndExclusiveMs; weekMs += WEEK_IN_MS) {
    let inSystem = 0;
    let processing = 0;

    rows.forEach((row) => {
      if (weekMs >= row.startMs && weekMs < row.endExclusiveMs) {
        processing += row.amount;
      }
      if (weekMs >= firstWeekStartMs && weekMs < row.endExclusiveMs) {
        inSystem += row.amount;
      }
    });

    labels.push(formatUtcMsAsIsoDate(weekMs));
    inSystemSeries.push(inSystem);
    processingSeries.push(processing);
  }

  return { labels, inSystemSeries, processingSeries };
}

function renderInvoiceEvolutionChart(results) {
  const section = document.getElementById("invoice-evolution-section");
  const note = document.getElementById("invoice-evolution-note");
  if (!section) return;

  const planningMode = results.planning_mode || results.metrics?.planning_mode || "single_week";
  if (planningMode !== "multi_week") {
    section.style.display = "none";
    if (note) note.style.display = "none";
    return;
  }

  const series = buildWeeklyInvoiceEvolutionSeries(results.weekly_plan || []);
  if (!series || !series.labels.length) {
    section.style.display = "none";
    if (note) note.style.display = "none";
    return;
  }

  section.style.display = "block";

  const notSelectedCount = Number(results.metrics?.not_selected_count || 0);
  if (note) {
    if (notSelectedCount > 0) {
      note.textContent = (
        `Chart includes only scheduled invoices. ${notSelectedCount.toLocaleString()} `
        + "invoices were not scheduled in the current horizon."
      );
      note.style.display = "block";
    } else {
      note.style.display = "none";
    }
  }

  const evolutionCtx = document.getElementById("invoice-evolution-chart")?.getContext("2d");
  if (!evolutionCtx) return;

  appState.charts.evolution = new Chart(evolutionCtx, {
    type: "line",
    data: {
      labels: series.labels,
      datasets: [
        {
          label: "Cumulative outstanding exposure (€)",
          data: series.inSystemSeries,
          borderColor: "rgba(220,53,69,1)",
          backgroundColor: "rgba(220,53,69,0.15)",
          pointBackgroundColor: "rgba(220,53,69,1)",
          pointRadius: 2,
          pointHoverRadius: 4,
          tension: 0,
        },
        {
          label: "Active invoices in lifetime (€)",
          data: series.processingSeries,
          borderColor: "rgba(40,167,69,1)",
          backgroundColor: "rgba(40,167,69,0.15)",
          pointBackgroundColor: "rgba(40,167,69,1)",
          pointRadius: 2,
          pointHoverRadius: 4,
          tension: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        title: { display: true, text: "Scheduled Invoice Evolution by Week" },
        subtitle: {
          display: true,
          text: "How scheduled invoice exposure builds and decays over the planning horizon",
          font: { size: 12, style: "italic" },
          padding: { bottom: 10 },
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${EUR_CURRENCY_FORMATTER.format(context.parsed.y || 0)}`,
          },
        },
      },
      scales: {
        y: {
          min: 0,
          ticks: {
            callback: (value) => EUR_CURRENCY_FORMATTER.format(Number(value) || 0),
          },
        },
      },
    },
  });
}

function extractWeeklyExposureRows(results) {
  if (Array.isArray(results.weekly_exposure) && results.weekly_exposure.length) {
    return results.weekly_exposure;
  }

  const metrics = results.metrics || {};
  const rows = [];

  const appendUsageRows = (usageByWeek, entityType) => {
    Object.entries(usageByWeek || {}).forEach(([weekStart, entities]) => {
      Object.entries(entities || {}).forEach(([entityId, usage]) => {
        rows.push({
          week_start: weekStart,
          entity_type: entityType,
          entity_id: entityId,
          ...usage,
        });
      });
    });
  };

  appendUsageRows(metrics.facility_weekly_usage, "facility");
  appendUsageRows(metrics.customer_weekly_usage, "customer");
  appendUsageRows(metrics.group_weekly_usage, "group");
  return rows;
}

function normalizeWeeklyExposureRows(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return [];
  return rows
    .map((row) => {
      const weekMs = parseWeekStartToUtcMs(row.week_start);
      if (!Number.isFinite(weekMs)) return null;

      const limit = toNumericAmount(row.limit);
      const usedTotal = toNumericAmount(row.used_total ?? row.used);
      const utilizationFromPayload = Number(row.utilization_pct);
      const utilizationPct = Number.isFinite(utilizationFromPayload)
        ? utilizationFromPayload
        : (limit > 0 ? (usedTotal / limit) * 100 : 0);

      return {
        weekMs,
        weekLabel: formatUtcMsAsIsoDate(weekMs),
        entityType: String(row.entity_type || "").trim().toLowerCase(),
        entityId: String(row.entity_id || "").trim() || "(unknown)",
        usedTotal,
        limit,
        utilizationPct,
      };
    })
    .filter(Boolean);
}

function alignToPlanningWeekMs(dateMs, planningStartMs) {
  if (!Number.isFinite(dateMs)) return null;
  if (!Number.isFinite(planningStartMs)) return dateMs;
  const weekOffset = Math.floor((dateMs - planningStartMs) / WEEK_IN_MS);
  return planningStartMs + (weekOffset * WEEK_IN_MS);
}

function parseDeferralConstraintWeights(reasonDetail, reason) {
  const text = String(reasonDetail || "");
  const facilityMatch = text.match(/facility\s*=\s*(\d+)/i);
  const customerMatch = text.match(/customer\s*=\s*(\d+)/i);
  const groupMatch = text.match(/group\s*=\s*(\d+)/i);

  const active = [];
  if (facilityMatch && Number(facilityMatch[1]) > 0) active.push("facility");
  if (customerMatch && Number(customerMatch[1]) > 0) active.push("customer");
  if (groupMatch && Number(groupMatch[1]) > 0) active.push("group");

  if (!active.length) {
    const reasonText = String(reason || "").toLowerCase();
    if (reasonText.includes("facility")) active.push("facility");
    else if (reasonText.includes("customer")) active.push("customer");
    else if (reasonText.includes("group")) active.push("group");
    else active.push("other");
  }

  const weights = { facility: 0, customer: 0, group: 0, other: 0 };
  const share = 1 / active.length;
  active.forEach((key) => {
    if (weights[key] != null) weights[key] += share;
  });
  return weights;
}

function normalizeDeferredRows(rows, planningStartMs) {
  if (!Array.isArray(rows) || rows.length === 0) return [];
  return rows
    .map((row) => {
      const eventMs = parseWeekStartToUtcMs(row.offer_file_date || row.summary_file_date || row.due_date);
      if (!Number.isFinite(eventMs)) return null;
      const weekMs = alignToPlanningWeekMs(eventMs, planningStartMs);
      return {
        weekMs,
        weekLabel: formatUtcMsAsIsoDate(weekMs),
        amount: toNumericAmount(row.purchase_price),
        reason: String(row.excluded_reason || "").trim() || "unknown",
        weights: parseDeferralConstraintWeights(row.excluded_reason_detail, row.excluded_reason),
      };
    })
    .filter(Boolean);
}

async function fetchExcludedInvoices(processId, stage = null) {
  if (!processId) return [];
  const pageSize = 500;
  let offset = 0;
  let total = Number.POSITIVE_INFINITY;
  const rows = [];

  while (offset < total) {
    const stagePart = stage ? `&stage=${encodeURIComponent(stage)}` : "";
    const response = await request(
      `/api/optimizer/processes/${processId}/results/excluded?limit=${pageSize}&offset=${offset}${stagePart}`
    );
    const invoices = Array.isArray(response.invoices) ? response.invoices : [];
    rows.push(...invoices);

    const fetchedTotal = rows.length;
    const reportedTotal = Number(response.total);
    total = Number.isFinite(reportedTotal) ? reportedTotal : fetchedTotal;

    if (!invoices.length || fetchedTotal >= total) break;
    offset += pageSize;
  }

  return rows;
}

async function fetchOptimizerExcludedInvoices(processId) {
  if (!processId) return [];
  if (appState.optimizerExcludedCache[processId]) {
    return appState.optimizerExcludedCache[processId];
  }
  let optimizerRows = [];
  try {
    optimizerRows = await fetchExcludedInvoices(processId, "optimizer");
  } catch (err) {
    console.warn("Failed fetching optimizer-only exclusions; retrying with all exclusions.", err);
  }

  if (!optimizerRows.length) {
    try {
      const allRows = await fetchExcludedInvoices(processId);
      optimizerRows = allRows.filter((row) => String(row.excluded_stage || "").toLowerCase() === "optimizer");
    } catch (err) {
      console.warn("Failed fetching fallback exclusions.", err);
    }
  }

  appState.optimizerExcludedCache[processId] = optimizerRows;
  return optimizerRows;
}

function buildFacilityHeadroomSeries(exposureRows) {
  const weekMap = new Map();
  const utilizationByWeekType = new Map();

  exposureRows.forEach((row) => {
    if (!(row.limit > 0)) return;
    if (!["facility", "customer", "group"].includes(row.entityType)) return;
    const weekData = weekMap.get(row.weekMs) || {
      minFacilityHeadroom: null,
      minCustomerHeadroom: null,
      minGroupHeadroom: null,
    };
    const headroom = Math.max(0, row.limit - row.usedTotal);
    if (row.entityType === "facility") {
      if (weekData.minFacilityHeadroom == null || headroom < weekData.minFacilityHeadroom) {
        weekData.minFacilityHeadroom = headroom;
      }
    } else if (row.entityType === "customer") {
      if (weekData.minCustomerHeadroom == null || headroom < weekData.minCustomerHeadroom) {
        weekData.minCustomerHeadroom = headroom;
      }
    } else if (row.entityType === "group") {
      if (weekData.minGroupHeadroom == null || headroom < weekData.minGroupHeadroom) {
        weekData.minGroupHeadroom = headroom;
      }
    }
    weekMap.set(row.weekMs, weekData);

    const key = `${row.weekMs}|${row.entityType}`;
    const currentMax = utilizationByWeekType.get(key) || 0;
    if (row.utilizationPct > currentMax) {
      utilizationByWeekType.set(key, row.utilizationPct);
    }
  });

  const weeks = Array.from(weekMap.keys()).sort((a, b) => a - b);
  if (!weeks.length) return null;
  const labels = weeks.map((weekMs) => formatUtcMsAsIsoDate(weekMs));

  const combinedBottleneckHeadroomSeries = [];
  const maxUtilizationPctSeries = [];
  const maxCustomerUtilizationPctSeries = [];
  const maxGroupUtilizationPctSeries = [];

  weeks.forEach((weekMs) => {
    const weekData = weekMap.get(weekMs) || {};
    const candidates = [
      weekData.minFacilityHeadroom,
      weekData.minCustomerHeadroom,
      weekData.minGroupHeadroom,
    ].filter((v) => Number.isFinite(v));
    const combinedBottleneckHeadroom = candidates.length ? Math.min(...candidates) : 0;

    combinedBottleneckHeadroomSeries.push(combinedBottleneckHeadroom);
    maxUtilizationPctSeries.push(utilizationByWeekType.get(`${weekMs}|facility`) ?? null);
    maxCustomerUtilizationPctSeries.push(utilizationByWeekType.get(`${weekMs}|customer`) ?? null);
    maxGroupUtilizationPctSeries.push(utilizationByWeekType.get(`${weekMs}|group`) ?? null);
  });

  return {
    labels,
    combinedBottleneckHeadroomSeries,
    maxUtilizationPctSeries,
    maxCustomerUtilizationPctSeries,
    maxGroupUtilizationPctSeries,
  };
}

function buildDeferralDriversSeries(deferredRows) {
  const byWeek = new Map();
  deferredRows.forEach((row) => {
    if (!Number.isFinite(row.weekMs)) return;
    const bucket = byWeek.get(row.weekMs) || {
      facilityCount: 0,
      customerCount: 0,
      groupCount: 0,
      otherCount: 0,
      totalCount: 0,
      totalAmount: 0,
    };
    const weights = row.weights || {};
    bucket.facilityCount += Number(weights.facility) || 0;
    bucket.customerCount += Number(weights.customer) || 0;
    bucket.groupCount += Number(weights.group) || 0;
    bucket.otherCount += Number(weights.other) || 0;
    bucket.totalCount += 1;
    bucket.totalAmount += Number(row.amount) || 0;
    byWeek.set(row.weekMs, bucket);
  });

  const weeks = Array.from(byWeek.keys())
    .sort((a, b) => a - b)
    .filter((weekMs) => (byWeek.get(weekMs)?.totalCount || 0) > 0);
  if (!weeks.length) return null;

  return {
    labels: weeks.map((weekMs) => formatUtcMsAsIsoDate(weekMs)),
    facilityCountSeries: weeks.map((weekMs) => byWeek.get(weekMs)?.facilityCount || 0),
    customerCountSeries: weeks.map((weekMs) => byWeek.get(weekMs)?.customerCount || 0),
    groupCountSeries: weeks.map((weekMs) => byWeek.get(weekMs)?.groupCount || 0),
    otherCountSeries: weeks.map((weekMs) => byWeek.get(weekMs)?.otherCount || 0),
    totalCountSeries: weeks.map((weekMs) => byWeek.get(weekMs)?.totalCount || 0),
    totalAmountSeries: weeks.map((weekMs) => byWeek.get(weekMs)?.totalAmount || 0),
  };
}

function buildDeferralDriversSummarySeries(reasonCounts) {
  const entries = Object.entries(reasonCounts || {});
  if (!entries.length) return null;

  let facilityCount = 0;
  let customerCount = 0;
  let groupCount = 0;
  let otherCount = 0;

  entries.forEach(([reason, rawCount]) => {
    const count = Number(rawCount) || 0;
    const weights = parseDeferralConstraintWeights("", reason);
    facilityCount += count * (Number(weights.facility) || 0);
    customerCount += count * (Number(weights.customer) || 0);
    groupCount += count * (Number(weights.group) || 0);
    otherCount += count * (Number(weights.other) || 0);
  });

  const totalCount = facilityCount + customerCount + groupCount + otherCount;
  if (totalCount <= 0) return null;

  return {
    labels: ["All planning weeks"],
    facilityCountSeries: [facilityCount],
    customerCountSeries: [customerCount],
    groupCountSeries: [groupCount],
    otherCountSeries: [otherCount],
    totalCountSeries: [totalCount],
    totalAmountSeries: [0],
  };
}

function configureInsightLayout({ multiWeek, showDeferralChart = false }) {
  const grid = document.getElementById("insight-charts-grid");
  const facilityContainer = document.getElementById("chart-container-facility");
  const customerContainer = document.getElementById("chart-container-customer");
  const exclusionContainer = document.getElementById("chart-container-exclusion");
  const funnelContainer = document.getElementById("chart-container-funnel");

  if (!grid || !facilityContainer || !customerContainer || !exclusionContainer || !funnelContainer) {
    return;
  }

  const useStacked = Boolean(multiWeek);
  grid.classList.toggle("charts-grid--stacked", useStacked);
  facilityContainer.classList.toggle("chart-container--tall", useStacked);
  customerContainer.classList.toggle("chart-container--tall", useStacked && showDeferralChart);
  customerContainer.style.display = multiWeek && !showDeferralChart ? "none" : "";

  if (useStacked) {
    exclusionContainer.style.display = "none";
    funnelContainer.style.display = "none";
  } else {
    exclusionContainer.style.display = "";
    funnelContainer.style.display = "";
  }
}

function renderSingleWeekUtilizationCharts(results) {
  const metrics = results.metrics || {};

  const facilityUsage = metrics.facility_usage || {};
  const facilityLabels = Object.keys(facilityUsage);
  const facilityUsed = facilityLabels.map((k) => facilityUsage[k]?.used || 0);
  const facilityLimits = facilityLabels.map((k) => facilityUsage[k]?.limit || 0);
  const facilityCtx = document.getElementById("facility-chart")?.getContext("2d");
  if (facilityCtx && facilityLabels.length > 0) {
    appState.charts.facility = new Chart(facilityCtx, {
      type: "bar",
      data: {
        labels: facilityLabels,
        datasets: [
          { label: "Used", data: facilityUsed, backgroundColor: "rgba(54,162,235,0.7)" },
          { label: "Limit", data: facilityLimits, backgroundColor: "rgba(201,203,207,0.5)" },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { title: { display: true, text: "Facility Utilization" } },
      },
    });
  }

  const customerUsage = metrics.customer_usage || {};
  const customerEntries = Object.entries(customerUsage)
    .map(([k, v]) => ({ customer: k, pct: v?.utilization_pct || 0 }))
    .sort((a, b) => b.pct - a.pct)
    .slice(0, 15);
  const customerCtx = document.getElementById("customer-chart")?.getContext("2d");
  if (customerCtx && customerEntries.length > 0) {
    appState.charts.customer = new Chart(customerCtx, {
      type: "bar",
      data: {
        labels: customerEntries.map((e) => e.customer),
        datasets: [
          {
            label: "Utilization %",
            data: customerEntries.map((e) => e.pct),
            backgroundColor: "rgba(75,192,192,0.7)",
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { title: { display: true, text: "Top 15 Customers by Utilization" } },
      },
    });
  }

  const excludedCtx = document.getElementById("exclusion-chart")?.getContext("2d");
  if (excludedCtx) {
    const reasonCounts = {};
    const ruleSummaries = results.rule_summaries || [];
    ruleSummaries.forEach((rs) => {
      if (rs.excluded_rows > 0) {
        reasonCounts[`Rule: ${formatRuleLabel(rs.rule_name)}`] = rs.excluded_rows;
      }
    });

    const notSelectedCount = metrics.not_selected_count || 0;
    if (notSelectedCount > 0) {
      reasonCounts["Optimizer: not selected"] = notSelectedCount;
    }

    const reasonLabels = Object.keys(reasonCounts);
    const reasonValues = Object.values(reasonCounts);
    if (reasonLabels.length > 0) {
      appState.charts.exclusion = new Chart(excludedCtx, {
        type: "doughnut",
        data: {
          labels: reasonLabels,
          datasets: [
            {
              data: reasonValues,
              backgroundColor: [
                "rgba(255,99,132,0.7)",
                "rgba(255,159,64,0.7)",
                "rgba(255,205,86,0.7)",
                "rgba(75,192,192,0.7)",
                "rgba(54,162,235,0.7)",
                "rgba(153,102,255,0.7)",
              ],
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { title: { display: true, text: "Exclusion Breakdown" } },
        },
      });
    }
  }

  const ruleFunnelCtx = document.getElementById("rule-funnel-chart")?.getContext("2d");
  const ruleSummaries = results.rule_summaries || [];
  if (ruleFunnelCtx && ruleSummaries.length > 0) {
    appState.charts.funnel = new Chart(ruleFunnelCtx, {
      type: "bar",
      data: {
        labels: ruleSummaries.map((rs) => formatRuleLabel(rs.rule_name)),
        datasets: [
          {
            label: "Input",
            data: ruleSummaries.map((rs) => rs.input_rows),
            backgroundColor: "rgba(54,162,235,0.5)",
          },
          {
            label: "Output",
            data: ruleSummaries.map((rs) => rs.output_rows),
            backgroundColor: "rgba(75,192,192,0.7)",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { title: { display: true, text: "Rule Funnel" } },
      },
    });
  }
}

async function renderMultiWeekInsightCharts(results) {
  const weeklyExposureRows = normalizeWeeklyExposureRows(extractWeeklyExposureRows(results));
  const planningStartMs = parseWeekStartToUtcMs(results.planning_start_date || results.cohort);
  let optimizerExcludedRows = [];

  try {
    optimizerExcludedRows = await fetchOptimizerExcludedInvoices(appState.currentProcessId);
  } catch (err) {
    console.error("Failed to load optimizer exclusions for charting:", err);
  }
  const deferredRows = normalizeDeferredRows(optimizerExcludedRows, planningStartMs);
  const deferralDriversSeries = (
    buildDeferralDriversSeries(deferredRows)
    || buildDeferralDriversSummarySeries(results.metrics?.deferred_reasons || results.deferred_reasons || {})
  );

  configureInsightLayout({ multiWeek: true, showDeferralChart: Boolean(deferralDriversSeries) });

  const facilityCtx = document.getElementById("facility-chart")?.getContext("2d");
  const headroomSeries = buildFacilityHeadroomSeries(weeklyExposureRows);
  if (facilityCtx && headroomSeries) {
    appState.charts.facility = new Chart(facilityCtx, {
      type: "line",
      data: {
        labels: headroomSeries.labels,
        datasets: [
          {
            label: "Combined bottleneck headroom (€)",
            data: headroomSeries.combinedBottleneckHeadroomSeries,
            borderColor: "rgba(40,167,69,1)",
            backgroundColor: "rgba(40,167,69,0.15)",
            yAxisID: "y",
            pointRadius: 2,
            pointHoverRadius: 4,
            tension: 0,
          },
          {
            label: "Max facility utilization (%)",
            data: headroomSeries.maxUtilizationPctSeries,
            borderColor: "rgba(220,53,69,1)",
            backgroundColor: "rgba(220,53,69,0.15)",
            yAxisID: "y1",
            pointRadius: 2,
            pointHoverRadius: 4,
            tension: 0,
          },
          {
            label: "Max customer utilization (%)",
            data: headroomSeries.maxCustomerUtilizationPctSeries,
            borderColor: "rgba(23,162,184,1)",
            backgroundColor: "rgba(23,162,184,0.12)",
            borderDash: [6, 4],
            yAxisID: "y1",
            pointRadius: 1.5,
            pointHoverRadius: 3,
            tension: 0,
          },
          {
            label: "Max group utilization (%)",
            data: headroomSeries.maxGroupUtilizationPctSeries,
            borderColor: "rgba(255,159,64,1)",
            backgroundColor: "rgba(255,159,64,0.12)",
            borderDash: [3, 4],
            yAxisID: "y1",
            pointRadius: 1.5,
            pointHoverRadius: 3,
            tension: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          title: { display: true, text: "Credit Bottleneck Timeline" },
        },
        scales: {
          y: {
            position: "left",
            ticks: {
              callback: (value) => EUR_CURRENCY_FORMATTER.format(Number(value) || 0),
            },
          },
          y1: {
            position: "right",
            grid: { drawOnChartArea: false },
            min: 0,
            max: 100,
            ticks: {
              callback: (value) => `${(Number(value) || 0).toFixed(0)}%`,
            },
          },
        },
      },
    });
  }

  const customerCtx = document.getElementById("customer-chart")?.getContext("2d");
  if (customerCtx && deferralDriversSeries) {
    appState.charts.customer = new Chart(customerCtx, {
      data: {
        labels: deferralDriversSeries.labels,
        datasets: [
          {
            type: "bar",
            label: "Facility-driven deferrals",
            data: deferralDriversSeries.facilityCountSeries,
            backgroundColor: "rgba(220,53,69,0.7)",
            yAxisID: "y",
            stack: "drivers",
            barPercentage: 0.9,
            categoryPercentage: 0.95,
          },
          {
            type: "bar",
            label: "Customer-driven deferrals",
            data: deferralDriversSeries.customerCountSeries,
            backgroundColor: "rgba(23,162,184,0.7)",
            yAxisID: "y",
            stack: "drivers",
            barPercentage: 0.9,
            categoryPercentage: 0.95,
          },
          {
            type: "bar",
            label: "Group-driven deferrals",
            data: deferralDriversSeries.groupCountSeries,
            backgroundColor: "rgba(255,159,64,0.75)",
            yAxisID: "y",
            stack: "drivers",
            barPercentage: 0.9,
            categoryPercentage: 0.95,
          },
          {
            type: "bar",
            label: "Other deferrals",
            data: deferralDriversSeries.otherCountSeries,
            backgroundColor: "rgba(108,117,125,0.7)",
            yAxisID: "y",
            stack: "drivers",
            barPercentage: 0.9,
            categoryPercentage: 0.95,
          },
          {
            type: "line",
            label: "Total deferred amount (€)",
            data: deferralDriversSeries.totalAmountSeries,
            borderColor: "rgba(40,167,69,1)",
            backgroundColor: "rgba(40,167,69,0.15)",
            yAxisID: "y1",
            tension: 0.2,
            pointRadius: 1.5,
            pointHoverRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          title: { display: true, text: "Deferral Drivers by Planning Week" },
        },
        scales: {
          y: {
            position: "left",
            stacked: true,
            beginAtZero: true,
            ticks: {
              precision: 0,
              callback: (value) => `${Number(value) || 0}`,
            },
          },
          y1: {
            position: "right",
            grid: { drawOnChartArea: false },
            beginAtZero: true,
            ticks: {
              callback: (value) => EUR_CURRENCY_FORMATTER.format(Number(value) || 0),
            },
          },
          x: {
            stacked: true,
          },
        },
      },
    });
  }
}

async function renderCharts(results) {
  Object.values(appState.charts).forEach((c) => {
    if (c && typeof c.destroy === "function") c.destroy();
  });
  appState.charts = {};

  const planningMode = results.planning_mode || results.metrics?.planning_mode || appState.planningMode || "single_week";
  const header = document.getElementById("insight-charts-header");
  if (planningMode === "multi_week") {
    configureInsightLayout({ multiWeek: true, showDeferralChart: false });
    if (header) header.setAttribute("title-text", "Multi-week Planning Insights (Top Signals)");
    await renderMultiWeekInsightCharts(results);
  } else {
    configureInsightLayout({ multiWeek: false, showDeferralChart: false });
    if (header) header.setAttribute("title-text", "Utilization Charts");
    renderSingleWeekUtilizationCharts(results);
  }

  renderInvoiceEvolutionChart(results);
}

async function loadSelectedInvoices() {
  if (!appState.currentProcessId) return;

  const offset = appState.selectedPage * appState.pageSize;
  try {
    const data = await request(
      `/api/optimizer/processes/${appState.currentProcessId}/results/selected?limit=${appState.pageSize}&offset=${offset}`
    );
    renderInvoiceTable("selected-invoices-table", data.invoices, "selected");
    updatePagination("selected", data.total, offset);

    const header = document.getElementById("selected-panel-header");
    if (header) header.setAttribute("title-text", `Selected Invoices (${data.total})`);
  } catch (err) {
    console.error("Failed to load selected invoices:", err);
  }
}

async function loadExcludedInvoices() {
  if (!appState.currentProcessId) return;

  const offset = appState.excludedPage * appState.pageSize;
  const stageFilter = document.getElementById("excluded-stage-filter")?.value || "";
  const reasonFilter = document.getElementById("excluded-reason-filter")?.value || "";

  let url = `/api/optimizer/processes/${appState.currentProcessId}/results/excluded?limit=${appState.pageSize}&offset=${offset}`;
  if (stageFilter) url += `&stage=${encodeURIComponent(stageFilter)}`;
  if (reasonFilter) url += `&reason=${encodeURIComponent(reasonFilter)}`;

  try {
    const data = await request(url);
    renderInvoiceTable("excluded-invoices-table", data.invoices, "excluded");
    updatePagination("excluded", data.total, offset);

    const header = document.getElementById("excluded-panel-header");
    if (header) header.setAttribute("title-text", `Excluded Invoices (${data.total})`);
  } catch (err) {
    console.error("Failed to load excluded invoices:", err);
  }
}

function renderInvoiceTable(tableId, invoices, type) {
  const table = document.getElementById(tableId);
  if (!table) return;

  const existing = table.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
  existing.forEach((r) => r.remove());

  if (!invoices || invoices.length === 0) {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `<ui5-table-cell colspan="6"><div class="table-empty">No invoices</div></ui5-table-cell>`;
    table.appendChild(row);
    return;
  }

  invoices.forEach((inv) => {
    const row = document.createElement("ui5-table-row");
    const price = inv.purchase_price != null ? formatAmount(inv.purchase_price) : "-";

    if (type === "selected") {
      row.innerHTML = `
        <ui5-table-cell>${inv.invoice_ref || "-"}</ui5-table-cell>
        <ui5-table-cell>${inv.company_code || "-"}</ui5-table-cell>
        <ui5-table-cell>${inv.customer || "-"}</ui5-table-cell>
        <ui5-table-cell>${price}</ui5-table-cell>
        <ui5-table-cell>${inv.due_date || "-"}</ui5-table-cell>
        <ui5-table-cell>${inv.status || "-"}</ui5-table-cell>
      `;
    } else {
      row.innerHTML = `
        <ui5-table-cell>${inv.invoice_ref || "-"}</ui5-table-cell>
        <ui5-table-cell>${inv.company_code || "-"}</ui5-table-cell>
        <ui5-table-cell>${inv.customer || "-"}</ui5-table-cell>
        <ui5-table-cell>${price}</ui5-table-cell>
        <ui5-table-cell><span class="status-badge ${inv.excluded_stage || ""}">${inv.excluded_stage || "-"}</span></ui5-table-cell>
        <ui5-table-cell><span class="excluded-reason-text">${inv.excluded_reason || "-"}</span></ui5-table-cell>
      `;
    }
    table.appendChild(row);
  });
}

function updatePagination(type, total, offset) {
  const pageSize = appState.pageSize;
  const currentPage = Math.floor(offset / pageSize) + 1;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  document.getElementById(`${type}-page-info`).textContent = `Page ${currentPage} of ${totalPages}`;
  document.getElementById(`${type}-prev-btn`).disabled = currentPage <= 1;
  document.getElementById(`${type}-next-btn`).disabled = currentPage >= totalPages;
}

async function downloadFile(type) {
  if (!appState.currentProcessId) return;
  const planningMode = getEffectivePlanningModeForDownloads();
  const isWeeklyDownload = type === "weekly-plan" || type === "weekly-exposure";
  if (isWeeklyDownload && planningMode !== "multi_week") {
    showError("Weekly plan and weekly exposure downloads are only available for multi-week optimization.");
    return;
  }

  try {
    const url = `${API_BASE_URL}/api/optimizer/processes/${appState.currentProcessId}/download/${type}`;
    const response = await fetch(url, {
      headers: { "X-API-Key": API_KEY },
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = downloadUrl;

    const extensions = {
      selected: "xlsx",
      excluded: "xlsx",
      "weekly-plan": "xlsx",
      "weekly-exposure": "xlsx",
      summary: "md",
      pdf: "pdf",
      report: "zip",
    };
    a.download = `optimizer_${type}.${extensions[type] || "bin"}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(downloadUrl);
  } catch (err) {
    showError(`Download failed: ${err.message}`);
  }
}

// UI helpers

function setupHelpPopover(iconId, popoverId) {
  const icon = document.getElementById(iconId);
  const popover = document.getElementById(popoverId);
  if (!icon || !popover) return;
  icon.addEventListener("click", () => {
    if (popover.open) {
      popover.open = false;
    } else {
      popover.opener = iconId;
      popover.open = true;
    }
  });
}

function showProgress() {
  const el = document.getElementById("opt-progress-section");
  if (el) el.style.display = "block";
  appState.processStartedAtMs = null;
  appState.lastElapsedSeconds = 0;
  renderOptimizationProgress(null, "running");
  hideDashboard();
}

function hideProgress() {
  const el = document.getElementById("opt-progress-section");
  if (el) el.style.display = "none";
  appState.lastProgress = null;
  appState.processStartedAtMs = null;
  appState.lastElapsedSeconds = 0;
}

function formatElapsedSeconds(totalSeconds) {
  const seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rem = seconds % 60;
  return `${minutes}m ${rem}s`;
}

function renderOptimizationProgress(progress, processStatus = "running", startedAt = null) {
  const stepEl = document.getElementById("opt-progress-step");
  const detailEl = document.getElementById("opt-progress-details");
  const elapsedEl = document.getElementById("opt-progress-text");
  if (!stepEl || !detailEl || !elapsedEl) return;

  if (startedAt && appState.processStartedAtMs == null) {
    const parsed = Date.parse(String(startedAt));
    if (!Number.isNaN(parsed)) {
      appState.processStartedAtMs = parsed;
    }
  }

  if (!progress || typeof progress !== "object") {
    stepEl.textContent = "Step 0/10: Starting optimizer process";
    detailEl.textContent = processStatus === "running" ? "Preparing run..." : "Waiting for status updates...";
    const elapsedSeconds = appState.processStartedAtMs == null
      ? 0
      : Math.max(0, Math.floor((Date.now() - appState.processStartedAtMs) / 1000));
    appState.lastElapsedSeconds = Math.max(appState.lastElapsedSeconds, elapsedSeconds);
    elapsedEl.textContent = `Elapsed: ${formatElapsedSeconds(appState.lastElapsedSeconds)}`;
    return;
  }

  appState.lastProgress = progress;
  const stepIndex = Number(progress.step_index || 0);
  const stepTotal = Number(progress.step_total || 10);
  const label = progress.step_label || "Running optimizer";
  stepEl.textContent = `Step ${stepIndex}/${stepTotal}: ${label}`;

  let elapsedSeconds = 0;
  if (appState.processStartedAtMs != null) {
    elapsedSeconds = Math.max(0, Math.floor((Date.now() - appState.processStartedAtMs) / 1000));
  } else {
    elapsedSeconds = Math.max(0, Math.floor(Number(progress.elapsed_seconds) || 0));
  }
  appState.lastElapsedSeconds = Math.max(appState.lastElapsedSeconds, elapsedSeconds);
  elapsedEl.textContent = `Elapsed: ${formatElapsedSeconds(appState.lastElapsedSeconds)}`;

  const details = progress.details || {};
  if (
    Number.isFinite(Number(details.batches_total))
    && Number.isFinite(Number(details.batches_completed))
  ) {
    const batchesTotal = Number(details.batches_total);
    const batchesCompleted = Number(details.batches_completed);
    const pct = batchesTotal > 0
      ? Math.max(0, Math.min(100, Math.round((batchesCompleted / batchesTotal) * 100)))
      : 0;
    detailEl.textContent = `RPT-1 progress: ${pct}%`;
    return;
  }

  if (details && typeof details === "object" && Object.keys(details).length > 0) {
    if (details.skipped) {
      detailEl.textContent = "Skipped for current planning mode.";
      return;
    }
  }
  detailEl.textContent = "Running...";
}

function showDashboard() {
  const el = document.getElementById("opt-dashboard");
  if (el) el.style.display = "block";
}

function hideDashboard() {
  const el = document.getElementById("opt-dashboard");
  if (el) el.style.display = "none";
  const weekly = document.getElementById("opt-weekly-section");
  if (weekly) weekly.style.display = "none";
  hideWeeklyExposureEvolutionSection();
  const evolution = document.getElementById("invoice-evolution-section");
  if (evolution) evolution.style.display = "none";
  const evolutionNote = document.getElementById("invoice-evolution-note");
  if (evolutionNote) evolutionNote.style.display = "none";
}

function updateSelectedAmountDisplay(value) {
  const amountEl = document.getElementById("opt-selected-amount");
  if (!amountEl) return;

  const displayAmount = formatAmount(value);
  amountEl.textContent = displayAmount;

  amountEl.classList.remove(
    "stat-value--amount-medium",
    "stat-value--amount-small",
    "stat-value--amount-xsmall"
  );

  const displayLength = displayAmount.replace(/\s/g, "").length;
  if (displayLength >= 16) {
    amountEl.classList.add("stat-value--amount-xsmall");
  } else if (displayLength >= 14) {
    amountEl.classList.add("stat-value--amount-small");
  } else if (displayLength >= 12) {
    amountEl.classList.add("stat-value--amount-medium");
  }
}

function formatAmount(value) {
  if (value == null) return "-";
  return new Intl.NumberFormat("en-US", {
    style: "decimal",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function showError(message) {
  const el = document.getElementById("opt-error");
  const status = document.getElementById("opt-status");
  if (!message) {
    if (el) el.style.display = "none";
    return;
  }
  if (el) {
    el.textContent = message;
    el.style.display = "block";
  }
  if (status) status.style.display = "flex";
  // Hide success
  const suc = document.getElementById("opt-success");
  if (suc) suc.style.display = "none";
}

function showSuccess(message) {
  const el = document.getElementById("opt-success");
  const status = document.getElementById("opt-status");
  if (el) {
    el.textContent = message;
    el.style.display = "block";
  }
  if (status) status.style.display = "flex";
  // Hide error
  const err = document.getElementById("opt-error");
  if (err) err.style.display = "none";
  // Auto-hide after 5s
  setTimeout(() => {
    if (el) el.style.display = "none";
  }, 5000);
}
