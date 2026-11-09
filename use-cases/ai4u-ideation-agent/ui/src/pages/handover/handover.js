/**
 * Handover workspace page logic.
 *
 * Handles: list view, detail form, dynamic tables, business impact,
 * AI capabilities, checklist, file uploads (per-zone), context sources,
 * progress tracking, and autosave with toast feedback.
 */

import {
  createHandover,
  listHandovers,
  getHandover,
  updateHandover,
  completeHandover,
  deleteHandover,
  exportHandover,
  exportHandoverAs,
  uploadFile,
  listFiles,
  deleteFile,
  getHandoverFileUrl,
  addContext,
  addContextFile,
  listContext,
  removeContext,
  prefillFromContext,
  sendAgentMessage,
  getAgentHistory,
} from "../../services/handover.js";
import { request, getAuthHeaders } from "../../services/api.js";

/* UI5 Web Components */
import "@ui5/webcomponents/dist/ComboBox.js";
import "@ui5/webcomponents/dist/ComboBoxItem.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/DatePicker.js";
import "@ui5/webcomponents/dist/CheckBox.js";
import "@ui5/webcomponents/dist/Toast.js";
import "@ui5/webcomponents/dist/FileUploader.js";
import "@ui5/webcomponents/dist/Tag.js";
import "@ui5/webcomponents/dist/SegmentedButton.js";
import "@ui5/webcomponents/dist/SegmentedButtonItem.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Link.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";

// ── Constants ─────────────────────────────────────────────────────────────

// AI capabilities (12 predefined options with descriptions + unique icons)
const AI_CAPABILITIES = [
  { id: "doc_understanding", label: "Document Understanding", desc: "OCR, PDF parsing, form extraction", icon: "document-text" },
  { id: "classification", label: "Classification", desc: "Categorize documents, tickets, or requests", icon: "sorting-ranking" },
  { id: "anomaly_detection", label: "Anomaly Detection", desc: "Flag exceptions, outliers, unusual patterns", icon: "alert" },
  { id: "nlu", label: "Natural Language Understanding", desc: "Interpret text, extract entities, sentiment", icon: "text-formatting" },
  { id: "predictive", label: "Predictive / Forecasting", desc: "Demand planning, risk scoring, time-series", icon: "trend-up" },
  { id: "recommendation", label: "Recommendation Engine", desc: "Suggest next actions, products, or configurations", icon: "lightbulb" },
  { id: "conversational", label: "Conversational AI / Joule", desc: "Chatbots, virtual assistants, Q&A", icon: "discussion" },
  { id: "agentic", label: "Agentic / Multi-Step Automation", desc: "Orchestrated workflows, tool-using agents", icon: "process" },
  { id: "rag", label: "RAG (Retrieval-Augmented Generation)", desc: "Grounded answers from enterprise docs", icon: "database" },
  { id: "computer_vision", label: "Computer Vision", desc: "Image analysis, defect detection, spatial reasoning", icon: "camera" },
  { id: "generative_text", label: "Generative Text", desc: "Drafting, summarization, translation", icon: "write-new-document" },
  { id: "other", label: "Other", desc: "Specify in notes", icon: "add" },
];

// Dynamic table schemas (all 9 tables)
const TABLE_SCHEMAS = {
  data_sources: {
    columns: [
      { key: "name", label: "Source Name", placeholder: "e.g. SAP S/4HANA" },
      { key: "system", label: "System", placeholder: "e.g. ERP, CRM" },
      { key: "data_type", label: "Data Type", placeholder: "e.g. Transactional" },
      { key: "owner", label: "Owner", placeholder: "Team or person" },
      { key: "accessibility", label: "Access", placeholder: "API, Export, Manual" },
      { key: "export_method", label: "Export", placeholder: "e.g. OData, BAPI, File" },
      { key: "location", label: "Location", placeholder: "e.g. On-prem, Cloud" },
    ],
    initialRows: 4,
  },
  business_rules: {
    columns: [
      { key: "description", label: "Rule", placeholder: "e.g. 3-way match required" },
      { key: "condition", label: "Condition (If)", placeholder: "e.g. Invoice > €10k" },
      { key: "action", label: "Action (Then)", placeholder: "e.g. Route to manager" },
      { key: "source", label: "Source", placeholder: "e.g. SOP, Policy doc" },
    ],
    initialRows: 4,
    prefilled: [
      { description: "Standard case" },
      { description: "Exception - low severity" },
      { description: "Exception - high severity" },
      { description: "Escalation path" },
    ],
  },
  thresholds: {
    columns: [
      { key: "field", label: "Field / Variable", placeholder: "e.g. Invoice amount" },
      { key: "range", label: "Threshold / Range", placeholder: "e.g. < €1,000" },
      { key: "action_below", label: "Action Below", placeholder: "e.g. Auto-approve" },
      { key: "action_above", label: "Action Above", placeholder: "e.g. Manual review" },
    ],
    initialRows: 3,
  },
  approvals: {
    columns: [
      { key: "scenario", label: "Scenario", placeholder: "Scenario type" },
      { key: "approver", label: "Approver", placeholder: "Role or person" },
      { key: "escalation", label: "Escalation Path", placeholder: "Next level" },
      { key: "sla", label: "SLA", placeholder: "e.g. 24 hours" },
    ],
    initialRows: 4,
    prefilled: [
      { scenario: "Standard case" },
      { scenario: "Exception - low severity" },
      { scenario: "Exception - high severity" },
      { scenario: "Blocked or rejected" },
    ],
  },
  edge_cases: {
    columns: [
      { key: "case", label: "Edge Case", placeholder: "e.g. Duplicate invoice" },
      { key: "description", label: "Description", placeholder: "When/how it occurs" },
      { key: "handling", label: "How to Handle", placeholder: "Expected resolution" },
    ],
    initialRows: 3,
  },
  kpis: {
    columns: [
      { key: "metric", label: "KPI / Metric", placeholder: "e.g. Processing time" },
      { key: "current", label: "Current Baseline", placeholder: "e.g. 4.2 days" },
      { key: "target", label: "Target with AI", placeholder: "e.g. < 1 day" },
    ],
    initialRows: 3,
  },
  tech_components: {
    columns: [
      { key: "deployed", label: "✓", type: "checkbox" },
      { key: "component", label: "Component", static: true },
      { key: "version", label: "Version / Release", placeholder: "e.g. 2023 FPS02" },
      { key: "deployment", label: "Deployment", placeholder: "Cloud / On-Prem / Private" },
    ],
    initialRows: 9,
    prefilled: [
      { component: "SAP S/4HANA", deployed: false },
      { component: "SAP BTP", deployed: false },
      { component: "SAP AI Core / Gen AI Hub", deployed: false },
      { component: "SAP Ariba", deployed: false },
      { component: "SAP IBP", deployed: false },
      { component: "SAP TM / EWM", deployed: false },
      { component: "SAP SuccessFactors", deployed: false },
      { component: "Other SAP (specify)", deployed: false },
      { component: "Non-SAP system (specify)", deployed: false },
    ],
  },
  integrations: {
    columns: [
      { key: "system_a", label: "System A", placeholder: "e.g. S/4HANA" },
      { key: "system_b", label: "System B", placeholder: "e.g. Ariba" },
      { key: "method", label: "Method", placeholder: "e.g. API, RFC, File" },
      { key: "frequency", label: "Frequency", placeholder: "e.g. Real-time, Batch" },
    ],
    initialRows: 3,
  },
  contacts: {
    columns: [
      { key: "name", label: "Name", placeholder: "Contact person" },
      { key: "role", label: "Role", placeholder: "e.g. IT Architect" },
      { key: "contact", label: "Email / Phone", placeholder: "Contact info" },
      { key: "responsibility", label: "Responsibility", placeholder: "For this POC" },
    ],
    initialRows: 3,
  },
};

// Scalar form fields (id prefix "f-")
const SCALAR_FIELDS = [
  "customer", "industry", "workshop_date", "workshop_format",
  "advisor_name", "primary_contact", "tech_contact",
  "usecase_title", "priority", "problem_statement",
  "process_name", "process_freq", "process_volume", "process_team", "process_people",
  "data_input", "data_output", "data_usage",
  "tobe_narrative", "in_scope", "out_scope", "done_definition",
  "open_notes",
];

// Required fields for progress calculation
const REQUIRED_FIELDS = [
  "customer", "industry", "usecase_title", "problem_statement",
  "process_name", "process_freq", "data_input", "tobe_narrative",
];

// Section field map: defines which fields belong to each section for badge counting
// Fields prefixed with "f-" are scalar inputs; special keys use custom logic
const SECTION_FIELDS = {
  "section-customer": {
    fields: ["customer", "industry", "workshop_date", "advisor_name", "primary_contact", "tech_contact", "workshop_format"],
    required: ["customer", "industry"],
  },
  "section-usecase": {
    fields: ["usecase_title", "problem_statement", "priority"],
    required: ["usecase_title", "problem_statement"],
  },
  "section-process": {
    fields: ["process_name", "process_freq", "process_volume", "process_team", "process_people"],
    required: ["process_name", "process_freq"],
  },
  "section-data": {
    fields: ["data_input", "data_output", "data_usage"],
    tables: ["data_sources"],
    required: ["data_input"],
  },
  "section-impact": {
    special: "business_impact", // count filled impact rows
  },
  "section-ai": {
    special: "ai_capabilities", // count selected capabilities
  },
  "section-tobe": {
    fields: ["tobe_narrative", "in_scope", "out_scope", "done_definition"],
    required: ["tobe_narrative"],
  },
  "section-rules": {
    tables: ["business_rules", "thresholds", "approvals", "edge_cases", "kpis"],
  },
  "section-tech": {
    tables: ["tech_components", "integrations", "contacts"],
  },
  "section-checklist": {
    fields: ["open_notes"],
    special: "checklist", // count checked items
  },
};

let _currentHandoverId = null;
let _saveDebounce = null;
let _pendingDeleteId = null;
let _menuTargetId = null;
let _actionBusy = false;
let _currentCompletionPct = 0;
let _selectedConversationId = null; // tracks create-dialog conversation pick by ID

/** Toggle loading spinner + disabled state on a ui5-button. */
function _setBtnLoading(btn, loading) {
  if (!btn) return;
  btn.loading = loading;
  btn.loadingDelay = 0;
  btn.disabled = loading;
}

// ── Initialization ─────────────────────────────────────────────────────

export default function initHandoverPage() {
  const backBtn = document.getElementById("handover-back-btn");
  const newBtn = document.getElementById("handover-new-btn");
  const exportBtn = document.getElementById("handover-export-btn");
  const moreBtn = document.getElementById("handover-more-btn");
  const addContextBtn = document.getElementById("add-context-btn");

  // Navigation
  backBtn?.addEventListener("click", showListView);
  newBtn?.addEventListener("click", openCreateDialog);

  // Export menu (dropdown on Export button)
  exportBtn?.addEventListener("click", () => {
    const menu = document.getElementById("handover-export-menu");
    if (menu) {
      menu.opener = exportBtn;
      menu.open = true;
    }
  });

  const exportMenu = document.getElementById("handover-export-menu");
  exportMenu?.addEventListener("item-click", async (e) => {
    const text = e.detail?.item?.getAttribute("text") || e.detail?.item?.text;
    const formatMap = { PDF: "pdf", Markdown: "markdown", JSON: "json", "ZIP (with files)": "zip" };
    const format = formatMap[text];
    if (!format || !_currentHandoverId) return;

    _setBtnLoading(exportBtn, true);
    try {
      if (format === "json") {
        const data = await exportHandover(_currentHandoverId);
        downloadJson(data);
      } else {
        await exportHandoverAs(_currentHandoverId, format);
      }
    } catch (err) {
      console.error(`Failed to export ${format}:`, err);
      showToast("Export failed");
    } finally {
      _setBtnLoading(exportBtn, false);
    }
  });

  // Overflow menu (More button)
  moreBtn?.addEventListener("click", () => {
    const menu = document.getElementById("handover-detail-menu");
    if (menu) {
      menu.opener = moreBtn;
      menu.open = true;
    }
  });

  const detailMenu = document.getElementById("handover-detail-menu");
  detailMenu?.addEventListener("item-click", (e) => {
    const item = e.detail?.item;
    const text = item?.getAttribute("text") || item?.text;
    if (text === "Mark as Validated") {
      handleComplete();
    } else if (text === "Duplicate") {
      handleDuplicate();
    } else if (text === "Delete") {
      showDeleteConfirmation(_currentHandoverId);
    }
  });

  // Delete confirmation dialog
  const deleteCancel = document.getElementById("handover-delete-cancel");
  const deleteConfirm = document.getElementById("handover-delete-confirm");
  deleteCancel?.addEventListener("click", hideDeleteDialog);
  deleteConfirm?.addEventListener("click", handleConfirmDelete);

  // Create handover dialog
  const createCancel = document.getElementById("create-handover-cancel");
  const createConfirm = document.getElementById("create-handover-confirm");
  createCancel?.addEventListener("click", closeCreateDialog);
  createConfirm?.addEventListener("click", handleCreateConfirm);

  // Auto-fill from conversation selection
  const convSelect = document.getElementById("create-handover-conversation");
  convSelect?.addEventListener("selection-change", handleConversationSelect);

  // Reset title validation on input
  const createTitleInput = document.getElementById("create-handover-title");
  createTitleInput?.addEventListener("input", () => {
    if (createTitleInput.valueState === "Negative") {
      createTitleInput.valueState = "None";
    }
  });

  // Card overflow menu actions
  const cardMenu = document.getElementById("handover-card-menu");
  cardMenu?.addEventListener("item-click", async (e) => {
    const clickedItem = e.detail?.item;
    const text = clickedItem?.getAttribute("text") || clickedItem?.text;
    if (!_menuTargetId) return;

    const card = document.querySelector(`[data-handover-id="${CSS.escape(_menuTargetId)}"]`);

    if (text === "Delete") {
      showDeleteConfirmation(_menuTargetId);
    } else if (text === "PDF" || text === "Markdown" || text === "JSON") {
      const formatMap = { PDF: "pdf", Markdown: "markdown", JSON: "json", "ZIP (with files)": "zip" };
      const format = formatMap[text];
      if (card) card.classList.add("item-busy");
      try {
        if (format === "json") {
          const data = await exportHandover(_menuTargetId);
          downloadJson(data);
        } else {
          await exportHandoverAs(_menuTargetId, format);
        }
      } catch (err) {
        console.error(`Failed to export ${format}:`, err);
        showToast("Export failed");
      } finally {
        if (card) card.classList.remove("item-busy");
      }
    } else if (text === "Duplicate") {
      if (card) card.classList.add("item-busy");
      try {
        const source = await getHandover(_menuTargetId);
        const payload = { ...source };
        delete payload.id;
        delete payload.status;
        delete payload.created_at;
        delete payload.updated_at;
        delete payload.linked_conversation_id;
        delete payload.submitted_at;
        payload.usecase_title = (payload.usecase_title || "Untitled") + " (copy)";
        const result = await createHandover({ payload });
        showToast("Duplicate created");
        await loadHandoverList();
      } catch (err) {
        console.error("Failed to duplicate:", err);
        showToast("Duplicate failed");
      } finally {
        if (card) card.classList.remove("item-busy");
      }
    }
    _menuTargetId = null;
  });

  // Context sources
  addContextBtn?.addEventListener("click", handleAddContext);

  // Inline title editing
  initTitleEdit();

  // Initialize AI capabilities grid
  initAiCapabilities();

  // Initialize dynamic tables
  initDynamicTables();

  // Initialize upload zones
  initUploadZones();

  // Initialize checklist
  initChecklist();

  // Initialize custom panel headers (expand/collapse on click)
  initPanelHeaders();

  // Initialize required field touch tracking (for inline validation)
  initFieldTouchTracking();

  // Auto-save on field changes (debounced)
  initAutoSave();

  // Initialize agent chat
  initAgentChat();

  // Initialize prefill button (independent of agent chat)
  initPrefillButton();

  // Check if we should open a specific handover (from URL hash or query)
  const urlParams = new URLSearchParams(window.location.search);
  const openId = urlParams.get("id");
  if (openId) {
    openHandover(openId);
  } else {
    showListView();
  }
}

// ── Dynamic Tables Engine ─────────────────────────────────────────────────

function initDynamicTables() {
  for (const [tableKey, schema] of Object.entries(TABLE_SCHEMAS)) {
    const container = document.getElementById(`table-${tableKey}`);
    if (!container) continue;
    renderTable(container, tableKey, schema);
  }
}

function renderTable(container, tableKey, schema, existingData = null) {
  const cols = schema.columns;
  const numCols = cols.length + 1; // +1 for delete column

  // Determine rows to render
  let rowCount = schema.initialRows;
  if (existingData && existingData.length > rowCount) {
    rowCount = existingData.length + 1; // +1 empty for adding
  }

  let html = `<table class="step-table" data-table-key="${tableKey}">`;
  html += "<thead><tr>";
  for (const col of cols) {
    if (col.type === "checkbox") {
      html += `<th class="th-checkbox">${_esc(col.label)}</th>`;
    } else {
      html += `<th>${_esc(col.label)}</th>`;
    }
  }
  html += `<th class="td-delete"></th>`;
  html += "</tr></thead><tbody>";

  for (let i = 0; i < rowCount; i++) {
    html += buildTableRow(tableKey, schema, i, existingData?.[i]);
  }

  html += "</tbody></table>";
  container.innerHTML = html;

  // Bind events
  bindTableEvents(container, tableKey, schema);
}

function buildTableRow(tableKey, schema, index, rowData = null) {
  const cols = schema.columns;
  const prefilled = schema.prefilled?.[index];

  let html = `<tr data-row-index="${index}">`;
  for (const col of cols) {
    const savedValue = rowData?.[col.key] ?? "";
    const prefilledValue = prefilled?.[col.key] ?? "";
    const value = savedValue !== "" ? savedValue : prefilledValue;

    if (col.type === "checkbox") {
      const checked = value === true || value === "true";
      html += `<td class="td-checkbox"><input type="checkbox" data-col="${col.key}" ${checked ? "checked" : ""}></td>`;
    } else if (col.static && prefilled?.[col.key]) {
      // Static text (not editable) for prefilled rows
      html += `<td class="td-static" data-col="${col.key}" data-value="${_escAttr(String(value))}">${_esc(String(value))}</td>`;
    } else {
      const placeholder = col.placeholder || "";
      html += `<td><input class="td-input" data-col="${col.key}" value="${_escAttr(String(value))}" placeholder="${_escAttr(placeholder)}"></td>`;
    }
  }
  html += `<td class="td-delete"><button class="row-delete" title="Remove row">&times;</button></td>`;
  html += "</tr>";
  return html;
}

function bindTableEvents(container, tableKey, schema) {
  const table = container.querySelector(".step-table");
  if (!table) return;

  // Input event: auto-add row when typing in last row
  table.addEventListener("input", (e) => {
    if (!e.target.classList.contains("td-input")) return;
    const row = e.target.closest("tr");
    const tbody = table.querySelector("tbody");
    const rows = tbody.querySelectorAll("tr");
    const lastRow = rows[rows.length - 1];

    if (row === lastRow && e.target.value.trim()) {
      // Add a new empty row
      const newIndex = rows.length;
      const wrapper = document.createElement("tbody");
      wrapper.innerHTML = buildTableRow(tableKey, schema, newIndex);
      const newRow = wrapper.querySelector("tr");
      tbody.appendChild(newRow);
      bindRowDelete(newRow, table);
    }

    updateProgress();
    scheduleSave();
  });

  // Checkbox change events (for checkbox columns like "deployed")
  table.addEventListener("change", (e) => {
    if (e.target.type === "checkbox") {
      updateProgress();
      scheduleSave();
    }
  });

  // Delete buttons
  table.querySelectorAll(".row-delete").forEach((btn) => {
    bindRowDelete(btn.closest("tr"), table);
  });
}

function bindRowDelete(row, table) {
  const btn = row.querySelector(".row-delete");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const tbody = table.querySelector("tbody");
    const rows = tbody.querySelectorAll("tr");
    // Don't delete the last remaining row
    if (rows.length <= 1) return;
    row.remove();
    updateProgress();
    scheduleSave();
  });
}

function collectTableData(tableKey) {
  const container = document.getElementById(`table-${tableKey}`);
  if (!container) return [];

  const schema = TABLE_SCHEMAS[tableKey];
  const rows = container.querySelectorAll(".step-table tbody tr");
  const data = [];

  rows.forEach((row) => {
    const rowData = {};
    let hasValue = false;
    for (const col of schema.columns) {
      if (col.type === "checkbox") {
        const checkbox = row.querySelector(`input[type="checkbox"][data-col="${col.key}"]`);
        rowData[col.key] = checkbox?.checked || false;
        if (rowData[col.key]) hasValue = true;
      } else if (col.static) {
        // Static columns store their value in data-value or text
        const cell = row.querySelector(`[data-col="${col.key}"]`);
        const value = cell?.dataset?.value || cell?.textContent?.trim() || "";
        rowData[col.key] = value;
        if (value) hasValue = true;
      } else {
        const input = row.querySelector(`input[data-col="${col.key}"]`);
        const value = input?.value?.trim() || "";
        rowData[col.key] = value;
        if (value) hasValue = true;
      }
    }
    // Only include rows with at least one non-empty field
    if (hasValue) data.push(rowData);
  });

  return data;
}

/**
 * Check if a table has user-entered data beyond its prefilled defaults.
 * A row counts as "user data" if it has values in columns that aren't
 * part of the prefilled template for that row index.
 */
function hasUserTableData(tableKey) {
  const schema = TABLE_SCHEMAS[tableKey];
  if (!schema) return false;

  const container = document.getElementById(`table-${tableKey}`);
  if (!container) return false;

  const rows = container.querySelectorAll(".step-table tbody tr");
  const prefilled = schema.prefilled || [];

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const prefilledRow = prefilled[i] || {};

    for (const col of schema.columns) {
      // Skip columns that have a prefilled value for this row (static labels, default checkboxes)
      if (col.key in prefilledRow) {
        if (col.type === "checkbox") {
          // Checkbox counts if user CHECKED it (prefilled is false)
          const cb = row.querySelector(`input[type="checkbox"][data-col="${col.key}"]`);
          if (cb?.checked && !prefilledRow[col.key]) return true;
        }
        continue; // static/prefilled column — skip
      }

      // Non-prefilled column — any value means user input
      if (col.type === "checkbox") {
        const cb = row.querySelector(`input[type="checkbox"][data-col="${col.key}"]`);
        if (cb?.checked) return true;
      } else {
        const input = row.querySelector(`input[data-col="${col.key}"]`);
        if (input?.value?.trim()) return true;
      }
    }
  }

  // Also check rows beyond the prefilled count (user-added rows)
  if (rows.length > prefilled.length && rows.length > schema.initialRows) {
    return true;
  }

  return false;
}

function hydrateTable(tableKey, data) {
  const container = document.getElementById(`table-${tableKey}`);
  if (!container) return;
  const schema = TABLE_SCHEMAS[tableKey];
  renderTable(container, tableKey, schema, data);
}

// ── Business Impact Table ─────────────────────────────────────────────────

function collectBusinessImpact() {
  const impact = {};
  const inputs = document.querySelectorAll(".impact-table .td-input[data-impact]");
  inputs.forEach((input) => {
    const key = input.dataset.impact;
    const col = input.dataset.col; // "current" or "target"
    if (!impact[key]) impact[key] = {};
    impact[key][col] = input.value.trim();
  });

  return impact;
}

function hydrateBusinessImpact(data) {
  if (!data) return;
  const inputs = document.querySelectorAll(".impact-table .td-input[data-impact]");
  inputs.forEach((input) => {
    const key = input.dataset.impact;
    const col = input.dataset.col;
    if (data[key] != null && data[key][col] != null && data[key][col] !== "") {
      input.value = data[key][col];
    }
  });
}

// ── AI Capabilities ────────────────────────────────────────────────────

function initAiCapabilities() {
  const grid = document.getElementById("ai-capabilities-grid");
  if (!grid) return;

  grid.innerHTML = AI_CAPABILITIES.map((cap) => `
    <div class="ai-cap-item" data-capability="${_escAttr(cap.id)}">
      <ui5-icon name="${_escAttr(cap.icon)}"></ui5-icon>
      <div class="ai-cap-text">
        <span class="ai-cap-label">${_esc(cap.label)}</span>
        <span class="ai-cap-desc">${_esc(cap.desc)}</span>
      </div>
    </div>
  `).join("");

  grid.querySelectorAll(".ai-cap-item").forEach((el) => {
    el.addEventListener("click", () => {
      el.classList.toggle("selected");
      updateProgress();
      scheduleSave();
    });
  });
}

// ── Checklist ─────────────────────────────────────────────────────────────

function initChecklist() {
  const checkboxes = document.querySelectorAll("[data-chk]");
  checkboxes.forEach((cb) => {
    cb.addEventListener("change", () => {
      scheduleSave();
    });
  });

  // Clickable section references: click "(§N)" → expand + scroll to that section
  const SECTION_MAP = {
    1: "section-customer",
    2: "section-usecase",
    3: "section-process",
    4: "section-data",
    5: "section-impact",
    6: "section-ai",
    7: "section-tobe",
    8: "section-rules",
    9: "section-tech",
    10: "section-checklist",
  };

  const checklistContainer = document.getElementById("section-checklist");
  if (checklistContainer) {
    checklistContainer.addEventListener("click", (e) => {
      const target = e.target;
      // Check if click is on or near a checkbox label containing §N
      const text = target?.textContent || "";
      const match = text.match(/§(\d+)/);
      if (!match) return;
      // Don't navigate if user is clicking the checkbox itself
      if (target.tagName === "UI5-CHECKBOX") return;

      const sectionNum = parseInt(match[1], 10);
      const panelId = SECTION_MAP[sectionNum];
      if (!panelId) return;

      const panel = document.getElementById(panelId);
      if (panel) {
        panel.collapsed = false;
        panel.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  }
}

function collectChecklist() {
  const items = [];
  document.querySelectorAll("[data-chk]").forEach((cb) => {
    items.push({
      key: cb.dataset.chk,
      checked: cb.checked || false,
    });
  });
  return items;
}

function hydrateChecklist(data) {
  if (!data || !Array.isArray(data)) return;
  data.forEach((item) => {
    const cb = document.querySelector(`[data-chk="${CSS.escape(item.key)}"]`);
    if (cb) cb.checked = !!item.checked;
  });
}

// ── Upload Zones ──────────────────────────────────────────────────────────

function initUploadZones() {
  // Bind file-uploader change events
  document.querySelectorAll("[data-zone-upload]").forEach((uploader) => {
    uploader.addEventListener("change", async (e) => {
      const zone = uploader.dataset.zoneUpload;
      const files = e.detail?.files || e.target?.files;
      if (!files?.length || !_currentHandoverId) return;

      const zoneEl = uploader.closest(".inline-upload");
      if (zoneEl) zoneEl.classList.add("upload-zone-busy");

      let successCount = 0;
      let failCount = 0;
      try {
        for (const file of files) {
          try {
            await uploadFile(_currentHandoverId, file, zone);
            successCount++;
          } catch (err) {
            failCount++;
            console.error(`Failed to upload ${file.name}:`, err);
          }
        }
        if (failCount > 0 && successCount > 0) {
          showToast(`${successCount} uploaded, ${failCount} failed`);
        } else if (failCount > 0) {
          showToast(`Upload failed (${failCount} file${failCount > 1 ? "s" : ""})`);
        } else {
          showToast(`${successCount} file${successCount > 1 ? "s" : ""} uploaded`);
        }
        await loadFilesByZone(_currentHandoverId);
      } finally {
        if (zoneEl) zoneEl.classList.remove("upload-zone-busy");
      }
    });
  });

  // Drag-and-drop per inline-upload zone
  document.querySelectorAll(".inline-upload").forEach((zone) => {
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => {
      zone.classList.remove("dragover");
    });
    zone.addEventListener("drop", async (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      if (!_currentHandoverId) return;

      const zoneName = zone.dataset.zone;
      const files = e.dataTransfer?.files;
      if (!files?.length) return;

      zone.classList.add("upload-zone-busy");

      let successCount = 0;
      let failCount = 0;
      try {
        for (const file of files) {
          try {
            await uploadFile(_currentHandoverId, file, zoneName);
            successCount++;
          } catch (err) {
            failCount++;
            console.error(`Failed to upload ${file.name}:`, err);
          }
        }
        if (failCount > 0 && successCount > 0) {
          showToast(`${successCount} uploaded, ${failCount} failed`);
        } else if (failCount > 0) {
          showToast(`Upload failed (${failCount} file${failCount > 1 ? "s" : ""})`);
        } else {
          showToast(`${successCount} file${successCount > 1 ? "s" : ""} uploaded`);
        }
        await loadFilesByZone(_currentHandoverId);
      } finally {
        zone.classList.remove("upload-zone-busy");
      }
    });
  });
}

async function loadFilesByZone(handoverId) {
  try {
    const files = await listFiles(handoverId);

    // Clear all zone file lists
    document.querySelectorAll("[data-zone-files]").forEach((el) => {
      el.innerHTML = "";
    });

    // Group files by zone
    const byZone = {};
    files.forEach((f) => {
      if (!byZone[f.zone]) byZone[f.zone] = [];
      byZone[f.zone].push(f);
    });

    // Render files into their zones
    for (const [zone, zoneFiles] of Object.entries(byZone)) {
      const container = document.querySelector(`[data-zone-files="${zone}"]`);
      if (!container) continue;

      container.innerHTML = zoneFiles.map((f) => `
        <div class="inline-upload-file">
          <span class="inline-upload-file-name clickable"
                title="${_escAttr(f.filename)}"
                data-preview-file="${_escAttr(f.file_id)}"
                data-preview-filename="${_escAttr(f.filename)}"
                data-preview-ct="${_escAttr(f.content_type || "")}"
                >${_esc(f.filename)}</span>
          <button class="inline-upload-file-delete" data-delete-file="${_escAttr(f.file_id)}" title="Delete">&times;</button>
        </div>
      `).join("");

      // File preview/download on filename click
      container.querySelectorAll("[data-preview-file]").forEach((el) => {
        el.addEventListener("click", (e) => {
          e.stopPropagation();
          _handleFileClick(handoverId, el.dataset.previewFile, el.dataset.previewFilename, el.dataset.previewCt);
        });
      });

      container.querySelectorAll("[data-delete-file]").forEach((btn) => {
        btn.addEventListener("click", async (e) => {
          e.stopPropagation();
          const row = btn.closest(".inline-upload-file");
          if (row) row.classList.add("item-busy");
          try {
            await deleteFile(handoverId, btn.dataset.deleteFile);
            await loadFilesByZone(handoverId);
          } catch (err) {
            console.error("Failed to delete file:", err);
            showToast("Failed to delete file");
            if (row) row.classList.remove("item-busy");
          }
        });
      });
    }
  } catch (err) {
    console.error("Failed to load files:", err);
    showToast("Failed to load files");
  }
}

// ── File Preview / Download ─────────────────────────────────────────────

function _ensureHandoverLightbox() {
  if (document.getElementById("handover-lightbox")) return;
  document.body.insertAdjacentHTML("beforeend", `
    <div id="handover-lightbox" class="handover-lightbox" hidden>
      <button id="handover-lightbox-close" class="lightbox-close" aria-label="Close">&times;</button>
      <ui5-busy-indicator id="handover-lightbox-loader" size="L" active delay="0"></ui5-busy-indicator>
      <img id="handover-lightbox-img" class="lightbox-img" alt="File preview" hidden />
      <iframe id="handover-lightbox-pdf" class="lightbox-pdf" hidden></iframe>
    </div>
  `);

  // Close on backdrop click, close button, Escape
  const lb = document.getElementById("handover-lightbox");
  lb.addEventListener("click", (e) => {
    if (e.target === lb || e.target.id === "handover-lightbox-close") {
      _closeHandoverLightbox();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lb && !lb.hidden) _closeHandoverLightbox();
  });
}

function _closeHandoverLightbox() {
  const lb = document.getElementById("handover-lightbox");
  const img = document.getElementById("handover-lightbox-img");
  const pdf = document.getElementById("handover-lightbox-pdf");
  if (lb) lb.hidden = true;
  if (img) { img.src = ""; img.hidden = true; }
  if (pdf) { pdf.src = ""; pdf.hidden = true; }
}

async function _openHandoverLightboxImage(url) {
  _ensureHandoverLightbox();
  const lb = document.getElementById("handover-lightbox");
  const img = document.getElementById("handover-lightbox-img");
  const pdf = document.getElementById("handover-lightbox-pdf");
  const loader = document.getElementById("handover-lightbox-loader");
  if (!lb || !img) return;

  if (pdf) { pdf.src = ""; pdf.hidden = true; }
  img.src = ""; img.hidden = true;
  if (loader) loader.hidden = false;
  lb.hidden = false;

  try {
    const headers = await getAuthHeaders();
    const res = await fetch(url, { headers });
    if (!res.ok) throw new Error(res.status);
    const blob = await res.blob();
    img.src = URL.createObjectURL(blob);
    img.onload = () => { img.hidden = false; if (loader) loader.hidden = true; };
  } catch {
    lb.hidden = true;
  }
}

async function _openHandoverLightboxPdf(url) {
  _ensureHandoverLightbox();
  const lb = document.getElementById("handover-lightbox");
  const img = document.getElementById("handover-lightbox-img");
  const pdf = document.getElementById("handover-lightbox-pdf");
  const loader = document.getElementById("handover-lightbox-loader");
  if (!lb || !pdf) return;

  if (img) { img.src = ""; img.hidden = true; }
  pdf.hidden = true;
  if (loader) loader.hidden = false;
  lb.hidden = false;

  try {
    const headers = await getAuthHeaders();
    const res = await fetch(url, { headers });
    if (!res.ok) throw new Error(res.status);
    const blob = await res.blob();
    pdf.src = URL.createObjectURL(blob);
    pdf.hidden = false;
    if (loader) loader.hidden = true;
  } catch {
    lb.hidden = true;
  }
}

async function _downloadHandoverFile(url, filename) {
  try {
    const headers = await getAuthHeaders();
    const res = await fetch(url, { headers });
    if (!res.ok) { console.warn("Download failed:", res.status); return; }
    const blob = await res.blob();
    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = filename || "download";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
  } catch (err) {
    console.warn("Download error:", err);
  }
}

/**
 * Handle a file click: preview in lightbox (images/PDFs) or download.
 */
function _handleFileClick(handoverId, fileId, filename, contentType) {
  const url = getHandoverFileUrl(handoverId, fileId);
  const ct = (contentType || "").toLowerCase();
  const ext = (filename || "").split(".").pop()?.toLowerCase() || "";

  if (ct.startsWith("image/") || ["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(ext)) {
    _openHandoverLightboxImage(url);
  } else if (ct === "application/pdf" || ext === "pdf") {
    _openHandoverLightboxPdf(url);
  } else {
    _downloadHandoverFile(url, filename);
  }
}

// ── List View ──────────────────────────────────────────────────────────

async function showListView() {
  await flushPendingSave();
  _currentHandoverId = null;
  document.getElementById("handover-list-view").style.display = "";
  document.getElementById("handover-detail-view").style.display = "none";
  document.getElementById("handover-page-title").textContent = "Handover Packages";
  document.getElementById("handover-back-btn").style.display = "none";
  document.getElementById("handover-title-edit-icon").style.display = "none";
  document.getElementById("handover-title-wrapper").style.cursor = "default";
  document.getElementById("handover-title-wrapper").classList.remove("editable");
  document.getElementById("handover-progress").style.display = "none";
  document.getElementById("handover-new-btn").style.display = "";
  document.getElementById("handover-detail-actions").style.display = "none";

  // Update URL
  history.replaceState(null, "", "/handover");

  const listBusy = document.getElementById("handover-list-busy");
  if (listBusy) listBusy.active = true;
  try {
    await loadHandoverList();
  } finally {
    if (listBusy) listBusy.active = false;
  }
}

async function loadHandoverList() {
  const cardsEl = document.getElementById("handover-list-cards");
  const emptyEl = document.getElementById("handover-list-empty");

  try {
    const data = await listHandovers();
    const items = data.items || [];

    if (items.length === 0) {
      emptyEl.style.display = "";
      cardsEl.innerHTML = "";
      return;
    }

    emptyEl.style.display = "none";
    cardsEl.innerHTML = items.map((item) => {
      const pct = item.completion_pct || 0;
      const isValidated = item.status === "validated";
      const badge = isValidated
        ? `<span class="handover-card-status" data-status="validated">✓ Validated</span>`
        : `<span class="handover-card-pct">${pct}% complete</span>`;
      return `
      <div class="handover-card" data-id="${_escAttr(item.id)}">
        <ui5-button class="handover-card-overflow" data-overflow-id="${_escAttr(item.id)}" icon="overflow" design="Transparent" tooltip="More actions"></ui5-button>
        <div class="handover-card-title">${_esc(item.usecase_title || "Untitled")}</div>
        <div class="handover-card-customer">${_esc(item.customer || "No customer")}</div>
        <div class="handover-card-meta">
          ${badge}
          <span>${_formatDate(item.updated_at)}</span>
        </div>
      </div>`;
    }).join("");

    // Bind click handlers
    cardsEl.querySelectorAll(".handover-card").forEach((card) => {
      card.addEventListener("click", (e) => {
        if (e.target.closest(".handover-card-overflow")) return;
        openHandover(card.dataset.id);
      });
    });
    cardsEl.querySelectorAll(".handover-card-overflow").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        _menuTargetId = btn.dataset.overflowId;
        const menu = document.getElementById("handover-card-menu");
        if (menu) {
          menu.opener = btn;
          menu.open = true;
        }
      });
    });
  } catch (err) {
    console.error("Failed to load handovers:", err);
    cardsEl.innerHTML = `<p style="color: var(--sapNegativeTextColor)">Failed to load handovers.</p>`;
  }
}

// ── Detail View ────────────────────────────────────────────────────────

async function openHandover(handoverId) {
  await flushPendingSave();
  _currentHandoverId = handoverId;
  document.getElementById("handover-list-view").style.display = "none";
  document.getElementById("handover-detail-view").style.display = "";
  document.getElementById("handover-back-btn").style.display = "";
  document.getElementById("handover-title-edit-icon").style.display = "";
  document.getElementById("handover-title-wrapper").style.cursor = "pointer";
  document.getElementById("handover-title-wrapper").classList.add("editable");
  document.getElementById("handover-progress").style.display = "";
  document.getElementById("handover-new-btn").style.display = "none";
  document.getElementById("handover-detail-actions").style.display = "";

  // Update URL
  history.replaceState(null, "", `/handover?id=${handoverId}`);

  const detailBusy = document.getElementById("handover-detail-busy");
  if (detailBusy) detailBusy.active = true;

  try {
    const data = await getHandover(handoverId);

    document.getElementById("handover-page-title").textContent =
      data.usecase_title || "Untitled";

    // Clear stale suggestions from previous handover
    clearSuggestions();
    _pendingSuggestions = {};

    hydrateForm(data);
    updateProgress();
    await loadSidebarData(handoverId);

    // Reset agent chat before loading new history
    const chatMsgs = document.getElementById("agent-chat-messages");
    if (chatMsgs) chatMsgs.querySelectorAll(".agent-msg, .agent-summary-card").forEach((el) => el.remove());
    const agentWelcome = document.getElementById("agent-welcome");
    if (agentWelcome) agentWelcome.style.display = "";

    // Load agent conversation history (non-blocking)
    loadAgentHistory();
  } catch (err) {
    console.error("Failed to load handover:", err);
    showToast("Failed to load handover");
    showListView();
  } finally {
    if (detailBusy) detailBusy.active = false;
  }
}

function hydrateForm(data) {
  // Scalar fields
  for (const field of SCALAR_FIELDS) {
    const el = document.getElementById(`f-${field}`);
    if (!el) continue;
    const value = data[field] || "";
    if (el.tagName === "UI5-SELECT") {
      const options = el.querySelectorAll("ui5-option");
      options.forEach((opt) => {
        opt.selected = opt.getAttribute("value") === value;
      });
    } else {
      el.value = value;
    }
  }

  // AI capabilities
  const selected = data.ai_capabilities || [];
  document.querySelectorAll(".ai-cap-item").forEach((el) => {
    const cap = el.dataset.capability;
    el.classList.toggle("selected", selected.includes(cap));
  });

  // Dynamic tables
  for (const tableKey of Object.keys(TABLE_SCHEMAS)) {
    if (data[tableKey] && Array.isArray(data[tableKey])) {
      hydrateTable(tableKey, data[tableKey]);
    }
  }

  // Business impact
  hydrateBusinessImpact(data.business_impact);

  // Checklist
  hydrateChecklist(data.checklist);

  // Expand first non-empty section or section 2
  expandFirstRelevantSection(data);
}

function expandFirstRelevantSection(_data) {
  // Always open section 1 — the user orients top-down.
  // Section badges and the progress bar guide them to incomplete areas.
  const panel = document.getElementById("section-customer");
  if (panel) panel.collapsed = false;
}

function collectForm() {
  const payload = {};

  // Scalar fields
  for (const field of SCALAR_FIELDS) {
    const el = document.getElementById(`f-${field}`);
    if (!el) continue;
    if (el.tagName === "UI5-SELECT") {
      const selected = el.querySelector("ui5-option[selected]");
      payload[field] = selected ? selected.getAttribute("value") : "";
    } else {
      payload[field] = el.value || "";
    }
  }

  // AI capabilities
  const caps = [];
  document.querySelectorAll(".ai-cap-item.selected").forEach((el) => {
    caps.push(el.dataset.capability);
  });
  payload.ai_capabilities = caps;

  // Dynamic tables (always include — empty [] clears on merge-patch)
  for (const tableKey of Object.keys(TABLE_SCHEMAS)) {
    payload[tableKey] = collectTableData(tableKey);
  }

  // Business impact (always include — empty {} clears on merge-patch)
  payload.business_impact = collectBusinessImpact();

  // Checklist (always include — unchecked state clears on merge-patch)
  payload.checklist = collectChecklist();

  return payload;
}

// ── Progress ───────────────────────────────────────────────────────────

function updateProgress() {
  const payload = collectForm();

  // Compute per-section badges and sum for overall progress
  let totalFilled = 0;
  let totalCount = 0;

  for (const [sectionId, config] of Object.entries(SECTION_FIELDS)) {
    const badge = document.getElementById(`badge-${sectionId}`);
    const { filled, total, requiredDone } = getSectionCompletion(sectionId, config, payload);

    if (badge) updateBadgeDisplay(badge, filled, total, requiredDone);

    // AI capabilities: total is -1 sentinel, count as 1 item
    if (total === -1) {
      totalFilled += filled > 0 ? 1 : 0;
      totalCount += 1;
    } else {
      totalFilled += filled;
      totalCount += total;
    }
  }

  const pct = totalCount > 0 ? Math.round((totalFilled / totalCount) * 100) : 0;
  _currentCompletionPct = pct;

  const fillEl = document.getElementById("handover-progress-fill");
  const textEl = document.getElementById("handover-progress-text");
  if (fillEl) fillEl.style.width = `${pct}%`;
  if (textEl) textEl.textContent = `${pct}%`;

  // Update inline validation on required fields
  updateFieldValidation(payload);
}

// ── Custom Panel Header Toggle ────────────────────────────────────────

function initPanelHeaders() {
  document.querySelectorAll(".panel-header-custom").forEach((header) => {
    header.addEventListener("click", () => {
      const panelId = header.dataset.panel;
      const panel = document.getElementById(panelId);
      if (panel) panel.collapsed = !panel.collapsed;
    });
  });
}

// ── Section Badges ────────────────────────────────────────────────────

function getSectionCompletion(sectionId, config, payload) {
  // Special sections
  if (config.special === "business_impact") {
    const impact = collectBusinessImpact();
    if (!impact) return { filled: 0, total: 6, requiredDone: false };
    let count = 0;
    for (const key of Object.values(impact)) {
      if (key.current || key.target) count++;
    }
    return { filled: count, total: 6, requiredDone: count > 0 };
  }

  if (config.special === "ai_capabilities") {
    const count = document.querySelectorAll(".ai-cap-item.selected").length;
    return { filled: count, total: -1, requiredDone: count > 0 }; // -1 signals "N selected" display
  }

  if (config.special === "checklist") {
    const all = document.querySelectorAll("#section-checklist [data-chk]");
    let count = 0;
    all.forEach((cb) => { if (cb.checked) count++; });
    // Also count scalar fields (e.g. open_notes)
    let fieldsFilled = 0;
    const fieldCount = (config.fields || []).length;
    for (const field of config.fields || []) {
      const el = document.getElementById(`f-${field}`);
      if (el && (el.value || "").trim()) fieldsFilled++;
    }
    return { filled: count + fieldsFilled, total: all.length + fieldCount, requiredDone: count >= 10 };
  }

  // Standard field-based sections
  let filled = 0;
  let total = 0;
  let requiredFilled = 0;
  let requiredTotal = config.required ? config.required.length : 0;

  if (config.fields) {
    total += config.fields.length;
    for (const field of config.fields) {
      const el = document.getElementById(`f-${field}`);
      if (!el) continue;
      const val = el.tagName === "UI5-SELECT"
        ? (el.querySelector("ui5-option[selected]")?.getAttribute("value") || "")
        : (el.value || "").trim();
      if (val) {
        filled++;
        if (config.required?.includes(field)) requiredFilled++;
      }
    }
  }

  if (config.tables) {
    total += config.tables.length;
    for (const tableKey of config.tables) {
      if (hasUserTableData(tableKey)) filled++;
    }
  }

  // requiredDone: all required fields in this section are filled
  // Sections without required fields never reach green counter state
  const requiredDone = requiredTotal > 0
    ? requiredFilled >= requiredTotal
    : false;

  return { filled, total, requiredDone };
}

function updateBadgeDisplay(badge, filled, total, requiredDone) {
  if (total === -1) {
    // "N selected" mode (AI capabilities)
    badge.textContent = filled > 0 ? `${filled} selected` : "0 selected";
    if (requiredDone) {
      badge.design = "Positive";
      badge.colorScheme = "";
      badge.hideStateIcon = false;
    } else {
      badge.design = "Neutral";
      badge.colorScheme = "";
      badge.hideStateIcon = true;
    }
    return;
  }

  const isAllComplete = filled >= total && total > 0;
  const isPartial = filled > 0 && filled < total;

  if (isAllComplete) {
    badge.textContent = `${filled}/${total}`;
    badge.design = "Positive";
    badge.hideStateIcon = false; // subtle checkmark
    badge.colorScheme = "";
  } else if (isPartial) {
    badge.textContent = `${filled}/${total}`;
    badge.design = "Information";
    badge.hideStateIcon = true;
    badge.colorScheme = "";
  } else {
    badge.textContent = `${filled}/${total}`;
    badge.design = "Neutral";
    badge.hideStateIcon = true;
    badge.colorScheme = "";
  }
}

// ── Inline Field Validation ───────────────────────────────────────────

function updateFieldValidation(payload) {
  // No inline validation during editing — section badges and the progress
  // bar communicate completeness. Red errors only appear on explicit
  // "Complete" validation via validateRequiredFields().
  for (const field of REQUIRED_FIELDS) {
    const el = document.getElementById(`f-${field}`);
    if (!el) continue;
    // Clear any previous validation state (e.g. from a failed Complete attempt)
    if (el.valueState === "Negative" && (el.value || "").trim()) {
      el.valueState = "None";
    }
  }
}

/** Mark field as touched when user interacts, to enable validation on clear */
function initFieldTouchTracking() {
  for (const field of REQUIRED_FIELDS) {
    const el = document.getElementById(`f-${field}`);
    if (!el) continue;
    el.addEventListener("input", () => {
      if ((el.value || "").trim()) {
        el.dataset.touched = "true";
      }
    });
  }
}

/** Validate all required fields and show Negative state on empty ones (e.g. before export) */
function validateRequiredFields() {
  let allValid = true;
  const missing = [];

  for (const field of REQUIRED_FIELDS) {
    const el = document.getElementById(`f-${field}`);
    if (!el) continue;
    const val = (el.value || "").trim();
    if (!val) {
      el.valueState = "Negative";
      el.dataset.touched = "true";
      allValid = false;
      missing.push(field);
    }
  }
  return { valid: allValid, missing };
}

// ── Inline Title Edit ──────────────────────────────────────────────────

function initTitleEdit() {
  const wrapper = document.getElementById("handover-title-wrapper");
  const titleEl = document.getElementById("handover-page-title");
  const inputEl = document.getElementById("handover-title-input");
  if (!wrapper || !titleEl || !inputEl) return;

  wrapper.addEventListener("click", () => {
    if (!_currentHandoverId) return;
    const currentTitle = titleEl.textContent || "";
    inputEl.value = currentTitle === "Untitled" ? "" : currentTitle;
    wrapper.style.display = "none";
    inputEl.style.display = "";
    inputEl.focus();
  });

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commitTitleEdit();
    } else if (e.key === "Escape") {
      cancelTitleEdit();
    }
  });

  inputEl.addEventListener("focusout", () => {
    commitTitleEdit();
  });
}

function commitTitleEdit() {
  const wrapper = document.getElementById("handover-title-wrapper");
  const titleEl = document.getElementById("handover-page-title");
  const inputEl = document.getElementById("handover-title-input");
  if (!wrapper || !titleEl || !inputEl) return;

  const newTitle = (inputEl.value || "").trim();
  const displayTitle = newTitle || "Untitled";

  titleEl.textContent = displayTitle;
  wrapper.style.display = "";
  inputEl.style.display = "none";

  const formField = document.getElementById("f-usecase_title");
  if (formField) formField.value = newTitle;

  updateProgress();
  scheduleSave();
}

function cancelTitleEdit() {
  const wrapper = document.getElementById("handover-title-wrapper");
  const inputEl = document.getElementById("handover-title-input");
  if (wrapper) wrapper.style.display = "";
  if (inputEl) inputEl.style.display = "none";
}

// ── Auto-save ──────────────────────────────────────────────────────────

function initAutoSave() {
  const formPanel = document.querySelector(".handover-form-panel");
  if (!formPanel) return;

  formPanel.addEventListener("input", () => {
    updateProgress();
    scheduleSave();
  });
  formPanel.addEventListener("change", () => {
    updateProgress();
    scheduleSave();
  });

  // Bidirectional sync: usecase_title field → header title
  const usecaseTitleField = document.getElementById("f-usecase_title");
  if (usecaseTitleField) {
    usecaseTitleField.addEventListener("input", () => {
      const titleEl = document.getElementById("handover-page-title");
      if (titleEl && _currentHandoverId) {
        titleEl.textContent = usecaseTitleField.value.trim() || "Untitled";
      }
    });
  }

  // Flush pending save when page becomes hidden (side nav, tab close, etc.)
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden" && _saveDebounce) {
      flushPendingSave();
    }
  });

  // Safety net: warn if pending save hasn't flushed before unload
  window.addEventListener("beforeunload", (e) => {
    if (_saveDebounce) {
      e.preventDefault();
    }
  });
}

function scheduleSave() {
  if (_saveDebounce) clearTimeout(_saveDebounce);
  _saveDebounce = setTimeout(() => {
    _saveDebounce = null;
    if (_currentHandoverId) handleSave(true);
  }, 1000);
}

async function flushPendingSave() {
  if (_saveDebounce) {
    clearTimeout(_saveDebounce);
    _saveDebounce = null;
    if (_currentHandoverId) await handleSave(true);
  }
}

// ── Actions ────────────────────────────────────────────────────────────

async function openCreateDialog() {
  const dialog = document.getElementById("create-handover-dialog");
  const titleInput = document.getElementById("create-handover-title");
  const customerInput = document.getElementById("create-handover-customer");
  const convCombo = document.getElementById("create-handover-conversation");
  if (!dialog) return;

  if (titleInput) { titleInput.value = ""; titleInput.valueState = "None"; }
  if (customerInput) customerInput.value = "";
  _selectedConversationId = null;

  if (convCombo) {
    convCombo.value = "";
    convCombo.innerHTML = "";
    try {
      const data = await request("/api/ideation-agent/conversations");
      const convos = (data.conversations || [])
        .sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));
      for (const c of convos) {
        const item = document.createElement("ui5-cb-item");
        item.text = c.title || c.company_name || "Untitled";
        item.setAttribute("data-id", c.id);
        item.value = c.id;
        item.additionalText = _formatDate(c.updated_at);
        item.dataset.customer = c.company_name || "";
        convCombo.appendChild(item);
      }
    } catch (err) {
      console.error("Failed to load conversations:", err);
    }
  }

  dialog.open = true;
}

function closeCreateDialog() {
  const dialog = document.getElementById("create-handover-dialog");
  if (dialog) dialog.open = false;
}

function handleConversationSelect(e) {
  const titleInput = document.getElementById("create-handover-title");
  const customerInput = document.getElementById("create-handover-customer");

  const item = e.detail?.item;
  if (!item) {
    _selectedConversationId = null;
    return;
  }

  _selectedConversationId = item.getAttribute("data-id") || null;

  const convTitle = item.text || "";
  const customer = item.dataset?.customer || "";

  if (titleInput && !titleInput.value.trim()) {
    titleInput.value = convTitle;
  }
  if (customerInput && !customerInput.value.trim()) {
    customerInput.value = customer;
  }
}

async function handleCreateConfirm() {
  if (_actionBusy) return;
  _actionBusy = true;
  const confirmBtn = document.getElementById("create-handover-confirm");
  _setBtnLoading(confirmBtn, true);
  try {
    const titleInput = document.getElementById("create-handover-title");
    const customerInput = document.getElementById("create-handover-customer");
    const convCombo = document.getElementById("create-handover-conversation");

    const title = (titleInput?.value || "").trim();
    if (!title) {
      if (titleInput) {
        titleInput.valueState = "Negative";
        titleInput.focus();
      }
      return;
    }

    const customer = (customerInput?.value || "").trim();
    const linkedConversationId = _selectedConversationId || null;

    const result = await createHandover({
      linkedConversationId: linkedConversationId || undefined,
      payload: { usecase_title: title, customer },
    });
    closeCreateDialog();
    await openHandover(result.id);
  } catch (err) {
    console.error("Failed to create handover:", err);
    showToast("Failed to create handover");
  } finally {
    _setBtnLoading(confirmBtn, false);
    _actionBusy = false;
  }
}

async function handleSave(silent = false) {
  if (!_currentHandoverId) return;
  const payload = collectForm();
  payload.completion_pct = _currentCompletionPct;
  try {
    await updateHandover(_currentHandoverId, payload);
    if (!silent) {
      showToast();
    }
  } catch (err) {
    console.error("Failed to save handover:", err);
    if (silent) showToast("Changes not saved — check your connection");
  }
}

function showToast(message = "Changes saved") {
  const toast = document.getElementById("save-toast");
  if (!toast) return;
  toast.textContent = message;
  if (typeof toast.show === "function") {
    toast.show();
  } else {
    toast.open = true;
  }
}



function downloadJson(data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const safeTitle = (data.payload?.usecase_title || _currentHandoverId || "export")
    .replace(/[\/\\:*?"<>|]/g, "_").slice(0, 100);
  a.download = `handover-${safeTitle}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

async function handleComplete() {
  if (!_currentHandoverId || _actionBusy) return;
  _actionBusy = true;
  const detailBusy = document.getElementById("handover-detail-busy");
  try {
    // Validate required fields with visual feedback
    const { valid, missing } = validateRequiredFields();
    const payload = collectForm();
    const pct = _currentCompletionPct;

    if (pct < 60) {
      // Scroll to and expand the section containing the first missing field
      if (missing.length > 0) {
        const firstMissing = document.getElementById(`f-${missing[0]}`);
        const panel = firstMissing?.closest("ui5-panel");
        if (panel) {
          panel.collapsed = false;
          setTimeout(() => firstMissing.scrollIntoView({ behavior: "smooth", block: "center" }), 150);
        }
      }
      showToast(`Cannot validate: ${pct}% complete (need 60%)`);
      return;
    }

    if (!valid) {
      // Progress threshold met but required fields still empty
      if (missing.length > 0) {
        const firstMissing = document.getElementById(`f-${missing[0]}`);
        const panel = firstMissing?.closest("ui5-panel");
        if (panel) {
          panel.collapsed = false;
          setTimeout(() => firstMissing.scrollIntoView({ behavior: "smooth", block: "center" }), 150);
        }
      }
      showToast(`Required fields missing: ${missing.join(", ")}`);
      return;
    }

    if (detailBusy) detailBusy.active = true;
    await handleSave(true);
    await completeHandover(_currentHandoverId);
    const titleEl = document.getElementById("handover-page-title");
    if (titleEl && !titleEl.textContent.endsWith(" ✓")) titleEl.textContent += " ✓";
    showToast("Handover marked as validated");
  } catch (err) {
    console.error("Failed to complete:", err);
    showToast("Validation failed — please try again");
  } finally {
    if (detailBusy) detailBusy.active = false;
    _actionBusy = false;
  }
}

async function handleDuplicate() {
  if (!_currentHandoverId || _actionBusy) return;
  _actionBusy = true;
  const detailBusy = document.getElementById("handover-detail-busy");
  if (detailBusy) detailBusy.active = true;
  try {
    const payload = collectForm();
    payload.usecase_title = (payload.usecase_title || "Untitled") + " (copy)";
    const result = await createHandover({ payload });
    showToast("Duplicate created");
    await openHandover(result.id);
  } catch (err) {
    console.error("Failed to duplicate:", err);
    showToast("Duplicate failed");
  } finally {
    if (detailBusy) detailBusy.active = false;
    _actionBusy = false;
  }
}

// ── Delete Confirmation ───────────────────────────────────────────────

function showDeleteConfirmation(handoverId) {
  if (!handoverId) return;
  _pendingDeleteId = handoverId;
  const dialog = document.getElementById("handover-delete-dialog");
  if (dialog) dialog.open = true;
}

function hideDeleteDialog() {
  _pendingDeleteId = null;
  const dialog = document.getElementById("handover-delete-dialog");
  if (dialog) dialog.open = false;
}

async function handleConfirmDelete() {
  const id = _pendingDeleteId;
  hideDeleteDialog();
  if (!id || _actionBusy) return;
  _actionBusy = true;
  const listBusy = document.getElementById("handover-list-busy");
  if (listBusy) listBusy.active = true;

  try {
    await deleteHandover(id);
    if (_currentHandoverId === id) {
      await showListView();
    } else {
      await loadHandoverList();
    }
  } catch (err) {
    console.error("Failed to delete handover:", err);
    showToast("Delete failed");
  } finally {
    if (listBusy) listBusy.active = false;
    _actionBusy = false;
  }
}

// ── Sidebar Data ───────────────────────────────────────────────────────

async function loadSidebarData(handoverId) {
  await Promise.all([
    loadFilesByZone(handoverId),
    loadContextSources(handoverId),
  ]);
}

async function loadContextSources(handoverId) {
  const listEl = document.getElementById("context-sources-list");
  if (!listEl) return;

  try {
    const sources = await listContext(handoverId);
    const prefillBtn = document.getElementById("prefill-btn");
    const emptyEl = document.getElementById("context-sources-empty");

    if (!sources.length) {
      listEl.innerHTML = "";
      if (emptyEl) emptyEl.hidden = false;
      if (prefillBtn) prefillBtn.style.display = "none";
      return;
    }

    if (emptyEl) emptyEl.hidden = true;

    // Build lookup for context-zone files so document sources become clickable
    let contextFileMap = {};
    try {
      const files = await listFiles(handoverId);
      files.filter((f) => f.zone === "context").forEach((f) => {
        contextFileMap[f.filename] = f;
      });
    } catch { /* files list unavailable — documents won't be clickable */ }

    // Show prefill button when sources exist
    if (prefillBtn) prefillBtn.style.display = "";

    listEl.innerHTML = sources.map((s) => {
      let iconName = "document-text";
      let label;
      if (s.source_type === "manual") {
        iconName = "edit";
        label = `Note: ${_esc(s.source_ref || "-")}`;
      } else if (s.source_type === "ideation") {
        iconName = "business-suite/answered";
        const title = s.extracted_fields?.usecase_title;
        const linkText = title
          ? _esc(title.charAt(0).toUpperCase() + title.slice(1))
          : "Ideation conversation";
        // ui5-link with href for proper link semantics; click handler does SPA navigation
        label = `<ui5-link href="/?c=${_escAttr(s.source_ref)}" data-nav-conversation="${_escAttr(s.source_ref)}">${linkText}</ui5-link>`;
      } else if (s.source_type === "document") {
        const filename = s.source_ref || "document";
        const ext = (filename.split(".").pop() || "").toLowerCase();
        iconName = _getFileIcon(ext);
        const ctxFile = contextFileMap[filename];
        if (ctxFile) {
          label = `<span class="context-source-clickable"
                         data-preview-file="${_escAttr(ctxFile.file_id)}"
                         data-preview-filename="${_escAttr(filename)}"
                         data-preview-ct="${_escAttr(ctxFile.content_type || "")}"
                         >${_esc(filename)}</span>`;
        } else {
          label = _esc(filename);
        }
      } else {
        label = `${_esc(s.source_type)}: ${_esc(s.source_ref || "-")}`;
      }
      return `
      <div class="context-source-item" data-context-id="${_escAttr(s.id)}">
        <ui5-icon name="${iconName}" class="context-source-icon"></ui5-icon>
        <span class="context-source-label">${label}</span>
        <ui5-button icon="delete" design="Transparent" data-delete-ctx="${_escAttr(s.id)}" tooltip="Remove"></ui5-button>
      </div>
    `;
    }).join("");

    listEl.querySelectorAll("[data-delete-ctx]").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const ctxId = btn.dataset.deleteCtx;
        const row = btn.closest(".context-source-item");
        if (row) row.classList.add("item-busy");
        try {
          await removeContext(handoverId, ctxId);
          await loadContextSources(handoverId);
        } catch (err) {
          console.error("Failed to remove context:", err);
          showToast("Failed to remove context source");
          if (row) row.classList.remove("item-busy");
        }
      });
    });

    // Navigate to ideation conversation on link click (same tab, SPA)
    listEl.querySelectorAll("[data-nav-conversation]").forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const convId = link.dataset.navConversation;
        if (convId) window.location.href = `/?c=${encodeURIComponent(convId)}`;
      });
    });

    // File preview/download on context source document click
    listEl.querySelectorAll("[data-preview-file]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        _handleFileClick(handoverId, el.dataset.previewFile, el.dataset.previewFilename, el.dataset.previewCt);
      });
    });

    // ── Auto-suggestions: render cached fields or trigger extraction ──
    const needsExtraction = sources.some(
      (s) => s.source_type === "ideation" && (!s.extracted_fields || Object.keys(s.extracted_fields).length === 0)
    );

    if (needsExtraction) {
      // First open — auto-trigger prefill (same UX as clicking the button)
      handlePrefill();
    } else {
      // Refresh / return — render cached suggestions for still-empty fields
      const allFields = {};
      for (const s of sources) {
        if (s.extracted_fields && typeof s.extracted_fields === "object") {
          Object.assign(allFields, s.extracted_fields);
        }
      }
      if (Object.keys(allFields).length > 0) {
        renderSuggestions(allFields);
      }
    }
  } catch (err) {
    console.error("Failed to load context:", err);
    showToast("Failed to load context sources");
  }
}

async function handleAddContext() {
  if (!_currentHandoverId) return;

  const dialog = document.getElementById("add-context-dialog");
  if (!dialog) return;

  // ── Reset state ──
  let _ctxSelectedConvId = null;
  let _ctxSelectedFile = null;

  // ── Reset file state + description early ──
  const fileChip = document.getElementById("context-file-chip");
  const filePrompt = document.getElementById("context-file-prompt");
  if (fileChip) { fileChip.innerHTML = ""; fileChip.style.display = "none"; }
  if (filePrompt) filePrompt.style.display = "";

  const textarea = document.getElementById("context-manual-text");
  if (textarea) textarea.value = "";
  const combobox = document.getElementById("context-conversation-select");
  if (combobox) {
    combobox.value = "";
    combobox.innerHTML = "";
    combobox.loading = true;
  }
  dialog.open = true;

  // ── Populate conversation combobox (async, dialog already open) ──
  if (combobox) {
    try {
      const [data, existingSources] = await Promise.all([
        request("/api/ideation-agent/conversations", "GET"),
        listContext(_currentHandoverId),
      ]);
      const linkedIds = new Set(
        existingSources
          .filter((s) => s.source_type === "ideation")
          .map((s) => s.source_ref)
      );
      const convos = (data.conversations || [])
        .filter((c) => !linkedIds.has(c.id))
        .sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));
      for (const conv of convos) {
        const item = document.createElement("ui5-cb-item");
        item.text = conv.title || conv.company_name || "Untitled";
        item.setAttribute("data-id", conv.id);
        item.additionalText = _formatDate(conv.updated_at);
        combobox.appendChild(item);
      }
    } catch (err) {
      console.error("Failed to load conversations:", err);
    } finally {
      combobox.loading = false;
    }
    combobox.addEventListener("selection-change", (e) => {
      const item = e.detail?.item;
      _ctxSelectedConvId = item?.getAttribute("data-id") || null;
    });
  }

  // ── Wire file upload zone ──
  const fileZone = document.getElementById("context-file-zone");
  const fileInput = document.getElementById("context-file-input");

  const showFileChip = (file) => {
    _ctxSelectedFile = file;
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    const iconName = _getFileIcon(ext);
    fileChip.innerHTML = `
      <ui5-icon name="${iconName}"></ui5-icon>
      <span class="file-chip-name" title="${_escAttr(file.name)}">${_esc(file.name)}</span>
      <ui5-button id="ctx-file-remove" icon="decline" design="Transparent" tooltip="Remove"></ui5-button>
    `;
    fileChip.style.display = "";
    filePrompt.style.display = "none";
    fileChip.querySelector("#ctx-file-remove").addEventListener("click", (e) => {
      e.stopPropagation();
      clearFileSelection();
    });
  };

  const clearFileSelection = () => {
    _ctxSelectedFile = null;
    fileChip.innerHTML = "";
    fileChip.style.display = "none";
    filePrompt.style.display = "";
  };

  // File input change
  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      const files = e.detail?.files || e.target?.files;
      if (files && files.length > 0) showFileChip(files[0]);
    });
  }

  // Drag-and-drop on the file zone
  if (fileZone) {
    fileZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      fileZone.classList.add("drag-over");
    });
    fileZone.addEventListener("dragleave", () => {
      fileZone.classList.remove("drag-over");
    });
    fileZone.addEventListener("drop", (e) => {
      e.preventDefault();
      fileZone.classList.remove("drag-over");
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) showFileChip(files[0]);
    });
  }

  // ── Wire buttons ──
  const cancelBtn = document.getElementById("context-dialog-cancel");
  const confirmBtn = document.getElementById("context-dialog-confirm");

  const cleanup = () => {
    dialog.open = false;
    cancelBtn.removeEventListener("click", handleCancel);
    confirmBtn.removeEventListener("click", handleConfirm);
    dialog.removeEventListener("close", handleDialogClose);
  };

  const handleDialogClose = () => cleanup();
  const handleCancel = () => cleanup();

  const handleConfirm = async () => {
    const convId = _ctxSelectedConvId;
    const file = _ctxSelectedFile;
    const text = document.getElementById("context-manual-text")?.value?.trim() || "";

    // Validate: at least one must be filled
    if (!convId && !file && !text) {
      showToast("Please provide at least one context source");
      return;
    }

    confirmBtn.disabled = true;
    confirmBtn.textContent = "Extracting...";
    _setBtnLoading(confirmBtn, true);

    try {
      const results = [];

      // Add ideation context
      if (convId) {
        const r = await addContext(_currentHandoverId, "ideation", convId);
        results.push(r);
      }

      // Add file context
      if (file) {
        const r = await addContextFile(_currentHandoverId, file);
        results.push(r);
      }

      // Add description context
      if (text) {
        const r = await addContext(_currentHandoverId, "manual", text);
        results.push(r);
      }

      await loadContextSources(_currentHandoverId);

      // Show suggestions from any extraction that returned fields
      for (const result of results) {
        if (result?.extracted_fields && Object.keys(result.extracted_fields).length > 0) {
          renderSuggestions(result.extracted_fields);
          break; // Show first set of suggestions
        }
      }

      cleanup();
      const count = results.length;
      const hasFields = results.some((r) => r?.extracted_fields);
      showToast(
        `${count} context source${count > 1 ? "s" : ""} added` +
        (hasFields ? " — suggestions ready" : "")
      );
    } catch (err) {
      console.error("Failed to add context:", err);
      showToast("Failed to add context");
    } finally {
      _setBtnLoading(confirmBtn, false);
      confirmBtn.disabled = false;
      confirmBtn.textContent = "Add & Extract";
    }
  };

  cancelBtn.addEventListener("click", handleCancel);
  confirmBtn.addEventListener("click", handleConfirm);
  dialog.addEventListener("close", handleDialogClose);
}

// ── Inline Suggestions (Phase 2) ────────────────────────────────────────

let _pendingSuggestions = {};

function renderSuggestions(fields) {
  // Clear any existing suggestions first
  clearSuggestions();
  _pendingSuggestions = { ...fields };
  let count = 0;

  // Scalar fields: show suggestion chip below input
  for (const [fieldId, value] of Object.entries(fields)) {
    if (Array.isArray(value)) continue; // tables handled separately
    if (value && typeof value === "object") continue; // special objects (business_impact) handled below
    if (!value || (typeof value === "string" && !value.trim())) continue;

    const el = document.getElementById(`f-${fieldId}`);
    if (!el) { delete _pendingSuggestions[fieldId]; continue; }
    // Skip if field already has the same content as the suggestion
    let currentVal = "";
    if (el.tagName === "UI5-SELECT") {
      const selectedOpt = el.querySelector("ui5-option[selected]");
      currentVal = (selectedOpt?.getAttribute("value") || selectedOpt?.textContent || "").trim();
    } else {
      currentVal = (el.value || el.getAttribute("value") || "").trim();
    }
    if (currentVal && currentVal.toLowerCase() === String(value).trim().toLowerCase()) { delete _pendingSuggestions[fieldId]; continue; }

    const chip = createSuggestionChip(fieldId, value);
    el.closest(".form-field")?.appendChild(chip);
    count++;
  }

  // Table fields: show suggestion chip above table
  for (const tableKey of Object.keys(TABLE_SCHEMAS)) {
    if (!fields[tableKey]?.length) continue;
    const container = document.getElementById(`table-${tableKey}`);
    if (!container) { delete _pendingSuggestions[tableKey]; continue; }
    // Check if table already has data
    if (hasUserTableData(tableKey)) { delete _pendingSuggestions[tableKey]; continue; }

    const chip = createTableSuggestionChip(tableKey, fields[tableKey]);
    container.parentElement?.insertBefore(chip, container);
    count++;
  }

  // Business impact — special object with {dimension: {current, target}}
  if (fields.business_impact && typeof fields.business_impact === "object" && !Array.isArray(fields.business_impact)) {
    const impactObj = fields.business_impact;
    const filledDimensions = Object.entries(impactObj).filter(
      ([, v]) => v && typeof v === "object" && ((v.current || "").trim() || (v.target || "").trim()),
    );
    if (filledDimensions.length > 0) {
      // Only suggest if impact table is currently empty
      const anyFilled = [...document.querySelectorAll(".impact-table .td-input[data-impact]")].some(
        (inp) => (inp.value || "").trim(),
      );
      if (!anyFilled) {
        const chip = _createImpactSuggestionChip(filledDimensions);
        const impactTable = document.querySelector(".impact-table");
        impactTable?.parentElement?.insertBefore(chip, impactTable);
        count++;
      } else {
        delete _pendingSuggestions.business_impact;
      }
    } else {
      delete _pendingSuggestions.business_impact;
    }
  }

  // AI capabilities — array of capability IDs
  if (Array.isArray(fields.ai_capabilities) && fields.ai_capabilities.length > 0) {
    const grid = document.getElementById("ai-capabilities-grid");
    const alreadySelected = grid?.querySelectorAll(".ai-cap-item.selected").length > 0;
    if (grid && !alreadySelected) {
      const chip = _createAiCapsSuggestionChip(fields.ai_capabilities);
      grid.parentElement?.insertBefore(chip, grid);
      count++;
    } else {
      delete _pendingSuggestions.ai_capabilities;
    }
  }

  // Count individual items (not just chips) for the banner
  const itemCount = _countPendingSuggestionItems();

  // Show "Accept all" banner if multiple suggestions
  if (count > 1) showSuggestionBanner(itemCount);

  // Update section suggestion badges
  _updateSectionSuggestionBadges();

  // Show prefill button for re-runs
  const prefillBtn = document.getElementById("prefill-btn");
  if (prefillBtn) prefillBtn.style.display = "";

  if (count > 0) showToast(`${itemCount} suggestion${itemCount > 1 ? "s" : ""} ready to review`);
}

// ── Section Suggestion Badges ────────────────────────────────────────────

// Maps each field/table key to the section panel that contains it.
const _FIELD_TO_SECTION = (() => {
  const map = {};
  for (const [sectionId, config] of Object.entries(SECTION_FIELDS)) {
    for (const f of config.fields || []) map[f] = sectionId;
    for (const t of config.tables || []) map[t] = sectionId;
    if (config.special === "business_impact") map.business_impact = sectionId;
    if (config.special === "ai_capabilities") map.ai_capabilities = sectionId;
  }
  return map;
})();

/**
 * Count the number of individual suggested items in a pending suggestion value.
 * Arrays count their length, business_impact counts filled dimensions, scalars count 1.
 */
function _countSuggestionItems(key, value) {
  if (Array.isArray(value)) return value.length || 1;
  if (key === "business_impact" && typeof value === "object" && value) {
    return Object.values(value).filter(
      (v) => v && typeof v === "object" && ((v.current || "").trim() || (v.target || "").trim()),
    ).length || 1;
  }
  return 1;
}

/** Total individual items across all pending suggestions. */
function _countPendingSuggestionItems() {
  let total = 0;
  for (const [key, value] of Object.entries(_pendingSuggestions)) {
    total += _countSuggestionItems(key, value);
  }
  return total;
}

/**
 * Count pending suggestion items per section and show/hide a small AI badge
 * next to each section's existing completion badge.
 */
function _updateSectionSuggestionBadges() {
  // Count individual suggested items per section
  const counts = {};
  for (const [key, value] of Object.entries(_pendingSuggestions)) {
    const section = _FIELD_TO_SECTION[key];
    if (!section) continue;
    counts[section] = (counts[section] || 0) + _countSuggestionItems(key, value);
  }

  // Update or remove AI badges on each section header
  for (const sectionId of Object.keys(SECTION_FIELDS)) {
    const existing = document.getElementById(`ai-badge-${sectionId}`);
    const count = counts[sectionId] || 0;

    if (count > 0) {
      if (existing) {
        existing.querySelector(".section-ai-badge-count").textContent = count;
      } else {
        // Create a plain badge matching the suggestion chip design
        const sectionBadge = document.getElementById(`badge-${sectionId}`);
        if (!sectionBadge) continue;
        const aiBadge = document.createElement("span");
        aiBadge.id = `ai-badge-${sectionId}`;
        aiBadge.className = "section-ai-badge";
        aiBadge.innerHTML = `<ui5-icon name="ai" class="section-ai-badge-icon"></ui5-icon><span class="section-ai-badge-count">${count}</span>`;
        sectionBadge.insertAdjacentElement("afterend", aiBadge);
      }
    } else if (existing) {
      existing.remove();
    }
  }
}

function createSuggestionChip(fieldId, value) {
  const chip = document.createElement("div");
  chip.className = "field-suggestion";
  chip.dataset.fieldId = fieldId;
  const displayVal = _truncate(String(value), 80);
  chip.innerHTML = `
    <ui5-icon name="ai" class="field-suggestion-icon"></ui5-icon>
    <span class="field-suggestion-value" title="${_escAttr(String(value))}">${_esc(displayVal)}</span>
    <ui5-button class="field-suggestion-accept" icon="accept" design="Transparent" tooltip="Accept"></ui5-button>
    <ui5-button class="field-suggestion-dismiss" icon="decline" design="Transparent" tooltip="Dismiss"></ui5-button>
  `;
  chip.querySelector(".field-suggestion-accept").addEventListener("click", (e) => {
    e.stopPropagation();
    acceptSuggestion(fieldId, value);
    chip.remove();
    _updateSectionSuggestionBadges();
    updateSuggestionBanner();
  });
  chip.querySelector(".field-suggestion-dismiss").addEventListener("click", (e) => {
    e.stopPropagation();
    delete _pendingSuggestions[fieldId];
    chip.remove();
    _updateSectionSuggestionBadges();
    updateSuggestionBanner();
  });
  return chip;
}

function createTableSuggestionChip(tableKey, rows) {
  const chip = document.createElement("div");
  chip.className = "table-suggestion field-suggestion";
  chip.dataset.fieldId = tableKey;
  chip.innerHTML = `
    <ui5-icon name="ai" class="field-suggestion-icon"></ui5-icon>
    <span class="field-suggestion-value">${rows.length} row${rows.length > 1 ? "s" : ""} suggested for ${tableKey.replace(/_/g, " ")}</span>
    <ui5-button class="field-suggestion-accept" icon="accept" design="Transparent" tooltip="Accept"></ui5-button>
    <ui5-button class="field-suggestion-dismiss" icon="decline" design="Transparent" tooltip="Dismiss"></ui5-button>
  `;
  chip.querySelector(".field-suggestion-accept").addEventListener("click", (e) => {
    e.stopPropagation();
    acceptTableSuggestion(tableKey, rows);
    chip.remove();
    _updateSectionSuggestionBadges();
    updateSuggestionBanner();
  });
  chip.querySelector(".field-suggestion-dismiss").addEventListener("click", (e) => {
    e.stopPropagation();
    delete _pendingSuggestions[tableKey];
    chip.remove();
    _updateSectionSuggestionBadges();
    updateSuggestionBanner();
  });
  return chip;
}

function _createImpactSuggestionChip(filledDimensions) {
  const chip = document.createElement("div");
  chip.className = "table-suggestion field-suggestion";
  chip.dataset.fieldId = "business_impact";
  chip.innerHTML = `
    <ui5-icon name="ai" class="field-suggestion-icon"></ui5-icon>
    <span class="field-suggestion-value">${filledDimensions.length} impact metric${filledDimensions.length > 1 ? "s" : ""} suggested</span>
    <ui5-button class="field-suggestion-accept" icon="accept" design="Transparent" tooltip="Accept"></ui5-button>
    <ui5-button class="field-suggestion-dismiss" icon="decline" design="Transparent" tooltip="Dismiss"></ui5-button>
  `;
  chip.querySelector(".field-suggestion-accept").addEventListener("click", (e) => {
    e.stopPropagation();
    // Write values into the impact table inputs
    for (const [dim, pair] of filledDimensions) {
      for (const col of ["current", "target"]) {
        if (pair[col]?.trim()) {
          const input = document.querySelector(`.impact-table .td-input[data-impact="${CSS.escape(dim)}"][data-col="${col}"]`);
          if (input) input.value = pair[col].trim();
        }
      }
    }
    delete _pendingSuggestions.business_impact;
    chip.remove();
    _updateSectionSuggestionBadges();
    updateSuggestionBanner();
    updateProgress();
    scheduleSave();
  });
  chip.querySelector(".field-suggestion-dismiss").addEventListener("click", (e) => {
    e.stopPropagation();
    delete _pendingSuggestions.business_impact;
    chip.remove();
    _updateSectionSuggestionBadges();
    updateSuggestionBanner();
  });
  return chip;
}

function _createAiCapsSuggestionChip(capIds) {
  const chip = document.createElement("div");
  chip.className = "table-suggestion field-suggestion";
  chip.dataset.fieldId = "ai_capabilities";
  const labels = capIds
    .map((id) => AI_CAPABILITIES.find((c) => c.id === id)?.label || id)
    .join(", ");
  chip.innerHTML = `
    <ui5-icon name="ai" class="field-suggestion-icon"></ui5-icon>
    <span class="field-suggestion-value" title="${_escAttr(labels)}">${capIds.length} AI capabilit${capIds.length > 1 ? "ies" : "y"} suggested</span>
    <ui5-button class="field-suggestion-accept" icon="accept" design="Transparent" tooltip="Accept"></ui5-button>
    <ui5-button class="field-suggestion-dismiss" icon="decline" design="Transparent" tooltip="Dismiss"></ui5-button>
  `;
  chip.querySelector(".field-suggestion-accept").addEventListener("click", (e) => {
    e.stopPropagation();
    // Toggle selected state on matching capability grid items
    for (const id of capIds) {
      const item = document.querySelector(`.ai-cap-item[data-capability="${CSS.escape(id)}"]`);
      if (item && !item.classList.contains("selected")) {
        item.classList.add("selected");
      }
    }
    delete _pendingSuggestions.ai_capabilities;
    chip.remove();
    _updateSectionSuggestionBadges();
    updateSuggestionBanner();
    updateProgress();
    scheduleSave();
  });
  chip.querySelector(".field-suggestion-dismiss").addEventListener("click", (e) => {
    e.stopPropagation();
    delete _pendingSuggestions.ai_capabilities;
    chip.remove();
    _updateSectionSuggestionBadges();
    updateSuggestionBanner();
  });
  return chip;
}

function acceptSuggestion(fieldId, value) {
  const el = document.getElementById(`f-${fieldId}`);
  if (!el) { delete _pendingSuggestions[fieldId]; return; }

  if (el.tagName === "UI5-SELECT") {
    // For dropdowns: find the matching option (case-insensitive)
    const normalised = String(value).trim().toLowerCase();
    const options = el.querySelectorAll("ui5-option");
    let matched = false;
    for (const opt of options) {
      const optVal = (opt.getAttribute("value") || "").toLowerCase();
      const optText = (opt.textContent || "").trim().toLowerCase();
      if (optVal === normalised || optText === normalised) {
        opt.selected = true;
        matched = true;
        break;
      }
    }
    if (!matched) { delete _pendingSuggestions[fieldId]; return; }
  } else {
    el.value = String(value);
    // Trigger UI5 input event for proper binding
    el.fireEvent?.("input");
  }

  el.classList.add("suggestion-accepted");
  setTimeout(() => el.classList.remove("suggestion-accepted"), 1500);
  delete _pendingSuggestions[fieldId];
  updateProgress();
  scheduleSave();
}

function acceptTableSuggestion(tableKey, rows) {
  // Re-render the table with the suggested data
  const container = document.getElementById(`table-${tableKey}`);
  if (!container || !TABLE_SCHEMAS[tableKey]) return;

  renderTable(container, tableKey, TABLE_SCHEMAS[tableKey], rows);
  container.classList.add("suggestion-accepted");
  setTimeout(() => container.classList.remove("suggestion-accepted"), 1500);

  delete _pendingSuggestions[tableKey];
  updateProgress();
  scheduleSave();
}

function acceptAllSuggestions() {
  // Accept scalar suggestions
  document.querySelectorAll(".field-suggestion:not(.table-suggestion)").forEach((chip) => {
    const fieldId = chip.dataset.fieldId;
    const value = _pendingSuggestions[fieldId];
    if (value && !Array.isArray(value)) {
      acceptSuggestion(fieldId, value);
    }
    chip.remove();
  });
  // Accept table suggestions
  document.querySelectorAll(".table-suggestion").forEach((chip) => {
    chip.querySelector(".field-suggestion-accept")?.click();
  });
  hideSuggestionBanner();
  showToast("All suggestions applied");
}

function dismissAllSuggestions() {
  _pendingSuggestions = {};
  document.querySelectorAll(".field-suggestion, .table-suggestion").forEach((el) => el.remove());
  _updateSectionSuggestionBadges();
  hideSuggestionBanner();
}

function clearSuggestions() {
  document.querySelectorAll(".field-suggestion, .table-suggestion, .suggestion-banner, .section-ai-badge").forEach((el) => el.remove());
  _pendingSuggestions = {};
}

function showSuggestionBanner(count) {
  hideSuggestionBanner(); // Remove existing if any
  const formPanel = document.querySelector(".handover-form-panel");
  if (!formPanel) return;

  const banner = document.createElement("div");
  banner.className = "suggestion-banner";
  banner.id = "suggestion-banner";
  banner.innerHTML = `
    <ui5-icon name="ai"></ui5-icon>
    <span class="suggestion-banner-text">${count} suggestions from context</span>
    <ui5-button design="Emphasized" id="accept-all-btn">Accept All</ui5-button>
    <ui5-button design="Transparent" id="dismiss-all-btn">Dismiss All</ui5-button>
  `;
  // Place below the source-materials card, above the form sections
  const sourcesCard = document.getElementById("source-materials-card");
  if (sourcesCard) {
    sourcesCard.insertAdjacentElement("afterend", banner);
  } else {
    formPanel.insertBefore(banner, formPanel.firstChild);
  }

  banner.querySelector("#accept-all-btn").addEventListener("click", acceptAllSuggestions);
  banner.querySelector("#dismiss-all-btn").addEventListener("click", dismissAllSuggestions);
}

function hideSuggestionBanner() {
  document.getElementById("suggestion-banner")?.remove();
}

function updateSuggestionBanner() {
  const remainingChips = document.querySelectorAll(".field-suggestion, .table-suggestion").length;
  if (remainingChips <= 1) {
    hideSuggestionBanner();
  } else {
    const itemCount = _countPendingSuggestionItems();
    const textEl = document.querySelector(".suggestion-banner-text");
    if (textEl) textEl.textContent = `${itemCount} suggestions from context`;
  }
}

// ── Pre-fill Button ─────────────────────────────────────────────────────

function initPrefillButton() {
  const btn = document.getElementById("prefill-btn");
  if (!btn) return;
  btn.addEventListener("click", handlePrefill);
}

async function handlePrefill() {
  if (!_currentHandoverId) return;
  const btn = document.getElementById("prefill-btn");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Extracting...";
    _setBtnLoading(btn, true);
  }

  try {
    const result = await prefillFromContext(_currentHandoverId);
    if (result?.fields && Object.keys(result.fields).length > 0) {
      renderSuggestions(result.fields);
    } else {
      showToast("No additional fields could be extracted");
    }
  } catch (err) {
    console.error("Failed to prefill:", err);
    showToast("Pre-fill failed");
  } finally {
    if (btn) {
      _setBtnLoading(btn, false);
      btn.disabled = false;
      btn.textContent = "Pre-fill from context";
    }
  }
}

// ── Agent Chat (Phase 3) ────────────────────────────────────────────────

function initAgentChat() {
  const input = document.getElementById("agent-chat-input");
  const sendBtn = document.getElementById("agent-send-btn");

  if (!input || !sendBtn) return;

  // Send on button click
  sendBtn.addEventListener("click", () => sendAgentMsg());

  // Send on Enter key
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendAgentMsg();
    }
  });

  // Wire suggestion chips
  const suggestionsContainer = document.getElementById("agent-suggestions");
  if (suggestionsContainer) {
    suggestionsContainer.addEventListener("click", (e) => {
      const chip = e.target.closest(".agent-suggestion");
      if (chip?.dataset.msg) {
        input.value = chip.dataset.msg;
        sendAgentMsg();
      }
    });
  }
}

async function sendAgentMsg(targetSection) {
  const input = document.getElementById("agent-chat-input");
  if (!input || !_currentHandoverId) return;

  const text = input.value?.trim();
  if (!text) return;

  // Render user message
  renderAgentMessage("user", text);
  input.value = "";
  setAgentInputState(false);
  showAgentThinking("Thinking...");

  // Hide welcome on first message
  const welcome = document.getElementById("agent-welcome");
  if (welcome) welcome.style.display = "none";

  try {
    const result = await sendAgentMessage(_currentHandoverId, text, targetSection);

    hideAgentThinking();

    if (result.kind === "error") {
      renderAgentMessage("assistant", result.message || "Sorry, something went wrong.");
    } else {
      // Render agent response
      const chips = result.suggested_replies || [];
      renderAgentMessage("assistant", result.message, chips);

      // If agent proposed field values, show inline suggestions
      if (result.kind === "fill" && result.fields && Object.keys(result.fields).length > 0) {
        renderSuggestions(result.fields);
      }

      // If agent returned a summary with gaps, render as summary card
      if (result.kind === "summary" && result.gaps) {
        renderAgentSummaryCard(result.gaps);
      }
    }
  } catch (err) {
    hideAgentThinking();
    renderAgentMessage("assistant", "Connection error. Please try again.");
    console.error("Agent chat error:", err);
  } finally {
    setAgentInputState(true);
  }
}

function renderAgentMessage(role, text, chips) {
  const container = document.getElementById("agent-chat-messages");
  if (!container) return;

  const msg = document.createElement("div");
  msg.className = `agent-msg ${role}`;

  // Simple markdown: bold, lists
  let html = _esc(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n- /g, "\n• ")
    .replace(/\n/g, "<br>");

  msg.innerHTML = `<div class="agent-msg-bubble">${html}</div>`;

  // Add suggestion chips if provided
  if (chips?.length) {
    const chipsEl = document.createElement("div");
    chipsEl.className = "agent-msg-chips";
    for (const chipText of chips) {
      const chip = document.createElement("span");
      chip.className = "agent-msg-chip";
      chip.textContent = chipText;
      chip.onclick = () => {
        const input = document.getElementById("agent-chat-input");
        if (input) {
          input.value = chipText;
          sendAgentMsg();
        }
      };
      chipsEl.appendChild(chip);
    }
    msg.appendChild(chipsEl);
  }

  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
}

function renderAgentSummaryCard(gaps) {
  const container = document.getElementById("agent-chat-messages");
  if (!container) return;

  const card = document.createElement("div");
  card.className = "agent-summary-card";
  card.innerHTML = `
    <h4>Gaps to Fill</h4>
    <ul class="agent-summary-gaps">
      ${gaps.map((g) => `<li data-section="${_escAttr(g.section || "")}">${_esc(g.label || g)}</li>`).join("")}
    </ul>
  `;

  // Wire gap clicks to scroll to section
  card.querySelectorAll("li[data-section]").forEach((li) => {
    li.addEventListener("click", () => {
      const sectionId = li.dataset.section;
      if (sectionId) scrollToSection(sectionId);
    });
  });

  container.appendChild(card);
  container.scrollTop = container.scrollHeight;
}

function showAgentThinking(label) {
  const el = document.getElementById("agent-thinking");
  if (!el) return;
  const labelEl = el.querySelector(".agent-thinking-label");
  if (labelEl) labelEl.textContent = label || "Thinking...";
  el.style.display = "";
}

function hideAgentThinking() {
  const el = document.getElementById("agent-thinking");
  if (el) el.style.display = "none";
}

function setAgentInputState(enabled) {
  const input = document.getElementById("agent-chat-input");
  const btn = document.getElementById("agent-send-btn");
  if (input) input.disabled = !enabled;
  if (btn) btn.disabled = !enabled;
}

function scrollToSection(sectionId) {
  // Find the panel with matching section ID and expand it
  const panels = document.querySelectorAll(".handover-form-panel ui5-panel");
  for (const panel of panels) {
    if (panel.id === `section-${sectionId}`) {
      panel.collapsed = false;
      panel.scrollIntoView({ behavior: "smooth", block: "start" });
      break;
    }
  }
}

async function loadAgentHistory() {
  if (!_currentHandoverId) return;
  try {
    const result = await getAgentHistory(_currentHandoverId);
    if (result?.messages?.length) {
      // Hide welcome
      const welcome = document.getElementById("agent-welcome");
      if (welcome) welcome.style.display = "none";

      for (const msg of result.messages) {
        const chips = msg.chips || [];
        renderAgentMessage(msg.role, msg.content, chips);
      }
    }
  } catch (err) {
    // Silently ignore — history is optional
    console.debug("Agent history load skipped:", err.message);
  }
}

// ── Helpers ────────────────────────────────────────────────────────────

function _esc(str) {
  if (str == null) return "";
  const el = document.createElement("span");
  el.textContent = String(str);
  return el.innerHTML;
}

function _escAttr(str) {
  if (str == null) return "";
  return String(str).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function _truncate(str, max) {
  if (str == null) return "";
  return str.length > max ? str.slice(0, max - 1) + "…" : str;
}

function _formatDate(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch {
    return "";
  }
}

/** Map file extension to a UI5 icon name. */
function _getFileIcon(ext) {
  ext = (ext || "").toLowerCase();
  if (ext === "pdf") return "pdf-attachment";
  if (["doc", "docx", "rtf", "odt"].includes(ext)) return "doc-attachment";
  if (["ppt", "pptx", "odp"].includes(ext)) return "ppt-attachment";
  if (["xls", "xlsx", "ods", "csv", "tsv"].includes(ext)) return "excel-attachment";
  if (["png", "jpg", "jpeg", "gif", "svg", "webp"].includes(ext)) return "picture";
  if (["json", "xml", "yaml", "yml", "py", "js", "ts", "html", "css", "sql"].includes(ext)) return "syntax";
  if (["txt", "md", "markdown"].includes(ext)) return "document-text";
  if (["eml", "mht", "mhtml"].includes(ext)) return "email";
  return "document";
}
