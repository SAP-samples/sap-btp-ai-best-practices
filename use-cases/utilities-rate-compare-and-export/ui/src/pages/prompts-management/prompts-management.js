import { request } from "../../services/api.js";

// UI5 imports for Table
import "@ui5/webcomponents/dist/Table.js";
import "@ui5/webcomponents/dist/TableRow.js";
import "@ui5/webcomponents/dist/TableCell.js";
import "@ui5/webcomponents/dist/TableHeaderRow.js";
import "@ui5/webcomponents/dist/TableHeaderCell.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents-icons/dist/edit.js";
import "@ui5/webcomponents-icons/dist/delete.js";
import "@ui5/webcomponents-icons/dist/add.js";
import "@ui5/webcomponents-icons/dist/refresh.js";
import "@ui5/webcomponents-icons/dist/nav-back.js";
import "@ui5/webcomponents-icons/dist/save.js";
import "@ui5/webcomponents-icons/dist/add-folder.js";
import "@ui5/webcomponents-icons/dist/download.js";

let currentMapData = [];
let isEditing = false;

const DEFAULT_META_TEMPLATE = `# System Prompt: Rate Schedule Converter (MD to CSV)

## **Role & Objective**

You are a **Deterministic Utility Data Extraction Engine**. Your input is a Markdown (MD) document representing a Rate Schedule. Your output is a strict, **semicolon-separated CSV file**.

**Core Directives:**

1.  **Mandatory Header:** The output **MUST** start with the specific CSV header row defined below.
2.  **Dynamic Parsing:** Do not rely on hardcoded row counts for tiered tables. Parse row-by-row.
3.  **Column decoupling:** For tables with "Winter" and "Summer" columns, you must explicitly generate **separate rows with different Keys** for each column value.
4.  **Strict Formatting:** Follow the column schema rules without deviation.

---

## **1. Global Parsing Rules**

### **A. Effective Date (ABDATUM)**

1.  Scan the document header for the line **"Effective: [Month] [Day], [Year]"**.
2.  Convert this to \`MM/DD/YYYY\`.
    - _Example:_ "Effective: November 1, 2025" -> \`11/01/2025\`
3.  This date applies to **every single row** in column 3 (\`ABDATUM\`).

### **B. Number Formatting (PREISBTR_1)**

1.  **Decimals:** Ensure decimal separator is a period (\`.\`).
2.  **Precision:** Pad ALL prices/rates to **8 decimal places**.
    - \`1.00\` -> \`1.00000000\`
    - \`0.40\` -> \`0.40000000\`
    - \`1.2345\` -> \`1.23450000\`
3.  **Clean:** Remove \`$\` symbols and commas inside numbers (e.g., \`15,000\` -> \`15000\`).

### **C. Tier/Zone Logic (The Accumulator Algorithm)**

For **Tiered Rates** (tables with "First", "Next", "Over"), do not hardcode values. Use this algorithm:

1.  **Initialize** \`current_cumulative = 0\`.
2.  **Iterate** through every row in the rate table.
3.  **Parse** the "Units" text (remove commas):
    - **Case "First X":**
      - \`VONZONE_1\` = \`0\`
      - \`BISZONE_1\` = \`X\`
      - Update \`current_cumulative\` = \`X\`
    - **Case "Next Y":**
      - \`VONZONE_1\` = \`current_cumulative\`
      - \`BISZONE_1\` = \`current_cumulative\` + \`Y\`
      - Update \`current_cumulative\` = \`BISZONE_1\`
    - **Case "Over Z"**:
      - \`VONZONE_1\` = \`Z\` (or \`current_cumulative\`)
      - \`BISZONE_1\` = \`9999999999\`

---

## **2. Output CSV Schema**

**Line 1 (Header):**
\`PRICE_TEMPLATE;PREIS;ABDATUM;BISDATUM;VONZONE_1;BISZONE_1;PREISBTR_1;TEXT30;PREISTYP;PREISART;SPARTE;MASS;RUNDART;RUNDUNG;TWAERS;MNGBASIS;AKLASSE;TIMBASIS;TIMTYP\`

**Data Rows (19 Columns):**

1.  **PRICE_TEMPLATE**: Leave Empty (Row starts with \`;\`).
2.  **PREIS**: The **Key** from the Master Map.
3.  **ABDATUM**: Extracted Effective Date (\`MM/DD/YYYY\`).
4.  **BISDATUM**: \`12/31/9999\`
5.  **VONZONE_1**: Start Value (Empty for Flat Rates).
6.  **BISZONE_1**: End Value (\`9999999999\` for Flat Rates).
7.  **PREISBTR_1**: The Price (period decimal, 8 digits).
8.  **TEXT30**: The Description from the Master Map.
9.  **PREISTYP**: \`1\`
10. **PREISART**: \`0\` (Flat) or \`1\` (Tiered).
11. **SPARTE**: \`02\`
12. **MASS**: \`thm\` (Always use this value).
13. **RUNDART**: Empty.
14. **RUNDUNG**: Empty.
15. **TWAERS**: \`USD\`
16. **MNGBASIS**: \`1\`
17. **AKLASSE**: \`NCGR\` (Res), \`NCGC\` (Comm), \`NCGT\` (Trans).
18. **TIMBASIS**: \`1\` (if Tiered), else Empty.
19. **TIMTYP**: \`1\` (if Tiered), else Empty.

---`;

export default function init() {
  loadPrompts();
  setupEventListeners();
}

function setupEventListeners() {
  const btnRefresh = document.getElementById("btn-refresh-list");
  if (btnRefresh) btnRefresh.addEventListener("click", loadPrompts);

  const btnCreate = document.getElementById("btn-create-prompt");
  if (btnCreate) btnCreate.addEventListener("click", showCreateView);

  const btnBack = document.getElementById("btn-back");
  if (btnBack) btnBack.addEventListener("click", showListView);

  const btnDownloadEditor = document.getElementById("btn-download-editor");
  if (btnDownloadEditor) btnDownloadEditor.addEventListener("click", downloadPromptDraft);

  const btnSave = document.getElementById("btn-save");
  if (btnSave) btnSave.addEventListener("click", savePrompt);

  const btnAddGroup = document.getElementById("btn-add-group");
  if (btnAddGroup) btnAddGroup.addEventListener("click", addGroup);
}

async function loadPrompts() {
  const table = document.getElementById("prompts-table");
  if (!table) return;

  // Clear existing rows (keep headers)
  const rows = table.querySelectorAll("ui5-table-row");
  rows.forEach((row) => table.removeChild(row));

  try {
    const prompts = await request("/api/prompts/");
    prompts.forEach((p) => {
      const row = document.createElement("ui5-table-row");
      row.innerHTML = `
                <ui5-table-cell><span>${p.label}</span></ui5-table-cell>
                <ui5-table-cell><span>${p.filename}</span></ui5-table-cell>
                <ui5-table-cell>
                    <div style="display: flex; gap: 0.5rem;">
                        <ui5-button icon="download" design="Transparent" class="btn-download" data-filename="${p.filename}" tooltip="Download"></ui5-button>
                        <ui5-button icon="edit" class="btn-edit" data-filename="${p.filename}" tooltip="Edit"></ui5-button>
                        <ui5-button icon="delete" design="Negative" class="btn-delete" data-filename="${p.filename}" tooltip="Delete"></ui5-button>
                    </div>
                </ui5-table-cell>
            `;

      const btnDownload = row.querySelector(".btn-download");
      btnDownload.addEventListener("click", () => downloadPrompt(p.filename));

      const btnEdit = row.querySelector(".btn-edit");
      btnEdit.addEventListener("click", () => loadPromptDetail(p.filename));

      const btnDelete = row.querySelector(".btn-delete");
      btnDelete.addEventListener("click", () => deletePrompt(p.filename));

      table.appendChild(row);
    });
  } catch (e) {
    console.error("Failed to load prompts", e);
  }
}

async function loadPromptDetail(filename) {
  try {
    const data = await request(`/api/prompts/${filename}`);

    const inputFilename = document.getElementById("input-filename");
    inputFilename.value = filename;
    inputFilename.disabled = true; // No rename for now

    document.getElementById("input-meta").value = data.meta;
    document.getElementById("input-execution").value = data.execution;

    // Map data comes from backend as 'class' alias or 'class_'?
    // Pydantic outputs alias by default if set.
    // Let's assume standard JSON from backend has "class".
    currentMapData = data.map_data || [];

    // Sanitize data keys for JS usage
    currentMapData.forEach((g) => {
      g.rows.forEach((r) => {
        // Ensure 'class' property exists if it came as something else
        if (r.class_ !== undefined && r.class === undefined) r.class = r.class_;
      });
    });

    renderMapEditor();

    isEditing = true;
    document.getElementById("editor-title").textContent = "Edit Prompt: " + filename;

    showEditorView();
  } catch (e) {
    alert("Error loading prompt: " + e.message);
  }
}

function showCreateView() {
  const inputFilename = document.getElementById("input-filename");
  inputFilename.value = "";
  inputFilename.disabled = false;

  document.getElementById("input-meta").value = DEFAULT_META_TEMPLATE;
  document.getElementById("input-execution").value = "## **4. Execution**\n\nProcess the input Markdown now. Output **ONLY** the CSV data block.";

  currentMapData = [];
  renderMapEditor();

  isEditing = false;
  document.getElementById("editor-title").textContent = "Create New Prompt";

  showEditorView();
}

function renderMapEditor() {
  const container = document.getElementById("map-container");
  container.innerHTML = "";

  currentMapData.forEach((group, gIndex) => {
    const groupEl = document.createElement("div");
    groupEl.className = "map-group";

    // Header
    groupEl.innerHTML = `
            <div class="map-group-header">
                <ui5-input class="group-name-input" value="${group.name}" placeholder="Group Name (e.g. 101 - Residential)" style="flex: 1; max-width: 400px;"></ui5-input>
                <div class="spacer"></div>
                <ui5-button icon="add" class="btn-add-row">Add Row</ui5-button>
                <ui5-button icon="delete" design="Negative" class="btn-delete-group" icon-only tooltip="Delete Group"></ui5-button>
            </div>
            <div class="map-group-notes">
                <ui5-text>Group Notes / Instructions (preserved from the original prompt)</ui5-text>
                <ui5-textarea class="group-notes-input" rows="4" placeholder="e.g. **CRITICAL INSTRUCTION:** ...">${group.notes || ""}</ui5-textarea>
            </div>
            <div class="map-rows-header">
                <ui5-text>Name</ui5-text>
                <ui5-text>Key</ui5-text>
                <ui5-text>Description</ui5-text>
                <ui5-text>Class</ui5-text>
                <ui5-text>Unit</ui5-text>
                <span></span>
            </div>
            <div class="map-rows-container"></div>
        `;

    // Event Listeners for Group
    const nameInput = groupEl.querySelector(".group-name-input");
    nameInput.addEventListener("input", (e) => {
      group.name = e.target.value;
    });

    const notesInput = groupEl.querySelector(".group-notes-input");
    if (notesInput) {
      notesInput.value = group.notes || "";
      notesInput.addEventListener("input", (e) => {
        group.notes = e.target.value;
      });
    }

    groupEl.querySelector(".btn-delete-group").addEventListener("click", () => {
      currentMapData.splice(gIndex, 1);
      renderMapEditor();
    });

    groupEl.querySelector(".btn-add-row").addEventListener("click", () => {
      group.rows.push({ name: "", key: "", desc: "", class: "", unit: "thm" });
      renderMapEditor();
    });

    const rowsContainer = groupEl.querySelector(".map-rows-container");

    group.rows.forEach((row, rIndex) => {
      const rowEl = document.createElement("div");
      rowEl.className = "map-row";
      rowEl.innerHTML = `
                <ui5-input value="${row.name}" placeholder="Name"></ui5-input>
                <ui5-input value="${row.key}" placeholder="Key"></ui5-input>
                <ui5-input value="${row.desc}" placeholder="Description"></ui5-input>
                <ui5-input value="${row.class || ""}" placeholder="Class"></ui5-input>
                <ui5-input value="${row.unit || ""}" placeholder="Unit"></ui5-input>
                <ui5-button icon="delete" design="Negative" icon-only class="btn-delete-row"></ui5-button>
            `;

      const inputs = rowEl.querySelectorAll("ui5-input");
      inputs[0].addEventListener("input", (e) => (row.name = e.target.value));
      inputs[1].addEventListener("input", (e) => (row.key = e.target.value));
      inputs[2].addEventListener("input", (e) => (row.desc = e.target.value));
      inputs[3].addEventListener("input", (e) => (row.class = e.target.value));
      inputs[4].addEventListener("input", (e) => (row.unit = e.target.value));

      rowEl.querySelector(".btn-delete-row").addEventListener("click", () => {
        group.rows.splice(rIndex, 1);
        renderMapEditor();
      });

      rowsContainer.appendChild(rowEl);
    });

    container.appendChild(groupEl);
  });
}

function addGroup() {
  currentMapData.push({ name: "New Group", rows: [] });
  renderMapEditor();
}

async function savePrompt() {
  const filenameInput = document.getElementById("input-filename");
  let filename = filenameInput.value.trim();
  if (!filename) {
    alert("Filename is required");
    return;
  }

  // Ensure .md extension for display/check, backend handles it too
  if (!filename.endsWith(".md")) {
    filename += ".md";
  }

  const payload = {
    structure: {
      meta: document.getElementById("input-meta").value,
      map_data: currentMapData,
      execution: document.getElementById("input-execution").value,
      is_structured: true
    }
  };

  if (!isEditing) {
    payload.filename = filename;
  }

  try {
    const method = isEditing ? "PUT" : "POST";
    const url = isEditing ? `/api/prompts/${filename}` : `/api/prompts/`;

    await request(url, method, payload);

    // Success
    // Use ui5-toast if available or alert
    alert("Saved successfully!");
    showListView();
    loadPrompts();
  } catch (e) {
    alert("Failed to save: " + e.message);
  }
}

async function deletePrompt(filename) {
  if (!confirm(`Are you sure you want to delete ${filename}?`)) return;
  try {
    await request(`/api/prompts/${filename}`, "DELETE");
    loadPrompts();
  } catch (e) {
    alert("Failed to delete: " + e.message);
  }
}

async function downloadPrompt(filename) {
  try {
    // Use the backend download endpoint directly via window.location for file download
    // But we need to handle API_KEY authentication if protected.
    // window.location.href sends a standard GET without custom headers.
    // If the API requires X-API-Key header, we must fetch with Blob.

    // We already have request() helper but it expects JSON response by default.
    // Let's implement a fetch with Blob here.

    const { API_BASE_URL, API_KEY } = await import("../../services/api.js");

    const response = await fetch(`${API_BASE_URL}/api/prompts/${filename}/download`, {
      headers: {
        "X-API-Key": API_KEY
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const contentDisposition = response.headers.get("content-disposition") || "";
    const getFilenameFromContentDisposition = (value) => {
      // Handles: attachment; filename="foo.md"
      // Minimal parsing (works for our server response)
      const m = /filename\*=UTF-8''([^;]+)|filename="([^"]+)"|filename=([^;]+)/i.exec(value || "");
      const raw = (m && (m[1] || m[2] || m[3])) || "";
      try {
        return decodeURIComponent(raw);
      } catch {
        return raw;
      }
    };

    const toSafeFilename = (name, fallback) => {
      const base = (name ?? "")
        .toString()
        .trim()
        .replace(/^["']|["']$/g, "");
      const cleaned = base
        // illegal on Windows + reserved URL-ish chars
        .replace(/[\/\\?%*:|"<>]/g, "-")
        .replace(/\s+/g, " ")
        .trim();
      return cleaned || fallback;
    };

    const headerFilename = getFilenameFromContentDisposition(contentDisposition);
    let downloadName = toSafeFilename(headerFilename || filename, "prompt.md");
    if (!downloadName.toLowerCase().endsWith(".md")) downloadName += ".md";

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = downloadName;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (e) {
    alert("Failed to download: " + e.message);
  }
}

function buildPromptMarkdownDraft() {
  const filenameInput = document.getElementById("input-filename");
  const meta = document.getElementById("input-meta")?.value || "";
  const execution = document.getElementById("input-execution")?.value || "";

  // Ensure we keep file format consistent with backend build_markdown_from_structure
  const lines = [];
  lines.push(meta.trim());
  lines.push("");
  lines.push("## **3. Master Extraction Map**");
  lines.push("");

  (currentMapData || []).forEach((group) => {
    lines.push(`### **${group.name || ""}**`);
    lines.push("");
    if ((group.notes || "").trim()) {
      lines.push(group.notes.trim());
      lines.push("");
    }
    (group.rows || []).forEach((row) => {
      const name = row.name || "";
      const key = row.key || "";
      const desc = row.desc || "";
      const klass = row.class || row.class_ || "";
      const unit = row.unit || "";
      lines.push(`- **${name}** -> \`${key}\` | \`${desc}\` | \`${klass}\` | \`${unit}\``);
    });
    lines.push("");
  });

  if (execution.trim()) {
    lines.push("---");
    lines.push("");
    lines.push(execution.trim());
  }
  lines.push("");
  return lines.join("\n");
}

async function downloadPromptDraft() {
  try {
    const toSafeFilename = (name, fallback) => {
      const base = (name ?? "")
        .toString()
        .trim()
        .replace(/^["']|["']$/g, "");
      const cleaned = base
        .replace(/[\/\\?%*:|"<>]/g, "-")
        .replace(/\s+/g, " ")
        .trim();
      return cleaned || fallback;
    };

    const defaultName = `prompt-draft-${new Date().toISOString().slice(0, 10)}.md`;
    let filename = toSafeFilename(document.getElementById("input-filename")?.value, defaultName);
    if (!filename.toLowerCase().endsWith(".md")) filename += ".md";

    const content = buildPromptMarkdownDraft();
    const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (e) {
    alert("Failed to download: " + e.message);
  }
}

function showListView() {
  document.getElementById("prompts-list-view").style.display = "block";
  document.getElementById("prompts-editor-view").style.display = "none";
}

function showEditorView() {
  document.getElementById("prompts-list-view").style.display = "none";
  document.getElementById("prompts-editor-view").style.display = "block";
}
