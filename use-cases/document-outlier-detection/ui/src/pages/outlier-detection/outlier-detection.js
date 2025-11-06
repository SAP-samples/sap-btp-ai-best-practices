// Page-scoped UI5 component imports
import "@ui5/webcomponents/dist/FileUploader.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents-fiori/dist/IllustratedMessage.js";
import "@ui5/webcomponents-fiori/dist/illustrations/SuccessScreen.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Table.js";
import "@ui5/webcomponents/dist/TableRow.js";
import "@ui5/webcomponents/dist/TableCell.js";
import "@ui5/webcomponents/dist/TableHeaderRow.js";
import "@ui5/webcomponents/dist/TableHeaderCell.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";

import { requestMultipart, request } from "../../services/api.js";

function $(id) {
  return document.getElementById(id);
}

function excelColLetter(zeroBasedIndex) {
  let n = Number(zeroBasedIndex);
  let s = "";
  while (n >= 0) {
    s = String.fromCharCode((n % 26) + 65) + s;
    n = Math.floor(n / 26) - 1;
  }
  return s;
}

function toNumberLoose(v) {
  if (v == null) return null;
  if (typeof v === "number") return v;
  const s = String(v).replace(/,/g, "").trim();
  const num = parseFloat(s);
  return Number.isNaN(num) ? null : num;
}

function renderPreview(preview, issues = []) {
  const container = $("preview");
  if (!preview || !preview.headers || !preview.firstRows) {
    container.classList.add("hidden");
    container.innerHTML = "";
    return;
  }
  const headers = preview.headers;
  const rows = preview.firstRows;
  const rowNumbers = Array.isArray(preview.rowNumbers) ? preview.rowNumbers : [];
  const colLetters = headers.map((_, i) => {
    const fromApi = Array.isArray(preview.colLetters) ? preview.colLetters[i] : undefined;
    return fromApi || excelColLetter(i);
  });
  // Build maps/sets for flagged items
  const nameToIndex = new Map();
  headers.forEach((h, i) => nameToIndex.set(String(h).toLowerCase(), i));
  const flagged = new Map();
  const flaggedRows = new Set();
  const flaggedCols = new Set();
  (Array.isArray(issues) ? issues : []).forEach((it) => {
    const r = typeof it.rowIndex === "number" ? it.rowIndex : null;
    let cIdx = typeof it.columnIndex === "number" ? it.columnIndex : null;
    if (cIdx == null && it.column != null) {
      if (typeof it.column === "number") {
        // Accept numeric index (assume 0-based); if 1-based, backend also handles
        cIdx = it.column;
      } else {
        const asString = String(it.column).trim();
        const idxByName = nameToIndex.get(asString.toLowerCase());
        if (typeof idxByName === "number") {
          cIdx = idxByName;
        } else {
          const maybeNum = Number(asString);
          if (!Number.isNaN(maybeNum)) cIdx = maybeNum;
        }
      }
    }
    if (r != null && cIdx != null) {
      flagged.set(`${r}:${cIdx}`, it.reason || "Flagged by validation");
      flaggedRows.add(r);
      flaggedCols.add(cIdx);
    }
  });

  const table = document.createElement("ui5-table");
  table.style.width = "100%";
  table.setAttribute("overflow-mode", "Scroll");

  const headerRow = document.createElement("ui5-table-header-row");
  headerRow.setAttribute("slot", "headerRow");
  // Top-left corner for row indices
  const hcIdx = document.createElement("ui5-table-header-cell");
  hcIdx.textContent = "#";
  // Use component API to define a narrow index column (preferred over CSS)
  hcIdx.setAttribute("width", "80px");
  hcIdx.setAttribute("min-width", "80px");
  headerRow.appendChild(hcIdx);
  // Column headers labeled with Excel letters only
  headers.forEach((_, i) => {
    const hc = document.createElement("ui5-table-header-cell");
    const letter = colLetters[i] || "";
    hc.textContent = letter || "";
    // Ensure minimum width for all data columns
    hc.setAttribute("min-width", "80px");
    if (flaggedCols.has(i)) {
      hc.classList.add("flagged-col-header");
    }
    headerRow.appendChild(hc);
  });
  table.appendChild(headerRow);

  rows.forEach((r, rIdx) => {
    const rowEl = document.createElement("ui5-table-row");
    const isRowFlagged = flaggedRows.has(rIdx);
    // Row index cell
    const rowNumCell = document.createElement("ui5-table-cell");
    rowNumCell.textContent = String(rowNumbers[rIdx] ?? rIdx + 1);
    if (isRowFlagged) rowNumCell.classList.add("flagged-row-cell");
    rowEl.appendChild(rowNumCell);
    r.forEach((c, cIdx) => {
      const cell = document.createElement("ui5-table-cell");
      const key = `${rIdx}:${cIdx}`;
      const reason = flagged.get(key);
      if (reason) {
        cell.classList.add("flagged-cell");
        const coord = `${colLetters[cIdx] || ""}${rowNumbers[rIdx] ?? rIdx + 1}`;
        cell.setAttribute("title", coord ? `${coord}: ${reason}` : reason);
      }
      if (isRowFlagged) cell.classList.add("flagged-row-cell");
      if (flaggedCols.has(cIdx)) cell.classList.add("flagged-col-cell");
      cell.textContent = c === null || c === undefined ? "" : String(c);
      rowEl.appendChild(cell);
    });
    table.appendChild(rowEl);
  });
  container.innerHTML = "";
  container.appendChild(table);
  container.classList.remove("hidden");
}

function renderIssues(result) {
  const container = $("issues");
  container.innerHTML = "";
  if (!result) {
    container.classList.add("hidden");
    return;
  }
  const ok = !!result.ok;
  const issues = Array.isArray(result.issues) ? result.issues : [];
  const summary = result.summary || (ok ? "No issues detected" : "Validation completed");

  const title = document.createElement("ui5-title");
  title.setAttribute("level", "H4");
  title.setAttribute("size", "H4");
  title.textContent = ok ? "No issues found" : `Issues (${issues.length})`;
  container.appendChild(title);

  const p = document.createElement("ui5-text");
  p.textContent = summary;
  container.appendChild(p);

  if (issues.length > 0) {
    const list = document.createElement("div");
    list.className = "issues-list";
    const last = window.__excelValidationLast || {};
    const preview = last.preview || {};
    const headers = Array.isArray(preview.headers) ? preview.headers : [];
    const colLetters = Array.isArray(preview.colLetters) ? preview.colLetters : headers.map((_, i) => excelColLetter(i));
    const rowNumbers = Array.isArray(preview.rowNumbers) ? preview.rowNumbers : [];
    const rows = Array.isArray(preview.firstRows) ? preview.firstRows : [];
    const nameToIndex = new Map(headers.map((h, i) => [String(h).toLowerCase(), i]));

    issues.forEach((it) => {
      const card = document.createElement("ui5-card");
      const content = document.createElement("div");
      content.style.padding = "0.5rem";
      let rIdx = typeof it.rowIndex === "number" ? it.rowIndex : null;
      // Prefer explicit columnIndex first
      let cIdx = typeof it.columnIndex === "number" ? it.columnIndex : null;
      // Handle numeric or numeric-string `column` values
      if (cIdx == null && it.column != null) {
        if (typeof it.column === "number") {
          cIdx = it.column;
        } else {
          const asString = String(it.column).trim();
          const idxByName = nameToIndex.get(asString.toLowerCase());
          if (typeof idxByName === "number") {
            cIdx = idxByName;
          } else {
            const maybeNum = Number(asString);
            if (!Number.isNaN(maybeNum)) cIdx = maybeNum;
          }
        }
      }
      if (rIdx != null && cIdx == null) {
        const row = rows[rIdx] || [];
        // try to match by value
        if (row && row.length) {
          const target = toNumberLoose(it.value);
          if (target != null) {
            // exact string match first
            for (let j = 0; j < row.length && cIdx == null; j++) {
              const cellStr = row[j] == null ? null : String(row[j]);
              if (cellStr === String(it.value)) cIdx = j;
            }
            // numeric tolerant match
            if (cIdx == null) {
              let bestJ = null;
              let bestErr = Infinity;
              for (let j = 0; j < row.length; j++) {
                const num = toNumberLoose(row[j]);
                if (num == null) continue;
                const err = Math.abs(num - target);
                if (err < bestErr) {
                  bestErr = err;
                  bestJ = j;
                }
              }
              // Accept if reasonably close
              if (bestJ != null && bestErr <= Math.max(1e-6, Math.abs(target) * 1e-6)) {
                cIdx = bestJ;
              }
            }
          }
        }
      }
      const excelRow = rIdx != null && rowNumbers[rIdx] != null ? rowNumbers[rIdx] : rIdx != null ? rIdx + 1 : "";
      const excelCol = cIdx != null && colLetters[cIdx] != null ? colLetters[cIdx] : "";
      let coord = "";
      if (typeof it.excelCell === "string" && it.excelCell.trim()) {
        coord = it.excelCell;
      } else if (excelCol && excelRow) {
        coord = `${excelCol}${excelRow}`;
      }
      content.innerHTML = `
        <div><b>Cell</b>: ${coord || ""}</div>
        <div><b>Value</b>: ${it.value}</div>
        <div><b>Reason</b>: ${it.reason}</div>
      `;
      card.appendChild(content);
      list.appendChild(card);
    });
    container.appendChild(list);
  }

  container.classList.remove("hidden");
}

async function handleUpload() {
  const status = $("status");
  const fileInput = $("fileInput");
  const busy = $("previewBusy");
  const previewDiv = $("preview");
  const issuesDiv = $("issues");
  const btnSave = $("btnSave");
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    status.textContent = "Please choose an Excel file (.xlsx/.xlsm).";
    return;
  }
  if (status) status.textContent = ""; // show text inside BusyIndicator instead
  btnSave.disabled = true;
  if (busy) busy.active = true;
  if (busy) busy.text = "Uploading and validating...";
  // Clear previous results while loading new file
  if (previewDiv) {
    previewDiv.innerHTML = "";
    previewDiv.classList.add("hidden");
  }
  if (issuesDiv) {
    issuesDiv.innerHTML = "";
    issuesDiv.classList.add("hidden");
  }
  window.__excelValidationLast = null;

  try {
    const formData = new FormData();
    formData.append("file", file);
    const resp = await requestMultipart("/api/excel-validation/validate", formData);
    if (!resp.success) {
      if (busy) busy.text = `Validation failed: ${resp.error || "unknown error"}`;
      if (busy) busy.active = false;
      return;
    }
    if (busy) busy.text = "Rendering preview...";
    renderPreview(resp.preview, resp.result?.issues || []);
    renderIssues(resp.result);
    console.log("resp.result", resp.result);
    console.log("resp.preview", resp.preview);
    if (busy) busy.text = "";
    btnSave.disabled = false;
    // Store last response for save call
    window.__excelValidationLast = resp;
  } catch (e) {
    if (busy) busy.text = `Error: ${e.message}`;
  }
  if (busy) busy.active = false;
}

async function handleSave() {
  const data = window.__excelValidationLast;
  const dialog = $("saveDialog");
  try {
    await request("/api/excel-validation/save", "POST", {
      preview: data?.preview || null,
      result: data?.result || null
    });
    dialog.open = true;
  } catch (e) {
    // If API fails, still show dialog to satisfy mock requirement
    dialog.open = true;
  }
}

export default function init() {
  const fu = $("fileInput");
  if (fu) {
    fu.addEventListener("change", handleUpload);
  }
  $("btnSave").addEventListener("click", handleSave);
  $("closeDialog").addEventListener("click", () => ($("saveDialog").open = false));
}
