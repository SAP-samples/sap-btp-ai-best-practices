import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/FileUploader.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Table.js";
import "@ui5/webcomponents/dist/TableRow.js";
import "@ui5/webcomponents/dist/TableCell.js";
import "@ui5/webcomponents/dist/TableHeaderRow.js";
import "@ui5/webcomponents/dist/TableHeaderCell.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents-icons/dist/document.js";
import "@ui5/webcomponents-icons/dist/table-view.js";
import "@ui5/webcomponents-icons/dist/full-screen.js";
import "@ui5/webcomponents-icons/dist/decline.js";

import { uploadPdf, request } from "../../services/api.js";

export default function initUtilitiesRateCompareAndExportPage() {
  const errorStrip = document.getElementById("rate-error-strip");
  const resultsCard = document.getElementById("rate-results-card");
  const diffCard = document.getElementById("rate-diff-card");
  const diffDownloadBtn = document.getElementById("download-diff-csv-button");
  const diffExpandBtn = document.getElementById("expand-diff-table-button");

  // State for two files
  const state = {
    1: { csv: "", markdown: "", headers: [], rows: [], processing: false },
    2: { csv: "", markdown: "", headers: [], rows: [], processing: false }
  };

  // State for diff between current month (Document 1) and previous month (Document 2)
  const diffState = {
    headersDisplay: [],
    rowsDisplay: [],
    csvString: ""
  };

  const setSectionBusy = (id, active, message = "") => {
    const busyEl = document.getElementById(`rate-busy-${id}`);
    const busyRow = busyEl?.parentElement;
    const uploader = document.getElementById(`rate-pdf-uploader-${id}`);
    const mdUploader = document.getElementById(`rate-md-uploader-${id}`);
    const jsonUploader = document.getElementById(`rate-json-uploader-${id}`);
    const chooseBtn = document.getElementById(`rate-choose-pdf-${id}`);
    const mdChooseBtn = document.getElementById(`rate-choose-md-${id}`);
    const jsonChooseBtn = document.getElementById(`rate-choose-json-${id}`);
    const typeSelect = document.getElementById(`rate-type-select-${id}`);

    if (busyEl) {
      busyEl.active = active;
      busyEl.text = message || (active ? "Processing..." : "");
    }

    // Show/hide the busy row
    if (busyRow) {
      busyRow.style.display = active ? "" : "none";
    }

    // Disable inputs only for this section
    if (uploader) uploader.disabled = active;
    if (mdUploader) mdUploader.disabled = active;
    if (jsonUploader) jsonUploader.disabled = active;
    if (chooseBtn) chooseBtn.disabled = active;
    if (mdChooseBtn) mdChooseBtn.disabled = active;
    if (jsonChooseBtn) jsonChooseBtn.disabled = active;
    if (typeSelect) typeSelect.disabled = active;
  };

  const showError = (msg) => {
    if (errorStrip) {
      errorStrip.textContent = msg || "An unexpected error occurred.";
      errorStrip.style.display = "";
    }
  };

  const clearError = () => {
    if (errorStrip) {
      errorStrip.style.display = "none";
      errorStrip.textContent = "";
    }
  };

  const clearTable = (id) => {
    const table = document.getElementById(`rate-mapping-table-${id}`);
    if (!table) return;
    while (table.firstChild) {
      table.removeChild(table.firstChild);
    }
    table.noDataText = `No data for Document ${id}`;
  };

  const clearDiffTable = () => {
    const table = document.getElementById("rate-diff-table");
    if (!table) return;
    while (table.firstChild) {
      table.removeChild(table.firstChild);
    }
    table.noDataText = "No changed rates detected";
  };

  const buildTable = (id, headers, rows) => {
    const table = document.getElementById(`rate-mapping-table-${id}`);
    if (!table) return;
    clearTable(id);

    if (!headers || !headers.length) {
      table.noDataText = "No mapped rows returned.";
      return;
    }

    const headerRow = document.createElement("ui5-table-header-row");
    headerRow.slot = "headerRow";
    headerRow.sticky = true;
    headers.forEach((h) => {
      const cell = document.createElement("ui5-table-header-cell");
      cell.textContent = h;
      cell.title = h; // show full header on hover
      cell.style.width = "120px";
      headerRow.appendChild(cell);
    });
    table.appendChild(headerRow);

    if (rows && rows.length) {
      rows.forEach((row) => {
        const tr = document.createElement("ui5-table-row");
        row.forEach((cellValue) => {
          const cell = document.createElement("ui5-table-cell");
          cell.textContent = cellValue ?? "";
          cell.title = cellValue != null ? String(cellValue) : "";
          cell.style.width = "120px";
          tr.appendChild(cell);
        });
        table.appendChild(tr);
      });
    }
  };

  const buildDiffTable = (headers, rows, rowChanges) => {
    const table = document.getElementById("rate-diff-table");
    if (!table) return;
    clearDiffTable();

    if (!headers || !headers.length) {
      table.noDataText = "No changed rates detected";
      return;
    }

    const headerRow = document.createElement("ui5-table-header-row");
    headerRow.slot = "headerRow";
    headerRow.sticky = true;
    headers.forEach((h) => {
      const cell = document.createElement("ui5-table-header-cell");
      cell.textContent = h;
      cell.title = h;
      cell.style.width = "140px";
      headerRow.appendChild(cell);
    });
    table.appendChild(headerRow);

    if (rows && rows.length) {
      rows.forEach((row, rowIndex) => {
        const tr = document.createElement("ui5-table-row");
        const status = row[0]; // RowStatus is first column
        const changedCols = rowChanges && rowChanges[rowIndex] ? rowChanges[rowIndex] : [];

        // Row-level styling
        if (status === "Changed") {
          tr.style.backgroundColor = "#fff8e1"; // Very light orange
        } else if (status === "Added") {
          tr.style.backgroundColor = "#dcedc8"; // Light green
        } else if (status === "Removed") {
          tr.style.backgroundColor = "#ffcdd2"; // Light red
        }
        // status === "Unchanged" gets no background color

        row.forEach((cellValue, colIndex) => {
          const cell = document.createElement("ui5-table-cell");
          const value = cellValue != null ? String(cellValue) : "";
          cell.textContent = value;
          cell.title = value;
          cell.style.width = "140px";

          // Cell-level styling for Changed rows
          if (status === "Changed") {
            const colName = headers[colIndex];
            // Highlight the changed column and its "OLD" counterpart if adjacent
            if (changedCols.includes(colName)) {
              cell.style.backgroundColor = "#ffe0b2"; // Darker orange
              cell.style.fontWeight = "bold";
            }
            // Check if this is an OLD column for a changed field
            else if (colName.endsWith("_OLD")) {
              const baseName = colName.replace("_OLD", "");
              if (changedCols.includes(baseName)) {
                cell.style.backgroundColor = "#ffe0b2"; // Darker orange
              }
            }
          }

          tr.appendChild(cell);
        });
        table.appendChild(tr);
      });
    }
  };

  const handleDownload = (id) => {
    const csv = state[id].csv;
    if (!csv) return;
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mapped_rates_${id}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  };

  const handleDownloadDiff = () => {
    const csv = diffState.csvString;
    if (!csv) return;
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "mapped_rates_changes.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  };

  const buildExpandedTable = (headers, rows) => {
    const table = document.getElementById("rate-expanded-table");
    if (!table) return;

    // Clear existing
    while (table.firstChild) {
      table.removeChild(table.firstChild);
    }

    if (!headers || !headers.length) {
      table.noDataText = "No data available";
      return;
    }

    const headerRow = document.createElement("ui5-table-header-row");
    headerRow.slot = "headerRow";
    headerRow.sticky = true;
    headers.forEach((h) => {
      const cell = document.createElement("ui5-table-header-cell");
      cell.textContent = h;
      cell.title = h;
      cell.style.width = "150px";
      headerRow.appendChild(cell);
    });
    table.appendChild(headerRow);

    if (rows && rows.length) {
      rows.forEach((row) => {
        const tr = document.createElement("ui5-table-row");
        row.forEach((cellValue) => {
          const cell = document.createElement("ui5-table-cell");
          cell.textContent = cellValue ?? "";
          cell.title = cellValue != null ? String(cellValue) : "";
          tr.appendChild(cell);
        });
        table.appendChild(tr);
      });
    }
  };

  const handleExpand = (id) => {
    const dialog = document.getElementById("rate-expanded-dialog");
    const title = document.getElementById("rate-expanded-title");
    if (!dialog) return;

    if (title) title.textContent = `Expanded View - Document ${id}`;

    const headers = state[id].headers;
    const rows = state[id].rows;
    buildExpandedTable(headers, rows);

    dialog.open = true;
  };

  const handleExpandDiff = () => {
    const dialog = document.getElementById("rate-expanded-dialog");
    const title = document.getElementById("rate-expanded-title");
    if (!dialog) return;

    if (title) title.textContent = "Expanded View - Changed Rates";

    const headers = diffState.headersDisplay;
    const rows = diffState.rowsDisplay;
    // We could pass rowChanges here if we updated buildExpandedTable to support styling
    // For now, we'll just show the data
    buildExpandedTable(headers, rows);

    dialog.open = true;
  };

  const setContentVisibility = (id, visible) => {
    const tableContainer = document.querySelector(`#rate-result-section-${id} .rate-mapping-table-container`);

    if (tableContainer) tableContainer.style.display = visible ? "" : "none";
  };

  const setDiffVisibility = (hasBothDocs, hasRows) => {
    if (!diffCard) return;
    if (!hasBothDocs) {
      diffCard.style.display = "none";
      return;
    }
    diffCard.style.display = "";
    if (diffDownloadBtn) {
      diffDownloadBtn.disabled = !hasRows || !diffState.csvString;
    }
    if (diffExpandBtn) {
      diffExpandBtn.disabled = !hasRows;
    }
  };

  const recomputeDiff = async () => {
    const csv1 = state[1].csv;
    const csv2 = state[2].csv;

    // Reset diff state
    diffState.headersDisplay = [];
    diffState.rowsDisplay = [];
    diffState.csvString = "";
    clearDiffTable();

    // Check if both CSVs are available
    if (!csv1 || !csv2) {
      setDiffVisibility(false, false);
      return;
    }

    try {
      const result = await request("/api/rate-mapping/compare", "POST", {
        old_csv: csv2, // Doc 2 is Previous
        new_csv: csv1 // Doc 1 is Current
      });

      const headers = result.headers || [];
      const rows = result.rows || [];
      const rowChanges = result.row_changes || [];

      diffState.headersDisplay = headers;
      diffState.rowsDisplay = rows;

      if (rows.length) {
        const headerLine = headers.join(";");
        const rowLines = rows.map((row) => row.map((v) => (v != null ? String(v) : "")).join(";"));
        diffState.csvString = [headerLine, ...rowLines].join("\n");

        buildDiffTable(headers, rows, rowChanges);
        setDiffVisibility(true, true);
      } else {
        setDiffVisibility(true, false);
      }
    } catch (e) {
      console.error("Diff failed", e);
      // setDiffVisibility(true, false);
    }
  };

  const processFile = async (id, fileType) => {
    let uploader, file;
    if (fileType === "pdf") {
      uploader = document.getElementById(`rate-pdf-uploader-${id}`);
      file = uploader?.files?.[0];
    } else if (fileType === "md") {
      uploader = document.getElementById(`rate-md-uploader-${id}`);
      file = uploader?.files?.[0];
    } else if (fileType === "json") {
      uploader = document.getElementById(`rate-json-uploader-${id}`);
      file = uploader?.files?.[0];
    }

    if (!file) return;

    // Basic validation
    const name = file.name?.toLowerCase() || "";
    if (fileType === "pdf" && !name.endsWith(".pdf")) {
      showError("Please select a .pdf file.");
      return;
    }
    if (fileType === "md" && !name.endsWith(".md") && !name.endsWith(".markdown") && !name.endsWith(".txt")) {
      showError("Please select a .md, .markdown, or .txt file.");
      return;
    }
    if (fileType === "json" && !name.endsWith(".json")) {
      showError("Please select a .json file.");
      return;
    }

    clearError();
    state[id].processing = true;

    // Ensure results card is visible
    if (resultsCard) resultsCard.style.display = "";

    // Show specific result section
    const resultSection = document.getElementById(`rate-result-section-${id}`);
    if (resultSection) resultSection.style.visibility = "visible";

    // Set filename
    const filenameEl = document.getElementById(`rate-result-filename-${id}`);
    if (filenameEl) filenameEl.textContent = file.name || "";

    setSectionBusy(id, true, `Processing Document ${id}...`);

    // Hide content while processing
    setContentVisibility(id, false);

    // Reset UI for this ID
    clearTable(id);
    state[id].csv = "";
    state[id].markdown = "";
    state[id].headers = [];
    state[id].rows = [];

    const downloadBtn = document.getElementById(`download-csv-button-${id}`);
    const expandBtn = document.getElementById(`expand-table-button-${id}`);
    const mdPanel = document.getElementById(`rate-markdown-panel-${id}`);
    const mdOutput = document.getElementById(`rate-markdown-output-${id}`);
    const typeSelect = document.getElementById(`rate-type-select-${id}`);

    // Get selected file type
    const fileTypeKey = typeSelect && typeSelect.selectedOption ? typeSelect.selectedOption.value : "nc-rates";

    if (downloadBtn) downloadBtn.disabled = true;
    if (expandBtn) expandBtn.disabled = true;
    if (mdPanel) mdPanel.style.display = "none";
    if (mdOutput) mdOutput.textContent = "";

    try {
      let result;

      if (fileType === "json") {
        // For JSON files, read directly and parse
        const text = await file.text();
        result = JSON.parse(text);
      } else {
        // For PDF and MD files, call the API
        const endpointBase = fileType === "pdf" ? "/api/rate-mapping/pdf" : "/api/rate-mapping/markdown";
        const endpoint = `${endpointBase}?file_type=${encodeURIComponent(fileTypeKey)}`;
        result = await uploadPdf(endpoint, file);
      }

      console.log("result", result); // Keep this for debugging

      if (result.tokenUsage) {
        console.log(`Token Usage for Document ${id}:`, result.tokenUsage);
        if (result.tokenUsage.total) {
          console.log(`Input Tokens (Prompt) for Document ${id}:`, result.tokenUsage.total.prompt_tokens);
          console.log(`Output Tokens (Completion) for Document ${id}:`, result.tokenUsage.total.completion_tokens);
          console.log(`Total Tokens for Document ${id}:`, result.tokenUsage.total.total_tokens);
        }
      }

      const headers = result.headers || [];
      const rows = result.rows || [];
      const csvString = result.csvString || "";
      const markdown = result.markdown || "";

      buildTable(id, headers, rows);

      state[id].csv = csvString;
      state[id].markdown = markdown;
      state[id].headers = headers;
      state[id].rows = rows;

      if (downloadBtn) downloadBtn.disabled = !csvString;
      if (expandBtn) expandBtn.disabled = !rows || !rows.length;

      if (mdOutput) mdOutput.textContent = markdown;

      if (mdPanel) {
        mdPanel.style.display = markdown ? "" : "none";
        // Ensure it starts collapsed so it doesn't overwhelm
        mdPanel.collapsed = true;
      }

      // Show content after successful processing
      setContentVisibility(id, true);
    } catch (err) {
      console.error(err);
      showError(err?.message || `Error processing Document ${id}.`);
      setContentVisibility(id, true);
    } finally {
      state[id].processing = false;
      setSectionBusy(id, false);

      // Clear uploader value to allow re-selecting same file
      try {
        if (uploader) uploader.value = "";
      } catch (_) {}
    }

    // Attempt to recompute diff whenever one side finishes processing (success or error)
    await recomputeDiff();
  };

  // Fetch File Types on init
  const fetchFileTypes = async () => {
    try {
      const types = await request("/api/rate-mapping/types");
      [1, 2].forEach((id) => {
        const select = document.getElementById(`rate-type-select-${id}`);
        if (select) {
          while (select.firstChild) {
            select.removeChild(select.firstChild);
          }
          types.forEach((t) => {
            const opt = document.createElement("ui5-option");
            opt.value = t.key;
            opt.textContent = t.label;
            select.appendChild(opt);
          });
          // If options exist, ui5-select selects the first one by default.
        }
      });
    } catch (e) {
      console.error("Failed to fetch file types", e);
      showError("Failed to load file types.");
    }
  };

  fetchFileTypes();

  // Setup event listeners for both sets
  [1, 2].forEach((id) => {
    const pdfUploader = document.getElementById(`rate-pdf-uploader-${id}`);
    const mdUploader = document.getElementById(`rate-md-uploader-${id}`);
    const jsonUploader = document.getElementById(`rate-json-uploader-${id}`);
    const choosePdfBtn = document.getElementById(`rate-choose-pdf-${id}`);
    const chooseMdBtn = document.getElementById(`rate-choose-md-${id}`);
    const chooseJsonBtn = document.getElementById(`rate-choose-json-${id}`);
    const downloadBtn = document.getElementById(`download-csv-button-${id}`);
    const expandBtn = document.getElementById(`expand-table-button-${id}`);

    if (choosePdfBtn && pdfUploader) {
      choosePdfBtn.addEventListener("click", () => {
        try {
          pdfUploader.value = "";
        } catch (_) {}
      });
    }
    if (chooseMdBtn && mdUploader) {
      chooseMdBtn.addEventListener("click", () => {
        try {
          mdUploader.value = "";
        } catch (_) {}
      });
    }
    if (chooseJsonBtn && jsonUploader) {
      chooseJsonBtn.addEventListener("click", () => {
        try {
          jsonUploader.value = "";
        } catch (_) {}
      });
    }

    if (pdfUploader) {
      pdfUploader.addEventListener("change", () => processFile(id, "pdf"));
    }
    if (mdUploader) {
      mdUploader.addEventListener("change", () => processFile(id, "md"));
    }
    if (jsonUploader) {
      jsonUploader.addEventListener("change", () => processFile(id, "json"));
    }

    if (downloadBtn) {
      downloadBtn.addEventListener("click", () => handleDownload(id));
    }
    if (expandBtn) {
      expandBtn.addEventListener("click", () => handleExpand(id));
    }
  });

  if (diffDownloadBtn) {
    diffDownloadBtn.addEventListener("click", handleDownloadDiff);
  }
  if (diffExpandBtn) {
    diffExpandBtn.addEventListener("click", handleExpandDiff);
  }

  // Dialog close handler
  const dialog = document.getElementById("rate-expanded-dialog");
  const closeBtn = document.getElementById("rate-expanded-close");
  if (closeBtn && dialog) {
    closeBtn.addEventListener("click", () => {
      dialog.open = false;
    });
  }
}
