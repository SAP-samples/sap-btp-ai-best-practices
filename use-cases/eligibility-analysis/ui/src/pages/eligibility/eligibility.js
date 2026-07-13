/* UI5 Web Components for Eligibility Page */
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/FileUploader.js";
import "@ui5/webcomponents/dist/DatePicker.js";
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
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/SegmentedButton.js";
import "@ui5/webcomponents/dist/SegmentedButtonItem.js";

import Chart from "chart.js/auto";
import { request, API_BASE_URL, API_KEY } from "../../services/api.js";

// Application state
let appState = {
  selectedFile: null,
  isAnalyzing: false,
  analysisResult: null,
  outputFileName: null,
  config: {
    purchaseDate: null,
    nddt: null,
    teih: null,
    isspur: null,
    currencies: []
  },
  currentSellerId: null,
  sellerSummary: null,
  sellerHistory: [],
  historyOffset: 0,
  historyLimit: 50,
  isLoadingSeller: false,
  insightsCharts: {},
  isLoadingInsights: false,
  insightsLookbackDays: 90
};

export default function initEligibilityPage() {
  console.log("Eligibility page initialized");

  // Set up event listeners
  setupEventListeners();

  // Load default configuration
  loadDefaultConfig();
}

function setupEventListeners() {
  // File uploader
  const fileUploader = document.getElementById("file-uploader");
  if (fileUploader) {
    fileUploader.addEventListener("change", handleFileSelect);
  }

  // Analyze button
  const analyzeBtn = document.getElementById("analyze-btn");
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", analyzeFile);
  }

  // Download button
  const downloadBtn = document.getElementById("download-btn");
  if (downloadBtn) {
    downloadBtn.addEventListener("click", downloadSummary);
  }

  // Load defaults button
  const loadDefaultsBtn = document.getElementById("load-defaults-btn");
  if (loadDefaultsBtn) {
    loadDefaultsBtn.addEventListener("click", loadDefaultConfig);
  }

  const resetLogsBtn = document.getElementById("reset-logs-btn");
  if (resetLogsBtn) {
    resetLogsBtn.addEventListener("click", resetCustomerLogs);
  }

  // Seller statistics
  const loadSellerBtn = document.getElementById("load-seller-btn");
  if (loadSellerBtn) {
    loadSellerBtn.addEventListener("click", () => {
      const sellerIdInput = document.getElementById("seller-id-input");
      if (sellerIdInput?.value?.trim()) {
        fetchSellerStatistics(sellerIdInput.value.trim());
      }
    });
  }

  // Enter key in seller ID input
  const sellerIdInput = document.getElementById("seller-id-input");
  if (sellerIdInput) {
    sellerIdInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && sellerIdInput.value?.trim()) {
        fetchSellerStatistics(sellerIdInput.value.trim());
      }
    });
  }

  // Load more history button
  const loadMoreHistoryBtn = document.getElementById("load-more-history-btn");
  if (loadMoreHistoryBtn) {
    loadMoreHistoryBtn.addEventListener("click", () => {
      if (appState.currentSellerId) {
        fetchSellerHistory(appState.currentSellerId, false);
      }
    });
  }

  // Insights - Analyze Patterns button
  const loadInsightsBtn = document.getElementById("load-insights-btn");
  if (loadInsightsBtn) {
    loadInsightsBtn.addEventListener("click", fetchInsights);
  }

  const insightsTimeframe = document.getElementById("insights-timeframe");
  if (insightsTimeframe) {
    const applyInsightsTimeframeFromItem = (item) => {
      const lookbackDays = getLookbackDaysFromItem(item);
      if (lookbackDays > 0) {
        appState.insightsLookbackDays = lookbackDays;
      }
    };

    insightsTimeframe.addEventListener("selection-change", (event) => {
      const selectedItem = event.detail?.selectedItem
        || event.detail?.targetItem
        || event.detail?.selectedItems?.[0];
      if (selectedItem) {
        applyInsightsTimeframeFromItem(selectedItem);
        return;
      }

      const pressedItem = Array.from(
        insightsTimeframe.querySelectorAll("ui5-segmented-button-item"),
      ).find((item) => item.pressed === true || item.getAttribute("aria-pressed") === "true");
      applyInsightsTimeframeFromItem(pressedItem);
    });

    insightsTimeframe.querySelectorAll("ui5-segmented-button-item").forEach((item) => {
      item.addEventListener("click", () => applyInsightsTimeframeFromItem(item));
    });

    const initialItem = Array.from(
      insightsTimeframe.querySelectorAll("ui5-segmented-button-item"),
    ).find((item) => item.pressed === true || item.getAttribute("aria-pressed") === "true");
    applyInsightsTimeframeFromItem(initialItem);
  }

  // Export PDF button
  const exportPdfBtn = document.getElementById("export-pdf-btn");
  if (exportPdfBtn) {
    exportPdfBtn.addEventListener("click", exportPatternPdf);
  }

  // Load filter options for dropdowns
  loadFilterOptions();
}

async function loadDefaultConfig() {
  try {
    setMainLoading(true);
    hideSuccess();

    const config = await request("/api/eligibility/config");

    // Update state
    appState.config = {
      purchaseDate: config.purchase_date || getTodayDate(),
      nddt: config.nddt,
      teih: config.teih,
      isspur: config.isspur,
      currencies: config.eligible_currencies || []
    };

    // Update UI
    updateConfigUI();

    console.log("Loaded default config:", config);
  } catch (error) {
    console.error("Failed to load default config:", error);
    showError(`Failed to load configuration: ${error.message}`);

    // Set reasonable defaults
    appState.config = {
      purchaseDate: getTodayDate(),
      nddt: 30,
      teih: 90,
      isspur: 120,
      currencies: ["USD", "EUR", "GBP"]
    };
    updateConfigUI();
  } finally {
    setMainLoading(false);
  }
}

function updateConfigUI() {
  const purchaseDatePicker = document.getElementById("purchase-date");
  const nddtInput = document.getElementById("nddt-input");
  const teihInput = document.getElementById("teih-input");
  const isspurInput = document.getElementById("isspur-input");
  const currenciesInput = document.getElementById("currencies-input");

  if (purchaseDatePicker && appState.config.purchaseDate) {
    purchaseDatePicker.value = appState.config.purchaseDate;
  }

  if (nddtInput && appState.config.nddt != null) {
    nddtInput.value = appState.config.nddt.toString();
  }

  if (teihInput && appState.config.teih != null) {
    teihInput.value = appState.config.teih.toString();
  }

  if (isspurInput && appState.config.isspur != null) {
    isspurInput.value = appState.config.isspur.toString();
  }

  if (currenciesInput && appState.config.currencies) {
    currenciesInput.value = appState.config.currencies.join(",");
  }
}

function getConfigFromUI() {
  const purchaseDatePicker = document.getElementById("purchase-date");
  const nddtInput = document.getElementById("nddt-input");
  const teihInput = document.getElementById("teih-input");
  const isspurInput = document.getElementById("isspur-input");
  const currenciesInput = document.getElementById("currencies-input");

  return {
    purchaseDate: purchaseDatePicker?.value || getTodayDate(),
    nddt: parseInt(nddtInput?.value) || appState.config.nddt,
    teih: parseInt(teihInput?.value) || appState.config.teih,
    isspur: parseInt(isspurInput?.value) || appState.config.isspur,
    currencies: currenciesInput?.value?.split(",").map((c) => c.trim()).filter(Boolean) || appState.config.currencies
  };
}

function handleFileSelect(event) {
  const files = event.target.files;

  if (files && files.length > 0) {
    appState.selectedFile = files[0];

    // Update filename display
    const filenameSpan = document.getElementById("selected-filename");
    if (filenameSpan) {
      filenameSpan.textContent = appState.selectedFile.name;
    }

    // Enable analyze button
    const analyzeBtn = document.getElementById("analyze-btn");
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
    }
  }
}

async function analyzeFile() {
  if (!appState.selectedFile) {
    showError("Please select an Excel file first.");
    return;
  }

  const config = getConfigFromUI();

  try {
    setAnalyzeLoading(true);
    hideSuccess();
    hideError();

    // Build query parameters
    const params = new URLSearchParams();
    params.append("purchase_date", config.purchaseDate);
    if (config.nddt != null) params.append("nddt", config.nddt.toString());
    if (config.teih != null) params.append("teih", config.teih.toString());
    if (config.isspur != null) params.append("isspur", config.isspur.toString());
    if (config.currencies.length > 0) {
      params.append("eligible_currencies", config.currencies.join(","));
    }

    // Create FormData for file upload
    const formData = new FormData();
    formData.append("file", appState.selectedFile);

    // Make the request (cannot use request() because it sets Content-Type: application/json)
    const url = `${API_BASE_URL}/api/eligibility/analyze?${params.toString()}`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "X-API-Key": API_KEY
      },
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    appState.analysisResult = result;
    appState.outputFileName = result.output_file;

    // Display results
    displayAnalysisResults(result);

    console.log("Analysis complete:", result);
  } catch (error) {
    console.error("Failed to analyze file:", error);
    showError(`Analysis failed: ${error.message}`);
  } finally {
    setAnalyzeLoading(false);
  }
}

function displayAnalysisResults(result) {
  // Show results section
  const resultsSection = document.getElementById("results-section");
  if (resultsSection) {
    resultsSection.style.display = "block";
  }

  // Update summary counts
  const fundedCount = result.funded_invoices?.length || 0;
  const nonFundedCount = result.non_funded_invoices?.length || 0;
  const totalCount = fundedCount + nonFundedCount;

  document.getElementById("funded-count").textContent = fundedCount.toString();
  document.getElementById("non-funded-count").textContent = nonFundedCount.toString();
  document.getElementById("total-count").textContent = totalCount.toString();

  // Update panel headers
  const fundedPanel = document.getElementById("funded-panel");
  if (fundedPanel) {
    fundedPanel.setAttribute("header-text", `Funded Invoices (${fundedCount})`);
  }

  const nonFundedPanel = document.getElementById("non-funded-panel");
  if (nonFundedPanel) {
    nonFundedPanel.setAttribute("header-text", `Non-Funded Invoices (${nonFundedCount})`);
  }

  // Populate funded invoices table
  populateFundedTable(result.funded_invoices || []);

  // Populate non-funded invoices table
  populateNonFundedTable(result.non_funded_invoices || []);
}

function populateFundedTable(invoices) {
  const table = document.getElementById("funded-table");
  if (!table) return;

  // Clear existing rows (except header)
  const existingRows = table.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
  existingRows.forEach((row) => row.remove());

  if (invoices.length === 0) {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell colspan="6">
        <div class="table-empty">No funded invoices</div>
      </ui5-table-cell>
    `;
    table.appendChild(row);
    return;
  }

  invoices.forEach((invoice) => {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell>${invoice.invoice_number || "-"}</ui5-table-cell>
      <ui5-table-cell>${invoice.seller_id || "-"}</ui5-table-cell>
      <ui5-table-cell>${invoice.debtor_id || "-"}</ui5-table-cell>
      <ui5-table-cell>${invoice.currency || "-"}</ui5-table-cell>
      <ui5-table-cell>${formatAmount(invoice.amount, invoice.currency)}</ui5-table-cell>
      <ui5-table-cell>${invoice.due_date || "-"}</ui5-table-cell>
    `;
    table.appendChild(row);
  });
}

function populateNonFundedTable(invoices) {
  const table = document.getElementById("non-funded-table");
  if (!table) return;

  // Clear existing rows (except header)
  const existingRows = table.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
  existingRows.forEach((row) => row.remove());

  if (invoices.length === 0) {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell colspan="6">
        <div class="table-empty">No non-funded invoices</div>
      </ui5-table-cell>
    `;
    table.appendChild(row);
    return;
  }

  invoices.forEach((invoice, index) => {
    const row = document.createElement("ui5-table-row");
    row.classList.add("non-funded-row");
    row.dataset.invoiceKey = buildInvoiceKey(invoice, index);
    const reasons = invoice.rejection_reason
      || (invoice.rejection_reasons?.length ? invoice.rejection_reasons.join(", ") : "-");
    row.innerHTML = `
      <ui5-table-cell>${invoice.invoice_number || "-"}</ui5-table-cell>
      <ui5-table-cell>${invoice.seller_id || "-"}</ui5-table-cell>
      <ui5-table-cell>${invoice.debtor_id || "-"}</ui5-table-cell>
      <ui5-table-cell>${invoice.currency || "-"}</ui5-table-cell>
      <ui5-table-cell>${formatAmount(invoice.amount, invoice.currency)}</ui5-table-cell>
      <ui5-table-cell>${reasons}</ui5-table-cell>
    `;
    row.addEventListener("click", () => toggleNonFundedDetails(row, invoice));
    table.appendChild(row);
  });
}

function buildInvoiceKey(invoice, index) {
  const invoiceNumber = invoice.invoice_number || invoice.invoice_ref || "unknown";
  const sellerId = invoice.seller_id || "unknown";
  return `${sellerId}|${invoiceNumber}|${index}`;
}

function toggleNonFundedDetails(row, invoice) {
  const table = row.parentElement;
  if (!table) return;

  const currentKey = row.dataset.invoiceKey;
  const nextRow = row.nextElementSibling;
  const isExpanded = nextRow?.dataset?.detailsRow === "true" && nextRow?.dataset?.invoiceKey === currentKey;

  // Collapse any existing detail rows
  const existingDetailRows = table.querySelectorAll("ui5-table-row[data-details-row='true']");
  existingDetailRows.forEach((detailRow) => detailRow.remove());
  table.querySelectorAll("ui5-table-row.expanded").forEach((expandedRow) => expandedRow.classList.remove("expanded"));

  if (isExpanded) {
    row.classList.remove("expanded");
    return;
  }

  const detailRow = buildDetailsRow(invoice, currentKey);
  row.classList.add("expanded");
  row.after(detailRow);
}

function buildDetailsRow(invoice, invoiceKey) {
  const detailRow = document.createElement("ui5-table-row");
  detailRow.dataset.detailsRow = "true";
  detailRow.dataset.invoiceKey = invoiceKey;
  detailRow.classList.add("details-row");
  detailRow.style.setProperty("--row-height", "auto");

  const cell = document.createElement("ui5-table-cell");
  cell.style.gridColumn = "1 / -1";

  const content = document.createElement("div");
  content.classList.add("diagnostic-content");

  const diagnostics = Array.isArray(invoice.rejection_diagnostics)
    ? invoice.rejection_diagnostics
    : [];

  if (diagnostics.length === 0) {
    const empty = document.createElement("div");
    empty.classList.add("diagnostic-empty");
    empty.textContent = "No diagnostics available for this invoice.";
    content.appendChild(empty);
  } else {
    diagnostics.forEach((diag) => {
      const section = document.createElement("div");
      section.classList.add("diagnostic-section");

      const title = document.createElement("div");
      title.classList.add("diagnostic-title");
      const description = diag.description ? `: ${diag.description}` : "";
      title.textContent = `${diag.rule_code}${description}`;
      section.appendChild(title);

      const bullets = Array.isArray(diag.bullets) ? diag.bullets : [];
      if (bullets.length > 0) {
        const list = document.createElement("ul");
        list.classList.add("diagnostic-list");
        bullets.forEach((bullet) => {
          const item = document.createElement("li");
          item.textContent = bullet;
          list.appendChild(item);
        });
        section.appendChild(list);
      } else {
        const fallback = document.createElement("div");
        fallback.classList.add("diagnostic-empty");
        fallback.textContent = "No detailed bullets provided.";
        section.appendChild(fallback);
      }

      content.appendChild(section);
    });
  }

  cell.appendChild(content);
  detailRow.appendChild(cell);

  return detailRow;
}

async function downloadSummary() {
  if (!appState.outputFileName) {
    showError("No output file available. Please analyze a file first.");
    return;
  }

  try {
    hideSuccess();
    hideError();
    const url = `${API_BASE_URL}/api/eligibility/download/${encodeURIComponent(appState.outputFileName)}`;
    const response = await fetch(url, {
      method: "GET",
      headers: {
        "X-API-Key": API_KEY
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Get the file as a blob
    const blob = await response.blob();

    // Create download link
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = appState.outputFileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(downloadUrl);

    console.log("Downloaded:", appState.outputFileName);
  } catch (error) {
    console.error("Failed to download file:", error);
    showError(`Download failed: ${error.message}`);
  }
}

async function fetchSellerStatistics(sellerId) {
  appState.currentSellerId = sellerId;
  appState.historyOffset = 0;
  appState.sellerHistory = [];

  try {
    setSellerLoading(true);
    hideSuccess();
    hideError();

    // Fetch summary and history in parallel
    const [summary, historyResult] = await Promise.all([
      request(`/api/eligibility/seller/${encodeURIComponent(sellerId)}/summary`),
      request(`/api/eligibility/seller/${encodeURIComponent(sellerId)}/history?limit=${appState.historyLimit}&offset=0`)
    ]);

    appState.sellerSummary = summary;
    appState.sellerHistory = historyResult.records || [];
    appState.historyOffset = appState.sellerHistory.length;

    // Display results
    displaySellerSummary(summary);
    displaySellerHistory(appState.sellerHistory, true);
    displayRejectionBreakdown(summary.rejection_breakdown || {});

    // Show seller summary section
    const summarySection = document.getElementById("seller-summary-section");
    if (summarySection) {
      summarySection.style.display = "block";
    }

    console.log("Loaded seller statistics for:", sellerId);
  } catch (error) {
    console.error("Failed to fetch seller statistics:", error);
    showError(`Failed to load seller statistics: ${error.message}`);

    // Hide summary section on error
    const summarySection = document.getElementById("seller-summary-section");
    if (summarySection) {
      summarySection.style.display = "none";
    }
  } finally {
    setSellerLoading(false);
  }
}

async function fetchSellerHistory(sellerId, reset = false) {
  if (reset) {
    appState.historyOffset = 0;
    appState.sellerHistory = [];
  }

  try {
    hideSuccess();
    const historyResult = await request(
      `/api/eligibility/seller/${encodeURIComponent(sellerId)}/history?limit=${appState.historyLimit}&offset=${appState.historyOffset}`
    );

    const newRecords = historyResult.records || [];
    appState.sellerHistory = [...appState.sellerHistory, ...newRecords];
    appState.historyOffset += newRecords.length;

    displaySellerHistory(appState.sellerHistory, reset);

    // Hide "Load More" if no more records
    const loadMoreBtn = document.getElementById("load-more-history-btn");
    if (loadMoreBtn) {
      loadMoreBtn.style.display = newRecords.length < appState.historyLimit ? "none" : "inline-flex";
    }
  } catch (error) {
    console.error("Failed to fetch seller history:", error);
    showError(`Failed to load history: ${error.message}`);
  }
}

function displaySellerSummary(summary) {
  document.getElementById("seller-total-invoices").textContent = summary.total_invoices?.toString() || "0";
  document.getElementById("seller-funded-invoices").textContent = summary.funded_invoices?.toString() || "0";
  document.getElementById("seller-non-funded-invoices").textContent = summary.non_funded_invoices?.toString() || "0";

  const rate = summary.eligibility_rate != null ? `${(summary.eligibility_rate * 100).toFixed(1)}%` : "0%";
  document.getElementById("seller-eligibility-rate").textContent = rate;
}

function displayRejectionBreakdown(breakdown) {
  const table = document.getElementById("rejection-table");
  const panel = document.getElementById("rejection-panel");
  if (!table) return;

  const rejectionDescriptions = {
    R1: "Due date too close to purchase date",
    R2: "Duplicate invoice",
    R11: "Currency not eligible",
    R13: "Invoice is overdue",
    R16: "Tenor exceeds maximum allowed days",
    R17: "Invoice issued too recently",
  };

  // Clear existing rows (except header)
  const existingRows = table.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
  existingRows.forEach((row) => row.remove());

  const entries = Object.entries(breakdown);

  if (panel) {
    panel.setAttribute("header-text", `Non-Eligibility Breakdown (${entries.length} reasons)`);
  }

  if (entries.length === 0) {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell colspan="3">
        <div class="table-empty">No non-eligibility data available</div>
      </ui5-table-cell>
    `;
    table.appendChild(row);
    return;
  }

  // Calculate total for percentages
  const total = entries.reduce((sum, [, count]) => sum + count, 0);

  // Sort by count descending
  entries.sort((a, b) => b[1] - a[1]);

  entries.forEach(([reason, count]) => {
    const percentage = total > 0 ? ((count / total) * 100).toFixed(1) : "0.0";
    const description = rejectionDescriptions[reason];
    const label = description ? `${reason}: ${description}` : reason;
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell>${label}</ui5-table-cell>
      <ui5-table-cell>${count}</ui5-table-cell>
      <ui5-table-cell>${percentage}%</ui5-table-cell>
    `;
    table.appendChild(row);
  });
}

function displaySellerHistory(records, reset = false) {
  const table = document.getElementById("history-table");
  const panel = document.getElementById("history-panel");
  if (!table) return;

  if (reset) {
    // Clear existing rows (except header)
    const existingRows = table.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
    existingRows.forEach((row) => row.remove());
  }

  if (panel) {
    panel.setAttribute("header-text", `Processing History (${records.length} records)`);
  }

  if (records.length === 0 && reset) {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell colspan="5">
        <div class="table-empty">No history records available</div>
      </ui5-table-cell>
    `;
    table.appendChild(row);
    return;
  }

  // If not reset, only add new records
  const startIndex = reset ? 0 : records.length - appState.historyLimit;
  const recordsToAdd = reset ? records : records.slice(Math.max(0, startIndex));

  recordsToAdd.forEach((record) => {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell>${record.invoice_number || "-"}</ui5-table-cell>
      <ui5-table-cell>${record.status || "-"}</ui5-table-cell>
      <ui5-table-cell>${formatAmount(record.amount, record.currency)}</ui5-table-cell>
      <ui5-table-cell>${record.currency || "-"}</ui5-table-cell>
      <ui5-table-cell>${record.processed_date || "-"}</ui5-table-cell>
    `;
    table.appendChild(row);
  });
}

// Utility functions
function formatAmount(amount, currency) {
  if (amount == null) return "-";
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: currency || "USD",
      minimumFractionDigits: 2
    }).format(amount);
  } catch {
    return `${amount} ${currency || ""}`;
  }
}

function getTodayDate() {
  return new Date().toISOString().split("T")[0];
}

function setMainLoading(isLoading) {
  const dataStatus = document.getElementById("data-status");
  const mainLoading = document.getElementById("main-loading");

  if (dataStatus) {
    dataStatus.style.display = isLoading ? "flex" : "none";
  }

  if (mainLoading) {
    if (isLoading) {
      mainLoading.setAttribute("active", "");
    } else {
      mainLoading.removeAttribute("active");
    }
  }
}

function setAnalyzeLoading(isLoading) {
  appState.isAnalyzing = isLoading;

  const analyzeBtn = document.getElementById("analyze-btn");
  const analysisLoading = document.getElementById("analysis-loading");

  if (analyzeBtn) {
    analyzeBtn.disabled = isLoading;
  }

  if (analysisLoading) {
    analysisLoading.style.display = isLoading ? "flex" : "none";
  }
}

function setSellerLoading(isLoading) {
  appState.isLoadingSeller = isLoading;

  const sellerLoading = document.getElementById("seller-loading");
  const loadSellerBtn = document.getElementById("load-seller-btn");

  if (sellerLoading) {
    sellerLoading.style.display = isLoading ? "block" : "none";
  }

  if (loadSellerBtn) {
    loadSellerBtn.disabled = isLoading;
  }
}

function showError(message) {
  const errorStatus = document.getElementById("error-status");
  const dataStatus = document.getElementById("data-status");

  if (errorStatus) {
    errorStatus.textContent = message;
    errorStatus.style.display = "block";
  }

  if (dataStatus) {
    dataStatus.style.display = "flex";
  }

  hideSuccess();
  console.error(message);
}

function hideError() {
  const errorStatus = document.getElementById("error-status");

  if (errorStatus) {
    errorStatus.style.display = "none";
  }
}

function showSuccess(message) {
  const successStatus = document.getElementById("success-status");
  const dataStatus = document.getElementById("data-status");

  if (successStatus) {
    successStatus.textContent = message;
    successStatus.style.display = "block";
  }

  if (dataStatus) {
    dataStatus.style.display = "flex";
  }
}

function hideSuccess() {
  const successStatus = document.getElementById("success-status");
  if (successStatus) {
    successStatus.style.display = "none";
  }
}

async function resetCustomerLogs() {
  const confirmed = window.confirm("This will delete all customer log history. This cannot be undone. Continue?");
  if (!confirmed) return;

  const resetLogsBtn = document.getElementById("reset-logs-btn");

  try {
    setMainLoading(true);
    hideSuccess();
    hideError();
    if (resetLogsBtn) {
      resetLogsBtn.disabled = true;
    }

    const result = await request("/api/eligibility/customer-logs/reset", "POST");
    showSuccess(`Customer logs reset. Deleted ${result.deleted || 0} entries.`);

    appState.currentSellerId = null;
    appState.sellerSummary = null;
    appState.sellerHistory = [];
    appState.historyOffset = 0;

    displaySellerHistory([], true);
    displayRejectionBreakdown({});

    const summarySection = document.getElementById("seller-summary-section");
    if (summarySection) {
      summarySection.style.display = "none";
    }
  } catch (error) {
    console.error("Failed to reset customer logs:", error);
    showError(`Failed to reset customer logs: ${error.message}`);
  } finally {
    if (resetLogsBtn) {
      resetLogsBtn.disabled = false;
    }
    setMainLoading(false);
  }
}

// ---------------------------------------------------------------------------
// Insights Panel
// ---------------------------------------------------------------------------

function getSelectedTimeframe() {
  const segmented = document.getElementById("insights-timeframe");
  if (!segmented) return appState.insightsLookbackDays || 90;

  const items = Array.from(segmented.querySelectorAll("ui5-segmented-button-item"));
  const selectedItem = items.find(
    (item) => item.pressed === true || item.getAttribute("aria-pressed") === "true",
  );
  const lookbackDays = getLookbackDaysFromItem(selectedItem);

  if (lookbackDays > 0) {
    appState.insightsLookbackDays = lookbackDays;
    return lookbackDays;
  }

  return appState.insightsLookbackDays || 90;
}

function getLookbackDaysFromItem(item) {
  const rawValue = item?.dataset?.lookbackDays || item?.getAttribute("data-lookback-days");
  const lookbackDays = parseInt(rawValue || "", 10);
  return Number.isFinite(lookbackDays) && lookbackDays > 0 ? lookbackDays : 0;
}

function buildFilterParams() {
  const sellerId = document.getElementById("insights-seller-filter")?.selectedOption?.value || "";
  const debtorId = document.getElementById("insights-debtor-filter")?.selectedOption?.value || "";
  const programa = document.getElementById("insights-programa-filter")?.selectedOption?.value || "";
  const insurerId = document.getElementById("insights-insurer-filter")?.selectedOption?.value || "";
  const lookbackDays = getSelectedTimeframe();

  const params = new URLSearchParams();
  params.append("lookback_days", lookbackDays.toString());
  if (sellerId) params.append("seller_id", sellerId);
  if (debtorId) params.append("debtor_id", debtorId);
  if (programa) params.append("programa", programa);
  if (insurerId) params.append("insurer_id", insurerId);
  return params;
}

async function loadFilterOptions() {
  try {
    const data = await request("/api/eligibility/patterns/filters?lookback_days=365");

    const populateSelect = (selectId, items, allLabel) => {
      const select = document.getElementById(selectId);
      if (!select) return;
      // Remove all options except the first "All" option
      while (select.children.length > 1) {
        select.removeChild(select.lastChild);
      }
      (items || []).forEach((item) => {
        const option = document.createElement("ui5-option");
        option.setAttribute("value", item.id);
        option.textContent = item.name || item.id;
        select.appendChild(option);
      });
    };

    populateSelect("insights-seller-filter", data.sellers, "All Sellers");
    populateSelect("insights-debtor-filter", data.debtors, "All Debtors");
    populateSelect("insights-programa-filter", data.programas, "All Programs");
    populateSelect("insights-insurer-filter", data.insurers, "All Insurers");
  } catch (error) {
    console.warn("Failed to load filter options:", error);
  }
}

async function fetchInsights() {
  try {
    setInsightsLoading(true);
    hideError();

    const params = buildFilterParams();
    const qs = params.toString();

    const [patterns, profiles, trend] = await Promise.all([
      request(`/api/eligibility/patterns?${qs}`),
      request(`/api/eligibility/patterns/debtor-profiles?${qs}`),
      request(`/api/eligibility/patterns/trend?${qs}`),
    ]);

    displayInsightsSummary(patterns);
    displayPatternAlerts(patterns.patterns || []);
    displayDebtorProfiles(profiles);
    renderTrendChart(trend);
    renderRuleDistChart(trend);

    const resultsDiv = document.getElementById("insights-results");
    if (resultsDiv) {
      resultsDiv.style.display = "block";
    }
  } catch (error) {
    console.error("Failed to fetch insights:", error);
    showError(`Failed to analyze patterns: ${error.message}`);
  } finally {
    setInsightsLoading(false);
  }
}

async function exportPatternPdf() {
  const params = buildFilterParams();
  const url = `${API_BASE_URL}/api/eligibility/patterns/report?${params.toString()}`;

  const exportBtn = document.getElementById("export-pdf-btn");
  if (exportBtn) exportBtn.disabled = true;

  try {
    const response = await fetch(url, { headers: { "X-API-Key": API_KEY } });
    if (!response.ok) {
      throw new Error(`Report generation failed: ${response.statusText}`);
    }
    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = "pattern_analysis_report.pdf";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(downloadUrl);
  } catch (error) {
    console.error("Failed to export PDF:", error);
    showError(`Failed to export PDF: ${error.message}`);
  } finally {
    if (exportBtn) exportBtn.disabled = false;
  }
}

function setInsightsLoading(isLoading) {
  appState.isLoadingInsights = isLoading;

  const loadingDiv = document.getElementById("insights-loading");
  const btn = document.getElementById("load-insights-btn");
  const resultsDiv = document.getElementById("insights-results");

  if (loadingDiv) {
    loadingDiv.style.display = isLoading ? "block" : "none";
  }
  if (btn) {
    btn.disabled = isLoading;
  }
  if (isLoading && resultsDiv) {
    resultsDiv.style.display = "none";
  }
}

function displayInsightsSummary(patterns) {
  const totalEl = document.getElementById("insights-total-patterns");
  const highEl = document.getElementById("insights-high-count");
  const mediumEl = document.getElementById("insights-medium-count");
  const rateEl = document.getElementById("insights-eligibility-rate");

  if (totalEl) totalEl.textContent = (patterns.total_patterns ?? 0).toString();
  if (highEl) highEl.textContent = (patterns.high_severity ?? 0).toString();
  if (mediumEl) mediumEl.textContent = (patterns.medium_severity ?? 0).toString();
  if (rateEl) {
    const rate = patterns.overall_eligibility_rate ?? 0;
    rateEl.textContent = `${(rate * 100).toFixed(1)}%`;
  }
}

function displayPatternAlerts(alerts) {
  const container = document.getElementById("alerts-container");
  if (!container) return;

  container.innerHTML = "";

  if (alerts.length === 0) {
    container.innerHTML = `<div class="table-empty">No patterns detected in the analysis window.</div>`;
    return;
  }

  const panel = document.getElementById("alerts-panel");
  if (panel) {
    panel.setAttribute("header-text", `Pattern Alerts (${alerts.length})`);
  }

  alerts.forEach((alert) => {
    const severity = alert.severity || "low";
    const card = document.createElement("div");
    card.className = `alert-card severity-${severity}`;
    card.innerHTML = `
      <div class="alert-card-header">
        <span class="alert-severity-badge">${severity}</span>
        <span class="alert-title">${escapeHtml(alert.title || "")}</span>
      </div>
      <div class="alert-description">${escapeHtml(alert.description || "")}</div>
      <div class="alert-recommendation">${escapeHtml(alert.recommendation || "")}</div>
    `;
    container.appendChild(card);
  });
}

function displayDebtorProfiles(profiles) {
  const table = document.getElementById("debtor-profiles-table");
  const panel = document.getElementById("debtor-profiles-panel");
  if (!table) return;

  const existingRows = table.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
  existingRows.forEach((row) => row.remove());

  const items = Array.isArray(profiles) ? profiles : [];

  if (panel) {
    panel.setAttribute("header-text", `Debtor Profiles (${items.length})`);
  }

  if (items.length === 0) {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell colspan="8">
        <div class="table-empty">No debtor profiles available</div>
      </ui5-table-cell>
    `;
    table.appendChild(row);
    return;
  }

  items.forEach((profile) => {
    const rate = profile.rejection_rate ?? 0;
    const rateClass = rate > 0.7 ? "rate-high" : rate > 0.4 ? "rate-medium" : "rate-low";
    const rateText = `${(rate * 100).toFixed(1)}%`;
    const amountText = profile.total_amount_rejected != null
      ? parseFloat(profile.total_amount_rejected).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
      : "-";
    const batchText = profile.batches_with_rejection != null
      ? `${profile.batches_with_rejection}/${profile.batch_count}`
      : "-";

    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell>${escapeHtml(profile.debtor_name || profile.debtor_id || "-")}</ui5-table-cell>
      <ui5-table-cell>${escapeHtml(profile.seller_name || profile.seller_id || "-")}</ui5-table-cell>
      <ui5-table-cell>${profile.total_invoices ?? 0}</ui5-table-cell>
      <ui5-table-cell>${profile.rejected_invoices ?? 0}</ui5-table-cell>
      <ui5-table-cell><span class="${rateClass}">${rateText}</span></ui5-table-cell>
      <ui5-table-cell>${escapeHtml(profile.dominant_rule || "-")}</ui5-table-cell>
      <ui5-table-cell>${amountText}</ui5-table-cell>
      <ui5-table-cell>${batchText}</ui5-table-cell>
    `;
    table.appendChild(row);
  });
}

function renderTrendChart(trendData) {
  const ctx = document.getElementById("trend-chart")?.getContext("2d");
  if (!ctx) return;

  if (appState.insightsCharts.trend && typeof appState.insightsCharts.trend.destroy === "function") {
    appState.insightsCharts.trend.destroy();
  }

  const points = Array.isArray(trendData) ? trendData : [];

  if (points.length === 0) {
    delete appState.insightsCharts.trend;
    return;
  }

  const labels = points.map((p) => p.period_start);
  const rejectionRates = points.map((p) => Math.round((p.rejection_rate ?? 0) * 1000) / 10);
  const totalInvoices = points.map((p) => p.total_invoices ?? 0);

  appState.insightsCharts.trend = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Non-Eligibility Rate (%)",
          data: rejectionRates,
          type: "line",
          borderColor: "rgba(220,53,69,0.9)",
          backgroundColor: "rgba(220,53,69,0.1)",
          pointRadius: 3,
          pointHoverRadius: 5,
          tension: 0.2,
          yAxisID: "y1",
          order: 1,
        },
        {
          label: "Total Invoices",
          data: totalInvoices,
          backgroundColor: "rgba(54,162,235,0.5)",
          yAxisID: "y",
          order: 2,
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
      scales: {
        y: {
          position: "left",
          beginAtZero: true,
          title: { display: true, text: "Invoices" },
        },
        y1: {
          position: "right",
          grid: { drawOnChartArea: false },
          min: 0,
          max: 100,
          title: { display: true, text: "Non-Eligibility Rate (%)" },
          ticks: {
            callback: (value) => value + "%",
          },
        },
      },
    },
  });
}

function renderRuleDistChart(trendData) {
  const ctx = document.getElementById("rule-dist-chart")?.getContext("2d");
  if (!ctx) return;

  if (appState.insightsCharts.ruleDist && typeof appState.insightsCharts.ruleDist.destroy === "function") {
    appState.insightsCharts.ruleDist.destroy();
  }

  // Aggregate rule counts from trend data points (each point has rejection_by_rule)
  const ruleCounts = {};
  const points = Array.isArray(trendData) ? trendData : [];
  points.forEach((p) => {
    if (p.rejection_by_rule) {
      for (const [rule, count] of Object.entries(p.rejection_by_rule)) {
        ruleCounts[rule] = (ruleCounts[rule] || 0) + count;
      }
    }
  });

  const totalRejections = Object.values(ruleCounts).reduce((sum, c) => sum + c, 0);
  const ruleEntries = Object.entries(ruleCounts).sort((a, b) => b[1] - a[1]);

  if (ruleEntries.length === 0 || totalRejections === 0) {
    delete appState.insightsCharts.ruleDist;
    return;
  }

  const colors = [
    "rgba(220,53,69,0.7)",
    "rgba(255,159,64,0.7)",
    "rgba(255,205,86,0.7)",
    "rgba(54,162,235,0.7)",
    "rgba(23,162,184,0.7)",
    "rgba(40,167,69,0.7)",
    "rgba(108,117,125,0.7)",
  ];

  const percentages = ruleEntries.map(([, count]) =>
    Math.round((count / totalRejections) * 1000) / 10
  );

  appState.insightsCharts.ruleDist = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ruleEntries.map(([rule]) => rule),
      datasets: [
        {
          label: "Share of Non-Eligible Invoices (%)",
          data: percentages,
          backgroundColor: ruleEntries.map((_, i) => colors[i % colors.length]),
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: {
          beginAtZero: true,
          max: 100,
          title: { display: true, text: "% of Non-Eligible Invoices" },
          ticks: {
            callback: (value) => value + "%",
          },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: (context) => {
              const idx = context.dataIndex;
              const [, count] = ruleEntries[idx];
              return `${context.parsed.x}% (${count} non-eligible)`;
            },
          },
        },
      },
    },
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
