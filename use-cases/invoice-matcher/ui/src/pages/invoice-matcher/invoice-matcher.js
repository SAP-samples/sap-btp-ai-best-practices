import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/CheckBox.js";
import "@ui5/webcomponents/dist/SegmentedButton.js";
import "@ui5/webcomponents/dist/SegmentedButtonItem.js";
import "@ui5/webcomponents/dist/Table.js";
import "@ui5/webcomponents/dist/TableHeaderRow.js";
import "@ui5/webcomponents/dist/TableHeaderCell.js";
import "@ui5/webcomponents/dist/TableRow.js";
import "@ui5/webcomponents/dist/TableCell.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";

import { API_BASE_URL } from "../../services/api.js";

let invoiceFileContent = null;
let paymentFileContent = null;
let allRows = [];
let worker = null;
let aiJobId = null;
let aiPollTimer = null;

const CONFIG = {
  invoiceNumberCol: "Invoice#",
  invoiceAmountCol: "Invoice Amt",
  paymentAmountCol: "TRANSACTION_AMT",
  paymentRefCol: "BANK_REF",
  paymentTextCols: ["BY_ORD_OF_NAME", "BY_ORD_OF_ADDR", "REMIT_NAME"],
  tolerance: 0.01,
};

function $(id) {
  return document.getElementById(id);
}

function setStatus(msg) {
  const el = $("statusMessage");
  if (el) el.textContent = msg;
}

function setBusy(busy) {
  const ind = $("busyIndicator");
  if (ind) {
    if (busy) ind.classList.remove("hidden");
    else ind.classList.add("hidden");
  }
}

function updateStats(stats) {
  $("statTotal").textContent = stats.total;
  $("statMatched").textContent = stats.matched;
  $("statAiMatched").textContent = stats.aiMatched || 0;
  $("statUnmatched").textContent = stats.unmatched;
  $("statsBar").classList.remove("hidden");
}

function formatAmount(v) {
  if (v === "" || v === null || v === undefined) return "—";
  const n = parseFloat(v);
  if (isNaN(n)) return String(v);
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function renderTable(rows) {
  const table = $("resultsTable");
  if (!table) return;

  table.querySelectorAll("ui5-table-row").forEach((r) => r.remove());

  for (const row of rows) {
    const tr = document.createElement("ui5-table-row");

    let rowClass = "row-unmatched";
    if (row.matchStatus === "matched") rowClass = "row-matched";
    else if (row.matchStatus === "ai_matched") rowClass = "row-ai-matched";
    tr.classList.add(rowClass);

    const statusText = row.matchStatus === "matched" ? "Matched" : row.matchStatus === "ai_matched" ? "AI Matched" : "Unmatched";
    const statusClass = row.matchStatus === "matched" ? "badge-success" : row.matchStatus === "ai_matched" ? "badge-warning" : "badge-error";

    const confText = row.matchStatus === "ai_matched" ? (row.confidence || "") : "";
    const confClass = row.confidence === "HIGH" ? "badge-success" : row.confidence === "MEDIUM" ? "badge-warning" : "badge-error";

    const reasonText = row.matchStatus === "ai_matched" ? (row.matchReason || "") : "";

    tr.innerHTML = `
      <ui5-table-cell><span>${row.invoiceNumber || ""}</span></ui5-table-cell>
      <ui5-table-cell><span>${formatAmount(row.invoiceAmount)}</span></ui5-table-cell>
      <ui5-table-cell><span>${formatAmount(row.paymentAmount)}</span></ui5-table-cell>
      <ui5-table-cell><span>${row.bankRef || ""}</span></ui5-table-cell>
      <ui5-table-cell><span class="${statusClass}">${statusText}</span></ui5-table-cell>
      <ui5-table-cell><span class="${confClass}">${confText}</span></ui5-table-cell>
      <ui5-table-cell><span>${reasonText}</span></ui5-table-cell>
    `;
    table.appendChild(tr);
  }

  table.classList.remove("hidden");
  $("searchRow").classList.remove("hidden");
}

function applyFilter(filterKey) {
  let filtered;
  if (filterKey === "Matched") filtered = allRows.filter((r) => r.matchStatus === "matched");
  else if (filterKey === "AI Matched") filtered = allRows.filter((r) => r.matchStatus === "ai_matched");
  else if (filterKey === "Unmatched") filtered = allRows.filter((r) => r.matchStatus === "unmatched");
  else filtered = allRows;

  const query = ($("tableSearch")?.value || "").trim().toUpperCase();
  if (query) filtered = filtered.filter((r) => (r.invoiceNumber || "").toUpperCase().includes(query));

  renderTable(filtered);
}

function readFileAsCSV(file, isLatin1) {
  return new Promise((resolve, reject) => {
    const isExcel = /\.xlsx?$/i.test(file.name);
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed to read file"));

    if (isExcel) {
      reader.onload = (e) => {
        try {
          const workbook = window.XLSX.read(e.target.result, { type: "array" });
          const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
          resolve(window.XLSX.utils.sheet_to_csv(firstSheet));
        } catch (err) {
          reject(err);
        }
      };
      reader.readAsArrayBuffer(file);
    } else {
      reader.onload = (e) => resolve(e.target.result);
      reader.readAsText(file, isLatin1 ? "windows-1252" : "utf-8");
    }
  });
}

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL("../../worker.js", import.meta.url), { type: "classic" });
  return worker;
}

function fireAIMatch() {
  const unmatchedInvoices = allRows.filter((r) => r.matchStatus === "unmatched").map((r) => r.invoiceNumber);

  if (unmatchedInvoices.length === 0) {
    setBusy(false);
    setStatus("All invoices matched by rules — no AI needed.");
    return;
  }

  setStatus(`AI matching (${unmatchedInvoices.length} invoices)...`);

  const tolerance = parseFloat($("toleranceInput")?.value) || 0.01;

  fetch(`${API_BASE_URL}/api/ai-match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      invoiceCSV: invoiceFileContent,
      paymentCSV: paymentFileContent,
      ruleMatchedInvoices: unmatchedInvoices,
      tolerance,
      colConfig: {
        invoiceNumber: CONFIG.invoiceNumberCol,
        invoiceAmount: CONFIG.invoiceAmountCol,
        textCols: CONFIG.paymentTextCols,
      },
    }),
  })
    .then((r) => {
      if (!r.ok) return r.json().then((d) => { throw new Error(d.detail || "Request failed"); });
      return r.json();
    })
    .then((data) => {
      aiJobId = data.jobId;
      startAIPoll();
    })
    .catch((err) => {
      setBusy(false);
      setStatus(`AI Match error: ${err.message}`);
    });
}

function startAIPoll() {
  if (aiPollTimer) clearInterval(aiPollTimer);

  aiPollTimer = setInterval(() => {
    fetch(`${API_BASE_URL}/api/ai-match/status/${aiJobId}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.progress) {
          let msg = data.progress.message || "Processing...";
          if (data.progress.totalPayers && data.progress.completedPayers) {
            msg += ` (${data.progress.completedPayers}/${data.progress.totalPayers} payers)`;
          }
          setStatus(msg);
        }
        if (data.status === "complete") {
          stopAIPoll();
          mergeAIResults(data.results);
        } else if (data.status === "error") {
          stopAIPoll();
          setBusy(false);
          setStatus(`AI Match error: ${data.error}`);
        }
      })
      .catch(() => {
        stopAIPoll();
        setBusy(false);
        setStatus("Lost connection to AI matching service.");
      });
  }, 3000);
}

function stopAIPoll() {
  if (aiPollTimer) {
    clearInterval(aiPollTimer);
    aiPollTimer = null;
  }
}

function mergeAIResults(aiResults) {
  const aiMap = {};
  for (const r of aiResults) aiMap[r.invoiceNumber] = r;

  let aiMatchedCount = 0;
  for (const row of allRows) {
    if (row.matchStatus === "unmatched" && aiMap[row.invoiceNumber]) {
      const ai = aiMap[row.invoiceNumber];
      row.matchStatus = "ai_matched";
      row.matched = true;
      row.paymentAmount = ai.paymentAmount;
      row.bankRef = ai.bankRef;
      row.confidence = ai.confidence;
      row.matchReason = ai.matchReason;
      row.branchProximity = ai.branchProximity;
      aiMatchedCount++;
    }
  }

  const total = allRows.length;
  const matched = allRows.filter((r) => r.matchStatus === "matched").length;
  const unmatched = allRows.filter((r) => r.matchStatus === "unmatched").length;

  updateStats({ total, matched, aiMatched: aiMatchedCount, unmatched });
  applyFilter("All");
  setBusy(false);
  setStatus(`${matched} rule-matched · ${aiMatchedCount} AI-matched · ${unmatched} unmatched · ${total} total`);
}

function onMatchPress() {
  if (!invoiceFileContent) {
    setStatus("Please upload an invoices file first.");
    return;
  }
  if (!paymentFileContent) {
    setStatus("Please upload a payments file first.");
    return;
  }

  const tolerance = parseFloat($("toleranceInput")?.value);
  if (isNaN(tolerance) || tolerance < 0) {
    setStatus("Please enter a valid non-negative tolerance value.");
    return;
  }

  CONFIG.tolerance = tolerance;
  setBusy(true);
  allRows = [];
  $("resultsTable")?.classList.add("hidden");
  $("statsBar")?.classList.add("hidden");

  const demoMode = $("demoModeCheckbox")?.checked;
  let invoiceCSV = invoiceFileContent;

  if (demoMode) {
    const lines = invoiceCSV.split("\n");
    const header = lines[0];
    let dataLines = lines.slice(1).filter((l) => l.trim().length > 0);
    if (dataLines.length > 100) {
      for (let i = dataLines.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [dataLines[i], dataLines[j]] = [dataLines[j], dataLines[i]];
      }
      dataLines = dataLines.slice(0, 100);
    }
    invoiceCSV = header + "\n" + dataLines.join("\n");
  }

  invoiceFileContent = invoiceCSV;

  setStatus(demoMode ? "Demo mode: running rule-based on 100 invoices..." : "Running rule-based matching...");

  const w = getWorker();
  w.onmessage = (e) => {
    const msg = e.data;
    switch (msg.type) {
      case "STATUS":
        setStatus(msg.text);
        break;
      case "PROGRESS":
        setStatus(`Matching... ${msg.percent}%`);
        break;
      case "RESULT":
        allRows = msg.rows;
        const stats = { ...msg.stats, aiMatched: 0 };
        updateStats(stats);
        applyFilter("All");
        setStatus(`Rule-based: ${stats.matched} matched. Starting AI on ${stats.unmatched} remaining...`);
        fireAIMatch();
        break;
      case "ERROR":
        setBusy(false);
        setStatus(`Error: ${msg.message}`);
        break;
    }
  };
  w.onerror = (e) => {
    setBusy(false);
    setStatus(`Worker error: ${e.message}`);
  };
  w.postMessage({
    type: "MATCH",
    invoiceCSV,
    paymentCSV: paymentFileContent,
    config: {
      invoiceNumberCol: CONFIG.invoiceNumberCol,
      invoiceAmountCol: CONFIG.invoiceAmountCol,
      paymentAmountCol: CONFIG.paymentAmountCol,
      paymentRefCol: CONFIG.paymentRefCol,
      paymentTextCols: CONFIG.paymentTextCols,
      tolerance,
    },
  });
}

export function init() {
  const invoiceInput = $("invoiceFile");
  const paymentInput = $("paymentFile");
  const matchBtn = $("matchButton");
  const filterToggle = $("filterToggle");
  const searchInput = $("tableSearch");

  if (invoiceInput) {
    invoiceInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        invoiceFileContent = await readFileAsCSV(file, false);
        $("invoiceFileName").textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      } catch (err) {
        setStatus(`Failed to read invoice file: ${err.message}`);
      }
    });
  }

  if (paymentInput) {
    paymentInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        paymentFileContent = await readFileAsCSV(file, true);
        $("paymentFileName").textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      } catch (err) {
        setStatus(`Failed to read payment file: ${err.message}`);
      }
    });
  }

  if (matchBtn) {
    matchBtn.addEventListener("click", onMatchPress);
  }

  if (filterToggle) {
    filterToggle.addEventListener("selection-change", (e) => {
      const selected = e.detail.selectedItems?.[0]?.textContent || "All";
      applyFilter(selected);
    });
  }

  if (searchInput) {
    searchInput.addEventListener("input", () => {
      const filterBtn = filterToggle?.querySelector("[pressed]");
      const filterKey = filterBtn?.textContent || "All";
      applyFilter(filterKey);
    });
  }
}

export default init;
