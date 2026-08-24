/**
 * docai-new-app.ts
 * ----------------
 * DOC AI NEW — Main pipeline UI component.
 *
 * Features:
 * - Single and multiple PDF upload
 * - PDF type detection display
 * - Free Prompt extraction results
 * - Customer name detection
 * - Template status (found / created)
 * - Annotations summary
 * - Line items table
 */

import { processInvoicesNew } from "../api/docai-new-client";
import type { DocAiNewResult, ExtractionResult } from "../api/docai-new-types";

// ─── Helpers ────────────────────────────────────────────────────────────────

function fmt(val: unknown): string {
  if (val === null || val === undefined) return '<span class="na">—</span>';
  return String(val);
}

function fmtMoney(val: number | null | undefined): string {
  if (val === null || val === undefined) return '<span class="na">—</span>';
  return val.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtConf(val: number | null | undefined): string {
  if (val === null || val === undefined) return "";
  const pct = Math.round(val * 100);
  const cls = pct >= 80 ? "conf-high" : pct >= 50 ? "conf-mid" : "conf-low";
  return `<span class="${cls}">${pct}%</span>`;
}

function routeBadge(route: string | null): string {
  const map: Record<string, string> = {
    existing_template: "badge-green",
    template_created: "badge-blue",
    free_prompt: "badge-yellow",
    free_prompt_only: "badge-yellow",
    error: "badge-red",
  };
  const cls = map[route ?? ""] ?? "badge-gray";
  const label = route?.replace(/_/g, " ").toUpperCase() ?? "UNKNOWN";
  return `<span class="badge ${cls}">${label}</span>`;
}

function pdfTypeBadge(type: string | null): string {
  if (type === "scanned") return '<span class="badge badge-yellow">SCANNED</span>';
  if (type === "searchable") return '<span class="badge badge-green">SEARCHABLE</span>';
  return '<span class="badge badge-gray">UNKNOWN</span>';
}

// ─── Extraction fields table ─────────────────────────────────────────────────

function renderExtractionTable(ext: ExtractionResult): string {
  const conf = ext.confidence || {};
  const fields: Array<[string, string]> = [
    ["Customer Name", fmt(ext.customer_name)],
    ["Customer Address", fmt(ext.customer_address)],
    ["Tax ID", fmt(ext.customer_tax_id)],
    ["Invoice Number", fmt(ext.invoice_number)],
    ["Invoice Date", fmt(ext.invoice_date)],
    ["Due Date", fmt(ext.due_date)],
    ["Subtotal", fmtMoney(ext.subtotal)],
    ["Tax Amount", fmtMoney(ext.tax_amount)],
    ["Total Amount", fmtMoney(ext.total_amount)],
    ["Currency", fmt(ext.currency)],
  ];

  const rows = fields
    .map(([label, value]) => {
      const key = label.toLowerCase().replace(/ /g, "_");
      const confVal = conf[key] ?? conf[label.toLowerCase()] ?? null;
      return `<tr>
        <td class="field-label">${label}</td>
        <td>${value}</td>
        <td>${fmtConf(confVal)}</td>
      </tr>`;
    })
    .join("");

  return `
    <table class="result-table">
      <thead>
        <tr><th>Field</th><th>Value</th><th>Confidence</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ─── Line items table ────────────────────────────────────────────────────────

function renderLineItems(ext: ExtractionResult): string {
  const items = ext.line_items || [];
  if (items.length === 0) return "<p class='muted'>No line items extracted.</p>";

  const rows = items
    .map(
      (item, i) => `<tr>
        <td>${i + 1}</td>
        <td>${fmt(item.description)}</td>
        <td>${fmt(item.quantity)}</td>
        <td>${fmtMoney(item.unit_price)}</td>
        <td>${fmtMoney(item.line_total)}</td>
      </tr>`
    )
    .join("");

  return `
    <table class="result-table">
      <thead>
        <tr><th>#</th><th>Description</th><th>Qty</th><th>Unit Price</th><th>Total</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ─── SAP DocAI result (template route) ──────────────────────────────────────

function renderSapResult(sapExt: { headerFields?: unknown[]; lineItems?: unknown[] }): string {
  const hf = (sapExt.headerFields ?? []) as Array<{ name?: string; value?: unknown; rawValue?: string; confidence?: number }>;
  const rawLi = sapExt.lineItems ?? [];

  // SAP returns each line item as [{name, value, ...}, ...] — normalize to flat object
  const li = rawLi.map((row) => {
    if (Array.isArray(row)) {
      const item: Record<string, unknown> = {};
      for (const field of row as Array<{ name?: string; value?: unknown; rawValue?: unknown }>) {
        if (field.name) item[field.name] = field.value ?? field.rawValue ?? null;
      }
      return item;
    }
    return row as Record<string, unknown>;
  });

  const hfRows = hf
    .map((f) => {
      const val = String(f.value ?? f.rawValue ?? "—");
      const pct = f.confidence != null ? Math.round(f.confidence * 100) : null;
      const cls = pct != null ? (pct >= 80 ? "conf-high" : pct >= 50 ? "conf-mid" : "conf-low") : "";
      return `<tr>
        <td class="field-label">${fmt(f.name)}</td>
        <td>${fmt(val)}</td>
        <td>${pct != null ? `<span class="${cls}">${pct}%</span>` : ""}</td>
      </tr>`;
    })
    .join("");

  const liRows = li
    .map(
      (item, i) => `<tr>
        <td>${i + 1}</td>
        <td>${fmt(item["description"])}</td>
        <td>${fmt(item["quantity"])}</td>
        <td>${fmt(item["unitPrice"])}</td>
        <td>${fmt(item["netAmount"])}</td>
      </tr>`
    )
    .join("");

  return `
    <div class="result-section">
      <h4>Extracted Fields (SAP DocAI + Template) — ${hf.length} fields</h4>
      <table class="result-table">
        <thead><tr><th>Field</th><th>Value</th><th>Confidence</th></tr></thead>
        <tbody>${hfRows}</tbody>
      </table>
    </div>
    ${li.length > 0 ? `
    <div class="result-section">
      <h4>Line Items (${li.length})</h4>
      <table class="result-table">
        <thead><tr><th>#</th><th>Description</th><th>Qty</th><th>Unit Price</th><th>Net Amount</th></tr></thead>
        <tbody>${liRows}</tbody>
      </table>
    </div>` : ""}`;
}

// ─── Single result card ──────────────────────────────────────────────────────

function renderResultCard(result: DocAiNewResult, index: number): string {
  const ext = result.extraction;
  const errors = Array.isArray(result.errors) ? result.errors : [];
  const errorsHtml =
    errors.length > 0
      ? `<div class="error-box"><strong>Errors:</strong><ul>${errors.map((e) => `<li>${e}</li>`).join("")}</ul></div>`
      : "";

  // Template info: prefer template_name from pipeline result directly
  const templateName = result.template_name || (result.template as Record<string, unknown> | null)?.name as string | null;
  const templateId = result.template_id || (result.template as Record<string, unknown> | null)?.id as string | null;
  const templateHtml = templateName
    ? `<p><strong>Template:</strong> ${fmt(templateName)} <span class="muted">(id: ${fmt(templateId)})</span>${result.template_created ? ' <span class="badge badge-blue">NEW</span>' : ""}</p>`
    : `<p class="muted">No template associated.</p>`;

  const annotations = Array.isArray(result.annotations) ? result.annotations : [];
  const annotationsHtml =
    annotations.length > 0
      ? `<p><strong>Annotations generated:</strong> ${annotations.length}</p>`
      : "";

  // When route=existing_template, show SAP DocAI result instead of LLM extraction
  const isTemplateRoute = result.route === "existing_template";
  const sapRaw = result.sap_result as { extraction?: { headerFields?: unknown[]; lineItems?: unknown[] }; document?: { headerFields?: unknown[]; lineItems?: unknown[] } } | null;
  const sapExt = sapRaw?.extraction ?? sapRaw?.document ?? null;

  const mainContentHtml = isTemplateRoute && sapExt
    ? renderSapResult(sapExt)
    : ext
      ? `
        <div class="result-section">
          <h4>Extracted Fields (LLM)</h4>
          ${renderExtractionTable(ext)}
        </div>
        <div class="result-section">
          <h4>Line Items</h4>
          ${renderLineItems(ext)}
        </div>`
      : '<p class="muted">No extraction data available.</p>';

  return `
    <div class="result-card" id="result-${index}">
      <div class="result-card-header">
        <span class="filename">📄 ${result.filename}</span>
        ${pdfTypeBadge(result.pdf_type)}
        ${routeBadge(result.route)}
      </div>

      <div class="result-section">
        <h4>Customer</h4>
        <p><strong>${fmt(result.customer_name)}</strong></p>
        ${templateHtml}
        ${annotationsHtml}
      </div>

      ${errorsHtml}
      ${mainContentHtml}

      <div class="result-section" style="display:flex;gap:.75rem;align-items:center;flex-wrap:wrap;margin-top:1rem;">
        <button class="btn btn-primary btn-post-fi" id="btn-post-fi-${index}" data-idx="${index}">POST S4</button>
        <span class="fi-post-status" id="fi-post-status-${index}" style="font-size:.85rem;"></span>
      </div>
    </div>`;
}

// ─── Main render ─────────────────────────────────────────────────────────────

export function renderDocAiNewApp(container: HTMLElement): void {
  container.innerHTML = `
    <div class="docai-new-app">
      <div class="page-header">
        <h2>DOC AI NEW</h2>
        <p class="subtitle">Free Prompt extraction pipeline with automatic template management</p>
      </div>

      <div class="upload-section card">
        <h3>Upload Invoice PDF(s)</h3>
        <div class="upload-area" id="new-upload-area">
          <input type="file" id="new-file-input" accept=".pdf" multiple hidden />
          <div class="upload-placeholder" id="new-upload-placeholder">
            <span class="upload-icon">📂</span>
            <p>Drop PDF files here or <button class="link-btn" id="new-browse-btn">browse</button></p>
            <p class="muted">Supports single and multiple PDFs · Searchable and scanned</p>
          </div>
          <div id="new-file-list" class="file-list hidden"></div>
        </div>

        <div class="upload-options">
          <label class="checkbox-label">
            <input type="checkbox" id="new-auto-template" checked />
            Auto-create template if not found
          </label>
        </div>

        <button class="btn btn-primary" id="new-process-btn" disabled>
          ▶ Process Invoice(s)
        </button>
      </div>

      <div id="new-status" class="status-bar hidden"></div>
      <div id="new-results" class="results-container"></div>
    </div>`;

  // ── Wire up events ──────────────────────────────────────────────────────
  const fileInput = container.querySelector<HTMLInputElement>("#new-file-input")!;
  const browseBtn = container.querySelector<HTMLButtonElement>("#new-browse-btn")!;
  const uploadArea = container.querySelector<HTMLDivElement>("#new-upload-area")!;
  const fileList = container.querySelector<HTMLDivElement>("#new-file-list")!;
  const placeholder = container.querySelector<HTMLDivElement>("#new-upload-placeholder")!;
  const processBtn = container.querySelector<HTMLButtonElement>("#new-process-btn")!;
  const autoTemplateChk = container.querySelector<HTMLInputElement>("#new-auto-template")!;
  const statusBar = container.querySelector<HTMLDivElement>("#new-status")!;
  const resultsDiv = container.querySelector<HTMLDivElement>("#new-results")!;

  let selectedFiles: File[] = [];

  function updateFileList(): void {
    if (selectedFiles.length === 0) {
      fileList.classList.add("hidden");
      placeholder.classList.remove("hidden");
      processBtn.disabled = true;
      return;
    }
    placeholder.classList.add("hidden");
    fileList.classList.remove("hidden");
    fileList.innerHTML = selectedFiles
      .map(
        (f, i) =>
          `<div class="file-item">
            <span>📄 ${f.name}</span>
            <span class="muted">${(f.size / 1024).toFixed(1)} KB</span>
            <button class="remove-file-btn link-btn" data-idx="${i}">✕</button>
          </div>`
      )
      .join("");
    processBtn.disabled = false;

    fileList.querySelectorAll<HTMLButtonElement>(".remove-file-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = parseInt(btn.dataset.idx ?? "0", 10);
        selectedFiles.splice(idx, 1);
        updateFileList();
      });
    });
  }

  browseBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    if (fileInput.files) {
      const newFiles = Array.from(fileInput.files).filter((f) => f.name.endsWith(".pdf"));
      selectedFiles = [...selectedFiles, ...newFiles];
      updateFileList();
    }
    fileInput.value = "";
  });

  // Drag & drop
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
  });
  uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    const dropped = Array.from(e.dataTransfer?.files ?? []).filter((f) =>
      f.name.endsWith(".pdf")
    );
    selectedFiles = [...selectedFiles, ...dropped];
    updateFileList();
  });

  // Process
  processBtn.addEventListener("click", async () => {
    if (selectedFiles.length === 0) return;

    processBtn.disabled = true;
    statusBar.className = "status-bar status-loading";
    statusBar.textContent = `Processing ${selectedFiles.length} file(s)…`;
    resultsDiv.innerHTML = "";

    try {
      const response = await processInvoicesNew(
        selectedFiles,
        "default",
        autoTemplateChk.checked
      );

      statusBar.className = "status-bar status-success";
      statusBar.textContent = `✓ Processed ${response.total} file(s) successfully.`;

      resultsDiv.innerHTML = response.results
        .map((r, i) => renderResultCard(r, i))
        .join("");

      // Wire up POST in FI buttons for each result card
      resultsDiv.querySelectorAll<HTMLButtonElement>(".btn-post-fi").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const idx = parseInt(btn.dataset.idx ?? "0", 10);
          const result = response.results[idx];
          const statusSpan = resultsDiv.querySelector<HTMLSpanElement>(`#fi-post-status-${idx}`);
          btn.disabled = true;
          if (statusSpan) { statusSpan.style.color = ""; statusSpan.textContent = "Posting…"; }

          const ext = result.extraction;
          const payload = {
            supplier_name: result.customer_name ?? "",
            invoice_number: ext?.invoice_number ?? "",
            invoice_date: ext?.invoice_date ?? "",
            total_amount: ext?.total_amount ?? 0,
            currency: ext?.currency ?? "USD",
          };

          try {
            const res = await fetch("/api/v1/fi/post-invoice", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });
            const data = await res.json() as { success: boolean; fi_document?: string; error?: string };
            if (data.success) {
              if (statusSpan) { statusSpan.style.color = "#107e3e"; statusSpan.textContent = `FI Doc: ${data.fi_document ?? ""}`; }
            } else {
              if (statusSpan) { statusSpan.style.color = "#bb0000"; statusSpan.textContent = `Error: ${data.error ?? "Unknown error"}`; }
              btn.disabled = false;
            }
          } catch (err) {
            if (statusSpan) { statusSpan.style.color = "#bb0000"; statusSpan.textContent = `Error: ${(err as Error).message}`; }
            btn.disabled = false;
          }
        });
      });
    } catch (err) {
      statusBar.className = "status-bar status-error";
      statusBar.textContent = `✗ Error: ${(err as Error).message}`;
    } finally {
      processBtn.disabled = false;
    }
  });
}