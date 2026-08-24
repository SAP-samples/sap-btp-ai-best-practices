/**
 * train-template-app.ts
 * ----------------------
 * DOC AI NEW — Train Template UI component.
 */

import { listTemplatesNew, trainTemplate } from "../api/docai-new-client";
import type { DocAiNewTemplate, TrainingResult } from "../api/docai-new-types";

function fmt(val: unknown): string {
  if (val === null || val === undefined) return "—";
  return String(val);
}

function statusBadge(status: string): string {
  const map: Record<string, string> = {
    triggered: "badge-green",
    skipped: "badge-yellow",
    failed: "badge-red",
  };
  return `<span class="badge ${map[status] ?? "badge-gray"}">${status.toUpperCase()}</span>`;
}

function renderTrainingResult(result: TrainingResult): string {
  const errors = Array.isArray(result.errors) ? result.errors : [];
  const errorsHtml =
    errors.length > 0
      ? `<div class="error-box"><strong>Errors:</strong><ul>${errors.map((e) => `<li>${e}</li>`).join("")}</ul></div>`
      : "";

  const extractionResults = Array.isArray(result.extraction_results) ? result.extraction_results : [];
  const extractionRows = extractionResults
    .map((er) => {
      const ext = er.extraction;
      return (
        `<tr>` +
        `<td>📄 ${er.filename}</td>` +
        `<td>${fmt(ext?.customer_name)}</td>` +
        `<td>${fmt(ext?.invoice_number)}</td>` +
        `<td>${fmt(ext?.invoice_date)}</td>` +
        `<td>${fmt(ext?.total_amount)}</td>` +
        `</tr>`
      );
    })
    .join("");

  const extractionTable =
    extractionResults.length > 0
      ? `<table class="result-table"><thead><tr><th>File</th><th>Customer</th><th>Invoice #</th><th>Date</th><th>Total</th></tr></thead><tbody>${extractionRows}</tbody></table>`
      : "<p class='muted'>No extraction data.</p>";

  const successClass = result.success ? "text-success" : "text-error";
  const successLabel = result.success ? "✓ SUCCESS" : "✗ FAILED";

  return (
    `<div class="training-result card">` +
    `<h3>Training Result</h3>` +
    `<div class="training-summary">` +
    `<div class="summary-item"><span class="summary-label">Template ID</span><span class="summary-value">${fmt(result.template_id)}</span></div>` +
    `<div class="summary-item"><span class="summary-label">Documents Processed</span><span class="summary-value">${result.documents_processed}</span></div>` +
    `<div class="summary-item"><span class="summary-label">Fields Annotated</span><span class="summary-value">${result.fields_annotated}</span></div>` +
    `<div class="summary-item"><span class="summary-label">Training Status</span><span class="summary-value">${statusBadge(result.training_status)}</span></div>` +
    `<div class="summary-item"><span class="summary-label">Result</span><span class="summary-value ${successClass}">${successLabel}</span></div>` +
    `</div>` +
    errorsHtml +
    `<div class="result-section"><h4>Extracted Values per Document</h4>${extractionTable}</div>` +
    `</div>`
  );
}

export function renderTrainTemplateApp(container: HTMLElement): void {
  container.innerHTML =
    `<div class="train-template-app">` +
    `<div class="page-header"><h2>Train Template</h2><p class="subtitle">Select a template and upload PDFs to train it with Free Prompt extraction</p></div>` +

    `<div class="card" id="tt-step-1">` +
    `<h3>Step 1 — Select Template</h3>` +
    `<div class="template-select-row">` +
    `<select id="tt-template-select" class="select-input" disabled><option value="">Loading templates…</option></select>` +
    `<button class="btn btn-secondary" id="tt-refresh-btn">↻ Refresh</button>` +
    `</div>` +
    `<div id="tt-template-info" class="template-info hidden"></div>` +
    `</div>` +

    `<div class="card" id="tt-step-2">` +
    `<h3>Step 2 — Upload Training PDF(s)</h3>` +
    `<div class="upload-area" id="tt-upload-area">` +
    `<input type="file" id="tt-file-input" accept=".pdf" multiple hidden />` +
    `<div class="upload-placeholder" id="tt-upload-placeholder">` +
    `<span class="upload-icon">📂</span>` +
    `<p>Drop PDF files here or <button class="link-btn" id="tt-browse-btn">browse</button></p>` +
    `<p class="muted">Supports 1 or multiple PDFs</p>` +
    `</div>` +
    `<div id="tt-file-list" class="file-list hidden"></div>` +
    `</div>` +
    `</div>` +

    `<div class="card">` +
    `<button class="btn btn-primary btn-large" id="tt-train-btn" disabled>🎓 Submit Training</button>` +
    `</div>` +

    `<div id="tt-status" class="status-bar hidden"></div>` +
    `<div id="tt-result" class="results-container"></div>` +
    `</div>`;

  const templateSelect = container.querySelector<HTMLSelectElement>("#tt-template-select")!;
  const refreshBtn = container.querySelector<HTMLButtonElement>("#tt-refresh-btn")!;
  const templateInfo = container.querySelector<HTMLDivElement>("#tt-template-info")!;
  const fileInput = container.querySelector<HTMLInputElement>("#tt-file-input")!;
  const browseBtn = container.querySelector<HTMLButtonElement>("#tt-browse-btn")!;
  const uploadArea = container.querySelector<HTMLDivElement>("#tt-upload-area")!;
  const fileList = container.querySelector<HTMLDivElement>("#tt-file-list")!;
  const placeholder = container.querySelector<HTMLDivElement>("#tt-upload-placeholder")!;
  const trainBtn = container.querySelector<HTMLButtonElement>("#tt-train-btn")!;
  const statusBar = container.querySelector<HTMLDivElement>("#tt-status")!;
  const resultDiv = container.querySelector<HTMLDivElement>("#tt-result")!;

  let selectedFiles: File[] = [];
  let templates: DocAiNewTemplate[] = [];

  function updateTrainBtn(): void {
    trainBtn.disabled = !templateSelect.value || selectedFiles.length === 0;
  }

  async function loadTemplates(): Promise<void> {
    templateSelect.disabled = true;
    templateSelect.innerHTML = '<option value="">Loading…</option>';
    try {
      const resp = await listTemplatesNew();
      templates = resp.templates;
      if (templates.length === 0) {
        templateSelect.innerHTML = '<option value="">No templates available</option>';
      } else {
        templateSelect.innerHTML =
          '<option value="">— Select a template —</option>' +
          templates.map((t) => `<option value="${t.id}">${t.name} (${t.id})</option>`).join("");
        templateSelect.disabled = false;
      }
    } catch (err) {
      templateSelect.innerHTML = '<option value="">Error loading templates</option>';
      console.error("Failed to load templates:", err);
    }
  }

  loadTemplates();
  refreshBtn.addEventListener("click", loadTemplates);

  templateSelect.addEventListener("change", () => {
    const id = templateSelect.value;
    const tpl = templates.find((t) => t.id === id);
    if (tpl) {
      templateInfo.classList.remove("hidden");
      templateInfo.innerHTML =
        `<p><strong>Name:</strong> ${tpl.name}</p>` +
        `<p><strong>ID:</strong> <code>${tpl.id}</code></p>` +
        (tpl.schemaName ? `<p><strong>Schema:</strong> ${tpl.schemaName}</p>` : "") +
        (tpl.status ? `<p><strong>Status:</strong> ${tpl.status}</p>` : "");
    } else {
      templateInfo.classList.add("hidden");
    }
    updateTrainBtn();
  });

  function updateFileList(): void {
    if (selectedFiles.length === 0) {
      fileList.classList.add("hidden");
      placeholder.classList.remove("hidden");
    } else {
      placeholder.classList.add("hidden");
      fileList.classList.remove("hidden");
      fileList.innerHTML = selectedFiles
        .map(
          (f, i) =>
            `<div class="file-item">` +
            `<span>📄 ${f.name}</span>` +
            `<span class="muted">${(f.size / 1024).toFixed(1)} KB</span>` +
            `<button class="remove-file-btn link-btn" data-idx="${i}">✕</button>` +
            `</div>`
        )
        .join("");

      fileList.querySelectorAll<HTMLButtonElement>(".remove-file-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const idx = parseInt(btn.dataset.idx ?? "0", 10);
          selectedFiles.splice(idx, 1);
          updateFileList();
          updateTrainBtn();
        });
      });
    }
    updateTrainBtn();
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

  trainBtn.addEventListener("click", async () => {
    const templateId = templateSelect.value;
    if (!templateId || selectedFiles.length === 0) return;

    trainBtn.disabled = true;
    statusBar.className = "status-bar status-loading";
    statusBar.textContent = `Training template with ${selectedFiles.length} file(s)…`;
    resultDiv.innerHTML = "";

    try {
      const result = await trainTemplate(templateId, selectedFiles);
      statusBar.className = result.success ? "status-bar status-success" : "status-bar status-error";
      statusBar.textContent = result.success
        ? `✓ Training triggered successfully.`
        : `✗ Training completed with errors.`;
      resultDiv.innerHTML = renderTrainingResult(result);
    } catch (err) {
      statusBar.className = "status-bar status-error";
      statusBar.textContent = `✗ Error: ${(err as Error).message}`;
    } finally {
      updateTrainBtn();
    }
  });
}