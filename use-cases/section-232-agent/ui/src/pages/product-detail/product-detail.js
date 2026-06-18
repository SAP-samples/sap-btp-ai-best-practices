import dayjs from "dayjs";

import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Tag.js";

import { pollClassificationJob, request, requestBinary } from "../../services/api.js";
import { pageRouter } from "../../modules/router.js";
import { escapeHtml, renderMarkdownHtml } from "../../utils/html.js";

const STEEL_SUBTYPE_LABELS = {
  electrical_steel: "Electrical steel",
  cold_rolled_coil_steel: "Cold rolled coil steel",
  hot_rolled_coil_steel: "Hot rolled coil steel",
  stainless_steel_304: "Stainless steel 304",
  stainless_steel_316: "Stainless steel 316",
  stainless_steel_bar: "Stainless steel bar",
  duplex_steel: "Duplex steel",
  cast_steel: "Cast steel"
};
const TOP_LEVEL_METAL_LABELS = {
  steel: "Steel",
  aluminum: "Aluminum",
  copper: "Copper",
  cast_iron: "Cast Iron"
};

const state = {
  itemId: null,
  detail: null,
  pendingDocumentPaths: [],
  reasoningMap: {},
  loading: false
};

function formatConfidence(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "—";
  }

  const scaled = numeric > 1 ? numeric : numeric * 100;
  return `${scaled.toFixed(2)}%`;
}

function confidenceToPercentage(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return numeric;
  }
  return numeric > 1 ? numeric : numeric * 100;
}

function getConfidencePresentation(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return {
      text: "—",
      label: "Unknown",
      design: "Neutral"
    };
  }

  const percentage = confidenceToPercentage(numeric);

  if (percentage >= 80) {
    return {
      text: formatConfidence(percentage),
      label: "High",
      design: "Positive"
    };
  }

  if (percentage >= 60) {
    return {
      text: formatConfidence(percentage),
      label: "Medium",
      design: "Critical"
    };
  }

  return {
    text: formatConfidence(percentage),
    label: "Low",
    design: "Negative"
  };
}

function renderConfidenceBadge(value) {
  const confidence = getConfidencePresentation(value);
  return `<ui5-tag design="${confidence.design}">${escapeHtml(`${confidence.text} ${confidence.label}`)}</ui5-tag>`;
}

function formatRoundedNumber(value, maximumFractionDigits = 3) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";

  const rounded = Number(numeric.toFixed(maximumFractionDigits));
  if (Number.isInteger(rounded)) {
    return String(rounded);
  }

  return rounded.toFixed(maximumFractionDigits).replace(/\.?0+$/, "");
}

function formatGrams(value) {
  const formatted = formatRoundedNumber(value, 3);
  return formatted === "—" ? formatted : `${formatted}g`;
}

function formatSourceStatus(value) {
  const normalized = String(value || "").trim();
  if (!normalized || normalized === "none") return "—";
  if (normalized === "documented") return "PDF";
  if (normalized === "estimated") return "Estimated";
  if (normalized === "needs_review") return "Needs review";
  if (normalized === "gcc_tracker") return "GCC tracker";
  return normalized.replaceAll("_", " ");
}

function detailHasDocuments(detail) {
  return Boolean((detail?.assigned_documents || []).length);
}

function getDetailDocumentMode() {
  const select = document.getElementById("detail-document-mode");
  const value = select?.selectedOption?.value || select?.value || "text_only";
  return value === "with_documents" ? "with_documents" : "text_only";
}

function hasCompletedClassification(detail) {
  return detail?.latest_classification?.status === "completed";
}

function openPdfRequiredDialog(message) {
  const dialog = document.getElementById("detail-pdf-required-dialog");
  const content = document.getElementById("detail-pdf-required-content");
  if (content) {
    content.textContent = message || "This item cannot be classified until a PDF is uploaded.";
  }
  if (dialog) {
    dialog.open = true;
  }
}

function closePdfRequiredDialog() {
  const dialog = document.getElementById("detail-pdf-required-dialog");
  if (dialog) {
    dialog.open = false;
  }
}

function buildDocumentReferenceMap() {
  const references = state.detail?.assigned_documents || [];

  return new Map(
    references
      .filter((reference) => reference?.path)
      .map((reference) => [reference.path, reference])
  );
}

function getDisplayDocumentPath(path, referenceMap) {
  return referenceMap.get(path)?.relative_path || path;
}

function getDisplayDocumentName(path, referenceMap) {
  return referenceMap.get(path)?.file_name || path.split("/").pop() || path;
}

function formatSourceDocument(documentRef) {
  const filename = documentRef?.filename || "Document";
  const pageNumber = Number(documentRef?.page_number);
  if (Number.isFinite(pageNumber) && pageNumber > 0) {
    return `${filename} (page ${pageNumber})`;
  }
  return filename;
}

function renderSourceCell(row) {
  const documents = Array.isArray(row?.source_documents) ? row.source_documents : [];
  if (documents.length) {
    return `
      <div class="product-detail-source-list">
        ${documents
          .map(
            (documentRef) =>
              `<div class="product-detail-source-item">${escapeHtml(formatSourceDocument(documentRef))}</div>`
          )
          .join("")}
      </div>
    `;
  }
  return `<span class="product-detail-table-source-status">${escapeHtml(formatSourceStatus(row?.source_status))}</span>`;
}

function buildFallbackMetalRows(finalComposition) {
  return Object.entries(TOP_LEVEL_METAL_LABELS).map(([key, label]) => ({
    type: label,
    weight_grams: finalComposition?.top_level_grams?.[key] ?? 0,
    source_documents: [],
    source_status: "none"
  }));
}

function buildFallbackSteelSubtypeRows(finalComposition) {
  return Object.entries(STEEL_SUBTYPE_LABELS).map(([key, label]) => ({
    type: label,
    weight_grams: finalComposition?.steel_subtype_grams?.[key] ?? 0,
    source_documents: [],
    source_status: "none"
  }));
}

function partitionCompositionRows(rows) {
  return rows.reduce(
    (result, row) => {
      if (Number(row?.weight_grams) > 0) {
        result.visibleRows.push(row);
      } else {
        result.hiddenRows.push(row);
      }
      return result;
    },
    { visibleRows: [], hiddenRows: [] }
  );
}

function renderCompositionTable(rows, typeHeader = "Type") {
  return `
    <div class="product-detail-composition-table">
      <div class="product-detail-composition-row product-detail-composition-row-header">
        <div>${escapeHtml(typeHeader)}</div>
        <div>Weight</div>
        <div>Source</div>
      </div>
      ${rows
        .map(
          (row) => `
            <div class="product-detail-composition-row">
              <div>${escapeHtml(row?.type || "—")}</div>
              <div>${escapeHtml(formatGrams(row?.weight_grams))}</div>
              <div class="product-detail-composition-source">${renderSourceCell(row)}</div>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderCompositionEmptyState(message) {
  return `<div class="product-detail-composition-empty">${escapeHtml(message)}</div>`;
}

function renderHiddenCompositionRows(rows, typeHeader, label) {
  if (!rows.length) {
    return "";
  }

  return `
    <details class="product-detail-composition-details">
      <summary>${escapeHtml(label)}</summary>
      ${renderCompositionTable(rows, typeHeader)}
    </details>
  `;
}

function renderCompositionSection(rows, { typeHeader = "Type", emptyMessage, hiddenLabel } = {}) {
  const { visibleRows, hiddenRows } = partitionCompositionRows(rows);

  return `
    ${visibleRows.length ? renderCompositionTable(visibleRows, typeHeader) : renderCompositionEmptyState(emptyMessage)}
    ${renderHiddenCompositionRows(hiddenRows, typeHeader, hiddenLabel)}
  `;
}

function splitHtsDescriptionParts(description) {
  return String(description || "")
    .split(/\s*>\s*/g)
    .map((part) => part.trim())
    .filter(Boolean);
}

function renderHtsDescription(description, emptyText = "No HTS candidate was produced.") {
  const parts = splitHtsDescriptionParts(description);
  if (!parts.length) {
    return `<div>${escapeHtml(emptyText)}</div>`;
  }
  if (parts.length === 1) {
    return `<div>${escapeHtml(parts[0])}</div>`;
  }

  return `
    <ul class="product-detail-hts-list">
      ${parts.map((part) => `<li>${escapeHtml(part)}</li>`).join("")}
    </ul>
  `;
}

function formatHtsDescriptionMarkdown(description, emptyText = "No HTS candidate was produced.") {
  const parts = splitHtsDescriptionParts(description);
  if (!parts.length) return emptyText;
  if (parts.length === 1) return parts[0];
  return parts.map((part) => `- ${part}`).join("\n");
}

function buildHtsReasoningWithSources(hts) {
  return hts?.reasoning || hts?.best_candidate?.reasoning || "No HTS reasoning was provided.";
}

function buildHtsCandidatesReasoning(candidates) {
  return candidates
    .map((candidate) => {
      const summaryLine = [
        `Confidence: ${formatConfidence(candidate.confidence)}`,
        formatHtsDescriptionMarkdown(candidate.description)
      ]
        .filter(Boolean)
        .join("  \n");

      const reasoning = candidate.reasoning || candidate.description || "No reasoning was provided.";

      return `### **${candidate.code}**\n\n${summaryLine}\n\n${reasoning}`;
    })
    .join("\n\n---\n\n");
}

function showMessage(text, design = "Information") {
  const strip = document.getElementById("detail-message");
  if (!strip) return;
  if (!text) {
    strip.style.display = "none";
    strip.textContent = "";
    return;
  }
  strip.design = design;
  strip.textContent = text;
  strip.style.display = "block";
}

function setBusy(isBusy) {
  state.loading = isBusy;
  const exportButton = document.getElementById("detail-export-button");
  const classifyButton = document.getElementById("detail-classify-button");
  const docSaveButton = document.getElementById("detail-doc-save-button");
  const docUploadButton = document.getElementById("detail-doc-upload-button");
  [exportButton, classifyButton, docSaveButton, docUploadButton].forEach((button) => {
    if (!button) return;
    button.disabled = isBusy;
    button.loading =
      isBusy &&
      (button === exportButton ||
        button === classifyButton ||
        button === docSaveButton ||
        button === docUploadButton);
  });
  updateHeader();
}

function renderGeneralInformation() {
  const container = document.getElementById("detail-general-content");
  if (!container) return;

  const detail = state.detail;
  if (!detail) {
    container.innerHTML = `<div class="product-detail-empty">Unable to load item details.</div>`;
    return;
  }

  container.innerHTML = `
    <div class="product-detail-read-grid">
      <div class="product-detail-field"><ui5-label>Product Code</ui5-label><div class="product-detail-read-value">${escapeHtml(detail.product_code || "—")}</div></div>
      <div class="product-detail-field"><ui5-label>Priority</ui5-label><div class="product-detail-read-value">${escapeHtml(detail.priority || "—")}</div></div>
      <div class="product-detail-field"><ui5-label>Business Segment</ui5-label><div class="product-detail-read-value">${escapeHtml(detail.business_segment || "—")}</div></div>
      <div class="product-detail-field"><ui5-label>Site</ui5-label><div class="product-detail-read-value">${escapeHtml(detail.site || "—")}</div></div>
      <div class="product-detail-field"><ui5-label>PN Revised/Standardized</ui5-label><div class="product-detail-read-value">${escapeHtml(detail.pn_revised_standardized || "—")}</div></div>
      <div class="product-detail-field"><ui5-label>Total Weight (Gram)</ui5-label><div class="product-detail-read-value">${escapeHtml(detail.total_weight_gram ?? "—")}</div></div>
      <div class="product-detail-field"><ui5-label>Part Description</ui5-label><div class="product-detail-read-value">${escapeHtml(detail.part_description || "—")}</div></div>
      <div class="product-detail-field"><ui5-label>New Part Description</ui5-label><div class="product-detail-read-value">${escapeHtml(detail.new_part_description || "—")}</div></div>
      <div class="product-detail-field"><ui5-label>Date Started</ui5-label><div class="product-detail-read-value">${escapeHtml(detail.date_started || "—")}</div></div>
      <div class="product-detail-field"><ui5-label>Date Completed</ui5-label><div class="product-detail-read-value">${escapeHtml(detail.date_completed || "—")}</div></div>
      <div class="product-detail-field"><ui5-label>Priority Detail</ui5-label><div class="product-detail-read-value">${escapeHtml(detail.priority_detail || "—")}</div></div>
    </div>
  `;
}

function renderAssignedDocuments() {
  const container = document.getElementById("detail-assigned-documents");
  if (!container) return;

  if (!state.pendingDocumentPaths.length) {
    const detail = state.detail;
    const message = detail?.assigned_documents?.length
      ? "Some assigned documents are missing and were not loaded."
      : "No uploaded PDF is assigned yet.";
    container.innerHTML = `<div class="product-detail-empty">${escapeHtml(message)}</div>`;
    return;
  }

  const referenceMap = buildDocumentReferenceMap();
  container.innerHTML = `
    <div class="product-detail-doc-list">
      ${state.pendingDocumentPaths
        .map(
          (path) => `
            <div class="product-detail-doc-item">
              <div class="product-detail-doc-copy">
                <div class="product-detail-doc-name">${escapeHtml(getDisplayDocumentName(path, referenceMap))}</div>
                <div class="product-detail-doc-path">${escapeHtml(getDisplayDocumentPath(path, referenceMap))}</div>
              </div>
              <ui5-button design="Transparent" icon="decline" data-role="remove-assigned" data-path="${escapeHtml(path)}"></ui5-button>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function registerReasoning(key, title, body) {
  if (!body) return "";
  state.reasoningMap[key] = { title, body };
  return `<ui5-button design="Transparent" icon="hint" data-role="open-reasoning" data-key="${escapeHtml(key)}"></ui5-button>`;
}

function renderClassification() {
  const container = document.getElementById("detail-classification-content");
  if (!container) return;

  state.reasoningMap = {};
  const detail = state.detail;
  const classification = detail?.latest_classification;
  if (!classification) {
    container.innerHTML = `<div class="product-detail-empty">No classification has been run for this item yet.</div>`;
    return;
  }

  const finalComposition = classification.final_composition;
  const hts = classification.hts_classification;
  const bestCandidate = hts?.best_candidate;
  const secondaryCandidates = (hts?.candidates || []).filter((candidate) => candidate?.code !== bestCandidate?.code).slice(0, 5);
  const section232 = classification.section_232_assessment;

  if (classification.status === "blocked") {
    const blockingReason = classification.blocking_reason || {};
    container.innerHTML = `
      <div class="product-detail-classification-grid">
        <div class="product-detail-result-card">
          <div class="product-detail-result-header">
            <div class="product-detail-result-title">Metal Composition</div>
          </div>
          <div class="product-detail-result-body">
            <div>Status: <ui5-tag design="Critical">${escapeHtml(classification.status)}</ui5-tag></div>
            <div>${escapeHtml(blockingReason.message || classification.error || "Prediction is blocked until PDF evidence is available.")}</div>
            <div>Documents: ${escapeHtml(blockingReason.docs_status || state.detail?.docs_status || "No PDFs assigned")}</div>
          </div>
        </div>
      </div>
    `;
    return;
  }

  const provenance = finalComposition?.provenance || {};
  const metalRows = Array.isArray(finalComposition?.metal_rows) && finalComposition.metal_rows.length
    ? finalComposition.metal_rows
    : buildFallbackMetalRows(finalComposition);
  const steelSubtypeRows = Array.isArray(finalComposition?.steel_subtype_rows) && finalComposition.steel_subtype_rows.length
    ? finalComposition.steel_subtype_rows
    : buildFallbackSteelSubtypeRows(finalComposition);

  container.innerHTML = `
    <div class="product-detail-classification-grid">
      <div class="product-detail-result-card">
        <div class="product-detail-result-header">
          <div class="product-detail-result-title">Metal Composition</div>
          ${registerReasoning("composition", "Metal Composition reasoning", finalComposition?.reasoning || "")}
        </div>
        <div class="product-detail-result-body">
          <div>Status: <ui5-tag design="${classification.status === "completed" ? "Positive" : "Critical"}">${escapeHtml(classification.status)}</ui5-tag></div>
          <div>Metal item: ${escapeHtml(finalComposition?.is_metal_item ?? "—")}</div>
          <div>Total metal grams: ${escapeHtml(formatGrams(finalComposition?.estimated_total_metal_grams))}</div>
          ${renderCompositionSection(metalRows, {
            typeHeader: "Type",
            emptyMessage: "No detected metals",
            hiddenLabel: `Show ${metalRows.filter((row) => Number(row?.weight_grams) <= 0).length} zero-weight entries`
          })}
          <div class="product-detail-subtype-section">
            <div class="product-detail-subtype-title">Steel subtypes</div>
            ${renderCompositionSection(steelSubtypeRows, {
              typeHeader: "Subtype",
              emptyMessage: "No detected steel subtypes",
              hiddenLabel: `Show ${steelSubtypeRows.filter((row) => Number(row?.weight_grams) <= 0).length} zero-weight entries`
            })}
          </div>
        </div>
      </div>

      <div class="product-detail-result-card">
        <div class="product-detail-result-header">
          <div class="product-detail-result-title">HTS code</div>
          ${registerReasoning("hts-best", "HTS code reasoning", buildHtsReasoningWithSources(hts))}
        </div>
        <div class="product-detail-result-body">
          <div class="product-detail-meta-row">
            <div>Best candidate: <strong>${escapeHtml(bestCandidate?.code || "—")}</strong></div>
            ${renderConfidenceBadge(bestCandidate?.confidence ?? hts?.confidence)}
          </div>
          <div class="product-detail-hts-description">${renderHtsDescription(bestCandidate?.description)}</div>
        </div>
      </div>

      <div class="product-detail-result-card">
        <div class="product-detail-result-header">
          <div class="product-detail-result-title">HTS Candidates</div>
          ${registerReasoning(
            "hts-secondary",
            "HTS Candidates",
            buildHtsCandidatesReasoning(secondaryCandidates)
          )}
        </div>
        <div class="product-detail-result-body">
          ${
            secondaryCandidates.length
              ? secondaryCandidates
                  .map(
                    (candidate) => `
                      <div class="product-detail-hts-candidate">
                        <div class="product-detail-meta-row">
                          <div><strong>${escapeHtml(candidate.code)}</strong></div>
                          ${renderConfidenceBadge(candidate.confidence)}
                        </div>
                        ${renderHtsDescription(candidate.description, "No description was provided.")}
                      </div>
                    `
                  )
                  .join("")
              : `<div>No secondary HTS candidates were retained.</div>`
          }
        </div>
      </div>

      <div class="product-detail-result-card">
        <div class="product-detail-result-header">
          <div class="product-detail-result-title">Section 232</div>
          ${registerReasoning("section-232", "Section 232 reasoning", section232?.basis_summary)}
        </div>
        <div class="product-detail-result-body">
          <div class="product-detail-meta-row">
            <div>Decision: <ui5-tag design="${section232?.decision === "subject" ? "Critical" : section232?.decision === "not_subject" ? "Positive" : "Information"}">${escapeHtml(section232?.decision || "—")}</ui5-tag></div>
            ${renderConfidenceBadge(section232?.confidence)}
          </div>
          ${
            section232?.needs_human_review
              ? `<div class="product-detail-meta-row"><ui5-tag design="Contrast">Human review recommended</ui5-tag></div>`
              : ""
          }
          <div>${escapeHtml(section232?.basis_summary || "No Section 232 assessment is available.")}</div>
        </div>
      </div>
    </div>
  `;
}

function updateHeader() {
  const title = document.getElementById("detail-title");
  const subtitle = document.getElementById("detail-subtitle");
  const kicker = document.getElementById("detail-kicker");
  const exportButton = document.getElementById("detail-export-button");
  const classifyButton = document.getElementById("detail-classify-button");
  const docSaveButton = document.getElementById("detail-doc-save-button");
  const uploadButton = document.getElementById("detail-doc-upload-button");
  const clearButton = document.getElementById("detail-doc-clear-button");

  const detail = state.detail;
  kicker.textContent = "GCC Tracker Item";
  title.textContent = detail?.product_code || "Product detail";
  subtitle.textContent = detail?.last_classified_at
    ? `Last classified ${dayjs(detail.last_classified_at).format("DD MMM YYYY HH:mm")}`
    : "No prediction has been run for this item yet.";
  exportButton.style.display = "inline-flex";
  classifyButton.style.display = "inline-flex";
  exportButton.disabled = state.loading || !hasCompletedClassification(detail);
  docSaveButton.disabled = state.loading;
  uploadButton.disabled = state.loading;
  clearButton.disabled = state.loading;
  classifyButton.disabled = state.loading;
}

async function loadDetail() {
  setBusy(true);
  try {
    const detail = await request(`/api/metal-composition/items/${encodeURIComponent(state.itemId)}`);
    state.detail = detail;
    state.pendingDocumentPaths = (detail.assigned_documents || []).map((doc) => doc.path);
    updateHeader();
    renderGeneralInformation();
    renderAssignedDocuments();
    renderClassification();
    showMessage(detail.warnings?.length ? detail.warnings.join(" ") : "", detail.warnings?.length ? "Information" : "Information");
  } catch (error) {
    showMessage(error.message, "Negative");
  } finally {
    setBusy(false);
  }
}

async function saveDocuments() {
  if (!state.itemId) {
    showMessage("Load a GCC tracker item before persisting document assignments.", "Critical");
    return;
  }

  setBusy(true);
  try {
    const detail = await request(
      `/api/metal-composition/items/${encodeURIComponent(state.itemId)}/documents`,
      "PUT",
      { document_paths: state.pendingDocumentPaths }
    );
    state.detail = detail;
    state.pendingDocumentPaths = (detail.assigned_documents || []).map((doc) => doc.path);
    renderAssignedDocuments();
    updateHeader();
    showMessage("Document assignment saved.", "Positive");
  } catch (error) {
    showMessage(error.message, "Negative");
  } finally {
    setBusy(false);
  }
}

async function uploadDocument(file) {
  if (!state.itemId) {
    showMessage("Load a GCC tracker item before uploading PDFs.", "Critical");
    return;
  }
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  setBusy(true);
  try {
    const detail = await request(
      `/api/metal-composition/items/${encodeURIComponent(state.itemId)}/documents/upload`,
      "POST",
      formData
    );
    state.detail = detail;
    state.pendingDocumentPaths = (detail.assigned_documents || []).map((doc) => doc.path);
    renderAssignedDocuments();
    updateHeader();
    showMessage("PDF uploaded and assigned.", "Positive");
  } catch (error) {
    showMessage(error.message, "Negative");
  } finally {
    setBusy(false);
  }
}

async function classifyItem() {
  if (!state.itemId) return;
  const documentMode = getDetailDocumentMode();
  if (documentMode === "with_documents" && !detailHasDocuments(state.detail)) {
    openPdfRequiredDialog("PDF evidence mode requires an uploaded PDF for this item.");
    return;
  }
  setBusy(true);
  try {
    const response = await request("/api/metal-composition/items/predict", "POST", {
      item_id: state.itemId,
      document_mode: documentMode
    });
    showMessage("Prediction started. Waiting for the background job to finish.", "Information");
    const job = await pollClassificationJob(response.job_id);
    await loadDetail();
    const latestClassification = state.detail?.latest_classification;
    if (latestClassification?.status === "blocked") {
      openPdfRequiredDialog(
        latestClassification.blocking_reason?.message || "This item is blocked because PDF evidence is unavailable."
      );
      showMessage(latestClassification.blocking_reason?.message || "Prediction blocked.", "Critical");
      return;
    }
    if (job.status === "completed") {
      showMessage("Prediction completed.", "Positive");
      return;
    }
    showMessage(job.error_message || "Prediction finished with failures.", "Negative");
  } catch (error) {
    showMessage(error.message, "Negative");
  } finally {
    setBusy(false);
  }
}

async function exportReport() {
  if (!state.itemId || !hasCompletedClassification(state.detail)) {
    return;
  }

  setBusy(true);
  try {
    const { blob, filename } = await requestBinary(
      `/api/metal-composition/items/${encodeURIComponent(state.itemId)}/export-report`
    );
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = filename || `${state.detail?.product_code || "classification-report"}-classification-report.pdf`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(downloadUrl);
    showMessage("Report exported.", "Positive");
  } catch (error) {
    showMessage(error.message, "Negative");
  } finally {
    setBusy(false);
  }
}

function bindEvents() {
  document.getElementById("detail-back-button")?.addEventListener("click", () => pageRouter.navigate("/products"));
  document.getElementById("detail-export-button")?.addEventListener("click", exportReport);
  document.getElementById("detail-classify-button")?.addEventListener("click", classifyItem);
  document.getElementById("detail-doc-save-button")?.addEventListener("click", saveDocuments);
  document.getElementById("detail-doc-clear-button")?.addEventListener("click", async () => {
    state.pendingDocumentPaths = [];
    renderAssignedDocuments();
    if (state.itemId) {
      await saveDocuments();
    }
  });
  document.getElementById("detail-doc-upload-button")?.addEventListener("click", () => {
    document.getElementById("detail-doc-upload-input")?.click();
  });
  document.getElementById("detail-doc-upload-input")?.addEventListener("change", async (event) => {
    const [file] = Array.from(event.target.files || []);
    event.target.value = "";
    await uploadDocument(file);
  });
  document.getElementById("detail-assigned-documents")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-role='remove-assigned']");
    if (!button) return;
    const path = button.dataset.path;
    if (!path) return;
    state.pendingDocumentPaths = state.pendingDocumentPaths.filter((candidate) => candidate !== path);
    renderAssignedDocuments();
  });
  document.getElementById("detail-classification-content")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-role='open-reasoning']");
    if (!button) return;
    const reasoningKey = button.dataset.key;
    const payload = state.reasoningMap[reasoningKey];
    if (!payload) return;
    const dialog = document.getElementById("detail-reasoning-dialog");
    const content = document.getElementById("detail-reasoning-content");
    content.dataset.reasoningKey = reasoningKey || "";
    dialog.headerText = payload.title;
    content.innerHTML = renderMarkdownHtml(payload.body);
    dialog.open = true;
  });
  document.getElementById("detail-reasoning-close-button")?.addEventListener("click", () => {
    const dialog = document.getElementById("detail-reasoning-dialog");
    const content = document.getElementById("detail-reasoning-content");
    delete content.dataset.reasoningKey;
    dialog.open = false;
  });
  document.getElementById("detail-pdf-required-close-button")?.addEventListener("click", closePdfRequiredDialog);
}

export default async function initProductDetailPage(ctx = null) {
  state.itemId = ctx?.params?.itemId || null;
  state.detail = null;
  state.pendingDocumentPaths = [];
  state.reasoningMap = {};
  if (!state.itemId) {
    pageRouter.navigate("/products");
    return;
  }
  bindEvents();
  updateHeader();
  renderGeneralInformation();
  renderAssignedDocuments();
  renderClassification();
  await loadDetail();
}
