import dayjs from "dayjs";

import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/CheckBox.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Link.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Tag.js";

import { buildQuery, pollClassificationJob, request } from "../../services/api.js";
import { pageRouter } from "../../modules/router.js";
import { escapeHtml } from "../../utils/html.js";

const PAGE_SIZE = 25;
const DEFAULT_FILTERS = Object.freeze({
  priority: "",
  business_segment: "",
  product_code: "",
  pn_revised_standardized: "",
  new_part_description: "",
  part_description: "",
  has_documents: "",
  is_classified: ""
});

const state = {
  filters: { ...DEFAULT_FILTERS },
  items: [],
  facets: {
    priority: [],
    business_segment: []
  },
  total: 0,
  offset: 0,
  loading: false,
  selectedIds: new Set(),
  selectedItemMeta: new Map(),
  missingDocumentItems: []
};

function statusDesign(status) {
  if (status === "completed") return "Positive";
  if (status === "blocked") return "Critical";
  if (status === "failed") return "Negative";
  if (status === "needs_disambiguation") return "Critical";
  return "Information";
}

function docDesign(status) {
  if (status.startsWith("Assigned")) return "Positive";
  return "Neutral";
}

function getProductsDocumentMode() {
  const select = document.getElementById("products-document-mode");
  const value = select?.selectedOption?.value || select?.value || "text_only";
  return value === "with_documents" ? "with_documents" : "text_only";
}

function showMessage(text, design = "Information") {
  const strip = document.getElementById("products-message");
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

function buildSelectedItemMeta(item) {
  return {
    item_id: item.item_id,
    product_code: item.product_code || item.item_id,
    priority: item.priority || null,
    docs_status: item.docs_status || "No documents",
    has_documents: Boolean(item.has_documents)
  };
}

function cacheSelectedItem(item) {
  if (!item?.item_id) return;
  state.selectedItemMeta.set(item.item_id, buildSelectedItemMeta(item));
}

function syncSelectedMetadataFromCurrentPage() {
  state.items.forEach((item) => {
    if (state.selectedIds.has(item.item_id)) {
      cacheSelectedItem(item);
    }
  });
}

function normalizeMissingDocumentItem(item) {
  return {
    item_id: item?.item_id || "",
    product_code: item?.product_code || item?.item_id || "Unknown item",
    priority: item?.priority || null,
    docs_status: item?.docs_status || "No documents"
  };
}

function formatMissingDocumentItem(item) {
  const parts = [];
  if (item.priority) parts.push(item.priority);
  if (item.item_id) parts.push(item.item_id);
  parts.push(item.docs_status);
  return `
    <div class="products-missing-docs-item">
      <div class="products-missing-docs-item-title">${escapeHtml(item.product_code)}</div>
      <div class="products-missing-docs-item-meta">${escapeHtml(parts.join(" · "))}</div>
    </div>
  `;
}

function renderMissingDocumentsDialog() {
  const container = document.getElementById("products-missing-docs-list");
  if (!container) return;
  container.innerHTML = state.missingDocumentItems.map(formatMissingDocumentItem).join("");
}

function openMissingDocumentsDialog(items) {
  const dialog = document.getElementById("products-missing-docs-dialog");
  state.missingDocumentItems = items.map(normalizeMissingDocumentItem);
  renderMissingDocumentsDialog();
  if (dialog) {
    dialog.open = true;
  }
}

function closeMissingDocumentsDialog() {
  const dialog = document.getElementById("products-missing-docs-dialog");
  if (dialog) {
    dialog.open = false;
  }
}

function extractMissingDocumentsFromError(error) {
  const detail = error?.payload?.detail;
  if (!detail || detail.code !== "pdf_required_blocked") {
    return [];
  }
  return Array.isArray(detail.items) ? detail.items.map(normalizeMissingDocumentItem) : [];
}

function setBusy(isBusy) {
  state.loading = isBusy;
  const loadingEl = document.getElementById("products-loading");
  const goButton = document.getElementById("products-go-button");
  const classifyButton = document.getElementById("products-classify-button");
  const prevButton = document.getElementById("products-prev-button");
  const nextButton = document.getElementById("products-next-button");
  if (loadingEl) {
    loadingEl.style.display = isBusy ? "block" : "none";
  }
  [goButton, classifyButton, prevButton, nextButton].forEach((button) => {
    if (!button) return;
    button.disabled = isBusy;
    button.loading = isBusy && button === classifyButton ? true : false;
  });
}

function syncFilterValues() {
  const priority = document.getElementById("filter-priority");
  const segment = document.getElementById("filter-business-segment");
  const productCode = document.getElementById("filter-product-code");
  const pn = document.getElementById("filter-pn");
  const newDescription = document.getElementById("filter-new-description");
  const partDescription = document.getElementById("filter-part-description");
  const hasDocuments = document.getElementById("filter-has-documents");
  const isClassified = document.getElementById("filter-is-classified");

  state.filters = {
    priority: priority?.selectedOption?.value || priority?.value || "",
    business_segment: segment?.selectedOption?.value || segment?.value || "",
    product_code: productCode?.value?.trim() || "",
    pn_revised_standardized: pn?.value?.trim() || "",
    new_part_description: newDescription?.value?.trim() || "",
    part_description: partDescription?.value?.trim() || "",
    has_documents: hasDocuments?.selectedOption?.value || hasDocuments?.value || "",
    is_classified: isClassified?.selectedOption?.value || isClassified?.value || ""
  };
}

function setSelectValue(elementId, value) {
  const element = document.getElementById(elementId);
  if (!element) return;
  Array.from(element.querySelectorAll("ui5-option")).forEach((option) => {
    option.selected = (option.value || option.getAttribute("value") || option.textContent || "") === value;
  });
}

function applyFilterValues() {
  const productCode = document.getElementById("filter-product-code");
  const pn = document.getElementById("filter-pn");
  const newDescription = document.getElementById("filter-new-description");
  const partDescription = document.getElementById("filter-part-description");

  if (productCode) productCode.value = state.filters.product_code;
  if (pn) pn.value = state.filters.pn_revised_standardized;
  if (newDescription) newDescription.value = state.filters.new_part_description;
  if (partDescription) partDescription.value = state.filters.part_description;

  setSelectValue("filter-priority", state.filters.priority);
  setSelectValue("filter-business-segment", state.filters.business_segment);
  setSelectValue("filter-has-documents", state.filters.has_documents);
  setSelectValue("filter-is-classified", state.filters.is_classified);
}

function renderSelectOptions(elementId, options, currentValue) {
  const element = document.getElementById(elementId);
  if (!element) return;
  const current = currentValue || "";
  element.innerHTML = [
    `<ui5-option value="" ${current === "" ? "selected" : ""}>All</ui5-option>`,
    ...options.map(
      (option) =>
        `<ui5-option value="${escapeHtml(option.value)}" ${current === option.value ? "selected" : ""}>${escapeHtml(`${option.value} (${option.count})`)}</ui5-option>`
    )
  ].join("");
}

function renderToolbarSummary() {
  const totalTag = document.getElementById("products-total-tag");
  const summary = document.getElementById("products-page-summary");
  const selection = document.getElementById("products-selection-count");
  const prevButton = document.getElementById("products-prev-button");
  const nextButton = document.getElementById("products-next-button");

  if (totalTag) {
    totalTag.textContent = `${state.total} product${state.total === 1 ? "" : "s"}`;
  }
  if (summary) {
    const start = state.total === 0 ? 0 : state.offset + 1;
    const end = Math.min(state.offset + PAGE_SIZE, state.total);
    summary.textContent = `${start}-${end} of ${state.total}`;
  }
  if (selection) {
    selection.textContent = `${state.selectedIds.size} selected`;
  }
  if (prevButton) {
    prevButton.disabled = state.loading || state.offset === 0;
  }
  if (nextButton) {
    nextButton.disabled = state.loading || state.offset + PAGE_SIZE >= state.total;
  }
  const classifyButton = document.getElementById("products-classify-button");
  if (classifyButton) {
    classifyButton.disabled = state.loading || state.selectedIds.size === 0;
  }
}

function renderRows() {
  const tbody = document.getElementById("products-table-body");
  const selectAll = document.getElementById("products-select-all");
  if (!tbody) return;

  if (!state.items.length) {
    tbody.innerHTML = `<tr><td colspan="10" class="products-empty-state">No products matched the current filters.</td></tr>`;
    if (selectAll) {
      selectAll.checked = false;
      selectAll.indeterminate = false;
    }
    renderToolbarSummary();
    return;
  }

  tbody.innerHTML = state.items
    .map((item) => {
      const checked = state.selectedIds.has(item.item_id) ? "checked" : "";
      return `
        <tr>
          <td class="products-checkbox-cell">
            <ui5-checkbox data-role="row-select" data-item-id="${escapeHtml(item.item_id)}" ${checked}></ui5-checkbox>
          </td>
          <td>
            <ui5-link href="#" class="products-product-link" data-role="open-item" data-item-id="${escapeHtml(item.item_id)}">${escapeHtml(item.product_code)}</ui5-link>
          </td>
          <td>${escapeHtml(item.pn_revised_standardized || "—")}</td>
          <td class="products-description-cell">${escapeHtml(item.new_part_description || "—")}</td>
          <td class="products-description-cell">${escapeHtml(item.part_description || "—")}</td>
          <td>${escapeHtml(item.priority || "—")}</td>
          <td>${escapeHtml(item.business_segment || "—")}</td>
          <td><ui5-tag design="${docDesign(item.docs_status)}">${escapeHtml(item.docs_status)}</ui5-tag></td>
          <td><ui5-tag design="${statusDesign(item.classification_status)}">${escapeHtml(item.classification_status)}</ui5-tag></td>
          <td class="products-muted">${escapeHtml(item.last_classified_at ? dayjs(item.last_classified_at).format("DD MMM YYYY HH:mm") : "—")}</td>
        </tr>
      `;
    })
    .join("");

  if (selectAll) {
    const selectedOnPage = state.items.filter((item) => state.selectedIds.has(item.item_id)).length;
    selectAll.checked = selectedOnPage > 0 && selectedOnPage === state.items.length;
    selectAll.indeterminate = selectedOnPage > 0 && selectedOnPage < state.items.length;
  }
  renderToolbarSummary();
}

async function loadItems({ resetOffset = false, syncFilters = false } = {}) {
  if (resetOffset) {
    state.offset = 0;
  }
  if (syncFilters) {
    syncFilterValues();
  }
  setBusy(true);
  showMessage("");

  try {
    const query = buildQuery({
      ...state.filters,
      limit: PAGE_SIZE,
      offset: state.offset
    });
    const response = await request(`/api/metal-composition/items${query}`);
    state.items = response.items || [];
    state.facets = response.facets || { priority: [], business_segment: [] };
    state.total = response.total || 0;
    syncSelectedMetadataFromCurrentPage();

    renderSelectOptions("filter-priority", state.facets.priority || [], state.filters.priority);
    renderSelectOptions(
      "filter-business-segment",
      state.facets.business_segment || [],
      state.filters.business_segment
    );
    applyFilterValues();
    renderRows();
  } catch (error) {
    state.items = [];
    state.total = 0;
    renderRows();
    showMessage(error.message, "Negative");
  } finally {
    setBusy(false);
  }
}

async function runBatchClassification() {
  const itemIds = [...state.selectedIds];
  if (!itemIds.length) return;
  const documentMode = getProductsDocumentMode();

  setBusy(true);
  try {
    const response = await request("/api/metal-composition/items/predict-batch", "POST", {
      item_ids: itemIds,
      document_mode: documentMode
    });
    showMessage("Prediction started. Waiting for the background job to finish.", "Information");
    const job = await pollClassificationJob(response.job_id);
    const failed = job.failed_count || 0;
    const completed = job.completed_count || 0;
    state.selectedIds.clear();
    state.selectedItemMeta.clear();
    state.missingDocumentItems = [];
    closeMissingDocumentsDialog();
    renderRows();
    showMessage(`Prediction finished. ${completed} completed, ${failed} failed.`, failed ? "Critical" : "Positive");
    await loadItems();
  } catch (error) {
    const missingDocumentItems = extractMissingDocumentsFromError(error);
    if (error.status === 409 && missingDocumentItems.length) {
      openMissingDocumentsDialog(missingDocumentItems);
      return;
    }
    showMessage(error.message, "Negative");
  } finally {
    setBusy(false);
  }
}

async function classifySelection() {
  const itemIds = [...state.selectedIds];
  if (!itemIds.length) return;
  const documentMode = getProductsDocumentMode();

  const missingDocumentItems = documentMode === "with_documents"
    ? itemIds
        .map((itemId) => state.selectedItemMeta.get(itemId))
        .filter((item) => item && !item.has_documents)
        .map(normalizeMissingDocumentItem)
    : [];

  if (missingDocumentItems.length) {
    openMissingDocumentsDialog(missingDocumentItems);
    return;
  }

  await runBatchClassification();
}

function bindEvents() {
  document.getElementById("products-go-button")?.addEventListener("click", () =>
    loadItems({ resetOffset: true, syncFilters: true })
  );
  document.getElementById("products-clear-button")?.addEventListener("click", () => {
    state.filters = { ...DEFAULT_FILTERS };
    state.selectedIds.clear();
    state.selectedItemMeta.clear();
    state.missingDocumentItems = [];
    closeMissingDocumentsDialog();
    applyFilterValues();
    loadItems({ resetOffset: true });
  });
  document.getElementById("products-classify-button")?.addEventListener("click", classifySelection);
  document.getElementById("products-prev-button")?.addEventListener("click", async () => {
    if (state.offset === 0) return;
    state.offset = Math.max(0, state.offset - PAGE_SIZE);
    await loadItems();
  });
  document.getElementById("products-next-button")?.addEventListener("click", async () => {
    if (state.offset + PAGE_SIZE >= state.total) return;
    state.offset += PAGE_SIZE;
    await loadItems();
  });
  document.getElementById("products-select-all")?.addEventListener("change", (event) => {
    if (event.target.checked) {
      state.items.forEach((item) => {
        state.selectedIds.add(item.item_id);
        cacheSelectedItem(item);
      });
    } else {
      state.items.forEach((item) => {
        state.selectedIds.delete(item.item_id);
        state.selectedItemMeta.delete(item.item_id);
      });
    }
    renderRows();
  });
  document.getElementById("products-table-body")?.addEventListener("change", (event) => {
    const checkbox = event.target;
    if (!(checkbox instanceof HTMLElement) || checkbox.dataset.role !== "row-select") return;
    const itemId = checkbox.dataset.itemId;
    if (!itemId) return;
    if (checkbox.checked) {
      state.selectedIds.add(itemId);
      const item = state.items.find((candidate) => candidate.item_id === itemId);
      if (item) cacheSelectedItem(item);
    } else {
      state.selectedIds.delete(itemId);
      state.selectedItemMeta.delete(itemId);
    }
    renderRows();
  });
  document.getElementById("products-table-body")?.addEventListener("click", (event) => {
    const link = event.target.closest("[data-role='open-item']");
    if (!link) return;
    event.preventDefault();
    const itemId = link.dataset.itemId;
    if (!itemId) return;
    pageRouter.navigate(`/products/${itemId}`);
  });
  [
    "filter-product-code",
    "filter-pn",
    "filter-new-description",
    "filter-part-description"
  ].forEach((id) => {
    document.getElementById(id)?.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        loadItems({ resetOffset: true, syncFilters: true });
      }
    });
  });
  document.getElementById("products-missing-docs-close-button")?.addEventListener("click", () => {
    closeMissingDocumentsDialog();
  });
}

export default async function initProductsPage() {
  state.selectedIds.clear();
  state.selectedItemMeta.clear();
  state.missingDocumentItems = [];
  closeMissingDocumentsDialog();
  bindEvents();
  await loadItems({ resetOffset: true });
}
