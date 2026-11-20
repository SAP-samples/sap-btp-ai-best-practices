import { uploadSources as uploadSelectedRecords } from "../services/ecn-api.js";
import { element } from "./workspace-dom.js";

/** Create a request/response client for the short-lived ECN parsing worker. */
export function createImportWorkerClient() {
  const worker = new Worker(new URL("./ecn-import-worker.js", import.meta.url), { type: "module" });
  const pending = new Map();
  let sequence = 0;

  worker.addEventListener("message", ({ data }) => {
    const request = pending.get(data.id);
    if (!request) return;
    pending.delete(data.id);
    if (data.error) request.reject(new Error(data.error));
    else request.resolve(data.result);
  });
  worker.addEventListener("error", (event) => {
    for (const request of pending.values()) request.reject(new Error(event.message || "ECN parsing failed."));
    pending.clear();
  });

  /** Send one correlated request to the worker. */
  const send = (type, payload = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, ...payload });
  });

  return {
    parse: (file) => send("parse", { file }),
    preview: (options) => send("preview", { options }),
    serialize: (positions) => send("serialize", { positions }),
    terminate: () => {
      worker.terminate();
      for (const request of pending.values()) request.reject(new DOMException("Import closed", "AbortError"));
      pending.clear();
    },
  };
}

/** Bind one fresh, selected-only ECN import lifecycle to the inbox dialog. */
export function bindEcnImportDialog({
  root,
  signal,
  imported,
  uploadSources = uploadSelectedRecords,
  createWorkerClient = createImportWorkerClient,
  uuid = () => crypto.randomUUID(),
}) {
  const dialog = root.querySelector("#import-dialog");
  const openButton = root.querySelector("#open-import");
  const fileInput = root.querySelector("#import-file");
  const searchInput = root.querySelector("#import-search");
  const rows = root.querySelector("#import-rows");
  const status = root.querySelector("#import-status");
  const selectionSummary = root.querySelector("#selection-summary");
  const pageSummary = root.querySelector("#import-page-summary");
  const previous = root.querySelector("#previous-import-page");
  const next = root.querySelector("#next-import-page");
  const selectAll = root.querySelector("#select-all-import");
  const deselectAll = root.querySelector("#deselect-all-import");
  const submit = root.querySelector("#import-selected");
  const cancel = root.querySelector("#cancel-import");
  const listeners = [];
  let client = null;
  let fileName = "";
  let total = 0;
  let page = 1;
  let pages = 1;
  let selected = new Set();
  let pendingBatches = [];
  let importedCount = 0;
  let busy = false;
  let previewSequence = 0;

  /** Register one event listener with its matching cleanup. */
  const on = (node, type, handler) => {
    node.addEventListener(type, handler);
    listeners.push(() => node.removeEventListener(type, handler));
  };

  /** Update selection counts and import-button availability. */
  const showSelection = () => {
    selectionSummary.textContent = `${selected.size} selected of ${total}`;
    submit.disabled = busy || selected.size === 0;
  };

  /** Enable or disable every control that can invalidate an active upload. */
  const setBusy = (value) => {
    busy = value;
    fileInput.disabled = value;
    cancel.disabled = value;
    searchInput.disabled = value || total === 0;
    selectAll.disabled = value || total === 0;
    deselectAll.disabled = value || total === 0;
    previous.disabled = value || total === 0 || page <= 1;
    next.disabled = value || total === 0 || page >= pages;
    showSelection();
  };

  /** Render the current worker-owned preview page using text-only cells. */
  const renderPreview = async () => {
    if (!client || total === 0) return;
    const sequence = ++previewSequence;
    const result = await client.preview({ query: searchInput.value, page, pageSize: 25 });
    if (sequence !== previewSequence || !client) return;
    page = result.page;
    pages = result.pages;
    rows.replaceChildren();
    for (const entry of result.entries) {
      const row = element("tr");
      const checkbox = element("input");
      checkbox.type = "checkbox";
      checkbox.checked = selected.has(entry.position);
      checkbox.setAttribute("aria-label", `Select ${entry.ecnNumber} for import`);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) selected.add(entry.position);
        else selected.delete(entry.position);
        pendingBatches = [];
        importedCount = 0;
        showSelection();
      });
      for (const value of [checkbox, entry.position, entry.ecnNumber, entry.description, entry.lifecycle]) {
        const cell = element("td");
        if (value instanceof Node) cell.append(value);
        else cell.textContent = String(value);
        row.append(cell);
      }
      rows.append(row);
    }
    pageSummary.textContent = `Page ${page} of ${pages} · ${result.matching} matching ECNs`;
    previous.disabled = busy || page <= 1;
    next.disabled = busy || page >= pages;
    showSelection();
  };

  /** Forget all file and selection state and terminate its worker. */
  const reset = () => {
    previewSequence += 1;
    client?.terminate();
    client = null;
    fileName = "";
    total = 0;
    page = 1;
    pages = 1;
    selected = new Set();
    pendingBatches = [];
    importedCount = 0;
    busy = false;
    fileInput.value = "";
    searchInput.value = "";
    rows.replaceChildren();
    status.textContent = "";
    pageSummary.textContent = "No file selected";
    setBusy(false);
  };

  /** Close the dialog and release its complete in-memory source. */
  const close = () => {
    reset();
    if (typeof dialog.close === "function" && dialog.open) dialog.close();
    else dialog.removeAttribute("open");
  };

  /** Open a brand-new import session with no prior selection. */
  const open = () => {
    reset();
    client = createWorkerClient();
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    fileInput.focus();
  };

  /** Parse a chosen file entirely inside the current worker. */
  const loadFile = async () => {
    const file = fileInput.files?.[0];
    if (!file || !file.name.toLocaleLowerCase().endsWith(".json")) {
      status.textContent = "Select a JSON ECN file.";
      return;
    }
    try {
      setBusy(true);
      total = 0;
      selected = new Set();
      pendingBatches = [];
      importedCount = 0;
      rows.replaceChildren();
      showSelection();
      status.textContent = "Reading ECNs…";
      const result = await client.parse(file);
      fileName = file.name;
      total = result.total;
      selected = new Set();
      pendingBatches = [];
      importedCount = 0;
      page = 1;
      status.textContent = total ? "Select the ECNs to import." : "This file contains no ECNs.";
      setBusy(false);
      await renderPreview();
    } catch (error) {
      setBusy(false);
      status.textContent = error.message;
    }
  };

  /** Upload selected records in retry-safe batches while retaining failed work. */
  const importSelected = async () => {
    if (busy || selected.size === 0) return;
    if (pendingBatches.length === 0) {
      const positions = [...selected].sort((left, right) => left - right);
      pendingBatches = [];
      for (let offset = 0; offset < positions.length; offset += 25) {
        pendingBatches.push({ positions: positions.slice(offset, offset + 25), requestId: uuid(), dataGroup: root.querySelector("#import-data-group")?.value || "customer", assessmentDate: root.querySelector("#import-assessment-date")?.value || "" });
      }
      importedCount = 0;
    }
    setBusy(true);
    try {
      while (pendingBatches.length) {
        const batch = pendingBatches[0];
        status.textContent = `Importing ${importedCount + 1}–${importedCount + batch.positions.length} of ${selected.size}…`;
        const content = await client.serialize(batch.positions);
        const ecnFile = new File([content], fileName, { type: "application/json" });
        await uploadSources({ requestId: batch.requestId, ecnFile, signal, dataGroup: batch.dataGroup, assessmentDate: batch.assessmentDate });
        importedCount += batch.positions.length;
        pendingBatches.shift();
      }
      close();
      await imported();
    } catch (error) {
      if (error.name !== "AbortError") {
        status.textContent = `${importedCount} ECNs imported. Retry to continue with the failed and remaining batches: ${error.message}`;
      }
    } finally {
      if (client) setBusy(false);
    }
  };

  on(openButton, "click", open);
  on(fileInput, "change", () => void loadFile());
  on(searchInput, "input", () => { page = 1; void renderPreview(); });
  on(previous, "click", () => { page -= 1; void renderPreview(); });
  on(next, "click", () => { page += 1; void renderPreview(); });
  on(selectAll, "click", () => {
    selected = new Set(Array.from({ length: total }, (_, index) => index + 1));
    pendingBatches = [];
    importedCount = 0;
    showSelection();
    void renderPreview();
  });
  on(deselectAll, "click", () => {
    selected.clear();
    pendingBatches = [];
    importedCount = 0;
    showSelection();
    void renderPreview();
  });
  on(submit, "click", () => void importSelected());
  on(cancel, "click", close);
  on(dialog, "cancel", (event) => { event.preventDefault(); if (!busy) close(); });
  signal.addEventListener("abort", close, { once: true });
  reset();

  /** Detach the component and discard any worker-owned source. */
  return () => {
    signal.removeEventListener("abort", close);
    listeners.forEach((remove) => remove());
    close();
  };
}
