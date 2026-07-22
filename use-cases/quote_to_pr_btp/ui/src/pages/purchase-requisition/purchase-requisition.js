import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/FileUploader.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/ProgressIndicator.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents-fiori/dist/Wizard.js";

import { request, uploadFiles } from "../../services/api.js";

const RECOMMENDED_PROFILE = {
  mode: "shortlist",
  include_docai: false,
  include_llm: true,
  selected_llm_models: ["gemini-2.5-flash"],
  selected_llm_scenarios: ["detailed_static_prompt"],
  selected_docai_scenarios: [],
  approach_profile: "customer_fast_extraction"
};

let activeRunId = "";
let activeDocument = "";
let pollTimer = null;
let isProcessing = false;

function showMessage(message, design = "Information") {
  const strip = document.getElementById("one-click-message");
  if (!strip) return;
  strip.hidden = false;
  strip.design = design;
  strip.textContent = message;
}

function clearMessage() {
  const strip = document.getElementById("one-click-message");
  if (strip) strip.hidden = true;
}

function selectWizardStep(stepId) {
  const stepIds = ["wizard-upload", "wizard-validate", "wizard-complete"];
  const selectedIndex = stepIds.indexOf(stepId);
  stepIds.forEach((id, index) => {
    const step = document.getElementById(id);
    if (!step) return;
    step.disabled = index > selectedIndex;
    step.selected = id === stepId;
  });
}

function showOnly(state) {
  const success = document.getElementById("one-click-success");
  const review = document.getElementById("one-click-review");
  if (success) success.hidden = state !== "success";
  if (review) review.hidden = state !== "review";
  if (state === "progress") selectWizardStep("wizard-validate");
  else if (["success", "review"].includes(state)) selectWizardStep("wizard-complete");
  else selectWizardStep("wizard-upload");
}

function updateProgress(status) {
  const step = Math.max(1, Number(status?.step || 1));
  const total = Math.max(step, Number(status?.total_steps || 6));
  const percent = Math.min(100, Math.max(8, Math.round((step / total) * 100)));
  const title = document.getElementById("one-click-progress-title");
  const text = document.getElementById("one-click-progress-text");
  const indicator = document.getElementById("one-click-progress");
  if (title) title.textContent = status?.stage || "Processing the quote";
  if (text) text.textContent = status?.message || "Validating the extracted values.";
  if (indicator) {
    indicator.value = percent;
    indicator.displayValue = `${percent}%`;
  }
}

function approachRequest(detail) {
  const approach = detail?.best_approach || {};
  return {
    run_id: activeRunId,
    document_name: activeDocument,
    method_family: approach.approach_family,
    model: approach.model,
    strategy: approach.strategy,
    overrides: {}
  };
}

function reviewReasons(detail, prepared) {
  const reasons = [];
  const approach = detail?.best_approach || {};
  const extracted = detail?.extracted || {};
  const header = extracted.header || {};
  const lineItems = Array.isArray(extracted.line_items) ? extracted.line_items : [];
  const warnings = Array.isArray(extracted.warnings) ? extracted.warnings : [];
  const missing = prepared?.purchase_requisition?.missing_fields || [];

  if (approach.status && approach.status !== "success") reasons.push("The extraction did not complete cleanly");
  if (approach.confidence !== null && approach.confidence !== undefined && approach.confidence !== "" && Number.isFinite(Number(approach.confidence)) && Number(approach.confidence) < 75) reasons.push("The extraction confidence needs review");
  if (!header.vendor_name) reasons.push("Supplier name was not confirmed");
  if (!lineItems.length) reasons.push("No line items were confirmed");
  if (warnings.length) reasons.push("The extraction reported a warning");
  if (missing.length) reasons.push(...missing.map((field) => `Missing ${field}`));
  return [...new Set(reasons)];
}

function openReview(reasons = []) {
  const message = reasons.length
    ? `${reasons.slice(0, 2).join(". ")}.`
    : "A required value could not be confirmed from the quote.";
  const text = document.getElementById("one-click-review-text");
  if (text) text.textContent = message;
  showOnly("review");
}

async function finishCompletedRun() {
  const overview = await request(`/api/purchase-requisition/overview?run_id=${encodeURIComponent(activeRunId)}`);
  activeDocument = overview?.documents?.[0] || activeDocument;
  if (!activeDocument) throw new Error("No extracted document was published by the runner.");

  const detail = await request(`/api/purchase-requisition/documents/${encodeURIComponent(activeDocument)}?run_id=${encodeURIComponent(activeRunId)}`);
  const body = approachRequest(detail);
  const prepared = await request("/api/purchase-requisition/s4/purchase-requisition/payload", "POST", body);
  const reasons = reviewReasons(detail, prepared);
  if (!prepared?.s4_integration_enabled) {
    openReview(["Extraction completed. S/4HANA creation is not configured in this deployment"]);
    return;
  }
  if (reasons.length || !prepared?.purchase_requisition?.ready_for_create) {
    openReview(reasons);
    return;
  }

  updateProgress({ step: 6, total_steps: 6, stage: "Creating purchase requisition", message: "Sending the validated values to S/4HANA." });
  const created = await request("/api/purchase-requisition/s4/purchase-requisition/create", "POST", { ...body, confirm_create: true });
  const prNumber = created?.s4_result?.purchase_requisition;
  if (!prNumber) throw new Error("S/4HANA accepted the request but did not return the created PR number.");
  document.getElementById("one-click-pr-number").textContent = prNumber;
  showOnly("success");
}

async function pollRunner() {
  if (!activeRunId) return;
  try {
    const status = await request(`/api/purchase-requisition/runs/${encodeURIComponent(activeRunId)}/status`);
    updateProgress(status);
    if (status.status === "completed") {
      window.clearTimeout(pollTimer);
      await finishCompletedRun();
      isProcessing = false;
      return;
    }
    if (["error", "stale"].includes(status.status)) throw new Error(status.message || "Processing stopped before completion.");
    pollTimer = window.setTimeout(pollRunner, 2500);
  } catch (error) {
    isProcessing = false;
    showOnly("");
    showMessage(`The quote could not be completed: ${error.message}`, "Negative");
  }
}

async function processQuote() {
  if (isProcessing) return;
  const uploader = document.getElementById("one-click-file");
  const file = uploader?.files?.[0];
  if (!file) {
    showMessage("Choose one PDF before processing the quote.", "Critical");
    return;
  }

  isProcessing = true;
  clearMessage();
  showOnly("progress");
  updateProgress({ step: 1, total_steps: 6, stage: "Uploading document", message: file.name });
  try {
    const upload = await uploadFiles("/api/purchase-requisition/documents/upload", [file]);
    activeDocument = upload?.documents?.[0]?.file_name || file.name;
    const experimentName = `${new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 12)}_one_click`;
    const run = await request("/api/purchase-requisition/research", "POST", {
      ...RECOMMENDED_PROFILE,
      experiment_name: experimentName,
      document_names: [activeDocument]
    });
    activeRunId = run.run_id;
    if (!run.live_runner_available) throw new Error("The extraction runner is not available.");
    updateProgress(run.runner_status || { step: 1, total_steps: 6, stage: "Extraction started", message: "Selecting the recommended extraction approach." });
    await pollRunner();
  } catch (error) {
    isProcessing = false;
    showOnly("");
    showMessage(`The quote could not be started: ${error.message}`, "Negative");
  }
}

function resetPage() {
  const uploader = document.getElementById("one-click-file");
  if (uploader) uploader.value = "";
  activeRunId = "";
  activeDocument = "";
  isProcessing = false;
  showOnly("");
  clearMessage();
  document.getElementById("one-click-file-title").textContent = "No PDF selected";
  document.getElementById("one-click-file-note").textContent = "Choose the quote you want to convert into a PR.";
  document.getElementById("one-click-run").disabled = true;
}

export default function initOneClickPage() {
  const uploader = document.getElementById("one-click-file");
  const runButton = document.getElementById("one-click-run");
  uploader?.addEventListener("change", (event) => {
    const file = event.detail?.files?.[0] || uploader.files?.[0];
    document.getElementById("one-click-file-title").textContent = file?.name || "No PDF selected";
    document.getElementById("one-click-file-note").textContent = file
      ? `${Math.max(1, Math.round(file.size / 1024))} KB - ready to process`
      : "Choose the quote you want to convert into a PR.";
    runButton.disabled = !file;
    showOnly("");
    clearMessage();
  });
  runButton?.addEventListener("click", processQuote);
  document.getElementById("one-click-another")?.addEventListener("click", resetPage);
  document.getElementById("one-click-open-review")?.addEventListener("click", () => {
    const params = new URLSearchParams();
    if (activeRunId) params.set("run_id", activeRunId);
    if (activeDocument) params.set("document", activeDocument);
    window.pageRouter.navigate(`/extraction?${params.toString()}`);
  });
}
