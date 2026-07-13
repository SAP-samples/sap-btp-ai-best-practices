/* Pharma Procurement Sales Order Agent page components */
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/CheckBox.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Title.js";

import { request } from "../../services/api.js";

let latestPharmaOrderRequestId = 0;

function clampNumber(value, fallback, min, max) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) return fallback;
  return Math.min(Math.max(parsed, min), max);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(statusStrip, message, design = "Information") {
  if (!statusStrip) return;
  statusStrip.hidden = false;
  statusStrip.design = design;
  statusStrip.textContent = message;
}

function asList(value) {
  return Array.isArray(value) ? value : [];
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function getToolPayload(toolResult) {
  if (toolResult?.content_json) return toolResult.content_json;
  const preview = toolResult?.content_preview;
  if (!preview || typeof preview !== "string") return null;
  try {
    return JSON.parse(preview);
  } catch (error) {
    return null;
  }
}

function summarizeRecord(record = {}) {
  const priorityKeys = [
    "customer_name",
    "customer_id",
    "material_name",
    "material_id",
    "material_number",
    "ndc",
    "sales_order",
    "sales_order_id",
    "po_number",
    "invoice_id",
    "billing_document",
    "batch_id",
    "status",
    "overall_status",
    "delivery_status",
    "net_price",
    "currency",
    "available_quantity",
    "expiry_date"
  ];
  const pairs = priorityKeys
    .filter((key) => record[key] !== undefined && record[key] !== null && record[key] !== "")
    .slice(0, 6)
    .map((key) => `${key}: ${record[key]}`);
  if (pairs.length) return pairs.join(" | ");
  const fallback = Object.entries(record)
    .filter(([, value]) => ["string", "number", "boolean"].includes(typeof value))
    .slice(0, 5)
    .map(([key, value]) => `${key}: ${value}`);
  return fallback.join(" | ") || "No compact evidence preview available.";
}

function summarizeToolPayload(payload) {
  const groupedResults = asList(payload?.data?.results);
  const sourceFiles = unique(groupedResults.map((item) => item.file));
  const matchedRecords = groupedResults.reduce((sum, item) => sum + Number(item.match_count || 0), 0);
  const evidence = [];

  groupedResults.forEach((group) => {
    asList(group.matches).slice(0, 2).forEach((match) => {
      if (match?.record) {
        evidence.push({ dataset: group.dataset, text: summarizeRecord(match.record) });
      }
    });
  });

  return {
    tool: payload?.tool || "Unknown tool",
    purpose: payload?.purpose || "No purpose returned.",
    dataStatus: payload?.data_status || "unknown",
    assumptions: asList(payload?.assumptions),
    sourceFiles,
    matchedRecords,
    evidence: evidence.slice(0, 4)
  };
}

function renderTraceCards(result) {
  const toolCalls = asList(result.tool_calls);
  const toolResults = asList(result.tool_results);

  if (!toolCalls.length && !toolResults.length) {
    return `<div class="order-muted">No trace was returned. Enable "Show tool trace" and try again.</div>`;
  }

  const header = `
    <div class="order-trace-summary">
      <div><strong>Model</strong><span>${escapeHtml(result.model || "N/A")}</span></div>
      <div><strong>Provider</strong><span>${escapeHtml(result.provider || "N/A")}</span></div>
      <div><strong>Tool calls</strong><span>${escapeHtml(result.tool_call_count ?? 0)}</span></div>
    </div>
  `;

  const callCards = toolCalls.map((call, index) => {
    const args = call.args ? Object.entries(call.args).map(([key, value]) => `${key}: ${value}`).join(" | ") : "No arguments";
    return `
      <div class="order-trace-card">
        <div class="order-trace-card-title">${index + 1}. ${escapeHtml(call.name || "Unknown tool")}</div>
        <div class="order-chip-row"><span class="order-chip">Tool call</span></div>
        <div class="order-evidence-line">${escapeHtml(args)}</div>
      </div>
    `;
  }).join("");

  const resultCards = toolResults.map((toolResult, index) => {
    const payload = getToolPayload(toolResult);
    if (!payload) {
      return `
        <div class="order-trace-card">
          <div class="order-trace-card-title">${index + 1}. ${escapeHtml(toolResult.name || toolResult.tool_call_id || "Tool result")}</div>
          <div class="order-muted">Could not parse structured tool result. Preview:</div>
          <div class="order-evidence-line">${escapeHtml(toolResult.content_preview || "No preview.")}</div>
        </div>
      `;
    }

    const summary = summarizeToolPayload(payload);
    const files = summary.sourceFiles.length ? summary.sourceFiles : ["No source file returned"];
    const assumptions = summary.assumptions.length ? summary.assumptions : ["No assumptions returned"];
    const evidence = summary.evidence.length ? summary.evidence : [{ dataset: "N/A", text: "No matching records preview returned." }];

    return `
      <div class="order-trace-card">
        <div class="order-trace-card-title">${index + 1}. ${escapeHtml(summary.tool)}</div>
        <div class="order-trace-purpose">${escapeHtml(summary.purpose)}</div>
        <div class="order-chip-row">
          <span class="order-chip">${escapeHtml(summary.dataStatus)}</span>
          <span class="order-chip">matched records: ${escapeHtml(summary.matchedRecords)}</span>
        </div>
        <div class="order-trace-section"><strong>Source files</strong>${files.map((file) => `<span>${escapeHtml(file)}</span>`).join("")}</div>
        <div class="order-trace-section"><strong>Assumptions</strong>${assumptions.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>
        <div class="order-trace-section"><strong>Evidence preview</strong>${evidence.map((item) => `<span>${escapeHtml(item.dataset)}: ${escapeHtml(item.text)}</span>`).join("")}</div>
      </div>
    `;
  }).join("");

  return `${header}<div class="order-trace-grid">${callCards}${resultCards}</div>`;
}

function renderCapabilities(capabilitiesContainer, noteStrip, payload) {
  const capabilities = asList(payload.capabilities);
  if (noteStrip && payload.ui_note) {
    noteStrip.hidden = false;
    noteStrip.textContent = payload.ui_note;
  }

  if (!capabilities.length) {
    capabilitiesContainer.innerHTML = `<div class="order-muted">No capabilities returned by the backend.</div>`;
    return;
  }

  capabilitiesContainer.innerHTML = capabilities.map((capability) => {
    const example = asList(capability.example_questions)[0] || payload.default_question || "";
    const examples = asList(capability.example_questions).slice(0, 2).map((question) => `<li>${escapeHtml(question)}</li>`).join("");
    const tools = asList(capability.expected_tools).map((tool) => `<span class="order-chip">${escapeHtml(tool)}</span>`).join("");
    const sources = asList(capability.source_structures).slice(0, 4).map((source) => `<span>${escapeHtml(source)}</span>`).join("");
    return `
      <button class="order-capability-card" type="button" data-example-question="${escapeHtml(example)}">
        <span class="order-capability-title">${escapeHtml(capability.title)}</span>
        <span class="order-capability-description">${escapeHtml(capability.description)}</span>
        <span class="order-capability-label">Expected tools</span>
        <span class="order-chip-row">${tools}</span>
        <span class="order-capability-label">Source structures</span>
        <span class="order-source-list">${sources}</span>
        <span class="order-capability-label">Examples</span>
        <ul>${examples}</ul>
      </button>
    `;
  }).join("");
}

export default function initPharmaOrderPage() {
  const questionInput = document.getElementById("pharma-order-question");
  const maxTokensInput = document.getElementById("pharma-order-max-tokens");
  const includeTraceInput = document.getElementById("pharma-order-include-trace");
  const submitButton = document.getElementById("pharma-order-submit-button");
  const statusStrip = document.getElementById("pharma-order-status");
  const responsePanel = document.getElementById("pharma-order-response-panel");
  const responseContainer = document.getElementById("pharma-order-response");
  const tracePanel = document.getElementById("pharma-order-trace-panel");
  const traceContainer = document.getElementById("pharma-order-trace");
  const capabilitiesContainer = document.getElementById("pharma-order-capabilities");
  const capabilitiesNote = document.getElementById("pharma-order-capabilities-note");

  if (!questionInput || !submitButton || !responsePanel || !responseContainer || !tracePanel || !traceContainer) {
    return;
  }

  const loadCapabilities = async () => {
    if (!capabilitiesContainer) return;
    try {
      const payload = await request("/api/pharma-order/capabilities");
      renderCapabilities(capabilitiesContainer, capabilitiesNote, payload);
      if (payload.default_question && !questionInput.value) {
        questionInput.value = payload.default_question;
      }
      capabilitiesContainer.querySelectorAll("[data-example-question]").forEach((card) => {
        card.addEventListener("click", () => {
          questionInput.value = card.getAttribute("data-example-question") || "";
          questionInput.focus();
        });
      });
    } catch (error) {
      capabilitiesContainer.innerHTML = `<div class="order-muted">Could not load capabilities: ${escapeHtml(error.message)}</div>`;
    }
  };

  const handleAsk = async () => {
    const requestId = ++latestPharmaOrderRequestId;
    const question = questionInput.value?.trim();
    if (!question) {
      responsePanel.collapsed = false;
      responseContainer.textContent = "Please type a service representative question first.";
      setStatus(statusStrip, "Question is required.", "Warning");
      return;
    }

    const includeTrace = Boolean(includeTraceInput?.checked);
    const maxTokens = clampNumber(maxTokensInput?.value || "900", 900, 128, 4000);

    submitButton.loading = true;
    responsePanel.collapsed = false;
    tracePanel.collapsed = false;
    responseContainer.textContent = "Pharma Procurement Sales Order Agent is resolving names, planning tools, and consolidating evidence...";
    traceContainer.innerHTML = `<div class="order-muted">Waiting for readable LangGraph tool trace...</div>`;
    setStatus(statusStrip, "Calling /api/pharma-order/ask on the FastAPI backend...", "Information");

    try {
      const result = await request("/api/pharma-order/ask", "POST", {
        question,
        temperature: 0.2,
        max_tokens: maxTokens,
        prompt_variant: "joule",
        include_trace: includeTrace
      });

      if (requestId !== latestPharmaOrderRequestId) return;

      if (!result.success) {
        responseContainer.textContent = `Error: ${result.error || "Could not get a response."}`;
        traceContainer.innerHTML = `<div class="order-muted">No tool trace loaded.</div>`;
        setStatus(statusStrip, "Backend returned an unsuccessful response.", "Negative");
        return;
      }

      responseContainer.textContent = result.answer || result.markdown || "No answer generated.";
      traceContainer.innerHTML = renderTraceCards(result);
      setStatus(statusStrip, `Answered with ${result.tool_call_count ?? 0} tool call(s).`, "Positive");
    } catch (error) {
      if (requestId !== latestPharmaOrderRequestId) return;
      responseContainer.textContent = `Error: ${error.message}`;
      traceContainer.innerHTML = `<div class="order-muted">No tool trace loaded.</div>`;
      setStatus(statusStrip, "Request failed. Check backend logs and VITE_API_BASE_URL / VITE_API_KEY.", "Negative");
    } finally {
      if (requestId === latestPharmaOrderRequestId) {
        submitButton.loading = false;
      }
    }
  };

  loadCapabilities();
  submitButton.addEventListener("click", handleAsk);
  questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleAsk();
    }
  });
}
