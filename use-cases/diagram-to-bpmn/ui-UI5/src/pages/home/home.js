/* Home page UI5 components */
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/Title.js";

import { checkHealth, generateBpmn } from "../../services/api.js";

const DEFAULT_MODEL = "gemini-2.5-pro";
const PREVIEW_CHARACTER_LIMIT = 6000;

const MODEL_OPTIONS = {
  "gemini-2.5-pro": "Gemini 2.5 Pro"
};

const MODEL_TO_PROVIDER = {
  "gemini-2.5-pro": "gemini"
};

const state = {
  bpmnXML: "",
  meta: null,
  error: ""
};

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function prettifyUsageKey(key) {
  return key
    .replace(/_/g, " ")
    .replace(/tokens/gi, " tokens")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function formatPrimitive(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toLocaleString() : String(value);
  }

  if (typeof value === "bigint") {
    return value.toString();
  }

  return String(value);
}

function renderUsageEntry(key, value, level = 0) {
  const indentClass = `token-level-${Math.min(level, 4)}`;
  const label = escapeHtml(prettifyUsageKey(key));

  if (isRecord(value)) {
    const entries = Object.entries(value);
    if (!entries.length) {
      return "";
    }

    const childrenHtml = entries
      .map(([childKey, childValue]) => renderUsageEntry(childKey, childValue, level + 1))
      .filter(Boolean)
      .join("");

    if (!childrenHtml) {
      return "";
    }

    return `<div class="token-group ${indentClass}">
      <span class="group-label">${label}</span>
      <div class="token-group-body">
        ${childrenHtml}
      </div>
    </div>`;
  }

  if (Array.isArray(value)) {
    if (!value.length) {
      return "";
    }

    const childrenHtml = value
      .map((item, index) => renderUsageEntry(`Entry ${index + 1}`, item, level + 1))
      .filter(Boolean)
      .join("");

    if (!childrenHtml) {
      return "";
    }

    return `<div class="token-group ${indentClass}">
      <span class="group-label">${label}</span>
      <div class="token-group-body">
        ${childrenHtml}
      </div>
    </div>`;
  }

  const formattedValue = escapeHtml(formatPrimitive(value));
  return `<div class="token-row ${indentClass}">
    <span class="label">${label}</span>
    <span class="value">${formattedValue}</span>
  </div>`;
}

function renderUsageEntries(usage) {
  return Object.entries(usage)
    .map(([key, value]) => renderUsageEntry(key, value))
    .filter(Boolean)
    .join("");
}

function setMessage(strip, text = "", design = "Information") {
  if (!strip) return;
  if (!text) {
    strip.hidden = true;
    strip.textContent = "";
    return;
  }

  strip.design = design;
  strip.textContent = text;
  strip.hidden = false;
}

function renderDetails(detailsContainer, detailsPanel) {
  if (!detailsContainer || !detailsPanel) {
    return;
  }

  if (!state.meta) {
    detailsContainer.innerHTML =
      "<ui5-text>Generation statistics will appear here after a successful run.</ui5-text>";
    detailsPanel.collapsed = true;
    return;
  }

  const modelLabel = MODEL_OPTIONS[state.meta.model] || state.meta.model || "Unknown";

  const detailRows = [
    `<div class="detail-row"><span class="label">Model</span><span class="value">${escapeHtml(
      modelLabel
    )}</span></div>`,
    `<div class="detail-row"><span class="label">Success</span><span class="value">${
      state.meta.success ? "Yes" : "No"
    }</span></div>`
  ];

  // const usage = state.meta.usage;
  // if (usage && typeof usage === "object") {
  //   const usageHtml = renderUsageEntries(usage);
  //   if (usageHtml) {
  //     detailRows.push(
  //       `<div class="token-list"><span class="label token-heading">Token Usage</span>${usageHtml}</div>`
  //     );
  //   }
  // }

  detailsContainer.innerHTML = detailRows.join("");
  detailsPanel.collapsed = false;
}

function renderPreview(previewTextarea, xmlPanel) {
  if (!previewTextarea || !xmlPanel) {
    return;
  }

  if (!state.bpmnXML) {
    previewTextarea.value = "";
    xmlPanel.collapsed = true;
    return;
  }

  let previewText = state.bpmnXML;
  if (state.bpmnXML.length > PREVIEW_CHARACTER_LIMIT) {
    previewText =
      state.bpmnXML.slice(0, PREVIEW_CHARACTER_LIMIT) +
      "\n... (truncated for preview, download the file for the full BPMN XML)";
  }

  previewTextarea.value = previewText;
  xmlPanel.collapsed = false;
}

function renderResult(resultStrip, downloadButton, detailsContainer, detailsPanel, previewTextarea, xmlPanel) {
  if (state.error) {
    setMessage(resultStrip, state.error, "Negative");
  } else if (state.bpmnXML) {
    setMessage(resultStrip, "BPMN XML generated successfully. Download is ready.", "Positive");
  } else {
    setMessage(resultStrip, "", "Information");
  }

  if (downloadButton) {
    downloadButton.disabled = !state.bpmnXML;
  }

  renderDetails(detailsContainer, detailsPanel);
  renderPreview(previewTextarea, xmlPanel);
}

function resetState() {
  state.bpmnXML = "";
  state.meta = null;
  state.error = "";
}

export default function initHomePage() {
  const fileInput = document.getElementById("diagram-input");
  const fileSelectButton = document.getElementById("file-select-button");
  const fileSelectedText = document.getElementById("file-selected-text");

  const generateButton = document.getElementById("generate-button");
  const resetButton = document.getElementById("reset-button");
  const downloadButton = document.getElementById("download-button");

  const feedbackStrip = document.getElementById("feedback-strip");
  const resultStrip = document.getElementById("result-strip");

  const detailsPanel = document.getElementById("details-panel");
  const detailsContainer = document.getElementById("generation-details");
  const xmlPanel = document.getElementById("xml-panel");
  const previewTextarea = document.getElementById("bpmn-preview");

  /*
  const healthButton = document.getElementById("health-button");
  const healthPanel = document.getElementById("health-panel");
  const healthResponse = document.getElementById("health-response");
  */

  if (
    !fileInput ||
    !fileSelectButton ||
    !generateButton ||
    !resetButton ||
    !downloadButton ||
    !feedbackStrip ||
    !resultStrip ||
    !detailsPanel ||
    !detailsContainer ||
    !xmlPanel ||
    !previewTextarea ||
    /*
    !healthButton ||
    !healthPanel ||
    !healthResponse ||
    */
    !fileSelectedText
  ) {
    console.error("Home page elements missing, cannot initialize UI");
    return;
  }

  function getSelectedFile() {
    return fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
  }

  function updateGenerateButtonState() {
    const hasFile = Boolean(getSelectedFile());
    generateButton.disabled = !hasFile || generateButton.loading;
  }

  function clearFeedback() {
    setMessage(feedbackStrip, "");
  }

  /*
  async function handleHealthCheck() {
    clearFeedback();
    setMessage(resultStrip, "", "Information");

    healthButton.loading = true;
    healthResponse.textContent = "Checking API health...";
    healthPanel.collapsed = false;

    try {
      const healthData = await checkHealth();
      healthResponse.textContent = JSON.stringify(healthData, null, 2);

      if (healthData && healthData.status === "healthy") {
        setMessage(feedbackStrip, "API server is healthy.", "Positive");
      } else {
        setMessage(feedbackStrip, "API health check returned a warning.", "Warning");
      }
    } catch (error) {
      healthResponse.textContent = `Error: ${error.message}`;
      setMessage(
        feedbackStrip,
        `Failed to contact API: ${error.message || "Unknown error"}`,
        "Negative"
      );
    } finally {
      healthButton.loading = false;
    }
  }
  */

  function updateFileState() {
    const selectedFile = getSelectedFile();
    if (selectedFile) {
      fileSelectedText.textContent = `${selectedFile.name} (${Math.round(selectedFile.size / 1024)} KB)`;
    } else {
      fileSelectedText.textContent = "No file selected";
    }
    updateGenerateButtonState();
  }

  function handleDownload() {
    if (!state.bpmnXML) {
      return;
    }

    const blob = new Blob([state.bpmnXML], { type: "application/xml" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "generated_process.bpmn";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  async function handleGenerate() {
    clearFeedback();
    state.error = "";
    setMessage(resultStrip, "", "Information");

    const selectedFile = getSelectedFile();
    if (!selectedFile) {
      setMessage(feedbackStrip, "Please upload a process diagram before generating.", "Warning");
      return;
    }

    const model = DEFAULT_MODEL;
    const provider = MODEL_TO_PROVIDER[model] || "gemini";

    generateButton.loading = true;
    updateGenerateButtonState();
    downloadButton.disabled = true;
    setMessage(resultStrip, "Generating BPMN XML. This can take a few minutes...", "Information");

    try {
      const response = await generateBpmn({
        file: selectedFile,
        filename: selectedFile.name,
        provider,
        model
      });

      if (response.success) {
        state.bpmnXML = response.bpmn_xml || "";
        state.meta = {
          provider: response.provider || provider,
          model: response.model || model,
          success: Boolean(response.success),
          usage: response.usage || null
        };
        state.error = "";
        setMessage(resultStrip, "BPMN XML generated successfully. Download is ready.", "Positive");
      } else {
        state.bpmnXML = "";
        state.meta = {
          provider,
          model,
          success: false,
          usage: response.usage || null
        };
        state.error = response.error || "Generation failed without error details.";
        setMessage(resultStrip, state.error, "Negative");
      }
    } catch (error) {
      state.bpmnXML = "";
      state.meta = {
        provider,
        model,
        success: false,
        usage: null
      };
      state.error = error.message || "Failed to generate BPMN XML.";
      setMessage(resultStrip, state.error, "Negative");
    } finally {
      generateButton.loading = false;
      updateGenerateButtonState();
      renderResult(resultStrip, downloadButton, detailsContainer, detailsPanel, previewTextarea, xmlPanel);
    }
  }

  function handleReset() {
    clearFeedback();
    resetState();
    setMessage(resultStrip, "", "Information");

    fileInput.value = "";
    updateFileState();

    renderResult(resultStrip, downloadButton, detailsContainer, detailsPanel, previewTextarea, xmlPanel);
    // healthPanel.collapsed = true;
  }

  updateGenerateButtonState();
  renderResult(resultStrip, downloadButton, detailsContainer, detailsPanel, previewTextarea, xmlPanel);

  fileSelectButton.addEventListener("click", () => {
    clearFeedback();
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    updateFileState();
    clearFeedback();
  });

  generateButton.addEventListener("click", handleGenerate);
  resetButton.addEventListener("click", handleReset);
  downloadButton.addEventListener("click", handleDownload);
  // healthButton.addEventListener("click", handleHealthCheck);
}
