/* UI5 Web Components for Grounding Page */
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Option.js";
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

import { marked } from "marked";
import { request } from "../../services/api.js";

// Default custom prompt template
const DEFAULT_CUSTOM_PROMPT = `You are a precise and reliable assistant. Using only the provided context, generate a concise and accurate summary relevant to the request. Do not infer or generate information beyond the given context. If the requested information is not available in the context, clearly state that. Request: {{ ?groundingRequest }} Context: {{ ?groundingOutput }}`;

// Global state
let appState = {
  pipelines: [],
  collections: [],
  mappedCollections: [],
  selectedCollectionId: "*",
  isLoading: false,
  errorMessage: null
};

export default function initGroundingPage() {
  console.log("Grounding page initialized");

  // Configure marked library
  marked.setOptions({
    breaks: true, // Enable line breaks
    gfm: true, // Enable GitHub Flavored Markdown
    smartLists: true,
    smartypants: true,
    headerIds: false, // Disable header IDs for security
    mangle: false // Disable email mangling
  });

  // Initialize default prompt
  const customPromptTextarea = document.getElementById("custom-prompt");
  if (customPromptTextarea) {
    customPromptTextarea.value = DEFAULT_CUSTOM_PROMPT;
  }

  // Set up event listeners
  setupEventListeners();

  // Load initial data
  loadInitialData();
}

function setupEventListeners() {
  // Collection selection
  const collectionSelect = document.getElementById("collection-select");
  if (collectionSelect) {
    collectionSelect.addEventListener("change", (event) => {
      const selectedValue = event.detail.selectedOption.getAttribute("data-value") || "*";
      handleCollectionSelection(selectedValue);
    });
  }

  // Submit button
  const submitBtn = document.getElementById("submit-btn");
  if (submitBtn) {
    submitBtn.addEventListener("click", () => {
      handleGroundingCompletion();
    });
  }

  // Enter key in grounding request input
  const groundingRequestInput = document.getElementById("grounding-request");
  if (groundingRequestInput) {
    groundingRequestInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        handleGroundingCompletion();
      }
    });
  }
}

async function loadInitialData() {
  setMainLoading(true);

  try {
    // Load mapped collections (this will also fetch pipelines and collections)
    const mappedCollections = await request("/api/grounding/mapped-collections");
    appState.mappedCollections = mappedCollections;

    console.log("Loaded mapped collections:", mappedCollections);

    if (mappedCollections && mappedCollections.length > 0) {
      // Update UI
      updatePipelinesTable(mappedCollections);
      updateCollectionSelect(mappedCollections);

      showSection("pipelines-section");
      showSection("selection-section");
      showSection("query-section");
    } else {
      updateAuthStatus("No document collections found. Please ensure your SAP AI Core instance has configured document pipelines and collections.", "Warning");
      showSection("pipelines-section");
      showSection("selection-section");
      showSection("query-section");
    }
  } catch (error) {
    console.error("Failed to load initial data:", error);
    let errorMessage = `Unable to load document collections: ${error.message}`;

    // Show more detailed error information
    if (error.response) {
      console.error("Error response:", error.response);
      if (error.response.detail) {
        errorMessage = `Service Error: ${error.response.detail}`;
      } else if (error.response.error) {
        errorMessage = `Service Error: ${error.response.error}`;
      }
    }

    updateAuthStatus(errorMessage, "Error");
  } finally {
    setMainLoading(false);
  }
}

function updatePipelinesTable(mappedCollections) {
  const table = document.getElementById("pipelines-table");
  const panel = document.getElementById("pipelines-panel");

  if (!table) return;

  // Update panel header with count
  if (panel) {
    const count = mappedCollections.length;
    const headerText = count > 0 ? `Document Collections (${count})` : "Document Collections (None Available)";
    panel.setAttribute("header-text", headerText);
  }

  // Clear existing rows (except header row)
  const existingRows = table.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
  existingRows.forEach((row) => row.remove());

  if (mappedCollections.length === 0) {
    // Show empty state
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell colspan="3">
        <div class="table-empty">No document collections available. Please check your SAP AI Core configuration.</div>
      </ui5-table-cell>
    `;
    table.appendChild(row);
    return;
  }

  // Add new rows
  mappedCollections.forEach((item) => {
    const row = document.createElement("ui5-table-row");

    row.innerHTML = `
      <ui5-table-cell>${item.pipeline_path}</ui5-table-cell>
      <ui5-table-cell>${item.collection_title}</ui5-table-cell>
      <ui5-table-cell>${item.collection_id}</ui5-table-cell>
    `;

    table.appendChild(row);
  });
}

function updateCollectionSelect(mappedCollections) {
  const select = document.getElementById("collection-select");
  if (!select) return;

  // Clear existing options (except the first one)
  const existingOptions = select.querySelectorAll("ui5-option:not(:first-child)");
  existingOptions.forEach((option) => option.remove());

  // Add collection options
  mappedCollections.forEach((item) => {
    const option = document.createElement("ui5-option");
    option.setAttribute("data-value", item.collection_id);
    option.textContent = `[${item.pipeline_path}] - ${item.collection_title}`;
    select.appendChild(option);
  });
}

async function handleCollectionSelection(collectionId) {
  appState.selectedCollectionId = collectionId;

  if (collectionId === "*") {
    hideSection("files-section");
    return;
  }

  try {
    setTableLoading("files-table", true);
    showSection("files-section");

    // Update panel header
    const filesPanel = document.getElementById("files-panel");
    const selectedCollection = appState.mappedCollections.find((c) => c.collection_id === collectionId);
    if (filesPanel && selectedCollection) {
      filesPanel.setAttribute("header-text", `Documents in: ${selectedCollection.collection_title}`);
    }

    // Load files for the selected collection
    const files = await request(`/api/grounding/collections/${collectionId}/files`);
    updateFilesTable(files);
  } catch (error) {
    console.error("Failed to load collection files:", error);
    updateFilesTable([]);
    showError(`Error loading files: ${error.message}`);
  } finally {
    setTableLoading("files-table", false);
  }
}

function updateFilesTable(files) {
  const table = document.getElementById("files-table");
  const panel = document.getElementById("files-panel");

  if (!table) return;

  // Update panel header with count
  if (panel) {
    const currentHeader = panel.getAttribute("header-text") || "Documents in Collection";
    const baseHeader = currentHeader.replace(/ \(\d+.*?\)$/, ""); // Remove existing count
    const count = files.length;
    const headerText = count > 0 ? `${baseHeader} (${count})` : `${baseHeader} (No Documents)`;
    panel.setAttribute("header-text", headerText);
  }

  // Clear existing rows (except header row)
  const existingRows = table.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
  existingRows.forEach((row) => row.remove());

  if (files.length === 0) {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell colspan="3">
        <div class="table-empty">No documents available in this collection</div>
      </ui5-table-cell>
    `;
    table.appendChild(row);
    return;
  }

  // Add file rows
  files.forEach((file) => {
    const row = document.createElement("ui5-table-row");

    row.innerHTML = `
      <ui5-table-cell>${file.file_name}</ui5-table-cell>
      <ui5-table-cell>${file.indexed_timestamp}</ui5-table-cell>
      <ui5-table-cell>${file.document_id}</ui5-table-cell>
    `;

    table.appendChild(row);
  });
}

async function handleGroundingCompletion() {
  const groundingRequestInput = document.getElementById("grounding-request");
  const customPromptTextarea = document.getElementById("custom-prompt");
  const chunkCountInput = document.getElementById("chunk-count");

  if (!groundingRequestInput?.value?.trim()) {
    showError("Please enter a business question to analyze the documents.");
    return;
  }

  const requestData = {
    grounding_request: groundingRequestInput.value.trim(),
    collection_id: appState.selectedCollectionId,
    custom_prompt: customPromptTextarea?.value?.trim() || null,
    max_chunk_count: parseInt(chunkCountInput?.value) || 50
  };

  try {
    setSubmitLoading(true);

    const response = await request("/api/grounding/completion", "POST", requestData);

    if (response.success) {
      displayResults(response);
      showSection("results-section");
    } else {
      showError(`Grounding completion failed: ${response.error}`);
    }
  } catch (error) {
    console.error("Failed to execute grounding completion:", error);
    showError(`Error: ${error.message}`);
  } finally {
    setSubmitLoading(false);
  }
}

function displayResults(response) {
  // Display LLM response with markdown rendering
  const llmResponseDiv = document.getElementById("llm-response");
  if (llmResponseDiv) {
    const responseText = response.llm_response || "No AI analysis results available.";
    try {
      // Parse and render markdown
      const htmlContent = marked.parse(responseText);
      llmResponseDiv.innerHTML = htmlContent;
    } catch (error) {
      console.error("Error parsing markdown:", error);
      // Fallback to plain text if markdown parsing fails
      llmResponseDiv.textContent = responseText;
    }
  }

  // Display grounding result
  const groundingResultTextarea = document.getElementById("grounding-result");
  if (groundingResultTextarea) {
    let groundingText = response.grounding_result || "No document context available.";

    // Try to format JSON if it's a JSON string
    try {
      const parsed = JSON.parse(groundingText);
      groundingText = JSON.stringify(parsed, null, 2);
    } catch (e) {
      // If not JSON, use as-is
    }

    groundingResultTextarea.value = groundingText;
  }
}

function setMainLoading(isLoading) {
  appState.isLoading = isLoading;
  updateUIState();
}

function setLoadingState(isLoading) {
  // Legacy function for compatibility - delegates to setMainLoading
  setMainLoading(isLoading);
}

function setSubmitLoading(isLoading) {
  const submitBtn = document.getElementById("submit-btn");
  const loadingIndicator = document.getElementById("loading-indicator");

  if (submitBtn) {
    submitBtn.disabled = isLoading;
  }

  if (loadingIndicator) {
    loadingIndicator.style.display = isLoading ? "block" : "none";
  }
}

function setTableLoading(tableId, isLoading) {
  const table = document.getElementById(tableId);
  if (!table) return;

  if (isLoading) {
    // Clear existing rows (except header row) and show loading
    const existingRows = table.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
    existingRows.forEach((row) => row.remove());

    const loadingRow = document.createElement("ui5-table-row");
    loadingRow.innerHTML = `
      <ui5-table-cell colspan="3">
        <div style="text-align: center; padding: 1rem;">
          <ui5-busy-indicator active></ui5-busy-indicator>
          <ui5-text>Loading documents...</ui5-text>
        </div>
      </ui5-table-cell>
    `;
    table.appendChild(loadingRow);
  }
}

function updateAuthStatus(message, type = "Information") {
  // Only store error messages in state
  if (type === "Error") {
    appState.errorMessage = message;
  } else {
    appState.errorMessage = null;
  }
  updateUIState();
}

function updateUIState() {
  // Get DOM elements once
  const mainLoading = document.getElementById("main-loading");
  const authStatus = document.getElementById("auth-status");
  const dataStatus = document.getElementById("data-status");

  if (!dataStatus) return;

  // Update loading indicator
  if (mainLoading) {
    if (appState.isLoading) {
      mainLoading.setAttribute("active", "");
    } else {
      mainLoading.removeAttribute("active");
    }
  }

  // Update error message
  if (authStatus) {
    if (appState.errorMessage) {
      authStatus.design = "Negative";
      authStatus.textContent = appState.errorMessage;
      authStatus.style.display = "block";
    } else {
      authStatus.style.display = "none";
    }
  }

  // Update data-status visibility based on state
  const shouldShowDataStatus = appState.isLoading || appState.errorMessage;
  dataStatus.style.display = shouldShowDataStatus ? "flex" : "none";
}

function showSection(sectionId) {
  const section = document.getElementById(sectionId);
  if (section) {
    section.style.display = "block";
  }
}

function hideSection(sectionId) {
  const section = document.getElementById(sectionId);
  if (section) {
    section.style.display = "none";
  }
}

function showError(message) {
  updateAuthStatus(message, "Error");

  // Also log to console for debugging
  console.error(message);
}
