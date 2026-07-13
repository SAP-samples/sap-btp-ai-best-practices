/* UI5 Web Components for Grounding Page */
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/TextArea.js";
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
import "@ui5/webcomponents/dist/Link.js";

import { marked } from "marked";
import { request } from "../../services/api.js";

// Hardcoded collection ID and default configuration values
const COLLECTION_ID = "9e01b5f6-6963-42ce-958e-f90f58f6bbe0";
// const DEFAULT_CUSTOM_PROMPT = `You are a precise and reliable assistant. Using only the provided context, generate a concise and accurate summary relevant to the request. Do not infer or generate information beyond the given context. If the requested information is not available in the context, clearly state that. Request: {{ ?groundingRequest }} Context: {{ ?groundingOutput }}`;

const DEFAULT_CUSTOM_PROMPT = `You are a precise and reliable assistant for extracting product information from catalog snippets provided as a JSON array.
ONLY use the provided JSON context; NEVER invent or infer details not present in the context.

PARAMS
- MIN_RESULTS = 20
- MAX_RESULTS = 200

CONTEXT FORMAT
- The context is a JSON array: [
  {
    "content": "<string>",
    "metadata": {
      "title": [...],
      "webUrlPageNo": [...],
      "webUrl": [...],
      "document_webUrl": [...],
      "source": [...]
    }
  }, ...
].

GROUPING (Within-page fusion)
- Treat each array element as a "chunk".
- You MAY merge information across multiple chunks ONLY when they share the exact same \`metadata.webUrlPageNo\` value (string-equality).
- When merging, form a "page group" of all chunks with that identical \`webUrlPageNo\`. You can use any field from any chunk inside the same page group.
- DO NOT merge across different \`webUrlPageNo\` values, even if other metadata matches.

REQUEST UNDERSTANDING (generic — any product type)
- Parse the user request into:
  • PRODUCT TERMS (chair, stool, bench, table, armchair, bar stool, visitor chair, swivel chair/armchair, cantilever, sled-base, lounge, etc.).
  • ATTRIBUTE TERMS (colors/finishes; base/frame; materials/shell; armrests/accessories; mechanisms/functions; castors/glides; dimensions; series/model codes).
- Matching is case-insensitive and robust to hyphens, newlines, and minor OCR artifacts.

PAGE-LOCAL LEXICON (no hard-coding)
- Per page group, build a local dictionary from headings/sections like: Price, Seat mm, Total mm, Backrest, Box (L x W x H), Weight, Volume, Functions/Mechanism, Armrests, Castors/Glides, Colour/Color/Finish, Table top, Wood/Veneer, Delivery, Options/Accessories.
- Learn label↔code pairs ONLY from the same page group (e.g., "chromed 2324", "white 2283", "premium white 4002A"). Do not assume mappings from other pages.

PRODUCT BOUNDARY & VARIANT DETECTION (recall-first)
- A product "variant" has a distinct model/article/order code near a spec/price block.
- Detect model codes using robust patterns: uppercase/lowercase alphanumerics (2–10 chars), possibly mixed letters/numbers (e.g., V321, 5C75, HU130, EV261, 1S346, 262S, LM740, SU271, V101K). Ignore ordinary numbers that are clearly prices, weights, volumes, page numbers, or section counters.
- When a row/column block lists several model codes (typical price/spec tables), emit ONE JSON object **per distinct model code** in that page group.
- If a page group has a clear category (e.g., “Visitor chairs”, “Bar stools”, “Lounge chair”, “Benches”, “Tables”), inherit that as the product "name" prefix when an individual row lacks a standalone label.

MATCHING & SELECTION (three-tier recall with MIN_RESULTS)
Tier 1 — Strict match:
  - Include products that match PRODUCT TERMS and ATTRIBUTE TERMS explicitly evidenced in the same page group (using the page-local lexicon).
Tier 2 — Relaxed attributes:
  - If results < MIN_RESULTS, drop ATTRIBUTE TERM filters but keep PRODUCT TERMS (type/category must still match).
Tier 3 — Any products:
  - If still < MIN_RESULTS, include products of any type from any page group (still obeying the evidence rules) until you reach MIN_RESULTS or exhaust the context.

EVIDENCE RULES (must be same page group)
- Include an item ONLY if its exact model/article/order code appears in the same page group as the other fields you output for that item.
- Attributes must be explicitly stated in the same page group. If not present, omit them (do not guess).
- Prefer structured/table values nearest to the model code. On conflicts:
  1) Prefer the value closest to the code,
  2) If still unclear, prefer the more specific/complete value (units, ranges),
  3) Otherwise, set the field to null.

ROBUST TEXT HANDLING
- Normalize broken lines/hyphenation; join wrapped tokens that belong together (e.g., “chromed\n2324” ⇒ “chromed 2324”).
- Treat repeated numeric columns common in price/spec tables cautiously; never assign a price unless it is clearly associated with the specific model row; otherwise set "price" to null.

MODEL UNIQUENESS (no duplicates across the output)
- The "model" must be **unique** per output. Maintain a global set of seen models using a normalized key: remove whitespace and convert to uppercase for comparison (e.g., "ev261", "EV 261" ⇒ key "EV261").
- Within a single page group, if the same model appears multiple times, keep only one object (choose the instance with the most complete, non-null fields; if tied, keep the earlier/upper occurrence in the content).
- If the **same model** appears in multiple page groups, keep exactly one object chosen by this priority:
  (1) More complete fields (fewer nulls) > (2) presence of a clear price bound to the model > (3) richer dimensions/specs > (4) lower page number if parseable from \`webUrlPageNo\` > (5) earlier encounter order.
- After deduplication, if results fall below MIN_RESULTS, continue selecting additional **unique** models from lower tiers or remaining pages until reaching MIN_RESULTS or exhausting the context. Never duplicate a model to pad the count.

FIELDS (Return a JSON array; use null where not available)
- "name": Series/product label as printed (e.g., "Visitor chairs — cantilever frame", "Bar stool", "Lounge table") if clearly inferable from headings/row labels in the **same page group**; else null.
- "model": Exact printed model/order/article code for the variant (never synthesize; print exactly as it appears in the chosen page group instance).
- "size": Dimensions exactly as printed with units and original separators (e.g., "74 x 77 x 124 cm" OR "Seat mm H: 410–510; W: 480; D: 480 | Total mm H: 1320; W(Ar): 580; D: 720"), else null.
- "price": Printed price string for that model (verbatim, including currency/symbols/notes) if clearly linked; else null.
- "title": For a page group, use the first non-empty string from any chunk’s \`metadata.title\`; if multiple differ, use the most frequent; else the first non-empty; else null.
- "webUrlPageNo": The exact \`metadata.webUrlPageNo\` of the page group.
- "webUrl": Prefer the first available \`metadata.document_webUrl\` from the page group; if none, use the first \`metadata.webUrl\`; else null.

ORDER & SIZE
- Within each page, preserve the natural top-to-bottom appearance when possible; across pages, sort by ascending page number if parseable from \`webUrlPageNo\`, otherwise keep encounter order.
- Return between MIN_RESULTS and MAX_RESULTS items. If more than MAX_RESULTS match, return the top MAX_RESULTS scored by: (a) Tier priority (1 > 2 > 3), then (b) stronger textual match to the request (more terms matched), then (c) proximity of attributes to the model code.

OUTPUT
- Return ONLY a valid JSON array (no preamble, no markdown). If no items satisfy the rules, return [].

Request: {{ ?groundingRequest }}
Context: {{ ?groundingOutput }}`;
const DEFAULT_CHUNK_COUNT = 150;

// Global state
let appState = {
  isLoading: false,
  errorMessage: null
};

export default function initGroundingPage() {
  console.log("Product Catalog Search page initialized");

  // Configure marked library
  marked.setOptions({
    breaks: true, // Enable line breaks
    gfm: true, // Enable GitHub Flavored Markdown
    smartLists: true,
    smartypants: true,
    headerIds: false, // Disable header IDs for security
    mangle: false // Disable email mangling
  });

  // Set up event listeners
  setupEventListeners();

  // Load initial data
  loadInitialData();
}

function setupEventListeners() {
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
    // Load files for the hardcoded collection
    await loadCollectionFiles();

    showSection("files-section");
    showSection("query-section");
  } catch (error) {
    console.error("Failed to load initial data:", error);
    let errorMessage = `Unable to load product catalogs: ${error.message}`;

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

async function loadCollectionFiles() {
  try {
    setTableLoading("files-table", true);

    // Update panel header
    const filesPanel = document.getElementById("files-panel");
    if (filesPanel) {
      filesPanel.setAttribute("header-text", "Available Product Catalogs");
    }

    // Load files for the hardcoded collection
    const files = await request(`/api/grounding/collections/${COLLECTION_ID}/files`);
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
    const count = files.length;
    const headerText = count > 0 ? `Available Product Catalogs (${count})` : "Available Product Catalogs (No Catalogs)";
    panel.setAttribute("header-text", headerText);
  }

  // Clear existing rows (except header row)
  const existingRows = table.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
  existingRows.forEach((row) => row.remove());

  if (files.length === 0) {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell colspan="3">
        <div class="table-empty">No product catalogs available</div>
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

  if (!groundingRequestInput?.value?.trim()) {
    showError("Please enter a product search query.");
    return;
  }

  const requestData = {
    grounding_request: groundingRequestInput.value.trim(),
    collection_id: COLLECTION_ID,
    custom_prompt: DEFAULT_CUSTOM_PROMPT,
    max_chunk_count: DEFAULT_CHUNK_COUNT
  };

  try {
    setSubmitLoading(true);

    const response = await request("/api/grounding/completion", "POST", requestData);

    if (response.success) {
      displayResults(response);
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
  // Try to parse LLM response as products JSON first
  let products = null;
  const responseText = response.llm_response || "No AI analysis results available.";

  try {
    // Check if the response is a JSON array of products
    const parsed = JSON.parse(responseText);
    if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].name) {
      products = parsed;
    }
  } catch (e) {
    // If not JSON, treat as regular markdown response
  }

  if (products) {
    // Display products in cards format
    displayProductCards(products);
  } else {
    // Hide products section if no products found
    const productsSection = document.getElementById("products-section");
    if (productsSection) {
      productsSection.style.display = "none";
    }
  }
}

function setMainLoading(isLoading) {
  appState.isLoading = isLoading;
  updateUIState();
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
          <ui5-text>Loading product catalogs...</ui5-text>
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

function displayProductCards(products) {
  const productsContainer = document.getElementById("products-container");
  if (!productsContainer) return;

  // Clear existing content
  productsContainer.innerHTML = "";

  // Show products section
  const productsSection = document.getElementById("products-section");
  if (productsSection) {
    productsSection.style.display = "block";
  }

  // Update the project count
  const projectCountElement = document.getElementById("project-count");
  if (projectCountElement) {
    projectCountElement.textContent = products.length;
  }

  // Create cards for each product
  products.forEach((product, index) => {
    const cardElement = createProductCard(product, index);
    productsContainer.appendChild(cardElement);
  });
}

function createProductCard(product, index) {
  const card = document.createElement("ui5-card");
  card.className = "product-card";
  card.style.marginBottom = "1rem";

  // Create card header
  const cardHeader = document.createElement("ui5-card-header");
  cardHeader.slot = "header";
  cardHeader.setAttribute("title-text", product.name || "Product");
  cardHeader.setAttribute("subtitle-text", `Model: ${product.model || "N/A"}`);

  const avatar = document.createElement("ui5-icon");
  avatar.name = "product";
  avatar.slot = "avatar";
  cardHeader.appendChild(avatar);

  card.appendChild(cardHeader);

  // Create card content
  const content = document.createElement("div");
  content.style.padding = "1rem";

  // Size information
  if (product.size) {
    const sizeDiv = document.createElement("div");
    sizeDiv.className = "product-detail";
    sizeDiv.innerHTML = `
      <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
        <ui5-icon name="dimension" style="margin-right: 0.5rem;"></ui5-icon>
        <ui5-label style="font-weight: bold;">Dimensions:</ui5-label>
      </div>
      <ui5-text style="font-size: 0.875rem; color: var(--sapNeutralColor);">${product.size}</ui5-text>
    `;
    content.appendChild(sizeDiv);
  }

  // Price information removed per user request

  // Document source
  if (product.webUrlPageNo) {
    const sourceDiv = document.createElement("div");
    sourceDiv.className = "product-detail";
    sourceDiv.style.marginTop = "1rem";

    const sourceHeader = document.createElement("div");
    sourceHeader.style.cssText = "display: flex; align-items: center; margin-bottom: 0.5rem;";
    sourceHeader.innerHTML = `
      <ui5-icon name="document-text" style="margin-right: 0.5rem;"></ui5-icon>
      <ui5-label style="font-weight: bold;">Source Catalog:</ui5-label>
    `;
    sourceDiv.appendChild(sourceHeader);

    if (product.title && product.webUrlPageNo) {
      // Extract page number from URL fragment
      let pageNumber = null;
      try {
        const url = new URL(product.webUrlPageNo);
        const pageMatch = url.hash.match(/#page=(\d+)/);
        if (pageMatch) {
          pageNumber = pageMatch[1];
        }
      } catch (e) {
        // If URL parsing fails, continue without page number
      }

      // Create inline container for title and page number
      const titleContainer = document.createElement("div");
      titleContainer.style.cssText = "margin-bottom: 0.25rem;";

      // Make the document title a clickable link
      const titleLink = document.createElement("ui5-link");
      titleLink.href = product.webUrlPageNo;
      titleLink.target = "_blank";
      let linkText = product.title;
      if (pageNumber) {
        linkText += ` (Page ${pageNumber})`;
      }
      titleLink.textContent = linkText;
      titleLink.style.cssText = "font-size: 0.875rem;";
      titleContainer.appendChild(titleLink);

      sourceDiv.appendChild(titleContainer);
    } else if (product.title) {
      // Fallback: if no webUrlPageNo, just show title as text
      const titleText = document.createElement("ui5-text");
      titleText.style.cssText = "font-size: 0.875rem; color: var(--sapNeutralColor); display: block; margin-bottom: 0.25rem;";
      titleText.textContent = product.title;
      sourceDiv.appendChild(titleText);
    }

    content.appendChild(sourceDiv);
  }

  card.appendChild(content);
  return card;
}

function showSection(sectionId) {
  const section = document.getElementById(sectionId);
  if (section) {
    section.style.display = "block";
  }
}

function showError(message) {
  updateAuthStatus(message, "Error");

  // Also log to console for debugging
  console.error(message);
}
