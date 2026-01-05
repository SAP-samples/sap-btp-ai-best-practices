/* UI5 Web Components for Grounding Page */
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Table.js";
import "@ui5/webcomponents/dist/TableRow.js";
import "@ui5/webcomponents/dist/TableCell.js";
import "@ui5/webcomponents/dist/TableHeaderRow.js";
import "@ui5/webcomponents/dist/TableHeaderCell.js";

import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Link.js";
import "@ui5/webcomponents/dist/SegmentedButton.js";
import "@ui5/webcomponents/dist/SegmentedButtonItem.js";

/* UI5 Icons */
import "@ui5/webcomponents-icons/dist/ai.js";
import "@ui5/webcomponents-icons/dist/arrow-right.js";
import "@ui5/webcomponents-icons/dist/document-text.js";
import "@ui5/webcomponents-icons/dist/activity-2.js";
import "@ui5/webcomponents-icons/dist/product.js";
import "@ui5/webcomponents-icons/dist/dimension.js";
import "@ui5/webcomponents-icons/dist/accept.js";
import "@ui5/webcomponents-icons/dist/decline.js";

import { marked } from "marked";
import { request, API_BASE_URL } from "../../services/api.js";

// Hardcoded collection ID and default configuration values
const COLLECTION_ID = "9e01b5f6-6963-42ce-958e-f90f58f6bbe0";
// const DEFAULT_CUSTOM_PROMPT = `You are a precise and reliable assistant. Using only the provided context, generate a concise and accurate summary relevant to the request. Do not infer or generate information beyond the given context. If the requested information is not available in the context, clearly state that. Request: {{ ?groundingRequest }} Context: {{ ?groundingOutput }}`;

// const DEFAULT_CUSTOM_PROMPT = `You are a precise and reliable assistant for extracting product information from catalog snippets provided as a JSON array.
// ONLY use the provided JSON context; NEVER invent or infer details not present in the context.

// CONTEXT FORMAT
// - The context is a JSON array: [
//   {
//     "content": "<string>",
//     "metadata": {
//       "title": [...],
//       "webUrlPageNo": [...],
//       "webUrl": [...],
//       "document_webUrl": [...],
//       "source": [...]
//     }
//   }, ...
// ].

// GROUPING (Within-page fusion)
// - Treat each array element as a "chunk".
// - You MAY merge information across multiple chunks ONLY when they share the **exact same** \`metadata.webUrlPageNo\` value (string-equality).
// - When merging, form a "page group" consisting of all chunks with that identical \`webUrlPageNo\`. You can use any field from any chunk inside the same page group.
// - DO NOT merge across different \`webUrlPageNo\` values, even if other metadata matches.

// TASK
// - Return a JSON array of products that match the user’s request.
// - Include an item ONLY if the exact model/order/article code (e.g., "162S", "262S", "EV311", "1S366", "LM740") is evidenced within the SAME page group as the other fields you output for that item.

// CATALOG RULES
// - Prefer structured evidence (tables/rows, spec blocks) over prose.
// - Preserve original language for names, series, finishes, and model IDs (do not translate).
// - When a page shows multiple variants as separate rows, output ONE JSON object per variant (per distinct model code).
// - If a field is not explicit for that model anywhere in the SAME page group, set it to null.
// - If the same model appears multiple times within a page group, output just one object. On conflicting values inside the page group:
// - Prefer the value printed closest (in the same row/block) to the model code.
// - If proximity is unclear, prefer the more specific/complete value (e.g., includes units).
// - If still ambiguous, use null.
// - NEVER pull fields from a different page group.

// FIELDS (Return a JSON array; use null where not available)
// - "name": Product/series name as printed (e.g., "Silver", "EVERYis1", "AIMis1"), optionally a subtype if clearly stated (e.g., "Swivel chair medium high").
// - "model": Exact printed model/order/article code for the variant (e.g., "262S", "EV311", "1S366"). Never synthesize.
// - "size": Dimensions exactly as printed with units and original separators (e.g., "74 x 77 x 124 cm", or "H: 400-525; W: 500; D: 470"); null if not explicit for that model within the page group.
// - "price": Printed price string for that model including currency and any notes present in the SAME page group; do NOT normalize or convert; if unclear which price belongs to the model, use null.
// - "title": For a page group, use the first string from any chunk’s \`metadata.title\` if present; if multiple differ, use the most frequent; else the first non-empty; else null.
// - "webUrlPageNo": The exact \`metadata.webUrlPageNo\` string of the page group (do not parse or alter).
// - "webUrl": Prefer the first available URL from \`metadata.document_webUrl\` from any chunk in the page group; if none, use the first from \`metadata.webUrl\`; else null.

// OUTPUT
// - Return ONLY a valid JSON array (no preamble, no markdown). If no items satisfy the rules, return [].

// Request: {{ ?groundingRequest }}
// Context: {{ ?groundingOutput }}`;

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
const DEFAULT_CHUNK_COUNT = 50;
// const DEFAULT_CHUNK_COUNT = 5;

const vendorCatalogMapping = [
  {
    vendor: "EliteWork Furnishings",
    contract: "EliteWork_Furnishings_contract.pdf",
    catalogs: ["EliteWork_Furnishings_Product_Catalog.pdf"],
    exhaustionRate: "38%"
  },
  {
    vendor: "NovaWork Interiors",
    contract: "NovaWork_Interiors_contract.pdf",
    catalog: ["NovaWork_Interiors_Product_Catalog.pdf"],
    exhaustionRate: "61%"
  }
];

// Global state
let appState = {
  isLoading: false,
  errorMessage: null,
  contractAnalysisCache: new Map() // Cache for contract analysis results
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

  // View toggle buttons
  const cardsViewBtn = document.getElementById("cards-view");
  const tableViewBtn = document.getElementById("table-view");

  if (cardsViewBtn) {
    cardsViewBtn.addEventListener("click", () => {
      switchToCardsView();
    });
  }

  if (tableViewBtn) {
    tableViewBtn.addEventListener("click", () => {
      switchToTableView();
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

  // Clear previous results immediately when starting new search
  clearProductResults();

  const requestData = {
    grounding_request: groundingRequestInput.value.trim(),
    collection_id: COLLECTION_ID,
    custom_prompt: DEFAULT_CUSTOM_PROMPT,
    max_chunk_count: DEFAULT_CHUNK_COUNT
  };

  try {
    setSubmitLoading(true);

    console.log("requestData", requestData);
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
      products = parsed.map((product) => {
        if (product.webUrlPageNo) {
          product.webUrlPageNo = transformSharepointUrl(product.webUrlPageNo);
        }
        return product;
      });
    }
  } catch (e) {
    // If not JSON, treat as regular markdown response
  }

  if (products) {
    // Display products in both cards and table formats
    displayProductCards(products);
    displayProductTable(products);
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

function getContractInfo(productTitle) {
  if (!productTitle) return null;

  // Search through vendor catalog mapping to find matching catalog
  for (const mapping of vendorCatalogMapping) {
    // Check if the product title matches any of the catalogs for this vendor
    const catalogs = mapping.catalogs || [mapping.catalog]; // Handle both 'catalogs' array and single 'catalog' property

    for (const catalog of catalogs) {
      if (catalog && productTitle.includes(catalog)) {
        return {
          vendor: mapping.vendor,
          contract: mapping.contract,
          exhaustionRate: mapping.exhaustionRate
        };
      }
    }
  }

  return null;
}

async function displayProductCards(products) {
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

  // Create cards for each product immediately (with loading states)
  products.forEach((product, index) => {
    const cardElement = createProductCard(product, index);
    productsContainer.appendChild(cardElement);
  });

  // Group products by contract for bulk analysis
  const productsByContract = new Map();

  products.forEach((product, index) => {
    const contractInfo = getContractInfo(product.title);
    if (contractInfo) {
      const contractName = contractInfo.contract.replace(".pdf", ""); // Remove .pdf extension
      if (!productsByContract.has(contractName)) {
        productsByContract.set(contractName, []);
      }
      productsByContract.get(contractName).push({ product, index });
    }
  });

  // Perform bulk contract analysis for each contract (in parallel)
  const analysisPromises = Array.from(productsByContract.entries()).map(async ([contractName, productItems]) => {
    try {
      const productsForAnalysis = productItems.map((item) => ({
        name: item.product.name || "Unknown Product",
        description: `Model: ${item.product.model || "N/A"}${item.product.size ? `, Dimensions: ${item.product.size}` : ""}`
      }));

      const analysisResult = await analyzeContractBulk(contractName, productsForAnalysis);

      // Store results in cache and update UI
      if (analysisResult && analysisResult.results) {
        analysisResult.results.forEach((result, idx) => {
          if (idx < productItems.length) {
            const cacheKey = `${contractName}:${productItems[idx].product.name}:${productItems[idx].product.model}`;
            appState.contractAnalysisCache.set(cacheKey, result);

            // Update the specific product card with analysis results
            updateProductCardAnalysis(productItems[idx].product, productItems[idx].index, result);
          }
        });
      } else {
        // Handle analysis failure - update all products for this contract
        productItems.forEach((item) => {
          updateProductCardAnalysisError(item.product, item.index, "Analysis failed");
        });
      }
    } catch (error) {
      console.error(`Failed to analyze contract ${contractName}:`, error);
      // Update all products for this contract with error state
      productItems.forEach((item) => {
        updateProductCardAnalysisError(item.product, item.index, error.message);
      });
    }
  });

  // Don't wait for analyses to complete - let them update asynchronously
  Promise.all(analysisPromises).then(() => {
    console.log("All contract analyses completed");
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
        const url = new URL(product.webUrlPageNo, window.location.origin);
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

  // Contract information (after Source Catalog)
  const contractInfo = getContractInfo(product.title);
  if (contractInfo) {
    const contractDiv = document.createElement("div");
    contractDiv.className = "product-detail";
    contractDiv.style.marginTop = "1rem";

    const contractHeader = document.createElement("div");
    contractHeader.style.cssText = "display: flex; align-items: center; margin-bottom: 0.5rem;";
    contractHeader.innerHTML = `
      <ui5-icon name="activity-2" style="margin-right: 0.5rem;"></ui5-icon>
      <ui5-label style="font-weight: bold;">Contract:</ui5-label>
    `;
    contractDiv.appendChild(contractHeader);

    const vendorText = document.createElement("ui5-text");
    vendorText.style.cssText = "font-size: 0.875rem; color: var(--sapNeutralColor); display: block; margin-bottom: 0.25rem;";
    vendorText.textContent = `Vendor: ${contractInfo.vendor}`;
    contractDiv.appendChild(vendorText);

    // Make the contract clickable
    const contractContainer = document.createElement("div");
    contractContainer.style.cssText = "margin-bottom: 0.25rem;";

    const contractLink = document.createElement("ui5-link");
    contractLink.href = `${API_BASE_URL}/api/storage/contracts/${encodeURIComponent(contractInfo.contract)}`;
    contractLink.target = "_blank";
    contractLink.textContent = contractInfo.contract;
    contractLink.style.cssText = "font-size: 0.875rem;";
    contractContainer.appendChild(contractLink);

    contractDiv.appendChild(contractContainer);

    // Contract Analysis Results
    const analysisResult = getContractAnalysisResult(product);
    if (analysisResult) {
      const analysisContainer = document.createElement("div");
      analysisContainer.style.cssText =
        "margin-top: 0.5rem; padding: 0.5rem; background-color: var(--sapObjectHeader_Background); border-radius: 0.25rem; border: 1px solid var(--sapField_BorderColor);";

      // Purchase Support Status
      const purchaseStatusDiv = document.createElement("div");
      purchaseStatusDiv.style.cssText = "display: flex; align-items: center; margin-bottom: 0.25rem;";

      const purchaseIcon = document.createElement("ui5-icon");
      purchaseIcon.name = analysisResult.supports_purchase ? "accept" : "decline";
      purchaseIcon.style.cssText = `margin-right: 0.25rem; color: ${analysisResult.supports_purchase ? "var(--sapPositiveColor)" : "var(--sapNegativeColor)"};`;

      const purchaseText = document.createElement("ui5-text");
      purchaseText.style.cssText = `font-size: 0.75rem; color: ${analysisResult.supports_purchase ? "var(--sapPositiveColor)" : "var(--sapNegativeColor)"};`;
      purchaseText.textContent = analysisResult.supports_purchase ? "Purchase Supported" : "Purchase Not Supported";

      purchaseStatusDiv.appendChild(purchaseIcon);
      purchaseStatusDiv.appendChild(purchaseText);
      analysisContainer.appendChild(purchaseStatusDiv);

      // Contract Validity Status
      const validityStatusDiv = document.createElement("div");
      validityStatusDiv.style.cssText = "display: flex; align-items: center; margin-bottom: 0.25rem;";

      const validityIcon = document.createElement("ui5-icon");
      validityIcon.name = analysisResult.is_valid ? "accept" : "decline";
      validityIcon.style.cssText = `margin-right: 0.25rem; color: ${analysisResult.is_valid ? "var(--sapPositiveColor)" : "var(--sapNegativeColor)"};`;

      const validityText = document.createElement("ui5-text");
      validityText.style.cssText = `font-size: 0.75rem; color: ${analysisResult.is_valid ? "var(--sapPositiveColor)" : "var(--sapNegativeColor)"};`;
      validityText.textContent = analysisResult.is_valid ? "Contract Valid" : "Contract Expired";

      validityStatusDiv.appendChild(validityIcon);
      validityStatusDiv.appendChild(validityText);
      analysisContainer.appendChild(validityStatusDiv);

      // Validity End Date (if available)
      if (analysisResult.validity_end_date) {
        const endDateDiv = document.createElement("div");
        endDateDiv.style.cssText = "margin-bottom: 0.25rem;";

        const endDateText = document.createElement("ui5-text");
        endDateText.style.cssText = "font-size: 0.75rem; color: var(--sapNeutralColor);";
        endDateText.textContent = `Valid until: ${analysisResult.validity_end_date}`;

        endDateDiv.appendChild(endDateText);
        analysisContainer.appendChild(endDateDiv);
      }

      // Exhaustion Rate (if available)
      if (contractInfo.exhaustionRate) {
        const exhaustionRateDiv = document.createElement("div");
        exhaustionRateDiv.style.cssText = "margin-bottom: 0.25rem;";

        const exhaustionRateText = document.createElement("ui5-text");
        exhaustionRateText.style.cssText = "font-size: 0.75rem; color: var(--sapNeutralColor);";
        exhaustionRateText.textContent = `Exhaustion rate: ${contractInfo.exhaustionRate}`;

        exhaustionRateDiv.appendChild(exhaustionRateText);
        analysisContainer.appendChild(exhaustionRateDiv);
      }

      // // Confidence Score
      // const confidenceDiv = document.createElement("div");
      // confidenceDiv.style.cssText = "margin-bottom: 0.25rem;";

      // const confidenceText = document.createElement("ui5-text");
      // confidenceText.style.cssText = "font-size: 0.75rem; color: var(--sapNeutralColor);";
      // const confidencePercent = Math.round(analysisResult.confidence_score * 100);
      // confidenceText.textContent = `Confidence: ${confidencePercent}%`;

      // confidenceDiv.appendChild(confidenceText);
      // analysisContainer.appendChild(confidenceDiv);

      // Analysis Reasoning (collapsible)
      if (analysisResult.reasoning) {
        const reasoningDiv = document.createElement("div");
        reasoningDiv.style.cssText = "margin-top: 0.5rem;";

        const reasoningToggle = document.createElement("ui5-link");
        reasoningToggle.textContent = "View Analysis Details";
        reasoningToggle.style.cssText = "font-size: 0.75rem; cursor: pointer;";

        const reasoningContent = document.createElement("div");
        reasoningContent.style.cssText =
          "display: none; margin-top: 0.25rem; padding: 0.25rem; background-color: var(--sapField_Background); border-radius: 0.25rem; font-size: 0.75rem; color: var(--sapNeutralColor);";
        reasoningContent.textContent = analysisResult.reasoning;

        reasoningToggle.addEventListener("click", () => {
          const isHidden = reasoningContent.style.display === "none";
          reasoningContent.style.display = isHidden ? "block" : "none";
          reasoningToggle.textContent = isHidden ? "Hide Analysis Details" : "View Analysis Details";
        });

        reasoningDiv.appendChild(reasoningToggle);
        reasoningDiv.appendChild(reasoningContent);
        analysisContainer.appendChild(reasoningDiv);
      }

      contractDiv.appendChild(analysisContainer);
    } else {
      // Show loading indicator for contract analysis
      const loadingDiv = document.createElement("div");
      loadingDiv.setAttribute("data-contract-loading", "true");
      loadingDiv.style.cssText =
        "margin-top: 0.5rem; padding: 0.5rem; background-color: var(--sapObjectHeader_Background); border-radius: 0.25rem; border: 1px solid var(--sapField_BorderColor); text-align: center;";

      const loadingIndicator = document.createElement("ui5-busy-indicator");
      loadingIndicator.setAttribute("active", "");
      loadingIndicator.setAttribute("delay", "0");
      loadingIndicator.setAttribute("size", "S");
      loadingIndicator.style.cssText = "margin-right: 0.5rem;";

      // <ui5-busy-indicator active id="main-loading" delay="0" text="Connecting to document collections..."></ui5-busy-indicator>;

      const loadingText = document.createElement("ui5-text");
      loadingText.style.cssText = "font-size: 0.75rem; color: var(--sapNeutralColor);";
      loadingText.textContent = "Analyzing contract...";

      loadingDiv.appendChild(loadingIndicator);
      loadingDiv.appendChild(loadingText);
      contractDiv.appendChild(loadingDiv);
    }

    content.appendChild(contractDiv);
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

async function analyzeContractBulk(contractName, products) {
  try {
    const requestData = {
      contract_name: contractName,
      products: products,
      current_date: new Date().toISOString().split("T")[0], // Current date in YYYY-MM-DD format
      temperature: 0.1,
      max_tokens: 3000
    };

    const response = await request("/api/contract-analysis/analyze-bulk", "POST", requestData);
    return response;
  } catch (error) {
    console.error(`Contract analysis failed for ${contractName}:`, error);
    return null;
  }
}

function getContractAnalysisResult(product) {
  const contractInfo = getContractInfo(product.title);
  if (!contractInfo) return null;

  const contractName = contractInfo.contract.replace(".pdf", "");
  const cacheKey = `${contractName}:${product.name}:${product.model}`;

  return appState.contractAnalysisCache.get(cacheKey) || null;
}

function updateProductCardAnalysis(product, productIndex, analysisResult) {
  // Find the product card by index
  const productsContainer = document.getElementById("products-container");
  if (!productsContainer) return;

  const productCards = productsContainer.querySelectorAll(".product-card");
  if (productIndex >= productCards.length) return;

  const productCard = productCards[productIndex];

  // Get contract info for exhaustion rate
  const contractInfo = getContractInfo(product.title);

  // Find the contract analysis loading div and replace it with results
  const loadingDiv = productCard.querySelector('[data-contract-loading="true"]');
  if (!loadingDiv) {
    // If no loading div found, try to find the contract div and update it
    const contractDiv = productCard.querySelector(".product-detail:last-child");
    if (contractDiv) {
      // Remove existing loading content
      const existingLoading = contractDiv.querySelector("ui5-busy-indicator");
      if (existingLoading) {
        existingLoading.parentElement.remove();
      }
    } else {
      return;
    }
  } else {
    loadingDiv.remove();
  }

  // Find the contract div to add analysis results
  const contractDiv = productCard.querySelector(".product-detail:last-child");
  if (!contractDiv) return;

  // Create analysis results container
  const analysisContainer = document.createElement("div");
  analysisContainer.style.cssText =
    "margin-top: 0.5rem; padding: 0.5rem; background-color: var(--sapObjectHeader_Background); border-radius: 0.25rem; border: 1px solid var(--sapField_BorderColor);";

  // Purchase Support Status
  const purchaseStatusDiv = document.createElement("div");
  purchaseStatusDiv.style.cssText = "display: flex; align-items: center; margin-bottom: 0.25rem;";

  const purchaseIcon = document.createElement("ui5-icon");
  purchaseIcon.name = analysisResult.supports_purchase ? "accept" : "decline";
  purchaseIcon.style.cssText = `margin-right: 0.25rem; color: ${analysisResult.supports_purchase ? "var(--sapPositiveColor)" : "var(--sapNegativeColor)"};`;

  const purchaseText = document.createElement("ui5-text");
  purchaseText.style.cssText = `font-size: 0.75rem; color: ${analysisResult.supports_purchase ? "var(--sapPositiveColor)" : "var(--sapNegativeColor)"};`;
  purchaseText.textContent = analysisResult.supports_purchase ? "Purchase Supported" : "Purchase Not Supported";

  purchaseStatusDiv.appendChild(purchaseIcon);
  purchaseStatusDiv.appendChild(purchaseText);
  analysisContainer.appendChild(purchaseStatusDiv);

  // Contract Validity Status
  const validityStatusDiv = document.createElement("div");
  validityStatusDiv.style.cssText = "display: flex; align-items: center; margin-bottom: 0.25rem;";

  const validityIcon = document.createElement("ui5-icon");
  validityIcon.name = analysisResult.is_valid ? "accept" : "decline";
  validityIcon.style.cssText = `margin-right: 0.25rem; color: ${analysisResult.is_valid ? "var(--sapPositiveColor)" : "var(--sapNegativeColor)"};`;

  const validityText = document.createElement("ui5-text");
  validityText.style.cssText = `font-size: 0.75rem; color: ${analysisResult.is_valid ? "var(--sapPositiveColor)" : "var(--sapNegativeColor)"};`;
  validityText.textContent = analysisResult.is_valid ? "Contract Valid" : "Contract Expired";

  validityStatusDiv.appendChild(validityIcon);
  validityStatusDiv.appendChild(validityText);
  analysisContainer.appendChild(validityStatusDiv);

  // Validity End Date (if available)
  if (analysisResult.validity_end_date) {
    const endDateDiv = document.createElement("div");
    endDateDiv.style.cssText = "margin-bottom: 0.25rem;";

    const endDateText = document.createElement("ui5-text");
    endDateText.style.cssText = "font-size: 0.75rem; color: var(--sapNeutralColor);";
    endDateText.textContent = `Valid until: ${analysisResult.validity_end_date}`;

    endDateDiv.appendChild(endDateText);
    analysisContainer.appendChild(endDateDiv);
  }

  // Exhaustion Rate (if available)
  if (contractInfo && contractInfo.exhaustionRate) {
    const exhaustionRateDiv = document.createElement("div");
    exhaustionRateDiv.style.cssText = "margin-bottom: 0.25rem;";

    const exhaustionRateText = document.createElement("ui5-text");
    exhaustionRateText.style.cssText = "font-size: 0.75rem; color: var(--sapNeutralColor);";
    exhaustionRateText.textContent = `Exhaustion rate: ${contractInfo.exhaustionRate}`;

    exhaustionRateDiv.appendChild(exhaustionRateText);
    analysisContainer.appendChild(exhaustionRateDiv);
  }

  // // Confidence Score
  // const confidenceDiv = document.createElement("div");
  // confidenceDiv.style.cssText = "margin-bottom: 0.25rem;";

  // const confidenceText = document.createElement("ui5-text");
  // confidenceText.style.cssText = "font-size: 0.75rem; color: var(--sapNeutralColor);";
  // const confidencePercent = Math.round(analysisResult.confidence_score * 100);
  // confidenceText.textContent = `Confidence: ${confidencePercent}%`;

  // confidenceDiv.appendChild(confidenceText);
  // analysisContainer.appendChild(confidenceDiv);

  // Analysis Reasoning (collapsible)
  if (analysisResult.reasoning) {
    const reasoningDiv = document.createElement("div");
    reasoningDiv.style.cssText = "margin-top: 0.5rem;";

    const reasoningToggle = document.createElement("ui5-link");
    reasoningToggle.textContent = "View Analysis Details";
    reasoningToggle.style.cssText = "font-size: 0.75rem; cursor: pointer;";

    const reasoningContent = document.createElement("div");
    reasoningContent.style.cssText =
      "display: none; margin-top: 0.25rem; padding: 0.25rem; background-color: var(--sapField_Background); border-radius: 0.25rem; font-size: 0.75rem; color: var(--sapNeutralColor);";
    reasoningContent.textContent = analysisResult.reasoning;

    reasoningToggle.addEventListener("click", () => {
      const isHidden = reasoningContent.style.display === "none";
      reasoningContent.style.display = isHidden ? "block" : "none";
      reasoningToggle.textContent = isHidden ? "Hide Analysis Details" : "View Analysis Details";
    });

    reasoningDiv.appendChild(reasoningToggle);
    reasoningDiv.appendChild(reasoningContent);
    analysisContainer.appendChild(reasoningDiv);
  }

  contractDiv.appendChild(analysisContainer);
}

function updateProductCardAnalysisError(product, productIndex, errorMessage) {
  // Find the product card by index
  const productsContainer = document.getElementById("products-container");
  if (!productsContainer) return;

  const productCards = productsContainer.querySelectorAll(".product-card");
  if (productIndex >= productCards.length) return;

  const productCard = productCards[productIndex];

  // Find the contract analysis loading div and replace it with error
  const loadingDiv = productCard.querySelector('[data-contract-loading="true"]');
  if (!loadingDiv) {
    // If no loading div found, try to find the contract div and update it
    const contractDiv = productCard.querySelector(".product-detail:last-child");
    if (contractDiv) {
      // Remove existing loading content
      const existingLoading = contractDiv.querySelector("ui5-busy-indicator");
      if (existingLoading) {
        existingLoading.parentElement.remove();
      }
    } else {
      return;
    }
  } else {
    loadingDiv.remove();
  }

  // Find the contract div to add error message
  const contractDiv = productCard.querySelector(".product-detail:last-child");
  if (!contractDiv) return;

  // Create error container
  const errorContainer = document.createElement("div");
  errorContainer.style.cssText =
    "margin-top: 0.5rem; padding: 0.5rem; background-color: var(--sapErrorBackground); border-radius: 0.25rem; border: 1px solid var(--sapNegativeColor); text-align: center;";

  const errorIcon = document.createElement("ui5-icon");
  errorIcon.name = "decline";
  errorIcon.style.cssText = "margin-right: 0.5rem; color: var(--sapNegativeColor);";

  const errorText = document.createElement("ui5-text");
  errorText.style.cssText = "font-size: 0.75rem; color: var(--sapNegativeColor);";
  errorText.textContent = `Contract analysis failed: ${errorMessage}`;

  errorContainer.appendChild(errorIcon);
  errorContainer.appendChild(errorText);
  contractDiv.appendChild(errorContainer);
}

function clearProductResults() {
  // Clear the products container (cards view)
  const productsContainer = document.getElementById("products-container");
  if (productsContainer) {
    productsContainer.innerHTML = "";
  }

  // Clear the products table (table view)
  const productsTable = document.getElementById("products-table");
  if (productsTable) {
    const existingRows = productsTable.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
    existingRows.forEach((row) => row.remove());
  }

  // Hide the products section
  const productsSection = document.getElementById("products-section");
  if (productsSection) {
    productsSection.style.display = "none";
  }

  // Clear the contract analysis cache
  appState.contractAnalysisCache.clear();

  // Reset the project count
  const projectCountElement = document.getElementById("project-count");
  if (projectCountElement) {
    projectCountElement.textContent = "0";
  }
}

function switchToCardsView() {
  const productsContainer = document.getElementById("products-container");
  const productsTableContainer = document.getElementById("products-table-container");
  const cardsViewBtn = document.getElementById("cards-view");
  const tableViewBtn = document.getElementById("table-view");

  if (productsContainer) {
    productsContainer.style.display = "grid";
  }
  if (productsTableContainer) {
    productsTableContainer.style.display = "none";
  }
  if (cardsViewBtn) {
    cardsViewBtn.pressed = true;
  }
  if (tableViewBtn) {
    tableViewBtn.pressed = false;
  }
}

function switchToTableView() {
  const productsContainer = document.getElementById("products-container");
  const productsTableContainer = document.getElementById("products-table-container");
  const cardsViewBtn = document.getElementById("cards-view");
  const tableViewBtn = document.getElementById("table-view");

  if (productsContainer) {
    productsContainer.style.display = "none";
  }
  if (productsTableContainer) {
    productsTableContainer.style.display = "block";
  }
  if (cardsViewBtn) {
    cardsViewBtn.pressed = false;
  }
  if (tableViewBtn) {
    tableViewBtn.pressed = true;
  }
}

async function displayProductTable(products) {
  const productsTable = document.getElementById("products-table");
  if (!productsTable) return;

  // Clear existing rows (except header row)
  const existingRows = productsTable.querySelectorAll("ui5-table-row:not([slot='headerRow'])");
  existingRows.forEach((row) => row.remove());

  if (products.length === 0) {
    const row = document.createElement("ui5-table-row");
    row.innerHTML = `
      <ui5-table-cell colspan="7">
        <div class="table-empty">No products found</div>
      </ui5-table-cell>
    `;
    productsTable.appendChild(row);
    return;
  }

  // Group products by contract for bulk analysis
  const productsByContract = new Map();

  products.forEach((product, index) => {
    const contractInfo = getContractInfo(product.title);
    if (contractInfo) {
      const contractName = contractInfo.contract.replace(".pdf", ""); // Remove .pdf extension
      if (!productsByContract.has(contractName)) {
        productsByContract.set(contractName, []);
      }
      productsByContract.get(contractName).push({ product, index });
    }
  });

  // Create table rows for each product immediately (with loading states)
  products.forEach((product, index) => {
    const row = createProductTableRow(product, index);
    productsTable.appendChild(row);
  });

  // Perform bulk contract analysis for each contract (in parallel)
  const analysisPromises = Array.from(productsByContract.entries()).map(async ([contractName, productItems]) => {
    try {
      const productsForAnalysis = productItems.map((item) => ({
        name: item.product.name || "Unknown Product",
        description: `Model: ${item.product.model || "N/A"}${item.product.size ? `, Dimensions: ${item.product.size}` : ""}`
      }));

      const analysisResult = await analyzeContractBulk(contractName, productsForAnalysis);

      // Store results in cache and update UI
      if (analysisResult && analysisResult.results) {
        analysisResult.results.forEach((result, idx) => {
          if (idx < productItems.length) {
            const cacheKey = `${contractName}:${productItems[idx].product.name}:${productItems[idx].product.model}`;
            appState.contractAnalysisCache.set(cacheKey, result);

            // Update the specific table row with analysis results
            updateProductTableRowAnalysis(productItems[idx].product, productItems[idx].index, result);
          }
        });
      } else {
        // Handle analysis failure - update all products for this contract
        productItems.forEach((item) => {
          updateProductTableRowAnalysisError(item.product, item.index, "Analysis failed");
        });
      }
    } catch (error) {
      console.error(`Failed to analyze contract ${contractName}:`, error);
      // Update all products for this contract with error state
      productItems.forEach((item) => {
        updateProductTableRowAnalysisError(item.product, item.index, error.message);
      });
    }
  });

  // Don't wait for analyses to complete - let them update asynchronously
  Promise.all(analysisPromises).then(() => {
    console.log("All contract analyses completed for table view");
  });
}

function createProductTableRow(product, index) {
  const row = document.createElement("ui5-table-row");
  row.setAttribute("data-product-index", index);

  // Product Name and Model
  const productCell = document.createElement("ui5-table-cell");
  productCell.innerHTML = `
    <div class="table-cell-content">
      <div class="table-cell-primary">${product.name || "Product"}</div>
      <div class="table-cell-secondary">Model: ${product.model || "N/A"}</div>
    </div>
  `;

  // Model
  const modelCell = document.createElement("ui5-table-cell");
  modelCell.innerHTML = `<ui5-text class="table-cell-primary">${product.model || "N/A"}</ui5-text>`;

  // Dimensions
  const dimensionsCell = document.createElement("ui5-table-cell");
  dimensionsCell.innerHTML = `<ui5-text class="table-cell-secondary">${product.size || "N/A"}</ui5-text>`;

  // Price
  const priceCell = document.createElement("ui5-table-cell");
  priceCell.innerHTML = `<ui5-text class="table-cell-primary">${product.price || "N/A"}</ui5-text>`;

  // Source
  const sourceCell = document.createElement("ui5-table-cell");
  if (product.title && product.webUrlPageNo) {
    // Extract page number from URL fragment
    let pageNumber = null;
    try {
      const url = new URL(product.webUrlPageNo, window.location.origin);
      const pageMatch = url.hash.match(/#page=(\d+)/);
      if (pageMatch) {
        pageNumber = pageMatch[1];
      }
    } catch (e) {
      // If URL parsing fails, continue without page number
    }

    let linkText = product.title;
    if (pageNumber) {
      linkText += ` (Page ${pageNumber})`;
    }

    sourceCell.innerHTML = `
      <div class="table-cell-content">
        <ui5-link href="${product.webUrlPageNo}" target="_blank" class="table-link">${linkText}</ui5-link>
      </div>
    `;
  } else if (product.title) {
    sourceCell.innerHTML = `<ui5-text class="table-cell-secondary">${product.title}</ui5-text>`;
  } else {
    sourceCell.innerHTML = `<ui5-text class="table-cell-secondary">N/A</ui5-text>`;
  }

  // Contract
  const contractCell = document.createElement("ui5-table-cell");
  const contractInfo = getContractInfo(product.title);
  if (contractInfo) {
    contractCell.innerHTML = `
      <div class="table-cell-content">
        <div class="table-cell-secondary">${contractInfo.vendor}</div>
        <ui5-link href="${API_BASE_URL}/api/storage/contracts/${encodeURIComponent(contractInfo.contract)}" target="_blank" class="table-link">${contractInfo.contract}</ui5-link>
      </div>
    `;
  } else {
    contractCell.innerHTML = `<ui5-text class="table-cell-secondary">N/A</ui5-text>`;
  }

  // Status
  const statusCell = document.createElement("ui5-table-cell");
  const analysisResult = getContractAnalysisResult(product);
  if (analysisResult) {
    statusCell.innerHTML = createStatusCellContent(analysisResult);
  } else if (contractInfo) {
    // Show loading indicator for contract analysis
    statusCell.innerHTML = `
      <div class="table-status-loading" data-contract-loading="true">
        <ui5-busy-indicator active delay="0" size="S"></ui5-busy-indicator>
        <ui5-text>Analyzing...</ui5-text>
      </div>
    `;
  } else {
    statusCell.innerHTML = `<ui5-text class="table-cell-secondary">N/A</ui5-text>`;
  }

  row.appendChild(productCell);
  row.appendChild(modelCell);
  row.appendChild(dimensionsCell);
  row.appendChild(priceCell);
  row.appendChild(sourceCell);
  row.appendChild(contractCell);
  row.appendChild(statusCell);

  return row;
}

function createStatusCellContent(analysisResult) {
  const purchaseStatus = analysisResult.supports_purchase ? "✓ Purchase Supported" : "✗ Purchase Not Supported";
  const purchaseColor = analysisResult.supports_purchase ? "var(--sapPositiveColor)" : "var(--sapNegativeColor)";

  const validityStatus = analysisResult.is_valid ? "✓ Contract Valid" : "✗ Contract Expired";
  const validityColor = analysisResult.is_valid ? "var(--sapPositiveColor)" : "var(--sapNegativeColor)";

  const confidencePercent = Math.round(analysisResult.confidence_score * 100);

  return `
    <div class="table-status-container">
      <div class="table-status-item" style="color: ${purchaseColor};">
        <ui5-text style="font-size: 0.75rem;">${purchaseStatus}</ui5-text>
      </div>
      <div class="table-status-item" style="color: ${validityColor};">
        <ui5-text style="font-size: 0.75rem;">${validityStatus}</ui5-text>
      </div>
      <div class="table-status-item">
        <ui5-text style="font-size: 0.75rem; color: var(--sapContent_LabelColor);">Confidence: ${confidencePercent}%</ui5-text>
      </div>
    </div>
  `;
}

function updateProductTableRowAnalysis(product, productIndex, analysisResult) {
  const productsTable = document.getElementById("products-table");
  if (!productsTable) return;

  const tableRow = productsTable.querySelector(`ui5-table-row[data-product-index="${productIndex}"]`);
  if (!tableRow) return;

  const statusCell = tableRow.querySelector("ui5-table-cell:last-child");
  if (!statusCell) return;

  statusCell.innerHTML = createStatusCellContent(analysisResult);
}

function updateProductTableRowAnalysisError(product, productIndex, errorMessage) {
  const productsTable = document.getElementById("products-table");
  if (!productsTable) return;

  const tableRow = productsTable.querySelector(`ui5-table-row[data-product-index="${productIndex}"]`);
  if (!tableRow) return;

  const statusCell = tableRow.querySelector("ui5-table-cell:last-child");
  if (!statusCell) return;

  statusCell.innerHTML = `
    <div class="table-status-container">
      <div class="table-status-item" style="color: var(--sapNegativeColor);">
        <ui5-icon name="decline" style="margin-right: 0.25rem;"></ui5-icon>
        <ui5-text style="font-size: 0.75rem;">Analysis failed</ui5-text>
      </div>
    </div>
  `;
}

function showError(message) {
  updateAuthStatus(message, "Error");

  // Also log to console for debugging
  console.error(message);
}

function transformSharepointUrl(url) {
  if (!url) return url;

  // Check if it's a SharePoint URL
  if (url.includes("sharepoint.com")) {
    // Extract the filename
    const parts = url.split("/");
    const filenameWithParams = parts[parts.length - 1];
    return `/storage/catalogs/${filenameWithParams}`;
  }

  return url;
}
