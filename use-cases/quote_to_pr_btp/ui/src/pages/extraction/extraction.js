import "@ui5/webcomponents/dist/Assets.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/DatePicker.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Option.js";
import "@ui5/webcomponents/dist/Select.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";

import { API_BASE_URL, API_KEY, request } from "../../services/api.js";

let activeRunId = "";
let requestedDocument = "";
let currentDocumentName = "";
let overview = null;
let detail = null;
let preparedPayload = null;
let masterDataSuggestions = null;
let purchasingIntelligence = null;
let selectedLineItemOverrides = [];
let currentMaterialRows = [];
let unresolvedMaterialIndexes = [];
let payloadReady = false;
let masterDataLoaded = false;
let s4IntegrationEnabled = false;
let pdfUrl = "";

const html = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
const list = (value) => (Array.isArray(value) ? value : []);

function money(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(2) : String(value ?? "");
}

function sapDateToIso(value) {
  const match = String(value || "").match(/\/Date\((\d+)\)\//);
  if (!match) return "";
  return new Date(Number(match[1])).toISOString().slice(0, 10);
}

function showMessage(message, design = "Information") {
  const strip = document.getElementById("review-message");
  if (!strip) return;
  strip.hidden = false;
  strip.design = design;
  strip.textContent = message;
}

function friendlyS4Error(error) {
  const technicalMessage = String(error?.message || error || "");
  console.error("S/4 purchase requisition creation failed", technicalMessage);
  const supplierOrgError = technicalMessage.match(/Supplier\s+([A-Z0-9_-]+)\s+not yet created by purchasing organization\s+([A-Z0-9_-]+)/i);
  if (supplierOrgError) {
    return `Supplier ${supplierOrgError[1]} is not maintained for purchasing organization ${supplierOrgError[2]}. Keep the supplier as a suggestion or complete its purchasing data in S/4HANA, then try again.`;
  }
  const plantErrors = [...technicalMessage.matchAll(/Material\s+([A-Z0-9_-]+)\s+not maintained in plant\s+([A-Z0-9_-]+)/gi)];
  if (plantErrors.length) {
    const materials = [...new Set(plantErrors.map((match) => match[1]))].join(", ");
    const plant = plantErrors[0][2];
    return `The suggested material${plantErrors.length === 1 ? "" : "s"} ${materials} ${plantErrors.length === 1 ? "is" : "are"} not available in plant ${plant}. Select a plant-ready material or extend the material master, then try again.`;
  }
  if (/unit .*not created|baseunit/i.test(technicalMessage)) {
    return "The unit of measure was not accepted by S/4HANA. Confirm the suggested Base unit and try again.";
  }
  if (/purchaserequisitionprice|invalid value/i.test(technicalMessage)) {
    return "The item price was not accepted by S/4HANA. Confirm the Unit price and try again.";
  }
  return "S/4HANA could not create the purchase requisition. Review the highlighted SAP values and try again.";
}

function selectedDocument() {
  return currentDocumentName || document.getElementById("review-document")?.selectedOption?.value || overview?.documents?.[0] || "";
}

function currentApproach() {
  const approach = detail?.best_approach || {};
  return {
    method_family: approach.approach_family,
    model: approach.model,
    strategy: approach.strategy
  };
}

function fieldValue(id) {
  const control = document.getElementById(id);
  const value = control?.value || control?.getAttribute?.("value") || "";
  return String(value).trim() || undefined;
}

function setControlValue(id, value, { onlyIfEmpty = false } = {}) {
  const control = document.getElementById(id);
  if (!control || value === undefined || value === null) return;
  if (onlyIfEmpty && fieldValue(id)) return;
  const text = String(value);
  control.value = text;
  control.setAttribute("value", text);
}

function updateCreateAvailability() {
  const createButton = document.getElementById("review-create");
  if (!createButton) return;
  if (!s4IntegrationEnabled) {
    createButton.textContent = "S/4 Integration Not Configured";
    createButton.disabled = true;
    return;
  }
  createButton.textContent = "Create PR";
  createButton.disabled = !payloadReady || !masterDataLoaded;
}

function overrides() {
  const values = {
    material: fieldValue("review-material"),
    plant: fieldValue("review-plant"),
    purchasing_group: fieldValue("review-purchasing-group"),
    company_code: fieldValue("review-company-code"),
    material_group: fieldValue("review-material-group"),
    supplier: fieldValue("review-supplier"),
    base_unit: fieldValue("review-base-unit"),
    unit_price: fieldValue("review-unit-price"),
    currency: fieldValue("review-currency"),
    delivery_date: fieldValue("review-delivery-date")
  };
  const lineItems = selectedLineItemOverrides.filter(Boolean);
  if (lineItems.length) values.line_items = lineItems;
  return values;
}

function requestBody(includeOverrides = true) {
  return {
    run_id: activeRunId,
    document_name: selectedDocument(),
    ...currentApproach(),
    overrides: includeOverrides ? overrides() : {}
  };
}

function renderDocuments() {
  const selector = document.getElementById("review-document");
  const documents = list(overview?.documents);
  const selectedName = requestedDocument && documents.includes(requestedDocument) ? requestedDocument : documents[0];
  currentDocumentName = selectedName || "";
  selector.innerHTML = documents
    .map((name) => `<ui5-option value="${html(name)}"${name === selectedName ? " selected" : ""}>${html(name)}</ui5-option>`)
    .join("");
}

function renderQuote() {
  const header = detail?.extracted?.header || {};
  const suppliers = list(detail?.extracted?.suppliers);
  const supplier = header.vendor_name || suppliers[0]?.name || "Not extracted";
  const facts = [
    ["Supplier", supplier],
    ["Quote number", header.quote_number || "Not extracted"],
    ["Quote date", header.quote_date || "Not extracted"],
    ["Currency", header.currency || "Not extracted"],
    ["Total", header.total_amount ?? "Not extracted"],
    ["Payment terms", header.payment_terms || "Not extracted"]
  ];
  document.getElementById("review-quote-fields").innerHTML = facts
    .map(([label, value]) => `<div class="fact"><span>${html(label)}</span><strong>${html(value)}</strong></div>`)
    .join("");

  const items = list(detail?.extracted?.line_items);
  document.getElementById("review-line-items").innerHTML = items.length
    ? items.map((item) => `<tr><td>${html(item.description || "-")}</td><td>${html(item.quantity ?? "-")}</td><td>${html(item.unit_of_measure || "-")}</td><td>${html(item.unit_price == null ? "-" : money(item.unit_price))}</td><td>${html(item.line_total == null ? "-" : money(item.line_total))}</td></tr>`).join("")
    : `<tr><td colspan="5">No line items were extracted. Review is required before creation.</td></tr>`;
}

function matchStateLabel(status) {
  if (status === "matched") return "Matched";
  if (status === "review") return "Review suggested";
  if (status === "no_reliable_match") return "No reliable match";
  if (status === "missing_source_value") return "Source value missing";
  return "Unavailable";
}

function resetMasterDataSection() {
  masterDataSuggestions = null;
  selectedLineItemOverrides = [];
  currentMaterialRows = [];
  unresolvedMaterialIndexes = [];
  masterDataLoaded = false;
  const loading = document.getElementById("master-data-loading");
  const content = document.getElementById("master-data-content");
  const status = document.getElementById("master-data-status");
  if (loading) loading.hidden = false;
  if (content) content.hidden = true;
  if (status) {
    status.dataset.state = "loading";
    status.textContent = "Searching";
  }
  updateCreateAvailability();
}

function resetPurchasingIntelligence() {
  purchasingIntelligence = null;
  const loading = document.getElementById("intelligence-loading");
  const content = document.getElementById("intelligence-content");
  const source = document.getElementById("intelligence-source");
  const history = document.getElementById("purchasing-history");
  if (loading) loading.hidden = false;
  if (content) content.hidden = true;
  if (history) history.hidden = true;
  if (source) {
    source.dataset.state = "loading";
    source.textContent = "Loading";
  }
}

function intelligenceFact(label, value, detail = "") {
  return `<div class="intelligence-fact"><span>${html(label)}</span><strong>${html(value || "-")}</strong>${detail ? `<small>${html(detail)}</small>` : ""}</div>`;
}

function renderPurchasingIntelligence(result) {
  purchasingIntelligence = result;
  const vendor = result?.vendor || {};
  const decision = result?.decision || {};
  const purchaseOrders = list(result?.purchase_orders);
  const purchaseRequisitions = list(result?.purchase_requisitions);
  const source = document.getElementById("intelligence-source");
  document.getElementById("intelligence-loading").hidden = true;
  document.getElementById("intelligence-content").hidden = false;
  source.dataset.state = "matched";
  source.textContent = result?.source?.label || "Available";

  const lastPo = purchaseOrders[0];
  const lastPr = purchaseRequisitions[0];
  document.getElementById("intelligence-facts").innerHTML = [
    intelligenceFact("SAP supplier", vendor.supplier_id || "Not found", vendor.name),
    intelligenceFact("Supplier status", vendor.preferred_label || "Unknown", vendor.reason),
    intelligenceFact("Latest similar PO", lastPo?.purchase_order || "No history", lastPo ? `${lastPo.creation_date} | ${money(lastPo.net_unit_price)} ${lastPo.currency}` : "No similar purchase was found"),
    intelligenceFact("Latest PR", lastPr?.purchase_requisition || "No history", lastPr ? `${lastPr.status} | ${lastPr.creation_date}` : "No previous requisition was found")
  ].join("");

  document.getElementById("purchase-history-rows").innerHTML = purchaseOrders.length
    ? purchaseOrders.map((row) => `<tr><td>${html(row.item_description || "-")}<small>${html(row.material || "No material")}</small></td><td>${html(row.purchase_order)}</td><td>${html(`${money(row.net_unit_price)} ${row.currency || ""}`)}</td><td>${html(`${row.quantity ?? "-"} ${row.unit || ""}`)}</td><td>${html(row.creation_date || "-")}</td></tr>`).join("")
    : `<tr><td colspan="5">No similar purchase history is available.</td></tr>`;
  document.getElementById("pr-history-rows").innerHTML = purchaseRequisitions.length
    ? purchaseRequisitions.map((row) => `<tr><td>${html(row.purchase_requisition)}</td><td>${html(row.status || "-")}</td><td>${html(row.creation_date || "-")}</td></tr>`).join("")
    : `<tr><td colspan="3">No previous purchase requisitions are available.</td></tr>`;

  document.getElementById("toggle-purchasing-history").hidden = !purchaseOrders.length && !purchaseRequisitions.length;
  document.getElementById("supplier-onboarding").hidden = !decision.supplier_review_required;
  document.getElementById("intelligence-notice").textContent = result?.source?.notice || "";
  const attention = document.getElementById("review-attention");
  attention.dataset.state = decision.status === "ready_after_validation" ? "ready" : "review";
  document.getElementById("review-attention-title").textContent = decision.title || "Review purchasing context";
  document.getElementById("review-attention-text").textContent = decision.message || "Review the proposed values before creating the PR.";
}

async function loadPurchasingIntelligence() {
  const documentName = selectedDocument();
  try {
    const result = await request("/api/purchase-requisition/purchasing-intelligence", "POST", requestBody(false));
    if (selectedDocument() !== documentName) return;
    renderPurchasingIntelligence(result);
  } catch (error) {
    if (selectedDocument() !== documentName) return;
    document.getElementById("intelligence-loading").hidden = true;
    document.getElementById("intelligence-content").hidden = false;
    document.getElementById("intelligence-source").dataset.state = "unavailable";
    document.getElementById("intelligence-source").textContent = "Unavailable";
    document.getElementById("intelligence-facts").innerHTML = intelligenceFact("Purchasing context", "Unavailable", "Continue with manual review");
    document.getElementById("intelligence-notice").textContent = "Purchasing history could not be loaded. This does not block manual PR review.";
    console.error("Purchasing intelligence failed", error);
  }
}

function isReliableMaterialCandidate(candidate) {
  return Boolean(
    candidate
    && candidate.confidence === "High"
    && Number(candidate.score || 0) >= 88
  );
}

function isReliablePartnerCandidate(candidate) {
  return Boolean(
    candidate
    && candidate.supplier
    && candidate.confidence === "High"
    && Number(candidate.score || 0) >= 88
  );
}

function applyPartnerCandidate(candidate, { explicit = false } = {}) {
  const input = document.getElementById("review-supplier");
  const note = document.querySelector('[data-note-for="supplier"]');
  const field = note?.closest(".sap-field");
  const reliable = isReliablePartnerCandidate(candidate);
  if (!input || !candidate?.supplier || (!explicit && !reliable)) return false;

  setControlValue("review-supplier", candidate.supplier);
  if (field) {
    field.dataset.defaulted = String(!reliable);
    field.dataset.matchState = reliable ? "matched" : "review";
  }
  if (note) {
    note.textContent = reliable
      ? `S/4 match - High confidence (${candidate.score}%)`
      : `Selected S/4 supplier - ${candidate.score}% ${candidate.confidence}; please confirm`;
  }
  return true;
}

function materialOverride(index, candidate) {
  if (!candidate?.material || candidate.plant_ready === false) return null;
  return {
    index,
    material: candidate.material,
    material_group: candidate.material_group || fieldValue("review-material-group"),
    base_unit: candidate.base_unit || fieldValue("review-base-unit")
  };
}

function updateFallbackConfirmation({ reset = false } = {}) {
  unresolvedMaterialIndexes = currentMaterialRows
    .map((_match, index) => (selectedLineItemOverrides[index] ? null : index))
    .filter((index) => index !== null);
  updateCreateAvailability();
}

function renderProposalStatus(lineIndex, proposal) {
  const status = document.querySelector(`[data-proposal-status="${lineIndex}"]`);
  const button = document.querySelector(`[data-proposal-index="${lineIndex}"]`);
  if (status) {
    status.dataset.state = "draft";
    status.textContent = `Material request ${proposal.proposal_id} prepared. No SAP material created yet.`;
  }
  if (button) button.textContent = "Reopen draft proposal";
}

async function loadSavedMaterialProposals() {
  try {
    const params = new URLSearchParams({ document_name: selectedDocument() });
    const result = await request(`/api/purchase-requisition/material-proposals?${params.toString()}`);
    list(result?.proposals).forEach((proposal) => renderProposalStatus(Number(proposal.line_index), proposal));
  } catch (error) {
    console.error("Saved material proposals could not be loaded", error);
  }
}

function materialOptionLabel(candidate) {
  return `${candidate.material} - ${candidate.material_description || "No description"} | ${candidate.score}% ${candidate.confidence}`;
}

function materialWarning(candidate) {
  if (!candidate) return "No match";
  if (!isReliableMaterialCandidate(candidate)) return "Review suggested";
  return "Matched";
}

function renderMasterDataSuggestions(result) {
  masterDataSuggestions = result;
  const loading = document.getElementById("master-data-loading");
  const content = document.getElementById("master-data-content");
  const status = document.getElementById("master-data-status");
  loading.hidden = true;
  content.hidden = false;

  if (result?.status === "unavailable") {
    status.dataset.state = "unavailable";
    status.textContent = "Unavailable";
    document.getElementById("partner-source-name").textContent = "Master-data search unavailable";
    document.getElementById("partner-match-note").textContent = result.message || "Enter the SAP values manually.";
    document.getElementById("review-partner-match").innerHTML = `<ui5-option value="">No candidates</ui5-option>`;
    document.getElementById("material-match-rows").innerHTML = `<tr><td colspan="3">SAP material suggestions are unavailable. Continue with the proposed PR values.</td></tr>`;
    document.getElementById("master-data-guardrail").textContent = "No automatic values were applied.";
    currentMaterialRows = list(detail?.extracted?.line_items).map((item) => ({ ...item, candidates: [] }));
    selectedLineItemOverrides = currentMaterialRows.map(() => null);
    masterDataLoaded = true;
    updateFallbackConfirmation({ reset: true });
    return;
  }

  const partner = result?.business_partner || {};
  const partnerCandidates = list(partner.candidates);
  const selectedPartner = partnerCandidates[0] || null;
  const partnerApplied = applyPartnerCandidate(selectedPartner);
  status.dataset.state = result?.status === "matched" ? "matched" : "review";
  status.textContent = result?.status === "matched" ? "Matched" : "Review needed";
  document.getElementById("partner-source-name").textContent = partner.query || "Supplier name not extracted";
  document.getElementById("partner-match-note").textContent = partnerCandidates.length
    ? partnerApplied
      ? `${matchStateLabel(partner.status)}. Supplier ${selectedPartner.supplier} is applied to the PR: ${selectedPartner.score}% ${selectedPartner.confidence}.`
      : `${matchStateLabel(partner.status)}. Showing best candidate: ${partnerCandidates[0].score}% ${partnerCandidates[0].confidence}. Confirm it before applying.`
    : "No supplier candidates were returned.";

  const partnerSelect = document.getElementById("review-partner-match");
  partnerSelect.innerHTML = [
    ...partnerCandidates.map((candidate) => `<ui5-option value="${html(candidate.supplier)}"${selectedPartner?.supplier === candidate.supplier ? " selected" : ""}>${html(candidate.name)} - ${html(candidate.supplier)} | ${html(candidate.score)}% ${html(candidate.confidence)}</ui5-option>`),
    ...(!partnerCandidates.length ? [`<ui5-option value="">No candidates</ui5-option>`] : [])
  ].join("");
  partnerSelect.onchange = () => {
    const candidate = partnerCandidates.find((item) => item.supplier === partnerSelect.selectedOption?.value);
    const applied = applyPartnerCandidate(candidate, { explicit: true });
    document.getElementById("partner-match-note").textContent = candidate
      ? applied
        ? `Supplier ${candidate.supplier} will be used in the PR: ${candidate.score}% ${candidate.confidence}.`
        : `${candidate.score}% ${candidate.confidence} match. This candidate has no usable SAP supplier number.`
      : "No supplier candidate selected.";
    const supplierValue = fieldValue("review-supplier");
    const materialValue = fieldValue("review-material") || "the reviewed fallback";
    document.getElementById("master-data-guardrail").textContent = supplierValue
      ? `PR payload: supplier ${supplierValue}; material ${materialValue}. Material suggestions only replace the fallback after they are plant-ready and explicitly selected.`
      : `PR payload: no supplier is applied; material ${materialValue}. Confirm the supplier before creating the PR.`;
  };

  const materialRows = list(result?.materials);
  currentMaterialRows = materialRows;
  selectedLineItemOverrides = materialRows.map((match, index) => {
    const best = list(match.candidates)[0];
    return isReliableMaterialCandidate(best) ? materialOverride(index, best) : null;
  });
  document.getElementById("material-match-rows").innerHTML = materialRows.length
    ? materialRows.map((match, index) => {
        const candidates = list(match.candidates);
        const best = candidates[0];
        const options = candidates.length
          ? candidates.map((candidate, candidateIndex) => `<ui5-option value="${html(candidate.material)}"${candidateIndex === 0 ? " selected" : ""}>${html(materialOptionLabel(candidate))}</ui5-option>`).join("")
          : `<ui5-option value="">No candidates</ui5-option>`;
        const reliable = isReliableMaterialCandidate(best);
        const proposalAllowed = !reliable;
        return `<tr class="${reliable ? "match-ready-row" : "match-review-row"}">
          <td>
            <strong>${html(match.description || `Item ${index + 1}`)}</strong>
            <small>${html(match.manufacturer_part_number || match.vendor_material_number || "No part number")}</small>
          </td>
          <td class="material-candidate-cell">
            <ui5-select data-material-index="${index}" accessible-name="Suggested material for item ${index + 1}">${options}</ui5-select>
            ${proposalAllowed ? `<div class="material-proposal-action"><ui5-button data-proposal-index="${index}" design="Transparent" icon="add">Prepare material request</ui5-button><small class="proposal-status" data-proposal-status="${index}"></small></div>` : ""}
          </td>
          <td><span class="confidence-value" data-match-confidence="${index}" data-confidence="${html(best?.confidence || "Low")}">${html(best ? `${best.score}% ${best.confidence}` : "No match")}</span><small class="match-decision" data-match-warning="${index}">${html(materialWarning(best))}</small></td>
        </tr>`;
      }).join("")
    : `<tr><td colspan="3">No line items are available for material matching.</td></tr>`;

  document.querySelectorAll("[data-material-index]").forEach((select) => {
    select.onchange = () => {
      const index = Number(select.dataset.materialIndex);
      const match = materialRows[index];
      const candidate = list(match?.candidates).find((item) => item.material === select.selectedOption?.value);
      const confidence = document.querySelector(`[data-match-confidence="${index}"]`);
      const warning = document.querySelector(`[data-match-warning="${index}"]`);
      if (confidence) {
        confidence.dataset.confidence = candidate?.confidence || "Low";
        confidence.textContent = candidate ? `${candidate.score}% ${candidate.confidence}` : "No match";
      }
      const applied = materialOverride(index, candidate);
      selectedLineItemOverrides[index] = applied;
      if (warning) {
        warning.dataset.state = isReliableMaterialCandidate(candidate) ? "ready" : "review";
        warning.textContent = materialWarning(candidate);
      }
      updateFallbackConfirmation();
    };
  });
  document.querySelectorAll("[data-proposal-index]").forEach((button) => {
    button.onclick = () => prepareMaterialProposal(Number(button.dataset.proposalIndex), button);
  });
  const elapsed = Number(result?.client_elapsed_ms || 0);
  const timing = elapsed > 0 ? ` Search completed in ${(elapsed / 1000).toFixed(1)} seconds.` : "";
  const reviewCount = materialRows.filter((match) => !isReliableMaterialCandidate(list(match.candidates)[0])).length;
  const reviewSummary = reviewCount
    ? `${reviewCount} of ${materialRows.length} item${materialRows.length === 1 ? " needs" : "s need"} review.`
    : "All quote items have a high-confidence suggestion.";
  document.getElementById("master-data-guardrail").textContent = `${reviewSummary}${timing}`;
  masterDataLoaded = true;
  updateFallbackConfirmation({ reset: true });
  void loadSavedMaterialProposals();
}

async function prepareMaterialProposal(lineIndex, button) {
  const status = document.querySelector(`[data-proposal-status="${lineIndex}"]`);
  let prepared = false;
  button.disabled = true;
  button.textContent = "Preparing...";
  try {
    const result = await request("/api/purchase-requisition/material-proposals", "POST", {
      ...requestBody(false),
      line_index: lineIndex
    });
    renderProposalStatus(lineIndex, result);
    prepared = true;
    showMessage(result.message, "Positive");
  } catch (error) {
    if (status) status.textContent = "The material proposal could not be prepared.";
    showMessage(`Could not prepare the material proposal: ${error.message}`, "Negative");
  } finally {
    button.disabled = false;
    if (!prepared) button.textContent = "Prepare material request";
  }
}

async function sendToBackOffice() {
  const button = document.getElementById("send-back-office");
  const status = document.getElementById("back-office-status");
  button.disabled = true;
  button.textContent = "Sending...";
  try {
    const result = await request("/api/purchase-requisition/back-office-referrals", "POST", requestBody(false));
    status.textContent = `${result.reused ? "Existing referral reopened" : "Referral submitted"}: ${result.referral_id}.`;
    showMessage(result.message, "Positive");
  } catch (error) {
    status.textContent = "The onboarding referral could not be prepared.";
    showMessage(`Could not send the supplier to back office: ${error.message}`, "Negative");
  } finally {
    button.disabled = false;
    button.textContent = "Send to back office";
  }
}

async function loadMasterDataSuggestions({ forceRefresh = false } = {}) {
  const documentName = selectedDocument();
  const started = performance.now();
  const refreshButton = document.getElementById("refresh-master-data");
  if (forceRefresh) {
    resetMasterDataSection();
    if (refreshButton) {
      refreshButton.disabled = true;
      refreshButton.textContent = "Refreshing...";
    }
  }
  try {
    const result = await request("/api/purchase-requisition/s4/master-data/suggestions", "POST", {
      ...requestBody(false),
      force_refresh: forceRefresh
    });
    if (selectedDocument() !== documentName) return;
    result.client_elapsed_ms = performance.now() - started;
    renderMasterDataSuggestions(result);
  } catch (error) {
    if (selectedDocument() !== documentName) return;
    renderMasterDataSuggestions({ status: "unavailable", message: "SAP master data could not be searched. Enter the values manually." });
    console.error("S/4 master-data suggestions failed", error);
  } finally {
    if (refreshButton) {
      refreshButton.disabled = false;
      refreshButton.textContent = "Search again";
    }
  }
}

async function loadPdf(documentName) {
  const response = await fetch(`${API_BASE_URL}/api/purchase-requisition/documents/${encodeURIComponent(documentName)}/file`, {
    headers: { "X-API-Key": API_KEY }
  });
  if (!response.ok) throw new Error(`Could not load the source PDF (HTTP ${response.status}).`);
  const blob = await response.blob();
  if (pdfUrl) URL.revokeObjectURL(pdfUrl);
  pdfUrl = URL.createObjectURL(blob);
  document.getElementById("review-pdf").src = pdfUrl;
}

function setInput(id, value) {
  setControlValue(id, value, { onlyIfEmpty: true });
}

function clearPreparedInputs() {
  [
    "review-material",
    "review-plant",
    "review-purchasing-group",
    "review-company-code",
    "review-material-group",
    "review-supplier",
    "review-base-unit",
    "review-unit-price",
    "review-currency",
    "review-delivery-date"
  ].forEach((id) => {
    setControlValue(id, "");
  });
  document.querySelectorAll("[data-note-for]").forEach((note) => { note.textContent = ""; });
  document.querySelectorAll(".sap-field").forEach((field) => {
    delete field.dataset.defaulted;
    delete field.dataset.matchState;
  });
  payloadReady = false;
  masterDataLoaded = false;
  s4IntegrationEnabled = false;
  currentMaterialRows = [];
  unresolvedMaterialIndexes = [];
  selectedLineItemOverrides = [];
  updateCreateAvailability();
}

function markDefaulted(name, used) {
  const note = document.querySelector(`[data-note-for="${name}"]`);
  const field = note?.closest(".sap-field");
  if (field) field.dataset.defaulted = String(Boolean(used));
  if (note) note.textContent = used ? "Suggested default - please confirm" : "";
}

function renderPrepared(result) {
  preparedPayload = result;
  s4IntegrationEnabled = result?.s4_integration_enabled === true;
  const prepared = result?.purchase_requisition || {};
  const payload = prepared.payload || {};
  const firstItem = payload?.to_PurchaseReqnItem?.results?.[0] || {};
  const defaults = prepared?.source_summary?.defaulted_fields || {};
  const missing = list(prepared.missing_fields);

  setInput("review-material", firstItem.Material);
  setInput("review-plant", firstItem.Plant);
  setInput("review-purchasing-group", firstItem.PurchasingGroup);
  setInput("review-company-code", firstItem.CompanyCode);
  setInput("review-material-group", firstItem.MaterialGroup);
  setInput("review-supplier", firstItem.Supplier);
  setInput("review-base-unit", firstItem.BaseUnit);
  setInput("review-unit-price", money(firstItem.PurchaseRequisitionPrice));
  setInput("review-currency", firstItem.PurReqnItemCurrency);
  setInput("review-delivery-date", sapDateToIso(firstItem.DeliveryDate));

  ["material", "plant", "purchasing_group", "company_code", "material_group", "base_unit", "unit_price", "delivery_date"].forEach((name) => markDefaulted(name, defaults[name]));
  const defaultCount = Object.values(defaults).filter(Boolean).length;
  const attention = document.getElementById("review-attention");
  const title = document.getElementById("review-attention-title");
  const text = document.getElementById("review-attention-text");
  if (missing.length || defaultCount) {
    attention.dataset.state = "review";
    title.textContent = `${missing.length + defaultCount} value${missing.length + defaultCount === 1 ? "" : "s"} need confirmation`;
    text.textContent = missing.length ? `Missing: ${missing.join(", ")}.` : "Highlighted values were supplied from approved prototype defaults.";
  } else {
    attention.dataset.state = "ready";
    title.textContent = "Ready to create";
    text.textContent = "The quote and required SAP values passed validation.";
  }
  payloadReady = Boolean(prepared.ready_for_create);
  if (!s4IntegrationEnabled) {
    attention.dataset.state = "review";
    title.textContent = "Extraction complete";
    text.textContent = "Review the proposed values. S/4HANA creation is optional and is not configured in this deployment.";
  }
  updateFallbackConfirmation();
}

async function preparePayload() {
  const result = await request("/api/purchase-requisition/s4/purchase-requisition/payload", "POST", requestBody(false));
  renderPrepared(result);
}

async function loadSelectedDocument() {
  const documentName = selectedDocument();
  if (!documentName) throw new Error("No completed document is available for review.");
  document.getElementById("review-source-name").textContent = documentName;
  clearPreparedInputs();
  const params = new URLSearchParams({ run_id: activeRunId });
  detail = await request(`/api/purchase-requisition/documents/${encodeURIComponent(documentName)}?${params.toString()}`);
  renderQuote();
  resetMasterDataSection();
  resetPurchasingIntelligence();
  await Promise.all([loadPdf(documentName), preparePayload()]);
  void Promise.all([loadMasterDataSuggestions(), loadPurchasingIntelligence()]);
}

async function resolveRunId() {
  if (activeRunId) return;
  const projects = await request("/api/purchase-requisition/projects");
  const project = projects?.projects?.[0];
  activeRunId = project?.runs?.[0] || project?.default_run_id || "";
}

async function loadPage() {
  const loading = document.getElementById("review-loading");
  const workspace = document.getElementById("review-workspace");
  loading.hidden = false;
  workspace.hidden = true;
  try {
    await resolveRunId();
    if (!activeRunId) throw new Error("No completed extraction is available yet.");
    overview = await request(`/api/purchase-requisition/overview?run_id=${encodeURIComponent(activeRunId)}`);
    renderDocuments();
    await loadSelectedDocument();
    loading.hidden = true;
    workspace.hidden = false;
  } catch (error) {
    loading.hidden = true;
    showMessage(`Could not open the review: ${error.message}`, "Negative");
  }
}

async function createPr() {
  const button = document.getElementById("review-create");
  button.disabled = true;
  button.textContent = "Creating...";
  showMessage("Creating the reviewed purchase requisition in S/4HANA.");
  try {
    const refreshed = await request("/api/purchase-requisition/s4/purchase-requisition/payload", "POST", requestBody(true));
    renderPrepared(refreshed);
    if (!refreshed?.purchase_requisition?.ready_for_create) {
      showMessage("Complete the highlighted required values before creating the PR.", "Critical");
      return;
    }
    const result = await request("/api/purchase-requisition/s4/purchase-requisition/create", "POST", { ...requestBody(true), confirm_create: true });
    const prNumber = result?.s4_result?.purchase_requisition;
    if (!prNumber) throw new Error("S/4HANA accepted the request but did not return the created PR number.");
    document.getElementById("review-pr-number").textContent = prNumber;
    document.getElementById("review-workspace").hidden = true;
    document.getElementById("review-success").hidden = false;
    document.getElementById("review-message").hidden = true;
  } catch (error) {
    showMessage(friendlyS4Error(error), "Negative");
  } finally {
    button.textContent = "Create PR";
    updateCreateAvailability();
  }
}

export default function initReviewPage() {
  const params = new URLSearchParams(window.location.search);
  activeRunId = params.get("run_id") || "";
  requestedDocument = params.get("document") || "";
  document.getElementById("review-document")?.addEventListener("change", async () => {
    currentDocumentName = document.getElementById("review-document")?.selectedOption?.value || "";
    requestedDocument = currentDocumentName;
    document.getElementById("review-loading").hidden = false;
    document.getElementById("review-workspace").hidden = true;
    try {
      await loadSelectedDocument();
      document.getElementById("review-loading").hidden = true;
      document.getElementById("review-workspace").hidden = false;
    } catch (error) {
      document.getElementById("review-loading").hidden = true;
      showMessage(`Could not load the selected document: ${error.message}`, "Negative");
    }
  });
  document.getElementById("review-create")?.addEventListener("click", createPr);
  document.getElementById("refresh-master-data")?.addEventListener("click", () => {
    void loadMasterDataSuggestions({ forceRefresh: true });
  });
  document.getElementById("review-material")?.addEventListener("input", () => {
    updateFallbackConfirmation();
  });
  document.getElementById("send-back-office")?.addEventListener("click", sendToBackOffice);
  document.getElementById("toggle-purchasing-history")?.addEventListener("click", (event) => {
    const history = document.getElementById("purchasing-history");
    history.hidden = !history.hidden;
    event.currentTarget.textContent = history.hidden ? "View purchasing history" : "Hide purchasing history";
  });
  document.getElementById("review-back")?.addEventListener("click", () => window.pageRouter.navigate("/purchase-requisition"));
  document.getElementById("review-new-quote")?.addEventListener("click", () => window.pageRouter.navigate("/purchase-requisition"));
  loadPage();
}
