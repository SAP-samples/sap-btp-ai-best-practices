import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Table.js";
import "@ui5/webcomponents/dist/TableHeaderRow.js";
import "@ui5/webcomponents/dist/TableHeaderCell.js";
import "@ui5/webcomponents/dist/TableRow.js";
import "@ui5/webcomponents/dist/TableCell.js";
import "@ui5/webcomponents/dist/BusyIndicator.js";
import { marked } from "marked";

import { OrdersAPI, AnomalyAPI } from "../../services/api.js";

/**
 * Format feature value - converts boolean features (starting with "Is ") to True/False
 */
function formatFeatureValue(feature, value) {
    if (feature.startsWith("Is ")) {
        const numValue = parseFloat(value);
        if (numValue === 1.0 || numValue === 1) return "True";
        if (numValue === 0.0 || numValue === 0) return "False";
    }
    return value;
}

/**
 * Convert anomaly score to human-readable probability label
 */
function getAnomalyProbabilityLabel(score) {
    if (score >= 0.05) return "Normal";
    if (score > -0.05) return "Low Probability Anomaly";
    if (score >= -0.1) return "Medium Probability Anomaly";
    return "High Probability Anomaly";
}

let currentDoc = null;
let currentItem = null;

export default function initOrders() {
    const searchBtn = document.getElementById("search-btn");
    const randomBtn = document.getElementById("random-btn");
    const explainBtn = document.getElementById("explain-btn");
    const classifyBtn = document.getElementById("classify-btn");
    
    searchBtn.addEventListener("click", performSearch);
    randomBtn.addEventListener("click", loadRandomAnomalousOrder);
    explainBtn.addEventListener("click", generateExplanation);
    classifyBtn.addEventListener("click", runClassification);

    // Check URL params for initial load
    const params = new URLSearchParams(window.location.search);
    const doc = params.get("doc");
    const item = params.get("item");
    if (doc && item) {
        document.getElementById("doc-num-input").value = doc;
        document.getElementById("doc-item-input").value = item;
        performSearch();
    }
}

async function performSearch() {
    const docNum = document.getElementById("doc-num-input").value;
    const docItem = document.getElementById("doc-item-input").value;

    if (!docNum || !docItem) {
        // Show toast or error
        alert("Please enter both Document Number and Item Number");
        return;
    }

    try {
        const data = await OrdersAPI.getOrder(docNum, docItem);
        currentDoc = docNum;
        currentItem = docItem;
        renderOrder(data);
    } catch (error) {
        alert("Order not found or error loading details");
        document.getElementById("order-content").style.display = "none";
    }
}

/**
 * Fetch and load a random anomalous order to mirror the Streamlit demo behavior.
 */
async function loadRandomAnomalousOrder() {
    const randomBtn = document.getElementById("random-btn");
    randomBtn.disabled = true;
    try {
        const data = await OrdersAPI.getRandomAnomalous();
        document.getElementById("doc-num-input").value = data.doc_number;
        document.getElementById("doc-item-input").value = data.doc_item;
        await performSearch();
    } catch (error) {
        alert("No anomalous orders available or failed to load.");
    } finally {
        randomBtn.disabled = false;
    }
}

function renderOrder(data) {
    document.getElementById("order-content").style.display = "block";
    const order = data.order;

    // Status Banner
    const banner = document.getElementById("status-banner");
    const isAnomaly = order.predicted_anomaly === 1;
    banner.textContent = isAnomaly ? "ANOMALY DETECTED" : "NORMAL ORDER";
    banner.className = `status-banner ${isAnomaly ? 'status-anomaly' : 'status-normal'}`;

    // Details (show sold-to and ship-to separately to avoid confusion)
    document.getElementById("val-sold-to").textContent = order["Sold To number"];
    document.getElementById("val-ship-to").textContent = order["Ship-To Party"];
    document.getElementById("val-material-number").textContent = order["Material Number"];
    document.getElementById("val-material-description").textContent = order["Material Description"];
    document.getElementById("val-qty").textContent = order["Sales Order item qty"];
    document.getElementById("val-value").textContent = `$${order["Order item value"].toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById("val-score").textContent = getAnomalyProbabilityLabel(order["anomaly_score"]);

    // SHAP Table
    const table = document.getElementById("shap-table");
    const rows = table.querySelectorAll("ui5-table-row");
    rows.forEach(r => r.remove());

    if (data.shap_explanation && data.shap_explanation.length > 0) {
        data.shap_explanation.forEach(row => {
            const tr = document.createElement("ui5-table-row");
            const formattedValue = formatFeatureValue(row.Feature, row.Value);
            [row.Feature, formattedValue, row.Contribution].forEach(text => {
                const cell = document.createElement("ui5-table-cell");
                cell.textContent = text;
                tr.appendChild(cell);
            });
            table.appendChild(tr);
        });
    } else {
        const tr = document.createElement("ui5-table-row");
        const cell = document.createElement("ui5-table-cell");
        cell.textContent = "No significant contributors found or SHAP not available.";
        tr.appendChild(cell);
        table.appendChild(tr);
    }

    // Reset AI areas
    document.getElementById("ai-result").style.display = "none";
    document.getElementById("binary-result").style.display = "none";
}

async function generateExplanation() {
    if (!currentDoc) return;

    const busy = document.getElementById("ai-busy");
    const resultBox = document.getElementById("ai-result");
    const btn = document.getElementById("explain-btn");

    busy.style.display = "block";
    btn.disabled = true;
    resultBox.style.display = "none";

    try {
        const data = await OrdersAPI.generateExplanation(currentDoc, currentItem);
        resultBox.innerHTML = marked.parse(data.explanation);
        resultBox.style.display = "block";
    } catch (error) {
        resultBox.textContent = "Error generating explanation.";
        resultBox.style.display = "block";
    } finally {
        busy.style.display = "none";
        btn.disabled = false;
    }
}

async function runClassification() {
    if (!currentDoc) return;

    const busy = document.getElementById("classify-busy");
    const resultBox = document.getElementById("binary-result");
    const btn = document.getElementById("classify-btn");

    busy.style.display = "block";
    btn.disabled = true;
    resultBox.style.display = "none";

    try {
        const data = await AnomalyAPI.explainBinary(currentDoc, currentItem);
        resultBox.textContent = data.classification;
        resultBox.style.display = "block";
    } catch (error) {
        resultBox.textContent = "Error running classification.";
        resultBox.style.display = "block";
    } finally {
        busy.style.display = "none";
        btn.disabled = false;
    }
}

