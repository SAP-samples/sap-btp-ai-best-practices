/* Joule page components */
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Panel.js";
import "@ui5/webcomponents/dist/Text.js";
import "@ui5/webcomponents/dist/Title.js";

import { API_BASE_URL, request } from "../../services/api.js";

function formatSummary(summary) {
  if (!summary || typeof summary !== "object") {
    return JSON.stringify(summary ?? {}, null, 2);
  }

  return [
    `total_items: ${summary.total_items}`,
    `active_count: ${summary.active_count}`,
    `status: ${summary.status}`,
    `last_updated: ${summary.last_updated}`
  ].join("\n");
}

export default function initJoulePage() {
  const summaryButton = document.getElementById("joule-summary-button");
  const openApiButton = document.getElementById("joule-goto-api-button");
  const summaryPanel = document.getElementById("joule-summary-panel");
  const summaryContainer = document.getElementById("joule-summary");

  if (!summaryButton || !openApiButton || !summaryPanel || !summaryContainer) {
    return;
  }

  const loadSummary = async () => {
    summaryButton.loading = true;
    summaryPanel.collapsed = false;
    summaryContainer.textContent = "Loading Joule sample summary...";

    try {
      const data = await request("/api/joule/summary");
      summaryContainer.textContent = formatSummary(data);
    } catch (error) {
      summaryContainer.textContent = `Error: ${error.message}`;
    } finally {
      summaryButton.loading = false;
    }
  };

  summaryButton.addEventListener("click", loadSummary);
  openApiButton.addEventListener("click", () => {
    const docsUrl = `${API_BASE_URL}/docs`;
    window.open(docsUrl, "_blank");
  });
}

