import { downloadBatchArtifact, runBatch } from "../../services/offerApi.js";

function triggerBrowserDownload(blob, fileName) {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}

function renderBatchResult(result, api) {
  const status = document.getElementById("batch-status");
  const summary = document.getElementById("batch-summary");
  const artifacts = document.getElementById("batch-artifacts");

  status.textContent = "Analysis completed";
  summary.innerHTML = `
    <div class="metric-grid">
      <div class="metric-card"><span>Total</span><strong>${result.summary.total_accounts}</strong></div>
      <div class="metric-card"><span>Residential</span><strong>${result.summary.residential_accounts}</strong></div>
      <div class="metric-card"><span>Commercial</span><strong>${result.summary.commercial_accounts}</strong></div>
      <div class="metric-card"><span>With Offer</span><strong>${result.summary.accounts_with_final_offer}</strong></div>
    </div>
  `;
  summary.hidden = false;

  artifacts.innerHTML = `
    <div class="result-card">
      <div class="section-label">Artifacts</div>
      <button type="button" class="link-button" data-artifact="nbo_recommendations.xlsx">Download Excel</button>
      <button type="button" class="link-button" data-artifact="nbo_recommendations.json">Download JSON</button>
    </div>
  `;
  artifacts.hidden = false;

  artifacts.querySelectorAll("[data-artifact]").forEach((button) => {
    button.addEventListener("click", async () => {
      const fileName = button.getAttribute("data-artifact");
      if (!fileName) {
        return;
      }
      const blob = await api.downloadBatchArtifact(result.run_id, fileName);
      triggerBrowserDownload(blob, fileName);
    });
  });
}

export async function initBatchPage(api = { runBatch, downloadBatchArtifact }) {
  const runButton = document.getElementById("batch-run");
  const status = document.getElementById("batch-status");
  const summary = document.getElementById("batch-summary");
  const artifacts = document.getElementById("batch-artifacts");

  summary.hidden = true;
  artifacts.hidden = true;

  runButton?.addEventListener("click", async () => {
    status.textContent = "Running batch analysis...";
    const result = await api.runBatch();
    renderBatchResult(result, api);
  });
}
