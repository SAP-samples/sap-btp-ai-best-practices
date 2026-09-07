import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Title.js";
import "@ui5/webcomponents-icons/dist/pdf-attachment.js";
import "@ui5/webcomponents-icons/dist/refresh.js";

import Chart from "chart.js/auto";

import { customerClassOptions } from "../assessment/customerClassScope.js";
import { getLanguage, t } from "../../modules/i18n.js";
import { request, requestBlob } from "../../services/api.js";
import { RequestGeneration } from "../../services/requestGeneration.js";
import {
  buildNestedBarChartConfig,
  buildScoreViewModel,
  qualitativeScoreMarkup,
  scoreAriaDescription
} from "./scoreViewModel.js";

const ASSESSMENT_ID = "demo-assessment";
const REPORT_JOB_STORAGE_PREFIX = "document-assessment-report-job";
const REPORT_POLL_INTERVAL_MS = 2500;

const scoreState = {
  chartInstances: new Map(),
  customerClassScope: null,
  benchmarkOptions: null,
  profile: null,
  payload: null
};

let activeReportJobId = null;
let activeReportLanguage = null;
let reportPollTimer = null;
let languageChangeListenerAttached = false;
const scoreLoadOwner = new RequestGeneration();

/**
 * Return whether a score load still owns the mounted score page.
 *
 * @param {number} requestGeneration - Generation captured before asynchronous reads.
 * @returns {boolean} True only for the newest load on the mounted route.
 */
function ownsMountedScoreLoad(requestGeneration) {
  return Boolean(
    scoreLoadOwner.owns(requestGeneration) &&
      document.getElementById("score-page-title")
  );
}

/**
 * Escape API-provided text before inserting it into templates.
 *
 * @param {unknown} value - Value to render as text.
 * @returns {string} HTML-escaped string.
 */
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/**
 * Format a benchmark timestamp as a localized calendar date.
 *
 * @param {string | null | undefined} value - ISO date/time from the API.
 * @returns {string} Localized date or missing-value dash.
 */
function formatDatasetDate(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleDateString(getLanguage());
}

/**
 * Show a score/report lifecycle message.
 *
 * @param {string} message - User-visible translated message.
 * @param {"info" | "error" | "success"} [tone] - Visual tone.
 * @returns {void}
 */
function showSummary(message, tone = "info") {
  const summary = document.getElementById("score-summary");
  if (!summary) {
    return;
  }
  summary.hidden = false;
  summary.className = `score-summary ${tone}`;
  summary.textContent = message;
}

/**
 * Hide and clear the score/report lifecycle message.
 *
 * @returns {void}
 */
function hideSummary() {
  const summary = document.getElementById("score-summary");
  if (summary) {
    summary.hidden = true;
    summary.textContent = "";
  }
}

/**
 * Assign translated text when an element exists on the mounted route.
 *
 * @param {string} id - Target DOM element ID.
 * @param {string} text - Text content to assign.
 * @returns {void}
 */
function setText(id, text) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = text;
  }
}

/**
 * Apply static score labels for the active English or Italian language.
 *
 * @returns {void}
 */
function renderLabels() {
  const labels = {
    "score-page-title": "score.title",
    "score-page-subtitle": "score.subtitle",
    "score-report-button": "score.calificationReport",
    "score-refresh-button": "score.refresh",
    "score-overview-title": "score.overviewTitle",
    "score-legend-best": "score.bestPeer",
    "score-legend-average": "score.peerAverage",
    "score-legend-company": "score.company",
    "score-chart-description": "score.chartDescription",
    "score-dimension-analysis-title": "score.dimensionAnalysis",
    "score-dimension-analysis-description": "score.dimensionAnalysisDescription"
  };
  Object.entries(labels).forEach(([id, key]) => setText(id, t(key)));
  document.querySelector(".score-legend")?.setAttribute("aria-label", t("score.legendLabel"));
}

/**
 * Return the browser resume key for one assessment/report language.
 *
 * @param {string} [language] - Report language associated with the job.
 * @returns {string} Stable localStorage key.
 */
function reportJobStorageKey(language = getLanguage()) {
  return `${REPORT_JOB_STORAGE_PREFIX}:${ASSESSMENT_ID}:${language}`;
}

/**
 * Set report button loading/disabled state to prevent duplicate jobs.
 *
 * @param {boolean} loading - Whether a report job is active.
 * @returns {void}
 */
function setReportButtonLoading(loading) {
  const button = document.getElementById("score-report-button");
  if (button) {
    button.loading = loading;
    button.disabled = loading;
  }
}

/**
 * Stop report polling and optionally discard the resumable browser job ID.
 *
 * @param {boolean} [removeStoredJob] - Whether the active job is terminal.
 * @returns {void}
 */
function stopReportPolling(removeStoredJob = false) {
  if (reportPollTimer) {
    window.clearTimeout(reportPollTimer);
    reportPollTimer = null;
  }
  if (removeStoredJob && activeReportLanguage) {
    localStorage.removeItem(reportJobStorageKey(activeReportLanguage));
  }
  activeReportJobId = null;
  activeReportLanguage = null;
  setReportButtonLoading(false);
}

/**
 * Translate one persisted report lifecycle status.
 *
 * @param {string} status - Backend report job status.
 * @returns {string} Localized nonnumeric progress message.
 */
function reportProgressMessage(status) {
  const keyByStatus = {
    pending: "score.reportQueued",
    generating: "score.reportGenerating",
    rendering: "score.reportRendering"
  };
  return keyByStatus[status]
    ? t(keyByStatus[status])
    : t("score.reportUnknownStatus");
}

/**
 * Download one completed PDF report through the existing resumable endpoint.
 *
 * @param {object} job - Completed job status with ID and filename.
 * @returns {Promise<void>} Promise resolved after browser download starts.
 */
async function downloadAssessmentReport(job) {
  const blob = await requestBlob(
    `/api/assessment/calification-reports/${encodeURIComponent(job.job_id)}/file`
  );
  const url = window.URL.createObjectURL(blob);
  const link = window.document.createElement("a");
  link.href = url;
  link.download = job.file_name || "calification-report.pdf";
  link.style.display = "none";
  window.document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => window.URL.revokeObjectURL(url), 0);
}

/**
 * Poll a report until it is downloadable, failed, or expired.
 *
 * @param {string} jobId - Persisted report job identifier.
 * @param {string} language - Language-specific browser owner of the job.
 * @returns {Promise<void>} Promise resolved after this polling attempt.
 */
async function pollAssessmentReport(jobId, language) {
  if (activeReportJobId !== jobId || activeReportLanguage !== language) {
    return;
  }
  try {
    const job = await request(
      `/api/assessment/calification-reports/${encodeURIComponent(jobId)}`
    );
    if (job.status === "completed" && job.download_ready) {
      try {
        await downloadAssessmentReport(job);
        showSummary(t("score.reportReady"), "success");
        stopReportPolling(true);
      } catch (error) {
        showSummary(t("score.reportDownloadFailed", { message: error.message }), "error");
        stopReportPolling(false);
      }
      return;
    }
    if (job.status === "failed") {
      showSummary(t("score.reportFailed"), "error");
      stopReportPolling(true);
      return;
    }
    if (!["pending", "generating", "rendering"].includes(job.status)) {
      showSummary(t("score.reportUnknownStatus"), "error");
      stopReportPolling(true);
      return;
    }
    showSummary(reportProgressMessage(job.status), "info");
  } catch (error) {
    if (error.status === 404 || error.message.includes("status: 404")) {
      showSummary(t("score.reportExpired"), "error");
      stopReportPolling(true);
      return;
    }
    showSummary(t("score.reportPollFailed", { message: error.message }), "error");
  }
  reportPollTimer = window.setTimeout(
    () => pollAssessmentReport(jobId, language),
    REPORT_POLL_INTERVAL_MS
  );
}

/**
 * Begin polling and persist the job ID for navigation/reload resumption.
 *
 * @param {string} jobId - Persisted report job identifier.
 * @param {string} language - Language associated with the immutable snapshot.
 * @returns {void}
 */
function startReportPolling(jobId, language) {
  if (reportPollTimer) {
    window.clearTimeout(reportPollTimer);
  }
  activeReportJobId = jobId;
  activeReportLanguage = language;
  localStorage.setItem(reportJobStorageKey(language), jobId);
  setReportButtonLoading(true);
  pollAssessmentReport(jobId, language);
}

/**
 * Resume a report job stored for the current assessment and language.
 *
 * @returns {void}
 */
function resumeStoredReport() {
  const language = getLanguage();
  const jobId = localStorage.getItem(reportJobStorageKey(language));
  if (jobId) {
    showSummary(t("score.reportResuming"), "info");
    startReportPolling(jobId, language);
  } else {
    setReportButtonLoading(false);
  }
}

/**
 * Fetch persisted profile and active identity-free selector options for Score.
 *
 * @returns {Promise<{scope: object, options: object, profile: object | null}>} Read-only profile context.
 */
async function fetchScoreProfileContext() {
  const [scope, options] = await Promise.all([
    request("/api/assessment/customer-class-scope"),
    request("/api/assessment/benchmark-options")
  ]);
  let profile = null;
  try {
    profile = await request(
      `/api/assessment/profile?assessment_id=${encodeURIComponent(ASSESSMENT_ID)}`
    );
  } catch (error) {
    if (error.status !== 404) {
      throw error;
    }
  }
  return { scope, options, profile };
}

/**
 * Return a localized label for a persisted normalized customer-class ID.
 *
 * @param {string | null | undefined} customerClass - Normalized class ID.
 * @param {object | null} [customerClassScope] - Backend-owned class label scope.
 * @returns {string} Localized label or a missing-profile fallback.
 */
function customerClassLabel(customerClass, customerClassScope = scoreState.customerClassScope) {
  return (
    customerClassOptions(customerClassScope, getLanguage()).find(
      (option) => option.value === customerClass
    )?.label ||
    customerClass ||
    t("score.profileMissing")
  );
}

/**
 * Snapshot and enqueue the current assessment by identity and language only.
 * The backend resolves the authoritative persisted class/NACE profile.
 *
 * @returns {Promise<void>} Promise resolved when report polling has started.
 */
async function createAssessmentReport() {
  if (activeReportJobId) {
    showSummary(reportProgressMessage("generating"), "info");
    return;
  }
  const language = getLanguage();
  const storedJobId = localStorage.getItem(reportJobStorageKey(language));
  if (storedJobId) {
    showSummary(t("score.reportResuming"), "info");
    startReportPolling(storedJobId, language);
    return;
  }

  setReportButtonLoading(true);
  showSummary(t("score.reportSubmitting"), "info");
  try {
    const requestBody = { assessment_id: ASSESSMENT_ID, language };
    const job = await request(
      "/api/assessment/calification-reports",
      "POST",
      requestBody
    );
    showSummary(t("score.reportQueued"), "info");
    startReportPolling(job.job_id, language);
  } catch (error) {
    setReportButtonLoading(false);
    showSummary(t("score.reportStartFailed", { message: error.message }), "error");
  }
}

/**
 * Destroy all canvases before rerendering or leaving a language-specific view.
 *
 * @returns {void}
 */
function destroyScoreCharts() {
  scoreState.chartInstances.forEach((chart) => chart.destroy());
  scoreState.chartInstances.clear();
}

/**
 * Create one noninteractive nested-bar chart and qualitative ARIA description.
 *
 * @param {string} canvasId - Mounted canvas element identifier.
 * @param {object[]} items - Dimension or applicable topic view-model items.
 * @returns {void}
 */
function renderNestedBarChart(canvasId, items) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || items.length === 0) {
    return;
  }
  canvas.setAttribute("aria-label", scoreAriaDescription(items, t));
  const chart = new Chart(
    canvas,
    buildNestedBarChartConfig(items, {
      company: t("score.company"),
      peerAverage: t("score.peerAverage"),
      bestPeer: t("score.bestPeer")
    })
  );
  scoreState.chartInstances.set(canvasId, chart);
}

/**
 * Render the seven-dimension overview chart and qualitative commentary cards.
 *
 * @param {object[]} dimensions - Seven dimension view-model groups.
 * @returns {void}
 */
function renderOverview(dimensions) {
  const positioning = document.getElementById("score-overview-positioning");
  const viewport = document.getElementById("score-overview-chart-viewport");
  if (positioning) {
    positioning.innerHTML = qualitativeScoreMarkup(dimensions, t);
  }
  if (viewport) {
    viewport.style.height = `${Math.max(22, dimensions.length * 3.4)}rem`;
  }
  renderNestedBarChart("score-overview-chart", dimensions);
}

/**
 * Render keyboard-expandable topic analysis for every available dimension.
 *
 * @param {object[]} dimensions - Dimension groups with applicable question topics.
 * @returns {void}
 */
function renderDimensionAnalysis(dimensions) {
  const container = document.getElementById("score-dimension-sections");
  if (!container) {
    return;
  }
  container.innerHTML = dimensions
    .map((dimension, index) => {
      const position = t(`score.position.${dimension.benchmark.positioning}`);
      const canvasId = `score-topic-chart-${index}`;
      return `
        <details class="score-dimension-card" data-dimension-index="${index}">
          <summary>
            <span>${escapeHtml(dimension.label)}</span>
            <span class="position-badge position-${escapeHtml(dimension.benchmark.positioning)}">${escapeHtml(position)}</span>
          </summary>
          <div class="score-dimension-content">
            ${
              dimension.topics.length
                ? `<div class="score-topic-chart-viewport" style="height: ${Math.max(11, dimension.topics.length * 3.4)}rem"><canvas id="${canvasId}" role="img"></canvas></div>${qualitativeScoreMarkup(dimension.topics, t)}`
                : `<p class="score-topic-empty">${escapeHtml(t("score.noApplicableTopics"))}</p>`
            }
          </div>
        </details>
      `;
    })
    .join("");

  // Initialize each responsive canvas only after its native details element is
  // open, because a collapsed canvas has no measurable width in the browser.
  container.querySelectorAll("[data-dimension-index]").forEach((details) => {
    details.addEventListener("toggle", () => {
      const index = Number(details.dataset.dimensionIndex);
      const dimension = dimensions[index];
      if (details.open && dimension?.topics.length) {
        const canvasId = `score-topic-chart-${index}`;
        if (!scoreState.chartInstances.has(canvasId)) {
          renderNestedBarChart(canvasId, dimension.topics);
        }
      }
    });
  });
}

/**
 * Render benchmark availability, positioning, and dataset date.
 *
 * @param {object} payload - Assessment score response with benchmark context.
 * @returns {void}
 */
function renderBenchmarkStatus(payload) {
  const container = document.getElementById("score-benchmark-status");
  if (!container) {
    return;
  }
  const context = payload.benchmark_context || { available: false, reason: "no_active_dataset" };
  if (!context.available) {
    const reasonKey = {
      no_active_dataset: "score.unavailableNoDataset",
      missing_profile_context: "score.unavailableNoProfile",
      insufficient_peer_sample: "score.unavailableSmallCohort"
    }[context.reason] || "score.unavailableGeneric";
    container.className = "score-benchmark-status unavailable";
    container.innerHTML = `
      <strong>${escapeHtml(t("score.benchmarkUnavailable"))}</strong>
      <span>${escapeHtml(t(reasonKey))}</span>
      ${
        context.dataset_activated_at
          ? `<span>${escapeHtml(t("score.datasetDate", { datasetDate: formatDatasetDate(context.dataset_activated_at) }))}</span>`
          : ""
      }
    `;
    return;
  }
  const positioning = t(`score.position.${payload.benchmark?.positioning || "unavailable"}`);
  container.className = "score-benchmark-status";
  container.innerHTML = `
    <strong>${escapeHtml(positioning)}</strong>
    <span>${escapeHtml(
      t("score.benchmarkAvailableDetail", {
        datasetDate: formatDatasetDate(context.dataset_activated_at)
      })
    )}</span>
  `;
}

/**
 * Load profile-aware score data and render every deterministic UI section.
 *
 * @returns {Promise<boolean>} True when this load rendered, false when stale or failed.
 */
async function loadScore() {
  const requestGeneration = scoreLoadOwner.begin();
  const language = getLanguage();
  try {
    const profileContext = await fetchScoreProfileContext();
    const payload = await request(
      `/api/assessment/score?assessment_id=${encodeURIComponent(ASSESSMENT_ID)}&language=${encodeURIComponent(language)}`
    );
    if (!ownsMountedScoreLoad(requestGeneration)) {
      return false;
    }

    // Canvas ownership changes only after every read has completed and this
    // load is confirmed as newest. A late response can therefore neither
    // destroy current charts nor create a second Chart for the same canvas.
    destroyScoreCharts();
    scoreState.customerClassScope = profileContext.scope;
    scoreState.benchmarkOptions = profileContext.options;
    scoreState.profile = profileContext.profile;
    scoreState.payload = payload;
    if (!activeReportJobId) {
      hideSummary();
    }
    const view = buildScoreViewModel(payload);
    setText(
      "score-context",
      profileContext.profile
        ? t("score.context", {
            customerClass: customerClassLabel(
              profileContext.profile.customer_class,
              profileContext.scope
            ),
            nace1: profileContext.profile.nace1
          })
        : t("score.contextUnavailable")
    );
    renderBenchmarkStatus(payload);
    renderOverview(view.dimensions);
    renderDimensionAnalysis(view.dimensions);
    return true;
  } catch (error) {
    if (ownsMountedScoreLoad(requestGeneration)) {
      showSummary(t("score.loadingError", { message: error.message }), "error");
    }
    return false;
  }
}

/**
 * Initialize score rendering and preserve existing report polling/download flow.
 *
 * @returns {void}
 */
export default function initScorePage() {
  // Revoke pending work from the previous route mount before touching its
  // canvases or scheduling a new localized score load.
  scoreLoadOwner.invalidate();
  renderLabels();
  if (!languageChangeListenerAttached) {
    document.addEventListener("language-change", () => {
      if (!document.getElementById("score-page-title")) {
        return;
      }
      stopReportPolling(false);
      renderLabels();
      loadScore();
      resumeStoredReport();
    });
    languageChangeListenerAttached = true;
  }
  document.getElementById("score-report-button")?.addEventListener("click", () => {
    createAssessmentReport();
  });
  document.getElementById("score-refresh-button")?.addEventListener("click", () => {
    loadScore();
  });
  loadScore();
  resumeStoredReport();
}
