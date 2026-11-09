/**
 * Admin dashboard page — displays aggregated usage stats on a single page.
 * Uses a tab bar to separate Dashboard vs Telemetry (SAP-hosted only).
 */
import "@ui5/webcomponents/dist/TabContainer.js";
import "@ui5/webcomponents/dist/Tab.js";
import { getStats } from "../../services/admin.js";

export default async function initAdminPage() {
  const loadingEl = document.getElementById("admin-loading");
  const deniedEl = document.getElementById("admin-denied");
  const contentEl = document.getElementById("admin-content");

  try {
    const stats = await getStats();
    loadingEl.hidden = true;
    contentEl.hidden = false;

    // ── Overview KPIs ─────────────────────────────────────────────────────
    document.getElementById("stat-total-sessions").textContent =
      stats.totalSessions.toLocaleString();
    document.getElementById("stat-unique-users").textContent =
      stats.uniqueUsers.toLocaleString();
    document.getElementById("stat-avg-messages").textContent =
      stats.avgMessagesPerSession.toFixed(1);
    document.getElementById("stat-completion-rate").textContent =
      `${(stats.completionRate * 100).toFixed(0)}%`;

    // Render time series chart (default: daily)
    _renderChart(stats.sessionsPerDay, "daily", "sessions-chart");

    // Toggle daily/weekly
    const toggle = document.getElementById("time-range-toggle");
    if (toggle) {
      toggle.addEventListener("selection-change", (e) => {
        const selected = e.detail.selectedItems?.[0]?.textContent?.trim();
        if (selected === "Weekly") {
          _renderChart(stats.sessionsPerWeek, "weekly", "sessions-chart");
        } else {
          _renderChart(stats.sessionsPerDay, "daily", "sessions-chart");
        }
      });
    }

    // Render top companies
    _renderCompanies(stats.topCompanies);

    // ── Users ───────────────────────────────────────────────────────────────
    if (stats.users && (stats.users.newUsers > 0 || stats.users.returningUsers > 0)) {
      const usersSection = document.getElementById("users-section");
      if (usersSection) usersSection.hidden = false;

      document.getElementById("stat-new-users").textContent =
        stats.users.newUsers.toLocaleString();
      document.getElementById("stat-returning-users").textContent =
        stats.users.returningUsers.toLocaleString();
      document.getElementById("stat-total-companies").textContent =
        stats.users.totalCompanies.toLocaleString();

      _renderHorizontalBars("engagement-bands", stats.users.engagementBands, "band", "count");
      _renderRankedList("top-user-companies", stats.users.topUserCompanies, "company", "userCount");
    }

    // ── Outcomes & Quality ──────────────────────────────────────────────
    if (stats.outcomes && stats.outcomes.totalSubmissions > 0) {
      const outcomesSection = document.getElementById("outcomes-section");
      if (outcomesSection) outcomesSection.hidden = false;

      document.getElementById("stat-total-submissions").textContent =
        stats.outcomes.totalSubmissions.toLocaleString();
      document.getElementById("stat-avg-score").textContent =
        stats.outcomes.avgScore.toLocaleString();
      document.getElementById("stat-score-range").textContent =
        `${stats.outcomes.minScore} – ${stats.outcomes.maxScore}`;

      _renderHorizontalBars("score-distribution", stats.outcomes.scoreDistribution, "band", "count");
      _renderAppTypeSplit(stats.outcomes.appTypeSplit);
      _renderRankedList("top-industries", stats.outcomes.topIndustries, "industry");
      _renderChart(stats.outcomes.submissionsPerDay, "daily", "submissions-chart");
    }

    // ── Collaboration & Events ──────────────────────────────────────────
    if (stats.collaboration && (stats.collaboration.totalSharedConversations > 0 || stats.collaboration.totalEvents > 0)) {
      const collabSection = document.getElementById("collaboration-section");
      if (collabSection) collabSection.hidden = false;

      document.getElementById("stat-shared-conversations").textContent =
        stats.collaboration.totalSharedConversations.toLocaleString();
      document.getElementById("stat-share-members").textContent =
        stats.collaboration.totalShareMembers.toLocaleString();
      document.getElementById("stat-total-events").textContent =
        stats.collaboration.totalEvents.toLocaleString();
      document.getElementById("stat-active-events").textContent =
        stats.collaboration.activeEvents.toLocaleString();
      document.getElementById("stat-event-members").textContent =
        stats.collaboration.totalEventMembers.toLocaleString();

      _renderTopEvents(stats.collaboration.topEvents);
    }

    // ── Telemetry (SAP-hosted instance only, non-blocking) ────────────────
    _loadTelemetrySection();
  } catch (err) {
    loadingEl.hidden = true;
    if (err?.message?.includes("403")) {
      deniedEl.hidden = false;
    } else {
      deniedEl.hidden = false;
      console.error("Failed to load admin stats:", err);
    }
  }
}

// ── Chart Helpers ─────────────────────────────────────────────────────────────

/**
 * Render a vertical bar chart from time-series data.
 * Each bar has a count label on top and a date label below.
 */
function _renderChart(data, mode, containerId) {
  const container = document.getElementById(containerId);
  if (!container || !data || data.length === 0) {
    if (container) container.innerHTML = `<div class="companies-empty"><ui5-text>No data for this period.</ui5-text></div>`;
    return;
  }

  const maxCount = Math.max(...data.map((d) => d.count), 1);

  container.innerHTML = data
    .map((item) => {
      const height = Math.max((item.count / maxCount) * 100, 1);
      const label = mode === "daily" ? _formatDayLabel(item.date) : item.week;
      return `
      <div class="chart-bar-wrapper" title="${item.date || item.week}: ${item.count}">
        <ui5-text class="chart-count">${item.count}</ui5-text>
        <div class="chart-bar" style="height: ${height}%"></div>
        <ui5-text class="chart-label">${label}</ui5-text>
      </div>`;
    })
    .join("");
}

/**
 * Format "2026-07-22" → "Jul 22"
 */
function _formatDayLabel(dateStr) {
  if (!dateStr) return "";
  const parts = dateStr.split("-");
  if (parts.length !== 3) return dateStr;
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const monthIdx = parseInt(parts[1], 10) - 1;
  const day = parseInt(parts[2], 10);
  return `${months[monthIdx]} ${day}`;
}

// ── List/Bar Helpers ──────────────────────────────────────────────────────────

/**
 * Render the top companies list with horizontal bars.
 */
function _renderCompanies(companies) {
  const container = document.getElementById("top-companies");
  const emptyEl = document.getElementById("companies-empty");

  if (!companies || companies.length === 0) {
    if (container) container.hidden = true;
    if (emptyEl) emptyEl.hidden = false;
    return;
  }

  const maxCount = companies[0]?.count || 1;

  container.innerHTML = companies
    .map((c, i) => {
      const barWidth = Math.max((c.count / maxCount) * 100, 2);
      return `
      <div class="company-row">
        <ui5-text class="company-rank">${i + 1}</ui5-text>
        <ui5-text class="company-name">${_escapeHtml(c.company)}</ui5-text>
        <div class="company-bar-container">
          <div class="company-bar" style="width: ${barWidth}%"></div>
        </div>
        <ui5-text class="company-count">${c.count}</ui5-text>
      </div>`;
    })
    .join("");
}

/**
 * Render horizontal bar chart (reusable for score bands, appropriateness, etc.).
 */
function _renderHorizontalBars(containerId, data, labelKey, countKey) {
  const container = document.getElementById(containerId);
  if (!container || !data || data.length === 0) return;

  const maxCount = Math.max(...data.map((d) => d[countKey]), 1);

  container.innerHTML = data
    .map((item) => {
      const barWidth = Math.max((item[countKey] / maxCount) * 100, 2);
      return `
      <div class="company-row">
        <ui5-text class="company-name">${_escapeHtml(String(item[labelKey]))}</ui5-text>
        <div class="company-bar-container">
          <div class="company-bar" style="width: ${barWidth}%"></div>
        </div>
        <ui5-text class="company-count">${item[countKey]}</ui5-text>
      </div>`;
    })
    .join("");
}

/**
 * Render app type split as KPI cards.
 */
function _renderAppTypeSplit(data) {
  const container = document.getElementById("app-type-split");
  if (!container || !data || data.length === 0) return;

  container.innerHTML = data
    .map((item) => `
      <div class="kpi-card">
        <ui5-title level="H1" class="kpi-value">${item.count.toLocaleString()}</ui5-title>
        <ui5-text class="kpi-label">${_escapeHtml(item.type)}</ui5-text>
      </div>`)
    .join("");
}

/**
 * Render a ranked list (industries, companies, etc.) with bars.
 */
function _renderRankedList(containerId, data, labelKey, countKey = "count") {
  const container = document.getElementById(containerId);
  if (!container || !data || data.length === 0) return;

  const maxCount = data[0]?.[countKey] || 1;

  container.innerHTML = data
    .map((item, i) => {
      const barWidth = Math.max((item[countKey] / maxCount) * 100, 2);
      return `
      <div class="company-row">
        <ui5-text class="company-rank">${i + 1}</ui5-text>
        <ui5-text class="company-name">${_escapeHtml(String(item[labelKey]))}</ui5-text>
        <div class="company-bar-container">
          <div class="company-bar" style="width: ${barWidth}%"></div>
        </div>
        <ui5-text class="company-count">${item[countKey]}</ui5-text>
      </div>`;
    })
    .join("");
}

/**
 * Render top events by member count.
 */
function _renderTopEvents(events) {
  const container = document.getElementById("top-events");
  const emptyEl = document.getElementById("events-empty");

  if (!events || events.length === 0) {
    if (container) container.hidden = true;
    if (emptyEl) emptyEl.hidden = false;
    return;
  }

  const maxCount = events[0]?.memberCount || 1;

  container.innerHTML = events
    .map((item, i) => {
      const barWidth = Math.max((item.memberCount / maxCount) * 100, 2);
      return `
      <div class="company-row">
        <ui5-text class="company-rank">${i + 1}</ui5-text>
        <ui5-text class="company-name">${_escapeHtml(item.name)}</ui5-text>
        <div class="company-bar-container">
          <div class="company-bar" style="width: ${barWidth}%"></div>
        </div>
        <ui5-text class="company-count">${item.memberCount}</ui5-text>
      </div>`;
    })
    .join("");
}

// ── Telemetry Section ────────────────────────────────────────────────────────

/**
 * Load telemetry data (SAP-hosted only). On success, shows the tab bar and
 * populates the telemetry panel. On 404/error, tab bar stays hidden.
 */
async function _loadTelemetrySection() {
  try {
    const { getTelemetryData } = await import("../../services/admin.js");
    const telemetry = await getTelemetryData();
    if (!telemetry || !telemetry.summary) return;

    // Show the tab bar now that telemetry is available
    const tabBar = document.getElementById("admin-tabs");
    if (tabBar) tabBar.hidden = false;

    // Wire up tab switching
    _initTabSwitching();

    // KPI cards
    document.getElementById("stat-total-installations").textContent =
      telemetry.summary.totalInstallations.toLocaleString();
    document.getElementById("stat-active-installations").textContent =
      telemetry.summary.activeInstallations7d.toLocaleString();
    document.getElementById("stat-total-conversations-all").textContent =
      telemetry.summary.totalConversationsAll.toLocaleString();
    document.getElementById("stat-total-users-all").textContent =
      telemetry.summary.totalUsersAll.toLocaleString();

    // Version distribution
    if (telemetry.summary.versionDistribution?.length) {
      _renderHorizontalBars(
        "telemetry-version-dist",
        telemetry.summary.versionDistribution,
        "version",
        "count",
      );
    }

    // Growth chart (aggregate daily unique installations)
    if (telemetry.daily?.length) {
      const dailyAgg = _aggregateDailyInstallations(telemetry.daily);
      _renderChart(dailyAgg, "daily", "telemetry-growth-chart");
    }

    // Installations list
    if (telemetry.installations?.length) {
      _renderInstallationsList(telemetry.installations);
    }
  } catch {
    // Endpoint returned 404 (not SAP-hosted) or failed — tab bar stays hidden
  }
}

/**
 * Wire up tab-select event to toggle between Dashboard and Telemetry panels.
 */
function _initTabSwitching() {
  const tabBar = document.getElementById("admin-tabs");
  if (!tabBar || tabBar.dataset.wired) return;
  tabBar.dataset.wired = "true";

  tabBar.addEventListener("tab-select", (e) => {
    const selectedTab = e.detail?.tab;
    const dashPanel = document.getElementById("admin-panel-dashboard");
    const telPanel = document.getElementById("admin-panel-telemetry");

    if (selectedTab?.id === "admin-tab-telemetry") {
      if (dashPanel) dashPanel.hidden = true;
      if (telPanel) telPanel.hidden = false;
    } else {
      if (dashPanel) dashPanel.hidden = false;
      if (telPanel) telPanel.hidden = true;
    }
  });
}

/**
 * Aggregate daily snapshots into unique installations per day for the chart.
 */
function _aggregateDailyInstallations(daily) {
  const byDate = {};
  for (const row of daily) {
    const d = row.snapshotDate;
    if (!byDate[d]) byDate[d] = 0;
    byDate[d]++;
  }
  return Object.entries(byDate)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, count]) => ({ date, count }));
}

/**
 * Render the installations list showing ID, version, stats, and last seen.
 */
function _renderInstallationsList(installations) {
  const container = document.getElementById("telemetry-installations-list");
  if (!container || !installations.length) return;

  container.innerHTML = installations
    .map((inst) => {
      const lastSeen = inst.lastSeen
        ? new Date(inst.lastSeen).toLocaleDateString()
        : "—";
      const idDisplay =
        inst.installationId.length > 20
          ? inst.installationId.substring(0, 20) + "…"
          : inst.installationId;
      return `
      <div class="company-row" title="${_escapeHtml(inst.installationId)}">
        <ui5-text class="company-name" style="flex:2">${_escapeHtml(idDisplay)}</ui5-text>
        <ui5-text class="company-count" style="min-width:3.5rem">${_escapeHtml(inst.currentVersion || "?")}</ui5-text>
        <ui5-text class="company-count">${(inst.conversations || 0).toLocaleString()} conv</ui5-text>
        <ui5-text class="company-count">${(inst.users || 0).toLocaleString()} users</ui5-text>
        <ui5-text class="company-count" style="min-width:5rem">${lastSeen}</ui5-text>
      </div>`;
    })
    .join("");
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function _escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
