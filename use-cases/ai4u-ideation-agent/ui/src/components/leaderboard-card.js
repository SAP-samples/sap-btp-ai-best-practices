/**
 * Shared Leaderboard Card Component
 *
 * Provides a unified expandable card for leaderboard entries,
 * used by both the company leaderboard dialog and the event page leaderboard.
 */

import "./leaderboard-card.css";

// --- Private Helpers ---

function _escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function _getScoreColor(score) {
  if (score >= 800) return "#1d7d39";
  if (score >= 600) return "#0070f2";
  if (score >= 400) return "#e76500";
  return "#bb0000";
}

function _relativeTime(timestamp) {
  if (!timestamp) return "";
  const diff = Date.now() - new Date(timestamp).getTime();
  if (diff < 0) return "just now";
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function _fullTimestamp(timestamp) {
  if (!timestamp) return "";
  const d = new Date(timestamp);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }) +
    " at " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function _renderStars(rating) {
  const max = 5;
  const filled = Math.min(Math.max(Math.round(rating || 0), 0), max);
  let html = "";
  for (let i = 1; i <= max; i++) {
    html += `<ui5-icon name="favorite" class="lb-card__star ${i <= filled ? "lb-card__star--filled" : "lb-card__star--empty"}"></ui5-icon>`;
  }
  return html;
}

function _getRankClass(rank) {
  if (rank === 1) return "lb-card__rank--gold";
  if (rank === 2) return "lb-card__rank--silver";
  if (rank === 3) return "lb-card__rank--bronze";
  return "";
}

/**
 * Truncate long text to a maximum number of characters,
 * breaking at the nearest sentence boundary.
 */
function _truncate(str, maxLen = 300) {
  if (!str || str.length <= maxLen) return str;
  // Find last sentence end before maxLen
  const sub = str.slice(0, maxLen);
  const lastDot = sub.lastIndexOf(". ");
  const lastNewline = sub.lastIndexOf("\n");
  const breakAt = Math.max(lastDot, lastNewline);
  if (breakAt > maxLen * 0.4) {
    return str.slice(0, breakAt + 1).trim() + "…";
  }
  return sub.trim() + "…";
}

/**
 * Extract a concise summary from a structured solution text.
 * The solution field often contains sections like "WHAT IT DOES:", "TRIGGER:", etc.
 * We only show the opening description for the leaderboard.
 */
function _summarizeSolution(text) {
  if (!text) return "";
  // If it starts with "WHAT IT DOES:" extract just that section
  const whatMatch = text.match(/^WHAT IT DOES:\s*([\s\S]*?)(?:\n\s*\n|\nTRIGGER:|\nDATA SOURCE|\nPROCESSING|\nUSER ROLE|\nOUTPUT)/i);
  if (whatMatch) {
    return _truncate(whatMatch[1].trim(), 300);
  }
  // Otherwise truncate the whole text
  return _truncate(text, 300);
}

// --- Public API ---

/**
 * Normalize a leaderboard entry from either API format (camelCase or snake_case)
 * into a common shape.
 *
 * @param {object} raw - Raw entry from API
 * @param {"company"|"event"} format - Which API format
 * @returns {object} Normalized entry
 */
export function normalizeEntry(raw, format = "company") {
  if (format === "event") {
    return {
      rank: raw.rank,
      displayName: raw.display_name || "",
      email: null,
      industry: raw.industry || "",
      title: raw.title || "",
      problem: raw.problem || "",
      solution: raw.solution || "",
      appType: raw.app_type || "",
      feedback: raw.evaluator_feedback || "",
      detailedAnalysis: raw.evaluator_detailed_analysis || "",
      suggestions: raw.evaluator_suggestions || [],
      appropriatenessLevel: raw.appropriateness_level || "",
      score: raw.score || 0,
      rating: raw.rating || 0,
      timestamp: raw.timestamp || "",
      contextId: raw.context_id || "",
      isCurrentUser: !!raw.is_current_user,
    };
  }

  // Company format (camelCase)
  return {
    rank: raw.rank,
    displayName: raw.displayName || "",
    email: raw.email || null,
    industry: raw.industry || "",
    title: raw.title || "",
    problem: raw.problem || "",
    solution: raw.solution || "",
    appType: raw.app_type || raw.appType || "",
    feedback: raw.evaluatorFeedback || "",
    detailedAnalysis: raw.evaluatorDetailedAnalysis || "",
    suggestions: raw.evaluatorSuggestions || [],
    appropriatenessLevel: raw.appropriatenessLevel || "",
    score: raw.score || 0,
    rating: raw.rating || 0,
    timestamp: raw.timestamp || "",
    contextId: raw.contextId || "",
    isCurrentUser: !!raw.isCurrentUser,
  };
}

/**
 * Render a single leaderboard entry card as HTML string.
 *
 * @param {object} entry - Normalized entry (from normalizeEntry)
 * @param {object} [options]
 * @param {boolean} [options.highlight] - Whether to mark this card as highlighted
 * @returns {string} HTML string
 */
export function renderLeaderboardEntry(entry, options = {}) {
  const rankClass = _getRankClass(entry.rank);
  const currentClass = entry.isCurrentUser ? "lb-card--current" : "";
  const highlightClass = options.highlight ? "lb-card--highlight" : "";

  const youBadge = entry.isCurrentUser ? `<span class="lb-card__you-badge">you</span>` : "";
  const appTypePill = entry.appType
    ? `<span class="lb-card__app-type">${_escapeHtml(entry.appType)}</span>`
    : "";
  const timeText = entry.timestamp
    ? `<span class="lb-card__time">${_relativeTime(entry.timestamp)}</span>`
    : "";

  // Detail sections (only render non-empty fields)
  let detailSections = "";

  if (entry.problem) {
    detailSections += `
      <div class="lb-card__section">
        <span class="lb-card__label">Problem</span>
        <span class="lb-card__text">${_escapeHtml(entry.problem)}</span>
      </div>`;
  }

  if (entry.industry) {
    detailSections += `
      <div class="lb-card__section">
        <span class="lb-card__label">Industry</span>
        <span class="lb-card__industry-badge">${_escapeHtml(entry.industry)}</span>
      </div>`;
  }

  if (entry.solution) {
    detailSections += `
      <div class="lb-card__section">
        <span class="lb-card__label">Solution</span>
        <span class="lb-card__text">${_escapeHtml(entry.solution)}</span>
      </div>`;
  }

  if (entry.feedback) {
    detailSections += `
      <div class="lb-card__section lb-card__section--evaluation">
        <span class="lb-card__label">Evaluation</span>
        <span class="lb-card__text">${_escapeHtml(entry.feedback)}</span>
      </div>`;
  }

  if (entry.suggestions && entry.suggestions.length > 0) {
    const items = entry.suggestions.map((s) => `<li>${_escapeHtml(s)}</li>`).join("");
    detailSections += `
      <div class="lb-card__section">
        <span class="lb-card__label">Suggestions</span>
        <ul class="lb-card__suggestions">${items}</ul>
      </div>`;
  }

  if (entry.rating > 0) {
    detailSections += `
      <div class="lb-card__section">
        <span class="lb-card__label">Rating</span>
        <span class="lb-card__stars">${_renderStars(entry.rating)}</span>
      </div>`;
  }

  if (entry.timestamp) {
    detailSections += `
      <div class="lb-card__section">
        <span class="lb-card__label">Submitted</span>
        <span class="lb-card__text">${_fullTimestamp(entry.timestamp)}</span>
      </div>`;
  }

  return `
    <div class="lb-card ${currentClass} ${highlightClass}" data-context-id="${_escapeHtml(entry.contextId)}">
      <div class="lb-card__header">
        <div class="lb-card__rank">
          <span class="lb-card__rank-badge ${rankClass}">${entry.rank}</span>
        </div>
        <div class="lb-card__info">
          <span class="lb-card__name">${_escapeHtml(entry.displayName)}${youBadge}</span>
          <span class="lb-card__title">${_escapeHtml(entry.title)}</span>
        </div>
        <div class="lb-card__meta">
          ${appTypePill}
          ${timeText}
        </div>
        <span class="lb-card__score" style="color: ${_getScoreColor(entry.score)}">${entry.score}</span>
        <ui5-icon name="slim-arrow-down" class="lb-card__chevron"></ui5-icon>
      </div>
      <div class="lb-card__detail">
        <div class="lb-card__detail-inner">
          <div class="lb-card__detail-content">
            ${detailSections}
          </div>
        </div>
      </div>
    </div>`;
}

/**
 * Attach accordion expand/collapse interactions to a container of leaderboard cards.
 * Only one card can be expanded at a time.
 *
 * @param {HTMLElement} containerEl - Container holding .lb-card elements
 */
export function attachLeaderboardInteractions(containerEl) {
  containerEl.addEventListener("click", (e) => {
    const header = e.target.closest(".lb-card__header");
    if (!header) return;

    const card = header.closest(".lb-card");
    if (!card) return;

    const isExpanding = !card.classList.contains("lb-card--expanded");

    // Collapse all others (accordion behavior)
    containerEl.querySelectorAll(".lb-card--expanded").forEach((other) => {
      if (other !== card) other.classList.remove("lb-card--expanded");
    });

    if (isExpanding) {
      card.classList.add("lb-card--expanded");
    } else {
      card.classList.remove("lb-card--expanded");
    }
  });
}
