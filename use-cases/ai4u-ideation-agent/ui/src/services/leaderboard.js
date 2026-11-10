import { request } from "./api.js";
import { normalizeEntry, renderLeaderboardEntry, attachLeaderboardInteractions } from "../components/leaderboard-card.js";

/**
 * Get the company leaderboard for the current user's corporate domain.
 * @param {number} [limit=50]
 * @returns {Promise<{companyDomain: string, totalEntries: number, entries: Array}>}
 */
export async function getCompanyLeaderboard(limit = 50) {
  return request(`/api/ideation-agent/leaderboard/company?limit=${limit}`);
}

function _escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Open the Company Leaderboard dialog. Works from any page.
 * @param {{ contextId?: string, score?: number } | null} highlightEntry
 */
export async function openLeaderboardDialog(highlightEntry = null) {
  const existingDialog = document.getElementById("leaderboard-dialog");
  if (existingDialog) existingDialog.remove();

  const dialog = document.createElement("ui5-dialog");
  dialog.id = "leaderboard-dialog";
  dialog.setAttribute("accessible-name", "Company Leaderboard");
  dialog.style.cssText = "min-width: 600px; max-width: 750px;";

  dialog.innerHTML = `
      <div slot="header" class="leaderboard-dialog-header">
          <span class="leaderboard-dialog-title">Company Leaderboard</span>
          <ui5-button id="leaderboard-close-btn" icon="decline" design="Transparent" tooltip="Close"></ui5-button>
      </div>
      <div class="leaderboard-content">
          <div class="leaderboard-loading">
              <ui5-busy-indicator active size="M" delay="0"></ui5-busy-indicator>
              <span>Loading leaderboard...</span>
          </div>
      </div>
  `;

  document.body.appendChild(dialog);
  dialog.open = true;

  dialog.querySelector("#leaderboard-close-btn").addEventListener("click", () => {
      dialog.open = false;
  });
  dialog.addEventListener("close", () => {
      document.body.removeChild(dialog);
  });

  try {
      const data = await getCompanyLeaderboard(50);
      const content = dialog.querySelector(".leaderboard-content");

      if (!data || !data.entries?.length) {
          content.innerHTML = `
              <div class="leaderboard-empty">
                  <ui5-icon name="competitor" class="leaderboard-empty-icon"></ui5-icon>
                  <span>No submissions yet for your company.</span>
                  <span class="leaderboard-hint">Be the first to submit an evaluated solution!</span>
              </div>
          `;
          return;
      }

      const cards = data.entries.map((entry) => {
          const normalized = normalizeEntry(entry, "company");
          const isHighlight = highlightEntry
              && entry.contextId === highlightEntry.contextId
              && entry.score === highlightEntry.score;
          return renderLeaderboardEntry(normalized, { highlight: isHighlight });
      }).join("");

      content.innerHTML = `
          <div class="leaderboard-header-info">
              <span class="lb-domain">${_escapeHtml(data.companyDomain)}</span>
              <span class="lb-count">${data.totalEntries} submission${data.totalEntries !== 1 ? "s" : ""}</span>
          </div>
          <div class="lb-card-list">${cards}</div>
      `;

      attachLeaderboardInteractions(content.querySelector(".lb-card-list"));

      // Scroll highlighted entry into view
      if (highlightEntry) {
          const highlightEl = content.querySelector(".lb-card--highlight");
          if (highlightEl) {
              setTimeout(() => {
                  highlightEl.scrollIntoView({ behavior: "smooth", block: "center" });
              }, 150);
          }
      }
  } catch (err) {
      const content = dialog.querySelector(".leaderboard-content");
      if (err?.message?.includes("403")) {
          content.innerHTML = `
              <div class="leaderboard-empty">
                  <ui5-icon name="locked" class="leaderboard-empty-icon"></ui5-icon>
                  <span>Company leaderboard is only available for corporate email users.</span>
              </div>
          `;
      } else {
          content.innerHTML = `
              <div class="leaderboard-empty">
                  <span>Failed to load leaderboard. Please try again.</span>
              </div>
          `;
      }
  }
}

/**
 * Open a detail popup for a single leaderboard entry.
 * Reuses the same dialog structure and CSS as openLeaderboardDialog.
 * @param {object} rawEntry - Raw entry in company API format
 * @param {Function} [onReturnToList] - Optional callback; when provided a
 *   "Return to list" button appears in the footer. Calling it closes the
 *   dialog and lets the caller handle the side effect (e.g. remove a
 *   canvas tile and re-add the card to the sidebar).
 */
export function openEntryDetailDialog(rawEntry, onReturnToList) {
    const existing = document.getElementById("entry-detail-dialog");
    if (existing) existing.remove();

    const dialog = document.createElement("ui5-dialog");
    dialog.id = "entry-detail-dialog";
    dialog.setAttribute("accessible-name", rawEntry.title || "Entry Detail");
    dialog.style.cssText = "min-width: 600px; max-width: 750px;";

    const returnBtn = onReturnToList
        ? `<ui5-button id="entry-detail-return-btn" icon="undo" design="Negative">Return to list</ui5-button>`
        : "";

    dialog.innerHTML = `
        <div slot="header" class="leaderboard-dialog-header">
            <span class="leaderboard-dialog-title">${_escapeHtml(rawEntry.title || "Entry Detail")}</span>
            <ui5-button id="entry-detail-close-btn" icon="decline" design="Transparent" tooltip="Close"></ui5-button>
        </div>
        <div class="leaderboard-content">
            <div class="lb-card-list"></div>
        </div>
        ${returnBtn ? `<div slot="footer" class="leaderboard-dialog-footer">${returnBtn}</div>` : ""}
    `;

    document.body.appendChild(dialog);
    dialog.open = true;

    dialog.querySelector("#entry-detail-close-btn").addEventListener("click", () => {
        dialog.open = false;
    });
    dialog.addEventListener("close", () => dialog.remove());

    if (onReturnToList) {
        dialog.querySelector("#entry-detail-return-btn").addEventListener("click", () => {
            dialog.open = false;
            onReturnToList();
        });
    }

    const normalized = normalizeEntry(rawEntry, "company");
    const list = dialog.querySelector(".lb-card-list");
    list.innerHTML = renderLeaderboardEntry(normalized, {});
    attachLeaderboardInteractions(list);

    // Pre-expand so full detail is immediately visible
    const card = list.querySelector(".lb-card");
    if (card) card.classList.add("lb-card--expanded");
}
