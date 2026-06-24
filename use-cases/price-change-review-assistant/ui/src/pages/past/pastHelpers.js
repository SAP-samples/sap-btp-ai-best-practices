export const HISTORY_PAGE_SIZE = 50;

/**
 * Build the API endpoint for one historical request page.
 *
 * @param {{ limit?: number, offset?: number }} page Page size and row offset.
 * @returns {string} Price-change history endpoint with encoded pagination parameters.
 */
export function buildHistoryEndpoint({ limit = HISTORY_PAGE_SIZE, offset = 0 } = {}) {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset)
  });
  return `/api/price-change-history?${params.toString()}`;
}

/**
 * Return the total number of UI pages for historical requests.
 *
 * @param {number} total Total history rows.
 * @param {number} limit Rows per page.
 * @returns {number} Page count, with one empty page when there are no rows.
 */
export function historyPageCount(total, limit = HISTORY_PAGE_SIZE) {
  return Math.max(1, Math.ceil(Number(total || 0) / Number(limit || HISTORY_PAGE_SIZE)));
}

/**
 * Return the one-based visible row range for a history page.
 *
 * @param {{ total?: number, limit?: number, offset?: number }} page Page metadata.
 * @returns {{ start: number, end: number }} Visible row range.
 */
export function historyPageRange({ total = 0, limit = HISTORY_PAGE_SIZE, offset = 0 } = {}) {
  const normalizedTotal = Number(total || 0);
  if (normalizedTotal <= 0) {
    return { start: 0, end: 0 };
  }
  const start = Number(offset || 0) + 1;
  const end = Math.min(Number(offset || 0) + Number(limit || HISTORY_PAGE_SIZE), normalizedTotal);
  return { start, end };
}

/**
 * Build the toolbar summary text for historical requests.
 *
 * @param {{ approved?: number, rejected?: number, total?: number, limit?: number, offset?: number }} summary History metadata.
 * @returns {string} User-facing toolbar summary.
 */
export function buildHistoryToolbarText(summary = {}) {
  const approved = Number(summary.approved || 0);
  const rejected = Number(summary.rejected || 0);
  const total = Number(summary.total || 0);
  const range = historyPageRange(summary);
  return `${approved} approved, ${rejected} rejected, ${total} total past requests. Showing ${range.start}-${range.end} of ${total}.`;
}
