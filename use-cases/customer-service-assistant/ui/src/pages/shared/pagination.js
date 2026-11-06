export const PAGE_SIZE = 50;

/**
 * Build a query string for fixed-size review list pagination.
 *
 * @param {string} endpoint - API endpoint path without pagination parameters.
 * @param {number} page - One-based page number to request.
 * @returns {string} Endpoint path with page and fixed page_size query parameters.
 */
export function paginatedEndpoint(endpoint, page) {
  const pageNumber = Math.max(1, Number(page) || 1);
  return `${endpoint}?page=${pageNumber}&page_size=${PAGE_SIZE}`;
}

/**
 * Normalize API pagination metadata into frontend-friendly names.
 *
 * @param {object} payload - API response containing pagination metadata.
 * @returns {{page: number, pageSize: number, total: number, totalPages: number, summary: object}} Normalized metadata.
 */
export function normalizePagination(payload) {
  const total = Math.max(0, Number(payload?.total) || 0);
  return {
    page: Math.max(1, Number(payload?.page) || 1),
    pageSize: Math.max(1, Number(payload?.page_size) || PAGE_SIZE),
    total,
    totalPages: Math.max(1, Number(payload?.total_pages) || 1),
    summary: payload?.summary || {}
  };
}

/**
 * Format the visible item range for a paginated table.
 *
 * @param {{page: number, pageSize: number, total: number}} meta - Pagination metadata.
 * @returns {string} Human-readable range label.
 */
export function paginationRangeLabel(meta) {
  if (!meta.total) {
    return "0 of 0";
  }
  const start = (meta.page - 1) * meta.pageSize + 1;
  const end = Math.min(meta.total, meta.page * meta.pageSize);
  return `${start}-${end} of ${meta.total}`;
}

/**
 * Render previous/next pagination controls.
 *
 * @param {HTMLElement} container - Target element for the pagination controls.
 * @param {{page: number, totalPages: number, pageSize: number, total: number}} meta - Pagination metadata.
 * @returns {void}
 */
export function renderPaginationControls(container, meta) {
  if (!container) return;
  container.innerHTML = `
    <div class="pagination-controls">
      <ui5-button data-page-direction="previous" design="Transparent" ${meta.page <= 1 ? "disabled" : ""}>Previous</ui5-button>
      <span class="pagination-status">Page ${meta.page} of ${meta.totalPages}</span>
      <span class="pagination-range">${paginationRangeLabel(meta)}</span>
      <ui5-button data-page-direction="next" design="Transparent" ${meta.page >= meta.totalPages ? "disabled" : ""}>Next</ui5-button>
    </div>
  `;
}

/**
 * Attach click handlers to already-rendered pagination controls.
 *
 * @param {HTMLElement} container - Pagination controls container.
 * @param {{page: number, totalPages: number}} meta - Current pagination metadata.
 * @param {(page: number) => void} onPageChange - Callback invoked with the requested page.
 * @returns {void}
 */
export function attachPaginationHandlers(container, meta, onPageChange) {
  if (!container) return;
  container.querySelectorAll("[data-page-direction]").forEach((button) => {
    button.addEventListener("click", () => {
      const direction = button.dataset.pageDirection;
      const nextPage = direction === "next" ? meta.page + 1 : meta.page - 1;
      if (nextPage < 1 || nextPage > meta.totalPages) return;
      onPageChange(nextPage);
    });
  });
}
