/**
 * Return a pluralized label for a numeric count.
 *
 * @param {number} count Numeric count to describe.
 * @param {string} singular Label to use when count is one.
 * @param {string} plural Label to use for every other count.
 * @returns {string} The matching singular or plural label.
 */
function pluralize(count, singular, plural = `${singular}s`) {
  return Number(count) === 1 ? singular : plural;
}

/**
 * Format a byte count for compact Settings-page file-selection copy.
 *
 * @param {number} sizeBytes File size in bytes.
 * @returns {string} Human-readable byte, KB, or MB text.
 */
function formatFileSize(sizeBytes = 0) {
  const bytes = Number(sizeBytes || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 bytes";
  }
  if (bytes < 1024) {
    return `${bytes} ${pluralize(bytes, "byte")}`;
  }
  const kilobytes = bytes / 1024;
  if (kilobytes < 1024) {
    return `${Number.isInteger(kilobytes) ? kilobytes : kilobytes.toFixed(1)} KB`;
  }
  const megabytes = kilobytes / 1024;
  return `${Number.isInteger(megabytes) ? megabytes : megabytes.toFixed(1)} MB`;
}

const GCC_TRACKER_WORKBOOK_EXTENSIONS = [".xlsb", ".xlsx"];
const GCC_TRACKER_WORKBOOK_EXTENSION_COPY = ".xlsb or .xlsx";

/**
 * Determine whether a browser File-like object is a supported GCC Tracker Excel workbook.
 *
 * @param {{ name?: string } | null | undefined} file Browser File-like object.
 * @returns {boolean} True when the filename ends with .xlsb or .xlsx.
 */
export function isGCCTrackerWorkbookFile(file) {
  const filename = String(file?.name || "").trim().toLowerCase();
  return GCC_TRACKER_WORKBOOK_EXTENSIONS.some((extension) => filename.endsWith(extension));
}

/**
 * Build Settings-page copy for the currently selected GCC Tracker file.
 *
 * @param {{ name?: string, size?: number } | null | undefined} file Browser File-like object.
 * @returns {string} Selection summary for the upload control.
 */
export function buildGCCTrackerSelectionText(file) {
  if (!file) {
    return `No GCC Tracker ${GCC_TRACKER_WORKBOOK_EXTENSION_COPY} file selected.`;
  }
  return `Selected file: ${file.name || "GCC Tracker workbook"} (${formatFileSize(file.size)}).`;
}

/**
 * Build the post-refresh success message shown on the Settings page.
 *
 * @param {object} response GCC Tracker refresh API response payload.
 * @returns {string} Human-readable refresh and cleanup result.
 */
export function buildGCCTrackerRefreshSuccessMessage(response = {}) {
  const filename = response.uploaded_filename || "GCC Tracker file";
  const rowCount = Number(response.source_row_count || 0);
  const clearedCount = Number(response.cleared_classification_count || 0);
  const cancelledCount = Number(response.cancelled_job_count || 0);
  return (
    `Uploaded ${filename} and refreshed ${rowCount} GCC ${pluralize(rowCount, "row")} in HANA. ` +
    `Cleared ${clearedCount} saved classification ${pluralize(clearedCount, "snapshot")} and ` +
    `cancelled ${cancelledCount} active classification ${pluralize(cancelledCount, "job")}.`
  );
}
