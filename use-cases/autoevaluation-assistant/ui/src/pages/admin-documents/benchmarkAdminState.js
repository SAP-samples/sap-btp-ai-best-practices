let benchmarkSelectionSequence = 0;
let benchmarkHistorySequence = 0;

/**
 * Format an active-import timestamp in the UI's selected language.
 *
 * @param {string | null | undefined} value - ISO timestamp returned by the API.
 * @param {"en" | "it" | string} language - Active UI language/locale.
 * @returns {string} Localized date/time or a missing-value dash.
 */
export function formatBenchmarkTimestamp(value, language) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString(language);
}

/**
 * Create the initial immutable UI state for benchmark workbook administration.
 *
 * @returns {object} Empty selection, validation, history, and request state.
 */
export function createBenchmarkAdminState() {
  return {
    selectedFile: null,
    selectionRevision: null,
    validation: null,
    validatedRevision: null,
    history: null,
    mutationLoading: null,
    historyLoading: false,
    historyRequestRevision: null,
    historyError: null,
    error: null
  };
}

/**
 * Replace the selected workbook and invalidate every result from an older file.
 *
 * @param {object} state - Current benchmark panel state.
 * @param {File | object | null} file - Newly selected browser file.
 * @returns {object} Fresh state whose activation permission has been cleared.
 */
export function selectBenchmarkWorkbook(state, file) {
  return {
    ...state,
    selectedFile: file,
    // Module lifetime spans route remounts, so this token is never reused when
    // navigation recreates otherwise-empty panel state.
    selectionRevision: ++benchmarkSelectionSequence,
    validation: null,
    validatedRevision: null,
    mutationLoading: null,
    error: null
  };
}

/**
 * Mark one workbook mutation as active without changing history ownership.
 *
 * @param {object} state - Current benchmark panel state.
 * @param {"validation" | "activation"} mutation - Mutation being performed.
 * @returns {object} State with destructive controls locked for the mutation.
 */
export function beginBenchmarkMutation(state, mutation) {
  return { ...state, mutationLoading: mutation, error: null };
}

/**
 * Begin an independently owned import-history refresh.
 *
 * @param {object} state - Current benchmark panel state.
 * @returns {{state: object, requestRevision: number}} Updated state and owner token.
 */
export function beginBenchmarkHistoryLoad(state) {
  const requestRevision = ++benchmarkHistorySequence;
  return {
    requestRevision,
    state: {
      ...state,
      historyLoading: true,
      historyRequestRevision: requestRevision,
      historyError: null
    }
  };
}

/**
 * Return whether a history response still owns the active refresh request.
 *
 * @param {object} state - Current benchmark panel state.
 * @param {number} requestRevision - History token captured before the request.
 * @returns {boolean} True only for the newest history request.
 */
export function isCurrentBenchmarkHistoryRequest(state, requestRevision) {
  return state.historyRequestRevision === requestRevision;
}

/**
 * Commit history only for its owning request while preserving mutation state.
 *
 * @param {object} state - Current benchmark panel state.
 * @param {object} history - Safe active/recent import metadata.
 * @param {number} requestRevision - History token captured before the request.
 * @returns {object} Updated state, or unchanged state for a stale response.
 */
export function completeBenchmarkHistoryLoad(state, history, requestRevision) {
  if (!isCurrentBenchmarkHistoryRequest(state, requestRevision)) {
    return state;
  }
  return {
    ...state,
    history,
    historyLoading: false,
    historyRequestRevision: null,
    historyError: null
  };
}

/**
 * Finish the owning failed history request without unlocking a mutation.
 *
 * @param {object} state - Current benchmark panel state.
 * @param {string} error - Safe request error retained for diagnostics.
 * @param {number} requestRevision - History token captured before the request.
 * @returns {object} Updated state, or unchanged state for a stale response.
 */
export function failBenchmarkHistoryLoad(state, error, requestRevision) {
  if (!isCurrentBenchmarkHistoryRequest(state, requestRevision)) {
    return state;
  }
  return {
    ...state,
    historyLoading: false,
    historyRequestRevision: null,
    historyError: error
  };
}

/**
 * Store a completed validation only when it belongs to the current selection.
 *
 * @param {object} state - Current benchmark panel state.
 * @param {object} validation - Public validation response returned by the API.
 * @param {number} [selectionRevision] - Revision captured before the request.
 * @returns {object} Updated state, or the unchanged state for a stale response.
 */
export function applyBenchmarkValidation(
  state,
  validation,
  selectionRevision = state.selectionRevision
) {
  if (selectionRevision !== state.selectionRevision) {
    return state;
  }
  return {
    ...state,
    validation,
    validatedRevision: selectionRevision,
    mutationLoading: null,
    error: null
  };
}

/**
 * Return whether an asynchronous result still owns the selected workbook.
 *
 * @param {object} state - Current benchmark panel state.
 * @param {number | null} selectionRevision - Token captured before the request.
 * @returns {boolean} True only while the same unique selection is current.
 */
export function isCurrentBenchmarkSelection(state, selectionRevision) {
  return Boolean(
    selectionRevision !== null &&
      state.selectedFile &&
      state.selectionRevision === selectionRevision
  );
}

/**
 * Return whether the destructive activation action is safe to expose.
 *
 * @param {object} state - Current benchmark panel state.
 * @returns {boolean} True only for a successful validation of the current file.
 */
export function canActivateBenchmark(state) {
  return Boolean(
    state.selectedFile &&
      state.validation?.success === true &&
      state.validatedRevision === state.selectionRevision &&
      !state.mutationLoading
  );
}

/**
 * Build the exact multipart contract shared by dry-run and write imports.
 *
 * @param {File | object} file - Selected XLSX workbook.
 * @param {boolean} write - Whether the backend may persist and activate it.
 * @param {typeof FormData} [FormDataType] - Browser FormData constructor.
 * @returns {FormData | object} Multipart body with workbook and explicit write.
 */
export function buildBenchmarkImportForm(
  file,
  write,
  FormDataType = FormData
) {
  const form = new FormDataType();
  form.append("workbook", file, file.name);
  form.append("write", write ? "true" : "false");
  return form;
}

/**
 * Normalize benchmark history into an explicit render state.
 *
 * @param {object | null | undefined} history - Import history API response.
 * @returns {{kind: string, reason: string | null, active: object | null}} Render state.
 */
export function benchmarkHistoryView(history) {
  if (!history?.available || !history.active) {
    return {
      kind: "unavailable",
      reason: history?.reason || "no_active_dataset",
      active: null
    };
  }
  return { kind: "active", reason: null, active: history.active };
}

/**
 * Validate the browser-side file contract before any upload is attempted.
 *
 * @param {File | object | null} file - Candidate browser file.
 * @returns {boolean} True only for one filename ending in `.xlsx`.
 */
export function isBenchmarkWorkbook(file) {
  return Boolean(file?.name && String(file.name).toLowerCase().endsWith(".xlsx"));
}
