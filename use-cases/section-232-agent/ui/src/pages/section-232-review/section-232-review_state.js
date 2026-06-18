function trimText(value) {
  return String(value ?? "").trim();
}

const UNSPECIFIED_SCOPE_KEYS = new Set([
  "",
  "all",
  "mixed",
  "mixed/unspecified",
  "multi",
  "multiple",
  "unknown",
  "unspecified"
]);
const METAL_SCOPE_TOKEN_RE = /\b(?:steel|aluminum|aluminium|copper)\b/gi;
const CANONICAL_METAL_SCOPE_ORDER = {
  aluminum: 0,
  copper: 1,
  steel: 2
};
const OPEN_ENDED_EFFECTIVE_TO = "9999-12-31";
export const SECTION_232_REVIEW_PAGE_SIZE = 100;

function toSearchParams(input) {
  if (input instanceof URLSearchParams) {
    return input;
  }
  if (typeof input === "string") {
    const normalized = input.startsWith("?") ? input.slice(1) : input;
    return new URLSearchParams(normalized);
  }
  if (input && typeof input === "object") {
    if (typeof input.search === "string") {
      return toSearchParams(input.search);
    }
    if (typeof input.querystring === "string") {
      return toSearchParams(input.querystring);
    }
  }
  return new URLSearchParams();
}

export function parseSection232ReviewRoute(input) {
  const params = toSearchParams(input);
  const batchId = trimText(params.get("batch"));
  const version = trimText(params.get("version"));
  const hasBatch = Boolean(batchId);
  const hasVersion = Boolean(version);

  if (hasBatch === hasVersion) {
    return {
      batchId,
      version,
      mode: "unknown",
      requestQuery: {},
      isValid: false,
      error: "Choose exactly one review target: draft batch or published version."
    };
  }

  if (hasBatch) {
    return {
      batchId,
      version: "",
      mode: "draft",
      requestQuery: { batch_id: batchId },
      isValid: true,
      error: ""
    };
  }

  return {
    batchId: "",
    version,
    mode: "published",
    requestQuery: { version },
    isValid: true,
    error: ""
  };
}

export function resolveSection232ReviewSelection(rows = [], selectedIds = [], { mode = "draft" } = {}) {
  const knownIds = new Set(rows.map((row) => trimText(row?.candidateId)));
  const normalizedIds = Array.from(new Set(selectedIds.map((value) => trimText(value)).filter((value) => knownIds.has(value))));
  const selectedRow = normalizedIds.length === 1 ? rows.find((row) => row.candidateId === normalizedIds[0]) || null : null;
  const isDraft = trimText(mode) === "draft";

  if (!normalizedIds.length) {
    return {
      selectedIds: [],
      selectedRow: null,
      canEditDetail: false,
      selectionLabel: isDraft ? "Select one row to review details." : "Select one row to inspect details."
    };
  }

  if (selectedRow) {
    return {
      selectedIds: normalizedIds,
      selectedRow,
      canEditDetail: isDraft,
      selectionLabel: isDraft ? "Editing row details." : "Inspecting row details."
    };
  }

  return {
    selectedIds: normalizedIds,
    selectedRow: null,
    canEditDetail: false,
    selectionLabel: isDraft ? "Select exactly one row to edit details." : "Select exactly one row to inspect details."
  };
}

export function buildSection232CatalogStatus(item = {}) {
  const matchType = trimText(item.catalog_match_type).toLowerCase();
  const representativeCode = trimText(item.catalog_representative_code);
  const familyCount = Number(item.catalog_family_match_count || 0);
  const warning = trimText(item.catalog_warning);

  if (matchType === "exact") {
    return {
      state: "Positive",
      label: "Exact match",
      detail: representativeCode ? `Matched managed catalog entry ${representativeCode}.` : "Matched managed catalog entry."
    };
  }

  if (matchType === "family") {
    return {
      state: "Information",
      label: "Family match",
      detail: representativeCode
        ? `Using ${representativeCode} as the family representative across ${familyCount} catalog entries.`
        : `Matched ${familyCount} related catalog entries in the same family.`
    };
  }

  return {
    state: "Negative",
    label: "Catalog missing",
    detail: warning || "HTS code not found in managed catalog."
  };
}

export function buildSection232ReviewInteractivity({
  mode = "",
  loading = false,
  mutationInFlight = false,
  publishedBy = "",
  rowCount = 0,
  selection = {},
  selectedCount = null
} = {}) {
  const isDraft = trimText(mode) === "draft";
  const canMutate = isDraft && !loading && !mutationInFlight;
  const hasPublishedBy = Boolean(trimText(publishedBy));
  const selectedIds = Array.isArray(selection.selectedIds) ? selection.selectedIds : [];
  const normalizedSelectedCount = selectedCount != null && Number.isFinite(Number(selectedCount))
    ? Math.max(0, Number(selectedCount))
    : selectedIds.length;
  const normalizedRowCount = Number.isFinite(Number(rowCount)) ? Math.max(0, Number(rowCount)) : 0;
  const hasAnyRows = normalizedRowCount > 0;
  const hasSelection = normalizedSelectedCount > 0;
  const hasUnselectedRows = normalizedSelectedCount < normalizedRowCount;
  const hasDetailSelection = Boolean(selection.selectedRow);

  return {
    isReadOnly: !isDraft,
    showBulkActions: isDraft,
    showDecisionEditor: isDraft,
    showDeleteAction: hasDetailSelection,
    enableRowCheckboxes: canMutate,
    canBulkReview: canMutate && hasSelection && hasPublishedBy,
    canSelectAllRows: canMutate && hasAnyRows && hasUnselectedRows,
    canClearSelection: canMutate && hasSelection,
    canEditDetailDecision: canMutate && hasPublishedBy && Boolean(selection.canEditDetail),
    canDeleteDetailCode: !loading && !mutationInFlight && hasDetailSelection && (isDraft || hasPublishedBy)
  };
}

export function createSection232DraftSelectionState() {
  return {
    mode: "explicit",
    candidateIds: [],
    excludedCandidateIds: []
  };
}

export function isSection232DraftRowSelected(selection = {}, candidateId = "") {
  const normalizedCandidateId = trimText(candidateId);
  if (!normalizedCandidateId) return false;
  if (trimText(selection.mode) === "all") {
    const excludedIds = new Set((selection.excludedCandidateIds || []).map((value) => trimText(value)).filter(Boolean));
    return !excludedIds.has(normalizedCandidateId);
  }
  const candidateIds = new Set((selection.candidateIds || []).map((value) => trimText(value)).filter(Boolean));
  return candidateIds.has(normalizedCandidateId);
}

export function getSection232DraftSelectionCount(selection = {}, totalRowCount = 0) {
  const normalizedTotal = Number.isFinite(Number(totalRowCount)) ? Math.max(0, Number(totalRowCount)) : 0;
  if (trimText(selection.mode) === "all") {
    const excludedCount = new Set(
      (selection.excludedCandidateIds || []).map((value) => trimText(value)).filter(Boolean)
    ).size;
    return Math.max(0, normalizedTotal - excludedCount);
  }
  return new Set((selection.candidateIds || []).map((value) => trimText(value)).filter(Boolean)).size;
}

export function toggleSection232DraftSelection(selection = {}, candidateId = "", isSelected = false) {
  const normalizedCandidateId = trimText(candidateId);
  if (!normalizedCandidateId) {
    return {
      mode: trimText(selection.mode) === "all" ? "all" : "explicit",
      candidateIds: Array.isArray(selection.candidateIds) ? [...selection.candidateIds] : [],
      excludedCandidateIds: Array.isArray(selection.excludedCandidateIds) ? [...selection.excludedCandidateIds] : []
    };
  }

  if (trimText(selection.mode) === "all") {
    const excludedIds = new Set((selection.excludedCandidateIds || []).map((value) => trimText(value)).filter(Boolean));
    if (isSelected) {
      excludedIds.delete(normalizedCandidateId);
    } else {
      excludedIds.add(normalizedCandidateId);
    }
    return {
      mode: "all",
      candidateIds: [],
      excludedCandidateIds: [...excludedIds]
    };
  }

  const candidateIds = new Set((selection.candidateIds || []).map((value) => trimText(value)).filter(Boolean));
  if (isSelected) {
    candidateIds.add(normalizedCandidateId);
  } else {
    candidateIds.delete(normalizedCandidateId);
  }
  return {
    mode: "explicit",
    candidateIds: [...candidateIds],
    excludedCandidateIds: []
  };
}

export function selectAllSection232DraftRows() {
  return {
    mode: "all",
    candidateIds: [],
    excludedCandidateIds: []
  };
}

export function clearSection232DraftSelection() {
  return createSection232DraftSelectionState();
}

export function buildSection232DraftBulkReviewPayload(selection = {}, decision = "accepted") {
  const normalizedDecision = trimText(decision) || "accepted";
  if (trimText(selection.mode) === "all") {
    return {
      selection_mode: "all",
      excluded_candidate_ids: Array.isArray(selection.excludedCandidateIds) ? [...selection.excludedCandidateIds] : [],
      decision: normalizedDecision
    };
  }
  return {
    selection_mode: "explicit",
    candidate_ids: Array.isArray(selection.candidateIds) ? [...selection.candidateIds] : [],
    decision: normalizedDecision
  };
}

export function buildSection232ReviewPublishState(rows = [], publishedBy = "") {
  const rowList = Array.isArray(rows) ? rows : [];
  const trimmedPublishedBy = trimText(publishedBy);
  let acceptedCount = 0;
  let pendingCount = 0;

  for (const row of rowList) {
    const decision = trimText(row?.reviewDecision).toLowerCase();
    if (decision === "accepted") {
      acceptedCount += 1;
    } else if (decision !== "rejected") {
      pendingCount += 1;
    }
  }

  return {
    acceptedCount,
    pendingCount,
    publishedBy: trimmedPublishedBy,
    canPublish: acceptedCount > 0 && Boolean(trimmedPublishedBy)
  };
}

export function buildSection232ReviewMutationRecoveryState(message, design = "Negative") {
  return {
    mutationInFlight: false,
    message: trimText(message),
    messageDesign: design
  };
}

function buildSourceSummary(sourceFilenames = []) {
  const values = Array.isArray(sourceFilenames) ? sourceFilenames.map((value) => trimText(value)).filter(Boolean) : [];
  if (!values.length) return "—";
  if (values.length === 1) return values[0];
  return `${values[0]} +${values.length - 1} more`;
}

export function buildSection232ReviewRequestQuery(route = {}, htsSearchQuery = "") {
  const requestQuery = {
    ...(route?.requestQuery || {})
  };
  const normalizedQuery = trimText(htsSearchQuery);
  if (normalizedQuery) {
    requestQuery.hts_query = normalizedQuery;
  }
  return requestQuery;
}

export function buildSection232HtsSearchControlState({
  inputQuery = "",
  appliedQuery = "",
  routeValid = true,
  loading = false,
  mutationInFlight = false
} = {}) {
  const normalizedInputQuery = trimText(inputQuery);
  const normalizedAppliedQuery = trimText(appliedQuery);
  const disabled = Boolean(loading || mutationInFlight || !routeValid);
  const hasPendingQuery = normalizedInputQuery !== normalizedAppliedQuery;
  return {
    inputQuery: normalizedInputQuery,
    appliedQuery: normalizedAppliedQuery,
    disabled,
    hasInput: Boolean(normalizedInputQuery),
    hasAppliedQuery: Boolean(normalizedAppliedQuery),
    hasPendingQuery,
    canSubmit: !disabled && hasPendingQuery,
    canClear: !disabled && Boolean(normalizedInputQuery || normalizedAppliedQuery)
  };
}

export function buildSection232HistoryItemViewModel(item = {}) {
  const coverageEffect = trimText(item.coverage_effect || item.coverageEffect);
  const sourceFilenames = Array.isArray(item.source_filenames || item.sourceFilenames)
    ? (item.source_filenames || item.sourceFilenames).map((value) => trimText(value)).filter(Boolean)
    : [];
  return {
    version: trimText(item.version),
    publishedAt: trimText(item.published_at || item.publishedAt),
    publishedBy: trimText(item.published_by || item.publishedBy),
    candidateId: trimText(item.candidate_id || item.candidateId),
    legalHtsCode: trimText(item.legal_hts_code || item.hts_code || item.legalHtsCode || item.htsCode),
    ruleType: trimText(item.rule_type || item.ruleType),
    coverageEffect,
    effectiveFrom: trimText(item.effective_from || item.effectiveFrom),
    effectiveTo: trimText(item.effective_to || item.effectiveTo),
    sourceFilenames,
    sourceSummary: buildSourceSummary(sourceFilenames),
    processedAt: trimText(item.processed_at || item.processedAt),
    actionLabel: coverageEffect.toLowerCase() === "remove" ? "Removed" : "Included"
  };
}

function normalizeTextSources(values) {
  return Array.isArray(values)
    ? values.map((value) => trimText(value).toLowerCase()).filter((value) => value === "plain" || value === "layout")
    : [];
}

function buildTextSourceLabel(values) {
  const textSources = normalizeTextSources(values);
  if (textSources.includes("plain") && textSources.includes("layout")) {
    return "Plain text + layout text";
  }
  if (textSources.includes("layout")) {
    return "Layout text";
  }
  return "Plain text";
}

function buildEffectiveWindow(item = {}) {
  const start = trimText(item.effective_from);
  const end = trimText(item.effective_to);
  if (start && end) {
    return `From ${start} to ${end}`;
  }
  if (start) {
    return "Starts " + start + ", open-ended";
  }
  if (end) {
    return `Until ${end}`;
  }
  return "No explicit effective window";
}

function buildSourceDocuments(item = {}) {
  const documents = Array.isArray(item.source_documents)
    ? item.source_documents.map((document) => ({
        sourceId: trimText(document?.source_id),
        filename: trimText(document?.filename) || "Unknown source",
        uploadedAt: trimText(document?.uploaded_at)
      }))
    : [];
  if (documents.length) {
    return documents;
  }
  const filenames = Array.isArray(item.source_filenames)
    ? item.source_filenames.map((value) => trimText(value)).filter(Boolean)
    : [];
  return filenames.map((filename) => ({
    sourceId: "",
    filename,
    uploadedAt: ""
  }));
}

function normalizeEffectiveDate(value) {
  const normalized = trimText(value);
  return normalized === "—" ? "" : normalized;
}

function normalizeScopeKey(value) {
  const normalized = trimText(value).toLowerCase();
  if (UNSPECIFIED_SCOPE_KEYS.has(normalized)) {
    return "unspecified";
  }
  const tokens = [];
  const seen = new Set();
  for (const match of normalized.matchAll(METAL_SCOPE_TOKEN_RE)) {
    let token = match[0].toLowerCase();
    if (token === "aluminium") {
      token = "aluminum";
    }
    if (seen.has(token)) continue;
    seen.add(token);
    tokens.push(token);
  }
  if (!tokens.length) {
    return normalized || "unspecified";
  }
  tokens.sort((left, right) => {
    const leftRank = CANONICAL_METAL_SCOPE_ORDER[left] ?? 99;
    const rightRank = CANONICAL_METAL_SCOPE_ORDER[right] ?? 99;
    return leftRank - rightRank || left.localeCompare(right);
  });
  return tokens.join("+");
}

function toPublicationWindow(row = {}) {
  return {
    effectiveFrom: normalizeEffectiveDate(row.effectiveFrom),
    effectiveTo: normalizeEffectiveDate(row.effectiveTo)
  };
}

function publicationWindowSortKey(window) {
  return `${window.effectiveFrom}|${window.effectiveTo || OPEN_ENDED_EFFECTIVE_TO}`;
}

function comparePublicationWindows(left, right) {
  return publicationWindowSortKey(left).localeCompare(publicationWindowSortKey(right));
}

function publicationWindowOverlaps(left, right) {
  const leftEnd = left.effectiveTo || OPEN_ENDED_EFFECTIVE_TO;
  const rightEnd = right.effectiveTo || OPEN_ENDED_EFFECTIVE_TO;
  return left.effectiveFrom <= rightEnd && right.effectiveFrom <= leftEnd;
}

function pickEarlierDate(left, right) {
  return left.localeCompare(right) <= 0 ? left : right;
}

function pickLaterEnd(left, right) {
  return (left || OPEN_ENDED_EFFECTIVE_TO).localeCompare(right || OPEN_ENDED_EFFECTIVE_TO) >= 0 ? left : right;
}

function buildPublicationWindowLabel(window = {}) {
  return buildEffectiveWindow({
    effective_from: window.effectiveFrom,
    effective_to: window.effectiveTo
  });
}

function buildMaterialRuleLabel(row = {}) {
  const code = trimText(row.legalHtsCode || row.htsCode);
  const ruleType = trimText(row.ruleType || row.coverageEffect || "include").toLowerCase() || "include";
  return `${code} · ${ruleType} · ${normalizeScopeKey(row.metalScope)}`;
}

export function buildSection232ReviewDiagnostics(rows = []) {
  const reviewableRows = (Array.isArray(rows) ? rows : []).filter(
    (row) => trimText(row?.reviewDecision).toLowerCase() !== "rejected"
  );
  if (!reviewableRows.length) {
    return {
      summary: "",
      items: []
    };
  }

  const duplicateGroups = new Map();
  const materialGroups = new Map();

  for (const row of reviewableRows) {
    const materialLabel = buildMaterialRuleLabel(row);
    const window = toPublicationWindow(row);
    const slotKey = `${materialLabel}|${publicationWindowSortKey(window)}`;
    const duplicateGroup = duplicateGroups.get(slotKey) || {
      label: materialLabel,
      window,
      count: 0
    };
    duplicateGroup.count += 1;
    duplicateGroups.set(slotKey, duplicateGroup);

    const materialWindows = materialGroups.get(materialLabel) || [];
    materialWindows.push(window);
    materialGroups.set(materialLabel, materialWindows);
  }

  const duplicateItems = Array.from(duplicateGroups.values())
    .filter((group) => group.count > 1)
    .map(
      (group) => `Duplicate rows share ${group.label} · ${buildPublicationWindowLabel(group.window)}.`
    );

  const overlapItems = [];
  for (const [materialLabel, windows] of materialGroups.entries()) {
    const uniqueWindows = Array.from(
      new Map(windows.map((window) => [publicationWindowSortKey(window), window])).values()
    ).sort(comparePublicationWindows);
    if (uniqueWindows.length < 2) continue;

    let clusterStart = uniqueWindows[0].effectiveFrom;
    let clusterEnd = uniqueWindows[0].effectiveTo;
    let clusterHasOverlap = false;

    for (const window of uniqueWindows.slice(1)) {
      const clusterWindow = {
        effectiveFrom: clusterStart,
        effectiveTo: clusterEnd
      };
      if (publicationWindowOverlaps(clusterWindow, window)) {
        clusterHasOverlap = true;
        clusterStart = pickEarlierDate(clusterStart, window.effectiveFrom);
        clusterEnd = pickLaterEnd(clusterEnd, window.effectiveTo);
        continue;
      }
      if (clusterHasOverlap) {
        overlapItems.push(
          `Overlapping rows share ${materialLabel}. Keep the widest window: ${buildPublicationWindowLabel({
            effectiveFrom: clusterStart,
            effectiveTo: clusterEnd
          })}.`
        );
      }
      clusterStart = window.effectiveFrom;
      clusterEnd = window.effectiveTo;
      clusterHasOverlap = false;
    }

    if (clusterHasOverlap) {
      overlapItems.push(
        `Overlapping rows share ${materialLabel}. Keep the widest window: ${buildPublicationWindowLabel({
          effectiveFrom: clusterStart,
          effectiveTo: clusterEnd
        })}.`
      );
    }
  }

  const items = [...duplicateItems, ...overlapItems];
  return {
    summary: items.length ? `${items.length} Section 232 review issues need attention before publish.` : "",
    items
  };
}

export function buildSection232ReviewRowViewModel(item = {}) {
  const catalogStatus = buildSection232CatalogStatus(item);
  const effectiveFrom = trimText(item.effective_from);
  const effectiveTo = trimText(item.effective_to);
  const rateText = trimText(item.rate_text);
  const candidateQuality = trimText(item.candidate_quality).toLowerCase() === "suspect" ? "suspect" : "normal";
  const candidateFlags = Array.isArray(item.candidate_flags)
    ? item.candidate_flags.map((value) => trimText(value)).filter(Boolean)
    : [];
  const effectiveWindowLabel = buildEffectiveWindow(item);
  const sourceDocuments = buildSourceDocuments(item);
  const sourceUploadedAt = trimText(item.source_uploaded_at);
  const processedAt = trimText(item.processed_at);
  const matchEvidence = Array.isArray(item.match_evidence)
    ? item.match_evidence.map((evidence) => {
        const pageNumber = Number(evidence?.page_number);
        return {
          sourceId: trimText(evidence?.source_id),
          sourceFilename: trimText(evidence?.source_filename),
          pageNumber: Number.isFinite(pageNumber) ? pageNumber : 0,
          pageLabel: Number.isFinite(pageNumber) && pageNumber > 0 ? `P${pageNumber}` : "Page —",
          matchedText: trimText(evidence?.matched_text) || "—",
          normalizedHtsCode: trimText(evidence?.normalized_hts_code),
          contextText: trimText(evidence?.context_text) || "No context available.",
          textSources: normalizeTextSources(evidence?.text_sources),
          textSourceLabel: buildTextSourceLabel(evidence?.text_sources)
        };
      })
    : [];
  const history = Array.isArray(item.history)
    ? item.history.map(buildSection232HistoryItemViewModel)
    : [];
  return {
    candidateId: trimText(item.candidate_id),
    legalHtsCode: trimText(item.legal_hts_code || item.hts_code),
    description: trimText(item.description) || "No catalog description available.",
    ruleType: trimText(item.rule_type) || "include",
    coverageEffect: trimText(item.coverage_effect) || "include",
    effectiveFrom: effectiveFrom || "—",
    effectiveTo: effectiveTo || "—",
    effectiveWindowLabel,
    effectiveWindowSummary: effectiveWindowLabel,
    metalScope: trimText(item.metal_scope) || "—",
    reviewDecision: trimText(item.review_decision) || "pending",
    sourceExcerpt: trimText(item.source_excerpt) || "No source excerpt available.",
    sourcePages: Array.isArray(item.source_pages) ? item.source_pages.map((value) => Number(value)).filter(Number.isFinite) : [],
    sourceFilenames: Array.isArray(item.source_filenames) ? item.source_filenames.map((value) => trimText(value)).filter(Boolean) : [],
    sourceDocuments,
    sourceUploadedAt: sourceUploadedAt || "—",
    processedAt: processedAt || "—",
    sourceSummary: buildSourceSummary(item.source_filenames),
    matchEvidence,
    matchEvidenceCount: matchEvidence.length,
    primaryMatchEvidence: matchEvidence[0] || null,
    history,
    historyCount: history.length,
    catalogStatus,
    rateText: rateText || "—",
    candidateQuality,
    candidateFlags,
    isSuspect: candidateQuality === "suspect"
  };
}

export function buildSection232WorkspaceSummary(workspace = {}, route = {}) {
  const rowCount = Number.isFinite(Number(workspace.total))
    ? Math.max(0, Number(workspace.total))
    : Array.isArray(workspace.rows)
      ? workspace.rows.length
      : 0;
  const mode = trimText(workspace.mode || route.mode || "unknown");
  if (mode !== "draft" && mode !== "published") {
    return {
      modeLabel: "Review workspace",
      title: "Review workspace",
      subtitle: "Choose a draft batch or published version in the route query string."
    };
  }

  if (mode === "draft") {
    return {
      modeLabel: "Draft batch",
      title: workspace.batch?.batch_id || route.batchId || "Draft review",
      subtitle: `${rowCount} row${rowCount === 1 ? "" : "s"} from ${workspace.source_filenames?.length || 0} source file${workspace.source_filenames?.length === 1 ? "" : "s"}`
    };
  }

  return {
    modeLabel: "Published ruleset",
    title: workspace.version || route.version || "Published review",
    subtitle: `${rowCount} published row${rowCount === 1 ? "" : "s"} in review workspace`
  };
}
