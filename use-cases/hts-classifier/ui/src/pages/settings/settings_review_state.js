const CATALOG_WARNING_MESSAGE = "HTS code not found in managed catalog";
const SUPPORTED_METAL_FAMILIES = ["steel", "aluminum", "copper"];
const SECTION232_RESET_CONFIRMATION_TEXT = "RESET SECTION 232";
const SETTINGS_SECTION_232_REVIEW_PATH = "/settings/section-232/review";

function trimText(value) {
  return String(value ?? "").trim();
}

function toArray(value) {
  return Array.isArray(value) ? value : [];
}

function hasBatchId(batch) {
  return Boolean(trimText(batch?.batch_id));
}

function isPendingReviewDraftBatch(batch) {
  return hasBatchId(batch) && trimText(batch?.status) === "pending_review";
}

function compareDescendingTimestamps(left, right) {
  return trimText(right?.created_at).localeCompare(trimText(left?.created_at));
}

function buildRateLabel(item) {
  if (item.rule_type !== "rate_schedule") {
    return "";
  }
  if (item.effective_to) {
    return `Reduced rate until ${item.effective_to}`;
  }
  return "Reduced rate";
}

function buildEffectiveWindow(item) {
  const start = trimText(item.effective_from) || "Unknown start";
  const end = trimText(item.effective_to);
  return end ? `${start} -> ${end}` : `${start} -> Open-ended`;
}

function buildSourceSummary(sourceDocuments) {
  if (!sourceDocuments.length) {
    return "—";
  }
  if (sourceDocuments.length === 1) {
    return sourceDocuments[0];
  }
  return `${sourceDocuments[0]} +${sourceDocuments.length - 1} more`;
}

function buildMetalFamilies(metalScope) {
  const normalizedScope = trimText(metalScope).toLowerCase();
  if (!normalizedScope) {
    return [];
  }
  const families = SUPPORTED_METAL_FAMILIES.filter((family) => normalizedScope.includes(family));
  return families.length ? families : [normalizedScope];
}

function matchesMetalScopeFilter(row, metalScopeFilter) {
  const normalizedFilter = trimText(metalScopeFilter).toLowerCase();
  if (!normalizedFilter || normalizedFilter === "all") {
    return true;
  }
  return row.metalFamilies.includes(normalizedFilter) || row.metalScope.toLowerCase() === normalizedFilter;
}

export function buildReviewRowViewModel(item) {
  const sourceFilenames = toArray(item.source_filenames).map((value) => trimText(value)).filter(Boolean);
  const sourcePages = toArray(item.source_pages).filter((value) => Number.isFinite(Number(value))).map((value) => Number(value));
  const catalogMatchFound = Boolean(item.catalog_match_found);
  const description = catalogMatchFound ? trimText(item.description) : "";
  const coverageEffect = trimText(item.coverage_effect) || "include";
  const metalScope = trimText(item.metal_scope) || "—";

  return {
    candidateId: trimText(item.candidate_id),
    htsCode: trimText(item.hts_code),
    description,
    changeLabel: coverageEffect === "remove" ? "Remove" : "Include",
    changeType: coverageEffect === "remove" ? "remove" : "include",
    ruleType: trimText(item.rule_type),
    rateLabel: buildRateLabel(item),
    effectiveFrom: trimText(item.effective_from),
    effectiveTo: trimText(item.effective_to),
    effectiveWindowLabel: buildEffectiveWindow(item),
    metalScope,
    metalFamilies: buildMetalFamilies(metalScope),
    sourceFilenames,
    sourceSummary: buildSourceSummary(sourceFilenames),
    sourcePagesLabel: sourcePages.length ? sourcePages.join(", ") : "—",
    sourceExcerpt: trimText(item.source_excerpt),
    interpreterConfidenceLabel: Number.isFinite(Number(item.interpreter_confidence))
      ? `${Math.round(Number(item.interpreter_confidence) * 100)}%`
      : "—",
    reviewDecision: trimText(item.review_decision) || "pending",
    catalogWarning: catalogMatchFound ? "" : CATALOG_WARNING_MESSAGE,
    hasCatalogWarning: !catalogMatchFound,
    searchText: [
      trimText(item.hts_code),
      description,
      trimText(item.source_excerpt),
      sourceFilenames.join(" "),
      trimText(item.metal_scope),
      coverageEffect,
      trimText(item.rule_type)
    ]
      .join(" ")
      .toLowerCase()
  };
}

export function filterReviewRowViewModels(rows, filters = {}) {
  const normalizedSearch = trimText(filters.search).toLowerCase();
  const normalizedChangeType = trimText(filters.changeType).toLowerCase();
  const normalizedSourceDocument = trimText(filters.sourceDocument).toLowerCase();
  const normalizedMetalScope = trimText(filters.metalScope).toLowerCase();
  const normalizedDecision = trimText(filters.reviewDecision).toLowerCase();
  const warningsOnly = Boolean(filters.warningsOnly);
  const sortBy = trimText(filters.sortBy) || "hts_code";

  const filtered = rows.filter((row) => {
    if (normalizedSearch && !row.searchText.includes(normalizedSearch)) {
      return false;
    }
    if (normalizedChangeType && normalizedChangeType !== "all" && row.changeType !== normalizedChangeType) {
      return false;
    }
    if (
      normalizedSourceDocument &&
      normalizedSourceDocument !== "all" &&
      !row.sourceFilenames.some((sourceFilename) => sourceFilename.toLowerCase() === normalizedSourceDocument)
    ) {
      return false;
    }
    if (!matchesMetalScopeFilter(row, normalizedMetalScope)) {
      return false;
    }
    if (normalizedDecision && normalizedDecision !== "all" && row.reviewDecision !== normalizedDecision) {
      return false;
    }
    if (warningsOnly && !row.hasCatalogWarning) {
      return false;
    }
    return true;
  });

  return [...filtered].sort((left, right) => {
    if (sortBy === "change_type") {
      return left.changeLabel.localeCompare(right.changeLabel) || left.htsCode.localeCompare(right.htsCode);
    }
    if (sortBy === "effective_from") {
      return left.effectiveFrom.localeCompare(right.effectiveFrom) || left.htsCode.localeCompare(right.htsCode);
    }
    if (sortBy === "warning_state") {
      return Number(right.hasCatalogWarning) - Number(left.hasCatalogWarning) || left.htsCode.localeCompare(right.htsCode);
    }
    return left.htsCode.localeCompare(right.htsCode);
  });
}

export function summarizeDraftReviewRows(rows) {
  return rows.reduce(
    (summary, row) => {
      summary.total += 1;
      if (row.reviewDecision === "accepted") {
        summary.accepted += 1;
      } else if (row.reviewDecision === "rejected") {
        summary.rejected += 1;
      } else {
        summary.pending += 1;
      }
      if (row.hasCatalogWarning) {
        summary.warningCount += 1;
      }
      return summary;
    },
    {
      total: 0,
      accepted: 0,
      rejected: 0,
      pending: 0,
      warningCount: 0
    }
  );
}

export function buildRulesetSummaryCards(summary) {
  return [
    { label: "Active ruleset", value: trimText(summary?.active_ruleset_version) || "None" },
    { label: "Eligible code count", value: String(summary?.eligible_hts_code_count || 0) },
    { label: "Pending drafts", value: String(summary?.pending_draft_batch_count || 0) },
    { label: "Last published", value: trimText(summary?.last_published_at) || "Never" }
  ];
}

export function buildSection232ResetImpactCards({ sourceCount = 0, rulesetSummary = {} } = {}) {
  return [
    { label: "Stored source PDFs", value: String(sourceCount || 0) },
    { label: "Pending drafts", value: String(rulesetSummary?.pending_draft_batch_count || 0) },
    { label: "Active ruleset", value: trimText(rulesetSummary?.active_ruleset_version) || "None" },
    { label: "Eligible codes", value: String(rulesetSummary?.eligible_hts_code_count || 0) }
  ];
}

export function buildOpenReviewHref(batch) {
  const batchId = trimText(batch?.batch_id);
  if (!batchId) {
    return "";
  }
  return `${SETTINGS_SECTION_232_REVIEW_PATH}?batch=${encodeURIComponent(batchId)}`;
}

export function buildPublishedReviewHref(version) {
  const normalizedVersion = trimText(version);
  if (!normalizedVersion) {
    return "";
  }
  return `${SETTINGS_SECTION_232_REVIEW_PATH}?version=${encodeURIComponent(normalizedVersion)}`;
}

export function isSection232ResetConfirmationValid(value) {
  return trimText(value) === SECTION232_RESET_CONFIRMATION_TEXT;
}

export function selectLatestPendingDraftBatch(input) {
  const pendingDraftBatches = Array.isArray(input) ? input : toArray(input?.pending_draft_batches);
  const [latestBatch] = pendingDraftBatches.filter(hasBatchId).sort(compareDescendingTimestamps);
  return latestBatch || null;
}

export function buildSettingsLaunchState({ currentDraftBatch = null, rulesetSummary = {} } = {}) {
  const batch = isPendingReviewDraftBatch(currentDraftBatch) ? currentDraftBatch : selectLatestPendingDraftBatch(rulesetSummary);
  const href = buildOpenReviewHref(batch);

  return {
    batch,
    href,
    hasLaunchTarget: Boolean(href),
    pendingDraftCount: Number(rulesetSummary?.pending_draft_batch_count || 0)
  };
}

export function buildCancelDraftState({
  currentDraftBatch = null,
  rulesetSummary = {},
  busy = false
} = {}) {
  const launchState = buildSettingsLaunchState({
    currentDraftBatch,
    rulesetSummary
  });
  const batchId = trimText(launchState.batch?.batch_id);
  const hasCancelTarget = Boolean(batchId);
  const isBusy = Boolean(busy);
  return {
    batch: launchState.batch,
    batchId,
    disabled: !hasCancelTarget || isBusy,
    busy: isBusy,
    hasCancelTarget
  };
}

export function decideDraftRestoreSource({
  serverPendingDraftCount = 0,
  restoredFromServer = false,
  persistedDraftBatch = null
} = {}) {
  if (restoredFromServer) {
    return "none";
  }
  if (!isPendingReviewDraftBatch(persistedDraftBatch)) {
    return Number(serverPendingDraftCount || 0) > 0 ? "server" : "none";
  }
  return "persisted";
}

export { CATALOG_WARNING_MESSAGE, SECTION232_RESET_CONFIRMATION_TEXT, isPendingReviewDraftBatch };
