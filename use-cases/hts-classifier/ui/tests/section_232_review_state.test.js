import test from "node:test";
import assert from "node:assert/strict";

import {
  buildSection232ReviewDiagnostics,
  buildSection232DraftBulkReviewPayload,
  buildSection232ReviewInteractivity,
  buildSection232ReviewPublishState,
  buildSection232CatalogStatus,
  buildSection232HistoryItemViewModel,
  buildSection232HtsSearchControlState,
  buildSection232ReviewMutationRecoveryState,
  buildSection232ReviewRequestQuery,
  buildSection232ReviewRowViewModel,
  clearSection232DraftSelection,
  createSection232DraftSelectionState,
  getSection232DraftSelectionCount,
  isSection232DraftRowSelected,
  parseSection232ReviewRoute,
  resolveSection232ReviewSelection,
  selectAllSection232DraftRows,
  toggleSection232DraftSelection
} from "../src/pages/section-232-review/section-232-review_state.js";

test("parseSection232ReviewRoute reads batch mode from the query string", () => {
  const route = parseSection232ReviewRoute("?batch=batch-2026-04-19");

  assert.deepEqual(route, {
    batchId: "batch-2026-04-19",
    version: "",
    mode: "draft",
    requestQuery: {
      batch_id: "batch-2026-04-19"
    },
    isValid: true,
    error: ""
  });
});

test("parseSection232ReviewRoute reads published mode from the query string", () => {
  const route = parseSection232ReviewRoute("?version=section232-v0007");

  assert.deepEqual(route, {
    batchId: "",
    version: "section232-v0007",
    mode: "published",
    requestQuery: {
      version: "section232-v0007"
    },
    isValid: true,
    error: ""
  });
});

test("parseSection232ReviewRoute rejects missing or conflicting query params", () => {
  assert.deepEqual(parseSection232ReviewRoute(""), {
    batchId: "",
    version: "",
    mode: "unknown",
    requestQuery: {},
    isValid: false,
    error: "Choose exactly one review target: draft batch or published version."
  });

  assert.deepEqual(parseSection232ReviewRoute("?batch=batch-1&version=section232-v0001"), {
    batchId: "batch-1",
    version: "section232-v0001",
    mode: "unknown",
    requestQuery: {},
    isValid: false,
    error: "Choose exactly one review target: draft batch or published version."
  });
});

test("resolveSection232ReviewSelection enables detail editing only for a single row", () => {
  assert.deepEqual(resolveSection232ReviewSelection([], [], { mode: "draft" }), {
    selectedIds: [],
    selectedRow: null,
    canEditDetail: false,
    selectionLabel: "Select one row to review details."
  });

  assert.deepEqual(
    resolveSection232ReviewSelection(
      [
        { candidateId: "candidate-1", legalHtsCode: "7308.90.95" },
        { candidateId: "candidate-2", legalHtsCode: "7604.29.10" }
      ],
      ["candidate-2"],
      { mode: "draft" }
    ),
    {
      selectedIds: ["candidate-2"],
      selectedRow: { candidateId: "candidate-2", legalHtsCode: "7604.29.10" },
      canEditDetail: true,
      selectionLabel: "Editing row details."
    }
  );

  assert.deepEqual(
    resolveSection232ReviewSelection(
      [
        { candidateId: "candidate-1", legalHtsCode: "7308.90.95" },
        { candidateId: "candidate-2", legalHtsCode: "7604.29.10" }
      ],
      ["candidate-1", "candidate-2"],
      { mode: "draft" }
    ),
    {
      selectedIds: ["candidate-1", "candidate-2"],
      selectedRow: null,
      canEditDetail: false,
      selectionLabel: "Select exactly one row to edit details."
    }
  );
});

test("resolveSection232ReviewSelection uses browse-only copy in published mode", () => {
  assert.deepEqual(
    resolveSection232ReviewSelection(
      [
        { candidateId: "candidate-1", legalHtsCode: "7308.90.95" },
        { candidateId: "candidate-2", legalHtsCode: "7604.29.10" }
      ],
      ["candidate-2"],
      { mode: "published" }
    ),
    {
      selectedIds: ["candidate-2"],
      selectedRow: { candidateId: "candidate-2", legalHtsCode: "7604.29.10" },
      canEditDetail: false,
      selectionLabel: "Inspecting row details."
    }
  );

  assert.deepEqual(resolveSection232ReviewSelection([], [], { mode: "published" }), {
    selectedIds: [],
    selectedRow: null,
    canEditDetail: false,
    selectionLabel: "Select one row to inspect details."
  });
});

test("draft selection tracks explicit candidate ids across page changes", () => {
  const initialSelection = createSection232DraftSelectionState();
  const updatedSelection = toggleSection232DraftSelection(initialSelection, "candidate-2", true);

  assert.equal(isSection232DraftRowSelected(updatedSelection, "candidate-2"), true);
  assert.equal(isSection232DraftRowSelected(updatedSelection, "candidate-3"), false);
  assert.equal(getSection232DraftSelectionCount(updatedSelection, 200), 1);
  assert.deepEqual(buildSection232DraftBulkReviewPayload(updatedSelection, "accepted"), {
    selection_mode: "explicit",
    candidate_ids: ["candidate-2"],
    decision: "accepted"
  });
});

test("draft selection supports workspace-wide select all with page-local exclusions", () => {
  const allSelection = selectAllSection232DraftRows();
  const excludedSelection = toggleSection232DraftSelection(allSelection, "candidate-42", false);

  assert.equal(isSection232DraftRowSelected(excludedSelection, "candidate-41"), true);
  assert.equal(isSection232DraftRowSelected(excludedSelection, "candidate-42"), false);
  assert.equal(getSection232DraftSelectionCount(excludedSelection, 200), 199);
  assert.deepEqual(buildSection232DraftBulkReviewPayload(excludedSelection, "rejected"), {
    selection_mode: "all",
    excluded_candidate_ids: ["candidate-42"],
    decision: "rejected"
  });
  assert.deepEqual(clearSection232DraftSelection(), {
    mode: "explicit",
    candidateIds: [],
    excludedCandidateIds: []
  });
});

test("buildSection232CatalogStatus renders family-match catalog rows as informational", () => {
  assert.deepEqual(
    buildSection232CatalogStatus({
      catalog_match_type: "family",
      catalog_representative_code: "7407.10.30.00",
      catalog_family_match_count: 3,
      catalog_warning: ""
    }),
    {
      state: "Information",
      label: "Family match",
      detail: "Using 7407.10.30.00 as the family representative across 3 catalog entries."
    }
  );
});

test("buildSection232ReviewRowViewModel exposes effective window summary fields without tariff labels", () => {
  const row = buildSection232ReviewRowViewModel({
    candidate_id: "candidate-42",
    legal_hts_code: "7308.90.95",
    description: "Structural steel component",
    rule_type: "rate_schedule",
    coverage_effect: "include",
    effective_from: "2026-04-07",
    effective_to: "2027-12-31",
    metal_scope: "steel derivative articles",
    review_decision: "pending"
  });

  assert.equal(row.effectiveWindowLabel, "From 2026-04-07 to 2027-12-31");
  assert.equal(row.effectiveWindowSummary, "From 2026-04-07 to 2027-12-31");
});

test("buildSection232ReviewRowViewModel formats partial and missing effective windows explicitly", () => {
  const startOnlyRow = buildSection232ReviewRowViewModel({
    candidate_id: "candidate-start",
    effective_from: "2026-04-07"
  });
  const endOnlyRow = buildSection232ReviewRowViewModel({
    candidate_id: "candidate-end",
    effective_to: "2027-12-31"
  });
  const openWindowRow = buildSection232ReviewRowViewModel({
    candidate_id: "candidate-open"
  });

  assert.equal(startOnlyRow.effectiveWindowLabel, "Starts 2026-04-07, open-ended");
  assert.equal(startOnlyRow.effectiveWindowSummary, "Starts 2026-04-07, open-ended");
  assert.equal(endOnlyRow.effectiveWindowLabel, "Until 2027-12-31");
  assert.equal(endOnlyRow.effectiveWindowSummary, "Until 2027-12-31");
  assert.equal(openWindowRow.effectiveWindowLabel, "No explicit effective window");
  assert.equal(openWindowRow.effectiveWindowSummary, "No explicit effective window");
});

test("buildSection232ReviewRowViewModel preserves occurrence evidence for the detail pane", () => {
  const row = buildSection232ReviewRowViewModel({
    candidate_id: "candidate-evidence",
    source_excerpt: "Synthetic review excerpt.",
    match_evidence: [
      {
        source_filename: "notice.pdf",
        page_number: 6,
        matched_text: "240814",
        normalized_hts_code: "2408.14",
        context_text: "[Docket No. 240814-0099]",
        text_sources: ["plain", "layout"]
      }
    ]
  });

  assert.equal(row.matchEvidenceCount, 1);
  assert.equal(row.primaryMatchEvidence.pageLabel, "P6");
  assert.equal(row.primaryMatchEvidence.contextText, "[Docket No. 240814-0099]");
  assert.equal(row.primaryMatchEvidence.textSourceLabel, "Plain text + layout text");
});

test("buildSection232ReviewRowViewModel maps source timing metadata for review details", () => {
  const row = buildSection232ReviewRowViewModel({
    candidate_id: "candidate-timing",
    source_filenames: ["notice.pdf"],
    source_documents: [
      {
        source_id: "source-1",
        filename: "notice.pdf",
        uploaded_at: "2026-04-19T10:00:00Z"
      }
    ],
    source_uploaded_at: "2026-04-19T10:00:00Z",
    processed_at: "2026-04-19T10:05:00Z"
  });

  assert.deepEqual(row.sourceDocuments, [
    {
      sourceId: "source-1",
      filename: "notice.pdf",
      uploadedAt: "2026-04-19T10:00:00Z"
    }
  ]);
  assert.equal(row.sourceUploadedAt, "2026-04-19T10:00:00Z");
  assert.equal(row.processedAt, "2026-04-19T10:05:00Z");
});

test("buildSection232ReviewInteractivity keeps published workspaces browse-only but allows delete with actor", () => {
  assert.deepEqual(
    buildSection232ReviewInteractivity({
      mode: "published",
      loading: false,
      mutationInFlight: false,
      publishedBy: "auditor",
      rowCount: 4,
      selection: {
        selectedIds: ["candidate-1"],
        selectedRow: { candidateId: "candidate-1" },
        canEditDetail: false
      }
    }),
    {
      isReadOnly: true,
      showBulkActions: false,
      showDecisionEditor: false,
      showDeleteAction: true,
      enableRowCheckboxes: false,
      canBulkReview: false,
      canSelectAllRows: false,
      canClearSelection: false,
      canEditDetailDecision: false,
      canDeleteDetailCode: true
    }
  );
});

test("buildSection232ReviewInteractivity disables draft review actions while loading or mutating", () => {
  assert.deepEqual(
    buildSection232ReviewInteractivity({
      mode: "draft",
      loading: true,
      mutationInFlight: false,
      publishedBy: "reviewer",
      rowCount: 3,
      selection: {
        selectedIds: ["candidate-1", "candidate-2"],
        canEditDetail: false
      }
    }),
    {
      isReadOnly: false,
      showBulkActions: true,
      showDecisionEditor: true,
      showDeleteAction: false,
      enableRowCheckboxes: false,
      canBulkReview: false,
      canSelectAllRows: false,
      canClearSelection: false,
      canEditDetailDecision: false,
      canDeleteDetailCode: false
    }
  );

  assert.deepEqual(
    buildSection232ReviewInteractivity({
      mode: "draft",
      loading: false,
      mutationInFlight: false,
      publishedBy: "reviewer",
      rowCount: 3,
      selection: {
        selectedIds: ["candidate-2"],
        selectedRow: { candidateId: "candidate-2" },
        canEditDetail: true
      }
    }),
    {
      isReadOnly: false,
      showBulkActions: true,
      showDecisionEditor: true,
      showDeleteAction: true,
      enableRowCheckboxes: true,
      canBulkReview: true,
      canSelectAllRows: true,
      canClearSelection: true,
      canEditDetailDecision: true,
      canDeleteDetailCode: true
    }
  );
});

test("buildSection232ReviewInteractivity disables select all when every draft row is already selected", () => {
  assert.deepEqual(
    buildSection232ReviewInteractivity({
      mode: "draft",
      loading: false,
      mutationInFlight: false,
      publishedBy: "reviewer",
      rowCount: 2,
      selection: {
        selectedIds: ["candidate-1", "candidate-2"],
        canEditDetail: false
      }
    }),
    {
      isReadOnly: false,
      showBulkActions: true,
      showDecisionEditor: true,
      showDeleteAction: false,
      enableRowCheckboxes: true,
      canBulkReview: true,
      canSelectAllRows: false,
      canClearSelection: true,
      canEditDetailDecision: false,
      canDeleteDetailCode: false
    }
  );
});

test("buildSection232ReviewInteractivity requires Published by before draft review actions unlock", () => {
  assert.deepEqual(
    buildSection232ReviewInteractivity({
      mode: "draft",
      loading: false,
      mutationInFlight: false,
      publishedBy: "   ",
      rowCount: 2,
      selection: {
        selectedIds: ["candidate-1"],
        canEditDetail: true
      }
    }),
    {
      isReadOnly: false,
      showBulkActions: true,
      showDecisionEditor: true,
      showDeleteAction: false,
      enableRowCheckboxes: true,
      canBulkReview: false,
      canSelectAllRows: true,
      canClearSelection: true,
      canEditDetailDecision: false,
      canDeleteDetailCode: false
    }
  );
});

test("buildSection232ReviewPublishState counts accepted and pending rows for publish gating", () => {
  assert.deepEqual(
    buildSection232ReviewPublishState(
      [
        { reviewDecision: "accepted" },
        { reviewDecision: "rejected" },
        { reviewDecision: "accepted" }
      ],
      "  section-232-review-ui  "
    ),
    {
      acceptedCount: 2,
      pendingCount: 0,
      publishedBy: "section-232-review-ui",
      canPublish: true
    }
  );

  assert.deepEqual(
    buildSection232ReviewPublishState(
      [
        { reviewDecision: "accepted" },
        { reviewDecision: "pending" }
      ],
      "section-232-review-ui"
    ),
    {
      acceptedCount: 1,
      pendingCount: 1,
      publishedBy: "section-232-review-ui",
      canPublish: true
    }
  );

  assert.deepEqual(
    buildSection232ReviewPublishState(
      [
        { reviewDecision: "accepted" },
        { reviewDecision: "rejected" }
      ],
      "   "
    ),
    {
      acceptedCount: 1,
      pendingCount: 0,
      publishedBy: "",
      canPublish: false
    }
  );
});

test("buildSection232ReviewDiagnostics reports every duplicate and overlapping rule group up front", () => {
  const diagnostics = buildSection232ReviewDiagnostics([
    buildSection232ReviewRowViewModel({
      candidate_id: "candidate-duplicate-1",
      legal_hts_code: "8207.30.6062",
      rule_type: "rate_schedule",
      coverage_effect: "include",
      metal_scope: "steel and aluminum",
      effective_from: "2024-04-01",
      effective_to: "2027-12-31",
      review_decision: "pending"
    }),
    buildSection232ReviewRowViewModel({
      candidate_id: "candidate-duplicate-2",
      legal_hts_code: "8207.30.6062",
      rule_type: "rate_schedule",
      coverage_effect: "include",
      metal_scope: "aluminum + steel",
      effective_from: "2024-04-01",
      effective_to: "2027-12-31",
      review_decision: "accepted"
    }),
    buildSection232ReviewRowViewModel({
      candidate_id: "candidate-overlap-1",
      legal_hts_code: "8207.30.6062",
      rule_type: "rate_schedule",
      coverage_effect: "include",
      metal_scope: "steel, aluminum",
      effective_from: "2026-04-07",
      effective_to: "2027-12-31",
      review_decision: "pending"
    }),
    buildSection232ReviewRowViewModel({
      candidate_id: "candidate-ignored",
      legal_hts_code: "8207.30.6062",
      rule_type: "rate_schedule",
      coverage_effect: "include",
      metal_scope: "steel and aluminum",
      effective_from: "2026-04-07",
      effective_to: "2027-12-31",
      review_decision: "rejected"
    })
  ]);

  assert.equal(diagnostics.summary, "2 Section 232 review issues need attention before publish.");
  assert.deepEqual(diagnostics.items, [
    "Duplicate rows share 8207.30.6062 · rate_schedule · aluminum+steel · From 2024-04-01 to 2027-12-31.",
    "Overlapping rows share 8207.30.6062 · rate_schedule · aluminum+steel. Keep the widest window: From 2024-04-01 to 2027-12-31."
  ]);
});

test("buildSection232ReviewMutationRecoveryState clears mutation state and preserves the error message", () => {
  assert.deepEqual(
    buildSection232ReviewMutationRecoveryState("Failed to update draft review decision.", "Negative"),
    {
      mutationInFlight: false,
      message: "Failed to update draft review decision.",
      messageDesign: "Negative"
    }
  );
});

test("buildSection232ReviewRequestQuery includes trimmed HTS search text", () => {
  const route = parseSection232ReviewRoute("?version=section232-v0002");

  assert.deepEqual(buildSection232ReviewRequestQuery(route, "  3403.99  "), {
    version: "section232-v0002",
    hts_query: "3403.99"
  });
  assert.deepEqual(buildSection232ReviewRequestQuery(route, "   "), {
    version: "section232-v0002"
  });
});

test("buildSection232HtsSearchControlState only enables fetch when submitted explicitly", () => {
  assert.deepEqual(
    buildSection232HtsSearchControlState({
      inputQuery: "3403.99",
      appliedQuery: "",
      routeValid: true,
      loading: false,
      mutationInFlight: false
    }),
    {
      inputQuery: "3403.99",
      appliedQuery: "",
      disabled: false,
      hasInput: true,
      hasAppliedQuery: false,
      hasPendingQuery: true,
      canSubmit: true,
      canClear: true
    }
  );

  assert.equal(
    buildSection232ReviewRequestQuery(parseSection232ReviewRoute("?version=section232-v0002"), "").hts_query,
    undefined
  );
});

test("buildSection232ReviewRowViewModel maps published history for detail labels", () => {
  const row = buildSection232ReviewRowViewModel({
    candidate_id: "candidate-current",
    legal_hts_code: "3403.99.00",
    history: [
      {
        version: "section232-v0001",
        published_at: "2026-04-23T12:20:00Z",
        published_by: "pytest",
        candidate_id: "candidate-older",
        legal_hts_code: "3403.99.00",
        hts_code: "3403.99.00",
        rule_type: "include",
        coverage_effect: "include",
        source_filenames: ["2025-15819.pdf"],
        processed_at: "2026-04-23T12:11:22Z"
      }
    ]
  });

  assert.equal(row.history.length, 1);
  assert.deepEqual(row.history[0], {
    version: "section232-v0001",
    publishedAt: "2026-04-23T12:20:00Z",
    publishedBy: "pytest",
    candidateId: "candidate-older",
    legalHtsCode: "3403.99.00",
    ruleType: "include",
    coverageEffect: "include",
    effectiveFrom: "",
    effectiveTo: "",
    sourceFilenames: ["2025-15819.pdf"],
    sourceSummary: "2025-15819.pdf",
    processedAt: "2026-04-23T12:11:22Z",
    actionLabel: "Included"
  });
});

test("buildSection232HistoryItemViewModel labels remove actions", () => {
  assert.deepEqual(
    buildSection232HistoryItemViewModel({
      version: "section232-v0002",
      candidate_id: "candidate-remove",
      legal_hts_code: "3403.99.00",
      rule_type: "remove",
      coverage_effect: "remove",
      source_filenames: ["ANNEXES-I-A-I-B-II-III-IV.pdf"]
    }),
    {
      version: "section232-v0002",
      publishedAt: "",
      publishedBy: "",
      candidateId: "candidate-remove",
      legalHtsCode: "3403.99.00",
      ruleType: "remove",
      coverageEffect: "remove",
      effectiveFrom: "",
      effectiveTo: "",
      sourceFilenames: ["ANNEXES-I-A-I-B-II-III-IV.pdf"],
      sourceSummary: "ANNEXES-I-A-I-B-II-III-IV.pdf",
      processedAt: "",
      actionLabel: "Removed"
    }
  );
});
