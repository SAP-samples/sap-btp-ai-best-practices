import test from "node:test";
import assert from "node:assert/strict";

import {
  SECTION232_RESET_CONFIRMATION_TEXT,
  buildOpenReviewHref,
  buildPublishedReviewHref,
  buildCancelDraftState,
  buildReviewRowViewModel,
  decideDraftRestoreSource,
  buildSettingsLaunchState,
  buildSection232ResetImpactCards,
  buildRulesetSummaryCards,
  filterReviewRowViewModels,
  isSection232ResetConfirmationValid,
  isPendingReviewDraftBatch,
  selectLatestPendingDraftBatch
} from "../src/pages/settings/settings_review_state.js";

test("buildReviewRowViewModel marks missing catalog entries without tariff-specific labels", () => {
  const row = buildReviewRowViewModel({
    candidate_id: "candidate-2",
    hts_code: "8421.29.00",
    description: null,
    rule_type: "rate_schedule",
    coverage_effect: "include",
    effective_from: "2026-04-07",
    effective_to: "2027-12-31",
    catalog_match_found: false,
    source_filenames: ["ANNEXES-I-A-I-B-II-III-IV.pdf"],
    source_pages: [40]
  });

  assert.equal(row.changeLabel, "Include");
  assert.equal(row.description, "");
  assert.equal(row.catalogWarning, "HTS code not found in managed catalog");
  assert.deepEqual(row.sourceFilenames, ["ANNEXES-I-A-I-B-II-III-IV.pdf"]);
  assert.equal(row.sourceSummary, "ANNEXES-I-A-I-B-II-III-IV.pdf");
  assert.match(row.searchText, /annexes-i-a-i-b-ii-iii-iv\.pdf/);
});

test("filterReviewRowViewModels matches review rows by source filename", () => {
  const rows = [
    buildReviewRowViewModel({
      candidate_id: "candidate-1",
      hts_code: "7308.90.95",
      description: "Steel derivative article",
      rule_type: "include",
      coverage_effect: "include",
      effective_from: "2026-04-07",
      catalog_match_found: true,
      source_filenames: ["steel-notice.pdf"],
      review_decision: "pending"
    }),
    buildReviewRowViewModel({
      candidate_id: "candidate-2",
      hts_code: "7604.29.10",
      description: "Aluminum derivative article",
      rule_type: "include",
      coverage_effect: "include",
      effective_from: "2026-04-07",
      catalog_match_found: true,
      source_filenames: ["aluminum-notice.pdf"],
      review_decision: "pending"
    })
  ];

  const filteredRows = filterReviewRowViewModels(rows, { sourceDocument: "steel-notice.pdf" });

  assert.deepEqual(filteredRows.map((row) => row.candidateId), ["candidate-1"]);
});

test("buildRulesetSummaryCards keeps eligible count separate from pending review rows", () => {
  const cards = buildRulesetSummaryCards({
    active_ruleset_version: "section232-v0001",
    eligible_hts_code_count: 863,
    pending_draft_batch_count: 1,
    last_published_at: "2026-04-19T08:40:00Z"
  });

  assert.deepEqual(cards[0], { label: "Active ruleset", value: "section232-v0001" });
  assert.deepEqual(cards[1], { label: "Eligible code count", value: "863" });
  assert.deepEqual(cards[2], { label: "Pending drafts", value: "1" });
  assert.deepEqual(cards[3], { label: "Last published", value: "2026-04-19T08:40:00Z" });
});

test("filterReviewRowViewModels matches metal families for derivative backend scopes", () => {
  const rows = [
    buildReviewRowViewModel({
      candidate_id: "steel-1",
      hts_code: "7308.90.95",
      description: "Steel derivative article",
      rule_type: "include",
      coverage_effect: "include",
      effective_from: "2026-04-07",
      metal_scope: "steel derivative articles",
      catalog_match_found: true,
      review_decision: "pending"
    }),
    buildReviewRowViewModel({
      candidate_id: "aluminum-1",
      hts_code: "7604.29.10",
      description: "Aluminum derivative article",
      rule_type: "include",
      coverage_effect: "include",
      effective_from: "2026-04-07",
      metal_scope: "aluminum derivatives",
      catalog_match_found: true,
      review_decision: "pending"
    })
  ];

  const steelRows = filterReviewRowViewModels(rows, { metalScope: "steel" });
  const aluminumRows = filterReviewRowViewModels(rows, { metalScope: "aluminum" });

  assert.deepEqual(steelRows.map((row) => row.candidateId), ["steel-1"]);
  assert.deepEqual(aluminumRows.map((row) => row.candidateId), ["aluminum-1"]);
});

test("buildOpenReviewHref returns the route-driven draft review href", () => {
  assert.equal(
    buildOpenReviewHref({
      batch_id: "batch-2026-04-19"
    }),
    "/settings/section-232/review?batch=batch-2026-04-19"
  );
});

test("buildPublishedReviewHref returns the route-driven published review href", () => {
  assert.equal(buildPublishedReviewHref("section232-v0007"), "/settings/section-232/review?version=section232-v0007");
});

test("selectLatestPendingDraftBatch returns the latest pending batch from the server summary", () => {
  const batch = selectLatestPendingDraftBatch({
    pending_draft_batches: [
      {
        batch_id: "batch-old",
        created_at: "2026-04-18T08:10:00Z",
        source_filenames: ["old-notice.pdf"]
      },
      {
        batch_id: "batch-new",
        created_at: "2026-04-19T08:10:00Z",
        source_filenames: ["new-notice.pdf"]
      }
    ]
  });

  assert.equal(batch.batch_id, "batch-new");
  assert.deepEqual(batch.source_filenames, ["new-notice.pdf"]);
});

test("selectLatestPendingDraftBatch ignores entries without batch ids and falls back to null", () => {
  assert.equal(
    selectLatestPendingDraftBatch({
      pending_draft_batches: [{ batch_id: "   " }, { created_at: "2026-04-19T08:10:00Z" }]
    }),
    null
  );
});

test("isPendingReviewDraftBatch only accepts snapshots that are still pending review", () => {
  assert.equal(
    isPendingReviewDraftBatch({
      batch_id: "batch-pending",
      status: "pending_review",
      pending_count: 2
    }),
    true
  );
  assert.equal(
    isPendingReviewDraftBatch({
      batch_id: "batch-ready-to-publish",
      status: "pending_review",
      pending_count: 0
    }),
    true
  );
  assert.equal(
    isPendingReviewDraftBatch({
      batch_id: "batch-published",
      status: "published",
      pending_count: 0
    }),
    false
  );
});

test("buildSettingsLaunchState prefers the current draft batch and exposes the review href", () => {
  const launchState = buildSettingsLaunchState({
    currentDraftBatch: {
      batch_id: "batch-session",
      status: "pending_review",
      pending_count: 4,
      created_at: "2026-04-19T09:00:00Z"
    },
    rulesetSummary: {
      pending_draft_batch_count: 2,
      pending_draft_batches: [
        {
          batch_id: "batch-server",
          created_at: "2026-04-19T08:00:00Z"
        }
      ]
    }
  });

  assert.equal(launchState.hasLaunchTarget, true);
  assert.equal(launchState.batch.batch_id, "batch-session");
  assert.equal(launchState.pendingDraftCount, 2);
  assert.equal(launchState.href, "/settings/section-232/review?batch=batch-session");
});

test("buildSettingsLaunchState still reopens a fully reviewed pending batch", () => {
  const launchState = buildSettingsLaunchState({
    currentDraftBatch: {
      batch_id: "batch-ready-to-publish",
      status: "pending_review",
      pending_count: 0,
      created_at: "2026-04-19T09:00:00Z"
    },
    rulesetSummary: {
      pending_draft_batch_count: 0,
      pending_draft_batches: []
    }
  });

  assert.equal(launchState.hasLaunchTarget, true);
  assert.equal(launchState.batch.batch_id, "batch-ready-to-publish");
  assert.equal(launchState.href, "/settings/section-232/review?batch=batch-ready-to-publish");
});

test("buildSettingsLaunchState ignores stale persisted batches that are no longer pending", () => {
  const launchState = buildSettingsLaunchState({
    currentDraftBatch: {
      batch_id: "batch-published",
      status: "published",
      pending_count: 0,
      created_at: "2026-04-19T09:00:00Z"
    },
    rulesetSummary: {
      pending_draft_batch_count: 0,
      pending_draft_batches: []
    }
  });

  assert.equal(launchState.hasLaunchTarget, false);
  assert.equal(launchState.batch, null);
  assert.equal(launchState.href, "");
});

test("buildSettingsLaunchState falls back to the latest server pending batch", () => {
  const launchState = buildSettingsLaunchState({
    currentDraftBatch: null,
    rulesetSummary: {
      pending_draft_batch_count: 1,
      pending_draft_batches: [
        {
          batch_id: "batch-server",
          created_at: "2026-04-19T08:00:00Z"
        }
      ]
    }
  });

  assert.equal(launchState.hasLaunchTarget, true);
  assert.equal(launchState.batch.batch_id, "batch-server");
  assert.equal(launchState.href, "/settings/section-232/review?batch=batch-server");
});

test("buildSettingsLaunchState disables launch when no pending draft batch is available", () => {
  const launchState = buildSettingsLaunchState({
    currentDraftBatch: null,
    rulesetSummary: {
      pending_draft_batch_count: 0,
      pending_draft_batches: []
    }
  });

  assert.equal(launchState.hasLaunchTarget, false);
  assert.equal(launchState.batch, null);
  assert.equal(launchState.href, "");
});

test("buildCancelDraftState enables cancellation for the current pending draft batch", () => {
  const cancelState = buildCancelDraftState({
    currentDraftBatch: {
      batch_id: "batch-session",
      status: "pending_review",
      created_at: "2026-04-19T09:00:00Z"
    },
    rulesetSummary: {
      pending_draft_batch_count: 1,
      pending_draft_batches: []
    },
    busy: false
  });

  assert.deepEqual(cancelState, {
    batch: {
      batch_id: "batch-session",
      status: "pending_review",
      created_at: "2026-04-19T09:00:00Z"
    },
    batchId: "batch-session",
    disabled: false,
    busy: false,
    hasCancelTarget: true
  });
});

test("buildCancelDraftState falls back to latest server pending batch and disables while busy", () => {
  const cancelState = buildCancelDraftState({
    currentDraftBatch: null,
    rulesetSummary: {
      pending_draft_batch_count: 1,
      pending_draft_batches: [
        {
          batch_id: "batch-server",
          status: "pending_review",
          created_at: "2026-04-19T08:00:00Z"
        }
      ]
    },
    busy: true
  });

  assert.equal(cancelState.hasCancelTarget, true);
  assert.equal(cancelState.batchId, "batch-server");
  assert.equal(cancelState.busy, true);
  assert.equal(cancelState.disabled, true);
});

test("buildCancelDraftState disables cancellation when no pending draft batch is available", () => {
  const cancelState = buildCancelDraftState({
    currentDraftBatch: null,
    rulesetSummary: {
      pending_draft_batch_count: 0,
      pending_draft_batches: []
    }
  });

  assert.equal(cancelState.hasCancelTarget, false);
  assert.equal(cancelState.batch, null);
  assert.equal(cancelState.batchId, "");
  assert.equal(cancelState.disabled, true);
});

test("decideDraftRestoreSource prefers a valid persisted draft snapshot over server pending batches", () => {
  assert.equal(
    decideDraftRestoreSource({
      serverPendingDraftCount: 2,
      restoredFromServer: false,
      persistedDraftBatch: {
        batch_id: "batch-session",
        status: "pending_review",
        pending_count: 1
      }
    }),
    "persisted"
  );
});

test("decideDraftRestoreSource falls back to the server batch when persisted state is stale", () => {
  assert.equal(
    decideDraftRestoreSource({
      serverPendingDraftCount: 2,
      restoredFromServer: false,
      persistedDraftBatch: {
        batch_id: "batch-session",
        status: "published",
        pending_count: 0
      }
    }),
    "server"
  );
});

test("decideDraftRestoreSource returns the server batch when persisted state is unavailable and server drafts remain", () => {
  assert.equal(
    decideDraftRestoreSource({
      serverPendingDraftCount: 2,
      restoredFromServer: false,
      persistedDraftBatch: null
    }),
    "server"
  );
});

test("decideDraftRestoreSource no-ops after a successful server restore", () => {
  assert.equal(
    decideDraftRestoreSource({
      serverPendingDraftCount: 2,
      restoredFromServer: true,
      persistedDraftBatch: {
        batch_id: "batch-session",
        status: "pending_review",
        pending_count: 1
      }
    }),
    "none"
  );
});

test("section 232 reset confirmation requires the exact typed phrase", () => {
  assert.equal(SECTION232_RESET_CONFIRMATION_TEXT, "RESET SECTION 232");
  assert.equal(isSection232ResetConfirmationValid("RESET SECTION 232"), true);
  assert.equal(isSection232ResetConfirmationValid(" RESET SECTION 232 "), true);
  assert.equal(isSection232ResetConfirmationValid("reset section 232"), false);
  assert.equal(isSection232ResetConfirmationValid("RESET SECTION"), false);
});

test("buildSection232ResetImpactCards summarizes destructive reset scope", () => {
  const cards = buildSection232ResetImpactCards({
    sourceCount: 4,
    rulesetSummary: {
      active_ruleset_version: "section232-v0002",
      eligible_hts_code_count: 863,
      pending_draft_batch_count: 2
    }
  });

  assert.deepEqual(cards, [
    { label: "Stored source PDFs", value: "4" },
    { label: "Pending drafts", value: "2" },
    { label: "Active ruleset", value: "section232-v0002" },
    { label: "Eligible codes", value: "863" }
  ]);
});
