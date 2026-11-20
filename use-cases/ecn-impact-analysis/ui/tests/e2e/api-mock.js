import {
  businessEvidenceEnvelope,
  detailEnvelope,
  listEnvelope,
  missingIdentifierRow,
  progressEnvelope,
  providerStatusEnvelope,
  plantSettingsEnvelope,
  referenceDatasetsEnvelope,
  rowOne,
  uploadEnvelope,
} from "../fixtures/api-responses.js";

const duplicateRow = {
  ...rowOne,
  ecn_row_id: "66666666-6666-4666-8666-666666666666",
  source_ordinal: 2,
};
const failedRow = {
  ...rowOne,
  ecn_row_id: "77777777-7777-4777-8777-777777777777",
  ecn_number: "ECN-FAILED",
  display_ecn_number: "ECN-FAILED",
  source_ordinal: 3,
  analysis_state: "FAILED",
  completeness: null,
};

const JOB_IDS = Object.freeze({
  [rowOne.ecn_row_id]: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
  [duplicateRow.ecn_row_id]: "88888888-8888-4888-8888-888888888888",
  [missingIdentifierRow.ecn_row_id]: "99999999-9999-4999-8999-999999999999",
  [failedRow.ecn_row_id]: "ffffffff-ffff-4fff-8fff-ffffffffffff",
});
const successfulStates = new Set(["COMPLETED", "COMPLETED_WITH_GAPS"]);

/** Return initial attempt counts for every browser-fixture job. */
const initialAttemptCounts = () => new Map(
  Object.keys(JOB_IDS).map((rowId) => [rowId, 1]),
);

/** Return mutually consistent list rows for a seeded or post-upload state. */
function initialRows(completedImmediately) {
  return [
    {
      ...rowOne,
      analysis_state: completedImmediately ? "COMPLETED_WITH_GAPS" : "READY",
      completeness: completedImmediately ? "COMPLETED_WITH_GAPS" : null,
    },
    {
      ...duplicateRow,
      analysis_state: "COMPLETED_WITH_GAPS",
      completeness: "COMPLETED_WITH_GAPS",
    },
    {
      ...missingIdentifierRow,
      analysis_state: "COMPLETED_WITH_GAPS",
      completeness: "COMPLETED_WITH_GAPS",
    },
    failedRow,
  ];
}

/** Derive the exact progress envelope from the list's current row states. */
function progressForRows(rows) {
  const count = (state) => rows.filter((row) => row.analysis_state === state).length;
  const queued_count = count("QUEUED");
  const blocked_count = count("BLOCKED");
  const running_count = count("RUNNING");
  const completed_count = count("COMPLETED");
  const completed_with_gaps_count = count("COMPLETED_WITH_GAPS");
  const failed_count = count("FAILED");
  return progressEnvelope({
    ready_count: count("READY"),
    queued_count,
    blocked_count,
    running_count,
    completed_count,
    completed_with_gaps_count,
    failed_count,
    unfinished_count: queued_count + blocked_count + running_count,
    total_count: rows.length,
  });
}

/** Install a deterministic same-origin HTTP double and expose its state. */
export async function installApiMock(
  page,
  { seeded = false, completedImmediately = false } = {},
) {
  const state = {
    rows: seeded ? initialRows(completedImmediately) : [],
    requests: [],
    uploadBodies: [],
    detailReads: new Map(),
    attemptCounts: initialAttemptCounts(),
    retriedRows: new Set(),

    /** Move one row to a stored completed-with-gaps result. */
    completeAnalysis(rowId) {
      this.rows = this.rows.map((row) => row.ecn_row_id === rowId
        ? {
          ...row,
          analysis_state: "COMPLETED_WITH_GAPS",
          completeness: "COMPLETED_WITH_GAPS",
        }
        : row);
      if (this.retriedRows.delete(rowId)) {
        this.attemptCounts.set(rowId, (this.attemptCounts.get(rowId) || 0) + 1);
      }
    },

    /** Toggle one row between queued and terminal for coherent reset progress. */
    setResetActive(active) {
      const rowId = this.rows[0]?.ecn_row_id;
      if (!rowId) return;
      this.rows = this.rows.map((row) => row.ecn_row_id === rowId
        ? {
          ...row,
          analysis_state: active ? "QUEUED" : "COMPLETED_WITH_GAPS",
          completeness: active ? null : "COMPLETED_WITH_GAPS",
        }
        : row);
    },
  };

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const method = request.method();
    state.requests.push({ method, path: url.pathname, search: url.search });

    if (method === "POST" && url.pathname === "/api/uploads") {
      state.uploadBodies.push(request.postData() || "");
      state.rows = initialRows(false);
      state.attemptCounts = initialAttemptCounts();
      await route.fulfill({
        status: 202,
        json: {
          ...uploadEnvelope,
          imported_ecn_count: 4,
          ready_count: 4,
          queued_count: 0,
          blocked_count: 0,
          running_count: 0,
          completed_count: 0,
          completed_with_gaps_count: 0,
          failed_count: 0,
        },
      });
      return;
    }
    if (method === "POST" && url.pathname === "/api/summary-batches") {
      const body = request.postDataJSON();
      const selectedIds = body.selection.mode === "explicit"
        ? body.selection.ecn_ids
        : state.rows
          .map((row) => row.ecn_row_id)
          .filter((rowId) => !body.selection.excluded_ecn_ids.includes(rowId));
      const scheduledIds = selectedIds.filter((rowId) => {
        const row = state.rows.find((item) => item.ecn_row_id === rowId);
        return row && ["READY", "FAILED", "BLOCKED"].includes(row.analysis_state);
      });
      state.rows = state.rows.map((row) => scheduledIds.includes(row.ecn_row_id)
        ? { ...row, analysis_state: "RUNNING", completeness: null }
        : row);
      scheduledIds.forEach((rowId) => state.detailReads.set(rowId, 0));
      await route.fulfill({
        status: 202,
        json: {
          batch_id: body.request_id,
          cancelled: false,
          selection: {
            ecn_ids: selectedIds,
            scheduled_job_ids: scheduledIds.map((rowId) => JOB_IDS[rowId]),
          },
          jobs: selectedIds.map((rowId) => ({
            job_id: JOB_IDS[rowId],
            ecn_row_id: rowId,
            state: scheduledIds.includes(rowId) ? "QUEUED" : "COMPLETED",
          })),
          finished: scheduledIds.length === 0,
        },
      });
      return;
    }
    if (method === "GET" && url.pathname === "/api/ecns") {
      await route.fulfill({ json: listEnvelope(state.rows) });
      return;
    }
    if (method === "GET" && url.pathname === "/api/analysis-progress") {
      await route.fulfill({ json: progressForRows(state.rows) });
      return;
    }

    const evidenceMatch = url.pathname.match(/^\/api\/ecns\/([^/]+)\/evidence$/);
    if (method === "GET" && evidenceMatch) {
      await route.fulfill({
        json: businessEvidenceEnvelope(url.searchParams.get("section")),
      });
      return;
    }

    if (method === "GET" && url.pathname === "/api/provider-status") {
      await route.fulfill({ json: providerStatusEnvelope() });
      return;
    }
    if (method === "POST" && url.pathname === "/api/provider-status/check") {
      await route.fulfill({ json: providerStatusEnvelope() });
      return;
    }
    if (method === "GET" && url.pathname === "/api/reference-datasets") {
      await route.fulfill({ json: referenceDatasetsEnvelope() });
      return;
    }
    if (method === "GET" && url.pathname === "/api/plant-settings") {
      await route.fulfill({ json: plantSettingsEnvelope() });
      return;
    }
    if (method === "PUT" && url.pathname === "/api/plant-settings") {
      await route.fulfill({ json: plantSettingsEnvelope(request.postDataJSON()) });
      return;
    }

    const retryMatch = url.pathname.match(/^\/api\/ecns\/([^/]+)\/analysis\/retry$/);
    if (method === "POST" && retryMatch) {
      const target = state.rows.find((row) => row.ecn_row_id === retryMatch[1]);
      if (!target) {
        await route.fulfill({
          status: 404,
          json: {
            code: "ECN_NOT_FOUND",
            message: "ECN occurrence not found",
            details: null,
          },
        });
        return;
      }
      if (target.analysis_state !== "FAILED") {
        await route.fulfill({
          status: 409,
          json: {
            code: "ANALYSIS_NOT_RETRYABLE",
            message: "Only a failed analysis can be retried",
            details: null,
          },
        });
        return;
      }
      state.rows = state.rows.map((row) => row.ecn_row_id === retryMatch[1]
        ? { ...row, analysis_state: "QUEUED", completeness: null }
        : row);
      state.retriedRows.add(retryMatch[1]);
      state.detailReads.set(retryMatch[1], 0);
      await route.fulfill({
        status: 202,
        json: {
          ecn_row_id: retryMatch[1],
          job_id: JOB_IDS[retryMatch[1]],
          state: "QUEUED",
        },
      });
      return;
    }

    const projectionMatch = url.pathname.match(/^\/api\/ecns\/([^/]+)\/rdf-projection$/);
    if (method === "GET" && projectionMatch) {
      const row = state.rows.find((item) => item.ecn_row_id === projectionMatch[1]);
      if (!row) {
        await route.fulfill({
          status: 404,
          json: { code: "RDF_PROJECTION_NOT_FOUND", message: "Projection not found", details: null },
        });
        return;
      }
      await route.fulfill({
        json: {
          ecn_row_id: row.ecn_row_id,
          status: "CURRENT",
          source_sha256: "a".repeat(64),
          projection_version: "1.0.0",
          verified_node_count: 12,
          verified_triple_count: 48,
          error_code: null,
          error_message: null,
          last_attempt_at: "2026-09-18T08:00:00Z",
          current_at: "2026-09-18T08:00:01Z",
        },
      });
      return;
    }

    const detailMatch = url.pathname.match(/^\/api\/ecns\/([^/]+)$/);
    if (method === "GET" && detailMatch) {
      const rowId = detailMatch[1];
      let row = state.rows.find((item) => item.ecn_row_id === rowId);
      if (!row) {
        await route.fulfill({
          status: 404,
          json: {
            code: "ECN_NOT_FOUND",
            message: "ECN occurrence not found",
            details: null,
          },
        });
        return;
      }
      const reads = (state.detailReads.get(rowId) || 0) + 1;
      state.detailReads.set(rowId, reads);
      if (
        rowId === rowOne.ecn_row_id
        && row.analysis_state === "RUNNING"
        && reads > 1
      ) {
        await new Promise((resolve) => { setTimeout(resolve, 250); });
        state.completeAnalysis(rowId);
        row = state.rows.find((item) => item.ecn_row_id === rowId);
      }
      const complete = successfulStates.has(row.analysis_state);
      const baseAnalysis = detailEnvelope().analysis;
      await route.fulfill({
        json: detailEnvelope({
          ecn_row_id: rowId,
          ecn_number: row.ecn_number,
          display_ecn_number: row.display_ecn_number,
          title: row.title,
          description_summary: row.description_summary,
          source_ordinal: row.source_ordinal,
          counts: row.counts,
          analysis: complete
            ? { ...baseAnalysis, completeness: row.completeness }
            : null,
          job: {
            job_id: JOB_IDS[rowId],
            state: complete ? "COMPLETED" : row.analysis_state,
            attempt_count: state.attemptCounts.get(rowId) || 0,
            error_code: row.analysis_state === "FAILED" ? "PROVIDER_TIMEOUT" : null,
            error_message: row.analysis_state === "FAILED" ? "Generation timed out" : null,
          },
        }),
      });
      return;
    }

    if (method === "DELETE" && url.pathname === "/api/history") {
      if (state.rows.some((row) => ["QUEUED", "BLOCKED", "RUNNING"].includes(row.analysis_state))) {
        await route.fulfill({
          status: 409,
          json: {
            code: "HISTORY_RESET_ACTIVE_JOBS",
            message: "Analysis is active",
            details: null,
          },
        });
        return;
      }
      const deletedAnalysisCount = state.rows.filter(
        (row) => successfulStates.has(row.analysis_state),
      ).length;
      const deletedAttemptCount = state.rows.reduce(
        (total, row) => total + (state.attemptCounts.get(row.ecn_row_id) || 0),
        0,
      );
      const deletedJobCount = state.rows.length;
      state.rows = [];
      state.detailReads.clear();
      state.retriedRows.clear();
      state.attemptCounts = initialAttemptCounts();
      await route.fulfill({
        json: {
          deleted_attempt_count: deletedAttemptCount,
          deleted_analysis_count: deletedAnalysisCount,
          deleted_job_count: deletedJobCount,
          deleted_ecn_count: deletedJobCount,
          deleted_upload_count: 1,
          deleted_bom_row_count: 99,
          deleted_source_artifact_count: 2,
        },
      });
      return;
    }

    await route.fulfill({
      status: 404,
      json: { code: "NOT_FOUND", message: "No mocked route", details: null },
    });
  });
  return state;
}
