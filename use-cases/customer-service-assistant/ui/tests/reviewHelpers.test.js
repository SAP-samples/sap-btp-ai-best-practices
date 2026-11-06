import test from "node:test";
import assert from "node:assert/strict";

import {
  draftDecisionPayload,
  formatFileSize,
  mapDraftFromApi,
  manualEmailPayload,
  parseJsonField,
  validateManualEmail,
  summarizeFetchResult,
  describeRun
} from "../src/pages/home/reviewHelpers.js";

test("parseJsonField returns fallback for invalid values", () => {
  assert.deepEqual(parseJsonField(null, []), []);
  assert.deepEqual(parseJsonField("not-json", []), []);
});

test("formatFileSize uses readable units", () => {
  assert.equal(formatFileSize(42), "42 B");
  assert.equal(formatFileSize(2048), "2.0 KB");
});

test("mapDraftFromApi normalizes active draft rows", () => {
  const item = mapDraftFromApi({
    draft_id: "draft-1",
    gmail_message_id: "m1",
    client_country: "GB",
    to_email: "customer@example.com",
    subject: "Re: Documents",
    body: "Attached.",
    status: "ready_for_review",
    selected_documents_json: JSON.stringify([
      { document_id: "doc-1", file_name: "Invoice_2741747.PDF", document_type: "invoice", size_bytes: 2048 }
    ]),
    validation_errors_json: "[]",
    sender_email: "customer@example.com",
    original_subject: "Documents",
    original_body: "Please send invoice 2741747.",
    email_date: "2026-04-03 11:10:57",
    confidence: "0.9",
    explanation: "Selected 1 document."
  });

  assert.equal(item.rowId, "draft-1");
  assert.equal(item.country, "GB");
  assert.equal(item.attachments[0].fileName, "Invoice_2741747.PDF");
  assert.equal(item.originalEmail.body, "Please send invoice 2741747.");
});

test("summarizeFetchResult describes processed emails", () => {
  assert.equal(
    summarizeFetchResult({
      fetched: 2,
      skipped_existing: 1,
      not_document_request: 0,
      extraction_failed: 0,
      agent_failed: 1,
      drafts_created: 1
    }),
    "Fetched 2, skipped 1, non-requests 0, extraction failures 0, agent failures 1, drafts 1"
  );
});

test("validateManualEmail requires from subject and body", () => {
  assert.deepEqual(
    validateManualEmail({ from: "", subject: "Order", body: "Status?" }),
    ["From is required"]
  );
  assert.deepEqual(
    validateManualEmail({ from: "Customer <customer@example.com>", subject: "Order", body: "Status?" }),
    []
  );
});

test("manualEmailPayload trims manual email fields", () => {
  assert.deepEqual(
    manualEmailPayload({ from: " Customer <customer@example.com> ", subject: " Order 5030000237 ", body: " Please advise. " }),
    {
      from: "Customer <customer@example.com>",
      subject: "Order 5030000237",
      body: "Please advise."
    }
  );
});

test("draftDecisionPayload preserves the current edited draft body", () => {
  assert.deepEqual(draftDecisionPayload("  Edited final reply.\n"), {
    body: "  Edited final reply.\n"
  });
});

test("describeRun reports progress, failure, and final summary", () => {
  assert.equal(
    describeRun({ status: "running", summary: { total: 5, fetched: 2, skipped_existing: 1 } }),
    "Processing 3 of 5 emails..."
  );
  assert.equal(describeRun({ status: "running", summary: { total: 0 } }), "Looking for new emails...");
  assert.equal(describeRun({ status: "failed", error_message: "Gmail 403" }), "Run failed: Gmail 403");
  assert.match(
    describeRun({
      status: "success",
      summary: { fetched: 1, skipped_existing: 0, not_document_request: 0, extraction_failed: 0, agent_failed: 0, drafts_created: 1 }
    }),
    /Fetched 1/
  );
});
