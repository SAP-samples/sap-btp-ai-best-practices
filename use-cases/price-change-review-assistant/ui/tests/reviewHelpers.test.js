import test from "node:test";
import assert from "node:assert/strict";

import {
  buildProcessingProgressView,
  buildManualEmailPayload,
  buildManualEmailFormData,
  buildDraftPatchPayload,
  createProcessingRunId,
  attachmentDownloadEndpoint,
  formatApiErrorMessage,
  formatAttachmentSize,
  formatEmailSourceLabel,
  formatValidationError,
  getDraftDecisionActions,
  manualEmailHasAttachments,
  mapDraftFromApi,
  summarizeFetchResult
} from "../src/pages/home/reviewHelpers.js";

test("mapDraftFromApi converts live draft payload to UI row", () => {
  const row = mapDraftFromApi({
    draft_id: "draft-1",
    gmail_message_id: "m1",
    sender_email: "supplier@example.com",
    subject: "Price update",
    email_date: "2026-04-27T10:00:00",
    body: "Email body",
    supplier_id: "SUP001",
    supplier_name: "Tech Solutions",
    material_code: "8510031",
    material_description: "Apex Carbon Road Bike",
    original_price: "2150.00",
    requested_new_price: "2450.00",
    currency: "EUR",
    uom: "EA",
    effective_from: "2026-05-01",
    status: "ready_for_review",
    confidence: 0.91,
    explanation: "Resolved",
    candidate_materials_json: "[]",
    candidate_suppliers_json: "[]",
    validation_errors_json: "[]"
  });

  assert.equal(row.rowId, "draft-1");
  assert.equal(row.materialCode, "8510031");
  assert.equal(row.requestedNewPrice, "2450.00");
  assert.equal(row.originalEmail.date, "2026-04-27T10:00:00Z");
  assert.equal(row.originalEmail.body, "Email body");
  assert.deepEqual(row.attachments, []);
});

test("mapDraftFromApi exposes attachment metadata without content", () => {
  const row = mapDraftFromApi({
    draft_id: "draft-1",
    gmail_message_id: "m1",
    validation_errors_json: "[]",
    candidate_materials_json: "[]",
    candidate_suppliers_json: "[]",
    attachments_json: JSON.stringify([
      {
        attachment_id: "att-1",
        filename: "request.pdf",
        mime_type: "application/pdf",
        file_extension: "pdf",
        size_bytes: 1024
      }
    ])
  });

  assert.deepEqual(row.attachments, [
    {
      attachment_id: "att-1",
      filename: "request.pdf",
      mime_type: "application/pdf",
      file_extension: "pdf",
      size_bytes: 1024
    }
  ]);
});

test("buildDraftPatchPayload keeps editable fields only", () => {
  const payload = buildDraftPatchPayload({
    supplierId: "SUP001",
    supplierName: "Tech Solutions",
    supplierEmail: "supplier@example.com",
    materialCode: "8510031",
    materialDescription: "Apex Carbon Road Bike",
    originalPrice: "2150.00",
    requestedNewPrice: "2450.00",
    currency: "EUR",
    uom: "EA",
    effectiveFrom: "2026-05-01",
    effectiveTo: "",
    explanation: "Resolved"
  });

  assert.deepEqual(payload, {
    supplier_id: "SUP001",
    supplier_name: "Tech Solutions",
    supplier_email: "supplier@example.com",
    material_number: "8510031",
    material_description: "Apex Carbon Road Bike",
    original_price: "2150.00",
    requested_new_price: "2450.00",
    currency: "EUR",
    uom: "EA",
    effective_from: "2026-05-01",
    effective_to: null,
    explanation: "Resolved"
  });
});

test("formatValidationError renders user-facing field labels", () => {
  assert.equal(formatValidationError("effective_from is required"), "Effective from is required");
  assert.equal(formatValidationError("supplier_id is required"), "Supplier ID is required");
  assert.equal(formatValidationError("Custom validation message"), "Custom validation message");
});

test("getDraftStatusLabel renders review status labels for reviewers", async () => {
  const helpers = await import("../src/pages/home/reviewHelpers.js");

  assert.equal(helpers.getDraftStatusLabel("ready_for_review"), "Ready to Approve");
  assert.equal(helpers.getDraftStatusLabel("needs_human_review"), "Needs Review");
});

test("formatApiErrorMessage prefers structured backend detail messages", () => {
  const error = new Error("HTTP error! status: 424");
  error.detail = {
    message: "S/4 price update could not be applied: SAP_COM_0294 required.",
    s4_status: "service_unavailable"
  };

  assert.equal(
    formatApiErrorMessage(error),
    "S/4 price update could not be applied: SAP_COM_0294 required."
  );
});

test("summarizeFetchResult renders fetch counters", () => {
  const text = summarizeFetchResult({
    fetched: 2,
    skipped_existing: 3,
    not_price_request: 1,
    extraction_failed: 0,
    agent_failed: 0,
    drafts_created: 2
  });

  assert.equal(text, "Fetched 2, skipped 3, non-price 1, extraction failures 0, agent failures 0, drafts 2");
});

test("buildManualEmailPayload trims required manual email fields", () => {
  const payload = buildManualEmailPayload(
    {
      senderEmail: " supplier@example.com ",
      subject: " Price update ",
      body: " Please change material 8510031. "
    },
    "run-manual-1"
  );

  assert.deepEqual(payload, {
    sender_email: "supplier@example.com",
    subject: "Price update",
    body: "Please change material 8510031.",
    processing_run_id: "run-manual-1"
  });
});

test("buildManualEmailFormData keeps fields and repeated attachments", async () => {
  const file = new Blob(["sku,new_price\nA1,12.50\n"], { type: "text/csv" });
  const formData = buildManualEmailFormData(
    {
      senderEmail: " supplier@example.com ",
      subject: " Price update ",
      body: " See attached. "
    },
    [{ file, name: "prices.csv" }],
    "run-upload-1"
  );

  assert.equal(formData.get("sender_email"), "supplier@example.com");
  assert.equal(formData.get("subject"), "Price update");
  assert.equal(formData.get("body"), "See attached.");
  assert.equal(formData.get("processing_run_id"), "run-upload-1");
  assert.equal(formData.getAll("attachments").length, 1);
});

test("createProcessingRunId returns browser-safe unique ids", () => {
  const first = createProcessingRunId("manual");
  const second = createProcessingRunId("manual");

  assert.match(first, /^manual-/);
  assert.match(second, /^manual-/);
  assert.notEqual(first, second);
});

test("buildProcessingProgressView formats current and stale progress", () => {
  const fresh = buildProcessingProgressView(
    {
      run: {
        status: "running",
        current_message: "Reading current supplier price",
        last_heartbeat_at: "2026-06-05T08:00:30Z",
        started_at: "2026-06-05T08:00:00Z"
      },
      events: [
        { sequence_number: 1, message: "Classifying email", event_time: "2026-06-05T08:00:05Z" },
        { sequence_number: 2, message: "Reading current supplier price", event_time: "2026-06-05T08:00:30Z" }
      ]
    },
    Date.parse("2026-06-05T08:00:40Z")
  );

  assert.equal(fresh.currentMessage, "Reading current supplier price");
  assert.equal(fresh.helperText, "Last update 10s ago");
  assert.equal(fresh.isStale, false);
  assert.deepEqual(fresh.recentEvents.map((event) => event.message), [
    "Classifying email",
    "Reading current supplier price"
  ]);

  const stale = buildProcessingProgressView(
    {
      run: {
        status: "running",
        current_message: "Reading current supplier price",
        last_heartbeat_at: "2026-06-05T08:00:00Z",
        started_at: "2026-06-05T07:59:00Z"
      },
      events: []
    },
    Date.parse("2026-06-05T08:00:50Z")
  );

  assert.equal(stale.isStale, true);
  assert.equal(stale.helperText, "Still waiting on the model or S/4. Last update 50s ago.");
});

test("buildProcessingProgressView treats timezone-less HANA timestamps as UTC", () => {
  const view = buildProcessingProgressView(
    {
      run: {
        status: "running",
        current_message: "Preparing analysis",
        last_heartbeat_at: "2026-06-05 08:00:30",
        started_at: "2026-06-05 08:00:00"
      },
      events: []
    },
    Date.parse("2026-06-05T08:00:40Z")
  );

  assert.equal(view.helperText, "Last update 10s ago");
  assert.equal(view.isStale, false);
});

test("manualEmailHasAttachments detects selected files", () => {
  assert.equal(manualEmailHasAttachments([]), false);
  assert.equal(manualEmailHasAttachments([{ name: "prices.csv" }]), true);
});

test("formatAttachmentSize and download endpoint are audit friendly", () => {
  assert.equal(formatAttachmentSize(1024), "1 KB");
  assert.equal(formatAttachmentSize(1536), "1.5 KB");
  assert.equal(attachmentDownloadEndpoint("att 1"), "/api/email-attachments/att%201/download");
});

test("buildManualEmailPayload rejects blank manual email fields", () => {
  assert.throws(
    () => buildManualEmailPayload({ senderEmail: "supplier@example.com", subject: " ", body: "Body" }),
    /Sender email, subject, and body are required/
  );
});

test("formatEmailSourceLabel separates manual and Gmail messages", () => {
  assert.equal(formatEmailSourceLabel("manual-abc123"), "Manual entry manual-abc123");
  assert.equal(formatEmailSourceLabel("gmail-abc123"), "Gmail gmail-abc123");
  assert.equal(formatEmailSourceLabel(""), "Source not set");
});

test("getDraftDecisionActions marks reject red and separates approve state", () => {
  const actions = getDraftDecisionActions("ready_for_review");

  assert.deepEqual(actions, [
    {
      key: "reject",
      label: "Reject",
      design: "Negative",
      className: "reject-draft-button",
      disabled: false
    },
    {
      key: "approve",
      label: "Accept",
      design: "Positive",
      className: "approve-draft-button",
      disabled: false
    }
  ]);
});

test("draft explanation fields use a growing multiline editor", async () => {
  const helpers = await import("../src/pages/home/reviewHelpers.js");

  assert.equal(typeof helpers.getDraftFieldEditorConfig, "function");
  assert.deepEqual(helpers.getDraftFieldEditorConfig("explanation"), {
    component: "textarea",
    growing: true,
    rows: 4
  });
  assert.deepEqual(helpers.getDraftFieldEditorConfig("supplier-id"), {
    component: "input"
  });
});

test("draft editor value reader uses the native textarea value for explanations", async () => {
  const helpers = await import("../src/pages/home/reviewHelpers.js");

  assert.equal(typeof helpers.readDraftEditorValue, "function");
  assert.equal(
    helpers.readDraftEditorValue({
      tagName: "UI5-TEXTAREA",
      value: "",
      shadowRoot: {
        querySelector: () => ({ value: "Resolved supplier from S/4." })
      }
    }),
    "Resolved supplier from S/4."
  );
  assert.equal(
    helpers.readDraftEditorValue({
      tagName: "UI5-TEXTAREA",
      value: "Original value",
      shadowRoot: {
        querySelector: () => ({ value: "" })
      }
    }),
    ""
  );
  assert.equal(
    helpers.readDraftEditorValue({
      tagName: "UI5-INPUT",
      value: "SUP003"
    }),
    "SUP003"
  );
});

test("getDraftDecisionActions disables accept until a draft is ready", () => {
  const actions = getDraftDecisionActions("needs_human_review");

  assert.equal(actions.find((action) => action.key === "reject")?.disabled, false);
  assert.equal(actions.find((action) => action.key === "approve")?.disabled, true);
});
