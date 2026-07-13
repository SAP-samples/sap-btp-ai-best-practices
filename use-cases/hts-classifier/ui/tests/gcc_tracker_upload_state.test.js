import test from "node:test";
import assert from "node:assert/strict";

import {
  buildGCCTrackerRefreshSuccessMessage,
  buildGCCTrackerSelectionText,
  isGCCTrackerWorkbookFile
} from "../src/pages/settings/gcc_tracker_upload_state.js";

test("isGCCTrackerWorkbookFile accepts supported Excel workbooks", () => {
  assert.equal(isGCCTrackerWorkbookFile({ name: "GCC Tracker.xlsb" }), true);
  assert.equal(isGCCTrackerWorkbookFile({ name: "gcc-tracker.XLSB" }), true);
  assert.equal(isGCCTrackerWorkbookFile({ name: "GCC Tracker.xlsx" }), true);
  assert.equal(isGCCTrackerWorkbookFile({ name: "gcc-tracker.XLSX" }), true);
  assert.equal(isGCCTrackerWorkbookFile({ name: "GCC Tracker.csv" }), false);
  assert.equal(isGCCTrackerWorkbookFile(null), false);
});

test("buildGCCTrackerSelectionText describes selected GCC tracker file", () => {
  assert.equal(buildGCCTrackerSelectionText(null), "No GCC Tracker .xlsb or .xlsx file selected.");
  assert.equal(
    buildGCCTrackerSelectionText({ name: "GCC Tracker.xlsb", size: 2048 }),
    "Selected file: GCC Tracker.xlsb (2 KB)."
  );
});

test("buildGCCTrackerRefreshSuccessMessage includes refresh and cleanup counts", () => {
  assert.equal(
    buildGCCTrackerRefreshSuccessMessage({
      uploaded_filename: "GCC Tracker.xlsb",
      source_row_count: 12,
      cleared_classification_count: 3,
      cancelled_job_count: 1
    }),
    "Uploaded GCC Tracker.xlsb and refreshed 12 GCC rows in HANA. Cleared 3 saved classification snapshots and cancelled 1 active classification job."
  );
});
