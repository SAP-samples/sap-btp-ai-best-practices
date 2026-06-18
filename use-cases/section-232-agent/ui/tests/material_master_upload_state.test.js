import test from "node:test";
import assert from "node:assert/strict";

import {
  buildMaterialMasterRefreshSuccessMessage,
  buildMaterialMasterSelectionText,
  isMaterialMasterWorkbookFile
} from "../src/pages/settings/material_master_upload_state.js";

test("isMaterialMasterWorkbookFile accepts supported Excel workbooks", () => {
  assert.equal(isMaterialMasterWorkbookFile({ name: "Material Master.xlsb" }), true);
  assert.equal(isMaterialMasterWorkbookFile({ name: "material-master.XLSB" }), true);
  assert.equal(isMaterialMasterWorkbookFile({ name: "Material Master.xlsx" }), true);
  assert.equal(isMaterialMasterWorkbookFile({ name: "material-master.XLSX" }), true);
  assert.equal(isMaterialMasterWorkbookFile({ name: "Material Master.csv" }), false);
  assert.equal(isMaterialMasterWorkbookFile(null), false);
});

test("buildMaterialMasterSelectionText describes selected Material Master file", () => {
  assert.equal(buildMaterialMasterSelectionText(null), "No Material Master .xlsb or .xlsx file selected.");
  assert.equal(
    buildMaterialMasterSelectionText({ name: "Material Master.xlsb", size: 2048 }),
    "Selected file: Material Master.xlsb (2 KB)."
  );
});

test("buildMaterialMasterRefreshSuccessMessage includes refresh and cleanup counts", () => {
  assert.equal(
    buildMaterialMasterRefreshSuccessMessage({
      uploaded_filename: "Material Master.xlsb",
      source_row_count: 12,
      cleared_classification_count: 3,
      cancelled_job_count: 1
    }),
    "Uploaded Material Master.xlsb and refreshed 12 Material Master rows in HANA. Cleared 3 saved classification snapshots and cancelled 1 active classification job."
  );
});
