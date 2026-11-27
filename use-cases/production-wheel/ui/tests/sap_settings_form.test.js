// SAP settings section of the plant profile form: read/write round trip, checks, card lines.
// Run with: cd ui && node --test tests/sap_settings_form.test.js
import { test } from "node:test";
import assert from "node:assert/strict";

/** In-memory stand-in for form controls, keyed by id. */
function fakeForm() {
  const controls = new Map();
  return (id) => {
    if (!controls.has(id)) controls.set(id, { value: "", checked: false });
    return controls.get(id);
  };
}

const STORED = {
  dc_min_lot: { all_classes: { rule: "pallet", number: null }, A: null, B: null, C: { rule: "litres", number: 300 } },
  dc_rounding: { all_classes: { rule: "layer_at_max_days", number: null }, A: { rule: "pallet", number: null }, B: null, C: null },
  abc_cycle_days: { A: 5, B: 10, C: 20, unclassified: null },
  calendar_frequency_code: "02W", calendar_odd_profile: "ODD-W1", calendar_even_profile: "EVEN-W2",
  calendar_lot_size: "PK", calendar_reset_lot_size: "EX", calendar_reset_text: "Reset via ticket - to be blank",
  recipe_characters: null, kpi_line_breakdown: false, apo_default_pv: "PV - 1", multi_tier_dc: true,
};

test("the form reads back exactly what was written", async () => {
  const { readSapForm, writeSapForm, sapFormMarkup } = await import("../src/workspace/sap_settings_form.js");
  const field = fakeForm();
  writeSapForm(field, STORED);
  assert.deepEqual(readSapForm(field), STORED);
  // Every control the accessors use exists in the markup.
  const markup = sapFormMarkup();
  for (const id of ["sap_calendar_odd_profile", "sap_min_lot_C", "sap_min_lot_litres_C", "sap_rounding_all_classes", "sap_cycle_days_unclassified", "sap-detect", "sap-settings", "sap-dc-rules"])
    assert.match(markup, new RegExp(`id="${id}"`));
});

test("same as all classes and blank codes become null; litres keep their amount", async () => {
  const { readSapForm, writeSapForm } = await import("../src/workspace/sap_settings_form.js");
  const field = fakeForm();
  writeSapForm(field, STORED);
  field("sap_min_lot_C").value = "";
  field("sap_calendar_even_profile").value = "  ";
  field("sap_min_lot_B").value = "litres";
  field("sap_min_lot_litres_B").value = "120";
  const read = readSapForm(field);
  assert.equal(read.dc_min_lot.C, null);
  assert.equal(read.calendar_even_profile, null);
  assert.deepEqual(read.dc_min_lot.B, { rule: "litres", number: 120 });
});

test("form checks explain what to fix before saving", async () => {
  const { sapFormProblems } = await import("../src/workspace/sap_settings_form.js");
  assert.deepEqual(sapFormProblems(STORED), []);
  const problems = sapFormProblems({ ...STORED, calendar_odd_profile: "NODASH", calendar_lot_size: "",
    dc_min_lot: { ...STORED.dc_min_lot, A: { rule: "litres", number: null } }, abc_cycle_days: { A: 0, B: 10, C: 20 } });
  assert.equal(problems.length, 4);
  assert.match(problems[0], /<profile>-<calendar>/);
  assert.ok(sapFormProblems({ ...STORED, calendar_even_profile: "ODD-W1" }).some((p) => /must differ/.test(p)));
});

test("approval cards describe SAP and nested settings in words", async () => {
  const { describeProposal } = await import("../src/workspace/proposals.js");
  const card = describeProposal({ profile_name: "P", profile_revision: 3, change: { action: "update_settings", settings: {
    sap_settings: { calendar_odd_profile: "ODD-W1", dc_min_lot: { C: { rule: "litres", number: 300 }, A: null }, multi_tier_dc: false },
    runner_thresholds: { high_max_days: 7 } } } });
  assert.deepEqual(card.lines, [
    "SAP settings · Odd-week lot-size profile: ODD-W1",
    "SAP settings · Minimum lot at DC (Class C): Litres: 300",
    "SAP settings · Minimum lot at DC (Class A): same as all classes",
    "SAP settings · Plant ships through an intermediate DC: no",
    "runner_thresholds.high_max_days: 7",
  ]);
});
