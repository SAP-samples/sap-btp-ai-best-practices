/** Plain-language extraction findings: node --test tests/findings.test.js */
import test from "node:test";
import assert from "node:assert/strict";
import { assistantQuestion, findingGroups, itemsLine, scopeNotes } from "../src/workspace/findings.js";

/** The findings of the primary-site workbook imported with its enrichment workbook. */
const ISSUES = [
  { rule_id: "SAP_LOOKUP_MISSING", severity: "warning", entity_key: "RecipeCharacters",
    reason: "recipe_characters: table not found in the inputs workbook" },
  { rule_id: "PALLET_NONBLANK_SOURCE_DISCREPANCY", severity: "warning", entity: "fini", observed: 33, expected: 0,
    entity_key: Array.from({ length: 10 }, (_, i) => `${5000 + i}/P1`).join("|") },
  ...Array.from({ length: 9 }, (_, i) => ({
    rule_id: "FINI_DATA_QUALITY", severity: "warning", entity_key: `${6000 + i}/P1`, reason: "missing_line_evidence" })),
];

test("findings are grouped by check and reason, in plain words", () => {
  const groups = findingGroups(ISSUES);
  assert.deepEqual(groups.map((group) => [group.rule_id, group.total]), [
    ["PALLET_NONBLANK_SOURCE_DISCREPANCY", 33], ["FINI_DATA_QUALITY", 9], ["SAP_LOOKUP_MISSING", 1]]);
  assert.equal(groups[1].title, "9 finished products excluded: no filling-line data");
  assert.equal(groups[0].title, "33 finished products with a different pallet size in the enrichment workbook");
  assert.equal(groups[2].title, "1 SAP lookup table missing from the enrichment workbook");
  assert.deepEqual(groups[1].link, { view: "fini_master", field: "exclusion_reason", value: "missing_line_evidence" });
  assert.equal(groups[2].link, null);
});

test("affected items list ten, then count the rest, including unlisted ones", () => {
  const [pallets, excluded] = findingGroups(ISSUES);
  assert.match(itemsLine(pallets), /^5000\/P1, .*5009\/P1 and 23 more$/);
  assert.match(itemsLine(excluded), /^6000\/P1, .*6008\/P1$/);
});

test("errors come first and unknown checks keep their own text", () => {
  const groups = findingGroups([
    ...ISSUES,
    { rule_id: "SOME_NEW_CHECK", severity: "error", entity_key: "x", observed: "missing", action: "Fix the source" },
    { rule_id: "FINI_DATA_QUALITY", severity: "warning", entity_key: "7000/P1", reason: "missing_pallet_data" },
    { rule_id: "SAP_LOOKUP_MISSING", severity: "warning", entity_key: "ABC_calculated", reason: "dc_abc_classes: table not found" },
  ]);
  assert.ok(groups.some((group) => group.title === "2 SAP lookup tables missing from the enrichment workbook"));
  assert.equal(groups[0].severity, "error");
  assert.equal(groups[0].title, "SOME_NEW_CHECK (1)");
  assert.equal(groups[0].meaning, "Observed: missing");
  assert.equal(groups[0].effect, "Fix the source");
  assert.ok(groups.some((group) => group.title === "1 finished product excluded: no pallet size"));
  assert.deepEqual(findingGroups(undefined), []);
});

test("frequency scope is a collapsed note, not a warning, and counts products", () => {
  const issues = [{ rule_id: "PRODUCTION_FREQUENCY_UNDETERMINED", severity: "warning",
    entity_key: "999.Not possible to determine", observed: "12 FINI rows" }];
  // The user already saw the 999 rows in the extraction settings: no warning repeats them.
  assert.deepEqual(findingGroups(issues), []);
  const metadata = {
    settings: { modeled_frequencies: ["01W", "02W"] },
    frequency_inventory: { codes: [{ code: "01W", rows: 238 }, { code: "02W", rows: 237 }, { code: "16W", rows: 691 }] },
  };
  const [notSelected, undetermined] = scopeNotes(issues, metadata);
  assert.equal(notSelected.title, "691 finished products with a production frequency not selected in the extraction settings");
  assert.match(notSelected.meaning, /modeled 01W, 02W; these products have 16W/);
  assert.deepEqual(notSelected.link, { view: "fini_master", field: "exclusion_reason", value: "production_frequency_16w_out_of_scope" });
  assert.equal(undetermined.title, '12 finished products with the production frequency "999.Not possible to determine"');
  assert.equal(undetermined.severity, "info");
  assert.equal(undetermined.products, 12);
  // Every workbook code modeled: only the 999 note remains.
  assert.equal(scopeNotes(issues, { ...metadata, settings: { modeled_frequencies: ["01W", "02W", "16W"] } }).length, 1);
  assert.deepEqual(scopeNotes(undefined, undefined), []);
});

test("products usable only with OPTIMIZED get their own warning", () => {
  const groups = findingGroups([
    ...["1000001", "1000002", "1000003"].map((material) => ({ rule_id: "FINI_OPTIMIZED_PV_ONLY", severity: "warning",
      entity_key: `${material}/P1`, reason: "missing_fixed_pv" })),
    ...Array.from({ length: 7 }, (_, i) => ({ rule_id: "FINI_DATA_QUALITY", severity: "warning",
      entity_key: `${8000 + i}/P1`, reason: "missing_fixed_pv", observed: "excluded with production version FIXED and OPTIMIZED" })),
  ]);
  assert.deepEqual(groups.map((group) => group.title), [
    "7 finished products excluded: no selected production version",
    "3 finished products only usable with the OPTIMIZED production version",
  ]);
  assert.match(groups[0].meaning, /"Selected PV" cell of these products is empty/);
  assert.match(groups[0].effect, /left out of every optimization run, with both production version settings/);
  assert.match(groups[1].effect, /FIXED \(the default\) these products are left out. With OPTIMIZED they are included/);
});

test("older PV findings do not claim OPTIMIZED excludes them too", () => {
  // Imports before the OPTIMIZED split stored no `observed` text on these findings.
  const [legacy] = findingGroups([{ rule_id: "FINI_DATA_QUALITY", severity: "warning", entity_key: "1/P1", reason: "missing_fixed_pv" }]);
  assert.match(legacy.effect, /OPTIMIZED can include those whose SEFI has a production version/);
  const [current] = findingGroups([{ rule_id: "FINI_DATA_QUALITY", severity: "warning", entity_key: "1/P1",
    reason: "missing_fixed_pv", observed: "excluded with production version FIXED and OPTIMIZED" }]);
  assert.match(current.effect, /both production version settings/);
});

test("each exclusion reason names the workbook column to fix", () => {
  const [group] = findingGroups([{ rule_id: "FINI_DATA_QUALITY", severity: "warning", entity_key: "1/P1", reason: "missing_demand" }]);
  assert.match(group.meaning, /"Total FINI forecast \(litres\)" cell/);
  assert.match(group.effect, /Fill in the forecast/);
});

test("the assistant question names the check and reason", () => {
  const [group] = findingGroups([{ rule_id: "FINI_DATA_QUALITY", severity: "warning", entity_key: "1/P1", reason: "missing_fixed_pv" }]);
  const question = assistantQuestion(group);
  assert.match(question, /"1 finished product excluded: no selected production version"/);
  assert.match(question, /check FINI_DATA_QUALITY, reason missing_fixed_pv/);
});

test("per-key duplicate FINI issues group cleanly with accurate count", () => {
  const duplicateIssues = Array.from({ length: 15 }, (_, i) => ({
    rule_id: "FINI_KEY_UNIQUE",
    severity: "error",
    entity: "fini",
    entity_key: `${9000 + i}/PL01`,
    observed: "2 rows",
    expected: "1 row",
    action: "Resolve duplicate FINI/plant identities",
  }));
  const [group] = findingGroups(duplicateIssues);
  assert.equal(group.rule_id, "FINI_KEY_UNIQUE");
  assert.equal(group.severity, "error");
  assert.equal(group.total, 15);
  assert.equal(group.title, "15 finished products appear more than once");
  assert.match(itemsLine(group), /^9000\/PL01, .*9009\/PL01 and 5 more$/);
});
