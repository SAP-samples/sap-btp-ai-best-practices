// SAP readiness helpers: quarter label, completion transition, gap lines and the after-run offer.
// Run with: cd ui && node --test tests/sap_readiness.test.js
import { test } from "node:test";
import assert from "node:assert/strict";
import { completedTransition, currentQuarterLabel, offerModel, readinessLines, sourceLine } from "../src/workspace/sap_readiness.js";

const READINESS = {
  available: true,
  profile: { origin: "live", name: "Plan", revision: 4, profile_id: "p1" },
  tabs: {
    cycle_stock: { status: "ready", message: "", gaps: [] },
    planning_calendar: { status: "blocked", message: "Set the odd- and even-week lot-size profiles", gaps: [{ code: "profile_setting", setting: "calendar_odd_profile, calendar_even_profile", fix: "Enter both codes." }] },
    recipe_combination: { status: "partial", message: "", gaps: [{ message: "few characters", code: "dataset_view", fix: "Upload inputs." }] },
  },
  suggestions: [{ status: "suggested", dataset_name: "Older", values: { calendar_odd_profile: "ODD-W1", calendar_even_profile: "EVEN-W2" } }],
  last_summary: null,
};
const DECISIONS = { knee_point_index: 5, selection: null, points: [{ point_index: 5, decision_label: "B" }] };

test("quarter labels follow the calendar quarter", () => {
  assert.equal(currentQuarterLabel(new Date(2026, 0, 1)), "26Q1");
  assert.equal(currentQuarterLabel(new Date(2026, 2, 31)), "26Q1");
  assert.equal(currentQuarterLabel(new Date(2026, 3, 1)), "26Q2");
  assert.equal(currentQuarterLabel(new Date(2026, 11, 31)), "26Q4");
});

test("only an observed active-to-completed change counts as just completed", () => {
  assert.equal(completedTransition(undefined, "succeeded"), false);
  assert.equal(completedTransition("running", "succeeded"), true);
  assert.equal(completedTransition("queued", "completed"), true);
  assert.equal(completedTransition("succeeded", "succeeded"), false);
  assert.equal(completedTransition("running", "failed"), false);
});

test("gap lines list blocked and partial tabs with their fix", () => {
  const lines = readinessLines(READINESS);
  assert.deepEqual(lines.map((line) => [line.label, line.status, line.code]), [
    ["Planning calendar", "blocked", "profile_setting"], ["Recipe combination", "partial", "dataset_view"]]);
  assert.equal(lines[1].text, "few characters");
  assert.deepEqual(readinessLines({ available: false }), []);
});

test("the offer names the balanced point without a selection and stays quiet when settings are current", () => {
  const offer = offerModel(DECISIONS, READINESS, null);
  assert.equal(offer.text, "Run completed. Generate SAP settings for scenario B?");
  assert.equal(offer.primaryLabel, "Select B and generate");
  assert.equal(offer.needsSelection, true);
  assert.equal(offer.pointIndex, 5);
  assert.match(offer.question, /blocked: Planning calendar/);
  assert.equal(offer.suggestion.values.calendar_odd_profile, "ODD-W1");
  const selected = offerModel({ ...DECISIONS, selection: { point_index: 7, decision_label: "CO1" } }, READINESS, null);
  assert.equal(selected.primaryLabel, "Generate SAP settings");
  assert.match(selected.text, /scenario CO1/);
  assert.equal(offerModel(DECISIONS, { ...READINESS, last_summary: { stale: false } }, { generated_at: "x" }), null);
  const stale = offerModel(DECISIONS, { ...READINESS, last_summary: { stale: true, stale_reason: "the plant profile changed" } }, { generated_at: "x" });
  assert.match(stale.text, /changed since the SAP settings were generated/);
  assert.equal(offerModel(DECISIONS, { available: false }, null), null);
});

test("the source line names the profile and flags stale settings", () => {
  assert.equal(sourceLine(READINESS), 'SAP settings rules from plant profile "Plan" revision 4.');
  assert.match(sourceLine({ ...READINESS, last_summary: { stale: true, stale_reason: "profile changed" } }), /out of date: profile changed/);
  assert.match(sourceLine({ ...READINESS, profile: { origin: "none" } }), /no plant profile/);
});
