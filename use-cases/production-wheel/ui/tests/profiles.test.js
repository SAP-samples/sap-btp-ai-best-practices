import test from "node:test";
import assert from "node:assert/strict";
import { cycleStockDefinition, profileSummary, runnerDefinition } from "../src/workspace/settings.js";
import { datasetPlants, mergeDraft, persistSelection, restoreSelection } from "../src/workspace/state.js";

test("profile identity persists and draft edits preserve fixed profile controls", () => {
  let saved;
  const storage = {setItem(_key, value) {saved = value;}, getItem() {return saved;}};
  persistSelection(storage, {plant_profile_id: "profile-1", settings: {private: true}});
  assert.deepEqual(restoreSelection(storage), {plant_profile_id: "profile-1"});
  const draft = {revision: 2, plant_profile_id: "profile-1", request: {config: {matrix_mode: "HARD", group_size: {base_limit: 4}}}};
  const result = mergeDraft(draft, {title: " Scenario ", matrix: "OFF", cap: "100", scope: "[]", constraints: "[]", budget: "{}"});
  assert.equal(result.title, "Scenario");
  // The profile mode is only a default: the scenario selection is sent.
  assert.equal(result.request.config.matrix_mode, "OFF");
  assert.equal(result.request.config.group_size.base_limit, 4);
  const thresholds = {high_max_days: 10, low_min_days: 80, high_max_ratio: 3, low_min_ratio: 10};
  const summary = profileSummary({name: "Profile", plant: "1", revision: 2, settings: {runner_thresholds: thresholds}});
  assert.match(summary, /High runner: one full batch lasts 10 days or less, or it lasts at most 3 times/);
  assert.match(summary, /Low runner: one batch lasts 80 days or more and at least 10 times/);
  assert.match(summary, /Medium runner: everything in between/);
  assert.match(runnerDefinition({high_max_days: ""}), /lasts \? days or less/);
});

test("cycle stock sentence explains each mode with the profile numbers", () => {
  const policy = {mode: "single", high_min_days: 0, medium_min_days: 20, low_min_days: 40, medium_batch_factor: 0.5, max_days: 200, mixed_groups_only: true};
  assert.match(cycleStockDefinition(policy), /Single RoC\. Every product of a subgroup is filled in every batch.*capped at 200 days/);
  const multiples = cycleStockDefinition({...policy, mode: "runner_multiple", low_min_days: 45});
  assert.match(multiples, /Runner multiples\. In every subgroup, the fastest product/);
  assert.match(multiples, /at least 0 days for a high runner, 20 for a medium runner and 45 for a low runner/);
  assert.match(cycleStockDefinition({...policy, mode: "optimizer_choice"}), /Optimizer chooses\. For subgroups that mix runner classes/);
  assert.match(cycleStockDefinition({...policy, mode: "optimizer_choice", mixed_groups_only: false}), /For every subgroup/);
  assert.match(cycleStockDefinition({}), /Single RoC.*capped at \? days/);
  assert.match(profileSummary({name: "P", plant: "1", revision: 1, settings: {cycle_stock_policy: policy}}), /Cycle stock: Single RoC/);
});

test("dataset plants drive profile routing and the pending plant is never persisted", () => {
  assert.deepEqual(datasetPlants({metadata: {plant: "PL02"}}), ["PL02"]);
  assert.deepEqual(datasetPlants({plants: ["PL01", 7]}), ["PL01", "7"]);
  assert.deepEqual(datasetPlants({metadata: {}}), []);
  assert.deepEqual(datasetPlants(null), []);
  let saved;
  const storage = {setItem(_key, value) {saved = value;}, getItem() {return saved;}};
  persistSelection(storage, {dataset_id: "d1", pending_profile_plant: "PL02"});
  assert.deepEqual(restoreSelection(storage), {dataset_id: "d1"});
});

test("proposal helpers describe changes and target the revision-checked changes endpoint", async () => {
  const { changeRequest, decisionNote, describeProposal, ruleSummary } = await import("../src/workspace/proposals.js");
  const rule = {kind: "group_rule", constraint_id: "line-cap", scope: {plant: "PL01"}, description: "Line 6 groups have at most two FINIs.", source_text: "max 2 FINIs on line 6"};
  assert.deepEqual(ruleSummary(rule), {title: "Line 6 groups have at most two FINIs.", meta: "line-cap · Rule for every group · plant PL01", source: "max 2 FINIs on line 6"});
  const proposal = {profile_id: "p/1", profile_name: "Plant", profile_revision: 4, summary: "Cap line 6", change: {action: "add_constraint", constraint: rule}};
  assert.match(describeProposal(proposal).heading, /Add constraint · Plant \(revision 4\)/);
  assert.ok(describeProposal(proposal).lines.includes("Planner wording: max 2 FINIs on line 6"));
  const matrix = describeProposal({...proposal, change: {action: "update_settings", settings: {maximum_group_size: 3}, matrix_rows: [{volume_a: "1", volume_b: "1"}, {volume_a: "1", volume_b: "2"}, {volume_a: "2", volume_b: "1"}, {volume_a: "2", volume_b: "2"}]}});
  assert.deepEqual(matrix.lines, ["maximum_group_size: 3", "Replace volume compatibility: 2 volumes, 4 pairs"]);
  const summarized = describeProposal({...proposal, matrix_summary: {added_volumes: ["2"], removed_volumes: [], pairs_by_status: {N: 1}, changed_pair_count: 0},
    change: {action: "update_settings", matrix_rows: [{volume_a: "1", volume_b: "1"}, {volume_a: "1", volume_b: "2"}, {volume_a: "2", volume_b: "1"}, {volume_a: "2", volume_b: "2"}]}});
  assert.deepEqual(summarized.lines.slice(1), ["Added volumes (L): 2", "Volume pairs by status: N 1", "Existing pairs changed: 0"]);
  assert.deepEqual(changeRequest("p/1", 4, proposal.change), {path: "/api/plant-profiles/p%2F1/changes", body: {revision: 4, change: proposal.change}});
  assert.match(decisionNote(proposal, true, 5), /^\[Approved in UI\] add_constraint line-cap .* revision 5\.$/);
  assert.match(decisionNote(proposal, false), /^\[Rejected in UI\]/);
});

test("compatibility modes show readable labels while keeping stored values", async () => {
  const { MATRIX_MODE_LABELS, matrixModeLabel, matrixModeOptions } = await import("../src/workspace/vocabulary.js");
  const { describeProposal } = await import("../src/workspace/proposals.js");
  assert.deepEqual(Object.keys(MATRIX_MODE_LABELS).sort(), ["DIAGNOSTIC", "FLEXIBLE", "HARD", "OFF"]);
  assert.equal(matrixModeLabel("DIAGNOSTIC"), "Prefer compatible");
  assert.equal(matrixModeLabel("UNKNOWN"), "UNKNOWN");
  // Select values stay API identifiers, so saved drafts and profiles are unaffected.
  assert.match(matrixModeOptions(), /<option value="DIAGNOSTIC">Prefer compatible<\/option>/);
  const card = describeProposal({ profile_name: "P", profile_revision: 1, change: { action: "update_settings", settings: { matrix_mode: "DIAGNOSTIC" } } });
  assert.ok(card.lines.includes("Compatibility mode: Prefer compatible"));
});

test("volume matching labels, class label order and reclassification lines on the card", async () => {
  const { volumeMatchingLabel, volumeMatchingOptions, compareVolumeLabels } = await import("../src/workspace/vocabulary.js");
  const { describeProposal } = await import("../src/workspace/proposals.js");
  assert.equal(volumeMatchingLabel("nearest_class"), "Nearest reference volume");
  assert.equal(volumeMatchingLabel(undefined), "Exact volumes");
  assert.match(volumeMatchingOptions(), /<option value="nearest_class">/);
  assert.deepEqual([">15", "2.5", "1", "<1", "15", "10"].sort(compareVolumeLabels), ["<1", "1", "2.5", "10", "15", ">15"]);
  const card = describeProposal({
    profile_name: "P", profile_revision: 2,
    change: { action: "update_settings", settings: { volume_matching: "nearest_class" }, matrix_rows: [{ volume_a: "2", volume_b: "2", status: "Y" }] },
    matrix_summary: { added_volumes: ["2"], removed_volumes: [], pairs_by_status: { Y: 1 }, changed_pair_count: 0,
      reclassified_volumes: [{ volume: "2.2", product_count: 3, from: "2.5", to: "2" }], unclassified_volumes: ["20"] },
  });
  assert.ok(card.lines.includes("Volume matching: Nearest reference volume"));
  assert.ok(!card.lines.includes("volume_matching: nearest_class"));
  assert.ok(card.lines.includes("2.2 L (3 products): class 2.5 → 2"));
  assert.ok(card.lines.some((line) => line.includes("Volumes without a class") && line.includes("20")));
});
