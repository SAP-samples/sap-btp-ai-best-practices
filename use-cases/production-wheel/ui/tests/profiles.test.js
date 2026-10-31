import test from "node:test";
import assert from "node:assert/strict";
import { profileSummary } from "../src/workspace/settings.js";
import { datasetPlants, mergeDraft, persistSelection, restoreSelection } from "../src/workspace/state.js";

test("profile identity persists and draft edits preserve fixed profile controls", () => {
  let saved;
  const storage = {setItem(_key, value) {saved = value;}, getItem() {return saved;}};
  persistSelection(storage, {plant_profile_id: "profile-1", settings: {private: true}});
  assert.deepEqual(restoreSelection(storage), {plant_profile_id: "profile-1"});
  const draft = {revision: 2, plant_profile_id: "profile-1", request: {config: {matrix_mode: "HARD", group_size: {base_limit: 4}}}};
  const result = mergeDraft(draft, {title: " Scenario ", matrix: "OFF", cap: "100", scope: "[]", constraints: "[]", budget: "{}"});
  assert.equal(result.title, "Scenario");
  assert.equal(result.request.config.matrix_mode, "HARD");
  assert.equal(result.request.config.group_size.base_limit, 4);
  assert.match(profileSummary({name: "Profile", plant: "1", revision: 2, settings: {high_runner_threshold_days: 15}}), /High ≤ 15 days; low > 15 days/);
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
