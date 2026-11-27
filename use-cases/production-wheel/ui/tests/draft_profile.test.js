import test from "node:test";
import assert from "node:assert/strict";
import { profileFreshness, recreatedNote } from "../src/workspace/draft_profile.js";

test("outdated draft warning offers a recreate button only when a newer revision exists", () => {
  assert.equal(profileFreshness(null), null);
  assert.equal(profileFreshness({outdated: false, removed: false}), null);
  const outdated = profileFreshness({outdated: true, current_profile_revision: 5, message: "uses revision 3"});
  assert.deepEqual(outdated, {text: "uses revision 3", button: "Recreate draft with revision 5"});
  assert.deepEqual(profileFreshness({removed: true, outdated: false, message: "was removed"}), {text: "was removed", button: null});
});

test("recreate note reports kept and skipped scenario choices", () => {
  const note = recreatedNote({profile_status: {current_profile_revision: 5}, carried_over: ["request.config.pv_mode"], skipped: []});
  assert.equal(note, "New draft created with profile revision 5; 1 scenario choice kept. Validate it before launching.");
  const skipped = recreatedNote({profile_status: {current_profile_revision: 2}, carried_over: [],
    skipped: [{path: "request.constraints", reason: "unknown SEFI"}]});
  assert.match(skipped, /0 scenario choices kept.*Not kept: request\.constraints \(unknown SEFI\)\.$/);
});
