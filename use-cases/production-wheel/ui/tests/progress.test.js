/** Run progress contracts: node --test ui/tests/progress.test.js */
import test from "node:test";
import assert from "node:assert/strict";
import { runProgress } from "../src/workspace/progress.js";

test("queued jobs do not imply solver execution and explain long waits", () => {
  const progress = runProgress({status:"queued", created_at:"2026-09-08T14:00:00Z"}, Date.parse("2026-09-08T14:16:00Z"));
  assert.equal(progress.fraction, null);
  assert.match(progress.text, /execution has not started/);
  assert.match(progress.text, /worker/);
  assert.match(progress.text, /960s/);
});

test("worker phase progress becomes a bounded fractional bar", () => {
  assert.deepEqual(runProgress({status:"running", stage:"blocks", progress:{current:23,total:46}}), {fraction:0.5,text:"blocks: 23 / 46 in this phase."});
  assert.equal(runProgress({status:"running",progress:{current:10,total:0}}).fraction,null);
});
