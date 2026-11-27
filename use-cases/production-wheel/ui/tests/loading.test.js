/** Page loading indicator contracts: node --test ui/tests/loading.test.js */
import test from "node:test";
import assert from "node:assert/strict";
import { createLoadTracker } from "../src/workspace/loading.js";

/**
 * Build a tracker with a manual clock and a log of show/hide calls.
 * Output: {tracker, events, fire} where fire() runs the pending delayed show, if any.
 */
function harness() {
  const events = [];
  let scheduled = null;
  const tracker = createLoadTracker({
    show: (label) => events.push(["show", label]),
    hide: () => events.push(["hide"]),
    delay: 250,
    setTimer: (callback) => (scheduled = callback),
    clearTimer: () => (scheduled = null),
  });
  const fire = () => {
    const callback = scheduled;
    scheduled = null;
    callback?.();
  };
  return { tracker, events, fire };
}

/** A promise plus its resolve/reject handles, so a test decides when a load ends. */
function deferred() {
  let resolve, reject;
  const promise = new Promise((ok, fail) => ((resolve = ok), (reject = fail)));
  return { promise, resolve, reject };
}

/** Let the tracker's promise callbacks run. */
const flush = () => new Promise((resolve) => setImmediate(resolve));

test("a load that ends before the delay never shows the indicator", async () => {
  const { tracker, events } = harness();
  tracker.begin();
  const load = deferred();
  tracker.track(load.promise, "Loading datasets…");
  load.resolve();
  await flush();
  assert.deepEqual(events, []);
  assert.equal(tracker.pending, 0);
});

test("a slow load shows its text and hides when it ends", async () => {
  const { tracker, events, fire } = harness();
  tracker.begin();
  const load = deferred();
  tracker.track(load.promise, "Loading plant profiles…");
  fire();
  assert.deepEqual(events, [["show", "Loading plant profiles…"]]);
  load.resolve();
  await flush();
  assert.deepEqual(events.at(-1), ["hide"]);
  assert.equal(tracker.visible, false);
});

test("the indicator stays until every load of the page has ended", async () => {
  const { tracker, events, fire } = harness();
  tracker.begin();
  const first = deferred();
  const second = deferred();
  tracker.track(first.promise, "First");
  tracker.track(second.promise, "Second");
  fire();
  first.resolve();
  await flush();
  assert.equal(tracker.visible, true);
  second.resolve();
  await flush();
  assert.equal(tracker.visible, false);
  assert.deepEqual(events, [["show", "Second"], ["hide"]]);
});

test("a failed load hides the indicator and still rejects for the caller", async () => {
  const { tracker, fire } = harness();
  tracker.begin();
  const load = deferred();
  const returned = tracker.track(load.promise);
  fire();
  load.reject(new Error("503: HANA unavailable"));
  await assert.rejects(returned, /HANA unavailable/);
  await flush();
  assert.equal(tracker.visible, false);
  assert.equal(tracker.pending, 0);
});

test("a load from the previous page does not hide the new page's indicator", async () => {
  const { tracker, fire } = harness();
  tracker.begin();
  const oldPage = deferred();
  tracker.track(oldPage.promise, "Old page");
  tracker.begin();
  const newPage = deferred();
  tracker.track(newPage.promise, "New page");
  fire();
  oldPage.resolve();
  await flush();
  assert.equal(tracker.visible, true);
  assert.equal(tracker.pending, 1);
  newPage.resolve();
  await flush();
  assert.equal(tracker.visible, false);
});

test("navigating away hides a visible indicator at once", () => {
  const { tracker, events, fire } = harness();
  tracker.begin();
  tracker.track(deferred().promise, "Never ends");
  fire();
  tracker.begin();
  assert.deepEqual(events, [["show", "Never ends"], ["hide"]]);
  assert.equal(tracker.pending, 0);
});
