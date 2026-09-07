import assert from "node:assert/strict";
import test from "node:test";

import { RequestGeneration } from "../src/services/requestGeneration.js";

/**
 * Create a manually resolved promise for deterministic request ordering tests.
 *
 * @returns {{promise: Promise<unknown>, resolve: (value: unknown) => void, reject: (error: Error) => void}} Deferred value.
 */
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolver, rejecter) => {
    resolve = resolver;
    reject = rejecter;
  });
  return { promise, resolve, reject };
}

test("a profile GET started before a PUT cannot replace the persisted result", async () => {
  const owner = new RequestGeneration();
  const slowGet = deferred();
  let profile = { customer_class: "class_3", nace1: "Mining" };
  const loadToken = owner.begin();
  const load = slowGet.promise.then((result) => {
    if (owner.owns(loadToken)) {
      profile = result;
    }
  });

  owner.invalidate();
  profile = { customer_class: "class_3", nace1: "Electricity and gas" };
  slowGet.resolve({ customer_class: "class_2", nace1: "Manufacturing" });
  await load;

  assert.deepEqual(profile, {
    customer_class: "class_3",
    nace1: "Electricity and gas"
  });
});

test("only the latest score load renders and owns chart canvases", async () => {
  const owner = new RequestGeneration();
  const slow = deferred();
  const staleFailure = deferred();
  const fast = deferred();
  const rendered = [];
  const errors = [];
  const chartOwners = new Map([["overview", "previous"]]);
  let destroyedCharts = 0;
  let createdCharts = 0;

  const runLoad = async (response, label) => {
    const token = owner.begin();
    try {
      const payload = await response.promise;
      if (!owner.owns(token)) {
        return;
      }
      if (chartOwners.has("overview")) {
        destroyedCharts += 1;
        chartOwners.delete("overview");
      }
      createdCharts += 1;
      chartOwners.set("overview", label);
      rendered.push(payload);
    } catch (error) {
      if (owner.owns(token)) {
        errors.push(error.message);
      }
    }
  };

  const slowLoad = runLoad(slow, "slow");
  const failedLoad = runLoad(staleFailure, "stale-failure");
  const fastLoad = runLoad(fast, "fast");
  fast.resolve({ language: "it" });
  await fastLoad;
  slow.resolve({ language: "en" });
  staleFailure.reject(new Error("old English request failed"));
  await slowLoad;
  await failedLoad;

  assert.deepEqual(rendered, [{ language: "it" }]);
  assert.deepEqual(errors, []);
  assert.equal(destroyedCharts, 1);
  assert.equal(createdCharts, 1);
  assert.deepEqual([...chartOwners.entries()], [["overview", "fast"]]);
});
