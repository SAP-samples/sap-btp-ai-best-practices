import test from "node:test";
import assert from "node:assert/strict";

import {
  buildHistoryEndpoint,
  buildHistoryToolbarText,
  historyPageCount,
  historyPageRange
} from "../src/pages/past/pastHelpers.js";

test("historyPageCount returns at least one page", () => {
  assert.equal(historyPageCount(0, 50), 1);
  assert.equal(historyPageCount(73, 50), 2);
});

test("historyPageRange returns the visible row block", () => {
  assert.deepEqual(historyPageRange({ total: 73, limit: 50, offset: 0 }), {
    start: 1,
    end: 50
  });
  assert.deepEqual(historyPageRange({ total: 73, limit: 50, offset: 50 }), {
    start: 51,
    end: 73
  });
  assert.deepEqual(historyPageRange({ total: 0, limit: 50, offset: 0 }), {
    start: 0,
    end: 0
  });
});

test("buildHistoryToolbarText includes global counts and visible range", () => {
  assert.equal(
    buildHistoryToolbarText({ approved: 1, rejected: 72, total: 73, limit: 50, offset: 0 }),
    "1 approved, 72 rejected, 73 total past requests. Showing 1-50 of 73."
  );
});

test("buildHistoryEndpoint adds fixed pagination parameters", () => {
  assert.equal(
    buildHistoryEndpoint({ limit: 50, offset: 50 }),
    "/api/price-change-history?limit=50&offset=50"
  );
});
