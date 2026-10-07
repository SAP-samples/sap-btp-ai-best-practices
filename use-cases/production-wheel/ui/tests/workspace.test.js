/** Run focused browser-state and streaming adapter tests: node --test ui/tests/*.test.js */
import test from "node:test";
import assert from "node:assert/strict";
import {
  restoreSelection,
  persistSelection,
  mergeDraft,
  isTerminal,
} from "../src/workspace/state.js";
import { decodeNDJSON, deleteEntity } from "../src/services/api.js";
import {
  applyBrowserContextSummary,
  clearBrowserContext,
  formatChatMarkdown,
} from "../src/workspace/chat.js";

/** Minimal storage fixture with the same string contract as localStorage. */
function storage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key),
    setItem: (key, value) => values.set(key, value),
  };
}

test("only identifiers are persisted, never dataset tables or draft requests", () => {
  const fixture = storage();
  persistSelection(fixture, {
    dataset_id: "dataset-a",
    point_index: 0,
    request: { secret: true },
    rows: [{ fini: 123 }],
    context_id: "page-session-must-not-survive",
  });
  assert.deepEqual(restoreSelection(fixture), {
    dataset_id: "dataset-a",
    point_index: 0,
  });
});

test("chat markdown renders tables and emphasis while neutralizing active HTML", () => {
  const formatted = formatChatMarkdown(
    "| Field | Value |\n|---|---|\n| Coverage | **12 days** |\n\n<img src=x onerror=alert(1)>",
  );
  assert.match(formatted, /<table>/);
  assert.match(formatted, /<strong>12 days<\/strong>/);
  assert.doesNotMatch(formatted, /<img/i);
  assert.doesNotMatch(formatted, /href="javascript:/i);
  assert.match(formatted, /&lt;img/);
});

test("browser-only chat compaction retains recent turns and reset clears its summary", () => {
  const history = [
    { role: "user", content: "old" },
    { role: "assistant", content: "old reply" },
    { role: "user", content: "recent" },
    { role: "assistant", content: "recent reply" },
    { role: "user", content: "current" },
  ];

  let summary = applyBrowserContextSummary(history, "older compact context");

  assert.equal(summary, "older compact context");
  assert.deepEqual(history, [
    { role: "user", content: "recent" },
    { role: "assistant", content: "recent reply" },
    { role: "user", content: "current" },
  ]);
  summary = clearBrowserContext(history);
  assert.equal(summary, null);
  assert.deepEqual(history, []);
});

test("corrupt persisted state is safely ignored", () => {
  const fixture = storage();
  fixture.setItem("production-wheel.selection", "{");
  assert.deepEqual(restoreSelection(fixture), {});
});

test("manual changes preserve model versions and express caps using governed policy", () => {
  const draft = {
    revision: 4,
    request: {
      config: {
        versions: { matrix_version: "VERSION_A" },
        target_band: { upper_days: 7 },
      },
    },
  };
  const patch = mergeDraft(draft, {
    pv: "OPTIMIZED",
    coverage: "PARETO",
    matrix: "FLEXIBLE",
    cap: "9",
    scope: '[{"plant":"P1"}]',
    constraints: "[]",
    budget: '{"wall_time_seconds":60}',
  });
  assert.equal(patch.revision, 4);
  assert.equal(patch.request.config.versions.matrix_version, "VERSION_A");
  assert.deepEqual(patch.request.config.group_size, {
    base_limit: 7,
    mode: "BOUNDED_RELAXATION",
    max_excess: 2,
  });
  assert.equal(draft.request.config.pv_mode, undefined);
});

test("invalid global cap and malformed JSON fail before HTTP mutation", () => {
  const fields = {
    pv: "FIXED",
    coverage: "PARETO",
    matrix: "HARD",
    cap: 6,
    scope: "[]",
    constraints: "[]",
    budget: "{}",
  };
  assert.throws(() => mergeDraft({ revision: 1 }, fields), /at least 7/);
  assert.throws(
    () => mergeDraft({ revision: 1 }, { ...fields, cap: 7, scope: "{" }),
    SyntaxError,
  );
});

test("stream parser retains split UTF-8, ordered events and final non-newline event", async () => {
  const bytes = new TextEncoder().encode(
    '{"type":"tool_call","name":"café"}\n{"type":"assistant","text":"Done"}',
  );
  const stream = new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += 3)
        controller.enqueue(bytes.slice(i, i + 3));
      controller.close();
    },
  });
  const events = [];
  await decodeNDJSON(stream, async (event) => events.push(event));
  assert.deepEqual(events, [
    { type: "tool_call", name: "café" },
    { type: "assistant", text: "Done" },
  ]);
});

test("stream callback failures propagate to visible error handling", async () => {
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('{"type":"error"}\n'));
      controller.close();
    },
  });
  await assert.rejects(
    decodeNDJSON(stream, () => {
      throw new Error("Server failed");
    }),
    /Server failed/,
  );
});

test("terminal run states stop polling but running states do not", () => {
  assert.equal(isTerminal("COMPLETED"), true);
  assert.equal(isTerminal("cancelled"), true);
  assert.equal(isTerminal("worker_lost"), true);
  assert.equal(isTerminal("running"), false);
});

test("technical Changeover fields keep API identities but receive display labels", async () => {
  const { fieldLabel } = await import("../src/workspace/dom.js");
  assert.equal(fieldLabel("j_ch"), "Changeover proxy");
  assert.equal(fieldLabel("j_ch_contribution"), "Group changeover proxy contribution");
  assert.equal(fieldLabel("baseline_j_ch"), "Baseline changeover proxy");
});

/** Verify eligibility preserves the distinction between false and missing across data sources. */
test("line permissions preserve numeric, boolean and unknown source values", async () => {
  const { fieldValue } = await import("../src/workspace/vocabulary.js");
  for (const value of [0, "0", false]) assert.equal(fieldValue("eligible", value), "No");
  for (const value of [1, "1", true]) assert.equal(fieldValue("eligible", value), "Yes");
  for (const value of [null, undefined, ""]) assert.equal(fieldValue("eligible", value), "Unknown");
  assert.equal(fieldValue("filling_line", "1"), "1");
  assert.equal(fieldValue("material", "00123"), "00123");
  assert.equal(fieldValue("eligible", "unexpected"), "unexpected");
});

test("delete treats an already-deleted entity as success but surfaces other failures", async () => {
  const original = globalThis.fetch;
  const reply = (status, detail) => async () =>
    new Response(JSON.stringify({ detail }), { status });
  try {
    globalThis.fetch = reply(404, "Unknown dataset, run, draft, or artifact identifier");
    assert.equal(await deleteEntity("/api/datasets/x"), null);
    // A missing route (wrong backend) must not masquerade as a completed delete.
    globalThis.fetch = reply(404, "Not Found");
    await assert.rejects(deleteEntity("/api/datasets/x"), /404: Not Found/);
    globalThis.fetch = reply(422, "This snapshot has active runs");
    await assert.rejects(deleteEntity("/api/datasets/x"), /422: This snapshot has active runs/);
  } finally {
    globalThis.fetch = original;
  }
});
