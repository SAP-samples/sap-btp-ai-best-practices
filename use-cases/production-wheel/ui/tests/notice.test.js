/**
 * Notice banner contracts: API failures versus page code bugs.
 *
 * Run from ui/:
 *   node --test tests/notice.test.js
 */
import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { ApiError } from "../src/services/api.js";
import { describeFailure, mountNotice, SESSION_EXPIRED_TEXT } from "../src/workspace/notice.js";

/** A hidden banner element plus a console stand-in that records error() calls. */
function harness() {
  const { window } = new JSDOM("<!doctype html><div id='notice' role='alert' hidden></div>");
  const element = window.document.querySelector("#notice");
  const logged = [];
  const notice = mountNotice(element, { error: (...values) => logged.push(values) });
  return { element, notice, logged };
}

test("API failures keep the status prefix callers test for", () => {
  const error = new ApiError(409, "draft revision changed");
  assert.equal(error.message, "409: draft revision changed");
  assert.ok(error.message.startsWith("409"));
  assert.deepEqual(describeFailure(error), { text: "409: draft revision changed", pageError: false });
});

test("an unreachable API reads as plain text", () => {
  assert.equal(new ApiError(0, "Cannot reach the workspace API").message, "Cannot reach the workspace API");
});

test("a page code bug is named as a display error and logged with its stack", () => {
  const { element, notice, logged } = harness();
  const bug = new TypeError("Cannot read properties of undefined (reading 'append')");
  notice.show(bug);
  assert.equal(element.hidden, false);
  assert.equal(element.dataset.kind, "page");
  assert.match(element.textContent, /error in the page itself/);
  assert.match(element.textContent, /data and runs are not affected/);
  assert.match(element.textContent, /reading 'append'/);
  assert.equal(logged.length, 1);
  assert.equal(logged[0][1], bug);
});

test("readable errors raised on purpose are shown unchanged", () => {
  const { element, notice, logged } = harness();
  notice.show(new Error("Global group cap must be an integer of at least 7."));
  assert.equal(element.textContent, "Global group cap must be an integer of at least 7.");
  assert.equal(element.dataset.kind, "request");
  assert.equal(logged.length, 0);
});

test("aborted requests do not show the banner", () => {
  const { element, notice } = harness();
  const aborted = new Error("The operation was aborted.");
  aborted.name = "AbortError";
  notice.show(aborted);
  assert.equal(element.hidden, true);
});

test("clear and a click both hide the banner", () => {
  const { element, notice } = harness();
  notice.show(new ApiError(500, "boom"));
  notice.clear();
  assert.equal(element.hidden, true);
  notice.show(new ApiError(500, "boom"));
  element.click();
  assert.equal(element.hidden, true);
});

test("an expired sign-in session asks to sign in again and a click reloads the page", () => {
  const { window } = new JSDOM("<!doctype html><div id='notice' role='alert' hidden></div>");
  const element = window.document.querySelector("#notice");
  let reloads = 0;
  const notice = mountNotice(element, { error: () => {} }, () => reloads++);
  notice.show(new ApiError(401, "Sign-in required"));
  assert.equal(element.hidden, false);
  assert.equal(element.dataset.kind, "session");
  assert.equal(element.textContent, SESSION_EXPIRED_TEXT);
  element.click();
  assert.equal(reloads, 1);
  assert.equal(element.hidden, false);
});

test("a missing role (403) is reported as a normal request failure, not as an expired session", () => {
  assert.deepEqual(describeFailure(new ApiError(403, "Your user does not have access to this application")), {
    text: "403: Your user does not have access to this application",
    pageError: false,
  });
});
