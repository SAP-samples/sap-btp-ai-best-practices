/**
 * Header user indicator and API authentication contracts.
 *
 * Run from ui/:
 *   node --test tests/user_session.test.js
 */
import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import {
  AJAX_HEADERS,
  API_BASE_URL,
  behindApprouter,
  CSRF_HEADER,
  csrfFetch,
  LOGOUT_PATH,
  needsCsrfToken,
  request,
  resetCsrfToken,
} from "../src/services/api.js";
import { mountUserSession } from "../src/workspace/user_session.js";

/** Header elements as they appear in index.html (both hidden at start). */
function header() {
  const { window } = new JSDOM(
    "<!doctype html><span id='user-name' hidden></span><a id='sign-out' href='/do/logout' hidden>Sign out</a>",
  );
  const document = window.document;
  return {
    name: /** @type {HTMLElement} */ (document.querySelector("#user-name")),
    signOut: /** @type {HTMLAnchorElement} */ (document.querySelector("#sign-out")),
  };
}

test("behind the approuter the header shows the user and the sign-out link", async () => {
  const elements = header();
  await mountUserSession(elements, {
    approuter: true,
    loadUser: async () => ({ display_name: "Jane Doe", email: "jane.doe@example.com" }),
  });
  assert.equal(elements.name.hidden, false);
  assert.equal(elements.name.textContent, "Jane Doe");
  assert.equal(elements.name.title, "jane.doe@example.com");
  assert.equal(elements.signOut.hidden, false);
  assert.equal(elements.signOut.getAttribute("href"), LOGOUT_PATH);
});

test("local development hides sign-out, which only the approuter provides", async () => {
  const elements = header();
  await mountUserSession(elements, { approuter: false, loadUser: async () => ({ display_name: "Local Developer" }) });
  assert.equal(elements.signOut.hidden, true);
  assert.equal(elements.name.textContent, "Local Developer");
});

test("the public deployment (approuter without sign-in) hides sign-out", async () => {
  const elements = header();
  await mountUserSession(elements, {
    approuter: true,
    loadUser: async () => ({ display_name: "Public access", email: null, sign_in_required: false }),
  });
  assert.equal(elements.signOut.hidden, true);
  assert.equal(elements.name.textContent, "Public access");
});

test("a failed user lookup leaves the name hidden without throwing", async () => {
  const elements = header();
  const warn = console.warn;
  console.warn = () => {};
  try {
    await mountUserSession(elements, { approuter: true, loadUser: async () => { throw new Error("401"); } });
  } finally {
    console.warn = warn;
  }
  assert.equal(elements.name.hidden, true);
});

test("an empty base URL means same-origin calls through the approuter", () => {
  assert.equal(behindApprouter(""), true);
  assert.equal(behindApprouter("http://127.0.0.1:8000"), false);
});

test("API calls carry the AJAX marker and no API key", async () => {
  const original = globalThis.fetch;
  /** @type {RequestInit | undefined} */
  let sent;
  let url = "";
  globalThis.fetch = async (target, init) => {
    url = String(target);
    sent = init;
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  };
  try {
    await request("/api/me");
  } finally {
    globalThis.fetch = original;
  }
  const headers = /** @type {Record<string, string>} */ (sent?.headers);
  assert.equal(url, `${API_BASE_URL}/api/me`);
  assert.equal(headers["X-Requested-With"], AJAX_HEADERS["X-Requested-With"]);
  assert.equal("X-API-Key" in headers, false);
});

test("only state-changing calls behind the approuter need a CSRF token", () => {
  assert.equal(needsCsrfToken("POST", ""), true);
  assert.equal(needsCsrfToken("delete", ""), true);
  assert.equal(needsCsrfToken("GET", ""), false);
  assert.equal(needsCsrfToken(undefined, ""), false);
  assert.equal(needsCsrfToken("POST", "http://127.0.0.1:8000"), false);
});

/**
 * Fake approuter: issues token "t1" then "t2" on fetch requests and accepts a POST
 * only with the newest token, answering 403 + "X-CSRF-Token: Required" otherwise.
 */
function fakeApprouter() {
  const calls = [];
  let issued = 0;
  const fetchImpl = async (url, init = {}) => {
    const headers = /** @type {Record<string, string>} */ (init.headers || {});
    calls.push({ url, method: init.method, token: headers[CSRF_HEADER] });
    if (headers[CSRF_HEADER] === "Fetch") {
      issued += 1;
      return new Response("{}", { status: 200, headers: { [CSRF_HEADER]: `t${issued}` } });
    }
    if (headers[CSRF_HEADER] === `t${issued}`) return new Response("{}", { status: 200 });
    return new Response("", { status: 403, headers: { [CSRF_HEADER]: "Required" } });
  };
  return { calls, fetchImpl, expire: () => (issued += 1) };
}

test("a POST behind the approuter fetches the CSRF token once and reuses it", async () => {
  resetCsrfToken();
  const router = fakeApprouter();
  const options = { fetchImpl: router.fetchImpl, baseUrl: "" };
  assert.equal((await csrfFetch("/api/runs", { method: "POST" }, options)).status, 200);
  assert.equal((await csrfFetch("/api/runs", { method: "POST" }, options)).status, 200);
  assert.deepEqual(router.calls.map((call) => call.token), ["Fetch", "t1", "t1"]);
  resetCsrfToken();
});

test("a rejected CSRF token is refreshed and the call retried once", async () => {
  resetCsrfToken();
  const router = fakeApprouter();
  const options = { fetchImpl: router.fetchImpl, baseUrl: "" };
  await csrfFetch("/api/runs", { method: "POST" }, options);
  router.expire();
  const response = await csrfFetch("/api/runs", { method: "DELETE" }, options);
  assert.equal(response.status, 200);
  assert.deepEqual(router.calls.map((call) => call.token), ["Fetch", "t1", "t1", "Fetch", "t3"]);
  resetCsrfToken();
});
