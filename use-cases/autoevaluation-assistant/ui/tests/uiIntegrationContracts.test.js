import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const adminHtml = readFileSync(
  new URL("../src/pages/admin-documents/admin-documents.html", import.meta.url),
  "utf8"
);
const adminJs = readFileSync(
  new URL("../src/pages/admin-documents/admin-documents.js", import.meta.url),
  "utf8"
);
const assessmentJs = readFileSync(
  new URL("../src/pages/assessment/assessment.js", import.meta.url),
  "utf8"
);
const scoreHtml = readFileSync(
  new URL("../src/pages/score/score.html", import.meta.url),
  "utf8"
);
const scoreJs = readFileSync(
  new URL("../src/pages/score/score.js", import.meta.url),
  "utf8"
);
const translations = readFileSync(
  new URL("../src/modules/i18n.js", import.meta.url),
  "utf8"
);

test("benchmark administration is separate from the multi-document RAG uploader", () => {
  assert.match(adminHtml, /class="benchmark-admin-section"/);
  assert.match(adminHtml, /id="benchmark-workbook-input"[^>]*accept="\.xlsx/);
  assert.doesNotMatch(
    adminHtml.match(/id="benchmark-workbook-input"[^>]*>/)?.[0] || "",
    /\bmultiple\b/
  );
  assert.match(adminJs, /window\.confirm\(t\("admin\.benchmark\.confirmActivation"\)\)/);
  assert.match(adminJs, /buildBenchmarkImportForm\(file, false\)/);
  assert.match(adminJs, /buildBenchmarkImportForm\(file, true\)/);
  assert.match(adminJs, /refreshButton\.disabled = mutationBusy \|\| benchmarkState\.historyLoading/);
  assert.match(adminJs, /completeBenchmarkHistoryLoad/);
});

test("assessment profile context is loaded from and persisted to the backend", () => {
  assert.match(assessmentJs, /\/api\/assessment\/benchmark-options/);
  assert.match(assessmentJs, /\/api\/assessment\/profile/);
  assert.match(assessmentJs, /"PUT"/);
  assert.match(assessmentJs, /assessmentLoadOwner\.owns\(requestGeneration\)/);
  assert.ok(
    assessmentJs.split("assessmentLoadOwner.invalidate()").length - 1 >= 3,
    "profile PUT and route remount must invalidate older profile reads"
  );
  assert.doesNotMatch(assessmentJs, /Energy/);
});

test("Score is profile-authoritative, localized, and removes numeric score markup", () => {
  assert.match(scoreJs, /language=\$\{encodeURIComponent\(language\)\}/);
  assert.match(scoreJs, /const requestBody = \{ assessment_id: ASSESSMENT_ID, language \}/);
  assert.doesNotMatch(scoreJs, /requestBody\.customer_class|requestBody\.sector/);
  assert.doesNotMatch(scoreJs, /DEFAULT_SECTOR|Energy|same_sector|same_size/);
  assert.match(scoreJs, /scoreLoadOwner\.owns\(requestGeneration\)/);
  const ownerCheck = scoreJs.indexOf("if (!ownsMountedScoreLoad(requestGeneration))");
  const chartReplacement = scoreJs.indexOf("destroyScoreCharts();", ownerCheck);
  assert.ok(ownerCheck >= 0 && chartReplacement > ownerCheck);
  assert.doesNotMatch(scoreHtml, /score-numeric-details|score-detail-table/);
  assert.doesNotMatch(scoreJs, /formatExactScore|renderNumericDetails/);
  assert.doesNotMatch(scoreJs, /peer_sample_size|sampleSize/);
  assert.doesNotMatch(
    translations,
    /"score\.[^"]+":\s*"[^"]*\{sampleSize\}[^"]*"/
  );
});

test("every new benchmark and positioning label exists in both languages", () => {
  [
    "admin.benchmark.validationFailed",
    "admin.benchmark.activationFailed",
    "assessment.profileUnavailable",
    "score.position.below_peers",
    "score.position.in_line_with_peers",
    "score.position.above_peers",
    "score.unavailableSmallCohort"
  ].forEach((key) => {
    const occurrences = translations.split(`"${key}"`).length - 1;
    assert.equal(occurrences, 2, `${key} must be present once in English and once in Italian`);
  });
});

test("visible chart descriptions are qualitative rather than numeric", () => {
  const descriptions = [
    ...translations.matchAll(/"score\.chartDescription":\s*"([^"]+)"/g)
  ].map((match) => match[1]);
  assert.equal(descriptions.length, 2);
  descriptions.forEach((description) => assert.doesNotMatch(description, /\d/));
});
