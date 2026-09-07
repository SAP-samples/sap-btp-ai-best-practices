import assert from "node:assert/strict";
import test from "node:test";

import {
  buildAssessmentProfilePayload,
  naceOptionsForClass,
  resolveAssessmentProfile
} from "../src/pages/assessment/assessmentProfileState.js";

const options = {
  available: true,
  cohorts: [
    { customer_class: "class_2", nace1: "Manufacturing" },
    { customer_class: "class_3", nace1: "Electricity and gas" },
    { customer_class: "class_3", nace1: "Mining" }
  ]
};

test("persisted profile is authoritative over the browser class hint", () => {
  const persisted = {
    assessment_id: "demo-assessment",
    display_name: "Demo assessment",
    source_company_id: null,
    customer_class: "class_2",
    nace1: "Manufacturing"
  };

  assert.deepEqual(resolveAssessmentProfile(options, persisted, "class_3"), persisted);
});

test("a browser class value is only a one-time default for a valid imported cohort", () => {
  assert.deepEqual(resolveAssessmentProfile(options, null, "class_3"), {
    customer_class: "class_3",
    nace1: "Electricity and gas",
    needsPersistence: true
  });
  assert.deepEqual(naceOptionsForClass(options, "class_3"), [
    "Electricity and gas",
    "Mining"
  ]);
});

test("profile writes carry the exact selected class and NACE-1 context", () => {
  assert.deepEqual(
    buildAssessmentProfilePayload("demo-assessment", "class_3", "Mining"),
    {
      assessment_id: "demo-assessment",
      display_name: "Demo assessment",
      source_company_id: null,
      customer_class: "class_3",
      nace1: "Mining"
    }
  );
});
