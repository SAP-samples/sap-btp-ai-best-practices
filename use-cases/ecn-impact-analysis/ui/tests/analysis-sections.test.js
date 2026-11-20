import { expect, it } from "vitest";
import { renderAnalysisSections } from "../src/components/analysis-sections.js";
import { detailEnvelope } from "./fixtures/api-responses.js";

it("renders readable narrative sections without evidence IDs, raw enums, or payloads", () => {
  const detail = detailEnvelope();
  detail.analysis.sections.change_intent.assessment_markdown += " [E-0123456789abcdef]";
  detail.analysis.sections.uncertainties_and_missing_evidence.missing_evidence = [
    "Confirm the source record E-fedcba9876543210.",
  ];
  const view = renderAnalysisSections(detail);
  document.body.replaceChildren(view);
  const headings = [...document.querySelectorAll("summary h3")].map((node) => node.textContent);
  expect(headings).toEqual([
    "Change Intent",
    "Affected Materials",
    "EBOM and MBOM Impact",
    "Configuration Scope",
    "Uncertainties and Missing Evidence",
  ]);
  expect(document.body.textContent).toContain("Not applicable");
  expect(document.body.textContent).toContain("Authoritative affected-to-resulting relation");
  expect(document.body.textContent).not.toContain("E-4");
  expect(document.body.textContent).not.toContain("E-0123456789abcdef");
  expect(document.body.textContent).not.toContain("E-fedcba9876543210");
  expect(document.body.textContent).not.toContain("INSUFFICIENT_EVIDENCE");
  expect(document.body.textContent).not.toContain("DETERMINISTIC_FINDING");
});

it.each(["QUEUED", "BLOCKED", "RUNNING", "FAILED"])(
  "shows five useful narrative placeholders when the job is %s",
  (state) => {
    const detail = detailEnvelope({
      analysis: null,
      job: {
        job_id: "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        state,
        attempt_count: 1,
        error_code: state === "FAILED" ? "PROVIDER_TIMEOUT" : null,
        error_message: state === "FAILED" ? "Generation timed out" : null,
      },
    });
    document.body.replaceChildren(renderAnalysisSections(detail));
    expect(document.querySelectorAll(".analysis-placeholder")).toHaveLength(5);
    expect(document.body.textContent).toContain(
      state === "BLOCKED" ? "waiting for provider recovery" : "business evidence",
    );
  },
);

it("preserves caller-controlled accordion choices without opening gaps automatically", () => {
  const detail = detailEnvelope();
  detail.analysis.completeness = "COMPLETED";
  detail.analysis.sections.uncertainties_and_missing_evidence = {
    status: "SUFFICIENT",
    assessment_markdown: "No unresolved evidence gap remains.",
    evidence_ids: [],
    missing_evidence: [],
  };
  document.body.replaceChildren(renderAnalysisSections(detail, {
    openSections: new Set(["configuration_scope"]),
  }));
  expect(document.querySelector('details[data-section-key="configuration_scope"]').open).toBe(true);
  expect(document.querySelector('details[data-section-key="uncertainties_and_missing_evidence"]').open).toBe(false);
});
