import { renderMarkdown, stripInternalEvidenceIds } from "../modules/markdown.js";

export const SECTION_DEFINITIONS = Object.freeze([
  ["change_intent", "Change Intent"],
  ["affected_materials", "Affected Materials"],
  ["ebom_mbom_impact", "EBOM and MBOM Impact"],
  ["configuration_scope", "Configuration Scope"],
  ["uncertainties_and_missing_evidence", "Uncertainties and Missing Evidence"],
]);

const badgeLabel = Object.freeze({
  SUFFICIENT: "Supported",
  INSUFFICIENT_EVIDENCE: "More evidence needed",
  NOT_APPLICABLE: "Not applicable",
});

/** Render AI narrative sections without exposing internal evidence identifiers. */
export function renderAnalysisSections(detail, { openSections = null } = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = "analysis-sections";
  for (const [key, title] of SECTION_DEFINITIONS) {
    const panel = document.createElement("details");
    panel.dataset.sectionKey = key;
    panel.open = openSections?.has(key) || false;
    const summary = document.createElement("summary");
    const heading = document.createElement("h3");
    heading.textContent = title;
    summary.append(heading);
    panel.append(summary);

    const stored = detail.analysis?.sections?.[key] || detail.analysis_stages?.SUMMARY?.sections?.[key] || null;
    if (!stored) {
      const placeholder = document.createElement("p");
      placeholder.className = "analysis-placeholder";
      placeholder.textContent = detail.job.state === "FAILED"
        ? "The AI narrative is unavailable; business evidence can still be inspected above."
        : detail.job.state === "BLOCKED"
          ? "The deterministic result is ready. AI generation is waiting for provider recovery."
          : "The AI narrative is being generated; business evidence is already available above.";
      panel.append(placeholder);
      wrapper.append(panel);
      continue;
    }

    const badge = document.createElement("span");
    badge.className = `status-badge status-${stored.status.toLowerCase()}`;
    badge.textContent = badgeLabel[stored.status] || "Assessment available";
    panel.append(badge);
    const markdown = document.createElement("div");
    markdown.className = "markdown";
    markdown.append(renderMarkdown(stored.assessment_markdown));
    panel.append(markdown);
    if (stored.missing_evidence?.length) {
      const missingHeading = document.createElement("h4");
      missingHeading.textContent = "What Is Still Needed";
      const missing = document.createElement("ul");
      missing.className = "missing-evidence";
      for (const value of stored.missing_evidence) {
        const item = document.createElement("li");
        item.textContent = stripInternalEvidenceIds(value);
        missing.append(item);
      }
      panel.append(missingHeading, missing);
    }
    wrapper.append(panel);
  }
  return wrapper;
}
