import { button, element } from "./workspace-dom.js";

const CODE_SEVERITY = {
  // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: MISSING_INFO_RECORD: "high",
  // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: MISSING_WORK_INSTRUCTION: "high",
  MISSING_BOM_MATCHES: "high",
  MBOM_NOT_UPDATED: "high",
  DECLARED_PHASE_OUT: "high",
  REPLACEMENT_NOT_CONFIRMED: "medium",
  EBOM_130_SOURCE: "medium",
};
const SEVERITY_BORDER = { high: "#aa0808", medium: "#8d5e00", low: "#256f3a" };
const SEVERITY_COUNT_TONE = { high: "tone-error", medium: "tone-warning", low: "tone-success" };

export function renderReadiness(host, snapshot, inspect, showDatasets, descriptions = []) {
  const opened = new Set([...host.querySelectorAll("details[data-code][open]")].map(node => node.dataset.code));
  host.replaceChildren();

  const counts = snapshot.counts;
  host.append(element("h3", "Readiness"));
  host.append(element("p", `${counts.located_objects} of ${counts.technical_objects} parts in the ECN were found in the assessed BOMs.`, "helper-text"));

  if (!snapshot.questions.length) {
    host.append(element("p", "Nothing needs attention for this assessment."));
    return;
  }

  const byCode = new Map(descriptions.map(item => [item.code, item.description]));

  const severityOrder = { high: 0, medium: 1, low: 2, undefined: 3 };
  const sorted = [...snapshot.questions].sort((a, b) =>
    (severityOrder[CODE_SEVERITY[a.code]] ?? 3) - (severityOrder[CODE_SEVERITY[b.code]] ?? 3)
  );

  // Severity summary chips
  const highCount = sorted.filter(q => CODE_SEVERITY[q.code] === "high").length;
  const medCount = sorted.filter(q => CODE_SEVERITY[q.code] === "medium").length;
  const otherCount = sorted.length - highCount - medCount;
  const summaryRow = element("div", "", "readiness-summary");
  if (highCount) summaryRow.append(element("span", `${highCount} blocking`, "semantic-pill tone-error"));
  if (medCount) summaryRow.append(element("span", `${medCount} advisory`, "semantic-pill tone-warning"));
  if (otherCount) summaryRow.append(element("span", `${otherCount} note${otherCount !== 1 ? "s" : ""}`, "semantic-pill tone-neutral"));
  host.append(summaryRow);

  const cards = element("div", "", "followup-cards");

  for (const question of sorted) {
    const severity = CODE_SEVERITY[question.code];
    const borderColor = SEVERITY_BORDER[severity] || "#d5dadd";
    const affectedCount = question.affected_count || question.subjects?.length || 0;

    const card = element("div", "", "followup-card");
    card.style.borderLeftColor = borderColor;

    const panel = element("details");
    panel.dataset.code = question.code;
    panel.open = opened.has(question.code);

    // Compact summary row: severity dot + title + affected count
    const summary = element("summary");
    const dot = element("span", "", "severity-dot");
    dot.dataset.severity = severity || "";
    const countPill = element("span", `${affectedCount} affected`, `severity-count ${SEVERITY_COUNT_TONE[severity] || ""}`);
    summary.append(dot, element("strong", question.question), countPill);
    panel.append(summary);

    // Why text (direct child of details — required by tests)
    panel.append(element("p", byCode.get(question.code) || question.why));

    for (const subject of question.subjects || []) {
      const row = element("div", "", "followup-subject");
      row.append(button(`View ${subject.number || "unnumbered object"}`, () => void inspect(subject.number || subject.object_key)));
      const facts = element("ul");
      for (const fact of (subject.details || []).slice(0, 3)) facts.append(element("li", fact));
      row.append(facts);
      if (subject.details?.length > 3) row.append(element("p", `${subject.details.length} source rows affected. Open the material to inspect all rows.`, "helper-text"));
      panel.append(row);
    }

    if (question.scopes?.length) {
      const scope = element("details"); scope.append(element("summary", "BOM Datasets Checked"));
      for (const value of question.scopes) scope.append(element("p", value));
      panel.append(scope);
    }

    const next = element("p", "", "followup-next");
    next.append(element("strong", "Next step: "), document.createTextNode(question.resolution));
    panel.append(next, button("View data used", showDatasets));

    card.append(panel);
    cards.append(card);
  }

  host.append(cards);
}
