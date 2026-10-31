import { el, evidence, display, fieldLabel } from "./dom.js";

/** Render metric deltas and population caveats ahead of complete assignment evidence. */
export function renderComparison(target, result) {
  target.replaceChildren();
  const metrics = el("table");
  const head = el("thead");
  const header = el("tr");
  for (const label of [
    "Metric",
    "Left point",
    "Right point",
    "Delta (right − left)",
  ])
    header.append(el("th", label));
  head.append(header);
  metrics.append(head);
  const body = el("tbody");
  for (const item of result.metrics || []) {
    const row = el("tr");
    for (const value of [
      fieldLabel(item.field),
      item.left,
      item.right,
      item.delta,
    ])
      row.append(
        el(
          "td",
          typeof value === "number"
            ? value.toLocaleString(undefined, { maximumFractionDigits: 4 })
            : display(value),
        ),
      );
    body.append(row);
  }
  metrics.append(body);
  const scroll = el("div", null, "table-scroll");
  scroll.append(metrics);
  target.append(scroll);
  const population = result.population || {};
  target.append(
    el(
      "p",
      `Population: ${population.left_count ?? "—"} left · ${population.right_count ?? "—"} right · ${population.intersection_count ?? "—"} shared. ${result.membership_changes?.length || 0} membership changes · ${result.pv_changes?.length || 0} PV changes.`,
      "muted",
    ),
  );
  const caveats = el("ul");
  for (const caveat of result.caveats || []) caveats.append(el("li", caveat));
  target.append(caveats);
  const details = el("details");
  details.append(
    el("summary", "Full membership, PV, configuration & proof evidence"),
  );
  const content = el("div");
  evidence(content, result);
  details.append(content);
  target.append(details);
}
