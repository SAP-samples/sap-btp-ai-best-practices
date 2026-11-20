import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/Tag.js";
import "@ui5/webcomponents-icons/dist/ai.js";
import * as api from "../services/workspace-api.js";
/* MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: curated Make-or-Buy charts are not rendered.
import { disposeChart, initChart, makeBuyRadarOption } from "./ecn-charts.js";
*/
import { button, element, table } from "./workspace-dom.js";
import { buildPlantNameIndex, plantLabel } from "../modules/plant-display.js";
import { engineeringDrawingPreview } from "./engineering-drawing-preview.js";
import { acceptanceIssueSection } from "./acceptance-materials.js";
import { createLatestRequestGate } from "../modules/request-generation.js";

/** Mark a section or header whose content is based on AI-extracted drawing inputs. */
function aiAttributionIcon(accessibleName) {
  const icon = document.createElement("ui5-icon");
  icon.className = "ai-attribution-icon";
  icon.setAttribute("name", "ai");
  icon.setAttribute("mode", "Image");
  icon.setAttribute("accessible-name", accessibleName);
  return icon;
}

function aiAttributedLabel(tag, text, className, accessibleName) {
  const label = element(tag, "", className);
  label.append(aiAttributionIcon(accessibleName), document.createTextNode(text));
  return label;
}

function markAiAttributedColumn(tableElement, columnIndex, accessibleName) {
  const header = tableElement.querySelectorAll("thead th")[columnIndex];
  if (header) header.prepend(aiAttributionIcon(accessibleName));
}

/** Translate controlled source codes without interpreting source text as HTML. */
export function readable(value) {
  return ({ RESULTING: "Resulting object", AFFECTED: "Affected object", BOM_ECN_REFERENCE: "BOM change reference", APPROVEDFORSERIES: "Approved for series", TOBEOBSOLETE: "To be obsolete", PHASE_OUT: "Phase-out", PHASE_IN: "Phase-in", REVISION: "Revision", REVIEW: "Needs review" })[value] || String(value || "Not supplied").replaceAll("_", " ").toLowerCase();
}
/** Convert JSON coordinates with or without an export index to readable one-based breadcrumbs. */
export function sourceLabel(path) {
  if (!path?.startsWith("$")) return path || "Source not supplied";
  const labels = { ImplementationPlan: "Implementation plan", ResultingLinks: "Resulting link", AffectedLinks: "Affected link", CNAffectLinks: "Affected link", ResultingObjects: "Object", AffectedObjects: "Object" };
  return path.replace(/^\$(?:\[\d+\])?\.?/, "").split(".").filter(Boolean).map(part => {
    const match = /^(\w+)(?:\[(\d+)\])?$/.exec(part);
    if (!match) return "Source field";
    const label = labels[match[1]] || match[1].replace(/([a-z])([A-Z])/g, "$1 $2").replaceAll("_", " ");
    return `${label}${match[2] == null ? "" : ` ${Number(match[2]) + 1}`}`;
  }).join(" › ") || "ECN source record";
}
/** Format server-normalized calendar dates without timezone conversion or guessing. */
export function validity(value, quality, raw) {
  if (quality === "LEGACY_PLACEHOLDER") return "Legacy placeholder";
  if (value === "9999-12-31") return "No planned end";
  if (!value) return !raw ? "Not supplied" : "Could not interpret";
  const [year, month, day] = value.split("-");
  return `${Number(day)} ${["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][Number(month) - 1]} ${year}`;
}
/** Return the observed source path without filtering or deriving configuration semantics. */
export function hierarchyNodes(location) {
  return (location.path || []).map(node => ({ node, tag: "" }));
}
/** Text of the "End date" line for one location, or "" when there is no planned end (empty or 9999-12-31). */
export function endDateLine(location) {
  const { valid_to_date: date, valid_to_quality: quality, valid_to: raw } = location;
  if (date === "9999-12-31" || (!date && !raw && quality !== "LEGACY_PLACEHOLDER")) return "";
  return `End date: ${validity(date, quality, raw)}`;
}

function sourceCounts(occurrences, sources) {
  const counts = new Map();
  for (const occurrence of occurrences) counts.set(occurrence.source, (counts.get(occurrence.source) || 0) + 1);
  return sources
    .filter(source => counts.get(source))
    .map(source => `${source}: ${counts.get(source)}`)
    .join("; ") || "None";
}

function coverageLabel(coverage) {
  const labels = {
    OBSERVED_IN_BOTH: "Found in EBOM and MBOM",
    ENGINEERING_ONLY: "EBOM only",
    MANUFACTURING_ONLY: "MBOM only",
    NOT_OBSERVED: "Not observed in supplied BOM sources",
  };
  return Object.hasOwn(labels, coverage) ? labels[coverage] : "BOM presence not supplied";
}

function coverageEffectiveInterval(coverage) {
  if (!coverage.effective_interval_count) return "Not supplied";
  if (coverage.effective_interval_count > 1) return `Multiple intervals (${coverage.effective_interval_count})`;
  return coverage.occurrences?.[0] ? occurrenceEffectiveInterval(coverage.occurrences[0]) : "Not supplied";
}

function occurrenceEffectiveInterval(occurrence) {
  const from = validity(occurrence.valid_from_date, occurrence.valid_from_quality, occurrence.valid_from);
  const to = validity(occurrence.valid_to_date, occurrence.valid_to_quality, occurrence.valid_to);
  return `${from} – ${to}`;
}

function occurrenceQuantity(occurrence) {
  return [occurrence.quantity, occurrence.unit].filter(value => value !== undefined && value !== null && value !== "").join(" ") || "Not supplied";
}

function changeReference(occurrence) {
  return [...new Set([occurrence.change_number, occurrence.change_number_to].filter(Boolean))].join(" · ") || "Not supplied";
}

function hierarchyNodeLabel(node) {
  if (node.description && node.number && node.description !== node.number) return `${node.description} (${node.number})`;
  return node.description || node.number || node.label || "Not supplied";
}

function occurrenceHierarchy(occurrence) {
  if (!Array.isArray(occurrence.path) || !occurrence.path.length) return "Not supplied";
  const chain = element("div", "", "hierarchy-stack");
  for (const { node } of hierarchyNodes(occurrence)) {
    const line = element("div", "", "hierarchy-line");
    line.append(element("span", hierarchyNodeLabel(node), "hierarchy-node"));
    chain.append(line);
  }
  return chain;
}

function materialLabel(obj) {
  if (obj.description && obj.number) return `${obj.description} (${obj.number})`;
  return obj.description || obj.number || "Unnumbered object";
}

function appendBomCoverage(body, obj) {
  const coverage = obj.material_brief?.bom_coverage;
  if (!coverage) return;
  const occurrences = coverage.occurrences || [];
  const names = buildPlantNameIndex([
    ...(obj.material_brief.sourcing || []),
    ...(obj.material_brief.linked_use || []).flatMap(link => link.plants || []),
  ]);

  const section = element("section", "", "material-story-block");
  section.append(element("h3", "Observed BOM Coverage"));
  const help = element("p", "BOM presence shows observations in the supplied engineering (EBOM) and manufacturing (MBOM) sources. It does not confirm complete or approved rollout, or approval for any plant or date.", "helper-text bom-presence-help");
  help.id = "material-bom-presence-help";
  section.append(help);
  const coverageTable = table(["Component", "Engineering occurrences", "Manufacturing occurrences", "Effective interval", "Plant", "BOM presence"], [[
    materialLabel(obj),
    sourceCounts(occurrences, ["Enriched EBOM", "130% EBOM"]),
    sourceCounts(occurrences, ["130% MBOM"]),
    coverageEffectiveInterval(coverage),
    coverage.plants?.map(code => plantLabel(code, names)).join(", ") || "Plant not supplied",
    coverageLabel(coverage.coverage),
  ]]);
  coverageTable.classList.add("material-evidence-table", "bom-coverage-table");
  coverageTable.tabIndex = 0;
  coverageTable.setAttribute("role", "region");
  coverageTable.setAttribute("aria-label", "Observed BOM coverage table.");
  coverageTable.setAttribute("aria-describedby", help.id);
  section.append(coverageTable);

  const count = occurrences.length;
  if (!count) {
    section.append(element("p", "No observed BOM occurrences were supplied for this material.", "helper-text"));
    body.append(section);
    return;
  }
  const details = element("details", "", "observed-occurrences-details");
  details.append(element("summary", `Show ${count} observed BOM occurrence${count === 1 ? "" : "s"}`));
  const occurrencesTable = table(["BOM source", "Assembly / hierarchy context", "Item or position", "Quantity", "Effective interval", "Change reference"], occurrences.map(occurrence => [
    occurrence.source,
    occurrenceHierarchy(occurrence),
    occurrence.item_or_position || "Not supplied",
    occurrenceQuantity(occurrence),
    occurrenceEffectiveInterval(occurrence),
    changeReference(occurrence),
  ]));
  occurrencesTable.classList.add("material-evidence-table", "observed-occurrences-table");
  occurrencesTable.tabIndex = 0;
  occurrencesTable.setAttribute("role", "region");
  occurrencesTable.setAttribute("aria-label", "Observed BOM occurrences. Scroll for additional columns or rows.");
  details.append(occurrencesTable);
  section.append(details);
  body.append(section);
}

function procurementMode(labels = []) {
  const make = labels.some(label => label === "In-house production" || label === "In-house and external procurement allowed");
  const buy = labels.some(label => label === "External procurement" || label === "In-house and external procurement allowed");
  return {
    make,
    buy,
    label: make && buy ? "Make or buy allowed" : make ? "Make allowed" : buy ? "Buy allowed" : "Permission not supplied",
  };
}

function sourcePurchasePriceLabel(source = {}) {
  switch (source?.status) {
    case "SOURCE_PRICE_AVAILABLE": {
      const supplied = value => value !== undefined && value !== null && value !== "";
      const amount = source.currency === "EUR" && supplied(source.net_price)
        ? `€${source.net_price}`
        : [source.currency, source.net_price].filter(supplied).join(" ") || "Source price supplied";
      const basis = [source.price_per, source.order_price_unit].filter(supplied).join(" ");
      return basis ? `${amount} / ${basis}` : amount;
    }
    case "ZERO_PRICE_REVIEW": return "Zero Source Price — Review";
    case "MULTIPLE_SOURCE_PRICES_REVIEW": return "Multiple Source Prices — Review";
    default: return "Not Supplied";
  }
}

function sourceLeadTimeLabel(source = {}) {
  return source.status === "SOURCE_LEAD_TIME_AVAILABLE" && source.display
    ? source.display
    : "Not Supplied";
}

// MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: material-dialog pricing and comparison.
// This curated price, recommendation, and radar-chart implementation is retained
// as comments while the dialog is validated using source-backed evidence only.
// function priceLabel(prices = []) {
//   const price = prices.find(item => item.status === "ACTIVE" && item.net_price) || prices.find(item => item.net_price);
//   if (!price) return "Not supplied";
//   const basis = [price.currency, price.net_price].filter(Boolean).join(" ") || "Price supplied";
//   const quantity = price.price_per && String(price.price_per) !== "1" ? `${price.price_per} ` : "";
//   return `${basis}${price.unit_of_measure ? ` / ${quantity}${price.unit_of_measure}` : ""}`;
// }
//
// function recommendationLabel(value) {
//   return String(value || "Decision pending").replaceAll("_", " ").toLowerCase().replace(/^./, letter => letter.toUpperCase());
// }
//
// function preferredScores(recommendation) {
//   if (recommendation === "MAKE") return [4, 2];
//   if (recommendation === "BUY") return [2, 4];
//   return [2, 3];
// }
//
// function costScores(assessment) {
//   const make = Number(assessment.make_cost);
//   const buy = Number(assessment.buy_cost?.net_price);
//   if (!Number.isFinite(make) || make <= 0 || !Number.isFinite(buy) || buy <= 0
//     || assessment.currency !== assessment.buy_cost?.currency
//     || String(assessment.price_per || "1") !== String(assessment.buy_cost?.price_per || "1")
//     || assessment.unit_of_measure !== assessment.buy_cost?.unit_of_measure) return preferredScores(assessment.recommendation);
//   const higherScore = 5;
//   const lowerScore = Math.max(1, Math.round((Math.min(make, buy) / Math.max(make, buy)) * 4));
//   return make < buy ? [higherScore, lowerScore] : [lowerScore, higherScore];
// }
//
// /** Translate the bounded curated decision evidence into a clearly marked 1–5 comparison profile. */
// export function makeBuyScores(assessment) {
//   const preferred = preferredScores(assessment.recommendation);
//   const capacity = String(assessment.capacity || "").toLowerCase();
//   const history = String(assessment.comparable_history || "").toLowerCase();
//   const tooling = String(assessment.tooling || "").toLowerCase();
//   const constrainedMake = /fully booked|no compatible|not .*available|unavailable/.test(capacity);
//   const availableMake = /available/.test(capacity) && !constrainedMake;
//   const leadTime = constrainedMake ? [1, assessment.recommendation === "BUY_PENDING_QUOTE" ? 3 : 4]
//     : availableMake ? [4, assessment.recommendation === "MAKE" ? 2 : 3] : preferred;
//   const impactedPlantScore = Math.max(1, 5 - Math.max(0, Number(assessment.impacted_plant_count || 1) - 1));
//   const historicalTrend = /externally sourced|externally molded/.test(history) ? [2, 5]
//     : /made equivalent|same process|in-house/.test(history) ? [5, 2]
//       : /no comparable/.test(history) ? [1, 4] : preferred;
//   const quality = /qualified/.test(tooling) ? [5, 3]
//     : /does not produce|no in-house|no approved|unavailable/.test(tooling) ? [1, 4]
//       : /can apply/.test(tooling) ? [2, 3] : preferred;
//   return {
//     make: [costScores(assessment)[0], leadTime[0], impactedPlantScore, historicalTrend[0], quality[0]],
//     buy: [costScores(assessment)[1], leadTime[1], impactedPlantScore, historicalTrend[1], quality[1]],
//   };
// }
//
// function assessmentCostLabel(assessment) {
//   const make = assessment.make_cost ? `${assessment.currency || ""} ${assessment.make_cost}`.trim() : "Not supplied";
//   const quantity = assessment.price_per && String(assessment.price_per) !== "1" ? `${assessment.price_per} ` : "";
//   const makeUnit = assessment.unit_of_measure ? ` / ${quantity}${assessment.unit_of_measure}` : "";
//   return `Make: ${make}${makeUnit}; Buy: ${priceLabel(assessment.buy_cost ? [assessment.buy_cost] : [])}`;
// }
//
// function appendMakeBuyComparison(body, assessments) {
//   if (!assessments.length) return [];
//   const section = element("section", "", "material-story-block make-buy-comparison");
//   section.append(
//     element("h3", "Make vs Buy Comparison"),
//     element("p", "Comparative scores run from 1 (lower) to 5 (stronger). Cost and plant scope use this ECN's evidence; lead time, history, and quality interpret the curated demonstration assessment.", "helper-text"),
//   );
//   const grid = element("div", "", "make-buy-comparison-grid");
//   const charts = [];
//   for (const assessment of assessments) {
//     const plant = assessment.plant_name ? `${assessment.plant_name} (${assessment.plant_code})` : assessment.plant_code || "Plant not supplied";
//     const scores = makeBuyScores(assessment);
//     const card = element("article", "", "make-buy-comparison-card");
//     card.append(
//       element("h4", plant),
//       element("p", `${recommendationLabel(assessment.recommendation)}: ${assessment.recommendation_summary || "Decision rationale not supplied."}`),
//     );
//     const chart = element("div", "", "make-buy-radar");
//     chart.setAttribute("role", "img");
//     chart.setAttribute("aria-label", `Make versus buy comparison for ${plant}. Make scores: ${scores.make.join(", ")}; buy scores: ${scores.buy.join(", ")}.`);
//     card.append(chart);
//     const evidence = element("details", "", "make-buy-comparison-evidence");
//     evidence.append(element("summary", "Decision evidence"));
//     const facts = [
//       ["Cost", assessmentCostLabel(assessment)],
//       ["Lead time", assessment.capacity],
//       ["Impacted plants", assessment.impacted_plant_count ? `${assessment.impacted_plant_count} plant${assessment.impacted_plant_count === 1 ? "" : "s"} in the current ECN scope.` : null],
//       ["Historical trend", assessment.comparable_history],
//       ["Quality and process", assessment.tooling],
//     ];
//     for (const [label, value] of facts) if (value) evidence.append(element("p", `${label}: ${value}`));
//     card.append(evidence);
//     grid.append(card);
//     charts.push({ chart, scores });
//   }
//   section.append(grid);
//   body.append(section);
//   return charts;
// }
// MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: material-dialog pricing and comparison.


function displayNumber(value) {
  if (value === null || value === undefined || value === "") return "Not available";
  const [integer, fraction] = String(value).split(".");
  const grouped = integer.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return fraction === undefined ? grouped : `${grouped}.${fraction}`;
}

/** Keep calculator mm text precision and signs; never round or group digits. */
function displayMillimetreNumber(value) {
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "Not available";
  if (typeof value !== "string" || !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) return "Not available";
  return value;
}

function millimetres(value) {
  const displayed = displayMillimetreNumber(value);
  return displayed === "Not available" ? displayed : `${displayed} mm`;
}

function millimetreRange(minimum, maximum) {
  const minimumValue = displayMillimetreNumber(minimum);
  const maximumValue = displayMillimetreNumber(maximum);
  if (minimumValue === "Not available" || maximumValue === "Not available") return "Not available";
  return `${minimumValue}–${maximumValue} mm`;
}

function degrees(value) {
  const displayed = displayNumber(value);
  return displayed === "Not available" ? displayed : `${displayed}°`;
}

function record(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function records(value) {
  return Array.isArray(value) ? value.map(record) : [];
}

function mappedSource(entry) {
  const source = record(entry);
  return [source.sourcePoint, source.sourceField].filter(Boolean).join(" ");
}

function measurementWithSource(value, entry, formatter = millimetres) {
  const source = mappedSource(entry);
  const measurement = formatter(value);
  return source ? `${measurement} (${source})` : measurement;
}

function tubeLengthInputStatus(status) {
  return status === "Cannot assess" ? "Cannot assess" : "Input";
}

function calculationGroups(criteria) {
  const byId = new Map(criteria.map(criterion => [criterion.id, criterion]));
  const groups = [];
  const material = record(record(byId.get("raw_material")?.calculator).raw_material);
  const derivedMaterial = record(record(byId.get("raw_material")?.calculator).material_designation);
  if (Object.keys(material).length || Object.keys(derivedMaterial).length) {
    const designation = material.designation || derivedMaterial.candidate;
    const listed = material.guidelineCoverage === "listed";
    groups.push({ title: "Material Designation", criteria: [byId.get("raw_material")], rows: [[
      "PS4 Raw-Material Designation",
      designation || "Not available",
      listed ? "Listed in the PS4 raw-material guideline" : "No explicit pass/fail disposition for an unlisted designation",
      listed ? "Pass" : "Cannot assess",
    ]] });
  }

  const length = record(record(byId.get("tube_length")?.calculator).tube_length);
  if (Object.keys(length).length) {
    const rows = [
      ["Outer Diameter", millimetres(length.diameterMm), "Input to the documented tube-length calculator", tubeLengthInputStatus(length.status)],
      ["Total Tube Length", millimetres(length.totalLengthMm), millimetreRange(length.minimumMm, length.maximumMm), length.status],
    ];
    if (length.wallThicknessMm !== undefined && length.wallThicknessMm !== "") {
      rows.splice(1, 0, ["Wall Thickness", millimetres(length.wallThicknessMm), "Input to the documented tube-length calculator", tubeLengthInputStatus(length.status)]);
    }
    groups.push({ title: "Tube Length", criteria: [byId.get("tube_length")], rows });
  }

  const angles = records(record(byId.get("bend_angles")?.calculator).bend_angles?.results);
  if (angles.length) {
    groups.push({ title: "Bend Angles", criteria: [byId.get("bend_angles")], rows: angles.map((angle, index) => [
      `Bend Angle ${index + 1}`,
      degrees(angle.angleDeg),
      "0° < angle < 180°",
      angle.status,
    ]) });
  }

  const crimpCalculator = record(byId.get("automatic_crimp_dimensions")?.calculator);
  const mapping = record(crimpCalculator.schedule_mapping);
  const autoCrimp = record(crimpCalculator.auto_crimp);
  const autoResults = record(autoCrimp.results);
  const l2Results = records(record(autoResults.l2).segments);
  const l2Mapping = records(mapping.l2Segments);
  const crimpRows = [];
  const dimensionRow = (name, result, mapped) => {
    const values = record(result);
    if (!Object.keys(values).length) return;
    const requirement = values.requiredMinimumMm === undefined
      ? values.formulaOrLookup || "Documented calculator result"
      : `≥ ${millimetres(values.requiredMinimumMm)}${values.formulaOrLookup ? ` — ${values.formulaOrLookup}` : ""}`;
    crimpRows.push([
      name,
      measurementWithSource(values.designValueMm, mapped),
      requirement,
      automaticCrimpDisplayStatus(values.status || autoCrimp.overallStatus),
    ]);
  };
  dimensionRow("Automatic-Crimp L1", autoResults.l1, mapping.l1);
  l2Results.forEach((segment, index) => dimensionRow(`Automatic-Crimp L2.${index + 1}`, segment, l2Mapping[index]));
  dimensionRow("Automatic-Crimp L3", autoResults.l3, mapping.l3);
  dimensionRow("Automatic-Crimp L4", autoResults.l4, mapping.l4);
  if (Object.keys(mapping).length) {
    const l3 = record(mapping.l3);
    const theta1 = record(mapping.theta1);
    const theta4 = record(mapping.theta4);
    if (Object.keys(theta1).length) crimpRows.push(["Schedule Angle θ1", measurementWithSource(theta1.valueDeg, theta1, degrees), "Input to the documented L1 calculator rule", automaticCrimpDisplayStatus(autoCrimp.overallStatus)]);
    if (Object.keys(theta4).length) crimpRows.push(["Schedule Angle θ4", measurementWithSource(theta4.valueDeg, theta4, degrees), "Input to the documented L3/L4 calculator rules", automaticCrimpDisplayStatus(autoCrimp.overallStatus)]);
    if (Object.keys(l3).length && l3.formula) crimpRows.push(["L3 Derivation", millimetres(l3.l3Mm), l3.formula, automaticCrimpDisplayStatus(autoCrimp.overallStatus)]);
  }
  if (crimpRows.length) groups.push({ title: "Automatic-Crimp Geometry", criteria: [byId.get("automatic_crimp_dimensions")], rows: crimpRows });

  const robot = record(record(byId.get("robot_pickup")?.calculator).robot_pickup);
  const robotRows = [];
  for (const [label, key] of [["Robot-Pickup L1 Branch", "l1Branch"], ["Robot-Pickup L2 Branch", "l2Branch"]]) {
    const branch = record(robot[key]);
    if (!Object.keys(branch).length) continue;
    robotRows.push([
      label,
      millimetres(branch.designValueMm),
      `≥ ${millimetres(branch.requiredMinimumMm)}`,
      branch.status || robot.status,
    ]);
  }
  if (robotRows.length) groups.push({ title: "Robot-Pickup Geometry", criteria: [byId.get("robot_pickup")], rows: robotRows });
  return groups;
}

function appendCalculationTrace(host, criteria) {
  const groups = calculationGroups(criteria);
  if (!groups.length) return;
  const disclosure = element("details", "", "drawing-feasibility-calculations");
  const traceSummary = element("summary");
  traceSummary.append(aiAttributedLabel(
    "h4",
    "Geometry and Plant A calculation trace",
    "drawing-feasibility-ai-heading",
    "AI-extracted drawing inputs and calculator-derived Plant A results",
  ));
  disclosure.append(traceSummary);
  for (const group of groups) {
    const section = element("details", "", "drawing-feasibility-calculation-group");
    const groupSummary = element("summary");
    const groupHeading = element("h5", group.title);
    const result = summaryStatus(group.criteria.filter(Boolean)) || "Cannot assess";
    const tag = assessmentStatus(result, `${group.title} Result`);
    tag.className = "drawing-feasibility-calculation-group-status";
    tag.interactive = false;
    tag.setAttribute("wrapping-type", "Normal");
    // Native summary owns keyboard/expanded semantics; expose status as text
    // without relying on read-only UI5 tag shadow-DOM ARIA forwarding.
    tag.setAttribute("aria-hidden", "true");
    groupHeading.append(tag, element("span", ` — Result: ${result}`, "drawing-feasibility-status-description"));
    groupSummary.append(groupHeading);
    section.append(groupSummary);
    const calculationTable = table(
      ["Measurement", "Drawing / Calculated Value", "Plant A Calculation Requirements", "Result"],
      group.rows.map(([measurement, value, requirement, status]) => [
        measurement,
        value,
        requirement,
        assessmentStatus(status || "Cannot assess", `${measurement} Calculation Result`),
      ]),
    );
    calculationTable.classList.add("material-evidence-table", "drawing-feasibility-calculation-table");
    calculationTable.tabIndex = 0;
    calculationTable.setAttribute("role", "region");
    calculationTable.setAttribute("aria-label", `${group.title} Plant A Calculation Trace.`);
    markAiAttributedColumn(calculationTable, 1, "AI-extracted drawing input or calculator-derived value");
    section.append(calculationTable);
    disclosure.append(section);
  }
  host.append(disclosure);
}

export function assessmentDesign(result) {
  if (result === "Feasible against the assessed PS4 requirements" || result === "Pass" || result === "Met") return "Positive";
  if (result === "Warning") return "Critical";
  if (result === "Input") return "Neutral";
  if (result === "Not feasible against the PS4 requirements" || result === "Fail" || result === "Not met") return "Negative";
  if (result === "Configuration-dependent") return "Information";
  return "Critical";
}

function assessmentStatus(result, label, size = "S", text = result) {
  const status = document.createElement("ui5-tag");
  status.setAttribute("design", assessmentDesign(result));
  status.setAttribute("size", size);
  status.setAttribute("aria-label", `${label}: ${result}`);
  status.textContent = text;
  return status;
}

function summaryStatus(criteria) {
  const priority = { Pass: 1, Met: 1, Warning: 2, "Cannot assess": 3, "Not met": 4, Fail: 5 };
  if (!criteria.length) return "Cannot assess";
  return criteria.reduce((selected, criterion) => (
    (priority[criterion.status] || 0) > (priority[selected] || 0) ? criterion.status : selected
  ), criteria[0].status);
}

function titleCase(value) {
  const text = String(value || "");
  return text ? `${text[0].toUpperCase()}${text.slice(1)}` : "Feature";
}

function directMaterialMessage(criterion) {
  const calculator = record(criterion?.calculator);
  const material = record(calculator.raw_material);
  const derived = record(calculator.material_designation);
  const designation = material.designation || derived.candidate;
  if (criterion?.status === "Pass") return designation ? `${designation} is listed for PS4.` : "Material designation is listed for PS4.";
  if (criterion?.status === "Cannot assess" && designation) return `${designation} is not listed; suitability cannot be confirmed.`;
  return "Material designation cannot be confirmed.";
}

function directLengthMessage(criterion) {
  const calculation = record(record(criterion?.calculator).tube_length);
  const range = millimetreRange(calculation.minimumMm, calculation.maximumMm);
  const length = millimetres(calculation.totalLengthMm);
  if (criterion?.status === "Pass" && length !== "Not available" && range !== "Not available") return `${length} meets the documented ${range} range.`;
  if (criterion?.status === "Fail" && length !== "Not available" && range !== "Not available") return `${length} does not meet the documented ${range} range.`;
  return "Tube length cannot be assessed.";
}

function directBendMessage(criterion) {
  const results = records(record(record(criterion?.calculator).bend_angles).results);
  if (criterion?.status === "Pass") return results.length ? `All ${results.length} extracted bend angle${results.length === 1 ? "" : "s"} meet PS4 limits.` : "All extracted bend angles meet PS4 limits.";
  if (criterion?.status === "Fail") return "One or more bend angles fail PS4 limits.";
  return "Bend angles cannot be assessed.";
}

function directStraightLengthMessage(criterion) {
  const results = record(record(record(criterion?.calculator).auto_crimp).results);
  const failed = [["L1", results.l1], ["L3", results.l3], ["L4", results.l4]]
    .filter(([, value]) => record(value).status === "Fail")
    .map(([name]) => name);
  const l2Failures = records(record(results.l2).segments)
    .filter(segment => segment.status === "Fail")
    .map(segment => `L2.${segment.segmentIndex || "?"}`);
  const failedMeasurements = [...failed, ...l2Failures];
  if (criterion?.status === "Pass") return "All mapped straight lengths meet PS4 minimums.";
  if (criterion?.status === "Fail") return failedMeasurements.length ? `${failedMeasurements.join(", ")} ${failedMeasurements.length === 1 ? "is" : "are"} below the PS4 minimum.` : "One or more straight lengths are below the PS4 minimum.";
  return "Automatic-crimp geometry cannot be fully assessed.";
}

function automaticCrimpDisplayStatus(status) {
  return status === "Pass" || status === "Fail" ? status : "Warning";
}

function automaticCrimpStatus(criterion) {
  return automaticCrimpDisplayStatus(criterion?.status);
}

/** Build concise, grouped decision rows from detailed deterministic PS4 criteria. */
export function assessmentSummary(criteria) {
  const byId = new Map(criteria.map(criterion => [criterion.id, criterion]));
  const prohibited = criteria.filter(criterion => criterion.id.startsWith("prohibited_"));
  const failedFeatures = prohibited
    .filter(criterion => criterion.status === "Fail")
    .map(criterion => titleCase(criterion.title.replace(/^Prohibited feature:\s*/i, "")));
  const prohibitedStatus = summaryStatus(prohibited);
  const end = byId.get("end_configuration");
  const endWarning = end?.status === "Warning" && end.explanation
    ? end.explanation
    : "Configuration-dependent end compatibility.";
  const robotGating = byId.get("robot_pickup_gating");
  const robot = robotGating || byId.get("robot_pickup");
  return [
    {
      title: "Material Designation",
      status: byId.get("raw_material")?.status || "Cannot assess",
      message: directMaterialMessage(byId.get("raw_material")),
    },
    {
      title: "Tube Length",
      status: byId.get("tube_length")?.status || "Cannot assess",
      message: directLengthMessage(byId.get("tube_length")),
    },
    {
      title: "End Type",
      status: end?.status || "Cannot assess",
      message: end?.status === "Pass" ? "Supported end configuration specified."
        : end?.status === "Warning" ? endWarning
          : end?.status === "Fail" ? "Unsupported end configuration specified."
            : "End configuration cannot be assessed.",
    },
    {
      title: "Prohibited Features",
      status: prohibitedStatus,
      message: prohibitedStatus === "Pass" ? "No prohibited features found in the completed drawing."
        : prohibitedStatus === "Warning" ? "See End type for the compatible and non-compatible end options."
          : prohibitedStatus === "Fail" && failedFeatures.length ? `${failedFeatures.join("; ")} specified.`
            : "One or more prohibited features cannot be confirmed.",
    },
    {
      title: "Bend Angles",
      status: byId.get("bend_angles")?.status || "Cannot assess",
      message: directBendMessage(byId.get("bend_angles")),
    },
    {
      title: "Minimum Straight Lengths",
      status: automaticCrimpStatus(byId.get("automatic_crimp_dimensions")),
      message: directStraightLengthMessage(byId.get("automatic_crimp_dimensions")),
    },
    {
      title: "Automatic-Crimp Objective",
      status: automaticCrimpStatus(byId.get("automatic_crimp_objective")),
      message: byId.get("automatic_crimp_objective")?.status === "Pass" ? "All assessed automatic-crimp geometry meets PS4 minimums."
        : byId.get("automatic_crimp_objective")?.status === "Fail" ? "One or more automatic-crimp dimensions does not meet PS4 minimums."
          : "Automatic-crimp geometry cannot be fully assessed.",
    },
    {
      title: "Robot Pickup",
      status: robot?.status || "Cannot assess",
      message: robotGating?.status === "Fail" ? "Robot pickup is required but the condition is not met."
        : robotGating?.status === "Cannot assess" ? "Robot pickup is required but cannot be assessed."
          : robot?.status === "Met" ? "Robot-pickup condition is met."
            : robot?.status === "Not met" ? "Robot-pickup condition is not met."
              : "Robot-pickup condition cannot be assessed.",
    },
  ];
}

/** Return one concise rationale selected from the deterministic assessment output. */
export function assessmentConclusion(result, summary) {
  if (result === "Not feasible against the PS4 requirements") {
    const failure = summary.find(item => item.status === "Fail") || summary.find(item => item.status === "Not met");
    return failure ? `Not feasible: ${failure.message}` : "Not feasible against the assessed PS4 requirements.";
  }
  if (result === "Inconclusive") {
    const unknown = summary.find(item => item.status === "Cannot assess");
    return unknown ? `Inconclusive: ${unknown.message}` : "Inconclusive against the assessed PS4 requirements.";
  }
  if (result === "Configuration-dependent") return "Select a configuration to view its PS4 decision summary.";
  if (result !== "Feasible against the assessed PS4 requirements") return "Overall manufacturing feasibility is unknown; no recognized assessment result was supplied.";
  const warning = summary.find(item => item.status === "Warning");
  return warning ? `Warning: ${warning.message}` : "All assessed PS4 feasibility checks pass.";
}

/** Present canonical results only; never derive an overall verdict from check rows. */
function updateOverallStatus(host, result, configuration = "") {
  const labels = {
    "Feasible against the assessed PS4 requirements": "Feasible",
    "Not feasible against the PS4 requirements": "Not feasible",
    Inconclusive: "Inconclusive",
    "Configuration-dependent": "Configuration-dependent",
    "Not assessed": "Not assessed",
    "Analyzing drawing…": "Analyzing…",
    "Assessment unavailable": "Assessment unavailable",
  };
  const text = Object.hasOwn(labels, result) ? labels[result] : "Unknown";
  const meaning = `Overall manufacturing feasibility${configuration ? ` for ${configuration}` : ""}: ${text === "Unknown" ? "Unknown — no recognized assessment result supplied" : result}. This technical assessment does not replace manufacturing approval.`;
  const tag = assessmentStatus(result, "Overall manufacturing feasibility", "S", text);
  tag.className = "drawing-feasibility-overall-tag";
  tag.interactive = false;
  tag.setAttribute("wrapping-type", "Normal");
  tag.setAttribute("aria-hidden", "true");
  tag.title = meaning;
  // Read-only UI5 tags have no focusable ARIA forwarding API. The live region
  // supplies the full scoped meaning as text, independently of the shadow DOM.
  const accessibleText = element("span", meaning, "drawing-feasibility-status-description");
  host.replaceChildren(tag, accessibleText);
}

function renderDrawingReport(host, report, overallStatus) {
  const overallSummary = element("p", "", "drawing-feasibility-assessment-summary");
  const content = element("div", "", "drawing-feasibility-report");
  content.append(overallSummary);
  const reports = Array.isArray(report?.candidate_reports) ? report.candidate_reports : [];
  const candidateHost = element("div", "", "drawing-feasibility-candidates");
  const renderCandidate = index => {
    const candidate = reports[index];
    candidateHost.replaceChildren();
    if (!candidate) {
      updateOverallStatus(overallStatus, "Not assessed");
      overallSummary.textContent = "Not assessed: No drawing configuration report is available.";
      return;
    }
    const criteria = Array.isArray(candidate.criteria) ? candidate.criteria : [];
    updateOverallStatus(overallStatus, criteria.length ? candidate.result : "Not assessed", candidate.label);
    const summary = assessmentSummary(criteria);
    overallSummary.textContent = criteria.length
      ? assessmentConclusion(candidate.result, summary)
      : "Not assessed: No assessment criteria are available for this configuration.";
    const summaryTable = table(["PS4 Check", "Result", "Key Finding"], summary.map(summary => [
      summary.title,
      assessmentStatus(summary.status, `${summary.title} Result`),
      summary.message,
    ]));
    markAiAttributedColumn(summaryTable, 1, "AI-assisted PS4 result");
    markAiAttributedColumn(summaryTable, 2, "AI-assisted PS4 key finding");
    summaryTable.classList.add("drawing-feasibility-summary-table");
    summaryTable.setAttribute("role", "region");
    summaryTable.setAttribute("aria-label", `PS4 Decision Summary for ${candidate.label}.`);
    candidateHost.append(summaryTable);
    appendCalculationTrace(candidateHost, criteria);
  };
  if (reports.length > 1) {
    const label = element("label", "Configuration");
    const selector = document.createElement("select"); selector.setAttribute("aria-label", "Drawing Configuration");
    reports.forEach((candidate, index) => {
      const option = document.createElement("option"); option.value = String(index); option.textContent = candidate.label; selector.append(option);
    });
    selector.addEventListener("change", () => renderCandidate(Number(selector.value)));
    label.append(selector); content.append(label);
  }
  content.append(candidateHost); renderCandidate(0); host.replaceChildren(content);
}

function appendDrawingFeasibility(body, signal, onError, automaticReport = null, drawingSelection) {
  const section = element("section", "", "material-story-block drawing-feasibility");
  const heading = aiAttributedLabel("h3", "Plant A Manufacturing Feasibility", "drawing-feasibility-ai-heading", "AI-assisted drawing assessment");
  const overallStatus = element("span", "", "drawing-feasibility-overall-status");
  overallStatus.setAttribute("role", "status");
  overallStatus.setAttribute("aria-live", "polite");
  overallStatus.setAttribute("aria-atomic", "true");
  updateOverallStatus(overallStatus, "Not assessed");
  heading.append(overallStatus);
  section.append(heading);
  if (automaticReport) {
    const report = element("div", "", "drawing-feasibility-report-host");
    section.append(report);
    renderDrawingReport(report, automaticReport, overallStatus);
    body.append(section);
    return () => {};
  }
  const input = drawingSelection.input;
  if (!input) { body.append(section); return () => {}; }
  section.append(element("p", "Analyze the drawing selected above for PS4 technical feasibility.", "helper-text"));
  const form = element("div", "", "drawing-feasibility-form");
  const submit = button("Analyze Drawing", () => void assess());
  const status = element("p", "", "helper-text"); status.setAttribute("role", "status");
  const error = element("p", ""); error.setAttribute("role", "alert");
  const report = element("div", "", "drawing-feasibility-report-host");
  form.append(submit); section.append(form, status, error, report); body.append(section);
  let disposed = false, assessing = false;
  const resetReport = () => {
    error.textContent = ""; status.textContent = ""; report.replaceChildren();
    updateOverallStatus(overallStatus, "Not assessed");
  };
  input.addEventListener("change", resetReport);
  async function assess() {
    if (disposed || assessing) return;
    const file = drawingSelection.getSelectedFile();
    error.textContent = ""; report.replaceChildren();
    updateOverallStatus(overallStatus, "Not assessed");
    if (!file) { error.textContent = "Select a drawing file before starting the assessment."; return; }
    assessing = true; input.disabled = true; submit.disabled = true; status.textContent = "Analyzing drawing…";
    updateOverallStatus(overallStatus, "Analyzing drawing…");
    try {
      const response = await api.assessDrawing(file, signal);
      if (signal.aborted || disposed) return;
      status.textContent = "Assessment complete.";
      renderDrawingReport(report, response, overallStatus);
    } catch (failure) {
      if (failure.name === "AbortError" || disposed) return;
      status.textContent = "";
      updateOverallStatus(overallStatus, "Assessment unavailable");
      error.textContent = failure.message || "Drawing assessment could not be completed.";
      onError(failure);
    } finally {
      assessing = false;
      if (!signal.aborted && !disposed) { input.disabled = false; submit.disabled = false; }
    }
  }
  return () => { disposed = true; input.removeEventListener("change", resetReport); };
}

function appendMaterialBrief(body, obj, inspect, signal, onError, loadDrawing) {
  const description = obj.description || obj.types?.join(", ") || "Material Description Not Supplied";
  const title = element("h2", description); title.id = "material-dialog-title";
  body.append(title, element("p", `Material: ${obj.number || "Unnumbered object"}`, "material-identifier"));

  if (obj.change) {
    const action = {
      PHASE_IN: "Phase in this material",
      PHASE_OUT: "Phase out this material",
      ADDED: "Add this material",
      REMOVED: "Remove this material",
    }[obj.change.action] || `${readable(obj.change.action)} this material`;
    const change = element("section", "", "material-story-block"); change.append(element("h3", "ECN Change"));
    const summary = element("p", action);
    const counterparts = obj.change.counterpart_details || (obj.change.counterparts || []).map(number => ({ number }));
    if (counterparts.length) {
      summary.append(document.createTextNode(obj.change.action === "PHASE_OUT" ? "; replaced by " : "; replaces "));
      counterparts.forEach((counterpart, index) => {
        if (index) summary.append(document.createTextNode(", "));
        const label = counterpart.description ? `${counterpart.description} (${counterpart.number})` : counterpart.number;
        summary.append(button(label, () => void inspect(counterpart.number)));
      });
    }
    if (obj.change.revision_from && obj.change.revision_to && obj.change.revision_from !== obj.change.revision_to) {
      summary.append(document.createTextNode(`; revision ${obj.change.revision_from} to ${obj.change.revision_to}`));
    }
    summary.append(document.createTextNode(".")); change.append(summary);
    if (obj.change.usage?.label) change.append(element("p", obj.change.usage.label, "helper-text"));
    if (obj.change.pairing_label) change.append(element("p", obj.change.pairing_label, "helper-text"));
    body.append(change);
  }

  appendBomCoverage(body, obj);

  const sourcing = obj.material_brief?.sourcing || [];
  const sourcingNames = buildPlantNameIndex(sourcing);
  const section = element("section", "", "material-story-block"); section.append(element("h3", "Procurement Permission"));
  if (sourcing.length) {
    /* MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: material-dialog price table.
    section.append(table(["Plant", "Permitted", "Make cost", "Buy cost"], sourcing.map(entry => {
      const mode = procurementMode(entry.procurement_type_labels);
      return [
        entry.plant_name ? `${entry.plant_name} (${entry.plant_code})` : entry.plant_code || "Plant not supplied",
        mode.label,
        mode.make ? "Not supplied" : "Not allowed",
        mode.buy ? priceLabel(entry.purchase_prices) : "Not allowed",
      ];
    })));
    MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: material-dialog price table. */
    const procurementTable = table(["Plant", "Permitted", "Lead Time", "Source Purchase Price"], sourcing.map(entry => {
      const mode = procurementMode(entry.procurement_type_labels);
      return [
        plantLabel(entry.plant_code, sourcingNames),
        mode.label,
        sourceLeadTimeLabel(entry.source_lead_time),
        mode.buy ? sourcePurchasePriceLabel(entry.source_purchase_price) : "Not Allowed",
      ];
    }));
    procurementTable.classList.add("material-evidence-table", "material-procurement-table");
    procurementTable.tabIndex = 0;
    procurementTable.setAttribute("role", "region");
    procurementTable.setAttribute("aria-label", "Source-backed procurement permissions. Scroll for additional columns or rows.");
    section.append(procurementTable);
  } else {
    section.append(element(
      "p",
      "No source-backed procurement permission is available for this material.",
      "helper-text",
    ));
  }
  body.append(section);
  // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: return appendMakeBuyComparison(body, obj.material_brief?.make_buy_assessments || []);
  const drawingSelection = engineeringDrawingPreview(body, signal, { drawing: obj.engineering_drawing, loadDrawing });
  const disposeFeasibility = appendDrawingFeasibility(body, signal, onError, obj.manufacturing_feasibility, drawingSelection);
  return () => { disposeFeasibility(); drawingSelection.dispose(); };
}

/** Mount an accessible native material dialog. */
export function materialDialog(id, signal, onError) {
  const dialog = element("dialog", "", "object-dialog"); dialog.setAttribute("aria-labelledby", "material-dialog-title"); document.body.append(dialog);
  const requests = createLatestRequestGate(signal);
  let opener, disposePreview = () => {};
  let body = element("div", "", "object-dialog-body");
  const footer = element("footer", "", "object-dialog-footer");
  const cancel = () => { requests.cancel(); disposePreview(); };
  footer.append(button("Close", () => { cancel(); dialog.close(); }));
  dialog.append(body, footer);
  // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: dialog.querySelectorAll(".make-buy-radar").forEach(disposeChart);
  const dispose = () => { requests.dispose(); disposePreview(); dialog.remove(); };
  signal.addEventListener("abort", dispose, { once: true });
  if (signal.aborted) dispose();
  dialog.addEventListener("cancel", cancel);
  dialog.addEventListener("close", () => {
    // Native close events are queued; ignore one belonging to an earlier opening.
    if (dialog.open) return;
    cancel();
    if (!signal.aborted && opener?.isConnected) opener.focus();
  });
  function replaceBody(nextBody) {
    const moveFocus = body.contains(document.activeElement) || document.activeElement === dialog;
    body.replaceWith(nextBody); body = nextBody;
    body.scrollTop = 0;
    const title = body.querySelector("h2"); title.tabIndex = -1;
    if (moveFocus) (body.querySelector(".material-acceptance-issue") || title).focus({ preventScroll: true });
    return title;
  }
  function stateBody(number) {
    const state = element("div", "", "object-dialog-body");
    const title = element("h2", "Material details"); title.id = "material-dialog-title";
    state.append(title, element("p", `Material: ${number}`, "material-identifier"));
    return state;
  }
  /** Read one material and preserve its original opening control across counterpart navigation. */
  async function inspect(number, technical = true, { acceptanceIssue, propagateError = false } = {}) {
    if (signal.aborted) return;
    const request = requests.next();
    if (!dialog.open) opener = document.activeElement;
    disposePreview(); disposePreview = () => {};
    const loading = stateBody(number); loading.setAttribute("aria-busy", "true");
    const status = element("p", "Loading material details…", "material-dialog-state helper-text");
    status.setAttribute("role", "status"); loading.append(status);
    const title = replaceBody(loading);
    if (!dialog.open) { dialog.showModal(); title.focus({ preventScroll: true }); }
    try {
      const obj = await api.getObject(id, { number, technical, page: 1, page_size: 25 }, request.signal);
      if (!request.isCurrent() || !dialog.open) return;
      const details = element("div", "", "object-dialog-body");
      disposePreview = appendMaterialBrief(details, obj, inspect, request.signal, onError,
        () => api.getEngineeringDrawing(id, obj.number, request.signal));
      if (acceptanceIssue?.number === obj.number) {
        const issue = acceptanceIssueSection(acceptanceIssue);
        details.querySelector(".material-identifier").after(issue);
      }
      // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: dialog.querySelectorAll(".make-buy-radar").forEach(disposeChart);
      replaceBody(details);
      // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: charts.forEach(({ chart, scores }) => initChart(chart, makeBuyRadarOption(scores)));
    } catch (error) {
      if (!request.isCurrent() || !dialog.open || error.name === "AbortError") return;
      disposePreview(); disposePreview = () => {};
      const failed = stateBody(number);
      const message = element("p", `Could not load material details. ${error.message || "Try again."}`, "material-dialog-state");
      message.setAttribute("role", "alert");
      failed.append(message, button("Retry", () => void inspect(number, technical, { acceptanceIssue })));
      replaceBody(failed);
      onError(error);
      if (propagateError) throw error;
    }
  }
  return inspect;
}
