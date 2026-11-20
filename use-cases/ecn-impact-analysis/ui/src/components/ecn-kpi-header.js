// Keep decision colors local so the summary header does not load the deferred
// charting bundle. Detailed chart rendering remains in ecn-charts.js.
const C = Object.freeze({
  positive: "#256f3a",
  positiveLight: "#eaf6ed",
  negative: "#aa0808",
  negativeLight: "#ffeaf4",
  critical: "#8d5e00",
  criticalLight: "#fff3e1",
  neutral: "#475e75",
  neutralLight: "#eef1f4",
  border: "#e5e5e5",
  label: "#556b82",
});

/**
 * Compute decision signals from the existing snapshot. No extra API calls are needed.
 */
export function computeKpis(snapshot) {
  const plants = snapshot.plant_impact?.plants || [];
  const lines = snapshot.bom_changes?.lines || [];
  const materials = snapshot.bom_changes?.materials || [];
  // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: const schedule = snapshot.phase_schedule || [];

  const confirmedPlants = plants.filter((p) => p.status === "CONFIRMED").length;
  const checkPlants = plants.length - confirmedPlants;

  const missingPriceParts = new Set(
    plants.flatMap((p) => (p.new_parts_without_purchasing || []).map((n) => n.number || n)),
  );
  const pricedParts = new Set(
    plants.flatMap((p) => (p.new_parts_with_purchasing || []).map((n) => n.number || n)),
  );
  // Subtract parts that are priced from missing count
  for (const n of pricedParts) missingPriceParts.delete(n);
  const totalNewParts = missingPriceParts.size + pricedParts.size;
  const missingPrice = missingPriceParts.size;

  const mbomGaps = lines.filter(
    (l) => l.qualifiers?.includes("MBOM_NOT_UPDATED"),
  ).length;
  const totalLines = lines.length;

  const phaseOutRisk = materials.filter((m) => m.status === "STILL_USED").length;
  // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: const unconfirmedTransitions = schedule.filter((item) => !item.planner_confirmed).length;
  const undatedPlants = plants.filter((plant) => !plant.cutover_from).length;
  const procurementTypes = new Map();
  for (const plant of plants) {
    for (const material of plant.materials || []) {
      if (!material.procurement_types?.length) continue;
      const types = procurementTypes.get(material.number) || new Set();
      material.procurement_types.forEach((type) => types.add(type));
      procurementTypes.set(material.number, types);
    }
  }
  const makeBuyChanges = lines.filter((line) => {
    const outgoing = [...(procurementTypes.get(line.outgoing?.number) || [])];
    const incoming = [...(procurementTypes.get(line.incoming?.number) || [])];
    return outgoing.length === 1 && incoming.length === 1 && outgoing[0] !== incoming[0];
  }).length;

  return {
    plants: {
      confirmed: confirmedPlants,
      check: checkPlants,
      total: plants.length,
      names: plants.map((plant) => [plant.plant_code, plant.plant_name].filter(Boolean).join(" ") || "Plant to confirm"),
    },
    sourcing: { priced: pricedParts.size, missing: missingPrice, total: totalNewParts, makeBuyChanges },
    mbom: { aligned: totalLines - mbomGaps, gaps: mbomGaps, total: totalLines },
    phaseOut: { risk: phaseOutRisk },
    manufacturing: {
      // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: scheduled: schedule.length,
      // MOCK_DATA_RETIREMENT_VALIDATION_DISABLED: unconfirmed: unconfirmedTransitions,
      undatedPlants,
    },
  };
}

/** Tone for a decision signal. */
function signalTone(isOk, isError, neutral = false) {
  if (neutral) return "neutral";
  if (isError) return "error";
  if (isOk) return "ok";
  return "warn";
}

/**
 * Render the five decisions that frame an ECN review.
 * host: the #ecn-kpi-header element inside the DynamicPageHeader.
 * kpis: return value of computeKpis().
 */
export function renderKpiHeader(host, kpis, { onSelect = null } = {}) {
  host.replaceChildren();

  const tiles = [
    {
      id: "plants",
      icon: "map",
      label: "Plants affected",
      primary: kpis.plants.total === 0 ? "No plants" : `${kpis.plants.total} plant${kpis.plants.total !== 1 ? "s" : ""}`,
      secondary:
        kpis.plants.total === 0
          ? "No plant data yet"
          : kpis.plants.names.slice(0, 3).join(" · ") + (kpis.plants.names.length > 3 ? ` +${kpis.plants.names.length - 3} more` : ""),
      tone: signalTone(kpis.plants.check === 0, false),
    },
    {
      id: "sourcing",
      icon: "money-bills",
      label: "Sourcing action",
      primary:
        kpis.sourcing.makeBuyChanges > 0
          ? `${kpis.sourcing.makeBuyChanges} make/buy change${kpis.sourcing.makeBuyChanges !== 1 ? "s" : ""}`
          : kpis.sourcing.total === 0
          ? "No new parts"
          : kpis.sourcing.missing > 0
            ? `${kpis.sourcing.missing} part${kpis.sourcing.missing !== 1 ? "s" : ""} need${kpis.sourcing.missing !== 1 ? "" : "s"} sourcing`
            : "Sourcing records found",
      secondary:
        kpis.sourcing.makeBuyChanges > 0
          ? "Confirm sourcing decision and cost"
          : kpis.sourcing.total === 0
          ? "No purchase setup required"
          : kpis.sourcing.missing === 0
            ? `${kpis.sourcing.priced} new part${kpis.sourcing.priced !== 1 ? "s" : ""} ready for review`
            : "Create sourcing records before cutover",
      tone: signalTone(
        kpis.sourcing.missing === 0 && kpis.sourcing.makeBuyChanges === 0,
        kpis.sourcing.missing > 0 || kpis.sourcing.makeBuyChanges > 0,
      ),
    },
    {
      id: "bom",
      icon: "detail-view",
      label: "MBOM alignment",
      primary:
        kpis.mbom.total === 0
          ? "No BOM changes"
          : `${kpis.mbom.total} BOM change${kpis.mbom.total !== 1 ? "s" : ""}`,
      secondary:
        kpis.mbom.gaps === 0
          ? "All changes in MBOM"
          : `${kpis.mbom.gaps} EBOM-only gap${kpis.mbom.gaps !== 1 ? "s" : ""}`,
      tone: signalTone(kpis.mbom.gaps === 0, false),
    },
    {
      id: "phase-out",
      icon: "alert",
      label: "Phase-out risk",
      primary:
        kpis.phaseOut.risk === 0
          ? "No risk found"
          : `${kpis.phaseOut.risk} part${kpis.phaseOut.risk !== 1 ? "s" : ""} at risk`,
      secondary:
        kpis.phaseOut.risk === 0
          ? "Removed parts not used elsewhere"
          : "Still used in other plants",
      tone: signalTone(kpis.phaseOut.risk === 0, kpis.phaseOut.risk > 0),
    },
    {
      id: "manufacturing",
      icon: "factory",
      label: "Manufacturing timing",
      primary:
        kpis.plants.total === 0
          ? "No plant timing"
          : kpis.manufacturing.undatedPlants === 0
            ? "Cutover dates recorded"
            : `${kpis.manufacturing.undatedPlants} plant${kpis.manufacturing.undatedPlants !== 1 ? "s" : ""} need${kpis.manufacturing.undatedPlants !== 1 ? "" : "s"} timing`,
      secondary:
        kpis.plants.total === 0
          ? "No source-backed plant data"
          : kpis.manufacturing.undatedPlants === 0
            ? "Use the recorded plant cutover dates"
            : "Confirm cutover dates from source evidence",
      tone: signalTone(
        kpis.plants.total > 0 && kpis.manufacturing.undatedPlants === 0,
        kpis.manufacturing.undatedPlants > 0,
        kpis.plants.total === 0,
      ),
    },
  ];

  const toneColors = {
    ok: { bg: C.positiveLight, border: C.positive, text: C.positive, icon: C.positive },
    warn: { bg: C.criticalLight, border: C.critical, text: C.critical, icon: C.critical },
    error: { bg: C.negativeLight, border: C.negative, text: C.negative, icon: C.negative },
    neutral: { bg: C.neutralLight, border: C.border, text: C.neutral, icon: C.neutral },
  };

  for (const tile of tiles) {
    const colors = toneColors[tile.tone] || toneColors.neutral;
    const card = document.createElement("button");
    card.type = "button";
    card.className = "kpi-tile";
    card.setAttribute("aria-label", `${tile.label}: ${tile.primary}. ${tile.secondary}`);
    if (onSelect) card.addEventListener("click", () => onSelect(tile.id));
    card.style.cssText = `
      background: ${colors.bg};
      border: 1px solid ${colors.border};
      border-radius: .75rem;
      padding: .75rem 1rem;
      min-width: 0;
    `;

    const labelEl = document.createElement("p");
    labelEl.className = "kpi-label";
    labelEl.textContent = tile.label;
    labelEl.style.cssText = `margin: 0 0 .2rem; font-size: .7rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; color: ${C.label};`;

    const primaryEl = document.createElement("p");
    primaryEl.className = "kpi-primary";
    primaryEl.textContent = tile.primary;
    primaryEl.style.cssText = `margin: 0 0 .15rem; font-size: 1rem; font-weight: 700; color: ${colors.text};`;

    const secondaryEl = document.createElement("p");
    secondaryEl.className = "kpi-secondary";
    secondaryEl.textContent = tile.secondary;
    secondaryEl.style.cssText = `margin: 0; font-size: .75rem; color: ${C.label};`;

    card.append(labelEl, primaryEl, secondaryEl);
    host.append(card);
  }
}
