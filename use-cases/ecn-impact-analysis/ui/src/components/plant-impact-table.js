import { element } from "./workspace-dom.js";
import { buildPlantNameIndex, plantLabel } from "../modules/plant-display.js";

const ALIGNMENT_LABELS = {
  ALIGNED: "BOM evidence aligned",
  MBOM_PENDING: "Manufacturing BOM update pending",
  CHECK: "BOM alignment needs review",
};

/** Return "Plant B (PK01), Plant A (PL10)" for rule-derived plants. */
export function plantsLabel(plants, fallback = "Plant not determined") {
  const names = buildPlantNameIndex(plants);
  return (plants || []).map((plant) => plant.plant_code
    ? plantLabel(plant.plant_code, names)
    : plant.plant_name).filter(Boolean).join(", ") || fallback;
}

function alignment(lines) {
  const qualifiers = new Set(lines.flatMap((line) => line.qualifiers || []));
  if (qualifiers.has("MBOM_NOT_UPDATED")) return "MBOM_PENDING";
  if (lines.length && lines.every((line) => ["EBOM", "MBOM"].every((level) => line.levels?.includes(level)))) return "ALIGNED";
  return "CHECK";
}

/** Support stored snapshots created before plant assembly summaries were added. */
function legacyAssemblies(plant, bomChanges) {
  const lines = (bomChanges?.lines || []).filter((line) => plant.plant_code
    ? line.plants?.some((item) => item.plant_code === plant.plant_code)
    : !line.plants?.length);
  const groups = new Map();
  for (const line of lines) {
    const key = `${line.model_range || ""}|${line.assembly || ""}`;
    groups.set(key, [...(groups.get(key) || []), line]);
  }
  return [...groups.values()].map((group) => ({
    change_count: group.length,
    bom_alignment: alignment(group),
  }));
}

function plantLines(plant, bomChanges) {
  return (bomChanges?.lines || []).filter((line) => plant.plant_code
    ? line.plants?.some((item) => item.plant_code === plant.plant_code)
    : !line.plants?.length);
}

/** Render the one-line plant rollup; BOM evidence below supplies the component detail. */
export function renderPlantImpact(host, plantImpact, bomChanges = null) {
  host.replaceChildren();
  const plants = plantImpact?.plants || [];
  if (!plants.length) {
    host.append(element("span", "No affected plant could be determined from the available BOM evidence."));
    return;
  }

  const names = buildPlantNameIndex(plants);
  const rollup = plants.map((plant) => {
    const label = plant.plant_code ? plantLabel(plant.plant_code, names) : plantsLabel([plant]);
    const summaries = plant.assemblies?.length ? plant.assemblies : legacyAssemblies(plant, bomChanges);
    const lines = plantLines(plant, bomChanges);
    if (!summaries.length) return `${label}: BOM scope needs review.`;
    const assemblies = `${summaries.length} affected assembl${summaries.length === 1 ? "y" : "ies"}`;
    const changeCount = summaries.reduce((total, summary) => total + summary.change_count, 0);
    const changes = `${changeCount} component change${changeCount === 1 ? "" : "s"}`;
    const reviewCount = lines.length
      ? lines.filter((line) => alignment([line]) !== "ALIGNED").length
      : summaries.filter((summary) => summary.bom_alignment !== "ALIGNED").length;
    const outcome = reviewCount
      ? `${reviewCount} BOM row${reviewCount === 1 ? "" : "s"} needs alignment review`
      : ALIGNMENT_LABELS.ALIGNED;
    return `${label}: ${assemblies}, ${changes}; ${outcome}.`;
  });
  host.append(element("p", rollup.join(" "), "plant-impact-rollup"));
}
