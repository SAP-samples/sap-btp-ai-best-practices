import { button, element } from "./workspace-dom.js";
import { plantsLabel } from "./plant-impact-table.js";

/** Join the server's exact blocker identities to source-backed material context. */
export function blockedMaterialIssues(plan, snapshot) {
  const parts = new Map((plan?.parts || []).map(part => [part.number, part]));
  const numbers = [...new Set([
    ...(plan?.missing_price || []),
    ...(plan?.parts || []).filter(part => part.price === "MISSING").map(part => part.number),
  ])].filter(Boolean);
  return numbers.map(number => {
    const part = parts.get(number) || {};
    const material = snapshot?.bom_changes?.materials?.find(item => item.number === number);
    const lines = (snapshot?.bom_changes?.lines || []).filter(line => line.incoming?.number === number);
    const plants = (snapshot?.plant_impact?.plants || []).filter(plant =>
      plant.materials?.some(item => item.number === number));
    const contexts = new Map();
    for (const plant of [...plants, ...lines.flatMap(line => line.plants || [])]) {
      const key = plant.plant_code || plant.plant_name || "unknown";
      const existing = contexts.get(key);
      contexts.set(key, { ...existing, ...plant, plant_name: existing?.plant_name || plant.plant_name });
    }
    return {
      number,
      description: part.description || material?.description || lines[0]?.incoming?.description || "Description not supplied",
      plants: plantsLabel([...contexts.values()]),
      assemblies: [...new Set(lines.map(line => line.assembly_description || line.assembly).filter(Boolean))],
      masterData: part.master_data || "Material master action not supplied",
      required: `Add a purchase-info-record entry for ${number} to the purchasing source used by this ECN.`,
    };
  });
}

/** Open an exact blocked material with its task, reporting read errors in place. */
export function acceptanceMaterialAction(issue, inspect) {
  const wrapper = element("span", "", "acceptance-material-action");
  const error = element("span", "", "acceptance-material-error");
  error.setAttribute("role", "alert");
  const action = button(issue.number, async () => {
    error.textContent = "";
    try {
      const request = inspect(issue.number, true, { acceptanceIssue: issue, propagateError: true });
      await request;
    } catch (failure) {
      if (action.isConnected && failure.name !== "AbortError") {
        error.textContent = `Could not open ${issue.number}. ${failure.message || "Try again."}`;
      }
    }
  });
  wrapper.append(action, error);
  return wrapper;
}

/** Keep the exact acceptance task visible at the top of the opened material. */
export function acceptanceIssueSection(issue) {
  const section = element("section", "", "material-acceptance-issue");
  section.tabIndex = -1;
  section.setAttribute("aria-labelledby", "material-acceptance-title");
  const title = element("h3", "Required before ECN acceptance");
  title.id = "material-acceptance-title";
  section.append(title, element("p", issue.required), element("p", `Affected plants: ${issue.plants}`));
  if (issue.assemblies.length) section.append(element("p", `Parent assemblies: ${issue.assemblies.join(", ")}`));
  section.append(element("p", `Material master: ${issue.masterData}`));
  section.append(element("p", "Add this entry to the Purchase Info Records workbook, publish it in Settings → Reference Datasets, then import this ECN again. The current material details are read-only.", "helper-text"));
  const sourceLink = element("a", "Open Purchase Info Records");
  sourceLink.href = "/settings?source=PURCHASE_INFO_RECORDS";
  section.append(sourceLink);
  return section;
}
