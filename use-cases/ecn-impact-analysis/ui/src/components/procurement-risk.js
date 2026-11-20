// MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_START: Procurement / Make-or-Buy renderer.
// This renderer synthesizes make costs and consumes temporary overlay prices and recommendations.
// import { button, element, table } from "./workspace-dom.js";
//
// function plantLabel(plant) {
//   if (!plant) return "Plant not determined";
//   if (plant.plant_name && plant.plant_code) return `${plant.plant_name} (${plant.plant_code})`;
//   return plant.plant_name || plant.plant_code || "Plant not determined";
// }
//
// function materialCell(number, inspect, description = null) {
//   const materialLink = (label) => {
//     if (!inspect) return element("span", label);
//     const link = button(label, () => void inspect(number));
//     link.className = "material-link";
//     return link;
//   };
//   if (!description) return materialLink(number);
//   const cell = element("div", "", "material-cell");
//   cell.append(
//     materialLink(description),
//     element("span", `Material: ${number}`, "material-identifier"),
//   );
//   return cell;
// }
//
// function recordKey(number, plantCode) {
//   return JSON.stringify([number, plantCode || ""]);
// }
//
// function unitCost(amount, currency, pricePer, unit) {
//   if (amount == null || amount === "") return "unavailable";
//   const price = [currency, amount].filter(Boolean).join(" ");
//   const quantity = pricePer && pricePer !== "1" ? `${pricePer} ` : "";
//   return `${price} / ${quantity}${unit || "unit"}`;
// }
//
// /** Generate a stable illustrative make cost where no curated demo estimate exists. */
// function mockMakeCost(materialNumber, plant, purchasePrice) {
//   const key = `${materialNumber}:${plant?.plant_code || plant?.plant_name || ""}`;
//   const seed = [...key].reduce((total, character) => total + character.charCodeAt(0), 0);
//   const purchaseAmount = Number(purchasePrice?.net_price);
//   const multipliers = [0.78, 0.91, 1.07, 1.18];
//   const amount = Number.isFinite(purchaseAmount) && purchaseAmount > 0
//     ? purchaseAmount * multipliers[seed % multipliers.length]
//     : 1 + (seed % 1900) / 100;
//   return {
//     amount: amount.toFixed(2),
//     currency: purchasePrice?.currency || "EUR",
//     pricePer: purchasePrice?.price_per || "1",
//     unit: purchasePrice?.unit_of_measure || "PC",
//   };
// }
//
// function evidenceList(entries, empty) {
//   if (!entries.length) return empty;
//   const list = element("div", "", "procurement-evidence-list");
//   for (const entry of entries) {
//     list.append(element("span", entry));
//   }
//   return list;
// }
//
// function manufacturingBomEvidence(line) {
//   if (line.qualifiers?.includes("MBOM_NOT_UPDATED")) return "PENDING";
//   if (line.levels?.includes("MBOM")) return "FOUND";
//   return null;
// }
//
// function hasPriorManufacturingEvidence(material, bomHistory) {
//   if (bomHistory.get(material.number)?.prior_manufacturing_bom_use) return true;
//   return material.ecnStatuses.has("UNCHANGED")
//     && [...material.manufacturingBomEvidence.values()].some((evidence) => evidence.has("FOUND"));
// }
//
// function materialHistoryStatus(material, bomHistory, purchaseHistoryMaterials) {
//   const bomMaterial = bomHistory.get(material.number);
//   const hasPriorBomUse = bomMaterial?.status === "ALREADY_USED";
//   const hasPurchaseHistory = purchaseHistoryMaterials.has(material.number);
//   const isAlreadyInBom = material.ecnStatuses.has("UNCHANGED");
//   const available = hasPriorBomUse || hasPurchaseHistory || isAlreadyInBom;
//   return element(
//     "span",
//     available ? "Available" : "New",
//     `semantic-pill tone-${available ? "neutral" : "success"} procurement-history-status`,
//   );
// }
//
// /** Render each BOM material with any recorded plant-specific make-or-buy recommendation. */
// export function renderProcurement(host, snapshot, inspect) {
//   host.replaceChildren();
//   if (!snapshot) {
//     host.append(element("p", "Procurement data is unavailable. Reload to try again.", "helper-text"));
//     return;
//   }
//
//   const descriptions = new Map();
//   const addDescription = (part) => {
//     if (part?.number && part.description && !descriptions.has(part.number)) descriptions.set(part.number, part.description);
//   };
//   (snapshot.part_descriptions || []).forEach(addDescription);
//   (snapshot.bom_changes?.materials || []).forEach(addDescription);
//   for (const line of snapshot.bom_changes?.lines || []) {
//     addDescription(line.incoming);
//     addDescription(line.outgoing);
//   }
//
//   const plants = new Map((snapshot.plant_impact?.plants || []).map((plant) => [plant.plant_code, plant]));
//   const prices = new Map();
//   for (const price of snapshot.purchase_prices || []) {
//     if (!price.net_price) continue;
//     const key = recordKey(price.material_number, price.plant_code);
//     const current = prices.get(key);
//     if (!current || price.status === "ACTIVE") prices.set(key, price);
//   }
//   const purchaseHistoryMaterials = new Set([...prices.values()].map((price) => price.material_number));
//   const recommendations = (snapshot.make_buy_recommendations || []).filter((recommendation) => recommendation.material_number).map((recommendation) => ({
//     ...recommendation,
//     description: recommendation.material_description || descriptions.get(recommendation.material_number) || recommendation.material_number,
//     plant: plants.get(recommendation.plant_code) || { plant_code: recommendation.plant_code },
//   })).sort((left, right) => (
//     left.description.localeCompare(right.description)
//     || plantLabel(left.plant).localeCompare(plantLabel(right.plant))
//   ));
//   const curatedMakeCosts = new Map(recommendations
//     .filter((recommendation) => recommendation.make_cost)
//     .map((recommendation) => [
//       recordKey(recommendation.material_number, recommendation.plant_code),
//       recommendation,
//     ]));
//   const materials = new Map();
//   const addMaterial = (number, description, affectedPlants = [], ecnStatus = null, bomEvidence = null) => {
//     if (!number) return;
//     const material = materials.get(number) || {
//       number,
//       description: "",
//       plants: new Map(),
//       recommendations: [],
//       ecnStatuses: new Set(),
//       procurementTypes: new Map(),
//       manufacturingBomEvidence: new Map(),
//     };
//     if (!material.description) material.description = description || descriptions.get(number) || "";
//     if (ecnStatus) material.ecnStatuses.add(ecnStatus);
//     for (const affectedPlant of affectedPlants) {
//       const plant = plants.get(affectedPlant.plant_code) || affectedPlant;
//       const key = plant?.plant_code || plant?.plant_name;
//       if (key) {
//         material.plants.set(key, plant);
//         if (bomEvidence) {
//           const evidence = material.manufacturingBomEvidence.get(key) || new Set();
//           evidence.add(bomEvidence);
//           material.manufacturingBomEvidence.set(key, evidence);
//         }
//       }
//     }
//     materials.set(number, material);
//   };
//   for (const line of snapshot.bom_changes?.lines || []) {
//     const unchanged = line.outgoing?.number && line.outgoing.number === line.incoming?.number;
//     const bomEvidence = manufacturingBomEvidence(line);
//     addMaterial(line.outgoing?.number, line.outgoing?.description, line.plants, unchanged ? "UNCHANGED" : "REMOVED", bomEvidence);
//     addMaterial(line.incoming?.number, line.incoming?.description, line.plants, unchanged ? "UNCHANGED" : "NEW", bomEvidence);
//   }
//   for (const recommendation of recommendations) {
//     addMaterial(recommendation.material_number, recommendation.description);
//     materials.get(recommendation.material_number).recommendations.push(recommendation);
//   }
//   for (const plant of snapshot.plant_impact?.plants || []) {
//     const plantKey = plant.plant_code || plant.plant_name;
//     for (const sourceMaterial of plant.materials || []) {
//       const material = materials.get(sourceMaterial.number);
//       if (!material || !plantKey) continue;
//       const scopedPlant = plants.get(plant.plant_code) || plant;
//       material.plants.set(plantKey, scopedPlant);
//       const types = material.procurementTypes.get(plantKey) || new Set();
//       for (const type of sourceMaterial.procurement_types || []) types.add(type);
//       material.procurementTypes.set(plantKey, types);
//     }
//   }
//   const materialRows = [...materials.values()].sort((left, right) => (
//     (left.description || left.number).localeCompare(right.description || right.number)
//     || left.number.localeCompare(right.number)
//   ));
//   const displayRows = materialRows.filter((material) => (
//     material.ecnStatuses.has("NEW") || material.ecnStatuses.has("UNCHANGED")
//   ));
//   const bomHistory = new Map((snapshot.bom_changes?.materials || []).map((material) => [material.number, material]));
//   const decision = element("section", "", "surface");
//   decision.append(
//     element("h2", "Make or Buy"),
//   );
//   if (!displayRows.length) {
//     decision.append(element("p", "No new or unchanged BOM materials are available for make-or-buy evidence.", "helper-text"));
//     host.append(decision);
//     return;
//   }
//
//   decision.append(table(
//     ["Material", "Affected plants", "Purchasing evidence", "Mock make cost", "Material history", "Decision status"],
//     displayRows.map((material) => {
//       const purchasing = [...material.plants.values()].flatMap((plant) => {
//         const price = prices.get(recordKey(material.number, plant.plant_code));
//         return price ? [`${plantLabel(plant)}: ${unitCost(price.net_price, price.currency, price.price_per, price.unit_of_measure)}`] : [];
//       });
//       const manufacturing = [...material.plants.entries()].flatMap(([key, plant]) => {
//         const evidence = material.manufacturingBomEvidence.get(key) || new Set();
//         if (evidence.has("PENDING")) {
//           return `${plantLabel(plant)}: manufacturing BOM update pending`;
//         }
//         if (!evidence.has("FOUND")) return `${plantLabel(plant)}: no manufacturing BOM record found`;
//         if (!hasPriorManufacturingEvidence(material, bomHistory)) return `${plantLabel(plant)}: No History`;
//         const purchasePrice = prices.get(recordKey(material.number, plant.plant_code));
//         const curated = curatedMakeCosts.get(recordKey(material.number, plant.plant_code));
//         const cost = curated || mockMakeCost(material.number, plant, purchasePrice);
//         return `${plantLabel(plant)}: ${unitCost(cost.make_cost || cost.amount, cost.currency, cost.price_per || cost.pricePer, cost.unit_of_measure || cost.unit)}`;
//       });
//       const types = new Set([...material.procurementTypes.values()].flatMap((values) => [...values]));
//       const canBuy = [...types].some((type) => /external procurement/i.test(type));
//       const canMake = [...types].some((type) => /in-house production|in-house and external procurement allowed/i.test(type));
//       const decisionStatus = canBuy && canMake
//         ? "Make or buy eligible in material master; comparison needed"
//         : canBuy ? "Buy eligible in material master; comparison needed"
//           : canMake ? "Make eligible in material master; comparison needed"
//             : "Decision needs review: no procurement type recorded";
//       return [
//         materialCell(material.number, inspect, material.description),
//         [...material.plants.values()].map(plantLabel).join(", ") || "Plant not determined",
//         evidenceList(purchasing, "No History"),
//         evidenceList(manufacturing, "No manufacturing evidence recorded"),
//         materialHistoryStatus(material, bomHistory, purchaseHistoryMaterials),
//         decisionStatus,
//       ];
//     }),
//   ));
//   host.append(decision);
// }
// MOCK_DATA_RETIREMENT_VALIDATION_DISABLED_END: Procurement / Make-or-Buy renderer.
