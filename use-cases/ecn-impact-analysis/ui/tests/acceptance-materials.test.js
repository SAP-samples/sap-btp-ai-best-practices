import { expect, it } from "vitest";
import { blockedMaterialIssues } from "../src/components/acceptance-materials.js";

it("joins exact blocked identities to their source plants and parent assemblies", () => {
  const plan = { missing_price: ["NEW-2", "NEW-1", "NEW-2"], parts: [
    { number: "NEW-1", description: "Hydraulic pipe", price: "MISSING", master_data: "Create and release in PL10" },
    { number: "OLD-1", description: "Old pipe", price: "NOT_NEEDED" },
    { number: "PRICED", price: "HAS_PRICE" },
  ] };
  const snapshot = { plant_impact: { plants: [
    { plant_code: "PL10", plant_name: "Plant A", materials: [{ number: "NEW-1" }] },
    { plant_code: "PK01", plant_name: "Plant B", materials: [{ number: "OTHER" }] },
  ] }, bom_changes: { lines: [
    { assembly: "A1", assembly_description: "Hydraulic supply", incoming: { number: "NEW-1" }, plants: [{ plant_code: "PL10" }] },
    { assembly: "A1", assembly_description: "Hydraulic supply", incoming: { number: "NEW-1" }, plants: [{ plant_code: "PL10" }] },
    { assembly: "A2", assembly_description: "Steering supply", incoming: { number: "NEW-1" }, plants: [] },
  ] } };
  const issues = blockedMaterialIssues(plan, snapshot);
  expect(issues.map(issue => issue.number)).toEqual(["NEW-2", "NEW-1"]);
  expect(issues[0].plants).toBe("Plant not determined");
  expect(issues[1]).toMatchObject({ description: "Hydraulic pipe", plants: "Plant A (PL10)",
    assemblies: ["Hydraulic supply", "Steering supply"], masterData: "Create and release in PL10" });
  expect(issues[1].required).toContain("purchase-info-record entry for NEW-1");
});

it("uses only explicit blockers without inferring them from prose or unrelated findings", () => {
  expect(blockedMaterialIssues({ block_reason: "4 parts have no price." }, { questions: [{ subjects: [{ number: "OTHER" }] }] })).toEqual([]);
  expect(blockedMaterialIssues({ parts: [{ number: "NEW", price: "MISSING" }] }, null)[0].number).toBe("NEW");
});
