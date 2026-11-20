import { expect, it, vi } from "vitest";
import { buildIdentifierIndex, describePartIdentifiers, linkifyIdentifiers } from "../src/components/identifier-glossary.js";

const snapshot = {
  bom_changes: {
    materials: [{ number: "916100410830", description: "Suction hose, hydraulic" }],
    lines: [{
      assembly: "G916100410040", model_range: "Q900V",
      incoming: { number: "501150040650", description: "Bracket" }, outgoing: null,
    }],
  },
  plant_impact: { plants: [{ materials: [{ number: "QKX2833570", description: "Filter" }] }] },
  part_descriptions: [{ number: "UNPLACED1", description: "Flexible shim ring" }],
  programs: [{ model_range: "Q938V", ebom: [{ number: "QKX463881A", description: "Valve", parent: "Q923VV2526401_Q900" }], mbom: [] }],
};

it("indexes parts (clickable) and assemblies/model ranges (annotated), parts winning ties", () => {
  const index = buildIdentifierIndex(snapshot);
  expect(index.get("916100410830")).toMatchObject({ kind: "part", clickable: true, description: "Suction hose, hydraulic" });
  expect(index.get("501150040650")).toMatchObject({ kind: "part", clickable: true });
  expect(index.get("QKX2833570").kind).toBe("part");
  expect(index.get("QKX463881A").kind).toBe("part");
  expect(index.get("UNPLACED1")).toMatchObject({ kind: "part", description: "Flexible shim ring" });
  expect(index.get("G916100410040")).toMatchObject({ kind: "assembly", clickable: false });
  expect(index.get("Q923VV2526401")).toMatchObject({ kind: "assembly", clickable: false });
  expect(index.get("Q900V")).toMatchObject({ kind: "model", clickable: false });
  expect(index.get("Q938V").kind).toBe("model");
});

it("turns part numbers into clickable buttons and leaves dates and quantities alone", () => {
  const index = buildIdentifierIndex(snapshot);
  const host = document.createElement("div");
  host.innerHTML = "<p>Adds new part 916100410830, quantity 1, in G916100410040 on 2022-05-30 for Q900V.</p>";
  const inspect = vi.fn();
  linkifyIdentifiers(host, index, inspect);

  const part = host.querySelector("button.id-ref--part");
  expect(part.textContent).toBe("916100410830");
  expect(part.title).toContain("Suction hose, hydraulic");
  part.click();
  expect(inspect).toHaveBeenCalledWith("916100410830");

  const assembly = host.querySelector("span.id-ref--assembly");
  expect(assembly.textContent).toBe("G916100410040");
  expect(assembly.title).toContain("Assembly");
  expect(host.querySelector("span.id-ref--model").textContent).toBe("Q900V");

  // The date and the quantity digit are not identifiers, so they stay plain text.
  expect(host.textContent).toContain("2022-05-30");
  expect(host.textContent).toContain("quantity 1");
  expect(host.querySelectorAll(".id-ref")).toHaveLength(3);
});

it("replaces only described part identifiers in a compact AI change brief", () => {
  const index = buildIdentifierIndex(snapshot);
  const brief = describePartIdentifiers(
    "New part 916100410830 is added to assembly G916100410040 on 2022-05-30.",
    index,
  );

  expect(brief).toBe("New part Suction hose, hydraulic is added to assembly G916100410040 on 2022-05-30.");
});

it("is idempotent and never nests refs when run twice", () => {
  const index = buildIdentifierIndex(snapshot);
  const host = document.createElement("div");
  host.innerHTML = "<p>Part 916100410830 in G916100410040.</p>";
  linkifyIdentifiers(host, index, vi.fn());
  linkifyIdentifiers(host, index, vi.fn());
  expect(host.querySelectorAll(".id-ref")).toHaveLength(2);
  expect(host.querySelector(".id-ref .id-ref")).toBeNull();
});
