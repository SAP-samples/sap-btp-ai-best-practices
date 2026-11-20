import { expect, it } from "vitest";
import { makeBuyRadarOption } from "../src/components/ecn-charts.js";

it("creates two filled areas for Make and Buy in the decision comparison", () => {
  const option = makeBuyRadarOption({ make: [5, 4, 3, 2, 1], buy: [1, 2, 3, 4, 5] });

  expect(option.radar.indicator.map(indicator => indicator.name)).toEqual(["Cost", "Lead time", "Impacted plants", "Historical trend", "Quality"]);
  expect(option.series[0].type).toBe("radar");
  expect(option.series[0].data).toEqual(expect.arrayContaining([
    expect.objectContaining({ name: "Make", value: [5, 4, 3, 2, 1], areaStyle: expect.any(Object) }),
    expect.objectContaining({ name: "Buy", value: [1, 2, 3, 4, 5], areaStyle: expect.any(Object) }),
  ]));
});
