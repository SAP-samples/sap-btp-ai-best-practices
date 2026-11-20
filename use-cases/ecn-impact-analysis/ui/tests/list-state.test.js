import { expect, it } from "vitest";
import { detailHref, parseListState, serializeListState } from "../src/modules/list-state.js";

it("keeps only valid API list parameters and encodes the internal row ID", () => {
  const state = parseListState(new URLSearchParams(
    "query=shaft&analysis_state=FAILED&has_warnings=false&sort=warning_count&order=asc&page=2&page_size=50&evil=x",
  ));
  expect(state).toEqual({
    query: "shaft", ecn_number: "", plant: "", analysis_state: "FAILED", has_warnings: "false",
    sort: "warning_count", order: "asc", page: 2, page_size: 50,
  });
  expect(serializeListState(state).has("evil")).toBe(false);
  expect(detailHref("row/id", state)).toContain("/ecns/row%2Fid?");
});

it("round-trips the plant filter and plant sort through the URL", () => {
  const state = parseListState("plant=PL10&sort=affected_plants");
  expect(state.plant).toBe("PL10");
  expect(state.sort).toBe("affected_plants");
  expect(serializeListState(state).get("plant")).toBe("PL10");
});

it.each(["SUMMARIES_AVAILABLE", "IN_PROGRESS"])("round-trips the %s analysis group through the URL and detail links", (group) => {
  const state = parseListState(`analysis_state=${group}&page=2`);
  expect(state.analysis_state).toBe(group);
  expect(parseListState(serializeListState(state))).toEqual(state);
  expect(detailHref("row-1", state)).toContain(`analysis_state=${group}`);
});

it("rejects unrecognized analysis groups", () => {
  expect(parseListState("analysis_state=UNKNOWN").analysis_state).toBe("");
});
