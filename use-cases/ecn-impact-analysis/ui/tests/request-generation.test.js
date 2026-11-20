import { expect, it } from "vitest";
import { createLatestRequestGate } from "../src/modules/request-generation.js";

it("supersedes only work in the same request lane", () => {
  const page = new AbortController();
  const listReads = createLatestRequestGate(page.signal);
  const uploadMutations = createLatestRequestGate(page.signal);
  const firstList = listReads.next();
  const upload = uploadMutations.next();
  const secondList = listReads.next();

  expect(firstList.signal.aborted).toBe(true);
  expect(secondList.signal.aborted).toBe(false);
  expect(upload.signal.aborted).toBe(false);
  expect(upload.isCurrent()).toBe(true);

  uploadMutations.cancel();
  const replacementUpload = uploadMutations.next();
  expect(upload.signal.aborted).toBe(true);
  expect(replacementUpload.signal.aborted).toBe(false);

  page.abort();
  expect(secondList.signal.aborted).toBe(true);
  expect(replacementUpload.signal.aborted).toBe(true);
});
