import { importPreview, parseImportText, serializeSelection } from "../modules/ecn-import-data.js";

let records = [];

/** Process one import request while retaining the complete JSON only in this worker. */
self.addEventListener("message", async ({ data }) => {
  const { id, type } = data;
  try {
    let result;
    if (type === "parse") {
      records = [];
      records = parseImportText(await data.file.text());
      result = { total: records.length };
    } else if (type === "preview") {
      result = importPreview(records, data.options);
    } else if (type === "serialize") {
      result = serializeSelection(records, data.positions);
    } else {
      throw new Error("Unknown ECN import worker request.");
    }
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
});
