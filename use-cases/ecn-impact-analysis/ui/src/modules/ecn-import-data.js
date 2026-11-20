/** Parse one ECN JSON array and reject roots or members the API cannot ingest. */
export function parseImportText(text) {
  const root = JSON.parse(text);
  if (!Array.isArray(root)) throw new Error("The ECN file must contain a JSON array.");
  const invalid = root.findIndex((record) => !record || typeof record !== "object" || Array.isArray(record));
  if (invalid >= 0) throw new Error(`ECN row ${invalid + 1} must be a JSON object.`);
  return root;
}

/** Render the source lifecycle without coercing objects to "[object Object]". */
export function lifecycleLabel(state) {
  if (typeof state === "string" && state.trim()) return state.trim();
  if (state && typeof state === "object") {
    for (const key of ["Display", "Value"]) {
      if (typeof state[key] === "string" && state[key].trim()) return state[key].trim();
    }
  }
  return "Not available";
}

/** Return one searchable, bounded preview page without copying source records. */
export function importPreview(records, { query = "", page = 1, pageSize = 25 } = {}) {
  const needle = query.trim().toLocaleLowerCase();
  const matching = records
    .map((record, index) => ({ record, position: index + 1 }))
    .filter(({ record }) => {
      const haystack = [record.Number, record.DescriptionSummary, record.Description, record.Name]
        .filter((value) => typeof value === "string")
        .join(" ")
        .toLocaleLowerCase();
      return !needle || haystack.includes(needle);
    });
  const pages = Math.max(1, Math.ceil(matching.length / pageSize));
  const currentPage = Math.min(Math.max(1, page), pages);
  const start = (currentPage - 1) * pageSize;
  return {
    matching: matching.length,
    page: currentPage,
    pages,
    entries: matching.slice(start, start + pageSize).map(({ record, position }) => ({
      position,
      ecnNumber: typeof record.Number === "string" && record.Number.trim()
        ? record.Number.trim()
        : "Not available",
      description: [record.DescriptionSummary, record.Description, record.Name]
        .find((value) => typeof value === "string" && value.trim())?.trim() || "Not available",
      lifecycle: lifecycleLabel(record.State),
    })),
  };
}

/** Serialize only the selected one-based positions in original file order. */
export function serializeSelection(records, positions) {
  const selected = [...new Set(positions)]
    .sort((left, right) => left - right)
    .map((position) => {
      if (!Number.isInteger(position) || position < 1 || position > records.length) {
        throw new Error("The ECN selection is outside the imported file.");
      }
      return records[position - 1];
    });
  return JSON.stringify(selected);
}
