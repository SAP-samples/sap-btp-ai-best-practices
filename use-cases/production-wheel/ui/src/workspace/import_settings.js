/**
 * Extraction settings for the workbook being imported on the Datasets page.
 *
 * The Datasets page previews the chosen production workbook
 * (`POST /api/datasets/preview`) and keeps the result in `app.pendingImport`:
 * file names, plant, the production frequency codes with FINI row counts, and the
 * overrides already applied. The chat sends it with each Datasets turn as
 * `pending_import`. The Workspace Assistant answers with an
 * `extraction_settings_proposal` card; Apply hands the overrides to the Datasets
 * page through `app.importSettings.apply`, which the page registers while mounted.
 * Nothing here calls the server or saves data.
 */
import { el } from "./dom.js";

/** Keyword the import resolves to every weekly code present in the workbook. */
export const ALL_AVAILABLE = "all_available";
/** Codes modeled when the planner does not override the frequencies. */
export const DEFAULT_FREQUENCIES = ["01W", "02W"];

/** Plain labels and default descriptions shown for each extraction setting. */
const SETTINGS = [
  ["modeled_frequencies", "Production frequencies", "01W / 02W"],
  ["demand_days", "Demand days", "workbook source horizon if present, otherwise 250"],
  ["productive_weeks", "Productive weeks", "source horizon / 5 if present, otherwise 50"],
  ["canonical_factor", "Canonical factor", "0.90"],
];

/**
 * Return the weekly codes the import will model.
 * @param {object} overrides Applied overrides (may hold the `all_available` keyword).
 * @param {object|null} inventory `frequency_inventory` from the preview.
 * @returns {string[]} Codes such as ["01W", "02W"].
 */
export function modeledCodes(overrides = {}, inventory = null) {
  const value = overrides.modeled_frequencies;
  if (value === ALL_AVAILABLE) return (inventory?.codes || []).map((item) => item.code);
  return Array.isArray(value) ? value : DEFAULT_FREQUENCIES;
}

/**
 * Describe each extraction setting in plain words, with its origin.
 * @param {object} overrides Applied overrides; absent keys keep their default.
 * @param {object|null} inventory `frequency_inventory` from the preview, if any.
 * @param {string} origin Word for non-default values: "applied" on the Datasets
 *   page, "proposed" on an assistant card that is not applied yet.
 * @returns {string[]} One line per setting, e.g. "Demand days: 240 (applied)".
 */
export function settingLines(overrides = {}, inventory = null, origin = "applied") {
  return SETTINGS.map(([key, label, fallback]) => {
    if (!(key in overrides)) return `${label}: ${fallback} (default)`;
    const value = overrides[key];
    if (key !== "modeled_frequencies") return `${label}: ${value} (${origin})`;
    if (value !== ALL_AVAILABLE) return `${label}: ${value.join(" / ")} (${origin})`;
    const codes = modeledCodes(overrides, inventory);
    return codes.length
      ? `${label}: all ${codes.length} codes in the workbook (${codes.join(", ")}) (${origin})`
      : `${label}: all codes in the workbook (${origin})`;
  });
}

/**
 * Count the FINI rows whose frequency the current settings admit. Extraction may
 * still exclude some of them for data quality (for example a missing pallet).
 * @param {object} overrides Applied overrides.
 * @param {object} inventory `frequency_inventory` from the preview.
 * @returns {{modeled: number, total: number}} Row counts.
 */
export function modeledRowCount(overrides, inventory) {
  const codes = new Set(modeledCodes(overrides, inventory));
  const modeled = (inventory?.codes || []).filter((item) => codes.has(item.code)).reduce((sum, item) => sum + item.rows, 0);
  return { modeled, total: inventory?.total_rows || 0 };
}

/**
 * Return the import the assistant may see: only on the Datasets page.
 * @param {object} state App state (uses `page`).
 * @param {object|null} pending `app.pendingImport`.
 * @returns {object|null} Pending import for the chat body; null elsewhere so the
 *   server session forgets a workbook from an earlier Datasets visit.
 */
export function pendingImportContext(state, pending) {
  return state.page === "datasets" && pending ? pending : null;
}

/**
 * Chat history note recording the planner's decision on a settings card.
 * @param {object} proposal `extraction_settings_proposal` event.
 * @param {boolean} applied Whether the planner clicked Apply.
 * @returns {string} Note the assistant reads on its next turn.
 */
export function importDecisionNote(proposal, applied) {
  return applied
    ? `[Applied in UI] Extraction settings for ${proposal.primary_filename}: ${JSON.stringify(proposal.overrides)}. They apply to the next Upload and extract.`
    : `[Rejected in UI] Extraction settings proposal for ${proposal.primary_filename}; nothing changed.`;
}

/**
 * Render an Apply/Reject card for an extraction settings proposal.
 * @param {object} proposal `extraction_settings_proposal` event from the chat stream.
 * @param {{apply: Function, reject: Function}} handlers `apply` throws when the
 *   proposal no longer matches the chosen workbook.
 * @returns {HTMLElement} Card element.
 */
export function renderImportSettingsCard(proposal, handlers) {
  const card = el("div", null, "chat-message proposal-card");
  const list = el("ul");
  const inventory = { codes: proposal.modeled_codes.map((code) => ({ code })) };
  const lines = settingLines(proposal.overrides, inventory, "proposed");
  if (proposal.total_rows)
    lines.push(`FINI rows in scope by frequency: ${proposal.modeled_rows.toLocaleString()} of ${proposal.total_rows.toLocaleString()}`);
  for (const line of lines) list.append(el("li", line));
  const status = el("p", "Not applied. Review, then Apply or Reject.", "muted");
  status.setAttribute("role", "status");
  const apply = el("button", "Apply", "proposal-approve");
  const reject = el("button", "Reject", "proposal-reject");
  apply.type = reject.type = "button";
  const actions = el("div", null, "actions");
  actions.append(apply, reject);
  card.append(
    el("strong", `Extraction settings · ${proposal.primary_filename}`),
    el("p", proposal.summary),
    list,
    ...(proposal.warnings || []).map((warning) => el("p", warning, "muted")),
    status,
    actions,
  );

  /** Apply or reject once; a stale card stays rejectable but can never be applied. */
  async function decide(applied) {
    apply.disabled = reject.disabled = true;
    try {
      if (applied) {
        await handlers.apply(proposal);
        status.textContent = "Applied. Click Upload and extract to import with these settings.";
      } else {
        await handlers.reject(proposal);
        status.textContent = "Rejected. Nothing changed.";
      }
      card.classList.add(applied ? "approved" : "rejected");
    } catch (error) {
      status.textContent = error.message;
      reject.disabled = false;
    }
  }
  apply.addEventListener("click", () => decide(true));
  reject.addEventListener("click", () => decide(false));
  return card;
}
