/**
 * Pure helpers for the SAP settings readiness view and the after-run offer.
 *
 * The API (GET /api/runs/{id}/sap-settings/readiness) reports, per SAP tab,
 * whether generation would succeed and what is missing. These helpers turn that
 * payload into display lines, decide whether to offer generation after a run,
 * and build the default planning period. No DOM access: tested with node --test.
 */
import { isTerminal } from "./state.js";

/** People-facing tab names, in template order (same keys as the summary). */
export const SAP_TAB_LABELS = {
  pv_check: "Check PV in subgroup",
  kpis: "RoC and change overs",
  cycle_stock: "Cycle stock",
  planning_calendar: "Planning calendar",
  recipe_combination: "Recipe combination",
  apo_lot_size: "APO lot size",
  central_dc_lot_size: "Central DC lot size",
  runner_audit: "Runner RoC audit",
};

/**
 * Planning period label of the quarter containing `date`, e.g. "26Q4".
 * @param {Date} date Any date.
 * @returns {string} Two-digit year, "Q" and quarter number.
 */
export function currentQuarterLabel(date) {
  const year = String(date.getFullYear() % 100).padStart(2, "0");
  return `${year}Q${Math.floor(date.getMonth() / 3) + 1}`;
}

/**
 * True only when a run was seen active and is now completed.
 * Reopening a run that finished earlier (no previous status) is not a transition.
 * @param {string|undefined} previous Last status seen while polling this run.
 * @param {string} current Status just read.
 * @returns {boolean} Whether the run has just completed.
 */
export function completedTransition(previous, current) {
  return Boolean(previous) && !isTerminal(previous) && /completed|succeeded/i.test(current || "");
}

/**
 * Lines for every tab that is blocked or partial, with its fix.
 * @param {object|null} readiness Readiness payload.
 * @returns {{tab: string, label: string, status: string, text: string, fix: string, setting: string|null, code: string|null}[]}
 */
export function readinessLines(readiness) {
  if (!readiness?.available) return [];
  const lines = [];
  for (const [tab, info] of Object.entries(readiness.tabs || {})) {
    if (info.status !== "blocked" && info.status !== "partial") continue;
    const gap = (info.gaps || [])[0] || {};
    lines.push({
      tab,
      label: SAP_TAB_LABELS[tab] || tab,
      status: info.status,
      text: info.message || gap.message || "",
      fix: gap.fix || "",
      setting: gap.setting || null,
      code: gap.code || null,
    });
  }
  return lines;
}

/** Label of a point for people: its decision label (B, CO1...) or "point N". */
function pointLabel(decisions, pointIndex) {
  const point = (decisions?.points || []).find((item) => Number(item.point_index) === Number(pointIndex));
  return point?.decision_label || `point ${pointIndex}`;
}

/**
 * Decide what the after-run notice says, or null when there is nothing to offer.
 * The scenario is the stored selection, else the balanced (knee) point.
 * @param {object|null} decisions Decision scenarios payload (points, knee_point_index, selection).
 * @param {object|null} readiness Readiness payload for that point.
 * @param {object|null} summary Stored SAP settings summary, if generated.
 * @returns {null | {text: string, primaryLabel: string, pointIndex: number, needsSelection: boolean,
 *   gaps: object[], suggestion: object|null, question: string}}
 */
export function offerModel(decisions, readiness, summary) {
  if (!readiness?.available) return null;
  const stale = Boolean(readiness.last_summary?.stale);
  if (summary && !stale) return null;
  const selection = decisions?.selection || null;
  const pointIndex = selection?.point_index ?? decisions?.knee_point_index;
  if (pointIndex === null || pointIndex === undefined) return null;
  const label = selection?.decision_label || pointLabel(decisions, pointIndex);
  const gaps = readinessLines(readiness);
  const suggestion = (readiness.suggestions || []).find((item) => item.status === "suggested") || null;
  const text = stale
    ? `The plant profile changed since the SAP settings were generated (${readiness.last_summary.stale_reason}). Generate them again for scenario ${label}?`
    : `Run completed. Generate SAP settings for scenario ${label}?`;
  const missing = gaps.filter((gap) => gap.status === "blocked").map((gap) => gap.label);
  const question = `The run just completed. Help me choose a decision scenario and generate the SAP settings${missing.length ? `; these tabs are blocked: ${missing.join(", ")}. How do I fix them?` : "."}`;
  return {
    text,
    primaryLabel: selection ? "Generate SAP settings" : `Select ${label} and generate`,
    pointIndex: Number(pointIndex),
    needsSelection: !selection,
    gaps,
    suggestion,
    question,
  };
}

/**
 * One line naming where the SAP rules come from, and whether stored settings are stale.
 * @param {object|null} readiness Readiness payload.
 * @returns {string} Text for the SAP settings card ("" before results).
 */
export function sourceLine(readiness) {
  if (!readiness?.available) return "";
  const profile = readiness.profile || {};
  const where = profile.origin === "live"
    ? `SAP settings rules from plant profile "${profile.name}" revision ${profile.revision}.`
    : profile.origin === "frozen"
      ? `SAP settings rules from the removed plant profile "${profile.name}" (copy saved with the run).`
      : "This run has no plant profile; SAP settings use the defaults.";
  const stale = readiness.last_summary?.stale ? ` Generated settings are out of date: ${readiness.last_summary.stale_reason}.` : "";
  return where + stale;
}
