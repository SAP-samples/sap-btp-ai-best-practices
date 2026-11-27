/**
 * Plant-profile change proposals made by the Workspace Assistant.
 *
 * The assistant validates a change and emits a `profile_proposal` chat event; nothing
 * is saved until the user clicks Approve on the card rendered here. Approve posts the
 * typed change to `/api/plant-profiles/{id}/changes` with the revision the proposal was
 * made against, so a profile edited in the meantime is rejected instead of overwritten.
 * The Settings constraint list reuses `ruleSummary` and `changeRequest` for removals.
 */
import { el } from "./dom.js";
import { describeSapPatch } from "./sap_settings_form.js";
import { matrixModeLabel, volumeMatchingLabel } from "./vocabulary.js";

const ACTION_LABELS = {
  add_constraint: "Add constraint",
  replace_constraint: "Change constraint",
  remove_constraint: "Remove constraint",
  update_settings: "Change profile settings",
};

/**
 * Describe one typed business rule for people.
 * @param {object} rule Profile rule with constraint_id, kind, scope, description, source_text.
 * @returns {{title: string, meta: string, source: string}} Display strings.
 */
export function ruleSummary(rule) {
  const scope = rule.scope || {};
  const where = [scope.plant && `plant ${scope.plant}`, scope.sefi && `SEFI ${scope.sefi}`].filter(Boolean).join(", ") || "whole profile";
  const kind = rule.kind === "selection_bound" ? "Limit across selected groups" : "Rule for every group";
  return {
    title: rule.description || rule.constraint_id,
    meta: `${rule.constraint_id} · ${kind} · ${where}`,
    source: rule.source_text || "",
  };
}

/**
 * Build the HTTP request that applies a change against an exact profile revision.
 * @param {string} profileId Profile to change.
 * @param {number} revision Revision the change was proposed or reviewed against.
 * @param {object} change Typed ProfileChange ({action, constraint | constraint_id | settings | matrix_rows}).
 * @returns {{path: string, body: object}} POST path and JSON body.
 */
export function changeRequest(profileId, revision, change) {
  return { path: `/api/plant-profiles/${encodeURIComponent(profileId)}/changes`, body: { revision, change } };
}

/**
 * Summarize a proposal card's content without touching the DOM.
 * @param {object} proposal `profile_proposal` event from the chat stream.
 * @returns {{heading: string, lines: string[], details: object}} Card text and typed evidence.
 */
export function describeProposal(proposal) {
  const change = proposal.change || {};
  const lines = [];
  if (change.constraint) {
    const rule = ruleSummary(change.constraint);
    lines.push(rule.title, rule.meta);
    if (rule.source) lines.push(`Planner wording: ${rule.source}`);
  }
  if (change.constraint_id) lines.push(`Constraint: ${change.constraint_id}`);
  for (const [key, value] of Object.entries(change.settings || {})) {
    if (key === "matrix_mode") lines.push(`Compatibility mode: ${matrixModeLabel(value)}`);
    else if (key === "volume_matching") lines.push(`Volume matching: ${volumeMatchingLabel(value)}`);
    else if (key === "sap_settings") lines.push(...describeSapPatch(value));
    // Nested sections (runner_thresholds, cycle_stock_policy) are merged key by key on the server.
    else if (value && typeof value === "object") for (const [field, item] of Object.entries(value)) lines.push(`${key}.${field}: ${item}`);
    else lines.push(`${key}: ${value}`);
  }
  if (change.matrix_rows) {
    const volumes = new Set(change.matrix_rows.map((row) => row.volume_a)).size;
    lines.push(`Replace volume compatibility: ${volumes} volumes, ${change.matrix_rows.length} pairs`);
    const summary = proposal.matrix_summary;
    if (summary) {
      if (summary.added_volumes.length) lines.push(`Added volumes (L): ${summary.added_volumes.join(", ")}`);
      if (summary.removed_volumes.length) lines.push(`Removed volumes (L): ${summary.removed_volumes.join(", ")}`);
      const statuses = Object.entries(summary.pairs_by_status).map(([status, count]) => `${status} ${count}`).join(", ");
      lines.push(`Volume pairs by status: ${statuses}`);
      lines.push(`Existing pairs changed: ${summary.changed_pair_count}`);
      // Class matrices: which dataset products change class with the new stops.
      for (const item of summary.reclassified_volumes || []) {
        lines.push(`${item.volume} L (${item.product_count} product${item.product_count === 1 ? "" : "s"}): class ${item.from ?? "none"} → ${item.to ?? "none"}`);
      }
      if (summary.unclassified_volumes?.length) lines.push(`Volumes without a class (a run would stop): ${summary.unclassified_volumes.join(", ")} L`);
    }
  }
  return {
    heading: `${ACTION_LABELS[change.action] || "Profile change"} · ${proposal.profile_name} (revision ${proposal.profile_revision})`,
    lines,
    details: change,
  };
}

/**
 * Text added to the page conversation so the next assistant turn knows the decision.
 * @param {object} proposal Proposal the user decided on.
 * @param {boolean} approved Whether it was approved.
 * @param {number} [revision] Saved profile revision after approval.
 * @returns {string} History note.
 */
export function decisionNote(proposal, approved, revision) {
  const change = proposal.change || {};
  const target = change.constraint?.constraint_id || change.constraint_id || "settings";
  return approved
    ? `[Approved in UI] ${change.action} ${target} on profile ${proposal.profile_id}; saved as revision ${revision}.`
    : `[Rejected in UI] ${change.action} ${target} on profile ${proposal.profile_id}; nothing was saved.`;
}

/**
 * Render an approval card. Buttons disable after a decision; errors stay on the card.
 * @param {object} proposal `profile_proposal` event.
 * @param {{approve: Function, reject: Function}} handlers Async approve returns the saved profile.
 * @returns {HTMLElement} Card element.
 */
export function renderProposalCard(proposal, handlers) {
  const { heading, lines, details } = describeProposal(proposal);
  const card = el("div", null, "chat-message proposal-card");
  const summary = el("p", proposal.summary);
  const list = el("ul");
  for (const line of lines) list.append(el("li", line));
  const evidence = el("details");
  evidence.append(el("summary", "Typed change"), el("pre", JSON.stringify(details, null, 2), "evidence"));
  const status = el("p", "Not saved. Review, then Approve or Reject.", "muted");
  status.setAttribute("role", "status");
  const approve = el("button", "Approve", "proposal-approve");
  const reject = el("button", "Reject", "proposal-reject");
  approve.type = reject.type = "button";
  const actions = el("div", null, "actions");
  actions.append(approve, reject);
  card.append(el("strong", heading), summary, list, evidence, status, actions);

  /** Apply one decision; after a failed save the card stays open for another attempt. */
  async function decide(approved) {
    approve.disabled = reject.disabled = true;
    try {
      if (approved) {
        const saved = await handlers.approve(proposal);
        status.textContent = `Approved and saved as profile revision ${saved.revision}.`;
      } else {
        await handlers.reject(proposal);
        status.textContent = "Rejected. Nothing was saved.";
      }
      card.classList.add(approved ? "approved" : "rejected");
    } catch (error) {
      // A stale revision can never succeed; other failures (network, validation) may be retried.
      const stale = /revision/i.test(error.message);
      status.textContent = stale
        ? "The profile changed after this proposal. Ask the Assistant to propose it again."
        : `Could not save: ${error.message}`;
      approve.disabled = stale;
      reject.disabled = false;
    }
  }
  approve.addEventListener("click", () => decide(true));
  reject.addEventListener("click", () => decide(false));
  return card;
}
