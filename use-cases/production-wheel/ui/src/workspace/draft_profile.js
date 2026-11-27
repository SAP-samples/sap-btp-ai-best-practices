/**
 * Texts for drafts whose frozen plant profile revision is older than the profile.
 *
 * A draft keeps the profile revision it was created with, so later profile edits
 * (settings, rules, volume matrix) do not reach it. The Optimizer workspace shows
 * a warning with a "Recreate draft" button; these pure helpers build its texts so
 * they can be tested without a DOM.
 */

/**
 * Build the warning shown above the draft actions.
 * @param {object|null} status Response of GET /api/run-drafts/{id}/profile-status.
 * @returns {{text: string, button: string|null}|null} Warning text and the
 *   recreate button label, or null when the draft's profile is current.
 */
export function profileFreshness(status) {
  if (!status || !status.message) return null;
  return {
    text: status.message,
    button: status.outdated ? `Recreate draft with revision ${status.current_profile_revision}` : null,
  };
}

/**
 * Summarize what a recreate kept and dropped, for the draft status line.
 * @param {object} result Response of POST /api/run-drafts/{id}/recreate-with-current-profile.
 * @returns {string} One status sentence (plus skipped values, if any).
 */
export function recreatedNote(result) {
  const revision = result.profile_status?.current_profile_revision ?? result.draft?.plant_profile?.revision;
  const kept = result.carried_over?.length || 0;
  let note = `New draft created with profile revision ${revision}; ${kept} scenario choice${kept === 1 ? "" : "s"} kept. Validate it before launching.`;
  if (result.skipped?.length)
    note += ` Not kept: ${result.skipped.map((item) => `${item.path} (${item.reason})`).join("; ")}.`;
  return note;
}
