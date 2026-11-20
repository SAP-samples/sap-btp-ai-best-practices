/**
 * Display text and tone for engineering review decisions.
 *
 * ACCEPTED (Accept ECN) and CHANGES_REQUESTED (Reject) are
 * the two decisions an engineer can submit.
 * The other three are retired outcomes that older reviews may still carry;
 * they stay readable in the history and show as "Reviewed" in summaries.
 * Shared by the ECN detail review card and the ECN list "Review" column.
 */
export const reviewPresentation = Object.freeze({
  ACCEPTED: ["Accepted", "success"],
  CHANGES_REQUESTED: ["Rejected", "error"],
  REVIEWED: ["Reviewed", "neutral"],
  CLARIFICATION_REQUESTED: ["Reviewed", "neutral"],
  CORRECTION_RECORDED: ["Reviewed", "neutral"],
});

/** History labels keep the retired outcomes' own wording. */
export const reviewHistoryLabels = Object.freeze({
  ACCEPTED: "Accepted",
  CHANGES_REQUESTED: "Rejected",
  REVIEWED: "Reviewed",
  CLARIFICATION_REQUESTED: "Clarification requested",
  CORRECTION_RECORDED: "Correction recorded",
});
