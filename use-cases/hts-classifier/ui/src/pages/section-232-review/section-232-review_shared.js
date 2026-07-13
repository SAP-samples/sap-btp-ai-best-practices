import { formatLocalTimestamp } from "../../utils/time.js";

export function trimText(value) {
  return String(value ?? "").trim();
}

export function formatReviewTimestamp(value) {
  return formatLocalTimestamp(trimText(value));
}
