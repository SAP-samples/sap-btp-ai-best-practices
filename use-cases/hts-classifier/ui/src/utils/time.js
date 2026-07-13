import dayjs from "dayjs";

export function formatTimestamp(value, pattern = "DD MMM YYYY HH:mm") {
  return value ? dayjs(value).format(pattern) : "—";
}

export function formatLocalTimestamp(value) {
  const normalized = String(value ?? "").trim();
  if (!normalized || normalized === "—") return "—";
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return normalized;
  return parsed.toLocaleString();
}
