/**
 * Format a confidence value (0-1) as a percentage string
 */
export function formatConfidence(value: number | undefined | null): string {
  if (value == null) return "—";
  return `${Math.round(value * 100)}%`;
}

/**
 * Get CSS class for confidence level
 */
export function confidenceClass(value: number | undefined | null): string {
  if (value == null) return "low";
  if (value >= 0.8) return "high";
  if (value >= 0.5) return "medium";
  return "low";
}

/**
 * Format a score (0-100) with color class
 */
export function scoreClass(score: number | undefined): string {
  if (score == null) return "";
  if (score >= 80) return "success";
  if (score >= 60) return "warning";
  return "error";
}

/**
 * Format a number value for display
 */
export function formatValue(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "number") {
    return value.toLocaleString("en-US", { maximumFractionDigits: 4 });
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

/**
 * Truncate a string to a max length
 */
export function truncate(str: string, maxLen = 60): string {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen) + "…";
}

/**
 * Convert camelCase field name to human-readable label
 */
export function fieldLabel(name: string): string {
  return name
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (s) => s.toUpperCase())
    .trim();
}

/**
 * Escape HTML special characters
 */
export function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/**
 * Format JSON for display
 */
export function formatJson(obj: unknown): string {
  return JSON.stringify(obj, null, 2);
}

/**
 * Get status icon character
 */
export function statusIcon(status: "pending" | "active" | "completed" | "error"): string {
  switch (status) {
    case "completed": return "✓";
    case "active": return "⟳";
    case "error": return "✗";
    default: return "○";
  }
}