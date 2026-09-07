/** Shared visual encodings for the nested benchmark bars. */
export const NESTED_BAR_STYLE = Object.freeze({
  bestPeer: { color: "#b7ddb7", thickness: 42 },
  peerAverage: { color: "#f2cd32", thickness: 28 },
  company: { color: "#2e83b7", thickness: 14 }
});

/**
 * Escape a value for insertion into qualitative score markup.
 *
 * @param {unknown} value - API or translation value to render.
 * @returns {string} HTML-safe text.
 */
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/**
 * Return a complete nullable benchmark metric for predictable rendering.
 *
 * @param {object | null | undefined} metric - Nested metric from the API.
 * @param {number | null | undefined} score - Compatibility company score.
 * @returns {object} Normalized company/peer values and qualitative status.
 */
function normalizedMetric(metric, score) {
  return {
    company_score: metric?.company_score ?? score ?? 0,
    peer_average: metric?.peer_average ?? null,
    best_peer: metric?.best_peer ?? null,
    positioning: metric?.positioning || "unavailable",
    commentary: metric?.commentary || ""
  };
}

/**
 * Convert a score response into seven dimension groups with applicable topics.
 *
 * @param {object} payload - Assessment score API response.
 * @returns {{dimensions: object[]}} Dimension/topic view model.
 */
export function buildScoreViewModel(payload) {
  const applicableQuestions = (payload?.questions || []).filter(
    (question) => question.applicable !== false
  );
  return {
    dimensions: (payload?.dimensions || []).map((dimension) => ({
      ...dimension,
      label: dimension.display_name || dimension.dimension,
      benchmark: normalizedMetric(dimension.benchmark, dimension.score),
      topics: applicableQuestions
        .filter((question) => question.dimension === dimension.dimension)
        .map((question) => ({
          ...question,
          label: question.topic_title || question.question_id,
          benchmark: normalizedMetric(question.benchmark, question.score)
        }))
    }))
  };
}

/**
 * Clamp a score to the non-disclosing chart's fixed internal 0-100 scale.
 *
 * @param {unknown} value - Nullable score value.
 * @returns {number | null} Clamped numeric value or null for unavailable peers.
 */
function chartValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return null;
  }
  return Math.min(100, Math.max(0, Number(value)));
}

/**
 * Wrap long Chart.js axis labels without shortening their words.
 *
 * @param {string} label - Full dimension or topic title.
 * @param {number} [lineLength] - Approximate maximum characters per line.
 * @returns {string[]} Wrapped lines for a categorical axis tick.
 */
export function wrapChartLabel(label, lineLength = 28) {
  const words = String(label || "").split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  words.forEach((word) => {
    if (line && `${line} ${word}`.length > lineLength) {
      lines.push(line);
      line = word;
    } else {
      line = line ? `${line} ${word}` : word;
    }
  });
  if (line) {
    lines.push(line);
  }
  return lines.length ? lines : [""];
}

/**
 * Build one overlapping nested-bar Chart.js configuration.
 *
 * Exact values stay inside the canvas renderer. Numeric axes, labels, legend,
 * interactions, and tooltips are disabled; surrounding DOM supplies semantic
 * qualitative descriptions without exposing numeric score labels.
 *
 * @param {object[]} items - Dimension or topic score view-model items.
 * @param {{company: string, peerAverage: string, bestPeer: string}} labels - Legend labels.
 * @returns {object} Chart.js configuration with a fixed hidden 0-100 scale.
 */
export function buildNestedBarChartConfig(items, labels) {
  const metrics = items.map((item) => normalizedMetric(item.benchmark, item.score));
  return {
    type: "bar",
    data: {
      labels: items.map((item) => wrapChartLabel(item.label || item.display_name || item.dimension)),
      datasets: [
        // Chart.js draws higher-order datasets first. Combined with
        // `grouped:false`, this layers the thick green bar behind the thinner
        // yellow and blue bars at the same categorical center.
        {
          label: labels.bestPeer,
          data: metrics.map((metric) => chartValue(metric.best_peer)),
          backgroundColor: NESTED_BAR_STYLE.bestPeer.color,
          barThickness: NESTED_BAR_STYLE.bestPeer.thickness,
          grouped: false,
          order: 3
        },
        {
          label: labels.peerAverage,
          data: metrics.map((metric) => chartValue(metric.peer_average)),
          backgroundColor: NESTED_BAR_STYLE.peerAverage.color,
          barThickness: NESTED_BAR_STYLE.peerAverage.thickness,
          grouped: false,
          order: 2
        },
        {
          label: labels.company,
          data: metrics.map((metric) => chartValue(metric.company_score)),
          backgroundColor: NESTED_BAR_STYLE.company.color,
          barThickness: NESTED_BAR_STYLE.company.thickness,
          grouped: false,
          order: 1
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      events: [],
      scales: {
        x: {
          min: 0,
          max: 100,
          display: false,
          stacked: false
        },
        y: {
          stacked: false,
          grid: { display: false },
          ticks: { autoSkip: false, color: "#1d2b36", font: { size: 12 } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false }
      }
    }
  };
}

/**
 * Render nonnumeric positioning and deterministic commentary for chart items.
 *
 * @param {object[]} items - Dimension or topic view-model items.
 * @param {(key: string) => string} translate - Translation lookup function.
 * @returns {string} Semantic list markup that never exposes exact score values.
 */
export function qualitativeScoreMarkup(items, translate) {
  return `<ul class="score-positioning-list">${items
    .map((item) => {
      const metric = normalizedMetric(item.benchmark, item.score);
      const position = translate(`score.position.${metric.positioning}`);
      return `<li><strong>${escapeHtml(item.label || item.display_name || item.dimension)}</strong><span class="position-badge position-${escapeHtml(metric.positioning)}">${escapeHtml(position)}</span><p>${escapeHtml(metric.commentary || translate("score.position.unavailable"))}</p></li>`;
    })
    .join("")}</ul>`;
}

/**
 * Build a qualitative accessible chart description without score numbers.
 *
 * @param {object[]} items - Dimension or topic view-model items.
 * @param {(key: string) => string} translate - Translation lookup function.
 * @returns {string} Plain-language item/status summary for canvas ARIA.
 */
export function scoreAriaDescription(items, translate) {
  return items
    .map((item) => {
      const metric = normalizedMetric(item.benchmark, item.score);
      const label = item.label || item.display_name || item.dimension;
      return `${label}: ${translate(`score.position.${metric.positioning}`)}`;
    })
    .join("; ");
}
