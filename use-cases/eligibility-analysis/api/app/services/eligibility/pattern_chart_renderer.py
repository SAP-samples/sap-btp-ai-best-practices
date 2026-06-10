"""Server-side chart rendering for pattern analysis PDF reports.

Uses matplotlib with the Agg backend (no display required) to render
trend and rule distribution charts as PNG files for embedding in PDFs.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

logger = logging.getLogger(__name__)

# Colors matching the UI Chart.js palette (matplotlib-compatible)
_BAR_COLOR = (54 / 255, 162 / 255, 235 / 255, 0.7)
_LINE_COLOR = "#dc3545"
_RULE_COLORS = [
    "#4472C4", "#ED7D31", "#A5A5A5", "#FFC000",
    "#5B9BD5", "#70AD47", "#264478", "#9B57A0",
]


def render_trend_chart(
    trend_data: List[dict],
    output_path: Path,
) -> Optional[Path]:
    """Render non-eligibility trend as a dual-axis bar+line chart (PNG).

    Args:
        trend_data: List of TrendDataPoint dicts (period_start, total_invoices,
            rejected_invoices, rejection_rate).
        output_path: Path to write the PNG file.

    Returns:
        The output path, or None if there is no data to plot.
    """
    if not trend_data:
        return None

    labels = [str(p.get("period_start", ""))[:10] for p in trend_data]
    totals = [p.get("total_invoices", 0) for p in trend_data]
    rates = [round((p.get("rejection_rate", 0)) * 100, 1) for p in trend_data]

    fig, ax1 = plt.subplots(figsize=(8, 4), dpi=300)

    # Bar chart: total invoices
    x = range(len(labels))
    ax1.bar(x, totals, color=_BAR_COLOR, label="Total Invoices")
    ax1.set_xlabel("Period")
    ax1.set_ylabel("Total Invoices", color="#36A2EB")
    ax1.tick_params(axis="y", labelcolor="#36A2EB")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    # Line chart: rejection rate
    ax2 = ax1.twinx()
    ax2.plot(list(x), rates, color=_LINE_COLOR, marker="o", markersize=4, linewidth=2, label="Non-Eligibility Rate (%)")
    ax2.set_ylabel("Non-Eligibility Rate (%)", color=_LINE_COLOR)
    ax2.tick_params(axis="y", labelcolor=_LINE_COLOR)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    fig.suptitle("Non-Eligibility Trend", fontsize=12, fontweight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.info("Trend chart saved to %s", output_path)
    return output_path


def render_rule_distribution_chart(
    trend_data: List[dict],
    output_path: Path,
) -> Optional[Path]:
    """Render rule distribution as a horizontal bar chart (PNG).

    Aggregates rejection_by_rule across all trend data points.

    Args:
        trend_data: List of TrendDataPoint dicts with rejection_by_rule.
        output_path: Path to write the PNG file.

    Returns:
        The output path, or None if there is no data to plot.
    """
    if not trend_data:
        return None

    aggregated: Counter = Counter()
    for point in trend_data:
        by_rule = point.get("rejection_by_rule", {})
        for rule, count in by_rule.items():
            aggregated[rule] += count

    if not aggregated:
        return None

    sorted_rules = aggregated.most_common()
    rules = [r for r, _ in sorted_rules]
    counts = [c for _, c in sorted_rules]

    fig, ax = plt.subplots(figsize=(6, max(3, len(rules) * 0.5)), dpi=300)
    colors = [_RULE_COLORS[i % len(_RULE_COLORS)] for i in range(len(rules))]

    y_pos = range(len(rules))
    ax.barh(list(y_pos), counts, color=colors, height=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(rules, fontsize=9)
    ax.set_xlabel("Number of Triggers")
    ax.invert_yaxis()

    fig.suptitle("Rule Distribution", fontsize=12, fontweight="bold")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.info("Rule distribution chart saved to %s", output_path)
    return output_path
