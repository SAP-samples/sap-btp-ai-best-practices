"""Draw nonnumeric nested-bar charts for deterministic assessment reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fpdf import FPDF

from app.models.reports import AssessmentReportMetric
from app.services.assessment_report_localization import pdf_safe_text, report_text
from app.services.assessment_report_pdf_theme import (
    BONE,
    CHARCOAL,
    LIGHT_YELLOW_ASSESSMENT,
    MINDARO,
    MOONSTONE,
    REPORT_FONT_FAMILY,
)

BEST_PEER_COLOR = MINDARO
"""Green assessment framework layer used for the independent best peer."""

PEER_AVERAGE_COLOR = LIGHT_YELLOW_ASSESSMENT
"""Yellow assessment framework layer used for the arithmetic peer average."""

COMPANY_COLOR = MOONSTONE
"""Teal assessment framework foreground layer used for the assessed company."""

PLOT_TRACK_COLOR = BONE
"""Warm neutral assessment framework track that discloses no numeric tick labels."""


@dataclass(frozen=True)
class NestedBarLayer:
    """One clamped geometry layer in a shared zero-to-one-hundred plot.

    Inputs:
        series: Stable semantic layer identity.
        color: RGB fill color.
        thickness: Visual bar thickness in millimeters.
        width: Clamped horizontal width within the supplied plot extent.

    Outputs:
        Immutable geometry drawn in best, average, then company order.
    """

    series: str
    color: tuple[int, int, int]
    thickness: float
    width: float


def _clamped_width(value: float, plot_width: float) -> float:
    """Convert one score into defensive fixed-scale plot geometry.

    Inputs:
        value: Trusted score that is nevertheless clamped to zero through one hundred.
        plot_width: Physical width representing the complete hidden scale.

    Outputs:
        float: Width between zero and ``plot_width`` millimeters.
    """

    clamped = max(0.0, min(float(value), 100.0))
    return plot_width * clamped / 100.0


def build_nested_bar_layers(
    metric: AssessmentReportMetric,
    plot_width: float,
) -> tuple[NestedBarLayer, ...]:
    """Build ordered nested-bar geometry without coercing null peers to zero.

    Inputs:
        metric: Frozen company and optional peer values for one scope.
        plot_width: Physical extent representing the hidden zero-to-one-hundred scale.

    Outputs:
        tuple[NestedBarLayer, ...]: Best peer, peer average, and company layers;
        missing peer layers are omitted while the company is always retained.
    """

    layers: list[NestedBarLayer] = []
    if metric.best_peer is not None:
        layers.append(
            NestedBarLayer(
                series="best_peer",
                color=BEST_PEER_COLOR,
                thickness=6.6,
                width=_clamped_width(metric.best_peer, plot_width),
            )
        )
    if metric.peer_average is not None:
        layers.append(
            NestedBarLayer(
                series="peer_average",
                color=PEER_AVERAGE_COLOR,
                thickness=4.4,
                width=_clamped_width(metric.peer_average, plot_width),
            )
        )
    layers.append(
        NestedBarLayer(
            series="company",
            color=COMPANY_COLOR,
            thickness=2.2,
            width=_clamped_width(metric.company_score, plot_width),
        )
    )
    return tuple(layers)


def _wrap_text(pdf: FPDF, text: str, width: float) -> list[str]:
    """Wrap one label by measured core-font width while preserving paragraphs.

    Inputs:
        pdf: Active FPDF document whose current font supplies measurements.
        text: Label to wrap.
        width: Maximum line width in millimeters.

    Outputs:
        list[str]: Non-empty measured lines safe for manual row layout.
    """

    # ``multi_cell`` reserves the document cell margin on both sides when no
    # explicit padding is supplied. Measure against the same usable width so a
    # late third line cannot escape the row and collide with its separator.
    usable_width = max(width - (2 * pdf.c_margin), 1.0)
    lines: list[str] = []
    for paragraph in pdf_safe_text(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if pdf.get_string_width(candidate) <= usable_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines or [""]


def _add_continuation_page(pdf: FPDF) -> None:
    """Add a page using the current portrait or landscape orientation.

    Inputs:
        pdf: Active FPDF document.

    Outputs:
        None. A new page is appended with the same orientation.
    """

    pdf.add_page(orientation=pdf.cur_orientation)


def _ensure_space(pdf: FPDF, required_height: float) -> bool:
    """Reserve a complete chart block and report whether a page was added.

    Inputs:
        pdf: Active FPDF document.
        required_height: Minimum vertical millimeters needed by the next block.

    Outputs:
        bool: ``True`` when a continuation page was created.
    """

    if pdf.get_y() + required_height <= pdf.h - pdf.b_margin:
        return False
    _add_continuation_page(pdf)
    return True


def _draw_legend(pdf: FPDF, language: str) -> None:
    """Draw a semantic legend containing no score values.

    Inputs:
        pdf: Active FPDF document.
        language: Report language used for legend labels.

    Outputs:
        None. Three color keys are drawn at the current vertical position.
    """

    _ensure_space(pdf, 10)
    entries = (
        (report_text(language, "best_peer"), BEST_PEER_COLOR),
        (report_text(language, "peer_average"), PEER_AVERAGE_COLOR),
        (report_text(language, "company"), COMPANY_COLOR),
    )
    pdf.set_font(REPORT_FONT_FAMILY, size=7.5)
    widths = [max(35.0, pdf.get_string_width(label) + 12.0) for label, _ in entries]
    start_x = pdf.l_margin + max(0.0, (pdf.epw - sum(widths)) / 2)
    y = pdf.get_y() + 1
    for (label, color), width in zip(entries, widths):
        pdf.set_fill_color(*color)
        pdf.rect(start_x, y + 1.2, 6, 3, style="F")
        pdf.set_xy(start_x + 8, y)
        pdf.set_text_color(*CHARCOAL)
        pdf.cell(width - 8, 5, pdf_safe_text(label))
        start_x += width
    pdf.set_y(y + 7)


def _draw_nested_bar_row(
    pdf: FPDF,
    *,
    label_lines: Sequence[str],
    metric: AssessmentReportMetric,
    label_width: float,
    row_height: float,
) -> None:
    """Draw one wrapping label and centered nested-bar geometry.

    Inputs:
        pdf: Active FPDF document.
        label_lines: Pre-wrapped localized dimension or coded-topic label lines.
        metric: Frozen metric controlling layer geometry and peer availability.
        label_width: Reserved wrapping column width.
        row_height: Measured vertical row height.

    Outputs:
        None. Text and nonnumeric layers are painted at the current position.
    """

    row_x = pdf.l_margin
    row_y = pdf.get_y()
    plot_x = row_x + label_width + 5
    plot_width = pdf.w - pdf.r_margin - plot_x
    label_y = row_y + max(0.0, (row_height - len(label_lines) * 3.8) / 2)

    pdf.set_xy(row_x, label_y)
    pdf.set_font(REPORT_FONT_FAMILY, size=7.5)
    pdf.set_text_color(*CHARCOAL)
    pdf.multi_cell(
        label_width,
        3.8,
        "\n".join(label_lines),
        align="L",
        new_x="RIGHT",
        new_y="TOP",
    )

    center_y = row_y + row_height / 2
    pdf.set_fill_color(*PLOT_TRACK_COLOR)
    pdf.rect(plot_x, center_y - 3.3, plot_width, 6.6, style="F")
    # Draw thick-to-thin so each smaller semantic series remains visible even
    # when its numeric extent is shorter or longer than a background layer.
    for layer in build_nested_bar_layers(metric, plot_width):
        pdf.set_fill_color(*layer.color)
        pdf.rect(
            plot_x,
            center_y - layer.thickness / 2,
            layer.width,
            layer.thickness,
            style="F",
        )
    pdf.set_y(row_y + row_height)


def write_nested_bar_chart(
    pdf: FPDF,
    items: Sequence[tuple[str, AssessmentReportMetric]],
    language: str,
    *,
    label_width: float = 68.0,
) -> None:
    """Render a naturally paginated qualitative nested-bar chart.

    Inputs:
        pdf: Active report document.
        items: Ordered localized labels paired with frozen metrics.
        language: Report language used for semantic legend and missing data.
        label_width: Wrapping column width reserved before the hidden-scale plot.

    Outputs:
        None. Rows paginate naturally and the semantic legend repeats after each
        chart continuation page; no numeric ticks, labels, or values are drawn.
    """

    if not items:
        return
    _ensure_space(pdf, 18)
    _draw_legend(pdf, language)
    for label, metric in items:
        pdf.set_font(REPORT_FONT_FAMILY, size=7.5)
        lines = _wrap_text(pdf, label, label_width)
        if metric.peer_average is None or metric.best_peer is None:
            lines.extend(
                _wrap_text(pdf, report_text(language, "peer_unavailable"), label_width)
            )
        row_height = max(10.0, len(lines) * 3.8 + 2.2)
        if _ensure_space(pdf, row_height + 1):
            _draw_legend(pdf, language)
        _draw_nested_bar_row(
            pdf,
            label_lines=lines,
            metric=metric,
            label_width=label_width,
            row_height=row_height,
        )
        pdf.set_draw_color(*BONE)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(1)
