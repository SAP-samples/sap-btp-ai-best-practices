"""Deterministic section writers for assessment PDF reports."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from fpdf import FPDF

from app.models.reports import (
    AssessmentReportDimension,
    AssessmentReportSource,
)
from app.services.assessment_report_localization import (
    pdf_safe_text,
    positioning_summary_sentence,
    report_text,
)
from app.services.assessment_report_pdf_charts import write_nested_bar_chart
from app.services.assessment_report_pdf_theme import (
    CHARCOAL,
    MOONSTONE,
    REPORT_FONT_FAMILY,
)


def _date_text(value: datetime | None, language: str) -> str:
    """Format a snapshotted timestamp as a stable report date.

    Inputs:
        value: Optional UTC-aware timestamp.
        language: Report language used for the unavailable label.

    Outputs:
        str: ISO calendar date or localized unavailable wording.
    """

    return value.strftime("%Y-%m-%d") if value else report_text(language, "unavailable")


def _ensure_space(pdf: FPDF, required_height: float) -> bool:
    """Reserve vertical space and add a same-orientation page when necessary.

    Inputs:
        pdf: Active report document.
        required_height: Minimum millimeters required for the next block.

    Outputs:
        bool: Whether a page was added.
    """

    if pdf.get_y() + required_height <= pdf.h - pdf.b_margin:
        return False
    pdf.add_page(orientation=pdf.cur_orientation)
    return True


def _section_heading(
    pdf: FPDF,
    title: str,
    *,
    minimum_following_height: float = 16.0,
    spacing_before: float = 0.0,
) -> None:
    """Render a heading with optional spacing only when it stays on the page.

    Inputs:
        pdf: Active report document.
        title: Localized section heading.
        minimum_following_height: Minimum space reserved after the heading.
        spacing_before: Optional same-page top separation in millimeters.

    Outputs:
        None. The heading is written without an orphan or new-page top gap.
    """

    page_added = _ensure_space(
        pdf,
        spacing_before + 11 + minimum_following_height,
    )
    if spacing_before and not page_added:
        pdf.ln(spacing_before)
    pdf.set_x(pdf.l_margin)
    pdf.set_font(REPORT_FONT_FAMILY, "B", 16)
    safe_title = pdf_safe_text(title)
    available_width = pdf.epw - (2 * pdf.c_margin)
    title_width = max(pdf.get_string_width(safe_title), 1.0)
    title_font_size = min(16.0, max(13.0, 16.0 * available_width / title_width))
    pdf.set_font(REPORT_FONT_FAMILY, "B", title_font_size)
    pdf.set_text_color(*CHARCOAL)
    pdf.multi_cell(
        0,
        7,
        safe_title,
        align="L",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_draw_color(*MOONSTONE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)


def _write_narrative_lines(pdf: FPDF, sentences: Sequence[str]) -> None:
    """Render localized comparison sentences with natural page breaks.

    Inputs:
        pdf: Active report document.
        sentences: Ordered score-safe narrative lines.

    Outputs:
        None. Sentences wrap and paginate without tabular framing.
    """

    pdf.set_font(REPORT_FONT_FAMILY, size=8.5)
    pdf.set_text_color(*CHARCOAL)
    for sentence in sentences:
        _ensure_space(pdf, 7)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            0,
            4.5,
            pdf_safe_text(sentence),
            align="L",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(1)


def write_cover(pdf: FPDF, source: AssessmentReportSource) -> None:
    """Render compact opening-page identity and provenance information.

    Inputs:
        pdf: Active first-page report document.
        source: Frozen deterministic report source.

    Outputs:
        None. Title, assessed identity, class/NACE, and dates are written.
    """

    language = source.language
    pdf.set_fill_color(*CHARCOAL)
    pdf.rect(0, 0, pdf.w, 40, style="F")
    pdf.set_xy(pdf.l_margin, 10)
    pdf.set_font(REPORT_FONT_FAMILY, "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(
        pdf.epw,
        8,
        pdf_safe_text(report_text(language, "report_title")),
        align="L",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font(REPORT_FONT_FAMILY, size=8.5)
    pdf.multi_cell(
        pdf.epw,
        4.5,
        pdf_safe_text(report_text(language, "report_subtitle")),
        align="L",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_y(46)
    metadata = [
        (report_text(language, "assessed_company"), source.display_name),
        (report_text(language, "assessment"), source.assessment_id),
    ]
    if source.source_company_id:
        metadata.append(
            (report_text(language, "source_company_id"), source.source_company_id)
        )
    metadata.extend(
        [
            (report_text(language, "company_class"), source.customer_class_label),
            (report_text(language, "nace1"), source.nace1),
            (
                report_text(language, "dataset_date"),
                _date_text(source.provenance.dataset_activated_at, language)
                if source.provenance.import_id
                else report_text(language, "dataset_unavailable"),
            ),
            (
                report_text(language, "generation_date"),
                _date_text(source.generated_at, language),
            ),
        ]
    )
    pdf.set_font(REPORT_FONT_FAMILY, "B", 7.2)
    label_width = max(
        pdf.get_string_width(pdf_safe_text(label)) for label, _ in metadata
    ) + 4
    for label, value in metadata:
        pdf.set_x(pdf.l_margin)
        pdf.set_font(REPORT_FONT_FAMILY, "B", 7.2)
        pdf.set_text_color(*CHARCOAL)
        pdf.cell(label_width, 4.8, pdf_safe_text(label))
        pdf.set_font(REPORT_FONT_FAMILY, size=8)
        pdf.set_text_color(*CHARCOAL)
        pdf.multi_cell(
            pdf.epw - label_width,
            4.8,
            pdf_safe_text(value),
            align="L",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(2)


def write_overall_overview(
    pdf: FPDF,
    source: AssessmentReportSource,
) -> None:
    """Render the seven deterministic dimension bars.

    Inputs:
        pdf: Active report document.
        source: Frozen deterministic report source.

    Outputs:
        None. No exact score strings are rendered in this section.
    """

    _section_heading(
        pdf,
        report_text(source.language, "overall_overview"),
        minimum_following_height=32,
        spacing_before=5,
    )
    write_nested_bar_chart(
        pdf,
        [(dimension.display_name, dimension.metric) for dimension in source.dimensions],
        source.language,
    )


def write_dimension_positioning_summary(
    pdf: FPDF,
    source: AssessmentReportSource,
) -> None:
    """Render one deterministic narrative sentence per applicable dimension.

    Inputs:
        pdf: Active report document.
        source: Frozen deterministic report source.

    Outputs:
        None. Localized narrative sentences paginate naturally without a table.
    """

    _section_heading(
        pdf,
        report_text(source.language, "dimension_summary"),
        minimum_following_height=12,
        spacing_before=6,
    )
    _write_narrative_lines(
        pdf,
        [
            positioning_summary_sentence(
                source.language,
                dimension.display_name,
                "dimension",
                dimension.metric,
            )
            for dimension in source.dimensions
        ],
    )


def write_dimension_analysis(
    pdf: FPDF,
    dimension: AssessmentReportDimension,
    language: str,
) -> None:
    """Render one dimension's deterministic analysis.

    Inputs:
        pdf: Active report document.
        dimension: Frozen applicable dimension and ordered topics.
        language: Report language for deterministic labels.

    Outputs:
        None. The section starts cleanly and continues over arbitrary pages.
    """

    pdf.add_page()
    _section_heading(
        pdf,
        f"{report_text(language, 'dimension_analysis')}: {dimension.display_name}",
        minimum_following_height=34,
    )
    write_nested_bar_chart(
        pdf,
        [
            (f"{topic.question_id} - {topic.topic_title}", topic.metric)
            for topic in dimension.topics
        ],
        language,
        label_width=80,
    )
    _section_heading(
        pdf,
        report_text(language, "topic_positioning"),
        minimum_following_height=20,
        spacing_before=6,
    )
    _write_narrative_lines(
        pdf,
        [
            positioning_summary_sentence(
                language,
                f"{topic.question_id} - {topic.topic_title}",
                "topic",
                topic.metric,
            )
            for topic in dimension.topics
        ],
    )


def write_methodology_and_provenance(
    pdf: FPDF,
    source: AssessmentReportSource,
) -> None:
    """Render deterministic cohort methodology and captured import provenance.

    Inputs:
        pdf: Active report document.
        source: Frozen deterministic report source.

    Outputs:
        None. Thresholds, benchmark date, and missing-data rules are documented
        without disclosing exact scores, peer counts, or internal provenance.
    """

    language = source.language
    pdf.add_page()
    _section_heading(
        pdf,
        report_text(language, "methodology"),
        minimum_following_height=40,
    )
    paragraphs = [
        report_text(language, "method.cohort"),
        report_text(language, "method.selection"),
        report_text(language, "method.statistics"),
        report_text(language, "method.thresholds"),
        report_text(language, "method.missing"),
    ]
    pdf.set_font(REPORT_FONT_FAMILY, size=9)
    pdf.set_text_color(*CHARCOAL)
    for paragraph in paragraphs:
        _ensure_space(pdf, 16)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            0,
            5,
            pdf_safe_text(paragraph),
            align="L",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(1.5)
