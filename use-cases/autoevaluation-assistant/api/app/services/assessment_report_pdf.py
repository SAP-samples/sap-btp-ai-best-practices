"""Orchestrate the deterministic, naturally paginated assessment PDF."""

from __future__ import annotations

from fpdf import FPDF

from app.models.reports import AssessmentReportSource
from app.services.assessment_report_localization import pdf_safe_text, report_text
from app.services.assessment_report_pdf_sections import (
    write_cover,
    write_dimension_analysis,
    write_dimension_positioning_summary,
    write_methodology_and_provenance,
    write_overall_overview,
)
from app.services.assessment_report_pdf_theme import (
    CHARCOAL,
    MOONSTONE,
    REPORT_FONT_FAMILY,
    register_report_fonts,
)


class AssessmentReportPdf(FPDF):
    """FPDF document with opening-page headers, footers, and natural breaks.

    Inputs:
        language: Active English or Italian report language.

    Outputs:
        Configured portrait-first PDF canvas used by decomposed section writers.
    """

    def __init__(self, language: str) -> None:
        """Initialize margins, portable font encoding, and page numbering.

        Inputs:
            language: Normalized report language.

        Outputs:
            None. The document is ready for portrait and landscape pages.
        """

        super().__init__(orientation="P", unit="mm", format="A4")
        register_report_fonts(self)
        self.language = language
        self.core_fonts_encoding = "latin-1"
        self.set_margins(14, 16, 14)
        self.set_auto_page_break(auto=True, margin=16)
        self.alias_nb_pages()

    def header(self) -> None:
        """Render the product label after the compact opening page.

        Inputs:
            None. Current page number and orientation come from FPDF state.

        Outputs:
            None. Page one is left to the compact title and metadata treatment.
        """

        if self.page_no() == 1:
            return
        self.set_font(REPORT_FONT_FAMILY, "B", 7.5)
        self.set_text_color(*CHARCOAL)
        self.cell(
            0,
            5,
            pdf_safe_text(report_text(self.language, "product")),
            align="L",
        )
        self.set_draw_color(*MOONSTONE)
        self.line(self.l_margin, 12, self.w - self.r_margin, 12)
        self.ln(4)

    def footer(self) -> None:
        """Render localized current/total page numbering on every page.

        Inputs:
            None. Current page number and aliased total come from FPDF state.

        Outputs:
            None. A centered footer is drawn inside the bottom margin.
        """

        self.set_y(-12)
        self.set_font(REPORT_FONT_FAMILY, size=7)
        self.set_text_color(*CHARCOAL)
        label = report_text(self.language, "page")
        self.cell(0, 6, f"{pdf_safe_text(label)} {self.page_no()}/{{nb}}", align="C")


def render_assessment_report_pdf(
    source: AssessmentReportSource,
) -> bytes:
    """Render one frozen deterministic assessment report.

    Inputs:
        source: Strict current deterministic snapshot.

    Outputs:
        bytes: Complete in-memory PDF ready for HANA BLOB persistence.
    """

    pdf = AssessmentReportPdf(source.language)
    # Stable document metadata keeps fixture output reproducible across retries;
    # the enqueue timestamp, not worker execution time, owns the report date.
    pdf.set_creation_date(source.generated_at)
    pdf.set_author(report_text(source.language, "product"))
    pdf.set_title(report_text(source.language, "report_title"))
    pdf.add_page()
    write_cover(pdf, source)
    write_overall_overview(pdf, source)
    write_dimension_positioning_summary(pdf, source)
    for dimension in source.dimensions:
        write_dimension_analysis(pdf, dimension, source.language)
    write_methodology_and_provenance(pdf, source)
    return bytes(pdf.output())
