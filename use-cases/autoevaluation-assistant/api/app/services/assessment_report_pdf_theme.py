"""Shared assessment framework typography and color tokens for assessment PDF reports."""

from pathlib import Path

from fpdf import FPDF

REPORT_FONT_FAMILY = "montserrat"
"""FPDF family name for the embedded Montserrat typeface."""

CHARCOAL = (44, 62, 80)
"""Primary text, heading, and dark-background color (#2C3E50)."""

LIGHT_YELLOW_ASSESSMENT = (255, 204, 0)
"""Yellow comparison accent from the supplied assessment framework palette (#FFCC00)."""

MINDARO = (195, 233, 145)
"""Green comparison accent from the supplied assessment framework palette (#C3E991)."""

MOONSTONE = (81, 175, 184)
"""Teal comparison and rule accent from the supplied assessment framework palette (#51AFB8)."""

BONE = (236, 226, 208)
"""Warm neutral track and border color from the supplied assessment framework palette (#ECE2D0)."""

_FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"


def register_report_fonts(pdf: FPDF) -> None:
    """Register embedded Montserrat regular and bold faces.

    Inputs:
        pdf: Active FPDF document that will render the assessment report.

    Outputs:
        None. The document can use Montserrat in regular and bold styles.
    """

    pdf.add_font(
        family=REPORT_FONT_FAMILY,
        fname=_FONT_DIR / "Montserrat-Regular.ttf",
    )
    pdf.add_font(
        family=REPORT_FONT_FAMILY,
        style="B",
        fname=_FONT_DIR / "Montserrat-Bold.ttf",
    )
