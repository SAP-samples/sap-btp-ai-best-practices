"""PDF export helpers for item classification reports."""

from __future__ import annotations

from io import BytesIO
import re
from typing import Iterable, List, Sequence, Tuple
from xml.sax.saxutils import escape

from app.models.metal_composition import HTSCandidate, MetalCompositionItemDetail


_PDF_TEXT_REPLACEMENTS = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
    }
)
_BOLD_MARKDOWN_PATTERN = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def _load_reportlab():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

    return {
        "colors": colors,
        "TA_CENTER": TA_CENTER,
        "LETTER": LETTER,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "inch": inch,
        "ListFlowable": ListFlowable,
        "ListItem": ListItem,
        "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
    }


def _is_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _format_label(value: str) -> str:
    return value.replace("_", " ").title()


def _format_item_type(value: str | None) -> str:
    if not value:
        return "Unknown"
    return {"gcc": "GCC"}.get(value.lower(), _format_label(value))


def _format_bool(value: object) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"


def _format_number(value: object, maximum_fraction_digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    rounded = round(numeric, maximum_fraction_digits)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.{maximum_fraction_digits}f}".rstrip("0").rstrip(".")


def _format_grams(value: object) -> str:
    formatted = _format_number(value)
    return f"{formatted}g" if formatted else ""


def _format_confidence(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{round(numeric * 100)}%"


def _safe_text(value: object) -> str:
    normalized = str(value or "").translate(_PDF_TEXT_REPLACEMENTS)
    segments: List[str] = []
    last_index = 0
    for match in _BOLD_MARKDOWN_PATTERN.finditer(normalized):
        start, end = match.span()
        if start > last_index:
            segments.append(escape(normalized[last_index:start]))
        segments.append(f"<b>{escape(match.group(1))}</b>")
        last_index = end
    if last_index < len(normalized):
        segments.append(escape(normalized[last_index:]))
    return "".join(segments).replace("\n", "<br/>")


def _append_paragraph(story: List[object], paragraph_cls, style, text: str) -> None:
    if not text.strip():
        return
    story.append(paragraph_cls(_safe_text(text), style))


def _append_key_value(story: List[object], paragraph_cls, style, label: str, value: object) -> None:
    if not _is_present(value):
        return
    story.append(paragraph_cls(f"<b>{escape(label)}:</b> {_safe_text(value)}", style))


def _build_styles(toolkit: dict[str, object]) -> dict[str, object]:
    colors = toolkit["colors"]
    ParagraphStyle = toolkit["ParagraphStyle"]
    getSampleStyleSheet = toolkit["getSampleStyleSheet"]

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0A6ED1"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#152935"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHeading",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#223548"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCompact",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#223548"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Meta",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#5B738B"),
            alignment=toolkit["TA_CENTER"],
            spaceAfter=2,
        )
    )
    return styles


def _general_info_rows(detail: MetalCompositionItemDetail) -> Sequence[Tuple[str, object]]:
    return [
        ("Product Code", detail.product_code),
        ("Priority", detail.priority),
        ("Business Segment", detail.business_segment),
        ("Site", detail.site),
        ("PN Revised/Standardized", detail.pn_revised_standardized),
        ("Total Weight (Gram)", detail.total_weight_gram),
        ("Part Description", detail.part_description),
        ("New Part Description", detail.new_part_description),
        ("Date Started", detail.date_started),
        ("Date Completed", detail.date_completed),
        ("Priority Detail", detail.priority_detail),
        ("Material Content Method", detail.material_content_method),
        ("Material Identified", detail.material_identified),
    ]


def _metal_breakdown_lines(rows: Iterable[object]) -> List[str]:
    lines: List[str] = []
    for row in rows or []:
        weight_grams = getattr(row, "weight_grams", 0.0)
        if float(weight_grams or 0.0) <= 0.0:
            continue
        grams = _format_grams(weight_grams)
        if grams:
            lines.append(f"{getattr(row, 'type', 'Unknown')}: {grams}")
    return lines


def _other_candidates(
    best_candidate: HTSCandidate | None,
    candidates: Sequence[HTSCandidate] | None,
) -> List[HTSCandidate]:
    best_code = best_candidate.code if best_candidate is not None else ""
    return [candidate for candidate in candidates or [] if candidate.code != best_code]


def build_classification_report_pdf(detail: MetalCompositionItemDetail) -> bytes:
    toolkit = _load_reportlab()
    styles = _build_styles(toolkit)
    SimpleDocTemplate = toolkit["SimpleDocTemplate"]
    Paragraph = toolkit["Paragraph"]
    Spacer = toolkit["Spacer"]
    ListFlowable = toolkit["ListFlowable"]
    ListItem = toolkit["ListItem"]
    inch = toolkit["inch"]

    classification = detail.latest_classification
    if classification is None:
        raise ValueError("This item does not have a saved classification.")

    final_composition = classification.final_composition
    hts = classification.hts_classification
    section_232 = classification.section_232_assessment

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=toolkit["LETTER"],
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=f"{detail.product_code} classification report",
        author="Section 232 Agent",
    )

    story: List[object] = []
    story.append(Paragraph(_safe_text(f"Classification Report - {detail.product_code}"), styles["ReportTitle"]))
    _append_paragraph(story, Paragraph, styles["Meta"], f"Item type: {_format_item_type(detail.item_type)}")
    if _is_present(detail.last_classified_at):
        _append_paragraph(story, Paragraph, styles["Meta"], f"Last classified: {detail.last_classified_at}")
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("1. General Info", styles["SectionHeading"]))
    for label, value in _general_info_rows(detail):
        _append_key_value(story, Paragraph, styles["BodyCompact"], label, value)

    if final_composition is not None:
        story.append(Paragraph("2. Metal Composition", styles["SectionHeading"]))
        _append_key_value(
            story,
            Paragraph,
            styles["BodyCompact"],
            "Metal item",
            _format_bool(final_composition.is_metal_item),
        )
        _append_key_value(
            story,
            Paragraph,
            styles["BodyCompact"],
            "Total metal grams",
            _format_grams(final_composition.estimated_total_metal_grams),
        )

        metal_lines = _metal_breakdown_lines(final_composition.metal_rows)
        if metal_lines:
            story.append(Paragraph("Metal breakdown", styles["SubHeading"]))
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(_safe_text(line), styles["BodyCompact"])) for line in metal_lines],
                    bulletType="bullet",
                    leftIndent=14,
                )
            )

        steel_lines = _metal_breakdown_lines(final_composition.steel_subtype_rows)
        if steel_lines:
            story.append(Paragraph("Steel subtype breakdown", styles["SubHeading"]))
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(_safe_text(line), styles["BodyCompact"])) for line in steel_lines],
                    bulletType="bullet",
                    leftIndent=14,
                )
            )

        if _is_present(final_composition.reasoning):
            story.append(Paragraph("Reasoning", styles["SubHeading"]))
            _append_paragraph(story, Paragraph, styles["BodyCompact"], final_composition.reasoning)

    if hts is not None:
        best_candidate = hts.best_candidate
        story.append(Paragraph("3. HTS Code", styles["SectionHeading"]))

        if best_candidate is not None:
            story.append(Paragraph("3.1 Best Candidate", styles["SubHeading"]))
            _append_key_value(story, Paragraph, styles["BodyCompact"], "Code", best_candidate.code)
            _append_key_value(story, Paragraph, styles["BodyCompact"], "Description", best_candidate.description)
            _append_key_value(
                story,
                Paragraph,
                styles["BodyCompact"],
                "Confidence",
                _format_confidence(best_candidate.confidence),
            )
            if _is_present(best_candidate.reasoning):
                _append_key_value(
                    story,
                    Paragraph,
                    styles["BodyCompact"],
                    "Candidate reasoning",
                    best_candidate.reasoning,
                )
            if _is_present(hts.reasoning):
                _append_key_value(
                    story,
                    Paragraph,
                    styles["BodyCompact"],
                    "Overall HTS reasoning",
                    hts.reasoning,
                )

        secondary_candidates = _other_candidates(best_candidate, hts.candidates)
        if secondary_candidates:
            story.append(Paragraph("3.2 Other Candidates", styles["SubHeading"]))
            for index, candidate in enumerate(secondary_candidates, start=1):
                story.append(Paragraph(f"Candidate {index}", styles["SubHeading"]))
                _append_key_value(story, Paragraph, styles["BodyCompact"], "Code", candidate.code)
                _append_key_value(story, Paragraph, styles["BodyCompact"], "Description", candidate.description)
                _append_key_value(
                    story,
                    Paragraph,
                    styles["BodyCompact"],
                    "Confidence",
                    _format_confidence(candidate.confidence),
                )
                _append_key_value(
                    story,
                    Paragraph,
                    styles["BodyCompact"],
                    "Reasoning",
                    candidate.reasoning or candidate.description,
                )

    if section_232 is not None:
        story.append(Paragraph("4. Section 232 Eligibility", styles["SectionHeading"]))
        _append_key_value(story, Paragraph, styles["BodyCompact"], "Decision", _format_label(section_232.decision))
        _append_key_value(
            story,
            Paragraph,
            styles["BodyCompact"],
            "Confidence",
            _format_confidence(section_232.confidence),
        )
        if _is_present(section_232.basis_summary):
            _append_key_value(
                story,
                Paragraph,
                styles["BodyCompact"],
                "Reasoning",
                section_232.basis_summary,
            )

    document.build(story)
    return buffer.getvalue()
