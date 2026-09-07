"""Tests for the deterministic score-safe assessment PDF renderer."""

from __future__ import annotations

import re
from io import BytesIO
from inspect import getsource

import pytest
from fpdf import FPDF
from pypdf import PdfReader

from app.services import (
    assessment_report_pdf,
    assessment_report_pdf_charts,
    assessment_report_pdf_sections,
)
from app.services.assessment_report_pdf import render_assessment_report_pdf
from app.services.assessment_report_localization import positioning_summary_sentence
from app.services.assessment_report_pdf_charts import (
    BEST_PEER_COLOR,
    COMPANY_COLOR,
    PEER_AVERAGE_COLOR,
    build_nested_bar_layers,
)
from app.services.assessment_report_pdf_theme import (
    BONE,
    CHARCOAL,
    LIGHT_YELLOW_ASSESSMENT,
    MINDARO,
    MOONSTONE,
    REPORT_FONT_FAMILY,
    register_report_fonts,
)
from app.services.customer_class_scope import (
    customer_class_options,
    load_customer_class_scope,
)
from app.services.framework_importer import ITALIAN_DIMENSION_FILES
from app.services.framework_topics import topic_title_for
from scripts import render_assessment_report_fixtures
from tests.report_fixtures import make_class3_source, make_metric


def _page_texts(pdf_bytes: bytes) -> list[str]:
    """Extract normalized text from every rendered PDF page.

    Inputs:
        pdf_bytes: Complete report PDF content.

    Outputs:
        list[str]: One whitespace-normalized text string per page.
    """

    reader = PdfReader(BytesIO(pdf_bytes))
    return [re.sub(r"\s+", " ", page.extract_text() or "").strip() for page in reader.pages]


@pytest.mark.parametrize(
    (
        "language",
        "company_score",
        "peer_average",
        "best_peer",
        "sample_size",
        "expected",
    ),
    [
        (
            "en",
            68.5,
            50.0,
            91.0,
            4,
            "The Strategy dimension scored 37% above the peer average.",
        ),
        (
            "en",
            40.0,
            50.0,
            91.0,
            4,
            "The Strategy dimension scored 20% below the peer average.",
        ),
        (
            "en",
            51.5,
            50.0,
            91.0,
            4,
            "The Strategy dimension scored in line with the peer average (3% above).",
        ),
        (
            "en",
            50.0,
            50.0,
            91.0,
            4,
            "The Strategy dimension matched the peer average.",
        ),
        (
            "en",
            49.8,
            50.0,
            91.0,
            4,
            "The Strategy dimension scored in line with the peer average "
            "(less than 1% below).",
        ),
        (
            "en",
            1.0,
            0.0,
            1.0,
            4,
            "The Strategy dimension scored above the peer average; "
            "relative percentage is not applicable.",
        ),
        (
            "en",
            40.0,
            None,
            None,
            0,
            "Peer comparison is unavailable for the Strategy dimension.",
        ),
        (
            "it",
            60.0,
            50.0,
            91.0,
            4,
            "La dimensione Strategia ha ottenuto un punteggio del 20% "
            "superiore alla media dei peer.",
        ),
        (
            "it",
            49.0,
            50.0,
            91.0,
            4,
            "La dimensione Strategia si è posizionata in linea con la media "
            "dei peer (2% al di sotto).",
        ),
        (
            "it",
            50.0,
            50.0,
            91.0,
            4,
            "La dimensione Strategia ha eguagliato la media dei peer.",
        ),
        (
            "it",
            49.8,
            50.0,
            91.0,
            4,
            "La dimensione Strategia si è posizionata in linea con la media "
            "dei peer (meno dell'1% al di sotto).",
        ),
        (
            "it",
            0.0,
            0.0,
            0.0,
            4,
            "La dimensione Strategia ha eguagliato la media dei peer; "
            "la differenza percentuale relativa non è applicabile.",
        ),
        (
            "it",
            40.0,
            None,
            None,
            0,
            "Il confronto con i peer non è disponibile per la dimensione Strategia.",
        ),
    ],
)
def test_positioning_summary_sentence_uses_shared_relative_rounding(
    language: str,
    company_score: float,
    peer_average: float | None,
    best_peer: float | None,
    sample_size: int,
    expected: str,
) -> None:
    """Verify localized narrative wording for every comparison edge case."""

    metric = make_metric(
        company_score,
        peer_average=peer_average,
        best_peer=best_peer,
        sample_size=sample_size,
        language=language,
    )

    dimension_name = "Strategia" if language == "it" else "Strategy"
    assert (
        positioning_summary_sentence(
            language,
            dimension_name,
            "dimension",
            metric,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("language", "title", "summary_sentence", "table_header", "analysis_title"),
    [
        (
            "en",
            "Assessment Benchmark Report",
            "The Strategy dimension scored 23% below the peer average.",
            "Dimension Positioning Commentary",
            "Dimension Analysis: Strategy",
        ),
        (
            "it",
            "Report di benchmark della valutazione",
            "La dimensione Strategia ha ottenuto un punteggio del 23% "
            "inferiore alla media dei peer.",
            "Dimensione Posizionamento Commento",
            "Analisi per dimensione: Strategia",
        ),
    ],
)
def test_opening_page_merges_identity_overview_and_narrative_summary(
    repo_root,
    language: str,
    title: str,
    summary_sentence: str,
    table_header: str,
    analysis_title: str,
) -> None:
    """Verify the former cover content shares page one with report analysis."""

    long_nace = "Synthetic professional, scientific and technical activities"
    source = make_class3_source(repo_root, language=language)
    source = source.model_copy(
        update={
            "nace1": long_nace,
            "provenance": source.provenance.model_copy(update={"nace1": long_nace}),
        }
    )
    pages = _page_texts(render_assessment_report_pdf(source))

    assert len(pages) == 9
    assert all(
        expected in pages[0]
        for expected in (
            title,
            "Benchmark assessment for class 3",
            long_nace,
            "Overall Performance Overview"
            if language == "en"
            else "Panoramica delle prestazioni complessive",
            "Dimension Positioning Summary"
            if language == "en"
            else "Riepilogo del posizionamento per dimensione",
            summary_sentence,
        )
    )
    assert table_header not in pages[0]
    assert analysis_title in pages[1]


@pytest.mark.parametrize(
    ("language", "sentence", "table_header"),
    [
        (
            "en",
            "Topic Q.STR.01.01 - Strategic Planning scored 23% below the peer average.",
            "Topic Positioning Commentary",
        ),
        (
            "it",
            "Il tema Q.STR.01.01 - Pianificazione Strategica ha ottenuto un "
            "punteggio del 23% inferiore alla media dei peer.",
            "Tema Posizionamento Commento",
        ),
    ],
)
def test_topic_positioning_summary_uses_narrative_lines(
    repo_root,
    language: str,
    sentence: str,
    table_header: str,
) -> None:
    """Verify each dimension replaces its topic table with localized prose."""

    pages = _page_texts(
        render_assessment_report_pdf(make_class3_source(repo_root, language=language))
    )

    assert sentence in pages[1]
    assert table_header not in pages[1]
    assert pages[1].count("Q.STR.01.01") == 2


@pytest.mark.parametrize(
    ("language", "removed_sentence", "table_header"),
    [
        (
            "en",
            "The report uses the import captured when the job was queued.",
            "Scope Title",
        ),
        (
            "it",
            "Il report usa l'importazione acquisita al momento dell'accodamento.",
            "Ambito Titolo",
        ),
    ],
)
def test_methodology_omits_snapshot_sentence_and_profile_table(
    repo_root,
    language: str,
    removed_sentence: str,
    table_header: str,
) -> None:
    """Verify methodology stops before snapshot wording and repeated metadata."""

    source = make_class3_source(repo_root, language=language)
    methodology_page = _page_texts(render_assessment_report_pdf(source))[-1]

    assert removed_sentence not in methodology_page
    assert table_header not in methodology_page
    assert source.customer_class not in methodology_page
    assert source.nace1 not in methodology_page


def test_nested_bar_geometry_uses_fixed_scale_order_and_missing_peer_omission() -> None:
    """Verify layer order, thickness, colors, clamping, and null handling."""

    metric = make_metric(100.0, peer_average=48.2345, best_peer=91.3456)
    metric = metric.model_copy(update={"company_score": 120.0})
    layers = build_nested_bar_layers(metric, plot_width=100.0)

    assert [layer.series for layer in layers] == [
        "best_peer",
        "peer_average",
        "company",
    ]
    assert [layer.color for layer in layers] == [
        BEST_PEER_COLOR,
        PEER_AVERAGE_COLOR,
        COMPANY_COLOR,
    ]
    assert layers[0].thickness > layers[1].thickness > layers[2].thickness
    assert layers[2].width == 100.0

    missing_peer_layers = build_nested_bar_layers(
        make_metric(
            37.1234,
            peer_average=None,
            best_peer=None,
            sample_size=0,
        ),
        plot_width=100.0,
    )
    assert [layer.series for layer in missing_peer_layers] == ["company"]
    assert missing_peer_layers[0].width == pytest.approx(37.1234)


def test_chart_wrap_measurement_reserves_fpdf_cell_margins() -> None:
    """Keep a third Italian label line inside its measured chart row."""

    pdf = assessment_report_pdf.AssessmentReportPdf("it")
    pdf.add_page()
    pdf.set_font(REPORT_FONT_FAMILY, size=7.5)
    lines = assessment_report_pdf_charts._wrap_text(
        pdf,
        (
            "Riservatezza delle informazioni aziendali nei processi interni, "
            "esterni e nelle responsabilità di governance delegate"
        ),
        68.0,
    )

    assert len(lines) == 3
    assert lines[-1] == "governance delegate"


def test_section_heading_omits_requested_spacing_after_adding_a_page() -> None:
    """Verify a page-aware heading starts without a redundant top-page gap."""

    page_break_pdf = FPDF(orientation="P", unit="mm", format="A4")
    register_report_fonts(page_break_pdf)
    page_break_pdf.set_margins(14, 16, 14)
    page_break_pdf.set_auto_page_break(auto=True, margin=16)
    page_break_pdf.add_page()
    page_break_pdf.set_y(page_break_pdf.h - page_break_pdf.b_margin - 30)
    assessment_report_pdf_sections._section_heading(
        page_break_pdf,
        "Summary",
        spacing_before=6,
    )

    fresh_page_pdf = FPDF(orientation="P", unit="mm", format="A4")
    register_report_fonts(fresh_page_pdf)
    fresh_page_pdf.set_margins(14, 16, 14)
    fresh_page_pdf.set_auto_page_break(auto=True, margin=16)
    fresh_page_pdf.add_page()
    assessment_report_pdf_sections._section_heading(fresh_page_pdf, "Summary")

    assert page_break_pdf.page_no() == 2
    assert page_break_pdf.get_y() == pytest.approx(fresh_page_pdf.get_y())


@pytest.mark.parametrize(
    "title",
    (
        "Dimension Analysis: Combined Assurance & Management Oversight",
        "Analisi per dimensione: Combined Assurance & Management Oversight",
    ),
)
def test_long_montserrat_section_heading_stays_on_one_line(title: str) -> None:
    """Keep the longest localized dimension headings free of orphan lines."""

    pdf = assessment_report_pdf.AssessmentReportPdf("en")
    pdf.add_page()
    start_y = pdf.get_y()

    assessment_report_pdf_sections._section_heading(pdf, title)

    assert pdf.get_y() - start_y == pytest.approx(11)


def test_report_embeds_montserrat_and_uses_assessment_palette(repo_root) -> None:
    """Verify generated reports carry the supplied assessment framework typography and colors."""

    report = PdfReader(BytesIO(render_assessment_report_pdf(make_class3_source(repo_root))))
    font_names = {
        str(font.get_object().get("/BaseFont", ""))
        for page in report.pages
        for font in page["/Resources"].get("/Font", {}).values()
    }

    assert any("Montserrat" in font_name for font_name in font_names)
    assert all("Helvetica" not in font_name for font_name in font_names)
    assert {
        "charcoal": CHARCOAL,
        "light_yellow_assessment": LIGHT_YELLOW_ASSESSMENT,
        "mindaro": MINDARO,
        "moonstone": MOONSTONE,
        "bone": BONE,
    } == {
        "charcoal": (44, 62, 80),
        "light_yellow_assessment": (255, 204, 0),
        "mindaro": (195, 233, 145),
        "moonstone": (81, 175, 184),
        "bone": (236, 226, 208),
    }


def test_fixture_cli_uses_configured_customer_class_choices_and_default() -> None:
    """Verify the fixture CLI follows the shared editable class configuration."""

    configured_classes = [
        option["value"] for option in customer_class_options("en")
    ]
    assert render_assessment_report_fixtures.parse_args(
        []
    ).customer_class == load_customer_class_scope()["default_customer_class"]
    assert [
        render_assessment_report_fixtures.parse_args(
            ["--customer-class", customer_class]
        ).customer_class
        for customer_class in configured_classes
    ] == configured_classes
    with pytest.raises(SystemExit):
        render_assessment_report_fixtures.parse_args(
            ["--customer-class", "unknown-class"]
        )


@pytest.mark.parametrize(
    "customer_class",
    [option["value"] for option in customer_class_options("en")],
)
def test_fixture_renderer_supports_every_configured_customer_class(
    repo_root,
    tmp_path,
    customer_class: str,
) -> None:
    """Verify the visual fixture path renders every configured class scope."""

    output_path = tmp_path / f"assessment-report-{customer_class}.pdf"
    page_count = render_assessment_report_fixtures.render_fixture(
        repo_root=repo_root,
        output_path=output_path,
        customer_class=customer_class,
        language="en",
    )

    assert page_count >= 1
    assert output_path.is_file()


@pytest.mark.parametrize(
    ("language", "section_titles", "forbidden_titles", "commentary"),
    [
        (
            "en",
            (
                "Overall Performance Overview",
                "Dimension Positioning Summary",
                "Dimension Analysis",
                "Methodology and Data Provenance",
            ),
            (
                "Executive synthesis",
                "AI-generated improvement suggestions",
                "Detailed Appendix",
                "Company score",
            ),
            "The Strategy dimension scored 23% below the peer average.",
        ),
        (
            "it",
            (
                "Panoramica delle prestazioni complessive",
                "Riepilogo del posizionamento per dimensione",
                "Analisi per dimensione",
                "Metodologia e provenienza dei dati",
            ),
            (
                "Sintesi direzionale",
                "Suggerimenti di miglioramento generati dall'IA",
                "Appendice dettagliata",
                "Punteggio azienda",
            ),
            "La dimensione Strategia ha ottenuto un punteggio del 23% "
            "inferiore alla media dei peer.",
        ),
    ],
)
def test_report_is_deterministic_and_never_discloses_exact_scores(
    repo_root,
    language: str,
    section_titles: tuple[str, ...],
    forbidden_titles: tuple[str, ...],
    commentary: str,
) -> None:
    """Verify bilingual structure, percentage commentary, and score hiding."""

    source = make_class3_source(repo_root, language=language)
    assert len(source.dimensions) == 7
    assert sum(len(dimension.topics) for dimension in source.dimensions) == 38

    page_texts = _page_texts(render_assessment_report_pdf(source))
    all_text = " ".join(page_texts)

    assert len(page_texts) > 6
    assert all(title in all_text for title in section_titles)
    assert all(title not in all_text for title in forbidden_titles)
    assert commentary in all_text
    assert all(
        exact_value not in all_text
        for exact_value in ("37.1234", "48.2345", "91.3456")
    )
    for internal_label in (
        "Peer sample",
        "Campione peer",
        "Import ID",
        "ID importazione",
        "Source filename",
        "Nome file sorgente",
        "SHA-256",
        "Scoring version",
        "Versione dello scoring",
    ):
        assert internal_label not in all_text
    assert all(
        internal_value not in all_text
        for internal_value in (
            source.provenance.import_id,
            source.provenance.source_filename,
            source.provenance.source_sha256,
            source.provenance.scoring_version,
        )
        if internal_value
    )
    for dimension in source.dimensions:
        assert dimension.display_name in all_text
        for topic in dimension.topics:
            assert topic.question_id in all_text
            assert topic.topic_title in all_text


@pytest.mark.parametrize(
    ("language", "unavailable"),
    [("en", "Peer comparison unavailable"), ("it", "Confronto con i peer non disponibile")],
)
def test_missing_peer_report_omits_peer_layers_and_exact_values(
    repo_root,
    language: str,
    unavailable: str,
) -> None:
    """Verify null peers remain absent and raw scores are never rendered."""

    source = make_class3_source(repo_root, language=language, include_peers=False)
    pdf_text = " ".join(_page_texts(render_assessment_report_pdf(source)))

    assert unavailable in pdf_text
    assert all(
        [layer.series for layer in build_nested_bar_layers(topic.metric, 100.0)]
        == ["company"]
        for dimension in source.dimensions
        for topic in dimension.topics
    )
    assert "37.1234" not in pdf_text


def test_renderer_source_has_no_ai_narrative_or_exact_appendix_contract() -> None:
    """Verify removed AI and exact-score report paths cannot be rendered."""

    source_text = "\n".join(
        getsource(module)
        for module in (
            assessment_report_pdf,
            assessment_report_pdf_charts,
            assessment_report_pdf_sections,
        )
    )

    for forbidden in (
        "Executive synthesis",
        "AI-generated improvement suggestions",
        "write_ai_recommendations",
        "write_detailed_appendix",
        "QUESTION_APPENDIX",
        "peer_sample_size",
        "scoring_version",
        "same_sector",
        "same_size",
        "source_filename",
        "source_sha256",
    ):
        assert forbidden not in source_text


def test_italian_fixture_uses_localized_dimension_and_topic_titles(repo_root) -> None:
    """Verify the representative Italian source freezes localized titles."""

    source = make_class3_source(repo_root, language="it")
    expected_dimensions = {labels[1] for labels in ITALIAN_DIMENSION_FILES.values()}

    assert {dimension.display_name for dimension in source.dimensions} == expected_dimensions
    assert all(
        topic.topic_title == topic_title_for(topic.question_id, "it")
        or topic.question_id == "Q.FDR.12.01"
        for dimension in source.dimensions
        for topic in dimension.topics
    )
