"""Central English and Italian labels for deterministic assessment reports."""

from __future__ import annotations

from typing import Literal

from app.models.reports import AssessmentReportMetric
from app.services.benchmark_aggregation import rounded_relative_difference


REPORT_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "product": "Evaluation Assessment Assistant",
        "report_title": "Assessment Benchmark Report",
        "report_subtitle": "Executive positioning against the active NACE peer cohort",
        "assessment": "Assessment",
        "assessed_company": "Assessed company",
        "source_company_id": "Source company ID",
        "company_class": "Company class",
        "nace1": "NACE-1 cohort",
        "dataset_date": "Benchmark dataset date",
        "generation_date": "Snapshot generation date",
        "dataset_unavailable": "No active benchmark dataset",
        "overall_overview": "Overall Performance Overview",
        "dimension_summary": "Dimension Positioning Summary",
        "dimension_analysis": "Dimension Analysis",
        "topic_positioning": "Topic positioning and commentary",
        "methodology": "Methodology and Data Provenance",
        "best_peer": "Best Peer",
        "peer_average": "Peer Average",
        "company": "Company",
        "unavailable": "Unavailable",
        "peer_unavailable": "Peer comparison unavailable",
        "page": "Page",
        "method.cohort": "The cohort is defined by the exact persisted company class and NACE-1 combination shown above.",
        "method.selection": "Only the latest released submission for each source company is considered. The assessed source company is excluded, and peer values are disclosed only when the cohort meets the internal eligibility threshold.",
        "method.statistics": "Peer average is the arithmetic mean for each scope. Best peer is the independent maximum for that same scope.",
        "method.thresholds": "Positioning uses the inclusive relative rule: below peers is under 90% of the peer average, in line is from 90% through 110%, and above peers is over 110%.",
        "method.missing": "When the minimum sample or a scope value is unavailable, peer bars are omitted and missing values are never converted to zero.",
    },
    "it": {
        "product": "Evaluation Assessment Assistant",
        "report_title": "Report di benchmark della valutazione",
        "report_subtitle": "Posizionamento direzionale rispetto alla coorte peer NACE attiva",
        "assessment": "Valutazione",
        "assessed_company": "Azienda valutata",
        "source_company_id": "ID azienda sorgente",
        "company_class": "Classe aziendale",
        "nace1": "Coorte NACE-1",
        "dataset_date": "Data del dataset di benchmark",
        "generation_date": "Data di generazione dell'istantanea",
        "dataset_unavailable": "Nessun dataset di benchmark attivo",
        "overall_overview": "Panoramica delle prestazioni complessive",
        "dimension_summary": "Riepilogo del posizionamento per dimensione",
        "dimension_analysis": "Analisi per dimensione",
        "topic_positioning": "Posizionamento e commento per tema",
        "methodology": "Metodologia e provenienza dei dati",
        "best_peer": "Miglior peer",
        "peer_average": "Media dei peer",
        "company": "Azienda",
        "unavailable": "Non disponibile",
        "peer_unavailable": "Confronto con i peer non disponibile",
        "page": "Pagina",
        "method.cohort": "La coorte è definita dalla combinazione esatta tra classe aziendale persistita e NACE-1 mostrata sopra.",
        "method.selection": "Viene considerata solo l'ultima compilazione rilasciata per ogni azienda sorgente. L'azienda valutata viene esclusa e i valori peer vengono mostrati solo quando la coorte soddisfa la soglia interna di idoneità.",
        "method.statistics": "La media dei peer è la media aritmetica per ciascun ambito. Il miglior peer è il massimo indipendente nello stesso ambito.",
        "method.thresholds": "Il posizionamento usa la regola relativa inclusiva: sotto i peer è inferiore al 90% della media, in linea va dal 90% al 110% inclusi e sopra i peer supera il 110%.",
        "method.missing": "Quando il campione minimo o un valore di ambito non è disponibile, le barre peer vengono omesse e i valori mancanti non diventano mai zero.",
    },
}
"""Static localized report copy keyed by normalized language and semantic ID."""


def report_text(language: str, key: str) -> str:
    """Return one deterministic localized report label.

    Inputs:
        language: Normalized report language, ``en`` or ``it``.
        key: Semantic label key present in ``REPORT_LABELS``.

    Outputs:
        str: Localized label, falling back to English for unknown language codes.

    Raises:
        KeyError: If the semantic label key does not exist.
    """

    return REPORT_LABELS["it" if language == "it" else "en"][key]


def pdf_safe_text(value: object) -> str:
    """Normalize text to glyphs supported by FPDF's portable core font.

    Inputs:
        value: Arbitrary label or persisted text rendered into the report.

    Outputs:
        str: Latin-1-safe text using ASCII hyphens and replacement characters
        only when a source glyph has no portable equivalent.
    """

    normalized = str(value).translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u2013": "-",
                "\u2014": "-",
                "\u2026": "...",
            }
        )
    )
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


def positioning_summary_sentence(
    language: str,
    subject_name: str,
    subject_kind: Literal["dimension", "topic"],
    metric: AssessmentReportMetric,
) -> str:
    """Return one localized narrative sentence for a benchmark comparison.

    Inputs:
        language: Normalized report language, ``en`` or ``it``.
        subject_name: Frozen localized dimension name or question-code/topic label.
        subject_kind: Whether the sentence describes a ``dimension`` or ``topic``.
        metric: Frozen company and peer values for the subject.

    Outputs:
        str: Score-safe narrative using the shared relative-difference rounding.
    """

    italian = language == "it"
    if subject_kind == "dimension":
        subject = (
            f"La dimensione {subject_name}"
            if italian
            else f"The {subject_name} dimension"
        )
        unavailable_subject = (
            f"la dimensione {subject_name}"
            if italian
            else f"the {subject_name} dimension"
        )
        positioned = "posizionata"
    else:
        subject = f"Il tema {subject_name}" if italian else f"Topic {subject_name}"
        unavailable_subject = (
            f"il tema {subject_name}" if italian else f"topic {subject_name}"
        )
        positioned = "posizionato"

    peer_average = metric.peer_average
    if peer_average is None or metric.positioning == "unavailable":
        return (
            f"Il confronto con i peer non è disponibile per {unavailable_subject}."
            if italian
            else f"Peer comparison is unavailable for {unavailable_subject}."
        )
    if peer_average == 0:
        if metric.positioning == "in_line_with_peers":
            position = (
                f"{subject} ha eguagliato la media dei peer"
                if italian
                else f"{subject} matched the peer average"
            )
        else:
            position = (
                f"{subject} si è {positioned} sopra la media dei peer"
                if italian
                else f"{subject} scored above the peer average"
            )
        suffix = (
            "la differenza percentuale relativa non è applicabile."
            if italian
            else "relative percentage is not applicable."
        )
        return f"{position}; {suffix}"

    difference = rounded_relative_difference(
        metric.company_score,
        peer_average,
    )
    if difference is None:
        return (
            f"{subject} ha eguagliato la media dei peer."
            if italian
            else f"{subject} matched the peer average."
        )

    magnitude, direction = difference
    if italian:
        percentage = "meno dell'1%" if magnitude == "less_than_one" else f"{magnitude}%"
        if metric.positioning == "in_line_with_peers":
            direction_text = "al di sopra" if direction == "above" else "al di sotto"
            return (
                f"{subject} si è {positioned} in linea con la "
                f"media dei peer ({percentage} {direction_text})."
            )
        comparison = "superiore" if direction == "above" else "inferiore"
        return (
            f"{subject} ha ottenuto un punteggio del "
            f"{percentage} {comparison} alla media dei peer."
        )

    percentage = "less than 1%" if magnitude == "less_than_one" else f"{magnitude}%"
    if metric.positioning == "in_line_with_peers":
        return (
            f"{subject} scored in line with the peer average "
            f"({percentage} {direction})."
        )
    return (
        f"{subject} scored {percentage} {direction} "
        "the peer average."
    )


def invalid_snapshot_message(language: str) -> str:
    """Return a safe localized message for malformed or unsupported snapshots.

    Inputs:
        language: Defensively read raw snapshot language.

    Outputs:
        str: Public rerun instruction without validation or source details.
    """

    if language == "it":
        return "L'istantanea del report non è valida. Genera nuovamente il report."
    return "The report snapshot is invalid. Generate the report again."
