"""Tests for canonical bilingual question-specific assessment topic titles."""

from __future__ import annotations

from pathlib import Path
from inspect import getsource

from app.services.benchmark_import.scoring import calculate_benchmark_scores
from app.services.framework_importer import (
    load_framework_seed,
    load_italian_framework_translations,
)
from app.services.hana_schema import HANA_SCHEMA_STATEMENTS, required_table_names
from scripts import import_assessment_framework


EXPECTED_TOPIC_TITLES = {
    "Q.STR.01.01": {
        "en": "Strategic Planning",
        "it": "Pianificazione Strategica",
    },
    "Q.STR.02.01": {
        "en": "Business Plan Risk Analysis",
        "it": "Analisi Rischi di Piano",
    },
    "Q.STR.03.01": {
        "en": "Definition of Objectives",
        "it": "Definizione degli Obiettivi",
    },
    "Q.RCG.01.01": {
        "en": "Control Levels",
        "it": "Livelli di Controllo",
    },
    "Q.RCG.02.01": {
        "en": "Independence of Control Functions",
        "it": "Indipendenza Funzioni di Controllo",
    },
    "Q.RCG.03.01": {
        "en": "Control Design Responsibility",
        "it": "Responsabilità Disegno dei Controlli",
    },
    "Q.RCG.04.01": {
        "en": "Financial Reporting",
        "it": "Informativa Finanziaria",
    },
    "Q.RCG.05.01": {
        "en": "Corporate Crime Prevention Model",
        "it": "MOG 231",
    },
    "Q.RCG.06.01": {
        "en": "Whistleblowing",
        "it": "Whistleblowing",
    },
    "Q.RCG.07.01": {
        "en": "Anti-Corruption",
        "it": "Anticorruzione",
    },
    "Q.RCG.08.01": {
        "en": "Conflict of Interest",
        "it": "Conflitto di Interessi",
    },
    "Q.RCG.09.01": {
        "en": "Enterprise Risk Management",
        "it": "Enterprise Risk Management",
    },
    "Q.RCG.10.01": {
        "en": "Risk Management Methodologies",
        "it": "Metodologie di Risk Management",
    },
    "Q.RCG.11.01": {
        "en": "Risk Management Systems",
        "it": "Sistemi di Risk Management",
    },
    "Q.RCG.12.01": {
        "en": "ICRMS Evaluation",
        "it": "Valutazione SCIGR",
    },
    "Q.ORG.01.01": {
        "en": "Delegations and Powers of Attorney",
        "it": "Deleghe e Procure",
    },
    "Q.ORG.02.01": {
        "en": "Organizational Structure",
        "it": "Assetto Organizzativo",
    },
    "Q.ORG.03.01": {
        "en": "Segregation of Duties",
        "it": "Segregation of Duties",
    },
    "Q.ORG.04.01": {
        "en": "Internal Regulatory System",
        "it": "Sistema Normativo Interno",
    },
    "Q.PCU.01.01": {
        "en": "Code of Ethics",
        "it": "Codice Etico",
    },
    "Q.PCU.02.01": {
        "en": "Corporate Culture and Values",
        "it": "Cultura Aziendale e Valori",
    },
    "Q.PCU.03.01": {
        "en": "Disciplinary System",
        "it": "Sistema Disciplinare",
    },
    "Q.PCU.04.01": {
        "en": "Skills and Competencies",
        "it": "Skill e Competenze",
    },
    "Q.PCU.05.01": {
        "en": "Talent Management",
        "it": "Talent Management",
    },
    "Q.PCU.06.01": {
        "en": "Awareness of Objectives",
        "it": "Consapevolezza degli Obiettivi",
    },
    "Q.PCU.07.01": {
        "en": "Performance Evaluation",
        "it": "Valutazione delle Performance",
    },
    "Q.CAM.01.01": {
        "en": "Combined Assurance Process",
        "it": "Processo di Combined Assurance",
    },
    "Q.CAM.02.01": {
        "en": "Risk Assessment & Monitoring",
        "it": "Risk Assessment & Monitoring",
    },
    "Q.CAM.03.01": {
        "en": "Planning",
        "it": "Pianificazione",
    },
    "Q.CAM.04.01": {
        "en": "Control Activities",
        "it": "Attività di Controllo",
    },
    "Q.CAM.05.01": {
        "en": "Information Flows",
        "it": "Flussi Informativi",
    },
    "Q.CAM.06.01": {
        "en": "Management Oversight",
        "it": "Management Oversight",
    },
    "Q.CAM.07.01": {
        "en": "Traceability",
        "it": "Tracciabilità",
    },
    "Q.SID.01.01": {
        "en": "IT Governance",
        "it": "Governance IT",
    },
    "Q.SID.02.01": {
        "en": "IT Project Management",
        "it": "IT Project Management",
    },
    "Q.SID.03.01": {
        "en": "Application Management",
        "it": "Application Management",
    },
    "Q.SID.04.01": {
        "en": "IT Asset Management",
        "it": "IT Asset Management",
    },
    "Q.SID.05.01": {
        "en": "Business Continuity",
        "it": "Business Continuity",
    },
    "Q.FDR.01.01": {
        "en": "Supplier Procurement",
        "it": "Procurement Fornitori",
    },
    "Q.FDR.02.01": {
        "en": "Supplier Contract Management",
        "it": "Contract Management Fornitori",
    },
    "Q.FDR.03.01": {
        "en": "Corporate Due Diligence",
        "it": "Corporate Due Diligence",
    },
    "Q.FDR.04.01": {
        "en": "Sustainability Reporting",
        "it": "Rendicontazione di Sostenibilità",
    },
    "Q.FDR.05.01": {
        "en": "HSE",
        "it": "HSE",
    },
    "Q.FDR.06.01": {
        "en": "Cyber Security",
        "it": "Cyber Security",
    },
    "Q.FDR.07.01": {
        "en": "Artificial Intelligence",
        "it": "Artificial Intelligence",
    },
    "Q.FDR.08.01": {
        "en": "Personal Data Protection",
        "it": "Privacy",
    },
    "Q.FDR.09.01": {
        "en": "Financial Risk",
        "it": "Financial Risk",
    },
    "Q.FDR.10.01": {
        "en": "Tax Control Framework",
        "it": "Tax Control Framework",
    },
    "Q.FDR.11.01": {
        "en": "Stakeholder Management",
        "it": "Stakeholder Management",
    },
    "Q.FDR.12.01": {
        "en": "Confidentiality of Company Information",
        "it": "Confidenzialità delle Informazioni",
    },
}
"""Exact approved one-to-one English/Italian title mapping for all questions."""


def test_canonical_topic_title_mapping_is_exact_and_complete() -> None:
    """Verify all 50 approved bilingual titles match the reviewed analysis."""

    from app.services.framework_topics import CANONICAL_TOPIC_TITLES

    assert CANONICAL_TOPIC_TITLES == EXPECTED_TOPIC_TITLES
    assert len(CANONICAL_TOPIC_TITLES) == 50
    assert all(set(titles) == {"en", "it"} for titles in CANONICAL_TOPIC_TITLES.values())


def test_framework_seed_and_italian_translations_expose_question_topics(
    repo_root: Path,
) -> None:
    """Verify canonical/imported framework models expose localized topic titles."""

    from app.services.framework_importer import build_topic_translation_rows

    seed = load_framework_seed(
        repo_root / "data" / "sanitized" / "assessment_framework.xlsx",
        repo_root / "data" / "sanitized" / "assessment_question_explanations.csv",
    )
    translations = load_italian_framework_translations(
        repo_root / "data" / "sanitized" / "IT",
        seed.questions,
    )

    assert {question.question_id: question.topic_title for question in seed.questions} == {
        question_id: titles["en"]
        for question_id, titles in EXPECTED_TOPIC_TITLES.items()
    }
    assert {
        question_id: translation.topic_title
        for question_id, translation in translations.question_translations.items()
    } == {
        question_id: titles["it"]
        for question_id, titles in EXPECTED_TOPIC_TITLES.items()
    }
    topic_rows = build_topic_translation_rows(seed.questions)
    assert len(topic_rows) == 100
    assert topic_rows[0] == {
        "question_id": "Q.STR.01.01",
        "language": "en",
        "topic_title": "Strategic Planning",
    }


def test_topic_titles_use_backward_compatible_hana_translation_table() -> None:
    """Verify existing framework tables need no destructive column migration."""

    schema = "\n".join(HANA_SCHEMA_STATEMENTS).lower()

    assert "assessment_question_topic_translations" in required_table_names()
    topic_sql = schema.split(
        "create table assessment_question_topic_translations", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "question_id nvarchar(64) not null" in topic_sql
    assert "language nvarchar(8) not null" in topic_sql
    assert "topic_title nvarchar(255) not null" in topic_sql
    assert "primary key (question_id, language)" in topic_sql

    writer_source = getsource(import_assessment_framework.write_framework_to_hana)
    assert "delete from assessment_question_topic_translations" in writer_source
    assert "insert into assessment_question_topic_translations" in writer_source


def test_benchmark_topic_scores_use_canonical_question_title_not_section(
    repo_root: Path,
) -> None:
    """Verify imported topic audit text comes from question metadata, never Sezione."""

    seed = load_framework_seed(
        repo_root / "data" / "sanitized" / "assessment_framework.xlsx",
        repo_root / "data" / "sanitized" / "assessment_question_explanations.csv",
    )
    question = next(item for item in seed.questions if item.question_id == "Q.STR.03.01")
    scores = calculate_benchmark_scores(
        import_id="import-1",
        questionnaire_id="questionnaire-1",
        customer_class="class_1",
        questions={question.question_id: question},
        responses=[],
        supplied_dimension_scores={},
    )
    topic = next(score for score in scores if score.scope_type == "topic")

    assert topic.question_id == "Q.STR.03.01"
    assert topic.topic == "Definition of Objectives"
    assert topic.topic != question.section
