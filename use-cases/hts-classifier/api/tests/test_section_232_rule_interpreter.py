from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.services.metal_composition.section_232_rule_interpreter import _normalize_date, interpret_section_232_batch
from app.services.metal_composition.section_232_sources import PersistedSection232Source
from app.services.metal_composition.workflow.token_usage import TokenUsageRecorder


class _FakeLLM:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def invoke_native_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        content = self.payload
        if isinstance(content, dict):
            content = json.dumps(content)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content,
                    )
                )
            ]
        )


class _QueuedFakeLLM:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def invoke_native_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No queued LLM response available")
        content = self.responses.pop(0)
        if isinstance(content, dict):
            content = json.dumps(content)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content,
                    )
                )
            ]
        )


def _source(
    source_id: str,
    filename: str,
    *,
    page_number: int,
    text: str,
    excerpt: str,
) -> PersistedSection232Source:
    return PersistedSection232Source(
        source_id=source_id,
        filename=filename,
        size_bytes=100,
        page_count=1,
        extraction_status="completed",
        full_text=f"[Page {page_number}]\n{text}",
        page_texts=[
            {
                "page_number": page_number,
                "text": text,
                "layout_aware_text": text,
                "page_excerpt": excerpt,
                "char_count": len(text),
                "hts_mentions": [],
            }
        ],
        hts_mentions=[],
        warnings=[],
        content_sha256=f"sha-{source_id}",
        uploaded_at="2026-04-19T10:00:00Z",
    )


def test_interpret_section_232_batch_uses_full_document_prompt_and_normalizes_candidates():
    source_one = _source(
        "source-1",
        "notice-1.pdf",
        page_number=3,
        text="Annex I adds HTS 7308.90.95 at 50 percent ad valorem.",
        excerpt="Annex I adds HTS 7308.90.95 at 50 percent ad valorem.",
    )
    source_two = _source(
        "source-2",
        "notice-2.pdf",
        page_number=7,
        text="HTS 8421.29.00 remains included at a temporary 25 percent rate through December 31, 2026.",
        excerpt="HTS 8421.29.00 remains included at a temporary 25 percent rate.",
    )
    llm = _FakeLLM(
        {
            "results": [
                {
                    "candidate_key": "source-1:7308.90.95",
                    "is_real_hts_candidate": True,
                    "rule_type": "include",
                    "effective_from": "2026-03-12",
                    "effective_to": None,
                    "metal_scope": "steel derivative articles",
                    "rate_text": None,
                    "source_pages": ["3"],
                    "source_excerpt": "Annex I adds HTS 7308.90.95 at 50 percent ad valorem.",
                    "interpreter_confidence": "82%",
                },
                {
                    "candidate_key": "source-2:8421.29.00",
                    "is_real_hts_candidate": True,
                    "rule_type": "temporary_reduced_rate",
                    "effective_from": "April 1, 2026",
                    "effective_to": "12/31/2026",
                    "metal_scope": "aluminum derivatives",
                    "rate_text": "25 percent ad valorem through December 31, 2026",
                    "source_pages": [7],
                    "source_excerpt": "HTS 8421.29.00 remains included at a temporary 25 percent rate.",
                    "interpreter_confidence": 0.67,
                },
            ]
        }
    )

    candidates = interpret_section_232_batch(
        batch_id="batch-123",
        extracted_sources=[source_one, source_two],
        llm=llm,
    )

    assert len(candidates) == 2

    include_candidate, reduced_rate_candidate = candidates

    assert include_candidate.batch_id == "batch-123"
    assert include_candidate.hts_code == "7308.90.95"
    assert include_candidate.rule_type == "include"
    assert include_candidate.coverage_effect == "include"
    assert include_candidate.source_document_ids == ["source-1"]
    assert include_candidate.source_pages == [3]
    assert include_candidate.review_decision == "pending"
    assert include_candidate.effective_from == "2026-03-12"
    assert include_candidate.effective_to is None
    assert include_candidate.rate_text is None
    assert include_candidate.candidate_quality == "normal"
    assert include_candidate.candidate_flags == []

    assert reduced_rate_candidate.rule_type == "rate_schedule"
    assert reduced_rate_candidate.coverage_effect == "include"
    assert reduced_rate_candidate.effective_from == "2026-04-01"
    assert reduced_rate_candidate.effective_to == "2026-12-31"
    assert reduced_rate_candidate.source_document_ids == ["source-2"]
    assert reduced_rate_candidate.source_pages == [7]
    assert reduced_rate_candidate.source_excerpt == "HTS 8421.29.00 remains included at a temporary 25 percent rate."
    assert reduced_rate_candidate.rate_text == "25 percent ad valorem through December 31, 2026"

    assert len(llm.calls) == 2
    first_prompt_text = llm.calls[0]["messages"][1]["content"]
    second_prompt_text = llm.calls[1]["messages"][1]["content"]
    assert "Section 232 source batch" in first_prompt_text
    assert "full_document_text" in first_prompt_text
    assert "layout_aware_text" in first_prompt_text
    assert "page_excerpt" in first_prompt_text
    assert "candidate_key" in first_prompt_text
    assert "source-1" in first_prompt_text
    assert "source-2" in second_prompt_text

    repeated_candidates = interpret_section_232_batch(
        batch_id="batch-123",
        extracted_sources=[source_one, source_two],
        llm=llm,
    )
    assert [candidate.candidate_id for candidate in repeated_candidates] == [
        candidate.candidate_id for candidate in candidates
    ]


def test_interpret_section_232_batch_matches_legacy_results_without_candidate_keys():
    source = _source(
        "source-9",
        "notice-9.pdf",
        page_number=11,
        text="HTS 7616.99.51.60 is included in Annex I.",
        excerpt="HTS 7616.99.51.60 is included in Annex I.",
    )
    llm = _FakeLLM(
        {
            "rules": [
                {
                    "hts_code": "7616.99.51.60",
                    "rule_type": "include",
                    "effective_from": "2026-04-07",
                    "metal_scope": "aluminum",
                    "source_document_ids": ["notice-9.pdf"],
                    "source_pages": ["11"],
                    "interpreter_confidence": "0.55",
                }
            ]
        }
    )

    candidates = interpret_section_232_batch(
        batch_id="batch-999",
        extracted_sources=[source],
        llm=llm,
    )

    assert len(candidates) == 1
    assert candidates[0].hts_code == "7616.99.5160"
    assert candidates[0].source_document_ids == ["source-9"]
    assert candidates[0].source_pages == [11]
    assert candidates[0].source_excerpt == source.page_texts[0]["page_excerpt"]
    assert candidates[0].candidate_quality == "normal"


def test_interpret_section_232_batch_downgrades_unrecognized_metal_scope_to_unspecified():
    source = _source(
        "source-20",
        "notice-20.pdf",
        page_number=5,
        text="HTS 8302.41.00 is included in the notice.",
        excerpt="HTS 8302.41.00 is included in the notice.",
    )
    llm = _FakeLLM(
        {
            "results": [
                {
                    "candidate_key": "source-20:8302.41.00",
                    "is_real_hts_candidate": True,
                    "rule_type": "include",
                    "effective_from": None,
                    "metal_scope": "applies to all covered products in the notice",
                    "source_pages": [5],
                }
            ]
        }
    )

    candidates = interpret_section_232_batch(
        batch_id="batch-200",
        extracted_sources=[source],
        llm=llm,
    )

    assert len(candidates) == 1
    assert candidates[0].metal_scope == "unspecified"
    assert candidates[0].effective_from is None
    assert candidates[0].effective_to is None
    assert candidates[0].candidate_flags == []


def test_interpret_section_232_batch_marks_admin_reference_candidates_as_suspect():
    source = _source(
        "source-30",
        "notice-30.pdf",
        page_number=2,
        text="Docket No. 250226-0029. Submit comments before the close of business.",
        excerpt="Docket No. 250226-0029. Submit comments before the close of business.",
    )
    llm = _FakeLLM(
        {
            "results": [
                {
                    "candidate_key": "source-30:2502.26",
                    "is_real_hts_candidate": False,
                    "rule_type": None,
                    "effective_from": None,
                    "effective_to": None,
                    "metal_scope": None,
                    "rate_text": None,
                    "source_pages": [2],
                    "source_excerpt": "Docket number only; not an HTS code.",
                    "interpreter_confidence": 0.2,
                }
            ]
        }
    )

    candidates = interpret_section_232_batch(
        batch_id="batch-suspect",
        extracted_sources=[source],
        llm=llm,
    )

    assert len(candidates) == 1
    assert candidates[0].hts_code == "2502.26"
    assert candidates[0].candidate_quality == "suspect"
    assert "administrative_reference_context" in candidates[0].candidate_flags
    assert "llm_marked_not_real_hts_candidate" in candidates[0].candidate_flags
    assert "rule_type_undefined" in candidates[0].candidate_flags


def test_interpret_section_232_batch_retries_with_page_chunks_but_keeps_full_document_context():
    page_texts = []
    expected_codes = []
    for index in range(41):
        code = f"7308.90.{index:04d}"
        expected_codes.append(code)
        page_texts.append(
            {
                "page_number": 1,
                "text": f"HTS {code} remains covered.",
                "layout_aware_text": f"HTS {code} remains covered.",
                "page_excerpt": f"HTS {code} remains covered.",
                "char_count": 30,
                "hts_mentions": [code],
            }
        )

    source = PersistedSection232Source(
        source_id="source-11",
        filename="huge-notice.pdf",
        size_bytes=100,
        page_count=1,
        extraction_status="completed",
        full_text="Large notice with page markers",
        page_texts=page_texts,
        hts_mentions=expected_codes,
        warnings=[],
        content_sha256="sha-source-11",
        uploaded_at="2026-04-19T10:00:00Z",
    )
    first_chunk = [
        {
            "candidate_key": f"source-11:{code}",
            "is_real_hts_candidate": True,
            "rule_type": "include",
            "effective_from": None,
            "effective_to": None,
            "metal_scope": "steel",
            "rate_text": None,
            "source_pages": [1],
            "source_excerpt": "The listed HTS codes remain covered.",
            "interpreter_confidence": 0.9,
        }
        for code in expected_codes[:40]
    ]
    second_chunk = [
        {
            "candidate_key": f"source-11:{expected_codes[40]}",
            "is_real_hts_candidate": True,
            "rule_type": "include",
            "effective_from": None,
            "effective_to": None,
            "metal_scope": "steel",
            "rate_text": None,
            "source_pages": [1],
            "source_excerpt": "The listed HTS codes remain covered.",
            "interpreter_confidence": 0.9,
        }
    ]
    llm = _QueuedFakeLLM(
        [
            '{"results": [{"candidate_key": "source-11:7308.90.0000"',
            {"results": first_chunk},
            {"results": second_chunk},
        ]
    )

    candidates = interpret_section_232_batch(
        batch_id="batch-large",
        extracted_sources=[source],
        llm=llm,
    )

    assert [candidate.hts_code for candidate in candidates] == expected_codes
    assert len(llm.calls) == 3
    assert "full_document_text" in llm.calls[1]["messages"][1]["content"]
    assert "full_document_text" in llm.calls[2]["messages"][1]["content"]
    assert "candidate-chunk" not in llm.calls[1]["messages"][1]["content"]


def test_interpret_section_232_batch_passes_usage_recorder_to_primary_and_chunked_calls():
    """Verify Section 232 ingestion LLM calls receive the request usage recorder."""

    page_texts = []
    expected_codes = []
    for index in range(41):
        code = f"7308.90.{index:04d}"
        expected_codes.append(code)
        page_texts.append(
            {
                "page_number": 1,
                "text": f"HTS {code} remains covered.",
                "layout_aware_text": f"HTS {code} remains covered.",
                "page_excerpt": f"HTS {code} remains covered.",
                "char_count": 30,
                "hts_mentions": [code],
            }
        )

    source = PersistedSection232Source(
        source_id="source-usage",
        filename="usage-notice.pdf",
        size_bytes=100,
        page_count=1,
        extraction_status="completed",
        full_text="Large notice with page markers",
        page_texts=page_texts,
        hts_mentions=expected_codes,
        warnings=[],
        content_sha256="sha-source-usage",
        uploaded_at="2026-04-19T10:00:00Z",
    )
    usage_recorder = TokenUsageRecorder()
    llm = _QueuedFakeLLM(
        [
            '{"results": [{"candidate_key": "source-usage:7308.90.0000"',
            {
                "results": [
                    {
                        "candidate_key": f"source-usage:{code}",
                        "is_real_hts_candidate": True,
                        "rule_type": "include",
                        "effective_from": None,
                        "effective_to": None,
                        "metal_scope": "steel",
                        "rate_text": None,
                        "source_pages": [1],
                        "source_excerpt": "The listed HTS codes remain covered.",
                        "interpreter_confidence": 0.9,
                    }
                    for code in expected_codes[:40]
                ]
            },
            {
                "results": [
                    {
                        "candidate_key": f"source-usage:{expected_codes[40]}",
                        "is_real_hts_candidate": True,
                        "rule_type": "include",
                        "effective_from": None,
                        "effective_to": None,
                        "metal_scope": "steel",
                        "rate_text": None,
                        "source_pages": [1],
                        "source_excerpt": "The listed HTS codes remain covered.",
                        "interpreter_confidence": 0.9,
                    }
                ]
            },
        ]
    )

    candidates = interpret_section_232_batch(
        batch_id="batch-usage",
        extracted_sources=[source],
        llm=llm,
        usage_recorder=usage_recorder,
    )

    assert [candidate.hts_code for candidate in candidates] == expected_codes
    assert len(llm.calls) == 3
    assert all(call["usage_recorder"] is usage_recorder for call in llm.calls)


def test_interpret_section_232_batch_skips_oversized_primary_prompt_before_chunking():
    page_texts = []
    expected_codes = []
    for index in range(41):
        code = f"7308.90.{index:04d}"
        expected_codes.append(code)
        page_texts.append(
            {
                "page_number": 1,
                "text": f"HTS {code} remains covered.",
                "layout_aware_text": f"HTS {code} remains covered.",
                "page_excerpt": f"HTS {code} remains covered.",
                "char_count": 30,
                "hts_mentions": [code],
            }
        )

    source = PersistedSection232Source(
        source_id="source-12",
        filename="oversized-primary.pdf",
        size_bytes=100,
        page_count=1,
        extraction_status="completed",
        full_text="Large notice with page markers",
        page_texts=page_texts,
        hts_mentions=expected_codes,
        warnings=[],
        content_sha256="sha-source-12",
        uploaded_at="2026-04-19T10:00:00Z",
    )
    llm = _QueuedFakeLLM(
        [
            {
                "results": [
                    {
                        "candidate_key": f"source-12:{code}",
                        "is_real_hts_candidate": True,
                        "rule_type": "include",
                        "effective_from": None,
                        "effective_to": None,
                        "metal_scope": "steel",
                        "rate_text": None,
                        "source_pages": [1],
                        "source_excerpt": "The listed HTS codes remain covered.",
                        "interpreter_confidence": 0.9,
                    }
                    for code in expected_codes[:40]
                ]
            },
            {
                "results": [
                    {
                        "candidate_key": f"source-12:{expected_codes[40]}",
                        "is_real_hts_candidate": True,
                        "rule_type": "include",
                        "effective_from": None,
                        "effective_to": None,
                        "metal_scope": "steel",
                        "rate_text": None,
                        "source_pages": [1],
                        "source_excerpt": "The listed HTS codes remain covered.",
                        "interpreter_confidence": 0.9,
                    }
                ]
            },
        ]
    )
    llm.settings = SimpleNamespace(section_232_max_prompt_chars=1)

    candidates = interpret_section_232_batch(
        batch_id="batch-oversized-primary",
        extracted_sources=[source],
        llm=llm,
    )

    assert [candidate.hts_code for candidate in candidates] == expected_codes
    assert len(llm.calls) == 2
    assert "full_document_text" in llm.calls[0]["messages"][1]["content"]


@pytest.mark.parametrize(
    "metal_scope",
    [
        "steel only, not copper",
        "aluminum except steel fasteners",
    ],
)
def test_interpret_section_232_batch_downgrades_exclusionary_metal_scope_to_unspecified(metal_scope):
    source = _source(
        "source-21",
        "notice-21.pdf",
        page_number=6,
        text="HTS 8302.41.00 is included in the notice.",
        excerpt="HTS 8302.41.00 is included in the notice.",
    )
    llm = _FakeLLM(
        {
            "results": [
                {
                    "candidate_key": "source-21:8302.41.00",
                    "is_real_hts_candidate": True,
                    "rule_type": "include",
                    "effective_from": "2026-04-07",
                    "metal_scope": metal_scope,
                    "source_pages": [6],
                }
            ]
        }
    )

    candidates = interpret_section_232_batch(
        batch_id="batch-201",
        extracted_sources=[source],
        llm=llm,
    )

    assert len(candidates) == 1
    assert candidates[0].metal_scope == "unspecified"


def test_normalize_date_extracts_notice_phrase_dates():
    assert _normalize_date("on or after 12:01 a.m. eastern daylight time on August 1, 2025") == "2025-08-01"
