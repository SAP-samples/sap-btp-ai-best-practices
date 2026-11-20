"""Independent impact, readiness, and sequential summary generation stages."""

from __future__ import annotations

import asyncio
import logging
import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING

from app.ecn.analysis_models import (
    AnalysisResponse, ImpactFactors, ImpactNextSteps, derive_overall_impact,
)
from app.ecn.assistant.discovery import DiscoveryResult
from app.ecn.llm.provider import ProviderCallStage, ProviderCallStatus
from app.ecn.llm.grounded_language import normalize_replacement_prose
from app.ecn.llm.staged import (
    EXECUTIVE_SUMMARY_PATTERN, FactorStageResult, ReadinessStageResult, StagePrompt,
    SummaryStageResult, build_stage_prompt,
)
from app.ecn.llm.two_pass import FinalEvidenceProjection
from app.ecn.models import AnalysisSectionKey
from app.persistence.records import (
    ClaimedJob, ProviderCallOutcome, ProviderStageRecord, ValidationOutcome,
)

from .blocking_io import run_blocking

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .generation import AnalysisGenerationService, JobProcessingResult


async def run_staged_analysis(
    service: AnalysisGenerationService, claimed: ClaimedJob, projection: FinalEvidenceProjection,
    snapshot: dict, discovery_records: tuple[ProviderStageRecord, ...],
    discovery: DiscoveryResult,
) -> JobProcessingResult:
    """Run four independent perspectives and a summary, returning the job result.

    The inputs are the claimed job, bounded evidence projection, deterministic
    snapshot, prior discovery audit rows, and source-only discovery result.
    Validated stage outputs are stored and reused across failed attempts.
    """
    try:
        completed = await run_blocking(
            service._repository.list_stage_results, claimed.job_id,
            claimed.evidence.evidence_hash, claimed.generation_config.prompt_version,
            claimed.generation_config.output_schema_version,
        )
    except Exception:
        if discovery_records:
            return await service._finish_after_calls(
                claimed, discovery_records, "DB_PERSISTENCE_ERROR", ValidationOutcome.NOT_RUN,
            )
        return await service._finish_without_provider_call(claimed, "DB_PERSISTENCE_ERROR")

    stages = ("MANUFACTURING", "PROCUREMENT", "CONFIGURATION", "DATA_READINESS")
    if not snapshot.get("questions") and "DATA_READINESS" not in completed:
        empty = ReadinessStageResult().model_dump(mode="json")
        try:
            await run_blocking(
                service._repository.save_stage_result, str(service._uuid_factory()), claimed.job_id,
                claimed.attempt.attempt_id, "DATA_READINESS", claimed.evidence.evidence_hash,
                claimed.generation_config.prompt_version,
                claimed.generation_config.output_schema_version, empty, service._clock.now(),
            )
            completed["DATA_READINESS"] = empty
        except Exception:
            if discovery_records:
                return await service._finish_after_calls(
                    claimed, discovery_records, "DB_PERSISTENCE_ERROR", ValidationOutcome.NOT_RUN,
                )
            return await service._finish_without_provider_call(claimed, "DB_PERSISTENCE_ERROR")

    # Discovery re-runs on every attempt and mints new fact IDs, so a factor
    # saved by an earlier attempt may cite IDs this attempt cannot accept.
    permitted = _permitted_citations(projection)
    completed = {stage: payload for stage, payload in completed.items()
                 if stage not in _FACTOR_STAGES or set(payload["factor"].get("evidence_ids", ())) <= permitted}
    pending = [stage for stage in stages if stage not in completed]
    try:
        tasks = [call_staged(service,
            claimed, projection, stage, build_stage_prompt(projection, stage, snapshot=snapshot),
            len(discovery_records) + index + 1,
        ) for index, stage in enumerate(pending)]
    except Exception:
        if discovery_records:
            return await service._finish_after_calls(
                claimed, discovery_records, "PROMPT_PROJECTION_ERROR", ValidationOutcome.NOT_RUN,
            )
        return await service._finish_without_provider_call(claimed, "PROMPT_PROJECTION_ERROR")
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    records = list(discovery_records)
    failures = []
    for stage, outcome in zip(pending, outcomes):
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, BaseException):
            failures.append("PROVIDER_ADAPTER_CONTRACT_ERROR")
            continue
        payload, record, error = outcome
        if record is not None:
            records.append(record)
        if payload is not None:
            completed[stage] = payload
        if error:
            failures.append(error)
    if failures:
        if not records:
            return await service._finish_without_provider_call(claimed, failures[0])
        return await service._finish_after_calls(
            claimed, tuple(records), failures[0],
            ValidationOutcome.INVALID if failures[0] == "ANALYSIS_VALIDATION_ERROR" else ValidationOutcome.NOT_RUN,
        )

    try:
        prompt = build_stage_prompt(
            projection, "SUMMARY", completed={stage: completed[stage] for stage in stages},
        )
    except Exception:
        if not records:
            return await service._finish_without_provider_call(claimed, "PROMPT_PROJECTION_ERROR")
        return await service._finish_after_calls(
            claimed, tuple(records), "PROMPT_PROJECTION_ERROR", ValidationOutcome.NOT_RUN,
        )
    payload, record, error, provider_result = await call_staged(service,
        claimed, projection, "SUMMARY", prompt, len(discovery_records) + len(pending) + 1,
        return_provider=True,
    )
    if record is not None:
        records.append(record)
    if error:
        return await service._finish_after_calls(
            claimed, tuple(records), error,
            ValidationOutcome.INVALID if error == "ANALYSIS_VALIDATION_ERROR" else ValidationOutcome.NOT_RUN,
        )
    try:
        # Saved stage payloads are JSON values; strict enum parsing must use the JSON path.
        factors = ImpactFactors(
            manufacturing=FactorStageResult.model_validate_json(json.dumps(completed["MANUFACTURING"])).factor,
            procurement=FactorStageResult.model_validate_json(json.dumps(completed["PROCUREMENT"])).factor,
            configuration=FactorStageResult.model_validate_json(json.dumps(completed["CONFIGURATION"])).factor,
            data_readiness=ImpactFactors().data_readiness,
        )
        descriptions = ReadinessStageResult.model_validate_json(json.dumps(completed["DATA_READINESS"])).descriptions
        summary = SummaryStageResult.model_validate_json(json.dumps(payload))
        combined = AnalysisResponse(
            change_brief=summary.change_brief, sections=summary.sections, impact_factors=factors,
            overall_impact=derive_overall_impact(factors),
            data_readiness_descriptions=descriptions,
            additional_next_steps=ImpactNextSteps(**{
                name.lower(): completed[name]["additional_next_steps"]
                for name in ("MANUFACTURING", "PROCUREMENT", "CONFIGURATION")
            }),
        )
        synthetic = replace(provider_result, raw_json=combined.model_dump_json())
    except Exception:
        return await service._finish_after_calls(
            claimed, tuple(records), "ANALYSIS_VALIDATION_ERROR", ValidationOutcome.INVALID,
        )
    return await service._validate_and_complete(
        claimed, projection, synthetic, tuple(records), discovery,
    )

_FACTOR_STAGES = frozenset({"MANUFACTURING", "PROCUREMENT", "CONFIGURATION"})


def _permitted_citations(projection: FinalEvidenceProjection) -> set[str]:
    """Return citation IDs a factor may use in this attempt's bounded projection."""
    permitted = set(projection.header_citation_ids)
    permitted.update(cid for item in projection.records for cid in item.citation_ids)
    permitted.update(fact.evidence_id for fact in projection.discovered_facts)
    return permitted


async def call_staged(
    service: AnalysisGenerationService, claimed: ClaimedJob, projection: FinalEvidenceProjection,
    stage_name: str, prompt: StagePrompt, call_number: int, *, return_provider: bool = False,
):
    """Call and validate one named stage, returning its payload, audit row, and error.

    Inputs are the service, claimed job, bounded projection, named stage,
    structured prompt, call number, and optional provider-result return flag.
    Sibling stage failures are reported without cancelling other calls.
    """
    stage = ProviderCallStage(stage_name)
    started = service._clock.now()
    try:
        async def invoke():
            """Apply the stage timeout and return the provider adapter result."""
            async with asyncio.timeout(claimed.generation_config.request_timeout_seconds * 2 + 5):
                return await service._provider.generate(prompt, claimed.generation_config, stage)

        if service._provider_capacity is None:
            result = await invoke()
        else:
            async with service._provider_capacity:
                result = await invoke()
        record = service._stage_record_from_result(
            claimed, result, stage, started, service._clock.now(), call_number=call_number,
        )
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        record = service._failed_stage_record(
            claimed, stage, started, ProviderCallOutcome.TIMEOUT,
            "PROVIDER_TIMEOUT", call_number=call_number,
        )
        result = None
    except Exception:
        record = service._failed_stage_record(
            claimed, stage, started, ProviderCallOutcome.ERROR,
            "PROVIDER_ADAPTER_CONTRACT_ERROR", call_number=call_number,
        )
        result = None

    if result is not None:
        service._emit_usage(
            claimed, result.usage, result.model_deployment, result.latency_ms,
            "success" if result.status is ProviderCallStatus.RECEIVED else "error",
            stage, model_name=result.model_name,
        )
    try:
        await run_blocking(service._repository.record_provider_call, record)
    except Exception:
        response = (None, record, "DB_PERSISTENCE_ERROR")
        return (*response, result) if return_provider else response

    if result is None or result.status is ProviderCallStatus.ERROR:
        response = (None, record, record.error_code or "PROVIDER_RESPONSE_INVALID")
        return (*response, result) if return_provider else response
    try:
        if result.response_status != "completed" or result.refusal_present or not result.raw_json:
            raise ValueError("provider response incomplete")
        parsed = prompt.schema.model_validate_json(result.raw_json)
        parsed = prompt.schema.model_validate(normalize_replacement_prose(
            parsed.model_dump(mode="python"), projection.snapshot_context.get("bom_changes") or {},
        ))
        permitted = _permitted_citations(projection)
        if stage_name in _FACTOR_STAGES:
            if set(parsed.factor.evidence_ids) - permitted:
                raise ValueError("unknown factor citation")
            lines = [line.strip() for line in parsed.factor.assessment.splitlines() if line.strip()]
            if not 1 <= len(lines) <= 2 or any(not line.startswith(("- ", "* ")) for line in lines):
                raise ValueError("factor assessment must use bounded bullets")
        elif stage_name == "DATA_READINESS":
            expected = [q["code"] for q in prompt.payload["questions"]]
            actual = [item.code for item in parsed.descriptions]
            if len(actual) != len(set(actual)) or set(actual) != set(expected):
                raise ValueError("data readiness coverage mismatch")
        elif stage_name == "SUMMARY":
            if len(parsed.change_brief.split()) > 16:
                raise ValueError("change brief must contain at most 16 words")
            lines = [line.strip() for line in parsed.sections.executive_summary.assessment_markdown.splitlines() if line.strip()]
            if not re.fullmatch(EXECUTIVE_SUMMARY_PATTERN, "\n".join(lines)):
                raise ValueError("executive summary must follow what changes / what is impacted / where")
            for key in AnalysisSectionKey:
                section = getattr(parsed.sections, key.value)
                if set(section.evidence_ids) - projection.allowed_evidence_ids_by_section.get(key, frozenset()):
                    raise ValueError("unknown narrative citation")
                if key in projection.sections_requiring_gap and section.status.value != "INSUFFICIENT_EVIDENCE":
                    raise ValueError("required narrative gap missing")
        payload = parsed.model_dump(mode="json")
    except Exception as exc:
        # Our own check messages are safe to log; schema errors may echo model output.
        reason = str(exc) if type(exc) is ValueError else type(exc).__name__
        logger.warning("Stage %s failed validation for job %s: %s", stage_name, claimed.job_id, reason)
        response = (None, record, "ANALYSIS_VALIDATION_ERROR")
        return (*response, result) if return_provider else response
    try:
        await run_blocking(
            service._repository.save_stage_result, str(service._uuid_factory()), claimed.job_id,
            claimed.attempt.attempt_id, stage_name, claimed.evidence.evidence_hash,
            claimed.generation_config.prompt_version,
            claimed.generation_config.output_schema_version, payload, service._clock.now(),
        )
        response = (payload, record, None)
    except Exception:
        response = (None, record, "DB_PERSISTENCE_ERROR")
    return (*response, result) if return_provider else response
