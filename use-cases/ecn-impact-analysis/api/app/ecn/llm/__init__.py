"""Bounded prompting, native provider access, and strict ECN analysis validation."""

from .prompts import (
    PromptEvidenceItem,
    PromptEvidenceProjection,
    PromptProjectionError,
    build_impact_input,
    build_impact_instructions,
    project_evidence_for_prompt,
)
from .provider import (
    EcnAnalysisProvider,
    NativeImpactProvider,
    ProviderCallResult,
    ProviderCallStatus,
    create_impact_provider,
)
from .validation import (
    AnalysisValidationError,
    ValidatedAnalysis,
    parse_and_validate_analysis,
)

__all__ = [
    "AnalysisValidationError",
    "EcnAnalysisProvider",
    "NativeImpactProvider",
    "PromptEvidenceItem",
    "PromptEvidenceProjection",
    "PromptProjectionError",
    "ProviderCallResult",
    "ProviderCallStatus",
    "ValidatedAnalysis",
    "build_impact_input",
    "build_impact_instructions",
    "create_impact_provider",
    "parse_and_validate_analysis",
    "project_evidence_for_prompt",
]
