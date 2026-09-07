"""Shared language validation helpers for localized assessment behavior."""

from typing import Literal


SupportedLanguage = Literal["en", "it"]
"""Language codes supported by the assessment UI and AI review flow."""

SUPPORTED_LANGUAGES: set[str] = {"en", "it"}
"""Set of accepted language codes for framework text and AI generation."""

DEFAULT_LANGUAGE: SupportedLanguage = "en"
"""Default language used when callers do not submit a language selection."""


def normalize_language(value: str | None) -> SupportedLanguage:
    """Normalize and validate a user or API supplied language code.

    Inputs:
        value: Language code supplied by a browser, route query, form field, or
            stored job row. ``None`` and blank values fall back to English.

    Outputs:
        SupportedLanguage: Lowercase supported language code.

    Raises:
        ValueError: Raised when the language is not one of ``en`` or ``it``.
    """

    language = (value or DEFAULT_LANGUAGE).strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"Unsupported language '{value}'. Supported languages: {supported}.")
    return language  # type: ignore[return-value]
