import logging
from functools import lru_cache
from typing import Tuple, Any

from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.service import OrchestrationService

from .prompt_registry import get_prompt_content

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-pro"


def _build_config(system_prompt: str) -> OrchestrationConfig:
    return OrchestrationConfig(
        template=Template(
            messages=[
                SystemMessage(system_prompt),
            ]
        ),
        llm=LLM(name=MODEL_NAME),
    )


@lru_cache(maxsize=1)
def _get_orchestration_service() -> OrchestrationService:
    return OrchestrationService()


async def map_rates(
    markdown: str, master_csv: str, prompt_key: str = "nc-rates"
) -> Tuple[str, Any]:
    """Run the rate-mapping LLM and return the raw CSV string and token usage.

    The caller is responsible for validating/parsing the CSV.
    """
    if not markdown.strip():
        raise ValueError("Empty markdown content for rate mapping.")
    if not master_csv.strip():
        raise ValueError("Empty master CSV content for rate mapping.")

    # Load the specific prompt based on the key
    try:
        system_prompt = get_prompt_content(prompt_key)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Failed to load prompt for key '%s': %s", prompt_key, e)
        raise ValueError(f"Invalid prompt key: {prompt_key}") from e

    orchestration = _get_orchestration_service()
    config = _build_config(system_prompt)

    user_content = (
        "Please process the following Rate Schedule Markdown document according to the System Prompt instructions.\n"
        "--- START OF MARKDOWN INPUT ---\n"
        f"{markdown.strip()}\n"
        "--- END OF MARKDOWN INPUT ---\n"
    )

    logger.info(
        "Calling LLM for rate mapping (model=%s, prompt_key=%s)", MODEL_NAME, prompt_key
    )
    result = await orchestration.arun(
        config=config,
        history=[UserMessage(user_content)],
    )

    text = result.orchestration_result.choices[0].message.content or ""
    usage = result.orchestration_result.usage
    csv_text = text.strip()
    if not csv_text:
        raise ValueError("Rate mapping LLM returned empty output.")

    return csv_text, usage


def extract_csv_from_llm_output(llm_output: str, master_header: str) -> str:
    """Best-effort extraction of CSV content from the LLM output.

    We look for the first occurrence of the exact master header line and
    return from there to the end. If not found, fall back to the full text.
    """
    text = (llm_output or "").strip()
    if not text:
        raise ValueError("Empty LLM output when extracting CSV.")

    header = master_header.strip()
    idx = text.find(header)
    if idx == -1:
        # Header not found; assume the whole output is the CSV
        logger.warning(
            "Master header not found in LLM output; returning full text as CSV."
        )
        return text

    return text[idx:].strip()
