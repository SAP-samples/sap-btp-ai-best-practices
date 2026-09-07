"""Tests for deployment dependency declarations."""

from pathlib import Path


def test_genai_hub_sdk_installs_langchain_proxy_extras(repo_root: Path) -> None:
    """Verify CF installs all optional Gen AI Hub proxy dependencies.

    Inputs:
        repo_root: Absolute path to the repository root.

    Outputs:
        None. Assertions validate that the requirements file installs the SAP
        Gen AI Hub SDK with the full optional dependency set needed by the
        application.
    """

    requirements = (
        repo_root / "api" / "requirements.txt"
    ).read_text(encoding="utf-8").splitlines()
    normalized = [line.strip() for line in requirements if line.strip()]

    assert "sap-ai-sdk-gen[all]>=6.10.0" in normalized
    assert "sap-ai-sdk-gen[amazon,google]>=6.10.0" not in normalized
    assert "sap-ai-sdk-gen>=6.10.0" not in normalized
