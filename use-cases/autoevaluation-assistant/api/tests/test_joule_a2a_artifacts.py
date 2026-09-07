"""Tests for Joule A2A design-time artifacts."""

from pathlib import Path

import yaml


def test_joule_a2a_artifacts_round_trip_agent_context(repo_root: Path) -> None:
    """Verify Joule YAML preserves the A2A context ID between turns.

    Inputs:
        repo_root: Absolute path to the repository root.

    Outputs:
        None. Assertions validate the function/scenario/capability-context
        contract required by Joule for multi-turn remote agent calls.
    """

    capability_context = yaml.safe_load(
        (repo_root / "joule" / "a2a" / "capability_context.yaml").read_text()
    )
    function = yaml.safe_load(
        (
            repo_root
            / "joule"
            / "a2a"
            / "functions"
            / "assessment_knowledge_agent_function.yaml"
        ).read_text()
    )
    scenario = yaml.safe_load(
        (
            repo_root
            / "joule"
            / "a2a"
            / "scenarios"
            / "agent"
            / "assessment_knowledge_agent_scenario.yaml"
        ).read_text()
    )

    assert capability_context == {"variables": [{"name": "agent_context_id"}]}
    assert {"name": "agent_context_id", "optional": True} in function["parameters"]
    assert function["result"]["agent_context_id"] == "<? apiResponse.body.contextId ?>"
    assert scenario["target"]["parameters"] == [
        {
            "name": "agent_context_id",
            "value": "$capability_context.agent_context_id",
        }
    ]
    assert scenario["capability_context"] == [
        {
            "name": "agent_context_id",
            "value": "$target_result.agent_context_id",
        }
    ]
    assert "response_context" not in scenario


def test_joule_a2a_function_targets_mounted_agent_path(repo_root: Path) -> None:
    """Verify Joule calls the mounted FastAPI A2A endpoint.

    Inputs:
        repo_root: Absolute path to the repository root.

    Outputs:
        None. Assertions validate the remote agent request path used with the
        BTP destination URL.
    """

    function = yaml.safe_load(
        (
            repo_root
            / "joule"
            / "a2a"
            / "functions"
            / "assessment_knowledge_agent_function.yaml"
        ).read_text()
    )
    agent_request = function["action_groups"][0]["actions"][0]

    assert agent_request["type"] == "agent-request"
    assert agent_request["agent_type"] == "remote"
    assert agent_request["system_alias"] == "ASSESSMENT_KNOWLEDGE_AGENT_API"
    assert agent_request["path"] == "/a2a/"


def test_joule_a2a_capability_uses_code_based_agent_schema(repo_root: Path) -> None:
    """Verify the A2A capability points at the expected BTP destination.

    Inputs:
        repo_root: Absolute path to the repository root.

    Outputs:
        None. Assertions confirm schema version and system alias naming.
    """

    assistant = yaml.safe_load((repo_root / "da.sapdas.yaml").read_text())
    capability = yaml.safe_load(
        (repo_root / "joule" / "a2a" / "capability.sapdas.yaml").read_text()
    )

    assert assistant["schema_version"] == "1.4.0"
    assert assistant["name"] == "document_assessment_assistant"
    assert assistant["capabilities"] == [{"type": "local", "folder": "./joule/a2a"}]
    assert capability["schema_version"] == "3.28.0"
    assert capability["system_aliases"] == {
        "ASSESSMENT_KNOWLEDGE_AGENT_API": {"destination": "ASSESSMENT_KNOWLEDGE_AGENT_API"}
    }
