"""Tests for the Cloud Foundry deployment helper script."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def test_deploy_script_passes_manifest_vars_from_api_env(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    """Verify deploy.sh maps api/.env values into cf push variables.

    Inputs:
        tmp_path: Temporary directory used to isolate the script execution.
        repo_root: Absolute path to the repository root containing deploy.sh.

    Outputs:
        None. Assertions confirm the fake ``cf`` command receives all manifest
        variables required by Cloud Foundry.
    """

    script_path = tmp_path / "deploy.sh"
    shutil.copy(repo_root / "deploy.sh", script_path)
    script_path.chmod(0o755)

    api_dir = tmp_path / "api"
    api_dir.mkdir()
    (api_dir / ".env").write_text(
        "\n".join(
            [
                'API_KEY="stable-api-key"',
                'ALLOWED_ORIGIN="https://ui.example.test"',
                'API_BASE_URL="https://api.example.test"',
                'AGENT_PUBLIC_URL="https://agent.example.test"',
                'HANA_ADDRESS="hana.example.test"',
                'HANA_PORT="443"',
                'HANA_USER="hana-user"',
                'HANA_PASSWORD="hana-password"',
                'HANA_ENCRYPT="true"',
                'AICORE_AUTH_URL="https://auth.example.test"',
                'AICORE_CLIENT_ID="client-id"',
                'AICORE_CLIENT_SECRET="client-secret"',
                'AICORE_BASE_URL="https://api.example.test"',
                'AICORE_RESOURCE_GROUP="default"',
                'GENAI_DEFAULT_MODEL="gpt-5.4"',
                'GENAI_REVIEW_MODEL="gpt-5.4"',
                'GENAI_REASONING_EFFORT="low"',
                'GENAI_EMBEDDING_MODEL="text-embedding-3-large"',
                'GENAI_TEMPERATURE="0.1"',
                'GENAI_MAX_TOKENS="4096"',
                'JOULE_A2A_MODEL_NAME="gpt-4.1"',
                'LOG_USER_HASH_SALT="stable-salt"',
            ]
        ),
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    captured_args_path = tmp_path / "cf-args.jsonl"
    fake_cf = bin_dir / "cf"
    fake_cf.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, sys",
                f"open({str(captured_args_path)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )
    fake_cf.chmod(0o755)

    result = subprocess.run(
        [str(script_path)],
        cwd=tmp_path,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cf_calls = [
        json.loads(line)
        for line in captured_args_path.read_text(encoding="utf-8").splitlines()
    ]
    assert cf_calls[0] == [
        "push",
        "--var",
        "api_key=stable-api-key",
        "--var",
        "allowed_origin=https://ui.example.test",
        "--var",
        "api_base_url=https://api.example.test",
        "--var",
        "agent_public_url=https://agent.example.test",
        "--var",
        "hana_address=hana.example.test",
        "--var",
        "hana_port=443",
        "--var",
        "hana_user=hana-user",
        "--var",
        "hana_password=hana-password",
        "--var",
        "hana_encrypt=true",
        "--var",
        "aicore_auth_url=https://auth.example.test",
        "--var",
        "aicore_client_id=client-id",
        "--var",
        "aicore_client_secret=client-secret",
        "--var",
        "aicore_base_url=https://api.example.test",
        "--var",
        "aicore_resource_group=default",
        "--var",
        "genai_default_model=gpt-5.4",
        "--var",
        "genai_review_model=gpt-5.4",
        "--var",
        "genai_reasoning_effort=low",
        "--var",
        "genai_embedding_model=text-embedding-3-large",
        "--var",
        "genai_temperature=0.1",
        "--var",
        "genai_max_tokens=4096",
        "--var",
        "joule_a2a_model_name=gpt-4.1",
        "--var",
        "log_user_hash_salt=stable-salt",
    ]
    assert cf_calls[1:] == [
        ["bind-service", "autoevaluation-assistant-api", "Cloud Logging"],
        ["bind-service", "autoevaluation-assistant-worker", "Cloud Logging"],
        ["restart", "autoevaluation-assistant-api"],
        ["restart", "autoevaluation-assistant-worker"],
    ]


def test_deploy_script_generates_log_user_hash_salt_when_missing(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    """Verify deploy.sh never passes an empty Cloud Foundry manifest variable.

    Inputs:
        tmp_path: Temporary directory used to isolate the script execution.
        repo_root: Absolute path to the repository root containing deploy.sh.

    Outputs:
        None. Assertions confirm a missing ``LOG_USER_HASH_SALT`` is replaced
        with a generated non-empty value before ``cf push``.
    """

    script_path = tmp_path / "deploy.sh"
    shutil.copy(repo_root / "deploy.sh", script_path)
    script_path.chmod(0o755)

    api_dir = tmp_path / "api"
    api_dir.mkdir()
    (api_dir / ".env").write_text(
        "\n".join(
            [
                'API_KEY="stable-api-key"',
                'HANA_ADDRESS="hana.example.test"',
                'HANA_PORT="443"',
                'HANA_USER="hana-user"',
                'HANA_PASSWORD="hana-password"',
                'AICORE_AUTH_URL="https://auth.example.test"',
                'AICORE_CLIENT_ID="client-id"',
                'AICORE_CLIENT_SECRET="client-secret"',
                'AICORE_BASE_URL="https://api.example.test"',
                'AICORE_RESOURCE_GROUP="default"',
            ]
        ),
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    captured_args_path = tmp_path / "cf-args.jsonl"
    fake_cf = bin_dir / "cf"
    fake_cf.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, sys",
                f"open({str(captured_args_path)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')",
            ]
        ),
        encoding="utf-8",
    )
    fake_cf.chmod(0o755)

    result = subprocess.run(
        [str(script_path)],
        cwd=tmp_path,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cf_push_args = json.loads(
        captured_args_path.read_text(encoding="utf-8").splitlines()[0]
    )
    salt_arg = next(
        arg for arg in cf_push_args if arg.startswith("log_user_hash_salt=")
    )
    generated_salt = salt_arg.split("=", maxsplit=1)[1]
    assert generated_salt
    assert "Generated temporary LOG_USER_HASH_SALT" in result.stdout


def test_deploy_script_fails_when_required_env_var_is_missing(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    """Verify deploy.sh fails before cf push when api/.env is incomplete.

    Inputs:
        tmp_path: Temporary directory used to isolate the script execution.
        repo_root: Absolute path to the repository root containing deploy.sh.

    Outputs:
        None. Assertions confirm missing HANA or AI Core values stop the deploy.
    """

    script_path = tmp_path / "deploy.sh"
    shutil.copy(repo_root / "deploy.sh", script_path)
    script_path.chmod(0o755)
    api_dir = tmp_path / "api"
    api_dir.mkdir()
    (api_dir / ".env").write_text('API_KEY="stable-api-key"\n', encoding="utf-8")

    result = subprocess.run(
        [str(script_path)],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Missing required value in api/.env: HANA_ADDRESS" in result.stderr
