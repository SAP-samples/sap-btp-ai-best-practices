"""Load and validate YAML configuration without exposing application secrets."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ModelSettings(BaseModel):
    """Configure one SAP Generative AI Hub chat-model deployment."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "gemini", "claude"]
    name: str
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    use_responses_api: bool = True


class SkillSettings(BaseModel):
    """Configure dynamic skill discovery and maximum injected content."""

    model_config = ConfigDict(extra="forbid")

    directory: Path
    max_loaded_characters: int = Field(default=200_000, ge=1)


class MCPServerSettings(BaseModel):
    """Describe one local or remote MCP server connection."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    required: bool = True
    transport: Literal["stdio", "sse", "streamable_http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Path | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0)

    def connection(self) -> dict[str, Any]:
        """Return a validated connection dictionary for MultiServerMCPClient."""

        if self.transport == "stdio":
            if not self.command:
                raise ValueError("An enabled stdio MCP server requires command")
            connection: dict[str, Any] = {
                "transport": "stdio",
                "command": self.command,
                "args": self.args,
            }
            if self.env:
                connection["env"] = _require_resolved(self.env)
            if self.cwd:
                connection["cwd"] = str(self.cwd)
            return connection
        if not self.url:
            raise ValueError(f"An enabled {self.transport} MCP server requires url")
        return {
            "transport": self.transport,
            "url": _require_resolved(self.url),
            "headers": _require_resolved(self.headers),
            "timeout": self.timeout_seconds,
            "sse_read_timeout": self.timeout_seconds,
        }


class MCPSettings(BaseModel):
    """Group all configured MCP server connections."""

    model_config = ConfigDict(extra="forbid")

    servers: dict[str, MCPServerSettings] = Field(default_factory=dict)


class MemorySettings(BaseModel):
    """Configure optional SAP HANA conversation persistence."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    table_name: str = "LANGGRAPH_AGENT_MEMORY"
    max_messages: int = Field(default=40, ge=2)

    @field_validator("table_name")
    @classmethod
    def validate_table_name(cls, value: str) -> str:
        """Normalize and validate the HANA table identifier."""

        normalized = value.strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,126}", normalized):
            raise ValueError(f"Unsafe HANA table identifier: {value!r}")
        return normalized


class A2ASettings(BaseModel):
    """Configure the optional dual-protocol A2A HTTP server.

    The public URL is used in discovery metadata and must therefore be an
    externally meaningful HTTP(S) URL whenever the server is enabled.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    name: str = Field(default="Portable LangGraph Agent", min_length=1)
    description: str = Field(
        default="Skill-aware LangGraph agent with MCP tools", min_length=1
    )
    version: str = Field(default="0.1.0", min_length=1)
    public_url: str | None = None
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8080, ge=1, le=65_535)

    @model_validator(mode="after")
    def validate_public_url(self) -> "A2ASettings":
        """Require and normalize an HTTP(S) discovery URL when enabled."""

        if not self.enabled:
            return self
        if not self.public_url:
            raise ValueError("An enabled A2A server requires public_url")
        resolved = _require_resolved(self.public_url)
        parsed = urlsplit(resolved)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("A2A public_url must use http:// or https://")
        self.public_url = resolved.rstrip("/")
        return self


class AgentConfig(BaseModel):
    """Represent the complete validated agent configuration."""

    model_config = ConfigDict(extra="forbid")

    base_prompt: str
    model: ModelSettings
    skills: SkillSettings
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    a2a: A2ASettings = Field(default_factory=A2ASettings)
    recursion_limit: int = Field(default=30, ge=2)


def _expand_env(value: Any) -> Any:
    """Recursively expand known ``${ENV_VAR}`` placeholders.

    Unknown placeholders remain intact so disabled optional integrations do not
    make otherwise valid configuration unusable.
    """

    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), match.group(0)), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _require_resolved(value: Any) -> Any:
    """Reject an enabled integration that still contains missing environment values."""

    if isinstance(value, str) and _ENV_PATTERN.search(value):
        missing = sorted(set(_ENV_PATTERN.findall(value)))
        raise ValueError(f"Missing environment variable(s): {', '.join(missing)}")
    if isinstance(value, dict):
        return {key: _require_resolved(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_require_resolved(item) for item in value]
    return value


def load_config(path: str | Path) -> AgentConfig:
    """Load ``.env`` and return an AgentConfig from one YAML file.

    Args:
        path: YAML file path. Relative skill and MCP working directories are
            resolved against the YAML file's directory.

    Returns:
        A fully validated configuration object.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    load_dotenv(config_path.parent.parent / ".env", override=False)
    load_dotenv(config_path.parent / ".env", override=False)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Agent configuration must be a YAML mapping")
    expanded = _expand_env(raw)
    skill_dir = Path(expanded["skills"]["directory"])
    if not skill_dir.is_absolute():
        expanded["skills"]["directory"] = str((config_path.parent / skill_dir).resolve())
    for server in expanded.get("mcp", {}).get("servers", {}).values():
        cwd = server.get("cwd")
        if cwd and not Path(cwd).is_absolute():
            server["cwd"] = str((config_path.parent / cwd).resolve())
    return AgentConfig.model_validate(expanded)
