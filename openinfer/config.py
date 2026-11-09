"""Configuration: the `openinfer.yaml` policy file plus environment overrides.

Loading order:
1. explicit path passed to `load_config`
2. `OPENINFER_CONFIG` environment variable
3. `./openinfer.yaml` if it exists
4. the packaged `default_config.yaml` (identical to the example at the repo root)

`${VAR}` and `${VAR:-default}` references are expanded from the environment before the
YAML is parsed, so the file can be committed while secrets stay in `.env`.
"""

from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

BackendType = Literal["echo", "ollama", "vllm", "cloud", "llamacpp"]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(text: str, env: dict[str, str] | None = None) -> str:
    """Replace ${VAR} / ${VAR:-default} with values from `env` (defaults to os.environ)."""
    source = os.environ if env is None else env

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = source.get(name)
        if value is None or value == "":
            return default if default is not None else ""
        return value

    return _ENV_PATTERN.sub(replace, text)


class ModelPrice(BaseModel):
    """USD per one million tokens."""

    input: float = 0.0
    output: float = 0.0


class BackendConfig(BaseModel):
    type: BackendType
    enabled: bool = True
    models: list[str] = Field(default_factory=list)
    embedding_models: list[str] = Field(default_factory=list)

    # HTTP backends (ollama, vllm, cloud)
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    preset: str | None = None
    timeout_s: float = 120.0
    extra_headers: dict[str, str] = Field(default_factory=dict)

    # llama.cpp backend
    model_path: str | None = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_threads: int | None = None
    embedding: bool = False

    # Routing metadata
    expected_latency_ms: int | None = None
    # Per model price overrides, e.g. an amortised rate for self hosted hardware.
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)
    # Backend specific knobs (echo: delay_ms).
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_key", "base_url", "model_path", "preset", mode="before")
    @classmethod
    def _empty_to_none(cls, value: Any) -> Any:
        return None if value == "" else value

    def resolved_api_key(self, env: dict[str, str] | None = None) -> str | None:
        source = os.environ if env is None else env
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return source.get(self.api_key_env) or None
        return None

    def is_active(self, env: dict[str, str] | None = None) -> bool:
        """Enabled, and if it needs a key, the key is present."""
        if not self.enabled:
            return False
        return not (self.type == "cloud" and self.resolved_api_key(env) is None)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_to_none(cls, value: Any) -> Any:
        return None if value in ("", None) else str(value)


class LedgerConfig(BaseModel):
    url: str = "sqlite:///./openinfer.db"


class PolicyConfig(BaseModel):
    default_alias: str = "fast"
    private_tag: str = "private"
    max_cost_usd: float | None = None
    max_latency_ms: int | None = None
    retries: int = 2
    backoff_base_ms: int = 200
    backoff_max_ms: int = 4000
    default_max_tokens: int = 512
    # How many ledger samples before measured latency overrides `expected_latency_ms`.
    latency_min_samples: int = 5


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    ledger: LedgerConfig = Field(default_factory=LedgerConfig)
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    # Where the config came from, for `openinfer config` and /health.
    source: str = "<default>"

    def active_backends(self, env: dict[str, str] | None = None) -> dict[str, BackendConfig]:
        return {name: cfg for name, cfg in self.backends.items() if cfg.is_active(env)}


def default_config_text() -> str:
    return resources.files("openinfer").joinpath("default_config.yaml").read_text("utf-8")


def find_config_path(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    env_path = os.environ.get("OPENINFER_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    local = Path("openinfer.yaml")
    return local if local.exists() else None


def parse_config(
    text: str, *, source: str = "<inline>", env: dict[str, str] | None = None
) -> AppConfig:
    data: dict[str, Any] = yaml.safe_load(expand_env(text, env)) or {}
    data.setdefault("source", source)
    return AppConfig.model_validate(data)


def load_config(path: str | Path | None = None, *, env: dict[str, str] | None = None) -> AppConfig:
    resolved = find_config_path(path)
    if resolved is not None:
        if not resolved.exists():
            raise FileNotFoundError(f"config file not found: {resolved}")
        return parse_config(resolved.read_text("utf-8"), source=str(resolved), env=env)
    return parse_config(default_config_text(), source="<packaged default>", env=env)
