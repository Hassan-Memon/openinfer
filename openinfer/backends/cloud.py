"""Cloud backends: named presets over the OpenAI compatible adapter.

All four providers below have a free tier. Only the base URL, the environment variable that
holds the key, a couple of courtesy headers and the pricing table name differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from openinfer.backends.openai_compat import OpenAICompatBackend
from openinfer.config import BackendConfig
from openinfer.errors import InvalidRequest


@dataclass(frozen=True)
class CloudPreset:
    base_url: str
    api_key_env: str
    pricing_provider: str
    headers: dict[str, str] = field(default_factory=dict)
    supports_stream_usage: bool = True


PRESETS: dict[str, CloudPreset] = {
    "groq": CloudPreset(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        pricing_provider="groq",
    ),
    "openrouter": CloudPreset(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        pricing_provider="openrouter",
        headers={
            "HTTP-Referer": "https://github.com/Hassan-Memon/openinfer",
            "X-Title": "OpenInfer",
        },
    ),
    "gemini": CloudPreset(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        pricing_provider="gemini",
        # Gemini's compatibility layer rejects unknown stream_options at the time of writing.
        supports_stream_usage=False,
    ),
    "openai": CloudPreset(
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        pricing_provider="openai",
    ),
}


class CloudBackend(OpenAICompatBackend):
    kind = "cloud"
    local = False

    def __init__(
        self,
        name: str,
        config: BackendConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        preset_name = config.preset or name
        preset = PRESETS.get(preset_name)
        if preset is None and not config.base_url:
            raise InvalidRequest(
                f"cloud backend {name!r}: unknown preset {preset_name!r} and no base_url given. "
                f"Known presets: {', '.join(sorted(PRESETS))}"
            )
        base_url = config.base_url or (preset.base_url if preset else "")
        api_key = config.resolved_api_key()
        if api_key is None and preset is not None and not config.api_key_env:
            import os

            api_key = os.environ.get(preset.api_key_env)
        self.preset_name = preset_name
        self.pricing_provider = preset.pricing_provider if preset else preset_name
        self.supports_stream_usage = preset.supports_stream_usage if preset else True
        super().__init__(
            name,
            config,
            base_url=base_url,
            api_key=api_key,
            headers=preset.headers if preset else None,
            transport=transport,
        )
