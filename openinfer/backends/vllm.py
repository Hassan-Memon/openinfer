"""vLLM (or any self hosted OpenAI compatible server): local, zero marginal cost."""

from __future__ import annotations

import httpx

from openinfer.backends.openai_compat import OpenAICompatBackend
from openinfer.config import BackendConfig


class VLLMBackend(OpenAICompatBackend):
    kind = "vllm"
    local = True
    pricing_provider = None

    def __init__(
        self,
        name: str,
        config: BackendConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            name,
            config,
            base_url=config.base_url or "http://localhost:8000/v1",
            api_key=config.resolved_api_key(),
            transport=transport,
        )
