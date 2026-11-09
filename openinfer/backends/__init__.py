"""Backend registry: config entry -> adapter instance."""

from __future__ import annotations

from collections.abc import Callable

from openinfer.backends.base import Backend
from openinfer.backends.cloud import PRESETS, CloudBackend
from openinfer.backends.echo import EchoBackend
from openinfer.backends.llamacpp import LlamaCppBackend
from openinfer.backends.ollama import OllamaBackend
from openinfer.backends.openai_compat import OpenAICompatBackend
from openinfer.backends.vllm import VLLMBackend
from openinfer.config import AppConfig, BackendConfig

BackendFactory = Callable[[str, BackendConfig], Backend]

REGISTRY: dict[str, BackendFactory] = {
    "echo": EchoBackend,
    "ollama": OllamaBackend,
    "vllm": VLLMBackend,
    "cloud": CloudBackend,
    "llamacpp": LlamaCppBackend,
}


def build_backend(name: str, config: BackendConfig) -> Backend:
    try:
        factory = REGISTRY[config.type]
    except KeyError as exc:
        raise ValueError(f"unknown backend type {config.type!r} for {name!r}") from exc
    return factory(name, config)


def build_backends(config: AppConfig) -> dict[str, Backend]:
    """Instantiate every *active* backend (enabled, and keyed if it needs a key).

    Instantiation is cheap: HTTP clients are created but nothing is connected, and
    llama.cpp loads its model lazily on first request.
    """
    return {name: build_backend(name, cfg) for name, cfg in config.active_backends().items()}


__all__ = [
    "PRESETS",
    "REGISTRY",
    "Backend",
    "CloudBackend",
    "EchoBackend",
    "LlamaCppBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "VLLMBackend",
    "build_backend",
    "build_backends",
]
