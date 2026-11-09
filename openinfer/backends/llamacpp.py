"""llama.cpp in-process backend (optional extra: `uv sync --extra llamacpp`).

`llama-cpp-python` compiles llama.cpp for whatever accelerator the wheel or CMAKE_ARGS
target: plain CPU (AVX2/AVX512/NEON), Apple Metal, CUDA, ROCm/HIP, Vulkan, SYCL. The same
GGUF file runs on all of them, which is the whole point of this project. See
docs/learn/03-concepts/inference-backends-llamacpp-vllm-ollama.md and
docs/learn/03-concepts/quantization-and-gguf.md.

The library is synchronous and not thread safe for one `Llama` instance, so calls are
serialised with a lock and run in a worker thread to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from openinfer.backends.base import Backend
from openinfer.backends.openai_compat import parse_chat_completion, parse_chunk
from openinfer.config import BackendConfig
from openinfer.errors import BackendError, BackendUnavailable, NotSupported
from openinfer.types import (
    ChatChunk,
    ChatRequest,
    ChatResult,
    EmbeddingRequest,
    EmbeddingResult,
    HealthStatus,
    Usage,
    estimate_tokens,
)

DEFAULT_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def llamacpp_available() -> bool:
    try:
        import llama_cpp  # noqa: F401
    except Exception:  # noqa: BLE001 - import can fail with ImportError or a loader error
        return False
    return True


class LlamaCppBackend(Backend):
    kind = "llamacpp"
    local = True

    def __init__(self, name: str, config: BackendConfig) -> None:
        super().__init__(name, config)
        self.model_path = Path(config.model_path or f"models/{DEFAULT_FILE}")
        if not self.models:
            self.models = [self.model_path.stem]
        self._llama: Any | None = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # -- lazy loading -----------------------------------------------------------------------

    def _ensure_loaded(self) -> Any:
        if self._llama is not None:
            return self._llama
        if self._load_error:
            raise BackendUnavailable(self._load_error, backend=self.name)
        if not llamacpp_available():
            self._load_error = (
                "llama-cpp-python is not installed. Run `uv sync --extra llamacpp` "
                "(see docs/deploy/oracle-always-free.md for build flags per accelerator)."
            )
            raise BackendUnavailable(self._load_error, backend=self.name)
        if not self.model_path.exists():
            self._load_error = (
                f"GGUF not found at {self.model_path}. Run `openinfer download-model` first."
            )
            raise BackendUnavailable(self._load_error, backend=self.name)
        from llama_cpp import Llama

        kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.config.n_ctx,
            "n_gpu_layers": self.config.n_gpu_layers,
            "embedding": self.config.embedding,
            "verbose": False,
        }
        if self.config.n_threads:
            kwargs["n_threads"] = self.config.n_threads
        kwargs.update(self.config.options)
        try:
            self._llama = Llama(**kwargs)
        except Exception as exc:
            self._load_error = f"failed to load {self.model_path.name}: {exc}"
            raise BackendUnavailable(self._load_error, backend=self.name) from exc
        return self._llama

    def _kwargs(self, request: ChatRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"messages": request.messages}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.stop:
            kwargs["stop"] = request.stop
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.tools:
            kwargs["tools"] = request.tools
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.response_format:
            kwargs["response_format"] = request.response_format
        return kwargs

    # -- inference -----------------------------------------------------------------------------

    async def chat(self, request: ChatRequest) -> ChatResult:
        def run() -> dict[str, Any]:
            llama = self._ensure_loaded()
            with self._lock:
                return llama.create_chat_completion(**self._kwargs(request))

        try:
            data = await asyncio.to_thread(run)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(str(exc), backend=self.name, retriable=False) from exc
        result = parse_chat_completion(data, request)
        result.model = request.model
        return result

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ChatChunk | Exception | None] = asyncio.Queue()

        def produce() -> None:
            try:
                llama = self._ensure_loaded()
                with self._lock:
                    stream: Iterator[dict[str, Any]] = llama.create_chat_completion(
                        stream=True, **self._kwargs(request)
                    )
                    for payload in stream:
                        chunk = parse_chunk(payload)
                        if chunk is not None:
                            loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=produce, daemon=True).start()
        completion_parts: list[str] = []
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, BackendError):
                raise item
            if isinstance(item, Exception):
                raise BackendError(str(item), backend=self.name, retriable=False) from item
            if item.delta:
                completion_parts.append(item.delta)
            if item.finish_reason and item.usage is None:
                # llama.cpp streams do not include usage; estimate so the ledger has something.
                item.usage = Usage(
                    prompt_tokens=estimate_tokens(request.prompt_text()),
                    completion_tokens=estimate_tokens("".join(completion_parts)),
                    exact=False,
                )
            yield item

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        if not self.config.embedding:
            raise NotSupported(
                f"backend {self.name!r} was loaded without `embedding: true` in openinfer.yaml"
            )

        def run() -> list[list[float]]:
            llama = self._ensure_loaded()
            with self._lock:
                return [llama.embed(text) for text in request.inputs]

        vectors = await asyncio.to_thread(run)
        usage = Usage(prompt_tokens=estimate_tokens(request.prompt_text()), exact=False)
        return EmbeddingResult(vectors=vectors, usage=usage, model=request.model)

    async def health(self) -> HealthStatus:
        if not llamacpp_available():
            return HealthStatus(ok=False, detail="llama-cpp-python not installed")
        if not self.model_path.exists():
            return HealthStatus(ok=False, detail=f"missing {self.model_path}")
        return HealthStatus(ok=True, detail="loaded" if self._llama else "ready (lazy load)")
