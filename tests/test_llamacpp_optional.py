"""The llama.cpp backend must degrade gracefully when the extra is not installed."""

from __future__ import annotations

import pytest

from openinfer.backends.llamacpp import LlamaCppBackend, llamacpp_available
from openinfer.config import BackendConfig
from openinfer.errors import BackendUnavailable, NotSupported
from openinfer.types import ChatRequest, EmbeddingRequest


@pytest.mark.skipif(llamacpp_available(), reason="llama-cpp-python is installed")
async def test_missing_extra_is_reported_not_crashed(tmp_path):
    cfg = BackendConfig(type="llamacpp", model_path=str(tmp_path / "m.gguf"))
    b = LlamaCppBackend("llamacpp", cfg)
    assert b.models == ["m"]
    status = await b.health()
    assert status.ok is False and "not installed" in status.detail
    with pytest.raises(BackendUnavailable, match="uv sync --extra llamacpp"):
        await b.chat(ChatRequest(model="m", messages=[{"role": "user", "content": "x"}]))
    with pytest.raises(BackendUnavailable):
        async for _ in b.chat_stream(
            ChatRequest(model="m", messages=[{"role": "user", "content": "x"}])
        ):
            pass


async def test_embed_requires_embedding_flag(tmp_path):
    cfg = BackendConfig(type="llamacpp", model_path=str(tmp_path / "m.gguf"), embedding=False)
    b = LlamaCppBackend("llamacpp", cfg)
    with pytest.raises(NotSupported):
        await b.embed(EmbeddingRequest(model="m", inputs=["x"]))
