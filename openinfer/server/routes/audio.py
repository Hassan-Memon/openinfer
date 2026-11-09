"""POST /v1/audio/transcriptions via faster-whisper (optional extra `whisper`).

Without the extra the route answers 501 with a clear message rather than pretending to
exist. With it, a `WhisperModel` is loaded lazily on first use and kept in app state.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import time
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, UploadFile

from openinfer.errors import InvalidRequest, NotSupported
from openinfer.ledger.base import LedgerEntry

router = APIRouter(prefix="/v1", tags=["audio"])

DEFAULT_WHISPER_MODEL = os.environ.get("OPENINFER_WHISPER_MODEL", "base")


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _get_model(request: Request, size: str) -> Any:
    cache = getattr(request.app.state, "whisper_models", None)
    if cache is None:
        cache = request.app.state.whisper_models = {}
    if size not in cache:
        from faster_whisper import WhisperModel

        cache[size] = WhisperModel(size, device="cpu", compute_type="int8")
    return cache[size]


@router.post("/audio/transcriptions")
async def transcribe(
    request: Request,
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = "whisper-1",
    language: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
) -> Any:
    if not whisper_available():
        raise NotSupported(
            "speech to text is not installed. Run `uv sync --extra whisper` "
            "(faster-whisper, CPU int8) and restart."
        )
    if response_format not in ("json", "text", "verbose_json"):
        raise InvalidRequest("response_format must be json, text or verbose_json")

    size = DEFAULT_WHISPER_MODEL if model in ("whisper-1", "") else model
    data = await file.read()
    if not data:
        raise InvalidRequest("empty audio file")

    started = time.perf_counter()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        whisper = _get_model(request, size)

        def run() -> tuple[list[Any], Any]:
            segments, info = whisper.transcribe(path, language=language, temperature=temperature)
            return list(segments), info

        segments, info = await asyncio.to_thread(run)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)

    text = " ".join(s.text.strip() for s in segments).strip()
    latency_ms = (time.perf_counter() - started) * 1000
    request.app.state.ledger.record(
        LedgerEntry(
            endpoint="transcription",
            requested_model=model,
            backend="whisper",
            model=size,
            latency_ms=round(latency_ms, 3),
        )
    )
    if response_format == "text":
        return text
    body: dict[str, Any] = {"text": text}
    if response_format == "verbose_json":
        body.update(
            {
                "task": "transcribe",
                "language": getattr(info, "language", language),
                "duration": getattr(info, "duration", None),
                "segments": [
                    {"id": i, "start": s.start, "end": s.end, "text": s.text}
                    for i, s in enumerate(segments)
                ],
            }
        )
    return body
