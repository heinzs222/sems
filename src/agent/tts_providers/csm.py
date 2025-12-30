from __future__ import annotations

import asyncio
import base64
import time
from typing import Any, AsyncGenerator, Optional

import httpx
import structlog

from src.agent.audio import wav_bytes_to_twilio_ulaw
from src.agent.config import get_config
from src.agent.tts_providers.base import TTSProvider
from src.agent.tts_types import TTSChunk

logger = structlog.get_logger(__name__)


class CsmMicroserviceTTS(TTSProvider):
    """
    Contextual TTS provider backed by a remote GPU microservice (see `csm_service/`).

    Returns 24kHz WAV which we convert to Twilio-ready 8kHz mu-law.
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self._cancelled = False
        self._inflight: Optional[asyncio.Task] = None

    def cancel(self) -> None:
        self._cancelled = True
        if self._inflight and not self._inflight.done():
            self._inflight.cancel()

    async def _request_tts(self, *, text: str, context: Optional[list[dict]]) -> tuple[bytes, dict[str, Any]]:
        url = self.config.csm_endpoint.rstrip("/") + "/tts"
        timeout_s = max(0.1, (self.config.csm_timeout_ms or 2500) / 1000.0)

        payload = {
            "speaker_id": 0,
            "prompt_text": text,
            "context": context or [],
            "voice_style": self.config.csm_voice_style or "default",
            "deterministic": True,
        }

        started = time.time()
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            resp = await client.post(url, json=payload, headers={"Accept": "audio/wav"})

        elapsed_ms = (time.time() - started) * 1000
        meta: dict[str, Any] = {"elapsed_ms": round(elapsed_ms, 2), "url": url}

        resp.raise_for_status()

        content_type = (resp.headers.get("content-type") or "").lower()
        if "audio/wav" in content_type or "audio/x-wav" in content_type or content_type.startswith("audio/"):
            return resp.content, meta

        # Fallback: JSON with base64.
        data = resp.json()
        audio_b64 = data.get("audio_base64_wav_24khz") or data.get("audio_base64_wav") or data.get("audio")
        if not audio_b64:
            raise ValueError("CSM service response missing audio")

        meta.update({k: v for k, v in data.items() if k != "audio_base64_wav_24khz"})
        return base64.b64decode(audio_b64), meta

    async def synthesize_streaming(
        self,
        text: str,
        *,
        voice_id: Optional[str] = None,
        context: Optional[list[dict]] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        # `voice_id` is ignored; the microservice controls voice selection.
        if not text or not text.strip():
            return

        self._cancelled = False
        task = asyncio.create_task(self._request_tts(text=text, context=context))
        self._inflight = task

        try:
            wav_24k, meta = await task
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("CSM microservice TTS failed", error=str(e))
            raise
        finally:
            if self._inflight is task:
                self._inflight = None

        if self._cancelled:
            yield TTSChunk(audio_bytes=b"", is_final=True)
            return

        ulaw = wav_bytes_to_twilio_ulaw(wav_24k)
        yield TTSChunk(audio_bytes=ulaw, is_final=True, audio_wav_24k=wav_24k, meta=meta)
