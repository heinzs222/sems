from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Optional

import structlog

from src.agent.audio import wav_bytes_to_twilio_ulaw
from src.agent.config import get_config
from src.agent.tts_providers.base import TTSProvider
from src.agent.tts_types import TTSChunk

logger = structlog.get_logger(__name__)


class OpenAITTS(TTSProvider):
    """
    OpenAI Text-to-Speech provider (non-streaming).

    This provider synthesizes a full WAV and yields it as one chunk.
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self._cancelled = False
        self._inflight: Optional[asyncio.Task] = None

    def cancel(self) -> None:
        self._cancelled = True
        if self._inflight and not self._inflight.done():
            self._inflight.cancel()

    async def _generate_wav(self, text: str) -> bytes:
        from openai import OpenAI  # Local import to keep module import light

        client = OpenAI(api_key=self.config.openai_api_key)

        def _call() -> bytes:
            resp = client.audio.speech.create(
                model=self.config.openai_tts_model,
                voice=self.config.openai_tts_voice,
                input=text,
                response_format="wav",
            )
            # SDKs have varied over time; handle several shapes.
            data = getattr(resp, "content", None)
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
            read = getattr(resp, "read", None)
            if callable(read):
                return read()
            return bytes(resp)

        return await asyncio.to_thread(_call)

    async def synthesize_streaming(
        self,
        text: str,
        *,
        voice_id: Optional[str] = None,
        context: Optional[list[dict]] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        # `voice_id` and `context` are ignored for OpenAI TTS.
        if not text or not text.strip():
            return

        self._cancelled = False
        task = asyncio.create_task(self._generate_wav(text))
        self._inflight = task

        try:
            wav_bytes = await task
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("OpenAI TTS failed", error=str(e))
            yield TTSChunk(audio_bytes=b"", is_final=True)
            return
        finally:
            if self._inflight is task:
                self._inflight = None

        if self._cancelled:
            yield TTSChunk(audio_bytes=b"", is_final=True)
            return

        ulaw = wav_bytes_to_twilio_ulaw(wav_bytes)
        yield TTSChunk(audio_bytes=ulaw, is_final=True)

