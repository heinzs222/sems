from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Optional

import structlog

from src.agent.config import get_config
from src.agent.tts_providers.base import TTSProvider
from src.agent.tts_providers.cartesia import CartesiaTTS, CartesiaTTSMetrics
from src.agent.tts_providers.csm import CsmMicroserviceTTS
from src.agent.tts_providers.openai_tts import OpenAITTS
from src.agent.tts_types import TTSChunk

logger = structlog.get_logger(__name__)


def _backchannel_ack_text(language: Optional[str]) -> str:
    lang = (language or "").strip().lower()
    if lang.startswith("fr"):
        return "Hum-hum."
    return "Mm-hm."


class TTSManager:
    """
    Per-call TTS manager with a pluggable provider system.

    - `cartesia`: streaming WebSocket TTS (default; best on CPU deployments)
    - `openai`: OpenAI Audio Speech API (non-streaming in this repo)
    - `csm`: contextual TTS via a GPU microservice (falls back to Cartesia)
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self._provider: Optional[TTSProvider] = None
        self._fallback: Optional[TTSProvider] = None

    async def start(self) -> None:
        tts = (self.config.tts_provider or "cartesia").strip().lower()

        if tts == "cartesia":
            self._provider = CartesiaTTS(self.config)
            self._fallback = None
            return

        if tts == "openai":
            self._provider = OpenAITTS(self.config)
            self._fallback = None
            return

        if tts == "csm":
            # CSM is best-effort; Cartesia remains the reliability fallback.
            self._provider = CsmMicroserviceTTS(self.config)
            self._fallback = CartesiaTTS(self.config)
            return

        raise ValueError(f"Unsupported TTS_PROVIDER: {self.config.tts_provider}")

    async def stop(self) -> None:
        self.cancel_current()
        if self._provider:
            await self._provider.close()
            self._provider = None
        if self._fallback:
            await self._fallback.close()
            self._fallback = None

    async def cancel_context(self) -> None:
        if self._provider:
            await self._provider.cancel_context()
        if self._fallback:
            await self._fallback.cancel_context()

    def cancel_current(self) -> None:
        if self._provider:
            self._provider.cancel()
        if self._fallback:
            self._fallback.cancel()

    @property
    def cartesia_metrics(self) -> Optional[CartesiaTTSMetrics]:
        provider = self._provider
        fallback = self._fallback
        if isinstance(provider, CartesiaTTS):
            return provider.metrics
        if isinstance(fallback, CartesiaTTS):
            return fallback.metrics
        return None

    async def synthesize_streaming(
        self,
        text: str,
        *,
        voice_id: Optional[str] = None,
        context: Optional[list[dict]] = None,
        language: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        if not self._provider:
            await self.start()

        provider = self._provider
        if not provider:
            yield TTSChunk(audio_bytes=b"", is_final=True)
            return

        # Fast path: non-CSM providers.
        if not isinstance(provider, CsmMicroserviceTTS):
            async for chunk in provider.synthesize_streaming(
                text,
                voice_id=voice_id,
                context=context,
            ):
                yield chunk
            return

        # CSM path with (a) quick backchannel if slow, and (b) fallback to Cartesia.
        fallback = self._fallback or CartesiaTTS(self.config)
        ack_after_s = 0.9

        gen = provider.synthesize_streaming(text, voice_id=voice_id, context=context)
        first_task = asyncio.create_task(gen.__anext__())

        try:
            done, _pending = await asyncio.wait({first_task}, timeout=ack_after_s)
            if first_task in done:
                first_chunk = first_task.result()
                yield first_chunk
                async for chunk in gen:
                    yield chunk
                return

            # Slow microservice: play a short backchannel quickly, then wait for CSM.
            ack_text = _backchannel_ack_text(language)
            async for ack_chunk in fallback.synthesize_streaming(ack_text, voice_id=voice_id):
                yield ack_chunk

            first_chunk = await first_task
            yield first_chunk
            async for chunk in gen:
                yield chunk

        except StopAsyncIteration:
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("CSM TTS failed, falling back to Cartesia", error=str(e))
            try:
                async for chunk in fallback.synthesize_streaming(text, voice_id=voice_id):
                    yield chunk
            except Exception as fallback_error:
                logger.error("Cartesia fallback failed", error=str(fallback_error))
                yield TTSChunk(audio_bytes=b"", is_final=True)
        finally:
            if not first_task.done():
                first_task.cancel()
            try:
                await gen.aclose()
            except Exception:
                pass

