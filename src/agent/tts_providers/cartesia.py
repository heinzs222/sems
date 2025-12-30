from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

import structlog
import websockets

from src.agent.audio import pcm_8k_to_ulaw
from src.agent.config import get_config
from src.agent.tts_providers.base import TTSProvider
from src.agent.tts_types import TTSChunk

logger = structlog.get_logger(__name__)

# Cartesia output format - 8kHz is supported per their docs.
# No resampling needed - direct output at Twilio's native rate.
CARTESIA_SAMPLE_RATE = 8000
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_API_VERSION = "2024-06-10"


@dataclass
class CartesiaTTSMetrics:
    """Metrics for TTS performance."""

    total_requests: int = 0
    total_characters: int = 0
    total_audio_ms: float = 0.0
    avg_first_byte_ms: float = 0.0
    avg_total_ms: float = 0.0

    def record_synthesis(
        self,
        *,
        characters: int,
        audio_ms: float,
        first_byte_ms: float,
        total_ms: float,
    ) -> None:
        self.total_requests += 1
        self.total_characters += characters
        self.total_audio_ms += audio_ms

        # Running averages
        n = self.total_requests
        self.avg_first_byte_ms = (self.avg_first_byte_ms * (n - 1) + first_byte_ms) / n
        self.avg_total_ms = (self.avg_total_ms * (n - 1) + total_ms) / n


class CartesiaTTS(TTSProvider):
    """
    Cartesia streaming TTS client using WebSocket API.

    Produces Twilio-ready mu-law 8kHz audio.
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self._metrics = CartesiaTTSMetrics()
        self._is_cancelled = False
        self._current_context_id: Optional[str] = None
        self._current_ws: Optional[Any] = None

    @property
    def metrics(self) -> CartesiaTTSMetrics:
        return self._metrics

    def cancel(self) -> None:
        self._is_cancelled = True
        logger.debug("Cartesia TTS cancelled")

    async def cancel_context(self) -> None:
        if self._current_ws and self._current_context_id:
            try:
                cancel_request = {"context_id": self._current_context_id, "cancel": True}
                await self._current_ws.send(json.dumps(cancel_request))
                logger.debug("Cartesia cancel context sent", context_id=self._current_context_id)
            except Exception as e:
                logger.warning("Cartesia cancel context failed", error=str(e))

        self._is_cancelled = True
        self._current_context_id = None

    def _reset_cancel(self) -> None:
        self._is_cancelled = False

    async def synthesize_streaming(
        self,
        text: str,
        *,
        voice_id: Optional[str] = None,
        context: Optional[list[dict]] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        # `context` is ignored by Cartesia; kept for a unified interface.
        if not text or not text.strip():
            return

        self._reset_cancel()
        if voice_id is None:
            voice_id = self.config.cartesia_voice_id

        start_time = time.time()
        first_byte_time: Optional[float] = None
        total_audio_bytes = 0

        try:
            url = (
                f"{CARTESIA_WS_URL}?api_key={self.config.cartesia_api_key}"
                f"&cartesia_version={CARTESIA_API_VERSION}"
            )

            async with websockets.connect(url) as ws:
                self._current_ws = ws

                import uuid

                context_id = str(uuid.uuid4()).replace("-", "")
                self._current_context_id = context_id

                request = {
                    "context_id": context_id,
                    "model_id": "sonic-english",
                    "transcript": text,
                    "voice": {"mode": "id", "id": voice_id},
                    "output_format": {
                        "container": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": CARTESIA_SAMPLE_RATE,
                    },
                    "continue": False,
                }

                await ws.send(json.dumps(request))

                async for message in ws:
                    if self._is_cancelled:
                        break

                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        data = None

                    if isinstance(data, dict):
                        msg_type = data.get("type", "")
                        if msg_type == "chunk":
                            audio_b64 = data.get("data")
                            if not audio_b64:
                                continue

                            audio_data = base64.b64decode(audio_b64)
                            if first_byte_time is None:
                                first_byte_time = time.time()

                            ulaw_audio = pcm_8k_to_ulaw(audio_data)
                            if ulaw_audio:
                                total_audio_bytes += len(ulaw_audio)
                                yield TTSChunk(audio_bytes=ulaw_audio, is_final=False)
                            continue

                        if msg_type == "done":
                            break

                        if msg_type == "error":
                            logger.error("Cartesia error", error=data.get("message"), details=data)
                            break

                    if isinstance(message, (bytes, bytearray)):
                        if first_byte_time is None:
                            first_byte_time = time.time()
                        ulaw_audio = pcm_8k_to_ulaw(bytes(message))
                        if ulaw_audio:
                            total_audio_bytes += len(ulaw_audio)
                            yield TTSChunk(audio_bytes=ulaw_audio, is_final=False)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Cartesia synthesis failed", error=str(e))
        finally:
            self._current_context_id = None
            self._current_ws = None

        if not self._is_cancelled:
            yield TTSChunk(audio_bytes=b"", is_final=True)

        end_time = time.time()
        if first_byte_time is None:
            first_byte_time = end_time

        audio_duration_ms = total_audio_bytes / 8.0
        self._metrics.record_synthesis(
            characters=len(text),
            audio_ms=audio_duration_ms,
            first_byte_ms=(first_byte_time - start_time) * 1000,
            total_ms=(end_time - start_time) * 1000,
        )

