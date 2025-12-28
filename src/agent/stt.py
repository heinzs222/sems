"""
Deepgram Speech-to-Text streaming client.

Late-2025 Best Practice:
- Accepts mu-law 8kHz directly from Twilio (no conversion needed!)
- Supports both v1 (nova-2) and v2 (Flux) endpoints
- Flux mode uses ~80ms chunks for better turn detection

This eliminates CPU-heavy resampling that causes choppy audio.
"""

import asyncio
import json
from typing import Optional, Callable, Awaitable, Any
from dataclasses import dataclass, field
import time

import structlog
import websockets

from src.agent.config import get_config
import os

logger = structlog.get_logger(__name__)

# Deepgram endpoints
DEEPGRAM_V1_URL = "wss://api.deepgram.com/v1/listen"
DEEPGRAM_V2_URL = "wss://api.deepgram.com/v2/listen"  # Flux endpoint

# Prefer Deepgram v2 (/v2/listen). If it fails, we fall back automatically.
# You can force v1 only with USE_DEEPGRAM_V1_ONLY=true.
USE_DEEPGRAM_V1_ONLY = os.getenv("USE_DEEPGRAM_V1_ONLY", "false").lower() == "true"


@dataclass
class TranscriptionResult:
    """Result from STT."""
    text: str
    is_final: bool
    confidence: float = 0.0
    speech_final: bool = False
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0


@dataclass
class STTMetrics:
    """Metrics for STT performance."""
    total_audio_ms: float = 0.0
    total_transcripts: int = 0
    final_transcripts: int = 0
    avg_latency_ms: float = 0.0
    
    def record_transcript(self, is_final: bool, latency_ms: float) -> None:
        self.total_transcripts += 1
        if is_final:
            self.final_transcripts += 1
        if self.total_transcripts > 0:
            self.avg_latency_ms = (
                (self.avg_latency_ms * (self.total_transcripts - 1) + latency_ms) 
                / self.total_transcripts
            )


class DeepgramSTT:
    """
    Deepgram streaming STT client using raw WebSocket.
    """
    
    def __init__(
        self,
        on_transcript: Optional[Callable[[TranscriptionResult], Awaitable[None]]] = None,
        on_speech_started: Optional[Callable[[], Awaitable[None]]] = None,
        language: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        if config is None:
            config = get_config()
        
        self.config = config
        self.language = language or getattr(config, "deepgram_language_en", "en-US")
        self._on_transcript = on_transcript
        self._on_speech_started = on_speech_started
        self._ws = None
        self._is_connected = False
        self._metrics = STTMetrics()
        self._last_audio_time: float = 0.0
        self._current_transcript = ""
        self._word_count = 0
        self._receive_task = None
        self._detected_language: Optional[str] = None
        self._language_confidence: Optional[float] = None
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def metrics(self) -> STTMetrics:
        return self._metrics
    
    @property
    def current_word_count(self) -> int:
        return self._word_count
    
    async def connect(self) -> bool:
        """Connect to Deepgram streaming API."""
        if self._is_connected:
            return True
        
        try:
            # Late-2025 best practice: Send mu-law 8kHz directly to Deepgram
            # No conversion needed - Deepgram accepts mulaw encoding natively!
            headers = {"Authorization": f"Token {self.config.deepgram_api_key}"}

            # Prefer v2 for modern features like language detection; fall back for compatibility.
            # Also try multiple models because availability can vary by account.
            candidates: list[tuple[str, str]] = []
            for model in ("nova-3", "nova-2"):
                common = (
                    f"?model={model}"
                    f"&encoding=mulaw"
                    f"&sample_rate=8000"
                    f"&channels=1"
                    f"&punctuate=true"
                    f"&interim_results=true"
                    f"&smart_format=true"
                )

                if not USE_DEEPGRAM_V1_ONLY:
                    candidates.append(
                        (
                            f"v2_detect_language_{model}",
                            DEEPGRAM_V2_URL
                            + common
                            + "&vad_events=true"
                            + "&detect_language=true",
                        )
                    )

                candidates.append(
                    (
                        f"v1_detect_language_{model}",
                        DEEPGRAM_V1_URL
                        + common
                        + "&vad_events=true"
                        + "&endpointing=300"
                        + "&detect_language=true",
                    )
                )

                # Last-resort fallback: pin language so the agent can still respond.
                candidates.append(
                    (
                        f"v1_pinned_language_{model}",
                        DEEPGRAM_V1_URL
                        + common
                        + "&vad_events=true"
                        + "&endpointing=300"
                        + f"&language={self.language}",
                    )
                )

            last_error: Optional[BaseException] = None
            for label, url in candidates:
                try:
                    logger.info("Connecting to Deepgram", endpoint=label)
                    self._ws = await websockets.connect(
                        url,
                        additional_headers=headers,
                        open_timeout=10,
                    )
                    logger.info("Deepgram STT connected", endpoint=label)
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Deepgram connection attempt failed",
                        endpoint=label,
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    self._ws = None

            if not self._ws:
                raise RuntimeError(
                    f"All Deepgram connection attempts failed: {type(last_error).__name__ if last_error else 'unknown'}"
                )

            self._is_connected = True
            
            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())
            return True
            
        except Exception as e:
            logger.error(
                "Deepgram connection failed",
                error_type=type(e).__name__,
                error=str(e),
            )
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Deepgram."""
        self._is_connected = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning("Error closing Deepgram connection", error=str(e))
        
        self._ws = None
        logger.info("Deepgram STT disconnected")
    
    async def send_audio(self, audio_bytes: bytes) -> None:
        """Send audio data to Deepgram."""
        if not self._is_connected or not self._ws:
            return
        
        try:
            self._last_audio_time = time.time()
            self._metrics.total_audio_ms += len(audio_bytes) / 32
            await self._ws.send(audio_bytes)
        except Exception as e:
            logger.error("Failed to send audio to Deepgram", error=str(e))
    
    async def _receive_loop(self) -> None:
        """Receive and process messages from Deepgram."""
        try:
            async for message in self._ws:
                if not self._is_connected:
                    break
                
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from Deepgram")
                except Exception as e:
                    logger.error("Error processing Deepgram message", error=str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Deepgram connection closed")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Deepgram receive loop error", error=str(e))
        finally:
            self._is_connected = False
    
    async def _handle_message(self, data: dict) -> None:
        """Handle a message from Deepgram."""
        msg_type = data.get("type", "")
        msg_type_norm = msg_type.lower() if isinstance(msg_type, str) else ""

        self._maybe_update_language_detection(data)
        
        if msg_type_norm == "results":
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])
            
            if not alternatives:
                return
            
            transcript = alternatives[0].get("transcript", "")
            confidence = alternatives[0].get("confidence", 0.0)
            is_final = data.get("is_final", False)
            speech_final = data.get("speech_final", False)
            
            if not transcript:
                return
            
            # Track word count
            self._word_count = len(transcript.split())
            if is_final:
                self._current_transcript = transcript
            
            # Calculate latency
            latency_ms = 0.0
            if self._last_audio_time > 0:
                latency_ms = (time.time() - self._last_audio_time) * 1000
            
            self._metrics.record_transcript(is_final, latency_ms)
            
            result = TranscriptionResult(
                text=transcript,
                is_final=is_final,
                confidence=confidence,
                speech_final=speech_final,
                detected_language=self._detected_language,
                language_confidence=self._language_confidence,
                latency_ms=latency_ms,
            )
            
            logger.debug(
                "STT transcript",
                text=transcript[:50] if len(transcript) > 50 else transcript,
                is_final=is_final,
                speech_final=speech_final,
            )
            
            if self._on_transcript:
                await self._on_transcript(result)
        
        elif msg_type_norm in ("speechstarted", "speech_started", "speech_start"):
            logger.debug("STT speech started")
            if self._on_speech_started:
                await self._on_speech_started()
                
        elif msg_type_norm in ("utteranceend", "utterance_end"):
            logger.debug("Utterance end detected")
            if self._current_transcript and self._on_transcript:
                result = TranscriptionResult(
                    text=self._current_transcript,
                    is_final=True,
                    speech_final=True,
                    detected_language=self._detected_language,
                    language_confidence=self._language_confidence,
                )
                await self._on_transcript(result)
            self._current_transcript = ""
            self._word_count = 0
            
        elif msg_type_norm == "error":
            logger.error(
                "Deepgram error",
                error=data.get("message", "Unknown"),
                details=data,
            )
    
    def reset_state(self) -> None:
        """Reset transcript state."""
        self._current_transcript = ""
        self._word_count = 0

    def _maybe_update_language_detection(self, data: dict) -> None:
        detected, confidence = self._extract_language_detection(data)
        if detected:
            self._detected_language = detected
        if confidence is not None:
            self._language_confidence = confidence

    @staticmethod
    def _extract_language_detection(data: dict) -> tuple[Optional[str], Optional[float]]:
        """
        Best-effort extraction of Deepgram language detection fields.

        Deepgram may provide these at top-level, under `metadata`, or under `channel`
        depending on model/endpoint.
        """
        detected = None
        confidence = None

        candidates: list[dict] = [data]
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            candidates.append(metadata)
        channel = data.get("channel")
        if isinstance(channel, dict):
            candidates.append(channel)

        for c in candidates:
            if detected is None:
                value = c.get("detected_language")
                if isinstance(value, str) and value.strip():
                    detected = value.strip()
            if confidence is None:
                value = c.get("language_confidence")
                if isinstance(value, (int, float)):
                    confidence = float(value)
                elif isinstance(value, str):
                    try:
                        confidence = float(value)
                    except ValueError:
                        pass

        return detected, confidence


class STTManager:
    """Manager for STT operations in a call."""
    
    def __init__(self):
        self._stt: Optional[DeepgramSTT] = None
        self._transcript_callback: Optional[Callable[[TranscriptionResult], Awaitable[None]]] = None
        self._speech_started_callback: Optional[Callable[[], Awaitable[None]]] = None
    
    def set_transcript_callback(
        self, 
        callback: Callable[[TranscriptionResult], Awaitable[None]]
    ) -> None:
        self._transcript_callback = callback
    
    def set_speech_started_callback(
        self,
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        self._speech_started_callback = callback
    
    async def start(self, language: Optional[str] = None) -> bool:
        return await self.restart(language=language)
    
    async def restart(self, language: Optional[str] = None) -> bool:
        """
        (Re)start STT, optionally changing language.
        
        If STT is already running, this attempts to connect the new STT first and
        only swaps over on success (to avoid dropping the existing connection).
        """
        # If there is no active STT, just start a new one.
        if self._stt is None:
            self._stt = DeepgramSTT(
                on_transcript=self._transcript_callback,
                on_speech_started=self._speech_started_callback,
                language=language,
            )
            return await self._stt.connect()
        
        previous = self._stt
        candidate = DeepgramSTT(
            on_transcript=self._transcript_callback,
            on_speech_started=self._speech_started_callback,
            language=language,
        )
        
        ok = await candidate.connect()
        if not ok:
            try:
                await candidate.disconnect()
            except Exception:
                pass
            return False
        
        # Swap to the new STT and then close the previous one.
        self._stt = candidate
        try:
            await previous.disconnect()
        except Exception:
            pass
        
        return True
    
    async def stop(self) -> None:
        if self._stt:
            await self._stt.disconnect()
            self._stt = None
    
    async def send_audio(self, audio_bytes: bytes) -> None:
        if self._stt and self._stt.is_connected:
            await self._stt.send_audio(audio_bytes)
    
    @property
    def word_count(self) -> int:
        return self._stt.current_word_count if self._stt else 0
    
    def reset_state(self) -> None:
        if self._stt:
            self._stt.reset_state()
    
    @property
    def metrics(self) -> Optional[STTMetrics]:
        return self._stt.metrics if self._stt else None
