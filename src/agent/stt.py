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

# Feature flag for Flux mode
USE_DEEPGRAM_FLUX = os.getenv("USE_DEEPGRAM_FLUX", "false").lower() == "true"


@dataclass
class TranscriptionResult:
    """Result from STT."""
    text: str
    is_final: bool
    confidence: float = 0.0
    speech_final: bool = False
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
        config: Optional[Any] = None,
    ):
        if config is None:
            config = get_config()
        
        self.config = config
        self._on_transcript = on_transcript
        self._ws = None
        self._is_connected = False
        self._metrics = STTMetrics()
        self._last_audio_time: float = 0.0
        self._current_transcript = ""
        self._word_count = 0
        self._receive_task = None
    
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
            if USE_DEEPGRAM_FLUX:
                # Flux mode: better turn detection, ~80ms chunks recommended
                params = (
                    f"?model=nova-2"
                    f"&language=en-US"
                    f"&encoding=mulaw"  # Direct mu-law - no conversion!
                    f"&sample_rate=8000"  # Twilio's native rate
                    f"&channels=1"
                    f"&punctuate=true"
                    f"&interim_results=true"
                    f"&vad_events=true"
                    f"&smart_format=true"
                )
                url = DEEPGRAM_V2_URL + params
                logger.info("Using Deepgram Flux mode (v2)")
            else:
                # Standard mode with mu-law 8kHz - simplified params
                params = (
                    f"?model=nova-2"
                    f"&language=en-US"
                    f"&encoding=mulaw"
                    f"&sample_rate=8000"
                    f"&channels=1"
                    f"&punctuate=true"
                    f"&interim_results=true"
                    f"&smart_format=true"
                    f"&endpointing=500"
                )
                url = DEEPGRAM_V1_URL + params
            headers = {"Authorization": f"Token {self.config.deepgram_api_key}"}
            
            self._ws = await websockets.connect(url, additional_headers=headers)
            self._is_connected = True
            
            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            logger.info("Deepgram STT connected")
            return True
            
        except Exception as e:
            logger.error("Deepgram connection failed", error=str(e))
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
        
        if msg_type == "Results":
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
                
        elif msg_type == "UtteranceEnd":
            logger.debug("Utterance end detected")
            if self._current_transcript and self._on_transcript:
                result = TranscriptionResult(
                    text=self._current_transcript,
                    is_final=True,
                    speech_final=True,
                )
                await self._on_transcript(result)
            self._current_transcript = ""
            self._word_count = 0
            
        elif msg_type == "Error":
            logger.error("Deepgram error", error=data.get("message", "Unknown"))
    
    def reset_state(self) -> None:
        """Reset transcript state."""
        self._current_transcript = ""
        self._word_count = 0


class STTManager:
    """Manager for STT operations in a call."""
    
    def __init__(self):
        self._stt: Optional[DeepgramSTT] = None
        self._transcript_callback: Optional[Callable[[TranscriptionResult], Awaitable[None]]] = None
    
    def set_transcript_callback(
        self, 
        callback: Callable[[TranscriptionResult], Awaitable[None]]
    ) -> None:
        self._transcript_callback = callback
    
    async def start(self) -> bool:
        self._stt = DeepgramSTT(on_transcript=self._transcript_callback)
        return await self._stt.connect()
    
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
