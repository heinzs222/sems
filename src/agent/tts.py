"""
Cartesia Text-to-Speech streaming client using WebSocket API.

Late-2025 Best Practice:
- Request 8kHz PCM directly from Cartesia (no resampling!)
- Only conversion needed: PCM 8kHz -> mu-law 8kHz (fast)
- Supports cancel context for clean barge-in interruptions

This eliminates CPU-heavy 22050->8000 resampling that causes choppy audio.
"""

import asyncio
import json
import base64
from typing import AsyncGenerator, Optional, Any
from dataclasses import dataclass, field
import time

import structlog
import websockets

from src.agent.config import get_config
from src.agent.audio import pcm_8k_to_ulaw

logger = structlog.get_logger(__name__)

# Cartesia output format - 8kHz is supported per their docs (2025-12-25)
# No resampling needed - direct output at Twilio's native rate!
CARTESIA_SAMPLE_RATE = 8000
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_API_VERSION = "2024-06-10"  # Use supported API version


@dataclass
class TTSMetrics:
    """Metrics for TTS performance."""
    total_requests: int = 0
    total_characters: int = 0
    total_audio_ms: float = 0.0
    avg_first_byte_ms: float = 0.0
    avg_total_ms: float = 0.0
    
    def record_synthesis(
        self, 
        characters: int, 
        audio_ms: float, 
        first_byte_ms: float, 
        total_ms: float
    ) -> None:
        self.total_requests += 1
        self.total_characters += characters
        self.total_audio_ms += audio_ms
        
        # Running averages
        n = self.total_requests
        self.avg_first_byte_ms = (self.avg_first_byte_ms * (n - 1) + first_byte_ms) / n
        self.avg_total_ms = (self.avg_total_ms * (n - 1) + total_ms) / n


@dataclass
class TTSChunk:
    """A chunk of synthesized audio."""
    audio_bytes: bytes  # Raw mu-law bytes for Twilio
    is_final: bool = False
    timestamp: float = field(default_factory=time.time)


class CartesiaTTS:
    """
    Cartesia streaming TTS client using WebSocket API.
    
    Converts text to speech and outputs mu-law 8kHz audio for Twilio.
    """
    
    def __init__(self, config: Optional[Any] = None):
        if config is None:
            config = get_config()
        
        self.config = config
        self._metrics = TTSMetrics()
        self._is_cancelled = False
        self._current_context_id: Optional[str] = None
        self._current_ws = None
    
    @property
    def metrics(self) -> TTSMetrics:
        return self._metrics
    
    def cancel(self) -> None:
        """Cancel current synthesis."""
        self._is_cancelled = True
        logger.debug("TTS synthesis cancelled")
    
    async def cancel_context(self) -> None:
        """
        Cancel the current Cartesia context (for barge-in interruptions).
        
        Late-2025 best practice: Send cancel to Cartesia before clearing Twilio.
        This ensures clean interruption without audio artifacts.
        """
        if self._current_ws and self._current_context_id:
            try:
                cancel_request = {
                    "context_id": self._current_context_id,
                    "cancel": True
                }
                await self._current_ws.send(json.dumps(cancel_request))
                logger.debug("Sent Cartesia cancel context", context_id=self._current_context_id)
            except Exception as e:
                logger.warning("Failed to cancel Cartesia context", error=str(e))
        
        self._is_cancelled = True
        self._current_context_id = None
    
    def reset_cancel(self) -> None:
        """Reset cancellation flag."""
        self._is_cancelled = False
    
    async def synthesize_streaming(
        self,
        text: str,
        voice_id: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        Synthesize text to speech with streaming output using WebSocket.
        """
        if not text or not text.strip():
            return
        
        self.reset_cancel()
        
        if voice_id is None:
            voice_id = self.config.cartesia_voice_id
        
        start_time = time.time()
        first_byte_time = None
        total_audio_bytes = 0
        
        try:
            logger.debug("Starting TTS synthesis", text_length=len(text))
            
            # Connect to Cartesia WebSocket with API version
            url = f"{CARTESIA_WS_URL}?api_key={self.config.cartesia_api_key}&cartesia_version={CARTESIA_API_VERSION}"
            
            async with websockets.connect(url) as ws:
                self._current_ws = ws
                
                # Generate a valid context_id
                import uuid
                context_id = str(uuid.uuid4()).replace("-", "")
                self._current_context_id = context_id
                
                # Late-2025 best practice: Request 8kHz PCM directly!
                # This eliminates the 22050->8000 resampling that causes choppy audio
                request = {
                    "context_id": context_id,
                    "model_id": "sonic-english",
                    "transcript": text,
                    "voice": {
                        "mode": "id",
                        "id": voice_id
                    },
                    "output_format": {
                        "container": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": CARTESIA_SAMPLE_RATE  # 8kHz - no resampling needed!
                    },
                    "continue": False
                }
                
                await ws.send(json.dumps(request))
                
                # Receive audio chunks
                async for message in ws:
                    if self._is_cancelled:
                        logger.debug("TTS synthesis cancelled mid-stream")
                        break
                    
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")
                        
                        # Log for debugging
                        logger.debug("Cartesia message", msg_type=msg_type, keys=list(data.keys()))
                        
                        if msg_type == "chunk":
                            audio_b64 = data.get("data")
                            if audio_b64:
                                audio_data = base64.b64decode(audio_b64)
                                
                                if first_byte_time is None:
                                    first_byte_time = time.time()
                                    # Log audio format info on first chunk
                                    logger.info(
                                        "Cartesia first audio chunk",
                                        bytes_received=len(audio_data),
                                        expected_rate=CARTESIA_SAMPLE_RATE,
                                        samples=len(audio_data)//2  # PCM S16 = 2 bytes per sample
                                    )
                                
                                # Convert PCM 8kHz -> mu-law 8kHz (no resampling!)
                                ulaw_audio = pcm_8k_to_ulaw(audio_data)
                                
                                if ulaw_audio:
                                    total_audio_bytes += len(ulaw_audio)
                                    yield TTSChunk(audio_bytes=ulaw_audio, is_final=False)
                        
                        elif msg_type == "done":
                            logger.debug("Cartesia synthesis done")
                            break
                        
                        elif msg_type == "error":
                            logger.error("Cartesia error", error=data.get("message"), details=data)
                            break
                        
                        elif msg_type == "flush_done":
                            # Cartesia can emit `flush_done` before `done` with a small delay.
                            # Treat it as end-of-audio so we can flush the last partial 20ms frame immediately
                            # (prevents an audible pause before the final word).
                            logger.debug("Cartesia flush done")
                            break
                        
                        # Handle different response formats
                        elif "audio" in data:
                            audio_b64 = data.get("audio")
                            if audio_b64:
                                audio_data = base64.b64decode(audio_b64)
                                if first_byte_time is None:
                                    first_byte_time = time.time()
                                ulaw_audio = pcm_8k_to_ulaw(audio_data)
                                if ulaw_audio:
                                    total_audio_bytes += len(ulaw_audio)
                                    yield TTSChunk(audio_bytes=ulaw_audio, is_final=False)
                            
                    except json.JSONDecodeError:
                        # Might be binary audio data directly
                        if isinstance(message, bytes):
                            if first_byte_time is None:
                                first_byte_time = time.time()
                            ulaw_audio = pcm_8k_to_ulaw(message)
                            if ulaw_audio:
                                total_audio_bytes += len(ulaw_audio)
                                yield TTSChunk(audio_bytes=ulaw_audio, is_final=False)
                
                # Clear context tracking
                self._current_context_id = None
                self._current_ws = None
            
            # Final chunk marker
            if not self._is_cancelled:
                yield TTSChunk(audio_bytes=b"", is_final=True)
            
            # Record metrics
            end_time = time.time()
            audio_duration_ms = total_audio_bytes / 8
            
            self._metrics.record_synthesis(
                characters=len(text),
                audio_ms=audio_duration_ms,
                first_byte_ms=(first_byte_time - start_time) * 1000 if first_byte_time else 0,
                total_ms=(end_time - start_time) * 1000,
            )
            
            logger.debug(
                "TTS synthesis complete",
                text_length=len(text),
                audio_bytes=total_audio_bytes,
            )
            
        except Exception as e:
            logger.error("TTS synthesis failed", error=str(e))
            yield TTSChunk(audio_bytes=b"", is_final=True)
    
    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> bytes:
        """Synthesize text to speech (non-streaming)."""
        audio_chunks = []
        async for chunk in self.synthesize_streaming(text, voice_id):
            if chunk.audio_bytes:
                audio_chunks.append(chunk.audio_bytes)
        return b"".join(audio_chunks)
    
    async def close(self) -> None:
        """Close the client."""
        pass


class TTSManager:
    """
    Manager for TTS operations in a call.
    
    Handles synthesis lifecycle and provides interruption support.
    """
    
    def __init__(self):
        self._tts: Optional[CartesiaTTS] = None
        self._current_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Initialize TTS for a call."""
        self._tts = CartesiaTTS()
    
    async def stop(self) -> None:
        """Stop TTS and cleanup."""
        self.cancel_current()
        
        if self._tts:
            await self._tts.close()
            self._tts = None
    
    async def cancel_context(self) -> None:
        """Cancel the current Cartesia context (if any)."""
        if self._tts:
            await self._tts.cancel_context()
    
    def cancel_current(self) -> None:
        """Cancel current synthesis."""
        if self._tts:
            self._tts.cancel()
        
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            self._current_task = None
    
    async def synthesize_streaming(
        self, 
        text: str,
        voice_id: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        Synthesize text with streaming.
        
        Args:
            text: Text to synthesize
            
        Yields:
            TTSChunk with audio bytes
        """
        if not self._tts:
            self._tts = CartesiaTTS()
        
        async for chunk in self._tts.synthesize_streaming(text, voice_id=voice_id):
            yield chunk
    
    @property
    def metrics(self) -> Optional[TTSMetrics]:
        """Get TTS metrics."""
        return self._tts.metrics if self._tts else None


# Alternative: Simple async TTS function for one-off synthesis
async def synthesize_text(
    text: str,
    voice_id: Optional[str] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Convenience function to synthesize text.
    
    Args:
        text: Text to synthesize
        voice_id: Optional voice ID
        
    Yields:
        Mu-law audio bytes
    """
    tts = CartesiaTTS()
    
    try:
        async for chunk in tts.synthesize_streaming(text, voice_id):
            if chunk.audio_bytes:
                yield chunk.audio_bytes
    finally:
        await tts.close()
