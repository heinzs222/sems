"""Voice Pipeline Orchestration.

Late-2025 Best Practice:
- No audio resampling: Twilio mu-law 8kHz sent directly to Deepgram
- Outbound audio pacer: 20ms frame timing with jitter buffer
- Clean interruptions: Cancel Cartesia context + Twilio clear

Constructs and manages the voice pipeline:
inbound Twilio mu-law -> STT (mulaw/8000) -> (on final transcript) route -> 
(cached OR LLM) -> TTS (8kHz PCM) -> mu-law -> pacer -> Twilio outbound

Features:
- Barge-in with configurable word threshold
- Semantic routing for fast cached responses
- Background extraction
- Proper task cancellation handling
- Outbound audio pacing for smooth playback
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, List, Dict, Any
from enum import Enum
import time

import structlog
from twilio.rest import Client as TwilioClient

from src.agent.config import get_config, Config
from src.agent.audio import twilio_ulaw_passthrough, chunk_audio, TWILIO_FRAME_SIZE
from src.agent.twilio_protocol import (
    TwilioProtocolHandler,
    TwilioEventType,
    TwilioStartEvent,
    TwilioMediaEvent,
    parse_twilio_message,
    create_media_message,
)
from src.agent.stt import STTManager, TranscriptionResult
from src.agent.tts import TTSManager, TTSChunk
from src.agent.llm import GroqLLM, get_llm
from src.agent.routing import get_semantic_router, SemanticRouter
from src.agent.extract import ExtractionQueue

logger = structlog.get_logger(__name__)


class PipelineState(str, Enum):
    """Current state of the voice pipeline."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class TurnMetrics:
    """Metrics for a single conversation turn."""
    turn_id: int = 0
    start_time: float = 0.0
    stt_ms: float = 0.0
    llm_first_token_ms: float = 0.0
    llm_total_ms: float = 0.0
    tts_start_ms: float = 0.0
    tts_total_ms: float = 0.0
    total_turn_ms: float = 0.0
    was_routed: bool = False
    was_interrupted: bool = False
    
    def finalize(self) -> None:
        """Calculate total turn time."""
        if self.start_time > 0:
            self.total_turn_ms = (time.time() - self.start_time) * 1000


@dataclass
class CallMetrics:
    """Metrics for an entire call."""
    call_sid: str = ""
    stream_sid: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    turns: List[TurnMetrics] = field(default_factory=list)
    total_interruptions: int = 0
    total_routed_responses: int = 0
    
    @property
    def duration_seconds(self) -> float:
        end = self.end_time if self.end_time > 0 else time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_turns": len(self.turns),
            "total_interruptions": self.total_interruptions,
            "total_routed_responses": self.total_routed_responses,
            "avg_turn_ms": round(
                sum(t.total_turn_ms for t in self.turns) / len(self.turns), 2
            ) if self.turns else 0,
        }


class VoicePipeline:
    """
    Main voice pipeline orchestrator.
    
    Manages the flow of audio and text between Twilio, STT, LLM, and TTS.
    """
    
    def __init__(
        self,
        send_message: Callable[[str], Awaitable[None]],
        config: Optional[Config] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            send_message: Async function to send WebSocket messages to Twilio
            config: Optional configuration (uses default if not provided)
        """
        if config is None:
            config = get_config()
        
        self.config = config
        self._send_message = send_message
        
        # Components
        self._protocol = TwilioProtocolHandler()
        self._stt = STTManager()
        self._tts = TTSManager()
        self._llm: Optional[GroqLLM] = None
        self._router: Optional[SemanticRouter] = None
        self._extraction_queue: Optional[ExtractionQueue] = None
        
        # State
        self._state = PipelineState.IDLE
        self._is_running = False
        self._current_turn = 0
        self._current_turn_metrics: Optional[TurnMetrics] = None
        self._call_metrics = CallMetrics()
        
        # TTS state
        self._tts_task: Optional[asyncio.Task] = None
        self._pending_transcript: str = ""
        
        # Outbound audio pacer (Late-2025 best practice)
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._pacer_task: Optional[asyncio.Task] = None
        self._pacer_running: bool = False
        self._pacer_underruns: int = 0
        self._jitter_buffer_ms: int = 100  # Buffer 100ms before starting playback
        self._outbound_ulaw_remainder: bytes = b""
        
        # Silence detection
        self._last_speech_time: float = 0.0
        self._silence_task: Optional[asyncio.Task] = None
        
        # Twilio client for hangup
        self._twilio_client: Optional[TwilioClient] = None
    
    @property
    def state(self) -> PipelineState:
        return self._state
    
    @property
    def call_sid(self) -> str:
        return self._protocol.call_sid
    
    @property
    def stream_sid(self) -> str:
        return self._protocol.stream_sid
    
    @property
    def metrics(self) -> CallMetrics:
        return self._call_metrics
    
    async def start(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting voice pipeline")
        
        # Initialize LLM
        self._llm = get_llm()
        
        # Initialize router if enabled
        if self.config.router_enabled:
            self._router = get_semantic_router()
            self._router.initialize()
        
        # Initialize extraction queue
        self._extraction_queue = ExtractionQueue()
        await self._extraction_queue.start()
        
        # Set up STT callback
        self._stt.set_transcript_callback(self._on_transcript)
        
        # Initialize Twilio client for hangup
        if self.config.twilio_account_sid and self.config.twilio_auth_token:
            self._twilio_client = TwilioClient(
                self.config.twilio_account_sid,
                self.config.twilio_auth_token,
            )
        
        self._is_running = True
        self._state = PipelineState.IDLE
        
        logger.info("Voice pipeline started")
    
    async def stop(self) -> None:
        """Stop all pipeline components."""
        logger.info("Stopping voice pipeline")
        
        self._is_running = False
        
        # Cancel any ongoing tasks
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
        
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()
        
        # Stop components
        await self._stt.stop()
        await self._tts.stop()
        
        if self._extraction_queue:
            await self._extraction_queue.stop()
        
        # Finalize metrics
        self._call_metrics.end_time = time.time()
        
        logger.info(
            "Voice pipeline stopped",
            metrics=self._call_metrics.to_dict(),
        )
    
    async def handle_message(self, raw_message: str) -> None:
        """
        Handle an incoming WebSocket message from Twilio.
        
        Args:
            raw_message: Raw JSON message string
        """
        try:
            event_type, event = parse_twilio_message(raw_message)
        except ValueError as e:
            logger.warning("Failed to parse Twilio message", error=str(e))
            return
        
        if event_type == TwilioEventType.CONNECTED:
            logger.debug("Twilio connected")
            
        elif event_type == TwilioEventType.START:
            await self._handle_start(event)
            
        elif event_type == TwilioEventType.MEDIA:
            await self._handle_media(event)
            
        elif event_type == TwilioEventType.MARK:
            self._protocol.handle_mark(event)
            
        elif event_type == TwilioEventType.DTMF:
            logger.info("DTMF received", digit=event.digit)
            
        elif event_type == TwilioEventType.STOP:
            self._protocol.handle_stop()
            await self.stop()
    
    async def _handle_start(self, event: TwilioStartEvent) -> None:
        """Handle call start event."""
        self._protocol.handle_start(event)
        
        # Update metrics
        self._call_metrics.call_sid = event.call_sid
        self._call_metrics.stream_sid = event.stream_sid
        
        # Start STT
        if await self._stt.start():
            self._state = PipelineState.LISTENING
            logger.info(
                "Call started, listening",
                call_sid=event.call_sid,
                stream_sid=event.stream_sid,
            )
            
            # Send initial greeting
            await self._speak_response(
                f"Hello! This is {self.config.agent_name} from {self.config.company_name}. How can I help you today?"
            )
        else:
            logger.error("Failed to start STT")
    
    async def _handle_media(self, event: TwilioMediaEvent) -> None:
        """Handle incoming audio from Twilio."""
        if not self._is_running:
            return
        
        # Late-2025 best practice: Send mu-law directly to Deepgram (no conversion!)
        # Deepgram accepts encoding=mulaw&sample_rate=8000
        audio = twilio_ulaw_passthrough(event.payload)
        
        if audio:
            await self._stt.send_audio(audio)
            self._last_speech_time = time.time()
            
            # Start silence detection if not already running
            if self._silence_task is None or self._silence_task.done():
                self._silence_task = asyncio.create_task(self._silence_detector())
    
    async def _on_transcript(self, result: TranscriptionResult) -> None:
        """
        Handle STT transcript.
        
        Only processes final transcripts. Checks for interruption on partials.
        """
        if not self._is_running:
            return
        
        # Check for interruption during TTS
        if self._state == PipelineState.SPEAKING:
            word_count = len(result.text.split())
            
            if word_count >= self.config.min_interruption_words:
                logger.info(
                    "Interruption detected",
                    word_count=word_count,
                    threshold=self.config.min_interruption_words,
                )
                await self._handle_interruption()
                self._pending_transcript = result.text
        
        # Process final transcripts
        if result.is_final and result.speech_final:
            transcript = result.text.strip()
            
            if not transcript:
                return
            
            # Use pending transcript if we were interrupted
            if self._pending_transcript:
                transcript = self._pending_transcript
                self._pending_transcript = ""
            
            logger.info("Final transcript", text=transcript[:100])
            
            # Start new turn
            self._start_turn()
            self._current_turn_metrics.stt_ms = result.latency_ms
            
            # Process the transcript
            await self._process_transcript(transcript)
    
    async def _handle_interruption(self) -> None:
        """Handle user interruption during TTS playback."""
        if self._state != PipelineState.SPEAKING:
            return
        
        logger.info("Processing interruption")
        
        self._state = PipelineState.INTERRUPTED
        self._call_metrics.total_interruptions += 1
        
        if self._current_turn_metrics:
            self._current_turn_metrics.was_interrupted = True
        
        # Cancel TTS
        self._tts.cancel_current()
        
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            try:
                await self._tts_task
            except asyncio.CancelledError:
                pass
        
        # Send clear to Twilio to stop buffered audio
        clear_msg = self._protocol.create_clear()
        if clear_msg:
            await self._send_message(clear_msg)
        
        # Reset STT state
        self._stt.reset_state()
        
        self._state = PipelineState.LISTENING
    
    async def _process_transcript(self, transcript: str) -> None:
        """
        Process a final transcript.
        
        Tries routing first, then falls back to LLM.
        """
        self._state = PipelineState.PROCESSING
        
        # Try semantic routing first
        if self._router and self._router.is_enabled:
            route_name, audio_chunks, should_hangup = self._router.try_route(transcript)
            
            if route_name and audio_chunks:
                logger.info("Using cached response", route=route_name)
                
                self._call_metrics.total_routed_responses += 1
                if self._current_turn_metrics:
                    self._current_turn_metrics.was_routed = True
                
                # Play cached audio
                await self._play_cached_audio(audio_chunks)
                
                # Handle hangup for "stop" route
                if should_hangup:
                    await self._hangup_call()
                
                self._end_turn()
                return
        
        # Fall back to LLM
        await self._generate_llm_response(transcript)
    
    async def _generate_llm_response(self, transcript: str) -> None:
        """Generate and speak an LLM response."""
        if not self._llm:
            logger.error("LLM not initialized")
            return
        
        llm_start = time.time()
        first_token_time = None
        full_response = ""
        
        # Collect LLM response and stream to TTS
        try:
            self._state = PipelineState.SPEAKING
            
            async for chunk in self._llm.generate_streaming(transcript):
                if not self._is_running or self._state == PipelineState.INTERRUPTED:
                    break
                
                if first_token_time is None:
                    first_token_time = time.time()
                    if self._current_turn_metrics:
                        self._current_turn_metrics.llm_first_token_ms = (
                            first_token_time - llm_start
                        ) * 1000
                
                full_response += chunk
            
            # Record LLM metrics
            llm_end = time.time()
            if self._current_turn_metrics:
                self._current_turn_metrics.llm_total_ms = (llm_end - llm_start) * 1000
            
            # Speak the response if not interrupted
            if self._state != PipelineState.INTERRUPTED and full_response:
                await self._speak_response(full_response)
                
                # Submit for extraction (background, non-blocking)
                if self._extraction_queue:
                    await self._extraction_queue.submit(
                        user_message=transcript,
                        assistant_message=full_response,
                        conversation_history=self._llm.get_history_messages(),
                    )
            
        except asyncio.CancelledError:
            logger.debug("LLM generation cancelled")
        except Exception as e:
            logger.error("LLM generation failed", error=str(e))
        finally:
            self._end_turn()
            if self._state == PipelineState.SPEAKING:
                self._state = PipelineState.LISTENING
    
    async def _audio_pacer(self) -> None:
        """
        Outbound audio pacer - sends frames at exactly 20ms intervals.
        
        Late-2025 best practice: Pace outbound audio to prevent buffer underflows
        that cause choppy playback. Uses a jitter buffer before starting.
        """
        self._pacer_running = True
        frame_duration = 0.020  # 20ms per frame
        next_send_time = time.time()
        
        # Pre-buffer to reduce bursty producer/consumer underruns
        prebuffer_frames = max(3, int(self._jitter_buffer_ms / 20))
        buffered = 0
        frames_buffer = []
        
        try:
            # Pre-buffer phase
            while buffered < prebuffer_frames and self._pacer_running:
                try:
                    frame_msg = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.5
                    )
                    if frame_msg is None:
                        break
                    frames_buffer.append(frame_msg)
                    buffered += 1
                except asyncio.TimeoutError:
                    break
            
            # Main pacing loop
            while self._pacer_running and self._is_running:
                try:
                    # Get frame from buffer or queue
                    if frames_buffer:
                        frame_msg = frames_buffer.pop(0)
                    else:
                        # Non-blocking get to maintain timing
                        try:
                            frame_msg = self._audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            # No frame available, send silence to maintain timing
                            frame_msg = None
                    
                    if frame_msg is None:
                        # Check for new audio with small timeout
                        try:
                            frame_msg = await asyncio.wait_for(
                                self._audio_queue.get(),
                                timeout=0.015  # Most of the 20ms window
                            )
                        except asyncio.TimeoutError:
                            # Still no audio, continue loop
                            await asyncio.sleep(0.005)
                            continue
                    
                    if frame_msg is None:  # Poison pill
                        break
                    
                    # Wait until next send time for precise 20ms intervals
                    now = time.time()
                    wait_time = next_send_time - now
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    
                    # Send frame
                    await self._send_message(frame_msg)
                    
                    # Calculate next send time (exactly 20ms later)
                    next_send_time += frame_duration
                    
                    # If we're falling behind, reset timing
                    if time.time() > next_send_time + 0.040:  # More than 2 frames behind
                        self._pacer_underruns += 1
                        if self._pacer_underruns % 10 == 0:
                            logger.warning(
                                "Audio pacer reset timing", 
                                underruns=self._pacer_underruns,
                                behind_ms=(time.time() - next_send_time)*1000
                            )
                        next_send_time = time.time()
                        
                except asyncio.TimeoutError:
                    continue  # No audio available, keep waiting
                    
        except asyncio.CancelledError:
            pass
        finally:
            self._pacer_running = False
    
    async def _start_pacer(self) -> None:
        """Start the audio pacer task."""
        if self._pacer_task is None or self._pacer_task.done():
            self._audio_queue = asyncio.Queue()
            self._pacer_task = asyncio.create_task(self._audio_pacer())
    
    async def _stop_pacer(self) -> None:
        """Stop the audio pacer and clear queue."""
        self._pacer_running = False
        
        # Clear the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Send poison pill
        await self._audio_queue.put(None)
        
        if self._pacer_task and not self._pacer_task.done():
            self._pacer_task.cancel()
            try:
                await self._pacer_task
            except asyncio.CancelledError:
                pass
    
    async def _speak_response(self, text: str) -> None:
        """
        Synthesize and stream TTS response to Twilio via audio pacer.
        
        Late-2025 best practice: Queue audio frames to pacer for smooth 20ms timing.
        """
        if not text or not self._is_running:
            return
        
        tts_start = time.time()
        first_audio_time = None
        
        self._state = PipelineState.SPEAKING
        
        # Start pacer if not running
        await self._start_pacer()
        
        try:
            # Reset streaming frame remainder for this utterance.
            # Important: Do NOT pad to 160 bytes per TTS chunk; only pad once at the end,
            # otherwise we insert silence between streamed chunks which sounds like "ticking".
            self._outbound_ulaw_remainder = b""

            async for chunk in self._tts.synthesize_streaming(text):
                if not self._is_running or self._state == PipelineState.INTERRUPTED:
                    break

                if chunk.audio_bytes:
                    if first_audio_time is None:
                        first_audio_time = time.time()
                        if self._current_turn_metrics:
                            self._current_turn_metrics.tts_start_ms = (
                                first_audio_time - tts_start
                            ) * 1000
                    
                    # Queue audio frames to pacer for smooth playback.
                    # Preserve framing across streaming chunks to avoid injecting padding between chunks.
                    self._outbound_ulaw_remainder += chunk.audio_bytes
                    while len(self._outbound_ulaw_remainder) >= TWILIO_FRAME_SIZE:
                        frame = self._outbound_ulaw_remainder[:TWILIO_FRAME_SIZE]
                        self._outbound_ulaw_remainder = self._outbound_ulaw_remainder[TWILIO_FRAME_SIZE:]
                        msg = create_media_message(self._protocol.stream_sid, frame)
                        await self._audio_queue.put(msg)

                if chunk.is_final and self._outbound_ulaw_remainder:
                    frame = self._outbound_ulaw_remainder.ljust(TWILIO_FRAME_SIZE, b"\xff")
                    self._outbound_ulaw_remainder = b""
                    msg = create_media_message(self._protocol.stream_sid, frame)
                    await self._audio_queue.put(msg)
            
            # Record TTS metrics
            if self._current_turn_metrics:
                self._current_turn_metrics.tts_total_ms = (time.time() - tts_start) * 1000
            
            # Wait for queue to drain before sending mark
            while not self._audio_queue.empty() and self._state == PipelineState.SPEAKING:
                await asyncio.sleep(0.01)
            
            # Send mark to track when audio finishes
            mark_msg = self._protocol.create_mark()
            if mark_msg:
                await self._send_message(mark_msg)
                
        except asyncio.CancelledError:
            logger.debug("TTS streaming cancelled")
        except Exception as e:
            logger.error("TTS streaming failed", error=str(e))
        finally:
            if self._state == PipelineState.SPEAKING:
                self._state = PipelineState.LISTENING
    
    async def _play_cached_audio(self, audio_chunks: List[bytes]) -> None:
        """
        Play pre-rendered cached audio.
        
        Args:
            audio_chunks: List of 160-byte mu-law chunks
        """
        self._state = PipelineState.SPEAKING
        
        try:
            for chunk in audio_chunks:
                if not self._is_running or self._state == PipelineState.INTERRUPTED:
                    break
                
                msg = create_media_message(self._protocol.stream_sid, chunk)
                await self._send_message(msg)
                # 20ms per chunk
                await asyncio.sleep(0.015)  # Slightly less than 20ms to allow for processing
            
            # Send mark
            mark_msg = self._protocol.create_mark()
            if mark_msg:
                await self._send_message(mark_msg)
                
        except Exception as e:
            logger.error("Cached audio playback failed", error=str(e))
        finally:
            if self._state == PipelineState.SPEAKING:
                self._state = PipelineState.LISTENING
    
    async def _silence_detector(self) -> None:
        """
        Detect prolonged silence and prompt if needed.
        
        Runs as a background task.
        """
        try:
            while self._is_running:
                await asyncio.sleep(0.5)
                
                if self._state != PipelineState.LISTENING:
                    continue
                
                silence_duration = time.time() - self._last_speech_time
                
                if silence_duration > self.config.silence_timeout_seconds * 3:
                    # Extended silence - prompt user
                    logger.debug("Extended silence detected")
                    await self._speak_response(
                        "Are you still there? Is there anything else I can help you with?"
                    )
                    self._last_speech_time = time.time()
                    
        except asyncio.CancelledError:
            pass
    
    async def _hangup_call(self) -> None:
        """Hang up the call via Twilio REST API."""
        if not self._twilio_client or not self.call_sid:
            logger.warning("Cannot hangup - missing Twilio client or call_sid")
            return
        
        try:
            # Small delay to let the goodbye audio finish
            await asyncio.sleep(2.0)
            
            # Update call status to completed
            call = self._twilio_client.calls(self.call_sid).update(status="completed")
            logger.info("Call hung up", call_sid=self.call_sid)
            
        except Exception as e:
            logger.error("Failed to hang up call", error=str(e))
    
    def _start_turn(self) -> None:
        """Start a new conversation turn."""
        self._current_turn += 1
        self._current_turn_metrics = TurnMetrics(
            turn_id=self._current_turn,
            start_time=time.time(),
        )
    
    def _end_turn(self) -> None:
        """End the current conversation turn."""
        if self._current_turn_metrics:
            self._current_turn_metrics.finalize()
            self._call_metrics.turns.append(self._current_turn_metrics)
            
            logger.info(
                "Turn completed",
                turn_id=self._current_turn_metrics.turn_id,
                stt_ms=round(self._current_turn_metrics.stt_ms, 2),
                llm_first_token_ms=round(self._current_turn_metrics.llm_first_token_ms, 2),
                llm_total_ms=round(self._current_turn_metrics.llm_total_ms, 2),
                tts_start_ms=round(self._current_turn_metrics.tts_start_ms, 2),
                tts_total_ms=round(self._current_turn_metrics.tts_total_ms, 2),
                total_turn_ms=round(self._current_turn_metrics.total_turn_ms, 2),
                was_routed=self._current_turn_metrics.was_routed,
                was_interrupted=self._current_turn_metrics.was_interrupted,
            )
            
            self._current_turn_metrics = None


async def create_pipeline(
    send_message: Callable[[str], Awaitable[None]],
) -> VoicePipeline:
    """
    Create and start a new voice pipeline.
    
    Args:
        send_message: Function to send messages to Twilio WebSocket
        
    Returns:
        Initialized and started VoicePipeline
    """
    pipeline = VoicePipeline(send_message)
    await pipeline.start()
    return pipeline
