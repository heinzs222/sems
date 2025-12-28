"""
Twilio Media Streams WebSocket Protocol Handler.

Late-2025 Best Practice:
- Use mark messages to track playback completion
- Use clear messages for clean barge-in interruptions
- Track mark round-trip times for latency monitoring

Twilio sends JSON messages with events:
- connected: Initial connection
- start: Stream started, contains streamSid and callSid
- media: Audio data as base64 mu-law 8kHz
- mark: Playback marker acknowledgment
- dtmf: DTMF tone detected
- stop: Stream stopped

Outbound messages:
- media: Send audio as base64 mu-law 8kHz
- mark: Request playback acknowledgment
- clear: Clear buffered audio (for interruption)
"""

import base64
import msgspec
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

import structlog

from src.agent.audio import chunk_audio, TWILIO_FRAME_SIZE

logger = structlog.get_logger(__name__)

# Create global msgspec encoder/decoder
decoder = msgspec.json.Decoder()
encoder = msgspec.json.Encoder()


class TwilioEventType(str, Enum):
    """Twilio WebSocket event types."""
    CONNECTED = "connected"
    START = "start"
    MEDIA = "media"
    MARK = "mark"
    DTMF = "dtmf"
    STOP = "stop"


@dataclass
class TwilioStartEvent:
    """Parsed Twilio start event."""
    stream_sid: str
    call_sid: str
    account_sid: str
    tracks: List[str]
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_message(cls, message: Dict[str, Any]) -> "TwilioStartEvent":
        """Parse from Twilio message."""
        start = message.get("start", {})
        return cls(
            stream_sid=message.get("streamSid", ""),
            call_sid=start.get("callSid", ""),
            account_sid=start.get("accountSid", ""),
            tracks=start.get("tracks", []),
            custom_parameters=start.get("customParameters", {}),
        )


@dataclass
class TwilioMediaEvent:
    """Parsed Twilio media event."""
    stream_sid: str
    track: str
    chunk: int
    timestamp: str
    payload: bytes  # Decoded audio bytes (mu-law)
    
    @classmethod
    def from_message(cls, message: Dict[str, Any]) -> "TwilioMediaEvent":
        """Parse from Twilio message."""
        media = message.get("media", {})
        payload_b64 = media.get("payload", "")
        
        try:
            payload = base64.b64decode(payload_b64)
        except Exception:
            payload = b""
        
        return cls(
            stream_sid=message.get("streamSid", ""),
            track=media.get("track", "inbound"),
            chunk=int(media.get("chunk", 0)),
            timestamp=media.get("timestamp", ""),
            payload=payload,
        )


@dataclass
class TwilioMarkEvent:
    """Parsed Twilio mark event."""
    stream_sid: str
    name: str
    
    @classmethod
    def from_message(cls, message: Dict[str, Any]) -> "TwilioMarkEvent":
        """Parse from Twilio message."""
        mark = message.get("mark", {})
        return cls(
            stream_sid=message.get("streamSid", ""),
            name=mark.get("name", ""),
        )


@dataclass
class TwilioDTMFEvent:
    """Parsed Twilio DTMF event."""
    stream_sid: str
    digit: str
    
    @classmethod
    def from_message(cls, message: Dict[str, Any]) -> "TwilioDTMFEvent":
        """Parse from Twilio message."""
        dtmf = message.get("dtmf", {})
        return cls(
            stream_sid=message.get("streamSid", ""),
            digit=dtmf.get("digit", ""),
        )


@dataclass
class CallState:
    """State for an active Twilio call."""
    stream_sid: str = ""
    call_sid: str = ""
    account_sid: str = ""
    sequence_number: int = 0
    is_active: bool = True
    playback_generation_id: int = 0
    mark_sequence: int = 0
    pending_marks: Dict[str, float] = field(default_factory=dict)  # mark_name -> send_time
    mark_rtt_samples: List[float] = field(default_factory=list)  # RTT samples in ms
    
    def next_sequence(self) -> int:
        """Get next sequence number for outbound media."""
        self.sequence_number += 1
        return self.sequence_number
    
    @property
    def avg_mark_rtt_ms(self) -> float:
        """Average mark round-trip time in ms."""
        if not self.mark_rtt_samples:
            return 0.0
        return sum(self.mark_rtt_samples) / len(self.mark_rtt_samples)


def parse_twilio_message(raw_message: str) -> tuple[TwilioEventType, Any]:
    """
    Parse a raw Twilio WebSocket message.
    
    Args:
        raw_message: Raw JSON string from Twilio
        
    Returns:
        Tuple of (event_type, parsed_event)
        
    Raises:
        ValueError: If message cannot be parsed
    """
    try:
        # Fast msgspec decoding
        message = decoder.decode(raw_message.encode("utf-8") if isinstance(raw_message, str) else raw_message)
        # Convert to dict if needed (msgspec decodes to Struct by default if typed, or dict/list if untyped)
        # But here we just need it to be subscriptable
    except msgspec.DecodeError as e:
        logger.error("Failed to parse Twilio message", error=str(e))
        raise ValueError(f"Invalid JSON: {e}")
    
    event_type_str = message.get("event", "")
    
    try:
        event_type = TwilioEventType(event_type_str)
    except ValueError:
        logger.warning("Unknown Twilio event type", event_type=event_type_str)
        raise ValueError(f"Unknown event type: {event_type_str}")
    
    if event_type == TwilioEventType.CONNECTED:
        return event_type, message
    elif event_type == TwilioEventType.START:
        return event_type, TwilioStartEvent.from_message(message)
    elif event_type == TwilioEventType.MEDIA:
        return event_type, TwilioMediaEvent.from_message(message)
    elif event_type == TwilioEventType.MARK:
        return event_type, TwilioMarkEvent.from_message(message)
    elif event_type == TwilioEventType.DTMF:
        return event_type, TwilioDTMFEvent.from_message(message)
    elif event_type == TwilioEventType.STOP:
        return event_type, message
    else:
        return event_type, message


def create_media_message(
    stream_sid: str,
    audio_payload: bytes,
) -> str:
    """
    Create a Twilio media message.
    
    Args:
        stream_sid: The stream SID
        audio_payload: Raw mu-law audio bytes (should be 160 bytes for 20ms)
        
    Returns:
        JSON string to send to Twilio
    """
    payload_b64 = base64.b64encode(audio_payload).decode("utf-8")
    
    message = {
        "event": "media",
        "streamSid": stream_sid,
        "media": {
            "payload": payload_b64
        }
    }
    
    return encoder.encode(message).decode("utf-8")


def create_mark_message(stream_sid: str, name: str) -> str:
    """
    Create a Twilio mark message.
    
    Marks are used to get acknowledgment when audio has been played.
    
    Args:
        stream_sid: The stream SID
        name: Unique name for this mark
        
    Returns:
        JSON string to send to Twilio
    """
    message = {
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {
            "name": name
        }
    }
    
    return encoder.encode(message).decode("utf-8")


def create_clear_message(stream_sid: str) -> str:
    """
    Create a Twilio clear message.
    
    This clears any buffered audio on Twilio's side, used for interruption.
    
    Args:
        stream_sid: The stream SID
        
    Returns:
        JSON string to send to Twilio
    """
    message = {
        "event": "clear",
        "streamSid": stream_sid
    }
    
    return encoder.encode(message).decode("utf-8")


class TwilioProtocolHandler:
    """
    High-level handler for Twilio WebSocket protocol.
    
    Manages call state and provides methods for sending/receiving messages.
    """
    
    def __init__(self):
        self.call_state: Optional[CallState] = None
        self._mark_counter = 0
    
    @property
    def stream_sid(self) -> str:
        """Get the current stream SID."""
        return self.call_state.stream_sid if self.call_state else ""
    
    @property
    def call_sid(self) -> str:
        """Get the current call SID."""
        return self.call_state.call_sid if self.call_state else ""
    
    @property
    def is_active(self) -> bool:
        """Check if the call is active."""
        return self.call_state is not None and self.call_state.is_active
    
    def handle_start(self, event: TwilioStartEvent) -> None:
        """Handle a start event and initialize call state."""
        self.call_state = CallState(
            stream_sid=event.stream_sid,
            call_sid=event.call_sid,
            account_sid=event.account_sid,
        )
        logger.info(
            "Call started",
            stream_sid=event.stream_sid,
            call_sid=event.call_sid,
        )
    
    def handle_stop(self) -> None:
        """Handle a stop event."""
        if self.call_state:
            self.call_state.is_active = False
            logger.info(
                "Call stopped",
                stream_sid=self.call_state.stream_sid,
                call_sid=self.call_state.call_sid,
            )
    
    def handle_mark(self, event: TwilioMarkEvent) -> float:
        """
        Handle a mark acknowledgment and calculate RTT.
        
        Returns:
            Round-trip time in ms, or 0 if mark not found
        """
        import time
        rtt_ms = 0.0
        if self.call_state:
            mark_gen = self._parse_mark_generation(event.name)
            if (
                mark_gen is not None
                and mark_gen != self.call_state.playback_generation_id
            ):
                logger.debug(
                    "Ignoring stale mark acknowledgment",
                    mark_name=event.name,
                    mark_generation=mark_gen,
                    current_generation=self.call_state.playback_generation_id,
                )
                return 0.0
            send_time = self.call_state.pending_marks.pop(event.name, None)
            if send_time:
                rtt_ms = (time.time() - send_time) * 1000
                self.call_state.mark_rtt_samples.append(rtt_ms)
                # Keep only last 20 samples
                if len(self.call_state.mark_rtt_samples) > 20:
                    self.call_state.mark_rtt_samples.pop(0)
            logger.debug("Mark acknowledged", mark_name=event.name, rtt_ms=round(rtt_ms, 2))
        return rtt_ms
    
    def create_audio_messages(self, audio_bytes: bytes) -> List[str]:
        """
        Create chunked audio messages for Twilio.
        
        Chunks audio into 20ms frames (160 bytes) for smooth playback.
        
        Args:
            audio_bytes: Raw mu-law audio bytes
            
        Returns:
            List of JSON messages to send
        """
        if not self.call_state:
            return []
        
        messages = []
        for chunk in chunk_audio(audio_bytes, TWILIO_FRAME_SIZE):
            msg = create_media_message(self.call_state.stream_sid, chunk)
            messages.append(msg)
        
        return messages
    
    def create_mark(self, name: Optional[str] = None) -> str:
        """
        Create a mark message.
        
        Args:
            name: Optional mark name (auto-generated if not provided)
            
        Returns:
            JSON message to send
        """
        if not self.call_state:
            return ""
        
        if name is None:
            self._mark_counter += 1
            self.call_state.mark_sequence += 1
            name = f"g{self.call_state.playback_generation_id}_m{self.call_state.mark_sequence}"
        
        import time
        self.call_state.pending_marks[name] = time.time()  # Store send time for RTT calculation
        return create_mark_message(self.call_state.stream_sid, name)

    def bump_playback_generation(self) -> int:
        """
        Bump the playback generation id.

        Used to ignore stale marks after a Twilio `clear` (barge-in).
        """
        if not self.call_state:
            return 0

        self.call_state.playback_generation_id += 1
        self.call_state.mark_sequence = 0
        self.call_state.pending_marks.clear()
        logger.info(
            "Playback generation bumped",
            playback_generation_id=self.call_state.playback_generation_id,
        )
        return self.call_state.playback_generation_id

    def create_clear(self) -> str:
        """
        Create a clear message to flush buffered audio.
        
        Returns:
            JSON message to send
        """
        if not self.call_state:
            return ""
        
        logger.info("Clearing Twilio audio buffer", stream_sid=self.call_state.stream_sid)
        return create_clear_message(self.call_state.stream_sid)

    @staticmethod
    def _parse_mark_generation(mark_name: str) -> Optional[int]:
        """
        Parse a playback generation id from a mark name.

        Expected format for auto-generated marks: `g{gen}_m{seq}`.
        Returns None if the format doesn't match.
        """
        if not isinstance(mark_name, str):
            return None
        if not mark_name.startswith("g"):
            return None
        try:
            # g{gen}_m{seq}
            gen_part = mark_name.split("_", 1)[0]
            gen_str = gen_part[1:]
            return int(gen_str)
        except Exception:
            return None
