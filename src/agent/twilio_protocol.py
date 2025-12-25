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
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

import structlog

# 2025 Performance: Use msgspec for faster JSON (50+ messages/sec for audio)
try:
    import msgspec
    _decoder = msgspec.json.Decoder()
    _encoder = msgspec.json.Encoder()
    
    def json_loads(data):
        return _decoder.decode(data)
    
    def json_dumps(obj):
        return _encoder.encode(obj).decode('utf-8')
    
except ImportError:
    import json
    json_loads = json.loads
    json_dumps = json.dumps

from src.agent.audio import chunk_audio, TWILIO_FRAME_SIZE

logger = structlog.get_logger(__name__)


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
        message = json_loads(raw_message)
    except Exception as e:
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
    
    return json_dumps(message)


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
    
    return json_dumps(message)


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
    
    return json_dumps(message)


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
            name = f"mark_{self._mark_counter}"
        
        import time
        self.call_state.pending_marks[name] = time.time()  # Store send time for RTT calculation
        return create_mark_message(self.call_state.stream_sid, name)
    
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
