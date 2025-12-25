"""
Tests for Twilio protocol handling.
"""

import pytest
import json
import base64

from src.agent.twilio_protocol import (
    TwilioEventType,
    TwilioStartEvent,
    TwilioMediaEvent,
    TwilioMarkEvent,
    TwilioDTMFEvent,
    CallState,
    parse_twilio_message,
    create_media_message,
    create_mark_message,
    create_clear_message,
    TwilioProtocolHandler,
)


class TestMessageParsing:
    """Tests for parsing Twilio messages."""
    
    def test_parse_connected_event(self):
        """Test parsing connected event."""
        message = json.dumps({"event": "connected", "protocol": "Call"})
        event_type, event = parse_twilio_message(message)
        
        assert event_type == TwilioEventType.CONNECTED
    
    def test_parse_start_event(self):
        """Test parsing start event."""
        message = json.dumps({
            "event": "start",
            "streamSid": "MZ123",
            "start": {
                "callSid": "CA456",
                "accountSid": "AC789",
                "tracks": ["inbound"],
                "customParameters": {"key": "value"},
            }
        })
        
        event_type, event = parse_twilio_message(message)
        
        assert event_type == TwilioEventType.START
        assert isinstance(event, TwilioStartEvent)
        assert event.stream_sid == "MZ123"
        assert event.call_sid == "CA456"
        assert event.account_sid == "AC789"
        assert event.tracks == ["inbound"]
        assert event.custom_parameters == {"key": "value"}
    
    def test_parse_media_event(self):
        """Test parsing media event."""
        audio_data = b"\xff" * 160
        payload_b64 = base64.b64encode(audio_data).decode()
        
        message = json.dumps({
            "event": "media",
            "streamSid": "MZ123",
            "media": {
                "track": "inbound",
                "chunk": 1,
                "timestamp": "12345",
                "payload": payload_b64,
            }
        })
        
        event_type, event = parse_twilio_message(message)
        
        assert event_type == TwilioEventType.MEDIA
        assert isinstance(event, TwilioMediaEvent)
        assert event.stream_sid == "MZ123"
        assert event.track == "inbound"
        assert event.chunk == 1
        assert event.payload == audio_data
    
    def test_parse_mark_event(self):
        """Test parsing mark event."""
        message = json.dumps({
            "event": "mark",
            "streamSid": "MZ123",
            "mark": {
                "name": "mark_1",
            }
        })
        
        event_type, event = parse_twilio_message(message)
        
        assert event_type == TwilioEventType.MARK
        assert isinstance(event, TwilioMarkEvent)
        assert event.stream_sid == "MZ123"
        assert event.name == "mark_1"
    
    def test_parse_dtmf_event(self):
        """Test parsing DTMF event."""
        message = json.dumps({
            "event": "dtmf",
            "streamSid": "MZ123",
            "dtmf": {
                "digit": "5",
            }
        })
        
        event_type, event = parse_twilio_message(message)
        
        assert event_type == TwilioEventType.DTMF
        assert isinstance(event, TwilioDTMFEvent)
        assert event.digit == "5"
    
    def test_parse_stop_event(self):
        """Test parsing stop event."""
        message = json.dumps({"event": "stop", "streamSid": "MZ123"})
        event_type, event = parse_twilio_message(message)
        
        assert event_type == TwilioEventType.STOP
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON raises error."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_twilio_message("not valid json")
    
    def test_parse_unknown_event(self):
        """Test parsing unknown event type raises error."""
        message = json.dumps({"event": "unknown_event"})
        with pytest.raises(ValueError, match="Unknown event type"):
            parse_twilio_message(message)


class TestMessageCreation:
    """Tests for creating Twilio messages."""
    
    def test_create_media_message(self):
        """Test creating media message."""
        audio_data = b"\xff" * 160
        message = create_media_message("MZ123", audio_data)
        
        parsed = json.loads(message)
        
        assert parsed["event"] == "media"
        assert parsed["streamSid"] == "MZ123"
        assert "payload" in parsed["media"]
        
        # Verify payload decodes correctly
        decoded = base64.b64decode(parsed["media"]["payload"])
        assert decoded == audio_data
    
    def test_create_mark_message(self):
        """Test creating mark message."""
        message = create_mark_message("MZ123", "mark_42")
        
        parsed = json.loads(message)
        
        assert parsed["event"] == "mark"
        assert parsed["streamSid"] == "MZ123"
        assert parsed["mark"]["name"] == "mark_42"
    
    def test_create_clear_message(self):
        """Test creating clear message."""
        message = create_clear_message("MZ123")
        
        parsed = json.loads(message)
        
        assert parsed["event"] == "clear"
        assert parsed["streamSid"] == "MZ123"


class TestCallState:
    """Tests for CallState."""
    
    def test_call_state_defaults(self):
        """Test default CallState values."""
        state = CallState()
        
        assert state.stream_sid == ""
        assert state.call_sid == ""
        assert state.sequence_number == 0
        assert state.is_active is True
    
    def test_call_state_next_sequence(self):
        """Test sequence number incrementing."""
        state = CallState()
        
        assert state.next_sequence() == 1
        assert state.next_sequence() == 2
        assert state.next_sequence() == 3


class TestProtocolHandler:
    """Tests for TwilioProtocolHandler."""
    
    def test_handler_initial_state(self):
        """Test handler starts inactive."""
        handler = TwilioProtocolHandler()
        
        assert handler.stream_sid == ""
        assert handler.call_sid == ""
        assert handler.is_active is False
    
    def test_handler_handle_start(self):
        """Test handling start event."""
        handler = TwilioProtocolHandler()
        
        event = TwilioStartEvent(
            stream_sid="MZ123",
            call_sid="CA456",
            account_sid="AC789",
            tracks=["inbound"],
        )
        
        handler.handle_start(event)
        
        assert handler.stream_sid == "MZ123"
        assert handler.call_sid == "CA456"
        assert handler.is_active is True
    
    def test_handler_handle_stop(self):
        """Test handling stop event."""
        handler = TwilioProtocolHandler()
        handler.handle_start(TwilioStartEvent(
            stream_sid="MZ123",
            call_sid="CA456",
            account_sid="AC789",
            tracks=[],
        ))
        
        handler.handle_stop()
        
        assert handler.is_active is False
    
    def test_handler_create_audio_messages(self):
        """Test creating audio messages with chunking."""
        handler = TwilioProtocolHandler()
        handler.handle_start(TwilioStartEvent(
            stream_sid="MZ123",
            call_sid="CA456",
            account_sid="AC789",
            tracks=[],
        ))
        
        # 320 bytes = 2 chunks of 160
        audio = b"\xff" * 320
        messages = handler.create_audio_messages(audio)
        
        assert len(messages) == 2
        
        for msg in messages:
            parsed = json.loads(msg)
            assert parsed["event"] == "media"
            assert parsed["streamSid"] == "MZ123"
    
    def test_handler_create_mark(self):
        """Test creating mark messages."""
        handler = TwilioProtocolHandler()
        handler.handle_start(TwilioStartEvent(
            stream_sid="MZ123",
            call_sid="CA456",
            account_sid="AC789",
            tracks=[],
        ))
        
        mark1 = handler.create_mark()
        mark2 = handler.create_mark("custom_mark")
        
        parsed1 = json.loads(mark1)
        parsed2 = json.loads(mark2)
        
        assert parsed1["mark"]["name"] == "mark_1"
        assert parsed2["mark"]["name"] == "custom_mark"
    
    def test_handler_create_clear(self):
        """Test creating clear messages."""
        handler = TwilioProtocolHandler()
        handler.handle_start(TwilioStartEvent(
            stream_sid="MZ123",
            call_sid="CA456",
            account_sid="AC789",
            tracks=[],
        ))
        
        clear = handler.create_clear()
        
        parsed = json.loads(clear)
        assert parsed["event"] == "clear"
        assert parsed["streamSid"] == "MZ123"
    
    def test_handler_no_messages_when_inactive(self):
        """Test that handler returns empty when not active."""
        handler = TwilioProtocolHandler()
        
        messages = handler.create_audio_messages(b"\xff" * 160)
        mark = handler.create_mark()
        clear = handler.create_clear()
        
        assert messages == []
        assert mark == ""
        assert clear == ""
