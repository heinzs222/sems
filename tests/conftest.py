"""
Pytest configuration and fixtures.
"""

import pytest
import os
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for tests."""
    env_vars = {
        "PUBLIC_HOST": "test.ngrok.io",
        "PORT": "7860",
        "LOG_LEVEL": "DEBUG",
        "TWILIO_ACCOUNT_SID": "ACtest123456789",
        "TWILIO_AUTH_TOKEN": "test_auth_token",
        "DEEPGRAM_API_KEY": "test_deepgram_key",
        "CARTESIA_API_KEY": "test_cartesia_key",
        "GROQ_API_KEY": "test_groq_key",
        "GROQ_MODEL": "llama-3.3-70b-versatile",
        "ROUTER_ENABLED": "false",  # Disable for most tests
        "OUTLINES_ENABLED": "false",
    }
    
    with patch.dict(os.environ, env_vars):
        # Clear config cache
        from src.agent.config import get_config
        get_config.cache_clear()
        yield


@pytest.fixture
def sample_ulaw_audio():
    """Generate sample mu-law audio (silence)."""
    return b"\xff" * 160  # 20ms of silence


@pytest.fixture
def sample_pcm_audio():
    """Generate sample PCM audio (silence)."""
    return b"\x00\x00" * 160  # 20ms of silence at 8kHz


@pytest.fixture
def twilio_start_message():
    """Sample Twilio start message."""
    import json
    return json.dumps({
        "event": "start",
        "streamSid": "MZ123456",
        "start": {
            "callSid": "CA789012",
            "accountSid": "AC345678",
            "tracks": ["inbound"],
            "customParameters": {},
        }
    })


@pytest.fixture
def twilio_media_message(sample_ulaw_audio):
    """Sample Twilio media message."""
    import json
    import base64
    
    return json.dumps({
        "event": "media",
        "streamSid": "MZ123456",
        "media": {
            "track": "inbound",
            "chunk": 1,
            "timestamp": "12345",
            "payload": base64.b64encode(sample_ulaw_audio).decode(),
        }
    })


@pytest.fixture
def twilio_stop_message():
    """Sample Twilio stop message."""
    import json
    return json.dumps({
        "event": "stop",
        "streamSid": "MZ123456",
    })
