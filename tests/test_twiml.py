"""
Tests for TwiML generation.
"""

import pytest
from unittest.mock import patch
import os


class TestTwimlGeneration:
    """Tests for TwiML endpoint."""
    
    def test_twiml_contains_stream_element(self):
        """Test that TwiML contains Stream element."""
        with patch.dict(os.environ, {
            "PUBLIC_HOST": "test.ngrok.io",
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "token",
            "DEEPGRAM_API_KEY": "key",
            "CARTESIA_API_KEY": "key",
            "GROQ_API_KEY": "key",
            "GROQ_MODEL": "test-model",
        }):
            # Clear cached config
            from src.agent.config import get_config
            get_config.cache_clear()
            
            from fastapi.testclient import TestClient
            from server.app import app
            
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/twiml")
            
            assert response.status_code == 200
            assert "application/xml" in response.headers.get("content-type", "")
            
            content = response.text
            assert "<Response>" in content
            assert "<Connect>" in content
            assert "<Stream" in content
            assert "wss://test.ngrok.io/ws" in content
    
    def test_twiml_is_valid_xml(self):
        """Test that TwiML is valid XML."""
        with patch.dict(os.environ, {
            "PUBLIC_HOST": "test.ngrok.io",
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "token",
            "DEEPGRAM_API_KEY": "key",
            "CARTESIA_API_KEY": "key",
            "GROQ_API_KEY": "key",
            "GROQ_MODEL": "test-model",
        }):
            from src.agent.config import get_config
            get_config.cache_clear()
            
            from fastapi.testclient import TestClient
            from server.app import app
            import xml.etree.ElementTree as ET
            
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/twiml")
            
            # Should not raise an exception
            root = ET.fromstring(response.text)
            assert root.tag == "Response"
    
    def test_twiml_uses_correct_host(self):
        """Test that TwiML uses PUBLIC_HOST from config."""
        test_host = "my-custom-domain.example.com"
        
        with patch.dict(os.environ, {
            "PUBLIC_HOST": test_host,
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "token",
            "DEEPGRAM_API_KEY": "key",
            "CARTESIA_API_KEY": "key",
            "GROQ_API_KEY": "key",
            "GROQ_MODEL": "test-model",
        }):
            from src.agent.config import get_config
            get_config.cache_clear()
            
            from fastapi.testclient import TestClient
            from server.app import app
            
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/twiml")
            
            assert f"wss://{test_host}/ws" in response.text


class TestHealthEndpoint:
    """Tests for health endpoint."""
    
    def test_health_returns_ok(self):
        """Test health endpoint returns healthy status."""
        with patch.dict(os.environ, {
            "PUBLIC_HOST": "test.ngrok.io",
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "token",
            "DEEPGRAM_API_KEY": "key",
            "CARTESIA_API_KEY": "key",
            "GROQ_API_KEY": "key",
            "GROQ_MODEL": "test-model",
        }):
            from src.agent.config import get_config
            get_config.cache_clear()
            
            from fastapi.testclient import TestClient
            from server.app import app
            
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/health")
            
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""
    
    def test_metrics_returns_json(self):
        """Test metrics endpoint returns JSON."""
        with patch.dict(os.environ, {
            "PUBLIC_HOST": "test.ngrok.io",
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "token",
            "DEEPGRAM_API_KEY": "key",
            "CARTESIA_API_KEY": "key",
            "GROQ_API_KEY": "key",
            "GROQ_MODEL": "test-model",
        }):
            from src.agent.config import get_config
            get_config.cache_clear()
            
            from fastapi.testclient import TestClient
            from server.app import app
            
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/metrics")
            
            assert response.status_code == 200
            
            data = response.json()
            assert "uptime_seconds" in data
            assert "total_connections" in data
            assert "active_connections" in data
            assert "total_calls" in data
            assert "active_calls" in data
            assert "errors" in data
