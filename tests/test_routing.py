"""
Tests for semantic routing.
"""

import pytest
import os
from unittest.mock import patch, MagicMock


class TestSemanticRouter:
    """Tests for semantic router (without loading models)."""
    
    def test_router_disabled_returns_none(self):
        """Test that disabled router returns None for all inputs."""
        with patch.dict(os.environ, {"ROUTER_ENABLED": "false"}):
            # Clear cached config
            from src.agent.config import get_config
            get_config.cache_clear()
            
            from src.agent.routing import SemanticRouter
            
            router = SemanticRouter()
            router._initialized = True
            router._router = None  # Simulate disabled
            
            assert router.detect_route("what is the price") is None
            assert router.is_enabled is False
    
    def test_router_empty_text_returns_none(self):
        """Test that empty text returns None."""
        from src.agent.routing import SemanticRouter
        
        router = SemanticRouter()
        router._initialized = True
        router._router = MagicMock()  # Mock the router
        
        assert router.detect_route("") is None
        assert router.detect_route("  ") is None
        assert router.detect_route("a") is None  # Too short


class TestCachedAudioManager:
    """Tests for cached audio manager."""
    
    def test_audio_path_generation(self):
        """Test that audio paths are generated correctly."""
        from src.agent.routing import CachedAudioManager
        
        manager = CachedAudioManager("/test/audio")
        
        path = manager._get_audio_path("pricing")
        assert str(path).endswith("pricing.mulaw")
    
    def test_load_nonexistent_audio_returns_none(self):
        """Test loading non-existent file returns None."""
        from src.agent.routing import CachedAudioManager
        
        manager = CachedAudioManager("/nonexistent/path")
        
        result = manager.load_audio("pricing")
        assert result is None
    
    def test_has_audio_false_for_missing(self):
        """Test has_audio returns False for missing files."""
        from src.agent.routing import CachedAudioManager
        
        manager = CachedAudioManager("/nonexistent/path")
        
        assert manager.has_audio("pricing") is False
    
    def test_caching_behavior(self):
        """Test that audio is cached after first load."""
        from src.agent.routing import CachedAudioManager
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = os.path.join(tmpdir, "test.mulaw")
            with open(test_file, "wb") as f:
                f.write(b"\xff" * 160)
            
            manager = CachedAudioManager(tmpdir)
            
            # First load
            audio1 = manager.load_audio("test")
            
            # Delete the file
            os.remove(test_file)
            
            # Should still return cached version
            audio2 = manager.load_audio("test")
            
            assert audio1 == audio2
            assert audio1 == b"\xff" * 160


class TestRouteDetection:
    """Integration tests for route detection (requires model loading)."""
    
    @pytest.mark.skipif(
        os.environ.get("SKIP_MODEL_TESTS", "true").lower() == "true",
        reason="Skipping model-dependent tests"
    )
    def test_pricing_route_detection(self):
        """Test pricing route is detected."""
        from src.agent.routing import get_semantic_router
        
        router = get_semantic_router()
        if not router.initialize():
            pytest.skip("Router initialization failed")
        
        test_phrases = [
            "how much does it cost",
            "what's the pricing",
            "tell me about your rates",
        ]
        
        for phrase in test_phrases:
            route = router.detect_route(phrase)
            assert route == "pricing", f"Expected 'pricing' for '{phrase}', got '{route}'"
    
    @pytest.mark.skipif(
        os.environ.get("SKIP_MODEL_TESTS", "true").lower() == "true",
        reason="Skipping model-dependent tests"
    )
    def test_who_are_you_route_detection(self):
        """Test who_are_you route is detected."""
        from src.agent.routing import get_semantic_router
        
        router = get_semantic_router()
        if not router.initialize():
            pytest.skip("Router initialization failed")
        
        test_phrases = [
            "who are you",
            "what is your name",
            "are you a robot",
        ]
        
        for phrase in test_phrases:
            route = router.detect_route(phrase)
            assert route == "who_are_you", f"Expected 'who_are_you' for '{phrase}', got '{route}'"
    
    @pytest.mark.skipif(
        os.environ.get("SKIP_MODEL_TESTS", "true").lower() == "true",
        reason="Skipping model-dependent tests"
    )
    def test_stop_route_detection(self):
        """Test stop route is detected."""
        from src.agent.routing import get_semantic_router
        
        router = get_semantic_router()
        if not router.initialize():
            pytest.skip("Router initialization failed")
        
        test_phrases = [
            "goodbye",
            "hang up",
            "end the call",
        ]
        
        for phrase in test_phrases:
            route = router.detect_route(phrase)
            assert route == "stop", f"Expected 'stop' for '{phrase}', got '{route}'"


class TestTryRoute:
    """Tests for the try_route convenience method."""
    
    def test_try_route_no_match(self):
        """Test try_route when no route matches."""
        from src.agent.routing import SemanticRouter
        from types import SimpleNamespace
        
        router = SemanticRouter()
        router._initialized = True
        router._router = MagicMock(return_value=SimpleNamespace(name=None))
        
        route_name, chunks, should_hangup = router.try_route("random question")
        
        assert route_name is None
        assert chunks is None
        assert should_hangup is False
    
    def test_try_route_stop_sets_hangup(self):
        """Test that stop route sets should_hangup to True."""
        from src.agent.routing import SemanticRouter
        
        router = SemanticRouter()
        router._initialized = True
        
        # Mock route detection to return "stop"
        router.detect_route = MagicMock(return_value="stop")
        router.audio_manager.has_audio = MagicMock(return_value=False)
        
        route_name, chunks, should_hangup = router.try_route("goodbye")
        
        assert route_name == "stop"
        assert should_hangup is True
