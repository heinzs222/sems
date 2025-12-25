"""
Semantic routing for fast cached responses.

Uses semantic-router with fastembed to match user intents to predefined routes.
When a route matches, we can skip the LLM and play a pre-rendered audio response.

Routes:
- pricing: Questions about pricing/cost
- who_are_you: Questions about the agent's identity
- stop: Request to end the call
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import structlog

from src.agent.config import get_config
from src.agent.audio import chunk_audio_list, TWILIO_FRAME_SIZE

logger = structlog.get_logger(__name__)

# Lazy imports to avoid loading models at import time
_router = None
_encoder = None


def _get_router():
    """Lazy load the semantic router."""
    global _router, _encoder
    
    if _router is not None:
        return _router
    
    config = get_config()
    if not config.router_enabled:
        logger.info("Router disabled by configuration")
        return None
    
    try:
        from semantic_router import Route, RouteLayer
        from semantic_router.encoders import FastEmbedEncoder
    except ImportError:
        logger.warning("semantic-router not installed, routing disabled")
        return None
    
    try:
        
        logger.info("Initializing semantic router...")
        
        # Initialize encoder once
        _encoder = FastEmbedEncoder()
        
        # Define routes with example utterances
        pricing_route = Route(
            name="pricing",
            utterances=[
                "how much does it cost",
                "what's the price",
                "what are your rates",
                "pricing information",
                "how much do you charge",
                "what's the cost",
                "tell me about pricing",
                "what are the fees",
                "is it expensive",
                "what's your pricing model",
                "how much is it",
                "cost of service",
                "price list",
                "subscription cost",
                "monthly fee",
            ],
        )
        
        who_are_you_route = Route(
            name="who_are_you",
            utterances=[
                "who are you",
                "what are you",
                "what's your name",
                "tell me about yourself",
                "introduce yourself",
                "are you a robot",
                "are you an AI",
                "are you human",
                "what can you do",
                "who am I speaking with",
                "what is this",
                "who is this",
                "what company is this",
            ],
        )
        
        stop_route = Route(
            name="stop",
            utterances=[
                "goodbye",
                "bye",
                "end the call",
                "hang up",
                "stop",
                "that's all",
                "I'm done",
                "end call",
                "disconnect",
                "thanks bye",
                "thank you goodbye",
                "I have to go",
                "gotta go",
                "talk to you later",
                "see you",
            ],
        )
        
        # Create route layer
        _router = RouteLayer(
            encoder=_encoder,
            routes=[pricing_route, who_are_you_route, stop_route],
        )
        
        logger.info("Semantic router initialized", num_routes=3)
        return _router
        
    except Exception as e:
        logger.error("Failed to initialize semantic router", error=str(e))
        return None


class CachedAudioManager:
    """Manages pre-rendered audio files for cached responses."""
    
    def __init__(self, audio_dir: Optional[str] = None):
        if audio_dir is None:
            # Default to assets/audio in project root
            project_root = Path(__file__).parent.parent.parent
            audio_dir = project_root / "assets" / "audio"
        
        self.audio_dir = Path(audio_dir)
        self._cache: Dict[str, bytes] = {}
        self._chunks_cache: Dict[str, List[bytes]] = {}
    
    def _get_audio_path(self, route_name: str) -> Path:
        """Get the path to a cached audio file."""
        return self.audio_dir / f"{route_name}.mulaw"
    
    def load_audio(self, route_name: str) -> Optional[bytes]:
        """
        Load cached audio for a route.
        
        Args:
            route_name: Name of the route (e.g., "pricing")
            
        Returns:
            Raw mu-law audio bytes, or None if not found
        """
        if route_name in self._cache:
            return self._cache[route_name]
        
        audio_path = self._get_audio_path(route_name)
        
        if not audio_path.exists():
            logger.warning(
                "Cached audio not found",
                route_name=route_name,
                expected_path=str(audio_path),
            )
            return None
        
        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            
            self._cache[route_name] = audio_data
            logger.debug(
                "Loaded cached audio",
                route_name=route_name,
                size_bytes=len(audio_data),
            )
            return audio_data
            
        except Exception as e:
            logger.error(
                "Failed to load cached audio",
                route_name=route_name,
                error=str(e),
            )
            return None
    
    def get_audio_chunks(self, route_name: str) -> List[bytes]:
        """
        Get cached audio as chunks ready for Twilio.
        
        Args:
            route_name: Name of the route
            
        Returns:
            List of 160-byte chunks (20ms each)
        """
        if route_name in self._chunks_cache:
            return self._chunks_cache[route_name]
        
        audio_data = self.load_audio(route_name)
        if audio_data is None:
            return []
        
        chunks = chunk_audio_list(audio_data, TWILIO_FRAME_SIZE)
        self._chunks_cache[route_name] = chunks
        
        logger.debug(
            "Chunked cached audio",
            route_name=route_name,
            num_chunks=len(chunks),
        )
        
        return chunks
    
    def has_audio(self, route_name: str) -> bool:
        """Check if cached audio exists for a route."""
        if route_name in self._cache:
            return True
        return self._get_audio_path(route_name).exists()


class SemanticRouter:
    """
    High-level semantic routing interface.
    
    Combines route detection with cached audio playback.
    """
    
    def __init__(self):
        self.audio_manager = CachedAudioManager()
        self._router = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the router.
        
        Returns:
            True if initialized successfully
        """
        if self._initialized:
            return self._router is not None
        
        self._router = _get_router()
        self._initialized = True
        return self._router is not None
    
    @property
    def is_enabled(self) -> bool:
        """Check if routing is enabled and initialized."""
        return self._router is not None
    
    def detect_route(self, text: str) -> Optional[str]:
        """
        Detect which route (if any) matches the input text.
        
        Args:
            text: User's transcribed speech
            
        Returns:
            Route name if matched, None otherwise
        """
        if not self._router:
            return None
        
        if not text or len(text.strip()) < 2:
            return None
        
        try:
            result = self._router(text)
            
            if result.name:
                logger.info(
                    "Route detected",
                    text=text[:50],
                    route=result.name,
                )
                return result.name
            
            return None
            
        except Exception as e:
            logger.error("Route detection failed", error=str(e))
            return None
    
    def get_cached_response(self, route_name: str) -> Tuple[Optional[List[bytes]], bool]:
        """
        Get cached audio response for a route.
        
        Args:
            route_name: Name of the matched route
            
        Returns:
            Tuple of (audio_chunks, should_hangup)
            - audio_chunks: List of audio chunks to send, or None if not available
            - should_hangup: Whether to hang up after playing (for "stop" route)
        """
        should_hangup = route_name == "stop"
        
        if not self.audio_manager.has_audio(route_name):
            logger.warning(
                "No cached audio for route",
                route_name=route_name,
            )
            return None, should_hangup
        
        chunks = self.audio_manager.get_audio_chunks(route_name)
        
        if not chunks:
            return None, should_hangup
        
        return chunks, should_hangup
    
    def try_route(self, text: str) -> Tuple[Optional[str], Optional[List[bytes]], bool]:
        """
        Try to route the input and get cached response.
        
        This is a convenience method that combines detection and response retrieval.
        
        Args:
            text: User's transcribed speech
            
        Returns:
            Tuple of (route_name, audio_chunks, should_hangup)
            - route_name: Matched route or None
            - audio_chunks: Audio to play or None
            - should_hangup: Whether to hang up after
        """
        route_name = self.detect_route(text)
        
        if route_name is None:
            return None, None, False
        
        chunks, should_hangup = self.get_cached_response(route_name)
        
        return route_name, chunks, should_hangup


# Singleton instance
_semantic_router: Optional[SemanticRouter] = None


def get_semantic_router() -> SemanticRouter:
    """Get or create the semantic router singleton."""
    global _semantic_router
    
    if _semantic_router is None:
        _semantic_router = SemanticRouter()
    
    return _semantic_router


def initialize_router() -> bool:
    """
    Initialize the semantic router at application startup.
    
    Returns:
        True if router is enabled and initialized
    """
    router = get_semantic_router()
    return router.initialize()
