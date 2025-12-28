"""
Configuration management for the Twilio Voice Agent.

Loads environment variables and provides a strongly-typed configuration object.
Validates required keys at startup.
"""

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
import structlog

load_dotenv()

logger = structlog.get_logger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


@dataclass(frozen=True)
class Config:
    """Strongly-typed configuration object."""
    
    # Server
    public_host: str
    port: int = 7860
    log_level: str = "INFO"
    
    # Language
    # - default_language controls the agent's spoken language mode at call start ("en" or "fr")
    # - deepgram_language_* configure STT language codes passed to Deepgram
    default_language: str = "en"
    deepgram_language_en: str = "en-US"
    deepgram_language_fr: str = "fr"
    
    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    
    # Deepgram (STT)
    deepgram_api_key: str = ""
    
    # Cartesia (TTS)
    cartesia_api_key: str = ""
    cartesia_voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"
    cartesia_voice_id_fr: str = "dd951538-c475-4bde-a3f7-9fd7b3e4d8f5"
    
    # Groq (LLM)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    
    # Feature flags
    router_enabled: bool = True
    menu_only: bool = True
    memory_enabled: bool = False
    outlines_enabled: bool = False
    
    # Agent settings
    agent_name: str = "Sesame"
    company_name: str = "Sesame AI"
    max_history_turns: int = 10
    silence_timeout_seconds: float = 1.5
    min_interruption_words: int = 3
    
    @property
    def ws_url(self) -> str:
        """Get the WebSocket URL for Twilio."""
        return f"wss://{self.public_host}/ws"
    
    @property
    def base_url(self) -> str:
        """Get the base HTTP URL."""
        return f"https://{self.public_host}"
    
    def validate(self) -> None:
        """Validate that all required configuration is present."""
        missing = []
        
        if not self.public_host:
            missing.append("PUBLIC_HOST")
        if not self.twilio_account_sid:
            missing.append("TWILIO_ACCOUNT_SID")
        if not self.twilio_auth_token:
            missing.append("TWILIO_AUTH_TOKEN")
        if not self.deepgram_api_key:
            missing.append("DEEPGRAM_API_KEY")
        if not self.cartesia_api_key:
            missing.append("CARTESIA_API_KEY")
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.groq_model:
            missing.append("GROQ_MODEL")
        
        if missing:
            raise ConfigError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Please check your .env file."
            )
    
    def log_config(self) -> None:
        """Log configuration (without secrets)."""
        logger.info(
            "Configuration loaded",
            public_host=self.public_host,
            port=self.port,
            log_level=self.log_level,
            default_language=self.default_language,
            deepgram_language_en=self.deepgram_language_en,
            deepgram_language_fr=self.deepgram_language_fr,
            groq_model=self.groq_model,
            router_enabled=self.router_enabled,
            menu_only=self.menu_only,
            outlines_enabled=self.outlines_enabled,
            agent_name=self.agent_name,
            min_interruption_words=self.min_interruption_words,
            twilio_sid_prefix=self.twilio_account_sid[:6] + "..." if self.twilio_account_sid else "NOT SET",
            deepgram_key_set=bool(self.deepgram_api_key),
            cartesia_key_set=bool(self.cartesia_api_key),
            groq_key_set=bool(self.groq_api_key),
        )


def _get_bool(key: str, default: bool = False) -> bool:
    """Get a boolean from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def _get_int(key: str, default: int) -> int:
    """Get an integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    """Get a float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Get the application configuration.
    
    Uses lru_cache to ensure we only load config once.
    """
    default_language_raw = os.getenv("DEFAULT_LANGUAGE", "en").strip().lower()
    default_language = "fr" if default_language_raw.startswith("fr") else "en"
    
    config = Config(
        # Server
        public_host=os.getenv("PUBLIC_HOST", ""),
        port=_get_int("PORT", 7860),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        
        # Language
        default_language=default_language,
        deepgram_language_en=os.getenv("DEEPGRAM_LANGUAGE_EN", "en-US"),
        deepgram_language_fr=os.getenv("DEEPGRAM_LANGUAGE_FR", "fr"),
        
        # Twilio
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        
        # Deepgram
        deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", ""),
        
        # Cartesia
        cartesia_api_key=os.getenv("CARTESIA_API_KEY", ""),
        cartesia_voice_id=os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091"),
        cartesia_voice_id_fr=os.getenv("CARTESIA_VOICE_ID_FR", "dd951538-c475-4bde-a3f7-9fd7b3e4d8f5"),
        
        # Groq
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        
        # Feature flags
        router_enabled=_get_bool("ROUTER_ENABLED", True),
        menu_only=_get_bool("MENU_ONLY", True),
        memory_enabled=_get_bool("MEMORY_ENABLED", False),
        outlines_enabled=_get_bool("OUTLINES_ENABLED", False),
        
        # Agent settings
        agent_name=os.getenv("AGENT_NAME", "Sesame"),
        company_name=os.getenv("COMPANY_NAME", "Sesame AI"),
        max_history_turns=_get_int("MAX_HISTORY_TURNS", 10),
        silence_timeout_seconds=_get_float("SILENCE_TIMEOUT_SECONDS", 1.5),
        min_interruption_words=_get_int("MIN_INTERRUPTION_WORDS", 3),
    )
    
    return config


def init_config() -> Config:
    """
    Initialize and validate configuration.
    
    Call this at application startup to fail fast if config is invalid.
    """
    config = get_config()
    config.validate()
    config.log_config()
    return config
