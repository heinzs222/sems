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
    deepgram_eager_eot_threshold: float = 0.45
    deepgram_eot_threshold: float = 0.72
    deepgram_eot_timeout_ms: int = 4500
    
    # TTS Provider
    # - "cartesia": Cartesia API (default, low latency, CPU-friendly)
    # - "openai": OpenAI Audio Speech API (non-streaming in this repo)
    # - "csm": Contextual TTS via a GPU microservice (see csm_service/)
    tts_provider: str = "cartesia"  # "cartesia" | "openai" | "csm"
    
    # Cartesia (TTS)
    cartesia_api_key: str = ""
    cartesia_voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091"
    cartesia_voice_id_fr: str = "dd951538-c475-4bde-a3f7-9fd7b3e4d8f5"

    # OpenAI (TTS)
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"

    # CSM Microservice (contextual TTS)
    csm_endpoint: str = ""
    csm_timeout_ms: int = 2500
    csm_max_context_seconds: float = 24.0
    csm_voice_style: str = "default"
    
    # Groq (LLM)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # LLM Provider (Groq/OpenAI)
    # - Default is Groq for backwards compatibility.
    # - Set LLM_PROVIDER=openai + OPENAI_API_KEY/OPENAI_MODEL to use ChatGPT.
    llm_provider: str = "groq"  # "groq" | "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    
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

    # Turn-taking (avoid cutting callers off during short thinking pauses)
    # - grace: base hold after end-of-utterance before we act on a "final" transcript
    # - fallback: hold used when Deepgram emits is_final without speech_final
    # - short_utterance: extra patience for very short fragments (often mid-thought)
    # - incomplete: extra patience when the last token looks unfinished ("and", "mais", "...")
    turn_end_grace_ms: int = 650
    turn_end_fallback_ms: int = 950
    turn_end_short_utterance_ms: int = 1200
    turn_end_incomplete_ms: int = 1500

    # Silence check-ins (human-style, non-spammy)
    # - first: seconds after the agent finishes speaking with no user speech
    # - next: seconds between subsequent check-ins (if still silent)
    # - max_prompts: maximum check-ins before going quiet again until user speaks
    silence_checkin_first_s: float = 12.0
    silence_checkin_next_s: float = 25.0
    silence_checkin_max_prompts: int = 2
    min_interruption_words: int = 3

    # Audio pacing / jitter buffer (Twilio PSTN is jittery; this smooths outbound playback)
    # - jitter_buffer_ms: initial target prebuffer before starting an audio burst
    # - adaptive tuning will raise/lower within [min,max] based on underflows
    jitter_buffer_ms: int = 160
    jitter_buffer_min_ms: int = 80
    jitter_buffer_max_ms: int = 300
    jitter_buffer_adapt_step_ms: int = 20
    jitter_buffer_idle_threshold_ms: int = 250
    
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
        tts = (self.tts_provider or "cartesia").strip().lower()
        if tts not in ("cartesia", "openai", "csm"):
            raise ConfigError(
                f"Invalid TTS_PROVIDER '{self.tts_provider}'. Expected 'cartesia', 'openai', or 'csm'."
            )
        if tts in ("cartesia", "csm") and not self.cartesia_api_key:
            missing.append("CARTESIA_API_KEY")
        if tts == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if tts == "csm" and not self.csm_endpoint:
            missing.append("CSM_ENDPOINT")
        provider = (self.llm_provider or "groq").strip().lower()
        if provider not in ("groq", "openai"):
            raise ConfigError(
                f"Invalid LLM_PROVIDER '{self.llm_provider}'. Expected 'groq' or 'openai'."
            )

        if provider == "groq":
            if not self.groq_api_key:
                missing.append("GROQ_API_KEY")
            if not self.groq_model:
                missing.append("GROQ_MODEL")

        if provider == "openai":
            if not self.openai_api_key:
                missing.append("OPENAI_API_KEY")
            if not self.openai_model:
                missing.append("OPENAI_MODEL")
        
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
            deepgram_eager_eot_threshold=self.deepgram_eager_eot_threshold,
            deepgram_eot_threshold=self.deepgram_eot_threshold,
            deepgram_eot_timeout_ms=self.deepgram_eot_timeout_ms,
            tts_provider=self.tts_provider,
            llm_provider=self.llm_provider,
            llm_model=self.openai_model if self.llm_provider == "openai" else self.groq_model,
            router_enabled=self.router_enabled,
            menu_only=self.menu_only,
            outlines_enabled=self.outlines_enabled,
            agent_name=self.agent_name,
            min_interruption_words=self.min_interruption_words,
            turn_end_grace_ms=self.turn_end_grace_ms,
            turn_end_fallback_ms=self.turn_end_fallback_ms,
            turn_end_short_utterance_ms=self.turn_end_short_utterance_ms,
            turn_end_incomplete_ms=self.turn_end_incomplete_ms,
            silence_checkin_first_s=self.silence_checkin_first_s,
            silence_checkin_next_s=self.silence_checkin_next_s,
            silence_checkin_max_prompts=self.silence_checkin_max_prompts,
            jitter_buffer_ms=self.jitter_buffer_ms,
            jitter_buffer_min_ms=self.jitter_buffer_min_ms,
            jitter_buffer_max_ms=self.jitter_buffer_max_ms,
            jitter_buffer_adapt_step_ms=self.jitter_buffer_adapt_step_ms,
            jitter_buffer_idle_threshold_ms=self.jitter_buffer_idle_threshold_ms,
            csm_endpoint_set=bool(self.csm_endpoint),
            csm_timeout_ms=self.csm_timeout_ms,
            csm_max_context_seconds=self.csm_max_context_seconds,
            csm_voice_style=self.csm_voice_style,
            twilio_sid_prefix=self.twilio_account_sid[:6] + "..." if self.twilio_account_sid else "NOT SET",
            deepgram_key_set=bool(self.deepgram_api_key),
            cartesia_key_set=bool(self.cartesia_api_key),
            groq_key_set=bool(self.groq_api_key),
            openai_key_set=bool(self.openai_api_key),
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
        deepgram_eager_eot_threshold=_get_float("DEEPGRAM_EAGER_EOT_THRESHOLD", 0.45),
        deepgram_eot_threshold=_get_float("DEEPGRAM_EOT_THRESHOLD", 0.72),
        deepgram_eot_timeout_ms=_get_int("DEEPGRAM_EOT_TIMEOUT_MS", 4500),
        
        # TTS Provider
        tts_provider=os.getenv("TTS_PROVIDER", "cartesia").strip().lower(),
        
        # Cartesia
        cartesia_api_key=os.getenv("CARTESIA_API_KEY", ""),
        cartesia_voice_id=os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091"),
        cartesia_voice_id_fr=os.getenv("CARTESIA_VOICE_ID_FR", "dd951538-c475-4bde-a3f7-9fd7b3e4d8f5"),

        # OpenAI (TTS)
        openai_tts_model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        openai_tts_voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),

        # CSM Microservice (contextual TTS)
        csm_endpoint=os.getenv("CSM_ENDPOINT", "").strip(),
        csm_timeout_ms=_get_int("CSM_TIMEOUT_MS", 2500),
        csm_max_context_seconds=_get_float("CSM_MAX_CONTEXT_SECONDS", 24.0),
        csm_voice_style=os.getenv("CSM_VOICE_STYLE", "default").strip(),
        
        # Groq
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),

        # LLM Provider
        llm_provider=os.getenv("LLM_PROVIDER", "groq").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        
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
        turn_end_grace_ms=_get_int("TURN_END_GRACE_MS", 650),
        turn_end_fallback_ms=_get_int("TURN_END_FALLBACK_MS", 950),
        turn_end_short_utterance_ms=_get_int("TURN_END_SHORT_UTTERANCE_MS", 1200),
        turn_end_incomplete_ms=_get_int("TURN_END_INCOMPLETE_MS", 1500),
        silence_checkin_first_s=_get_float("SILENCE_CHECKIN_FIRST_S", 12.0),
        silence_checkin_next_s=_get_float("SILENCE_CHECKIN_NEXT_S", 25.0),
        silence_checkin_max_prompts=_get_int("SILENCE_CHECKIN_MAX_PROMPTS", 2),
        min_interruption_words=_get_int("MIN_INTERRUPTION_WORDS", 3),

        # Audio pacing / jitter buffer
        jitter_buffer_ms=_get_int("JITTER_BUFFER_MS", 160),
        jitter_buffer_min_ms=_get_int("JITTER_BUFFER_MIN_MS", 80),
        jitter_buffer_max_ms=_get_int("JITTER_BUFFER_MAX_MS", 300),
        jitter_buffer_adapt_step_ms=_get_int("JITTER_BUFFER_ADAPT_STEP_MS", 20),
        jitter_buffer_idle_threshold_ms=_get_int("JITTER_BUFFER_IDLE_THRESHOLD_MS", 250),
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
