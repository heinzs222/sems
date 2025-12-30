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
import re
import audioop
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, List, Dict, Any
from enum import Enum
import time
import unicodedata

import structlog
from twilio.rest import Client as TwilioClient

from src.agent.config import get_config, Config
from src.agent.audio import twilio_ulaw_passthrough, chunk_audio, TWILIO_FRAME_SIZE, TWILIO_SAMPLE_RATE
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
from src.agent.llm import GroqLLM, create_llm
from src.agent.routing import get_semantic_router, SemanticRouter
from src.agent.extract import ExtractionQueue
from src.agent.language import (
    detect_language_override,
    detect_language_switch_command,
    infer_language_from_text,
    normalize_detected_language,
    LangState,
    LanguageCode,
)
from src.agent.menu import (
    get_menu_catalog,
    looks_like_menu_request,
    looks_like_full_menu_request,
    find_menu_items,
    find_menu_sections,
)
from src.agent.voice_context import VoiceContextBuffer

logger = structlog.get_logger(__name__)


class PipelineState(str, Enum):
    """Current state of the voice pipeline."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


class CheckoutPhase(str, Enum):
    """High-level, menu-only checkout flow state."""

    ORDERING = "ordering"
    NAME = "name"
    NAME_CONFIRM = "name_confirm"
    NAME_SPELL = "name_spell"
    ADDRESS = "address"
    ADDRESS_CONFIRM = "address_confirm"
    PHONE = "phone"
    PHONE_CONFIRM = "phone_confirm"
    EMAIL = "email"
    EMAIL_CONFIRM = "email_confirm"
    COMPLETE = "complete"


class ToneMode(str, Enum):
    GREETING = "greeting"
    MENU_HELP = "menu_help"
    CHECKOUT_CAPTURE = "checkout_capture"
    REPAIR = "repair"


def _normalize_for_intent(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("’", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


_YES_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"(^|\b)(yes|yeah|yep|correct|thats right|that's right|right|ok|okay|sure)(\b|$)",
        r"(^|\b)(oui|ouais|d'accord|daccord|exact|c'est ca|cest ca)(\b|$)",
    )
)

_NO_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"(^|\b)(no|nope|nah|wrong|not correct)(\b|$)",
        r"(^|\b)(non|pas du tout|c'est pas ca|cest pas ca)(\b|$)",
    )
)

_DONE_ORDER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\b(thats all|that's all|thats it|that's it|nothing else|im done|i'm done|done)(\b|$)",
        r"\b(c'est tout|cest tout|rien d'autre|j'ai fini|jai fini|c'est bon|cest bon)(\b|$)",
    )
)

_MODIFY_ORDER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\b(add|actually|instead|change|wait|hold on)(\b|$)",
        r"\b(ajoute|ajouter|finalement|au fait|changer|attends)(\b|$)",
    )
)

_GOODBYE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\b(bye|goodbye|hang up|end call)(\b|$)",
        r"\b(au revoir|bye bye|raccroche|raccrocher)(\b|$)",
    )
)

_HANGUP_NOW_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\b(hang up|end call)(\b|$)",
        r"\b(raccroche|raccrocher)(\b|$)",
    )
)

_CA_POSTAL_RE = re.compile(r"\b([a-z]\d[a-z])\s?(\d[a-z]\d)\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

_LOG_PHONE_RE = re.compile(
    r"(?:\+?1[\s\-.]?)?(?:\(?\d{3}\)?[\s\-.]?)\d{3}[\s\-.]?\d{4}"
)


def _redact_transcript_for_logs(text: str) -> str:
    """
    Best-effort redaction for logs (to reduce accidental PII exposure).

    This is not a compliance-grade scrubber; it masks common patterns:
    - emails -> [EMAIL]
    - phone numbers -> [PHONE-***1234]
    - Canadian postal codes -> A1A ***
    """
    if not text:
        return ""

    redacted = _EMAIL_RE.sub("[EMAIL]", text)

    def _mask_phone(match: re.Match[str]) -> str:
        digits = re.sub(r"\D+", "", match.group(0) or "")
        last4 = digits[-4:] if len(digits) >= 4 else digits
        return f"[PHONE-***{last4}]"

    redacted = _LOG_PHONE_RE.sub(_mask_phone, redacted)
    redacted = _CA_POSTAL_RE.sub(lambda m: f"{m.group(1).upper()} ***", redacted)
    return redacted

_BARGE_IN_BACKCHANNEL_RE = re.compile(
    r"^(ok|okay|yeah|yep|yup|uh huh|uh-huh|mhm|mm hmm|mm-hmm|oui|ouais|d['’]accord|daccord)([.!?])?$"
)

_BARGE_IN_HARD_PHRASE_RE = re.compile(
    r"\b(stop|wait|hold on|one second|pause|cancel|actually|listen|arrete|attends|un instant|annule|ecoute)\b"
)

_BARGE_IN_RMS_THRESHOLD = 500
_BARGE_IN_HARD_CONTINUED_MS = 150
_BARGE_IN_BACKCHANNEL_CONTINUED_MS = 400

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

@dataclass
class QueuedTranscript:
    """A final transcript queued for turn processing."""
    text: str
    stt_latency_ms: float = 0.0
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    audio_ulaw_8k: Optional[bytes] = None


@dataclass(frozen=True)
class OutboundMessage:
    """Message to send to Twilio, optionally paced as audio."""

    message: str
    pace: bool = True  # True for 20ms media frames; False for control (mark/clear/etc.)


@dataclass
class SpeculativeLLMState:
    """
    Speculative (ephemeral) LLM generation started on Flux EagerEndOfTurn.

    This is used to reduce perceived latency by beginning LLM work during a pause
    and cancelling if Flux reports TurnResumed (user keeps talking).
    """

    transcript: str
    transcript_norm: str
    target_language: str
    extra_context: str
    started_at: float

    first_token_at: Optional[float] = None
    response_text: str = ""
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None


@dataclass
class CheckoutDebounceState:
    phase: CheckoutPhase
    text: str
    detected_language: Optional[str]
    language_confidence: Optional[float]
    stt_latency_ms: float
    created_at: float


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
        self._lang_state = LangState(
            current="fr" if self.config.default_language.startswith("fr") else "en"
        )
        self._turn_language: LanguageCode = self._lang_state.current
        
        # Components
        self._protocol = TwilioProtocolHandler()
        self._stt = STTManager()
        self._tts = TTSManager()
        self._llm: Optional[GroqLLM] = None
        self._router: Optional[SemanticRouter] = None
        self._extraction_queue: Optional[ExtractionQueue] = None
        self._menu_catalog = None

        # Contextual TTS history (used when TTS_PROVIDER=csm).
        self._voice_context = VoiceContextBuffer(max_context_seconds=self.config.csm_max_context_seconds)

        # Rolling inbound user audio (8kHz mu-law) for short context snippets.
        self._inbound_ulaw_history: bytearray = bytearray()
        self._inbound_ulaw_history_max_bytes: int = int(TWILIO_SAMPLE_RATE * 8)  # ~8 seconds
        
        # State
        self._state = PipelineState.IDLE
        self._is_running = False
        self._current_turn = 0
        self._current_turn_metrics: Optional[TurnMetrics] = None
        self._call_metrics = CallMetrics()

        # Menu-only checkout flow (order -> name -> address -> phone -> email -> confirm)
        self._checkout_phase: CheckoutPhase = CheckoutPhase.ORDERING
        self._customer_name: str = ""
        self._customer_address: str = ""
        self._customer_phone: str = ""
        self._customer_email: str = ""
        self._checkout_debounce: Optional[CheckoutDebounceState] = None
        self._checkout_debounce_task: Optional[asyncio.Task] = None

        # Track the last time we received any non-empty STT text (including interim results).
        # Used to avoid talking over the caller when Deepgram endpointing emits early finals.
        self._last_stt_text_time: float = 0.0
        self._last_interim_transcript: str = ""

        # Flux eager turn-taking: speculative LLM generation during pauses.
        self._speculative_llm: Optional[SpeculativeLLMState] = None
        
        # TTS state
        self._pending_transcript: str = ""
        
        # Turn processing (keep STT receive loop non-blocking)
        self._turn_queue: asyncio.Queue[QueuedTranscript] = asyncio.Queue()
        self._turn_worker_task: Optional[asyncio.Task] = None
        self._turn_task: Optional[asyncio.Task] = None
        self._stt_start_task: Optional[asyncio.Task] = None
        
        # Output task (e.g., greeting) and barge-in gating
        self._output_task: Optional[asyncio.Task] = None
        self._speech_started_during_tts: bool = False
        self._tts_soft_paused: bool = False
        self._barge_in_soft_started_at: float = 0.0
        self._barge_in_consecutive_speech_ms: int = 0
        self._barge_in_last_rms: int = 0
        self._barge_in_partial_text: str = ""
        self._barge_in_partial_words: int = 0
        self._barge_in_is_backchannel: bool = False
        self._soft_interrupt_watchdog_task: Optional[asyncio.Task] = None
        
        # Outbound audio pacer (Late-2025 best practice)
        self._outbound_queue: asyncio.Queue = asyncio.Queue()
        self._pacer_task: Optional[asyncio.Task] = None
        self._pacer_running: bool = False
        self._pacer_underruns: int = 0
        self._pacer_late_resets: int = 0
        self._queue_depth_max: int = 0
        self._jitter_buffer_base_ms: int = int(getattr(config, "jitter_buffer_ms", 160))
        self._jitter_buffer_ms: int = self._jitter_buffer_base_ms
        self._jitter_buffer_min_ms: int = int(getattr(config, "jitter_buffer_min_ms", 80))
        self._jitter_buffer_max_ms: int = int(getattr(config, "jitter_buffer_max_ms", 300))
        self._jitter_buffer_step_ms: int = int(getattr(config, "jitter_buffer_adapt_step_ms", 20))
        self._jitter_buffer_idle_threshold_ms: int = int(
            getattr(config, "jitter_buffer_idle_threshold_ms", 250)
        )
        self._jitter_buffer_stable_bursts: int = 0
        self._outbound_ulaw_remainder: bytes = b""
        
        # Silence detection
        self._last_audio_frame_time: float = 0.0
        self._last_speech_activity_time: float = 0.0
        self._last_silence_prompt_time: float = 0.0
        self._silence_task: Optional[asyncio.Task] = None

        # Debug flags (kept minimal to avoid noisy logs)
        self._logged_first_media: bool = False
        self._logged_first_transcript: bool = False
        
        # Twilio client for hangup
        self._twilio_client: Optional[TwilioClient] = None
    
    def _current_cartesia_voice_id(self, *, language: Optional[LanguageCode] = None) -> str:
        lang = language or self._turn_language or self._lang_state.current
        if lang == "fr":
            return self.config.cartesia_voice_id_fr or self.config.cartesia_voice_id
        return self.config.cartesia_voice_id

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        cleaned = (text or "").strip()
        if not cleaned:
            return []

        # Simple sentence boundary split (good enough for PSTN turn-taking).
        parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        sentences = [p.strip() for p in parts if p and p.strip()]
        return sentences or [cleaned]

    @staticmethod
    def _looks_like_repair_text(text: str) -> bool:
        t = _normalize_for_intent(text)
        if not t:
            return False
        return any(
            phrase in t
            for phrase in (
                "having trouble",
                "technical issue",
                "sorry",
                "desole",
                "souci technique",
                "probleme technique",
            )
        )

    def _infer_tone_mode(self, text: str, *, requested: Optional[ToneMode]) -> ToneMode:
        if requested is not None:
            return requested

        if self._looks_like_repair_text(text):
            return ToneMode.REPAIR

        # Greeting is only for the first outbound utterance on call start.
        normalized = _normalize_for_intent(text)
        if self._current_turn == 0 and any(
            normalized.startswith(prefix)
            for prefix in ("hi", "hello", "allô", "allo", "bonjour", "bonsoir")
        ):
            return ToneMode.GREETING

        if self._checkout_phase != CheckoutPhase.ORDERING:
            return ToneMode.CHECKOUT_CAPTURE

        return ToneMode.MENU_HELP

    @staticmethod
    def _apply_tone(text: str, *, language: LanguageCode, tone_mode: ToneMode) -> str:
        t = (text or "").strip()
        if not t:
            return t

        if tone_mode == ToneMode.GREETING:
            return t

        prefix_by_mode: dict[ToneMode, dict[LanguageCode, str]] = {
            ToneMode.MENU_HELP: {"en": "Got it—", "fr": "D’accord—"},
            ToneMode.CHECKOUT_CAPTURE: {"en": "Perfect—", "fr": "Parfait—"},
            ToneMode.REPAIR: {"en": "No worries—", "fr": "Pas de souci—"},
        }

        prefix = prefix_by_mode.get(tone_mode, {}).get(language)
        if not prefix:
            return t

        # Avoid double-prefixing when the message already begins with an acknowledgement.
        normalized = _normalize_for_intent(t)
        already_prefixed = any(
            normalized.startswith(p)
            for p in (
                "got it",
                "perfect",
                "no worries",
                "daccord",
                "parfait",
                "pas de souci",
            )
        )
        if already_prefixed:
            return t

        return f"{prefix} {t}"

    @staticmethod
    def _is_barge_in_backchannel(text: str) -> bool:
        t = _normalize_for_intent(text)
        return bool(t and _BARGE_IN_BACKCHANNEL_RE.match(t))

    @staticmethod
    def _contains_hard_interrupt_phrase(text: str) -> bool:
        t = _normalize_for_intent(text)
        return bool(t and _BARGE_IN_HARD_PHRASE_RE.search(t))

    def _begin_soft_interrupt(self, *, reason: str) -> None:
        if self._state != PipelineState.SPEAKING:
            return
        if self._tts_soft_paused:
            return

        now = time.time()
        self._tts_soft_paused = True
        self._speech_started_during_tts = False
        self._barge_in_soft_started_at = now
        self._barge_in_consecutive_speech_ms = 0
        self._barge_in_last_rms = 0
        self._barge_in_partial_text = ""
        self._barge_in_partial_words = 0
        self._barge_in_is_backchannel = False

        if self._soft_interrupt_watchdog_task and not self._soft_interrupt_watchdog_task.done():
            self._soft_interrupt_watchdog_task.cancel()
        self._soft_interrupt_watchdog_task = asyncio.create_task(
            self._soft_interrupt_watchdog(started_at=now)
        )

        logger.info(
            "Barge-in soft interrupt",
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            reason=reason,
            barge_in_soft_ms=0.0,
        )

    def _clear_soft_interrupt(self, *, reason: str) -> None:
        if not self._tts_soft_paused:
            return

        self._tts_soft_paused = False
        self._barge_in_soft_started_at = 0.0
        self._barge_in_consecutive_speech_ms = 0
        self._barge_in_last_rms = 0
        self._barge_in_partial_text = ""
        self._barge_in_partial_words = 0
        self._barge_in_is_backchannel = False

        if self._soft_interrupt_watchdog_task and not self._soft_interrupt_watchdog_task.done():
            self._soft_interrupt_watchdog_task.cancel()
        self._soft_interrupt_watchdog_task = None

        logger.debug(
            "Barge-in soft interrupt cleared",
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            reason=reason,
        )

    async def _soft_interrupt_watchdog(self, *, started_at: float) -> None:
        try:
            await asyncio.sleep(0.25)
            if (
                not self._is_running
                or self._state != PipelineState.SPEAKING
                or not self._tts_soft_paused
                or self._barge_in_soft_started_at != started_at
            ):
                return

            # If we aren't currently seeing continued speech energy, resume speaking.
            if self._barge_in_consecutive_speech_ms == 0:
                self._clear_soft_interrupt(reason="watchdog")
        except asyncio.CancelledError:
            pass

    async def _maybe_hard_interrupt_from_energy(self, ulaw_bytes: bytes) -> None:
        if (
            self._state != PipelineState.SPEAKING
            or not self._tts_soft_paused
            or not ulaw_bytes
        ):
            return

        try:
            pcm = audioop.ulaw2lin(ulaw_bytes, 2)
            rms = audioop.rms(pcm, 2)
        except Exception:
            rms = 0

        self._barge_in_last_rms = rms

        if rms >= _BARGE_IN_RMS_THRESHOLD:
            self._barge_in_consecutive_speech_ms += 20
        else:
            self._barge_in_consecutive_speech_ms = 0

        threshold_ms = _BARGE_IN_HARD_CONTINUED_MS
        if self._barge_in_is_backchannel and self._barge_in_partial_words <= 1:
            threshold_ms = _BARGE_IN_BACKCHANNEL_CONTINUED_MS

        if self._barge_in_consecutive_speech_ms >= threshold_ms:
            await self._handle_hard_interrupt(reason="continued_speech")

    @staticmethod
    def _matches_any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
        t = _normalize_for_intent(text)
        if not t:
            return False
        return any(p.search(t) for p in patterns)

    def _is_affirmative(self, text: str) -> bool:
        return self._matches_any(_YES_PATTERNS, text)

    def _is_negative(self, text: str) -> bool:
        return self._matches_any(_NO_PATTERNS, text)

    def _wants_done_order(self, text: str) -> bool:
        return self._matches_any(_DONE_ORDER_PATTERNS, text)

    def _wants_modify_order(self, text: str) -> bool:
        return self._matches_any(_MODIFY_ORDER_PATTERNS, text)

    def _wants_goodbye(self, text: str) -> bool:
        return self._matches_any(_GOODBYE_PATTERNS, text)

    def _maybe_sync_checkout_phase_from_assistant(self, assistant_text: str) -> None:
        """
        If the LLM asks for checkout info (name/address/phone/email), sync our deterministic
        checkout state so the next caller utterance is handled without calling the LLM again.
        """
        if not self.config.menu_only:
            return
        if self._checkout_phase != CheckoutPhase.ORDERING:
            return

        normalized = _normalize_for_intent(assistant_text)
        if not normalized:
            return

        # English patterns
        if re.search(
            r"\b(what(?:'s| is) your name|your (?:full )?name|name should i put|put it under)\b",
            normalized,
        ):
            self._checkout_phase = CheckoutPhase.NAME
            return
        if re.search(r"\b(what(?:'s| is) your address|your address|delivery address)\b", normalized):
            self._checkout_phase = CheckoutPhase.ADDRESS
            return
        if re.search(r"\b(phone number|your phone|telephone number)\b", normalized):
            self._checkout_phase = CheckoutPhase.PHONE
            return
        if re.search(r"\b(email address|your email|e-mail address)\b", normalized):
            self._checkout_phase = CheckoutPhase.EMAIL
            return

        # French patterns (normalized is accent-stripped)
        if re.search(r"\b(quel est votre nom|votre nom)\b", normalized):
            self._checkout_phase = CheckoutPhase.NAME
            return
        if re.search(r"\b(votre adresse|adresse de livraison)\b", normalized):
            self._checkout_phase = CheckoutPhase.ADDRESS
            return
        if re.search(r"\b(numero de telephone|telephone|votre telephone)\b", normalized):
            self._checkout_phase = CheckoutPhase.PHONE
            return
        if re.search(r"\b(votre email|adresse email|courriel|votre courriel)\b", normalized):
            self._checkout_phase = CheckoutPhase.EMAIL
            return

    def _extract_name(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""

        # Strip a few common "starter" fillers first.
        t = re.sub(r"^(um+|uh+|erm+|euh+|hum+)\b[, ]*", "", t, flags=re.IGNORECASE).strip()
        if not t:
            return ""

        normalized = _normalize_for_intent(t)

        # Remove common "my name is ..." prefixes, even if the user pauses and the STT finalizes
        # the prefix alone (e.g., "my name is" without a trailing name).
        # This prevents treating the prefix itself as a name.
        patterns = (
            r"^(my name is|my name'?s|this is|it'?s|it is)\b[,:\-–—]?\s*",
            r"^(je m'appelle|j'm'appelle|mon nom est|c'?est)\b[,:\-–—]?\s*",
        )

        normalized_stripped = normalized
        for pat in patterns:
            normalized_stripped = re.sub(pat, "", normalized_stripped, flags=re.IGNORECASE).strip()

        # Prefer original casing/accents when possible by removing prefixes again on the original.
        original = t.replace("’", "'")
        for pat in patterns:
            original = re.sub(pat, "", original, flags=re.IGNORECASE).strip()

        # If the user only said the prefix ("my name is") we should treat it as missing.
        if not normalized_stripped:
            return ""

        # Guard against accidental "names" that are just the prefix.
        prefix_only = {
            "my name is",
            "my name's",
            "this is",
            "it's",
            "it is",
            "je m'appelle",
            "j'm'appelle",
            "mon nom est",
            "c'est",
        }
        if normalized_stripped.strip(" .,!?:;-") in prefix_only:
            return ""

        # If they included extra details in the same breath ("... and my address is ..."),
        # keep only the name fragment.
        cut_markers = (
            r"\b(and my address is|and my address|my address is|address is|address)\b",
            r"\b(and my phone is|and my phone|my phone is|phone number is|phone)\b",
            r"\b(and my email is|and my email|my email is|email)\b",
            r"\b(et mon adresse|mon adresse|adresse)\b",
            r"\b(et mon numero|et mon numéro|mon numero|mon numéro|telephone|téléphone)\b",
            r"\b(et mon email|et mon courriel|mon email|mon courriel|courriel|email)\b",
        )
        normalized_name = normalized_stripped
        for marker in cut_markers:
            normalized_name = re.split(marker, normalized_name, maxsplit=1, flags=re.IGNORECASE)[0].strip()

        original_name = original
        for marker in cut_markers:
            original_name = re.split(marker, original_name, maxsplit=1, flags=re.IGNORECASE)[0].strip()

        normalized_name = normalized_name.strip(" .,!?:;-")
        original_name = original_name.strip(" .,!?:;-")

        return original_name if original_name else normalized_name

    def _extract_spelled_name(self, text: str) -> str:
        """
        Extract a name when the caller spells it letter-by-letter.

        Supports formats like:
        - "J O H N space D O E"
        - "J, O, H, N, espace, D, O, E"
        - "J O H N DOE" (mixed spelled + word)
        """
        t = (text or "").strip()
        if not t:
            return ""

        normalized = _normalize_for_intent(t)
        normalized = re.sub(
            r"^(my name is|my name'?s|this is|it'?s|it is)\b[,:\-–—]?\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"^(je m'appelle|j'm'appelle|mon nom est|c'?est)\b[,:\-–—]?\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )

        normalized = re.sub(r"\b(space|espace)\b", " | ", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"[^a-z0-9|]+", " ", normalized)
        tokens = [tok for tok in normalized.split() if tok]

        words: List[str] = []
        current_letters: List[str] = []

        for tok in tokens:
            if tok == "|":
                if current_letters:
                    words.append("".join(current_letters))
                    current_letters = []
                continue

            if len(tok) == 1 and tok.isalnum():
                current_letters.append(tok)
                continue

            # If the caller says a full chunk (e.g., "DOE"), treat it as a word.
            if tok.isalpha() and len(tok) > 1:
                if current_letters:
                    words.append("".join(current_letters))
                    current_letters = []
                words.append(tok)
                continue

        if current_letters:
            words.append("".join(current_letters))

        words = [w for w in words if w]
        if not words:
            return ""

        return " ".join(w.capitalize() for w in words).strip()

    def _extract_address(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""

        # Strip a few common "starter" fillers first.
        t = re.sub(r"^(um+|uh+|erm+|euh+|hum+)\b[, ]*", "", t, flags=re.IGNORECASE).strip()
        if not t:
            return ""

        normalized = _normalize_for_intent(t)
        patterns = (
            r"^(my address is|my address|address is|address|i live at|it's|it is)\b[,:\-\"]?\s*",
            r"^(mon adresse est|mon adresse|l'adresse est|adresse|j'habite|c'?est)\b[,:\-\"]?\s*",
        )

        normalized_stripped = normalized
        for pat in patterns:
            normalized_stripped = re.sub(pat, "", normalized_stripped, flags=re.IGNORECASE).strip()

        original = t.replace("ƒ?T", "'")
        for pat in patterns:
            original = re.sub(pat, "", original, flags=re.IGNORECASE).strip()

        if not normalized_stripped:
            return ""

        prefix_only = {
            "my address is",
            "my address",
            "address is",
            "address",
            "i live at",
            "it's",
            "it is",
            "mon adresse est",
            "mon adresse",
            "l'adresse est",
            "adresse",
            "j'habite",
            "c'est",
        }
        if normalized_stripped.strip(" .,!?:;-") in prefix_only:
            return ""

        return original if original else normalized_stripped

    def _extract_phone(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""

        original = t
        original = re.sub(
            r"^(my phone is|my number is|phone number is|phone is|number is)\s+",
            "",
            original,
            flags=re.IGNORECASE,
        ).strip()
        original = re.sub(
            r"^(mon numero est|mon num\S+ro est|numero|num\S+ro|telephone|t\S+l\S+phone)\s+",
            "",
            original,
            flags=re.IGNORECASE,
        ).strip()

        digits = re.sub(r"\D+", "", original)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) < 10:
            return ""
        return digits[-10:]

    def _spell_phone_digits(self, digits: str) -> str:
        digits = re.sub(r"\D+", "", digits or "")
        return ", ".join(list(digits))

    def _extract_email(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""

        normalized = t.lower()
        normalized = (
            normalized.replace(" at ", "@")
            .replace(" arrobase ", "@")
            .replace(" dot ", ".")
            .replace(" point ", ".")
            .replace(" point final ", ".")
        )
        match = _EMAIL_RE.search(normalized)
        return match.group(0) if match else ""

    def _spell_email(self, email: str, *, language: str) -> str:
        spoken: List[str] = []
        for ch in (email or "").strip():
            if ch.isalnum():
                spoken.append(ch.upper() if ch.isalpha() else ch)
            elif ch == "@":
                spoken.append("AROBAS" if language == "fr" else "AT")
            elif ch == ".":
                spoken.append("POINT" if language == "fr" else "DOT")
            elif ch == "-":
                spoken.append("TIRET" if language == "fr" else "DASH")
            elif ch == "_":
                spoken.append("UNDERSCORE")
        return ", ".join(spoken)

    def _spell_alnum(self, value: str) -> str:
        chars: List[str] = []
        for ch in (value or "").strip():
            if ch.isalnum():
                chars.append(ch.upper() if ch.isalpha() else ch)
        return ", ".join(chars)

    def _spell_alnum_hyphen(self, value: str) -> str:
        chars: List[str] = []
        for ch in (value or "").strip():
            if ch.isalnum():
                chars.append(ch.upper() if ch.isalpha() else ch)
        return "-".join(chars)

    def _spell_postal_code(self, address: str) -> Optional[str]:
        match = _CA_POSTAL_RE.search(address or "")
        if not match:
            return None
        code = (match.group(1) + match.group(2)).replace(" ", "")
        spelled = self._spell_alnum(code)
        return spelled or None

    def _spell_first_number(self, text: str) -> Optional[str]:
        match = re.search(r"\b(\d{1,6})\b", text or "")
        if not match:
            return None
        digits = match.group(1)
        return ", ".join(list(digits))

    async def _handle_checkout_flow(self, transcript: str) -> bool:
        """
        Handle menu-only checkout steps (name/address) deterministically.

        Returns True if handled (we spoke a response and should not call the LLM).
        """
        if not self.config.menu_only:
            return False

        language = self._turn_language

        # Let callers jump back to ordering at any time.
        if self._checkout_phase != CheckoutPhase.ORDERING and self._wants_modify_order(transcript):
            self._checkout_phase = CheckoutPhase.ORDERING
            msg = (
                "Bien sûr ! Qu’est-ce que vous voulez ajouter ?"
                if language == "fr"
                else "Absolutely! What would you like to add?"
            )
            await self._speak_response(msg)
            return True

        if self._checkout_phase == CheckoutPhase.ORDERING:
            if self._wants_goodbye(transcript):
                goodbye = (
                    "Merci ! À bientôt."
                    if language == "fr"
                    else "Thanks! Bye for now."
                )
                await self._speak_response(goodbye)
                # Avoid hanging up automatically on simple goodbyes; PSTN callers often
                # prefer to hang up themselves, and false positives are costly.
                if self._matches_any(_HANGUP_NOW_PATTERNS, transcript):
                    await self._hangup_call()
                return True

            normalized = _normalize_for_intent(transcript)

            mentioned_name = bool(
                re.match(
                    r"^(my name is|this is|it's|it is)\b|^(je m'appelle|mon nom est|c'est)\b",
                    normalized,
                )
            )
            mentioned_address = bool(
                _CA_POSTAL_RE.search(transcript)
                or re.search(r"\b(address|adresse)\b", normalized)
            )

            phone = self._extract_phone(transcript)
            email = self._extract_email(transcript)

            # If the caller starts giving their details without saying "that's all",
            # treat it as checkout start so we don't bounce to the LLM (and risk failures).
            wants_checkout = (
                self._wants_done_order(transcript)
                or mentioned_name
                or mentioned_address
                or bool(phone)
                or bool(email)
            )

            if not wants_checkout:
                return False

            # Opportunistically capture any details they already said.
            if mentioned_name and not self._customer_name:
                name = self._extract_name(transcript)
                if name:
                    self._customer_name = name
            if mentioned_address and not self._customer_address:
                address = self._extract_address(transcript)
                if address:
                    self._customer_address = address
            if phone and not self._customer_phone:
                self._customer_phone = phone
            if email and not self._customer_email:
                self._customer_email = email

            # Start checkout at the next missing/confirm step (name first).
            if not self._customer_name:
                self._checkout_phase = CheckoutPhase.NAME
                prompt = (
                    "Parfait ! Pour finaliser, quel est votre nom ?"
                    if language == "fr"
                    else "Perfect! To finalize, what name should I put it under?"
                )
                await self._speak_response(prompt)
                return True

            self._checkout_phase = CheckoutPhase.NAME_CONFIRM
            confirm = (
                f"Super ! J'ai noté {self._customer_name}. C'est bien ça ?"
                if language == "fr"
                else f"Perfect—your name is {self._customer_name}, right?"
            )
            await self._speak_response(confirm)
            return True

        if self._checkout_phase == CheckoutPhase.NAME:
            name = self._extract_name(transcript)
            if not name:
                retry = (
                    "Pas de souci — quel nom dois-je utiliser pour la commande ? (Votre prénom suffit.)"
                    if language == "fr"
                    else "No worries—I didn’t catch that. What name should I use for the order? (First name is totally fine.)"
                )
                await self._speak_response(retry)
                return True

            self._customer_name = name
            confirm = (
                f"Super ! J'ai noté {name}. C'est bien ça ?"
                if language == "fr"
                else f"Perfect—your name is {name}, right?"
            )
            self._checkout_phase = CheckoutPhase.NAME_CONFIRM
            await self._speak_response(confirm)
            return True

        if self._checkout_phase == CheckoutPhase.NAME_CONFIRM:
            treat_as_affirmative = self._is_affirmative(transcript)
            treat_as_negative = self._is_negative(transcript)

            # If the caller responds with a corrected name instead of "yes/no", accept it.
            if not treat_as_affirmative and not treat_as_negative:
                candidate = self._extract_name(transcript)
                if candidate:
                    if (
                        self._customer_name
                        and candidate.casefold() == self._customer_name.casefold()
                    ):
                        treat_as_affirmative = True
                    else:
                        self._customer_name = candidate
                        confirm = (
                            f"Merci ! Donc {candidate}. C'est bien ça ?"
                            if language == "fr"
                            else f"Got it—so {candidate}. Is that right?"
                        )
                        await self._speak_response(confirm)
                        return True

            if treat_as_affirmative:
                if self._customer_address:
                    spelled_number = self._spell_first_number(self._customer_address)
                    spelled_postal = self._spell_postal_code(self._customer_address)

                    parts: List[str] = []
                    if spelled_number:
                        parts.append(
                            f"numéro {spelled_number}"
                            if language == "fr"
                            else f"number {spelled_number}"
                        )
                    if spelled_postal:
                        parts.append(
                            f"code postal {spelled_postal}"
                            if language == "fr"
                            else f"postal code {spelled_postal}"
                        )
                    spelled_bits = "; ".join(parts)

                    if language == "fr":
                        confirm = (
                            f"Parfait. J'ai déjà noté l'adresse: {self._customer_address}. Pour confirmer: {spelled_bits}. C'est exact ?"
                            if spelled_bits
                            else f"Parfait. J'ai déjà noté l'adresse: {self._customer_address}. C'est exact ?"
                        )
                    else:
                        confirm = (
                            f"Great. I already noted your address as: {self._customer_address}. Just to confirm: {spelled_bits}. Is that correct?"
                            if spelled_bits
                            else f"Great. I already noted your address as: {self._customer_address}. Is that correct?"
                        )

                    self._checkout_phase = CheckoutPhase.ADDRESS_CONFIRM
                    await self._speak_response(confirm)
                    return True

                self._checkout_phase = CheckoutPhase.ADDRESS
                ask = (
                    "Parfait. Et votre adresse, s'il vous plaît ?"
                    if language == "fr"
                    else "Great. And what's your address?"
                )
                await self._speak_response(ask)
                return True

            if treat_as_negative:
                self._checkout_phase = CheckoutPhase.NAME_SPELL
                ask = (
                    "Pas de souci. Pouvez-vous l'épeler lettre par lettre, et dire « espace » entre le prénom et le nom ?"
                    if language == "fr"
                    else "No problem. Please spell it letter by letter, and say “space” between your first and last name."
                )
                await self._speak_response(ask)
                return True

            reprompt = (
                f"Juste pour être sûre : {self._customer_name}. C'est bien ça ? (Vous pouvez dire oui, ou me donner le bon nom.)"
                if language == "fr"
                else f"Just to make sure: {self._customer_name}. Is that right? (You can say yes, or tell me the correct name.)"
            )
            await self._speak_response(reprompt)
            return True

        if self._checkout_phase == CheckoutPhase.NAME_SPELL:
            spelled_name = self._extract_spelled_name(transcript)
            if not spelled_name:
                retry = (
                    "D'accord. Épelez-le lettre par lettre, comme: J-O-H-N, espace, D-O-E."
                    if language == "fr"
                    else "Okay. Spell it letter by letter, like: J-O-H-N, space, D-O-E."
                )
                await self._speak_response(retry)
                return True

            self._customer_name = spelled_name
            parts = [p for p in re.split(r"\s+", spelled_name.strip()) if p]
            spelled_parts = [self._spell_alnum_hyphen(p) for p in parts]
            separator = "ESPACE" if language == "fr" else "SPACE"
            spelled = (
                f" {separator} ".join(spelled_parts)
                if spelled_parts
                else self._spell_alnum_hyphen(spelled_name)
            )

            confirm = (
                f"Merci ! Je l'ai: {spelled}. Donc {spelled_name}. C'est bien ça ?"
                if language == "fr"
                else f"Got it: {spelled}. That's {spelled_name}. Is that correct?"
            )
            self._checkout_phase = CheckoutPhase.NAME_CONFIRM
            await self._speak_response(confirm)
            return True

        if self._checkout_phase == CheckoutPhase.ADDRESS:
            address = self._extract_address(transcript)
            if not address:
                retry = (
                    "Désolé, je n'ai pas compris l'adresse. Pouvez-vous la répéter ?"
                    if language == "fr"
                    else "Sorry—I didn’t catch the address. Can you repeat it?"
                )
                await self._speak_response(retry)
                return True

            self._customer_address = address
            spelled_number = self._spell_first_number(address)
            spelled_postal = self._spell_postal_code(address)

            parts: List[str] = []
            if spelled_number:
                parts.append(
                    f"numéro {spelled_number}"
                    if language == "fr"
                    else f"number {spelled_number}"
                )
            if spelled_postal:
                parts.append(
                    f"code postal {spelled_postal}"
                    if language == "fr"
                    else f"postal code {spelled_postal}"
                )
            spelled_bits = "; ".join(parts)

            if language == "fr":
                confirm = (
                    f"Parfait. J’ai noté: {address}. Pour confirmer: {spelled_bits}. C’est exact ?"
                    if spelled_bits
                    else f"Parfait. J’ai noté: {address}. C’est exact ?"
                )
            else:
                confirm = (
                    f"Great. I got: {address}. Just to confirm: {spelled_bits}. Is that correct?"
                    if spelled_bits
                    else f"Great. I got: {address}. Is that correct?"
                )
            self._checkout_phase = CheckoutPhase.ADDRESS_CONFIRM
            await self._speak_response(confirm)
            return True

        if self._checkout_phase == CheckoutPhase.ADDRESS_CONFIRM:
            treat_as_affirmative = self._is_affirmative(transcript)
            treat_as_negative = self._is_negative(transcript)

            # If the caller responds with a corrected address instead of "yes/no", accept it.
            if not treat_as_affirmative and not treat_as_negative:
                candidate = self._extract_address(transcript)
                if candidate:
                    if (
                        self._customer_address
                        and candidate.casefold() == self._customer_address.casefold()
                    ):
                        treat_as_affirmative = True
                    else:
                        self._customer_address = candidate

                        spelled_number = self._spell_first_number(candidate)
                        spelled_postal = self._spell_postal_code(candidate)

                        parts: List[str] = []
                        if spelled_number:
                            parts.append(
                                f"numéro {spelled_number}"
                                if language == "fr"
                                else f"number {spelled_number}"
                            )
                        if spelled_postal:
                            parts.append(
                                f"code postal {spelled_postal}"
                                if language == "fr"
                                else f"postal code {spelled_postal}"
                            )
                        spelled_bits = "; ".join(parts)

                        if language == "fr":
                            confirm = (
                                f"Parfait. J'ai noté: {candidate}. Pour confirmer: {spelled_bits}. C'est exact ?"
                                if spelled_bits
                                else f"Parfait. J'ai noté: {candidate}. C'est exact ?"
                            )
                        else:
                            confirm = (
                                f"Great. I got: {candidate}. Just to confirm: {spelled_bits}. Is that correct?"
                                if spelled_bits
                                else f"Great. I got: {candidate}. Is that correct?"
                            )
                        await self._speak_response(confirm)
                        return True

            if treat_as_affirmative:
                if self._customer_phone:
                    digits_spelled = self._spell_phone_digits(self._customer_phone)
                    confirm = (
                        f"Parfait. Pour confirmer le numéro: {digits_spelled}. C'est bien ça ?"
                        if language == "fr"
                        else f"Perfect. Just to confirm the number: {digits_spelled}. Did I get that right?"
                    )
                    self._checkout_phase = CheckoutPhase.PHONE_CONFIRM
                    await self._speak_response(confirm)
                    return True

                self._checkout_phase = CheckoutPhase.PHONE
                ask = (
                    "Parfait. Quel est votre numéro de téléphone ?"
                    if language == "fr"
                    else "Perfect. What's the best phone number?"
                )
                await self._speak_response(ask)
                return True

            if treat_as_negative:
                self._checkout_phase = CheckoutPhase.ADDRESS
                ask = (
                    "Oups, d’accord. Pouvez-vous répéter l’adresse lentement ?"
                    if language == "fr"
                    else "Oops—okay. Can you repeat the address slowly?"
                )
                await self._speak_response(ask)
                return True

            reprompt = (
                "Juste pour confirmer: est-ce que l’adresse est correcte ? (Vous pouvez dire oui, ou me donner la bonne adresse.)"
                if language == "fr"
                else "Just to confirm: is that address correct? (You can say yes, or tell me the correct address.)"
            )
            await self._speak_response(reprompt)
            return True

        if self._checkout_phase == CheckoutPhase.PHONE:
            phone = self._extract_phone(transcript)
            if not phone:
                retry = (
                    "Désolé, je n'ai pas bien compris le numéro. Vous pouvez le répéter ?"
                    if language == "fr"
                    else "Sorry—I didn't catch the phone number. Can you repeat it?"
                )
                await self._speak_response(retry)
                return True

            self._customer_phone = phone
            spelled = self._spell_phone_digits(phone)
            confirm = (
                f"Super. Je l'ai noté: {spelled}. C'est bien ça ?"
                if language == "fr"
                else f"Great. I got: {spelled}. Is that right?"
            )
            self._checkout_phase = CheckoutPhase.PHONE_CONFIRM
            await self._speak_response(confirm)
            return True

        if self._checkout_phase == CheckoutPhase.PHONE_CONFIRM:
            if self._is_affirmative(transcript):
                if self._customer_email:
                    spelled = self._spell_email(self._customer_email, language=language)
                    confirm = (
                        f"Parfait. Pour confirmer l'email: {spelled}. C'est bien ça ?"
                        if language == "fr"
                        else f"Perfect. Just to confirm the email: {spelled}. Did I get that right?"
                    )
                    self._checkout_phase = CheckoutPhase.EMAIL_CONFIRM
                    await self._speak_response(confirm)
                    return True

                self._checkout_phase = CheckoutPhase.EMAIL
                ask = (
                    "Parfait. Et quel est votre email ?"
                    if language == "fr"
                    else "Perfect. And what's your email?"
                )
                await self._speak_response(ask)
                return True

            if self._is_negative(transcript):
                self._checkout_phase = CheckoutPhase.PHONE
                ask = (
                    "Oups, d'accord. Pouvez-vous répéter votre numéro lentement ?"
                    if language == "fr"
                    else "Oops—okay. Can you repeat the number slowly?"
                )
                await self._speak_response(ask)
                return True

            reprompt = (
                "Juste pour vérifier: est-ce que le numéro est correct ?"
                if language == "fr"
                else "Just to double-check—did I get the phone number right?"
            )
            await self._speak_response(reprompt)
            return True

        if self._checkout_phase == CheckoutPhase.EMAIL:
            email = self._extract_email(transcript)
            if not email:
                retry = (
                    "Désolé, je n'ai pas compris l'email. Vous pouvez le répéter, ou l'épeler ?"
                    if language == "fr"
                    else "Sorry—I didn't catch the email. Can you repeat it, or spell it?"
                )
                await self._speak_response(retry)
                return True

            self._customer_email = email
            spelled = self._spell_email(email, language=language)
            confirm = (
                f"Parfait. Je l'épelle: {spelled}. C'est bien ça ?"
                if language == "fr"
                else f"Perfect. I'll spell it: {spelled}. Did I get that right?"
            )
            self._checkout_phase = CheckoutPhase.EMAIL_CONFIRM
            await self._speak_response(confirm)
            return True

        if self._checkout_phase == CheckoutPhase.EMAIL_CONFIRM:
            if self._is_affirmative(transcript):
                self._checkout_phase = CheckoutPhase.COMPLETE
                done = (
                    f"Parfait ! Votre commande est confirmée au nom de {self._customer_name}, à {self._customer_address}. Vous voulez ajouter autre chose ?"
                    if language == "fr"
                    else f"Perfect! Your order is confirmed under {self._customer_name} at {self._customer_address}. Anything else you’d like to add?"
                )
                await self._speak_response(done)
                return True

            if self._is_negative(transcript):
                self._checkout_phase = CheckoutPhase.EMAIL
                ask = (
                    "Oups, d'accord. Pouvez-vous répéter l'email lentement ?"
                    if language == "fr"
                    else "Oops—okay. Can you repeat the email slowly?"
                )
                await self._speak_response(ask)
                return True

            reprompt = (
                "Juste pour vérifier: est-ce que l'email est correct ?"
                if language == "fr"
                else "Just to double-check—is that email correct?"
            )
            await self._speak_response(reprompt)
            return True

        if self._checkout_phase == CheckoutPhase.COMPLETE:
            if self._wants_goodbye(transcript):
                goodbye = (
                    "Parfait, merci ! À bientôt."
                    if language == "fr"
                    else "Perfect—thank you! Bye for now."
                )
                await self._speak_response(goodbye)
                if self._matches_any(_HANGUP_NOW_PATTERNS, transcript):
                    await self._hangup_call()
                return True

            if self._wants_done_order(transcript) or self._is_negative(transcript):
                closing = (
                    "Parfait—merci ! Votre commande est confirmée. Si vous avez besoin d'autre chose, je suis là; sinon vous pouvez raccrocher quand vous voulez."
                    if language == "fr"
                    else "Perfect—thank you! Your order is confirmed. If you need anything else, I'm here; otherwise you can hang up whenever you're ready."
                )
                await self._speak_response(closing)
                return True

            # If they say "yes" here, they likely want to add more items.
            if self._is_affirmative(transcript):
                self._checkout_phase = CheckoutPhase.ORDERING
                msg = (
                    "Super ! Qu’est-ce que vous voulez ajouter ?"
                    if language == "fr"
                    else "Awesome! What would you like to add?"
                )
                await self._speak_response(msg)
                return True

            # Default: keep it friendly and guide them back.
            prompt = (
                "D’accord. Vous voulez ajouter quelque chose au menu, ou c’est tout ?"
                if language == "fr"
                else "Got it. Do you want to add anything from the menu, or is that all?"
            )
            await self._speak_response(prompt)
            return True

        return False
    
    async def _apply_language_override(self, target: LanguageCode) -> None:
        """Apply an explicit user override and confirm once."""
        if target == self._lang_state.current:
            return
        
        previous = self._lang_state.current
        now = time.time()
        self._lang_state.force(target, now)
        self._turn_language = target
        
        confirmation = (
            "D’accord, je continue en français."
            if target == "fr"
            else "Sure—switching to English."
        )

        logger.info(
            "Language decision",
            detected_language=None,
            language_confidence=None,
            current_language=self._lang_state.current,
            switched=True,
            reason="override",
            previous_language=previous,
        )

        await self._speak_response(confirmation, tone_mode=ToneMode.GREETING, language=target)
    
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
        self._llm = create_llm()
        
        # Initialize router if enabled
        if self.config.router_enabled and not self.config.menu_only:
            self._router = get_semantic_router()
            self._router.initialize()
        
        # Initialize extraction queue
        self._extraction_queue = ExtractionQueue()
        await self._extraction_queue.start()

        # Warm menu cache (best effort)
        self._menu_catalog = get_menu_catalog()
        
        # Set up STT callback
        self._stt.set_transcript_callback(self._on_transcript)
        self._stt.set_speech_started_callback(self._on_speech_started)
        self._stt.set_eager_end_of_turn_callback(self._on_eager_end_of_turn)
        self._stt.set_turn_resumed_callback(self._on_turn_resumed)
        
        # Initialize Twilio client for hangup
        if self.config.twilio_account_sid and self.config.twilio_auth_token:
            self._twilio_client = TwilioClient(
                self.config.twilio_account_sid,
                self.config.twilio_auth_token,
            )
        
        self._is_running = True
        self._state = PipelineState.IDLE
        
        # Start turn worker (runs LLM/TTS without blocking STT receive loop)
        if self._turn_worker_task is None or self._turn_worker_task.done():
            self._turn_worker_task = asyncio.create_task(self._turn_worker())
        
        logger.info("Voice pipeline started")
    
    async def stop(self) -> None:
        """Stop all pipeline components."""
        logger.info("Stopping voice pipeline")
        
        self._is_running = False
        
        # Cancel any ongoing tasks
        tasks_to_cancel: List[asyncio.Task] = []
        for task in (
            self._output_task,
            self._turn_task,
            self._turn_worker_task,
            self._silence_task,
            self._stt_start_task,
            self._soft_interrupt_watchdog_task,
        ):
            if task and not task.done():
                task.cancel()
                tasks_to_cancel.append(task)
        
        await self._stop_pacer()
        self._cancel_speculative_llm(reason="stop")
        self._cancel_checkout_debounce(reason="stop")
        
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        # Stop components
        await self._stt.stop()
        await self._tts.stop()
        
        if self._extraction_queue:
            await self._extraction_queue.stop()
        
        # Finalize metrics
        self._call_metrics.end_time = time.time()

        call_state = self._protocol.call_state
        metrics = self._call_metrics.to_dict()
        metrics.update(
            {
                "underflows": self._pacer_underruns,
                "late_resets": self._pacer_late_resets,
                "queue_depth_max": self._queue_depth_max,
                "jitter_buffer_ms": self._jitter_buffer_ms,
                "mark_rtt_ms": round(call_state.avg_mark_rtt_ms, 2) if call_state else 0.0,
            }
        )

        logger.info("Voice pipeline stopped", metrics=metrics)
    
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
            rtt_ms = self._protocol.handle_mark(event)
            if rtt_ms:
                call_state = self._protocol.call_state
                logger.debug(
                    "Twilio mark ack",
                    mark_name=event.name,
                    mark_rtt_ms=round(rtt_ms, 2),
                    avg_mark_rtt_ms=round(call_state.avg_mark_rtt_ms, 2) if call_state else 0.0,
                    playback_generation_id=getattr(call_state, "playback_generation_id", None),
                )
            
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

        # Enter listening state immediately; we can greet even if STT has issues.
        self._state = PipelineState.LISTENING
        now = time.time()
        self._last_audio_frame_time = now
        self._last_speech_activity_time = now
        self._last_silence_prompt_time = 0.0

        # Reset per-call contextual buffers.
        self._inbound_ulaw_history.clear()
        self._voice_context = VoiceContextBuffer(max_context_seconds=self.config.csm_max_context_seconds)
        logger.info(
            "Call started",
            call_sid=event.call_sid,
            stream_sid=event.stream_sid,
        )

        # Start STT in the background so a slow STT handshake doesn't block greeting audio.
        if self._stt_start_task and not self._stt_start_task.done():
            self._stt_start_task.cancel()
        self._stt_start_task = asyncio.create_task(self._start_stt_background())

        # Send initial greeting (always).
        greeting = (
            f"Allô ! Merci d’appeler {self.config.company_name} — je suis {self.config.agent_name}. Qu’est-ce que je vous prépare aujourd’hui ?"
            if self._lang_state.current == "fr"
            else f"Hi! Thanks for calling {self.config.company_name} — I’m {self.config.agent_name}. What can I get you from the menu today?"
        )
        # Speak greeting in a background task so STT can keep processing.
        if self._output_task and not self._output_task.done():
            self._output_task.cancel()
        self._output_task = asyncio.create_task(self._speak_response(greeting))

    async def _start_stt_background(self) -> None:
        """Start STT and log success/failure without blocking call audio output."""
        try:
            ok = await self._stt.start()
            if ok:
                logger.info("STT ready")
            else:
                logger.error("STT failed to start")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("STT start task error", error=str(e))
    
    async def _handle_media(self, event: TwilioMediaEvent) -> None:
        """Handle incoming audio from Twilio."""
        if not self._is_running:
            return
        
        # Late-2025 best practice: Send mu-law directly to Deepgram (no conversion!)
        # Deepgram accepts encoding=mulaw&sample_rate=8000
        audio = twilio_ulaw_passthrough(event.payload)
        
        if audio:
            # Keep a rolling buffer of the caller audio for contextual TTS (CSM mode).
            if (self.config.tts_provider or "cartesia").strip().lower() == "csm":
                try:
                    self._inbound_ulaw_history.extend(audio)
                    if len(self._inbound_ulaw_history) > self._inbound_ulaw_history_max_bytes:
                        del self._inbound_ulaw_history[:-self._inbound_ulaw_history_max_bytes]
                except Exception:
                    self._inbound_ulaw_history = bytearray(audio[-self._inbound_ulaw_history_max_bytes :])

            if not self._logged_first_media:
                self._logged_first_media = True
                logger.info(
                    "Inbound media received",
                    call_sid=self.call_sid,
                    stream_sid=self.stream_sid,
                    bytes=len(audio),
                    track=getattr(event, "track", None),
                )
            await self._stt.send_audio(audio)
            self._last_audio_frame_time = time.time()
             
            # Start silence detection if not already running
            if self._silence_task is None or self._silence_task.done():
                self._silence_task = asyncio.create_task(self._silence_detector())

            # Two-stage barge-in: while speaking and soft-paused, detect continued user speech via energy.
            if self._state == PipelineState.SPEAKING and self._tts_soft_paused:
                await self._maybe_hard_interrupt_from_energy(audio)
    
    async def _on_speech_started(self) -> None:
        """Handle VAD speech-start event from STT (used for barge-in)."""
        if not self._is_running:
            return

        self._last_speech_activity_time = time.time()
        
        if self._state == PipelineState.SPEAKING:
            self._speech_started_during_tts = True
            self._begin_soft_interrupt(reason="vad")
            logger.debug("Speech started detected during TTS")

    async def _on_eager_end_of_turn(self, transcript: str) -> None:
        """
        Handle Flux EagerEndOfTurn.

        We start speculative (ephemeral) LLM generation during a pause, and cancel if
        Flux reports TurnResumed (caller keeps talking).
        """
        if not self._is_running:
            return

        # Only speculate while we're listening; if we're already speaking/processing,
        # this doesn't help and may create needless cost.
        if self._state != PipelineState.LISTENING:
            return

        candidate = (transcript or "").strip() or (self._last_interim_transcript or "").strip()
        if not candidate:
            return

        # Don't speculate in deterministic checkout capture states.
        if self.config.menu_only and self._checkout_phase != CheckoutPhase.ORDERING:
            return

        # Avoid speculative calls for tiny backchannels / filler.
        if len(candidate) < 8:
            return
        if self._is_barge_in_backchannel(candidate) and len(candidate.split()) <= 2:
            return

        self._start_speculative_llm(candidate, reason="flux_eager_eot")

    async def _on_turn_resumed(self) -> None:
        """Handle Flux TurnResumed (cancel speculative work)."""
        if not self._is_running:
            return
        self._cancel_speculative_llm(reason="flux_turn_resumed")

    def _start_speculative_llm(self, transcript: str, *, reason: str) -> None:
        if not self._llm:
            return

        text = (transcript or "").strip()
        if not text:
            return

        target_language = self._lang_state.current
        inferred_language, inferred_confidence = infer_language_from_text(text)
        if (
            inferred_language is not None
            and inferred_confidence is not None
            and inferred_confidence >= 0.7
        ):
            target_language = inferred_language
        normalized = _normalize_for_intent(text)
        extra_context = self._build_extra_context(text, language=target_language)

        existing = self._speculative_llm
        if (
            existing
            and existing.transcript_norm == normalized
            and existing.target_language == target_language
            and existing.extra_context == extra_context
        ):
            return

        self._cancel_speculative_llm(reason="replaced")

        state = SpeculativeLLMState(
            transcript=text,
            transcript_norm=normalized,
            target_language=target_language,
            extra_context=extra_context,
            started_at=time.time(),
        )
        state.task = asyncio.create_task(self._run_speculative_llm(state))
        self._speculative_llm = state

        logger.debug(
            "Speculative LLM started",
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            reason=reason,
            language=target_language,
            transcript=_redact_transcript_for_logs(text)[:80],
        )

    def _cancel_speculative_llm(self, *, reason: str) -> None:
        state = self._speculative_llm
        if not state:
            return

        if state.task and not state.task.done():
            state.task.cancel()

        self._speculative_llm = None
        logger.debug(
            "Speculative LLM cancelled",
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            reason=reason,
        )

    async def _run_speculative_llm(self, state: SpeculativeLLMState) -> None:
        if not self._llm:
            return

        try:
            async for chunk in self._llm.generate_streaming_ephemeral(
                state.transcript,
                target_language=state.target_language,
                extra_context=state.extra_context,
            ):
                if state.first_token_at is None:
                    state.first_token_at = time.time()
                state.response_text += chunk
        except asyncio.CancelledError:
            pass
        except Exception as e:
            state.error = str(e)
        finally:
            # Nothing to do; the consumer will decide whether to use it.
            pass

    def _cancel_checkout_debounce(self, *, reason: str) -> None:
        if self._checkout_debounce_task and not self._checkout_debounce_task.done():
            self._checkout_debounce_task.cancel()
        self._checkout_debounce_task = None
        self._checkout_debounce = None
        logger.debug(
            "Checkout debounce cleared",
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            reason=reason,
        )

    @staticmethod
    def _meaningful_token_count(text: str) -> int:
        normalized = _normalize_for_intent(text)
        tokens = re.findall(r"[a-z0-9]+", normalized)
        return len([t for t in tokens if t])

    def _should_debounce_checkout_final(self, text: str) -> bool:
        if not self.config.menu_only:
            return False

        phase = self._checkout_phase
        normalized = _normalize_for_intent(text).strip(" .,!?:;-")

        if phase == CheckoutPhase.NAME:
            extracted = self._extract_name(text)
            if extracted:
                # If the name fragment is very short (e.g., "Hatem"), wait briefly for a
                # possible continuation ("Hatem Ben"). If nothing follows, we'll flush.
                return self._meaningful_token_count(extracted) < 2
            # Prefix-only finals like "my name is" often arrive right before the actual name.
            return bool(
                re.match(
                    r"^(my name is|my name'?s|my name|this is|it'?s|it is|je m'appelle|j'm'appelle|mon nom est|c'?est)$",
                    normalized,
                    flags=re.IGNORECASE,
                )
            )

        if phase == CheckoutPhase.ADDRESS:
            extracted = self._extract_address(text)
            if not extracted:
                return bool(
                    re.match(
                        r"^(my address is|my address|address is|address|mon adresse est|mon adresse|l'adresse est|adresse|c'?est)$",
                        normalized,
                        flags=re.IGNORECASE,
                    )
                )
            # If the address fragment is too short (e.g., "405"), wait briefly for the next final.
            return self._meaningful_token_count(extracted) < 2

        return False

    def _snapshot_recent_user_ulaw(self) -> Optional[bytes]:
        if not self._inbound_ulaw_history:
            return None
        return bytes(self._inbound_ulaw_history)

    async def _enqueue_turn_transcript(
        self,
        *,
        text: str,
        stt_latency_ms: float,
        detected_language: Optional[str],
        language_confidence: Optional[float],
        source: str,
    ) -> None:
        audio_ulaw: Optional[bytes] = None
        if (self.config.tts_provider or "cartesia").strip().lower() == "csm":
            audio_ulaw = self._snapshot_recent_user_ulaw()

        logger.info(
            "Final transcript",
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            menu_only=self.config.menu_only,
            checkout_phase=str(self._checkout_phase),
            source=source,
            text=_redact_transcript_for_logs(text)[:100],
        )
        await self._turn_queue.put(
            QueuedTranscript(
                text=text,
                stt_latency_ms=stt_latency_ms,
                detected_language=detected_language,
                language_confidence=language_confidence,
                audio_ulaw_8k=audio_ulaw,
            )
        )

    async def _flush_checkout_debounce_after(self, *, created_at: float, delay_s: float) -> None:
        try:
            await asyncio.sleep(delay_s)
            if not self._is_running:
                return

            pending = self._checkout_debounce
            if not pending or pending.created_at != created_at:
                return

            # If we're no longer in the same capture phase, don't hold the transcript.
            if pending.phase != self._checkout_phase:
                return

            await self._enqueue_turn_transcript(
                text=pending.text,
                stt_latency_ms=pending.stt_latency_ms,
                detected_language=pending.detected_language,
                language_confidence=pending.language_confidence,
                source="checkout_debounce_flush",
            )
        except asyncio.CancelledError:
            pass
        finally:
            # Clear regardless; worst case we ask again.
            if self._checkout_debounce and self._checkout_debounce.created_at == created_at:
                self._checkout_debounce = None
            self._checkout_debounce_task = None

    async def _maybe_debounce_checkout_final(
        self,
        *,
        text: str,
        stt_latency_ms: float,
        detected_language: Optional[str],
        language_confidence: Optional[float],
    ) -> bool:
        """
        Debounce short/prefix-only finals in NAME/ADDRESS capture.

        Deepgram can finalize a prefix ("my name is") right before the actual value.
        Waiting ~250–400ms and concatenating produces a single, correct utterance.
        """
        if not self.config.menu_only:
            return False

        now = time.time()
        pending = self._checkout_debounce

        # If we already have a pending fragment, try to combine with this one.
        if pending:
            same_phase = pending.phase == self._checkout_phase
            within_window = (now - pending.created_at) <= 0.45

            if same_phase and within_window and self._checkout_phase in (
                CheckoutPhase.NAME,
                CheckoutPhase.ADDRESS,
            ):
                combined = (pending.text + " " + text).strip()
                self._cancel_checkout_debounce(reason="combined")
                await self._enqueue_turn_transcript(
                    text=combined,
                    stt_latency_ms=stt_latency_ms,
                    detected_language=detected_language,
                    language_confidence=language_confidence,
                    source="checkout_debounce_combined",
                )
                return True

            # Stale or phase changed: flush the pending fragment immediately.
            self._cancel_checkout_debounce(reason="stale")
            await self._enqueue_turn_transcript(
                text=pending.text,
                stt_latency_ms=pending.stt_latency_ms,
                detected_language=pending.detected_language,
                language_confidence=pending.language_confidence,
                source="checkout_debounce_stale_flush",
            )

        if self._checkout_phase not in (CheckoutPhase.NAME, CheckoutPhase.ADDRESS):
            return False

        if not self._should_debounce_checkout_final(text):
            return False

        delay_s = 0.35  # 250–400ms window
        self._checkout_debounce = CheckoutDebounceState(
            phase=self._checkout_phase,
            text=text,
            detected_language=detected_language,
            language_confidence=language_confidence,
            stt_latency_ms=stt_latency_ms,
            created_at=now,
        )
        if self._checkout_debounce_task and not self._checkout_debounce_task.done():
            self._checkout_debounce_task.cancel()
        self._checkout_debounce_task = asyncio.create_task(
            self._flush_checkout_debounce_after(created_at=now, delay_s=delay_s)
        )

        logger.debug(
            "Checkout debounce started",
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            checkout_phase=str(self._checkout_phase),
            delay_ms=int(delay_s * 1000),
            text=_redact_transcript_for_logs(text)[:80],
        )
        return True

    async def _on_transcript(self, result: TranscriptionResult) -> None:
        """
        Handle STT transcript.
        
        Only processes final transcripts. Checks for interruption on partials.
        """
        if not self._is_running:
            return

        if result.text and result.text.strip():
            self._last_stt_text_time = time.time()
            self._last_speech_activity_time = time.time()
            # Track latest interim text for Flux EagerEndOfTurn speculation.
            self._last_interim_transcript = result.text.strip()

        if not self._logged_first_transcript and result.text:
            self._logged_first_transcript = True
            logger.info(
                "First STT transcript received",
                call_sid=self.call_sid,
                stream_sid=self.stream_sid,
                is_final=result.is_final,
                speech_final=result.speech_final,
                text=result.text[:80],
            )
        
        # Two-stage interruption while speaking:
        # - Soft: pause TTS enqueue on credible speech onset.
        # - Hard: Twilio clear + cancel TTS/LLM on confirmed interruption.
        if self._state == PipelineState.SPEAKING:
            text = (result.text or "").strip()
            if text:
                word_count = len(text.split())
                self._barge_in_partial_text = text
                self._barge_in_partial_words = word_count
                self._barge_in_is_backchannel = self._is_barge_in_backchannel(text)

                # Avoid soft-pausing the agent for short backchannels ("ok", "yeah", "oui", ...)
                # which can cause audible stutter/ticking if they occur frequently.
                soft_word_threshold = max(1, self.config.min_interruption_words - 1)
                should_soft_interrupt = (
                    self._contains_hard_interrupt_phrase(text)
                    or (
                        not self._barge_in_is_backchannel
                        and (self._speech_started_during_tts or word_count >= soft_word_threshold)
                    )
                )

                if not self._tts_soft_paused and should_soft_interrupt:
                    self._begin_soft_interrupt(
                        reason="speech_started" if self._speech_started_during_tts else "interim"
                    )

                self._speech_started_during_tts = False

                if self._contains_hard_interrupt_phrase(text):
                    logger.info(
                        "Interruption confirmed",
                        reason="phrase",
                        word_count=word_count,
                    )
                    self._pending_transcript = text
                    await self._handle_hard_interrupt(reason="phrase")
                    return

                hard_word_threshold = max(2, self.config.min_interruption_words)
                if word_count >= hard_word_threshold and not self._barge_in_is_backchannel:
                    logger.info(
                        "Interruption confirmed",
                        reason="words",
                        word_count=word_count,
                        threshold=hard_word_threshold,
                    )
                    self._pending_transcript = text
                    await self._handle_hard_interrupt(reason="words")
                    return

            # Backchannels while the agent is talking should not trigger a hard interrupt.
            if (result.is_final or result.speech_final) and self._is_barge_in_backchannel(result.text):
                if self._tts_soft_paused:
                    self._clear_soft_interrupt(reason="backchannel_final")
                return

            # Do not process transcripts into turns while the agent is still speaking.
            return
        
        # Process final transcripts.
        #
        # NOTE: Deepgram's `speech_final` may not always be present / true depending
        # on model + endpointing mode. Requiring it can lead to "greets but never
        # responds" if we only ever receive `is_final=true` segments.
        if result.is_final or result.speech_final:
            transcript = result.text.strip()
            
            if not transcript:
                return
            
            # Use pending transcript if we were interrupted
            if self._pending_transcript:
                transcript = self._pending_transcript
                self._pending_transcript = ""

            if await self._maybe_debounce_checkout_final(
                text=transcript,
                stt_latency_ms=result.latency_ms,
                detected_language=result.detected_language,
                language_confidence=result.language_confidence,
            ):
                return

            await self._enqueue_turn_transcript(
                text=transcript,
                stt_latency_ms=result.latency_ms,
                detected_language=result.detected_language,
                language_confidence=result.language_confidence,
                source="stt_final",
            )
    
    async def _handle_hard_interrupt(self, *, reason: str) -> None:
        """Hard interrupt (barge-in): clear Twilio + cancel active tasks."""
        if self._state != PipelineState.SPEAKING:
            return

        now = time.time()
        barge_in_hard_ms = (
            (now - self._barge_in_soft_started_at) * 1000
            if self._barge_in_soft_started_at
            else 0.0
        )

        logger.info(
            "Barge-in hard interrupt",
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            reason=reason,
            barge_in_hard_ms=round(barge_in_hard_ms, 2),
            last_rms=self._barge_in_last_rms,
            consecutive_speech_ms=self._barge_in_consecutive_speech_ms,
            partial_words=self._barge_in_partial_words,
            is_backchannel=self._barge_in_is_backchannel,
        )
        
        self._state = PipelineState.INTERRUPTED
        self._call_metrics.total_interruptions += 1
        self._speech_started_during_tts = False
        self._clear_soft_interrupt(reason="hard")
        self._cancel_speculative_llm(reason="hard_interrupt")
        self._cancel_checkout_debounce(reason="hard_interrupt")
        
        if self._current_turn_metrics:
            self._current_turn_metrics.was_interrupted = True
        
        # Cancel any active output/turn task so we stop producing audio immediately.
        if self._output_task and not self._output_task.done():
            self._output_task.cancel()
        
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        
        # Nuclear flush:
        # - Cancel Cartesia context (best effort)
        # - Stop our pacer + drop queued audio
        # - Clear Twilio's buffered audio
        # Bump playback generation so any late mark acks are ignored after `clear`.
        playback_generation_id = self._protocol.bump_playback_generation()

        # Tell Twilio to flush buffered audio immediately (authoritative for barge-in).
        clear_msg = self._protocol.create_clear()
        if clear_msg:
            try:
                await self._send_message(clear_msg)
                logger.info(
                    "Twilio clear sent",
                    call_sid=self.call_sid,
                    stream_sid=self.stream_sid,
                    playback_generation_id=playback_generation_id,
                    reason=reason,
                )
            except Exception as e:
                logger.warning("Failed to send Twilio clear", error=str(e))

        tasks: List[asyncio.Future] = []

        try:
            tasks.append(asyncio.create_task(self._tts.cancel_context()))
        except Exception:
            pass

        self._tts.cancel_current()
        tasks.append(asyncio.create_task(self._stop_pacer()))
        self._outbound_ulaw_remainder = b""

        # Reset adaptive jitter buffer to the baseline after a hard barge-in so the next turn
        # stays snappy; it will auto-increase again if underflows recur.
        self._jitter_buffer_ms = self._jitter_buffer_base_ms
        self._jitter_buffer_stable_bursts = 0

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.debug("Interruption cleanup task failed", error=str(r))
        
        # Reset STT state
        self._stt.reset_state()
        
        self._state = PipelineState.LISTENING
    
    async def _process_transcript(
        self,
        transcript: str,
        *,
        detected_language: Optional[str] = None,
        language_confidence: Optional[float] = None,
    ) -> None:
        """
        Process a final transcript.
        
        Tries routing first, then falls back to LLM.
        """
        self._state = PipelineState.PROCESSING
        
        # Explicit override phrases always win (do not route/LLM these).
        override_language = detect_language_override(transcript) or detect_language_switch_command(transcript)
        if override_language and override_language != self._lang_state.current:
            await self._apply_language_override(override_language)
            return

        # Stabilized auto-switch using Deepgram's language detection (when available),
        # otherwise fall back to lightweight text-based inference.
        previous_language = self._lang_state.current
        now = time.time()

        detection_source: Optional[str] = None
        effective_detected_language = detected_language
        effective_language_confidence = language_confidence

        if effective_detected_language and effective_language_confidence is not None:
            detection_source = "deepgram"
        else:
            inferred_language, inferred_confidence = infer_language_from_text(transcript)
            if inferred_language is not None:
                effective_detected_language = inferred_language
                effective_language_confidence = inferred_confidence
                detection_source = "text"

        # Reply language: follow the detected language of the *last user utterance* when it
        # is confident/meaningful, without announcing. Keep stabilization for `current`.
        reply_language: LanguageCode = self._lang_state.current
        normalized_detected = normalize_detected_language(effective_detected_language)
        try:
            confidence_value = (
                float(effective_language_confidence)
                if effective_language_confidence is not None
                else None
            )
        except (TypeError, ValueError):
            confidence_value = None

        normalized_text = _normalize_for_intent(transcript).strip(" .,!?:;-")
        greeting_tokens = {
            "bonjour",
            "salut",
            "allo",
            "bonsoir",
            "hello",
            "hi",
            "hey",
        }
        is_short_greeting = normalized_text in greeting_tokens

        if (
            normalized_detected
            and confidence_value is not None
            and confidence_value >= self._lang_state.CONFIDENCE_SWITCH
            and not self._is_barge_in_backchannel(transcript)
        ):
            if len(transcript.strip()) >= self._lang_state.MIN_CHARS or is_short_greeting:
                reply_language = normalized_detected

        self._turn_language = reply_language

        switched, reason = self._lang_state.update_from_detection(
            text=transcript,
            detected_language=effective_detected_language,
            language_confidence=effective_language_confidence,
            now=now,
        )

        logger.info(
            "Language decision",
            detected_language=effective_detected_language,
            language_confidence=effective_language_confidence,
            current_language=self._lang_state.current,
            reply_language=reply_language,
            switched=switched,
            reason=reason or None,
            previous_language=previous_language,
            detection_source=detection_source,
        )

        # Menu-only checkout flow: name/address confirmation (with spelling).
        checkout_phase_before = self._checkout_phase
        if await self._handle_checkout_flow(transcript):
            logger.info(
                "Routing decision",
                call_sid=self.call_sid,
                menu_only=self.config.menu_only,
                checkout_phase=str(checkout_phase_before),
                decision="checkout",
                next_checkout_phase=str(self._checkout_phase),
            )
            return
        
        # Try semantic routing first
        if (
            not self.config.menu_only
            and self._turn_language == "en"
            and self._router
            and self._router.is_enabled
        ):
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
                return
        
        # Fall back to LLM
        logger.info(
            "Routing decision",
            call_sid=self.call_sid,
            menu_only=self.config.menu_only,
            checkout_phase=str(self._checkout_phase),
            decision="llm",
        )
        await self._generate_llm_response(transcript)
    
    async def _generate_llm_response(self, transcript: str) -> None:
        """Generate and speak an LLM response."""
        if not self._llm:
            logger.error("LLM not initialized")
            return

        llm_start = time.time()
        first_token_time: Optional[float] = None
        full_response = ""

        extra_context = self._build_extra_context(transcript, language=self._turn_language)
        transcript_norm = _normalize_for_intent(transcript)

        speculative = self._speculative_llm
        speculative_match = bool(
            speculative
            and not speculative.error
            and speculative.transcript_norm == transcript_norm
            and speculative.target_language == self._turn_language
            and speculative.extra_context == extra_context
        )

        # If a speculative task exists but doesn't match this turn, cancel it to avoid
        # spending tokens on stale/incorrect partials.
        if speculative and not speculative_match:
            self._cancel_speculative_llm(reason="final_mismatch")
            speculative = None

        # Collect LLM response (keep processing state until we actually speak)
        try:
            self._state = PipelineState.PROCESSING

            used_speculative = False
            if speculative_match and speculative and speculative.task:
                logger.debug(
                    "Using speculative LLM",
                    call_sid=self.call_sid,
                    stream_sid=self.stream_sid,
                )

                # Wait for the speculative generation to finish (or be cancelled).
                await speculative.task

                if speculative.error or not (speculative.response_text or "").strip():
                    logger.debug(
                        "Speculative LLM unusable, falling back",
                        call_sid=self.call_sid,
                        stream_sid=self.stream_sid,
                        error=speculative.error,
                    )
                    self._speculative_llm = None
                else:
                    used_speculative = True
                    llm_start = speculative.started_at
                    first_token_time = speculative.first_token_at
                    full_response = speculative.response_text

                    # Consume speculative state to prevent re-use.
                    self._speculative_llm = None

                    # Commit history only when we actually use the output.
                    if full_response and self._state != PipelineState.INTERRUPTED:
                        self._llm.commit_turn(
                            user_message=transcript,
                            assistant_message=full_response,
                        )

            if not used_speculative:
                # Normal (non-speculative) generation path.
                llm_start = time.time()
                first_token_time = None
                full_response = ""

                async for chunk in self._llm.generate_streaming(
                    transcript,
                    target_language=self._turn_language,
                    extra_context=extra_context,
                ):
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
                if first_token_time is not None:
                    self._current_turn_metrics.llm_first_token_ms = (
                        first_token_time - llm_start
                    ) * 1000
                self._current_turn_metrics.llm_total_ms = (llm_end - llm_start) * 1000
            
            # Speak the response if not interrupted
            if self._state != PipelineState.INTERRUPTED and full_response:
                self._maybe_sync_checkout_phase_from_assistant(full_response)
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
            if self._state == PipelineState.SPEAKING:
                self._state = PipelineState.LISTENING
            self._speech_started_during_tts = False

    def _build_extra_context(
        self,
        transcript: str,
        *,
        language: Optional[LanguageCode] = None,
    ) -> Optional[str]:
        """
        Build optional extra system context for the LLM.

        Currently adds menu context when the caller asks about the menu, prices,
        or mentions a menu item.
        """
        catalog = self._menu_catalog or get_menu_catalog()
        if not catalog:
            return None

        language = language or self._turn_language
        wants_menu = looks_like_menu_request(transcript, language=language)
        wants_full_menu = looks_like_full_menu_request(transcript, language=language)
        section_hits = find_menu_sections(catalog, transcript, limit=2)
        matches = find_menu_items(catalog, transcript, limit=10)

        # If we're not going to reference menu data this turn, don't inject it.
        if not wants_menu and not wants_full_menu and not section_hits and not matches:
            return None

        sections_sorted = sorted(catalog.sections.keys())
        section_list = "; ".join(sections_sorted)

        if language == "fr":
            header = "DONNÉES MENU (source: menu.html)"
            sections_header = f"SECTIONS: {section_list}"
            matches_header = "ARTICLES PERTINENTS:"
        else:
            header = "MENU DATA (source: menu.html)"
            sections_header = f"SECTIONS: {section_list}"
            matches_header = "RELEVANT ITEMS:"

        lines: List[str] = [header, sections_header]

        if wants_full_menu:
            lines.extend(catalog.to_prompt_lines())
            logger.info("Menu context included", reason="full_menu", current_language=language)
            return "\n".join(lines)

        added: int = 0
        seen: set[tuple[str, str]] = set()

        def add_item(item_section: str, item_name: str, item_price: str) -> None:
            nonlocal added
            key = (item_section, item_name)
            if key in seen:
                return
            seen.add(key)
            lines.append(f"- {item_section}: {item_name} — {item_price}")
            added += 1

        # If the caller mentions a section (e.g., "desserts"), include that section excerpt.
        if section_hits:
            for section in section_hits:
                for item in catalog.sections.get(section, ()):
                    add_item(item.section, item.name, item.price_text)
                    if added >= 16:
                        break
                if added >= 16:
                    break

        # Otherwise, include direct item matches.
        if matches and added < 16:
            for item in matches:
                add_item(item.section, item.name, item.price_text)
                if added >= 16:
                    break

        if added > 0:
            lines.insert(2, matches_header)
            reason = "section_match" if section_hits else "menu_item_match"
            logger.info(
                "Menu context included",
                reason=reason,
                current_language=language,
                num_items=added,
                num_sections=len(section_hits),
            )
            return "\n".join(lines)

        # Menu request without a specific section/item: provide sections only (small context).
        logger.info("Menu context included", reason="menu_request", current_language=language)
        return "\n".join(lines)

    async def _run_turn(self, queued: QueuedTranscript) -> None:
        """Run a single conversation turn from a final transcript."""
        self._start_turn()
        if self._current_turn_metrics:
            self._current_turn_metrics.stt_ms = queued.stt_latency_ms

        # Feed the contextual TTS buffer with the latest user turn (best-effort).
        try:
            tts_provider = (self.config.tts_provider or "cartesia").strip().lower()
            if tts_provider == "csm":
                self._voice_context.add_turn(
                    role="user",
                    text=queued.text,
                    audio_ulaw_8k=queued.audio_ulaw_8k,
                )
            else:
                self._voice_context.add_turn(role="user", text=queued.text)
        except Exception:
            pass
        
        try:
            await self._process_transcript(
                queued.text,
                detected_language=queued.detected_language,
                language_confidence=queued.language_confidence,
            )
        except asyncio.CancelledError:
            logger.debug("Turn cancelled")
        except Exception as e:
            logger.error("Turn failed", error=str(e))
        finally:
            self._end_turn()
            if self._state == PipelineState.INTERRUPTED:
                self._state = PipelineState.LISTENING

    async def _turn_worker(self) -> None:
        """Background worker that processes queued final transcripts sequentially."""
        try:
            while self._is_running:
                try:
                    queued = await asyncio.wait_for(self._turn_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                try:
                    self._turn_task = asyncio.create_task(self._run_turn(queued))
                    try:
                        await self._turn_task
                    except asyncio.CancelledError:
                        # Turn cancellation is expected during barge-in.
                        pass
                finally:
                    self._turn_task = None
                    self._turn_queue.task_done()
        except asyncio.CancelledError:
            pass
    
    async def _audio_pacer(self) -> None:
        """
        Outbound audio pacer - sends frames at exactly 20ms intervals.
        
        Late-2025 best practice: Pace outbound audio to prevent buffer underflows
        that cause choppy playback. Uses a jitter buffer before starting.
        """
        self._pacer_running = True
        frame_duration = 0.020  # 20ms per frame
        next_send_time = time.time()
        frames_buffer: List[OutboundMessage] = []

        need_prebuffer = True
        audio_active = False
        last_audio_sent_time = 0.0
        burst_underflows = 0

        def _tune_jitter_buffer(*, increase: bool, reason: str) -> None:
            before = int(self._jitter_buffer_ms)
            if increase:
                bump = max(self._jitter_buffer_step_ms * 2, 40)
                after = min(self._jitter_buffer_max_ms, before + bump)
                self._jitter_buffer_stable_bursts = 0
            else:
                after = max(self._jitter_buffer_min_ms, before - self._jitter_buffer_step_ms)

            if after != before:
                self._jitter_buffer_ms = after
                logger.info(
                    "Adaptive jitter buffer tuned",
                    before_ms=before,
                    after_ms=after,
                    reason=reason,
                )

        try:
            # Main pacing loop
            while self._pacer_running and self._is_running:
                # If we were playing audio and have gone idle long enough, end the burst so the
                # next burst will prebuffer again.
                now = time.time()
                if (
                    audio_active
                    and not frames_buffer
                    and self._outbound_queue.empty()
                    and last_audio_sent_time
                    and (now - last_audio_sent_time) * 1000 >= self._jitter_buffer_idle_threshold_ms
                ):
                    audio_active = False
                    need_prebuffer = True
                    if burst_underflows == 0:
                        self._jitter_buffer_stable_bursts += 1
                        if self._jitter_buffer_stable_bursts >= 3:
                            _tune_jitter_buffer(increase=False, reason="stable")
                            self._jitter_buffer_stable_bursts = 0
                    else:
                        self._jitter_buffer_stable_bursts = 0
                    burst_underflows = 0

                # Get next message (prefer buffer).
                outbound: Optional[OutboundMessage]
                if frames_buffer:
                    outbound = frames_buffer.pop(0)
                else:
                    try:
                        outbound = self._outbound_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        outbound = None

                if outbound is None:
                    # No message ready. If we're in an active audio burst, this is an underflow
                    # (the producer didn't stay ahead of the 20ms schedule).
                    if audio_active:
                        try:
                            outbound = await asyncio.wait_for(self._outbound_queue.get(), timeout=0.020)
                        except asyncio.TimeoutError:
                            self._pacer_underruns += 1
                            burst_underflows += 1
                            _tune_jitter_buffer(increase=True, reason="underflow")
                            next_send_time = time.time()
                            continue
                    else:
                        await asyncio.sleep(0.01)
                        continue

                if outbound is None:  # Poison pill
                    break

                if not outbound.pace:
                    await self._send_message(outbound.message)
                    continue

                # Start of a new audio burst: prebuffer some frames for smooth playback.
                if need_prebuffer:
                    prebuffer_frames = max(3, int(self._jitter_buffer_ms / 20))
                    frames_buffer = [outbound]
                    buffered_audio = 1
                    prebuffer_deadline = time.time() + (self._jitter_buffer_ms / 1000.0) + 0.15
                    while (
                        buffered_audio < prebuffer_frames
                        and self._pacer_running
                        and self._is_running
                        and time.time() < prebuffer_deadline
                    ):
                        try:
                            nxt = await asyncio.wait_for(self._outbound_queue.get(), timeout=0.05)
                        except asyncio.TimeoutError:
                            break
                        if nxt is None:
                            break
                        frames_buffer.append(nxt)
                        if nxt.pace:
                            buffered_audio += 1
                    need_prebuffer = False
                    audio_active = True
                    next_send_time = time.time()
                    continue

                # Wait until next send time for precise 20ms intervals.
                wait_time = next_send_time - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                await self._send_message(outbound.message)
                last_audio_sent_time = time.time()
                next_send_time += frame_duration

                # If we're falling behind, reset timing (separate from underflows).
                if time.time() > next_send_time + 0.040:  # More than 2 frames behind
                    self._pacer_late_resets += 1
                    if self._pacer_late_resets % 10 == 0:
                        logger.warning(
                            "Audio pacer reset timing",
                            late_resets=self._pacer_late_resets,
                            behind_ms=(time.time() - next_send_time) * 1000,
                        )
                    next_send_time = time.time()
                    
        except asyncio.CancelledError:
            pass
        finally:
            self._pacer_running = False
    
    async def _start_pacer(self) -> None:
        """Start the audio pacer task."""
        if self._pacer_task is None or self._pacer_task.done():
            self._outbound_queue = asyncio.Queue()
            self._pacer_task = asyncio.create_task(self._audio_pacer())
    
    async def _stop_pacer(self) -> None:
        """Stop the audio pacer and clear queue."""
        self._pacer_running = False
        
        # Clear the queue
        while not self._outbound_queue.empty():
            try:
                self._outbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Send poison pill
        await self._outbound_queue.put(None)
        
        if self._pacer_task and not self._pacer_task.done():
            self._pacer_task.cancel()
            try:
                await self._pacer_task
            except asyncio.CancelledError:
                pass

    async def _wait_for_stt_quiet(self, *, min_quiet_ms: int = 250, max_wait_ms: int = 1200) -> None:
        """
        Wait briefly for STT to go quiet so we don't talk over the caller.

        We use the last time we saw any STT text (including interim results) as a proxy for
        "user is still talking". This helps in cases where Deepgram emits early finals.
        """
        if self._last_stt_text_time <= 0:
            return

        start = time.time()
        while self._is_running and self._state != PipelineState.INTERRUPTED:
            since_ms = (time.time() - self._last_stt_text_time) * 1000
            if since_ms >= min_quiet_ms:
                return
            if (time.time() - start) * 1000 >= max_wait_ms:
                return
            await asyncio.sleep(0.02)

    async def _enqueue_outbound(self, message: str, *, pace: bool) -> None:
        """Enqueue an outbound message and track queue depth."""
        if not message:
            return

        await self._outbound_queue.put(OutboundMessage(message, pace=pace))
        try:
            depth = self._outbound_queue.qsize()
        except Exception:
            return
        if depth > self._queue_depth_max:
            self._queue_depth_max = depth
    
    async def _speak_response(
        self,
        text: str,
        *,
        tone_mode: Optional[ToneMode] = None,
        language: Optional[LanguageCode] = None,
    ) -> None:
        """
        Synthesize and stream TTS response to Twilio via audio pacer.
        
        Late-2025 best practice: Queue audio frames to pacer for smooth 20ms timing.
        """
        if not text or not self._is_running:
            return

        lang = language or self._turn_language or self._lang_state.current
        mode = self._infer_tone_mode(text, requested=tone_mode)
        text = self._apply_tone(text, language=lang, tone_mode=mode)

        # Avoid talking over the caller when STT is still producing interim text.
        min_quiet_ms = 350 if mode in (ToneMode.CHECKOUT_CAPTURE, ToneMode.REPAIR) else 250
        await self._wait_for_stt_quiet(min_quiet_ms=min_quiet_ms)
        
        tts_start = time.time()
        first_audio_time = None
        
        self._state = PipelineState.SPEAKING
        
        # Start pacer if not running
        await self._start_pacer()

        tts_provider = (self.config.tts_provider or "cartesia").strip().lower()
        context_payload: Optional[list[dict]] = None
        if tts_provider == "csm":
            try:
                context_payload = self._voice_context.build_payload()
            except Exception:
                context_payload = None

        # Capture a short tail of what we actually send (for fallback context snippets).
        snippet_cap_s = 0.0
        if tts_provider == "csm":
            window_s = float(self.config.csm_max_context_seconds or 0.0)
            if window_s > 0:
                snippet_cap_s = min(6.0, max(3.0, window_s / 4.0))
        tail_max_bytes = int(TWILIO_SAMPLE_RATE * snippet_cap_s) if snippet_cap_s > 0 else 0
        assistant_ulaw_tail = bytearray()
        assistant_wav_24k: Optional[bytes] = None
        was_cancelled = False
        was_interrupted = False

        try:
            # Reset streaming frame remainder for this utterance.
            # Important: Do NOT pad to 160 bytes per TTS chunk; only pad once at the end,
            # otherwise we insert silence between streamed chunks which sounds like "ticking".
            self._outbound_ulaw_remainder = b""

            voice_id = self._current_cartesia_voice_id(language=lang)

            # For CSM, synthesize the full utterance in one request to preserve prosody and
            # reduce network roundtrips; keep sentence marks for fast streaming providers.
            segments = [text] if tts_provider == "csm" else self._split_sentences(text)

            # Best practice: Insert marks at sentence boundaries to track real playback and
            # make interruptions feel clean.
            for segment in segments:
                if not self._is_running or self._state == PipelineState.INTERRUPTED:
                    break

                async for chunk in self._tts.synthesize_streaming(
                    segment,
                    voice_id=voice_id,
                    context=context_payload,
                    language=lang,
                ):
                    if not self._is_running or self._state == PipelineState.INTERRUPTED:
                        break

                    if chunk.audio_bytes:
                        if chunk.audio_wav_24k and assistant_wav_24k is None:
                            assistant_wav_24k = chunk.audio_wav_24k

                        if first_audio_time is None:
                            first_audio_time = time.time()
                            if self._current_turn_metrics:
                                self._current_turn_metrics.tts_start_ms = (
                                    first_audio_time - tts_start
                                ) * 1000

                        # Queue audio frames to pacer for smooth playback.
                        # Preserve framing across streaming chunks to avoid injecting padding between chunks.
                        self._outbound_ulaw_remainder += chunk.audio_bytes
                        while (
                            not self._tts_soft_paused
                            and len(self._outbound_ulaw_remainder) >= TWILIO_FRAME_SIZE
                        ):
                            frame = self._outbound_ulaw_remainder[:TWILIO_FRAME_SIZE]
                            self._outbound_ulaw_remainder = self._outbound_ulaw_remainder[TWILIO_FRAME_SIZE:]
                            if tail_max_bytes > 0:
                                assistant_ulaw_tail += frame
                                if len(assistant_ulaw_tail) > tail_max_bytes:
                                    del assistant_ulaw_tail[:-tail_max_bytes]
                            msg = create_media_message(self._protocol.stream_sid, frame)
                            await self._enqueue_outbound(msg, pace=True)

                    if (
                        chunk.is_final
                        and self._outbound_ulaw_remainder
                        and not self._tts_soft_paused
                    ):
                        frame = self._outbound_ulaw_remainder.ljust(TWILIO_FRAME_SIZE, b"\xff")
                        self._outbound_ulaw_remainder = b""
                        if tail_max_bytes > 0:
                            assistant_ulaw_tail += frame
                            if len(assistant_ulaw_tail) > tail_max_bytes:
                                del assistant_ulaw_tail[:-tail_max_bytes]
                        msg = create_media_message(self._protocol.stream_sid, frame)
                        await self._enqueue_outbound(msg, pace=True)

                # If we soft-paused mid-sentence (backchannel/noise), wait briefly for resume so we
                # can flush the buffered tail and place a correct mark.
                while self._tts_soft_paused and self._is_running and self._state == PipelineState.SPEAKING:
                    await asyncio.sleep(0.01)

                if not self._is_running or self._state == PipelineState.INTERRUPTED:
                    break

                if self._outbound_ulaw_remainder:
                    while len(self._outbound_ulaw_remainder) >= TWILIO_FRAME_SIZE:
                        frame = self._outbound_ulaw_remainder[:TWILIO_FRAME_SIZE]
                        self._outbound_ulaw_remainder = self._outbound_ulaw_remainder[TWILIO_FRAME_SIZE:]
                        if tail_max_bytes > 0:
                            assistant_ulaw_tail += frame
                            if len(assistant_ulaw_tail) > tail_max_bytes:
                                del assistant_ulaw_tail[:-tail_max_bytes]
                        msg = create_media_message(self._protocol.stream_sid, frame)
                        await self._enqueue_outbound(msg, pace=True)

                    if self._outbound_ulaw_remainder:
                        frame = self._outbound_ulaw_remainder.ljust(TWILIO_FRAME_SIZE, b"\xff")
                        self._outbound_ulaw_remainder = b""
                        if tail_max_bytes > 0:
                            assistant_ulaw_tail += frame
                            if len(assistant_ulaw_tail) > tail_max_bytes:
                                del assistant_ulaw_tail[:-tail_max_bytes]
                        msg = create_media_message(self._protocol.stream_sid, frame)
                        await self._enqueue_outbound(msg, pace=True)

                mark_msg = self._protocol.create_mark()
                if mark_msg:
                    await self._enqueue_outbound(mark_msg, pace=False)
            
            # Record TTS metrics
            if self._current_turn_metrics:
                self._current_turn_metrics.tts_total_ms = (time.time() - tts_start) * 1000
            
        except asyncio.CancelledError:
            logger.debug("TTS streaming cancelled")
            was_cancelled = True
        except Exception as e:
            logger.error("TTS streaming failed", error=str(e))
        finally:
            was_interrupted = self._state == PipelineState.INTERRUPTED
            if self._state == PipelineState.SPEAKING:
                self._state = PipelineState.LISTENING

            if self._is_running and not was_cancelled and not was_interrupted:
                try:
                    if tts_provider == "csm":
                        if assistant_wav_24k:
                            self._voice_context.add_turn(
                                role="assistant",
                                text=text,
                                audio_wav_24k=assistant_wav_24k,
                            )
                        else:
                            self._voice_context.add_turn(
                                role="assistant",
                                text=text,
                                audio_ulaw_8k=bytes(assistant_ulaw_tail) if assistant_ulaw_tail else None,
                            )
                    else:
                        self._voice_context.add_turn(role="assistant", text=text)
                except Exception:
                    pass
    
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
            self._speech_started_during_tts = False
    
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
                
                # Speech-aware silence: use VAD/transcript activity, not raw Twilio frames (which
                # often continue even during silence).
                now = time.time()
                base = self._last_speech_activity_time or self._last_audio_frame_time or now
                silence_duration = now - base
                since_last_prompt = now - (self._last_silence_prompt_time or 0.0)
                
                if (
                    silence_duration > self.config.silence_timeout_seconds * 3
                    and since_last_prompt > self.config.silence_timeout_seconds * 3
                ):
                    # Extended silence - prompt user
                    logger.debug("Extended silence detected")
                    lang = self._turn_language or self._lang_state.current
                    prompt = (
                        "Vous êtes toujours là ? Est-ce que je peux vous aider avec autre chose ?"
                        if lang == "fr"
                        else "Are you still there? Is there anything else I can help you with?"
                    )
                    await self._speak_response(prompt, tone_mode=ToneMode.REPAIR, language=lang)
                    self._last_silence_prompt_time = now
                     
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
