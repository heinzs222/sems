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
from src.agent.audio import twilio_ulaw_passthrough, chunk_audio, TWILIO_FRAME_SIZE
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
    LangState,
    LanguageCode,
)
from src.agent.menu import get_menu_catalog, looks_like_menu_request, find_menu_items

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
    ADDRESS = "address"
    ADDRESS_CONFIRM = "address_confirm"
    PHONE = "phone"
    PHONE_CONFIRM = "phone_confirm"
    EMAIL = "email"
    EMAIL_CONFIRM = "email_confirm"
    COMPLETE = "complete"


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
        r"(^|\\b)(yes|yeah|yep|correct|thats right|that's right|right|ok|okay|sure)(\\b|$)",
        r"(^|\\b)(oui|ouais|d'accord|daccord|exact|c'est ca|cest ca)(\\b|$)",
    )
)

_NO_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"(^|\\b)(no|nope|nah|wrong|not correct)(\\b|$)",
        r"(^|\\b)(non|pas du tout|c'est pas ca|cest pas ca)(\\b|$)",
    )
)

_DONE_ORDER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\\b(thats all|that's all|thats it|that's it|nothing else|im done|i'm done|done)(\\b|$)",
        r"\\b(c'est tout|cest tout|rien d'autre|j'ai fini|jai fini|c'est bon|cest bon)(\\b|$)",
    )
)

_MODIFY_ORDER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\\b(add|actually|instead|change|wait|hold on)(\\b|$)",
        r"\\b(ajoute|ajouter|finalement|au fait|changer|attends)(\\b|$)",
    )
)

_GOODBYE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\\b(bye|goodbye|hang up|end call)(\\b|$)",
        r"\\b(au revoir|bye bye|raccroche|raccrocher)(\\b|$)",
    )
)

_HANGUP_NOW_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\\b(hang up|end call)(\\b|$)",
        r"\\b(raccroche|raccrocher)(\\b|$)",
    )
)

_CA_POSTAL_RE = re.compile(r"\\b([a-z]\\d[a-z])\\s?(\\d[a-z]\\d)\\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b", re.IGNORECASE)

_BARGE_IN_BACKCHANNEL_RE = re.compile(
    r"^(ok|okay|yeah|yep|yup|uh huh|uh-huh|mhm|mm hmm|mm-hmm|oui|ouais|d['’]accord|daccord)([.!?])?$"
)

_BARGE_IN_HARD_PHRASE_RE = re.compile(
    r"\\b(stop|wait|hold on|one second|pause|cancel|actually|listen|arrete|attends|un instant|annule|ecoute)\\b"
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


@dataclass(frozen=True)
class OutboundMessage:
    """Message to send to Twilio, optionally paced as audio."""

    message: str
    pace: bool = True  # True for 20ms media frames; False for control (mark/clear/etc.)


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
        
        # Components
        self._protocol = TwilioProtocolHandler()
        self._stt = STTManager()
        self._tts = TTSManager()
        self._llm: Optional[GroqLLM] = None
        self._router: Optional[SemanticRouter] = None
        self._extraction_queue: Optional[ExtractionQueue] = None
        self._menu_catalog = None
        
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

        # Track the last time we received any non-empty STT text (including interim results).
        # Used to avoid talking over the caller when Deepgram endpointing emits early finals.
        self._last_stt_text_time: float = 0.0
        
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
        self._queue_depth_max: int = 0
        self._jitter_buffer_ms: int = 100  # Buffer 100ms before starting playback
        self._outbound_ulaw_remainder: bytes = b""
        
        # Silence detection
        self._last_speech_time: float = 0.0
        self._silence_task: Optional[asyncio.Task] = None

        # Debug flags (kept minimal to avoid noisy logs)
        self._logged_first_media: bool = False
        self._logged_first_transcript: bool = False
        
        # Twilio client for hangup
        self._twilio_client: Optional[TwilioClient] = None
    
    def _current_cartesia_voice_id(self) -> str:
        if self._lang_state.current == "fr":
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

    def _extract_name(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""

        normalized = _normalize_for_intent(t)
        patterns = (
            r"^(my name is|this is|it's|it is)\\s+",
            r"^(je m'appelle|mon nom est|c'est)\\s+",
        )
        for pat in patterns:
            normalized = re.sub(pat, "", normalized, flags=re.IGNORECASE)

        # Prefer the original casing when possible by removing common prefixes again.
        original = t
        original = re.sub(
            r"^(my name is|this is|it's|it is)\\s+",
            "",
            original,
            flags=re.IGNORECASE,
        ).strip()
        original = re.sub(
            r"^(je m'appelle|mon nom est|c'est)\\s+",
            "",
            original,
            flags=re.IGNORECASE,
        ).strip()
        return original if original else normalized

    def _extract_address(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        original = t
        original = re.sub(
            r"^(my address is|address is|it's|it is)\\s+",
            "",
            original,
            flags=re.IGNORECASE,
        ).strip()
        original = re.sub(
            r"^(mon adresse est|l'adresse est|adresse|c'est)\\s+",
            "",
            original,
            flags=re.IGNORECASE,
        ).strip()
        return original if original else t

    def _extract_phone(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""

        original = t
        original = re.sub(
            r"^(my phone is|my number is|phone number is|phone is|number is)\\s+",
            "",
            original,
            flags=re.IGNORECASE,
        ).strip()
        original = re.sub(
            r"^(mon numero est|mon num\\S+ro est|numero|num\\S+ro|telephone|t\\S+l\\S+phone)\\s+",
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

    def _spell_postal_code(self, address: str) -> Optional[str]:
        match = _CA_POSTAL_RE.search(address or "")
        if not match:
            return None
        code = (match.group(1) + match.group(2)).replace(" ", "")
        spelled = self._spell_alnum(code)
        return spelled or None

    def _spell_first_number(self, text: str) -> Optional[str]:
        match = re.search(r"\\b(\\d{1,6})\\b", text or "")
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

        language = self._lang_state.current

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
                    r"^(my name is|this is|it's|it is)\\b|^(je m'appelle|mon nom est|c'est)\\b",
                    normalized,
                )
            )
            mentioned_address = bool(
                _CA_POSTAL_RE.search(transcript)
                or re.search(r"\\b(address|adresse)\\b", normalized)
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
            spelled = self._spell_alnum(self._customer_name)
            confirm = (
                f"Super ! J'ai noté {self._customer_name}. Je l'épelle: {spelled}. C'est bien ça ?"
                if language == "fr"
                else f"Awesome! I got {self._customer_name}. That's {spelled}. Did I spell that right?"
            )
            await self._speak_response(confirm)
            return True

        if self._checkout_phase == CheckoutPhase.NAME:
            name = self._extract_name(transcript)
            if not name:
                retry = (
                    "Désolé, je n’ai pas bien compris. Quel est votre nom ?"
                    if language == "fr"
                    else "Sorry—I didn’t catch that. What’s your name?"
                )
                await self._speak_response(retry)
                return True

            self._customer_name = name
            spelled = self._spell_alnum(name)
            confirm = (
                f"Super ! J’ai noté {name}. Je l’épelle: {spelled}. C’est bien ça ?"
                if language == "fr"
                else f"Awesome! I got {name}. That’s {spelled}. Did I spell that right?"
            )
            self._checkout_phase = CheckoutPhase.NAME_CONFIRM
            await self._speak_response(confirm)
            return True

        if self._checkout_phase == CheckoutPhase.NAME_CONFIRM:
            if self._is_affirmative(transcript):
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

            if self._is_negative(transcript):
                self._checkout_phase = CheckoutPhase.NAME
                ask = (
                    "Oups, d’accord. Pouvez-vous répéter votre nom lentement ?"
                    if language == "fr"
                    else "Oops—okay. Can you repeat your name slowly?"
                )
                await self._speak_response(ask)
                return True

            reprompt = (
                "Juste pour confirmer: est-ce que l’orthographe est correcte ? Oui ou non ?"
                if language == "fr"
                else "Quick check—did I spell your name correctly? Yes or no?"
            )
            await self._speak_response(reprompt)
            return True

        if self._checkout_phase == CheckoutPhase.ADDRESS:
            address = self._extract_address(transcript)
            if not address:
                retry = (
                    "Désolé, je n’ai pas compris l’adresse. Pouvez-vous la répéter ?"
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
            if self._is_affirmative(transcript):
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

            if self._is_negative(transcript):
                self._checkout_phase = CheckoutPhase.ADDRESS
                ask = (
                    "Oups, d’accord. Pouvez-vous répéter l’adresse lentement ?"
                    if language == "fr"
                    else "Oops—okay. Can you repeat the address slowly?"
                )
                await self._speak_response(ask)
                return True

            reprompt = (
                "Juste pour confirmer: est-ce que l’adresse est correcte ? Oui ou non ?"
                if language == "fr"
                else "Quick check—is that address correct? Yes or no?"
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

        await self._speak_response(confirmation)
    
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
                "queue_depth_max": self._queue_depth_max,
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
            self._last_speech_time = time.time()
            
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
        
        if self._state == PipelineState.SPEAKING:
            self._speech_started_during_tts = True
            self._begin_soft_interrupt(reason="vad")
            logger.debug("Speech started detected during TTS")

    async def _on_transcript(self, result: TranscriptionResult) -> None:
        """
        Handle STT transcript.
        
        Only processes final transcripts. Checks for interruption on partials.
        """
        if not self._is_running:
            return

        if result.text and result.text.strip():
            self._last_stt_text_time = time.time()

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

                if not self._tts_soft_paused and (self._speech_started_during_tts or word_count >= 1):
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
            
            logger.info(
                "Final transcript",
                text=transcript[:100],
                speech_final=result.speech_final,
            )
            await self._turn_queue.put(
                QueuedTranscript(
                    text=transcript,
                    stt_latency_ms=result.latency_ms,
                    detected_language=result.detected_language,
                    language_confidence=result.language_confidence,
                )
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
            switched=switched,
            reason=reason or None,
            previous_language=previous_language,
            detection_source=detection_source,
        )

        # Menu-only checkout flow: name/address confirmation (with spelling).
        if await self._handle_checkout_flow(transcript):
            return
        
        # Try semantic routing first
        if (
            not self.config.menu_only
            and self._lang_state.current == "en"
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
        await self._generate_llm_response(transcript)
    
    async def _generate_llm_response(self, transcript: str) -> None:
        """Generate and speak an LLM response."""
        if not self._llm:
            logger.error("LLM not initialized")
            return
        
        llm_start = time.time()
        first_token_time = None
        full_response = ""
        
        extra_context = self._build_extra_context(transcript)

        # Collect LLM response (keep processing state until we actually speak)
        try:
            self._state = PipelineState.PROCESSING
            
            async for chunk in self._llm.generate_streaming(
                transcript,
                target_language=self._lang_state.current,
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
                self._current_turn_metrics.llm_total_ms = (llm_end - llm_start) * 1000
            
            # Speak the response if not interrupted
            if self._state != PipelineState.INTERRUPTED and full_response:
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

    def _build_extra_context(self, transcript: str) -> Optional[str]:
        """
        Build optional extra system context for the LLM.

        Currently adds menu context when the caller asks about the menu, prices,
        or mentions a menu item.
        """
        catalog = self._menu_catalog or get_menu_catalog()
        if not catalog:
            return None

        language = self._lang_state.current
        wants_menu = looks_like_menu_request(transcript, language=language)
        matches = find_menu_items(catalog, transcript, limit=10) if not wants_menu else []

        if language == "fr":
            header = (
                "CONTEXTE MENU (source: menu.html)\n"
                "RÈGLES:\n"
                "- Utilise uniquement les articles et prix listés ici.\n"
                "- Aide l'appelant à choisir et à commander : demande la quantité, confirme ce que tu as noté, puis demande s'il veut autre chose.\n"
                "- Quand l'appelant dit que c'est tout : passe au checkout (nom, adresse, téléphone, email). Confirme chaque info en la répétant/épelant puis demande si c'est correct.\n"
                "- Adresse: épelle au moins le numéro et le code postal. Téléphone: répète les chiffres. Email: répète lentement et, si besoin, demande de l'épeler.\n"
                "- Ne confirme la commande (prise/confirmée) qu'après avoir obtenu ET fait confirmer: nom, adresse, téléphone et email.\n"
                "- Si l'appelant demande autre chose que le menu: réponds brièvement avec empathie, puis ramène doucement au menu.\n"
                "- Réponses courtes, 1 question à la fois.\n"
            )
        else:
            header = (
                "MENU CONTEXT (source: menu.html)\n"
                "RULES:\n"
                "- Use only the items and prices listed here.\n"
                "- Help the caller choose and place a (simulated) order: ask quantity, confirm what you recorded, then ask if they want anything else.\n"
                "- When the caller says they're done: move to checkout (name, address, phone number, email). Confirm each by repeating/spelling it back and asking if it's correct.\n"
                "- Address: spell at least the street number and postal code. Phone: repeat digits. Email: repeat slowly and ask them to spell it if needed.\n"
                "- Do not confirm the order as taken/confirmed until you have AND confirmed: name, address, phone number, and email.\n"
                "- If the caller asks about non-menu topics: respond briefly with empathy, then gently steer back to the menu.\n"
                "- Keep it short and ask 1 question at a time.\n"
            )

        if self.config.menu_only:
            menu_lines = "\n".join(catalog.to_prompt_lines())
            logger.info("Menu context included", reason="menu_only", current_language=language)
            return f"{header}\n{menu_lines}"

        if not wants_menu and not matches:
            return None

        if wants_menu:
            menu_lines = "\n".join(catalog.to_prompt_lines())
            logger.info("Menu context included", reason="menu_request", current_language=language)
            return f"{header}\n{menu_lines}"

        # Only provide relevant matches to keep context small.
        match_lines: List[str] = []
        for item in matches:
            match_lines.append(f"- {item.section}: {item.name} — {item.price_text}")

        logger.info(
            "Menu context included",
            reason="menu_item_match",
            current_language=language,
            num_matches=len(matches),
        )
        return f"{header}\nMATCHES:\n" + "\n".join(match_lines)

    async def _run_turn(self, queued: QueuedTranscript) -> None:
        """Run a single conversation turn from a final transcript."""
        self._start_turn()
        if self._current_turn_metrics:
            self._current_turn_metrics.stt_ms = queued.stt_latency_ms
        
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
        
        # Pre-buffer to reduce bursty producer/consumer underruns
        prebuffer_frames = max(3, int(self._jitter_buffer_ms / 20))
        buffered = 0
        frames_buffer = []
        
        try:
            # Pre-buffer phase
            while buffered < prebuffer_frames and self._pacer_running:
                try:
                    outbound = await asyncio.wait_for(
                        self._outbound_queue.get(),
                        timeout=0.5
                    )
                    if outbound is None:
                        break
                    frames_buffer.append(outbound)
                    if outbound.pace:
                        buffered += 1
                except asyncio.TimeoutError:
                    break
            
            # Main pacing loop
            while self._pacer_running and self._is_running:
                try:
                    # Get frame from buffer or queue
                    if frames_buffer:
                        outbound = frames_buffer.pop(0)
                    else:
                        # Non-blocking get to maintain timing
                        try:
                            outbound = self._outbound_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            # No frame available, send silence to maintain timing
                            outbound = None
                    
                    if outbound is None:
                        # Check for new audio with small timeout
                        try:
                            outbound = await asyncio.wait_for(
                                self._outbound_queue.get(),
                                timeout=0.015  # Most of the 20ms window
                            )
                        except asyncio.TimeoutError:
                            # Still no audio, continue loop
                            await asyncio.sleep(0.005)
                            continue
                    
                    if outbound is None:  # Poison pill
                        break

                    if not outbound.pace:
                        # Control frames (mark/clear/etc.) should be sent immediately and must not
                        # consume an audio pacing slot.
                        await self._send_message(outbound.message)
                        continue
                    
                    # Wait until next send time for precise 20ms intervals
                    now = time.time()
                    wait_time = next_send_time - now
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    
                    # Send frame
                    await self._send_message(outbound.message)
                    
                    # Calculate next send time (exactly 20ms later)
                    next_send_time += frame_duration
                    
                    # If we're falling behind, reset timing
                    if time.time() > next_send_time + 0.040:  # More than 2 frames behind
                        self._pacer_underruns += 1
                        if self._pacer_underruns % 10 == 0:
                            logger.warning(
                                "Audio pacer reset timing", 
                                underruns=self._pacer_underruns,
                                behind_ms=(time.time() - next_send_time)*1000
                            )
                        next_send_time = time.time()
                        
                except asyncio.TimeoutError:
                    continue  # No audio available, keep waiting
                    
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
    
    async def _speak_response(self, text: str) -> None:
        """
        Synthesize and stream TTS response to Twilio via audio pacer.
        
        Late-2025 best practice: Queue audio frames to pacer for smooth 20ms timing.
        """
        if not text or not self._is_running:
            return

        # Avoid talking over the caller when STT is still producing interim text.
        await self._wait_for_stt_quiet()
        
        tts_start = time.time()
        first_audio_time = None
        
        self._state = PipelineState.SPEAKING
        
        # Start pacer if not running
        await self._start_pacer()
        
        try:
            # Reset streaming frame remainder for this utterance.
            # Important: Do NOT pad to 160 bytes per TTS chunk; only pad once at the end,
            # otherwise we insert silence between streamed chunks which sounds like "ticking".
            self._outbound_ulaw_remainder = b""

            voice_id = self._current_cartesia_voice_id()

            # Best practice: Insert marks at sentence boundaries to track real playback and
            # make interruptions feel clean.
            for segment in self._split_sentences(text):
                if not self._is_running or self._state == PipelineState.INTERRUPTED:
                    break

                async for chunk in self._tts.synthesize_streaming(segment, voice_id=voice_id):
                    if not self._is_running or self._state == PipelineState.INTERRUPTED:
                        break

                    if chunk.audio_bytes:
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
                            msg = create_media_message(self._protocol.stream_sid, frame)
                            await self._enqueue_outbound(msg, pace=True)

                    if (
                        chunk.is_final
                        and self._outbound_ulaw_remainder
                        and not self._tts_soft_paused
                    ):
                        frame = self._outbound_ulaw_remainder.ljust(TWILIO_FRAME_SIZE, b"\xff")
                        self._outbound_ulaw_remainder = b""
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
                        msg = create_media_message(self._protocol.stream_sid, frame)
                        await self._enqueue_outbound(msg, pace=True)

                    if self._outbound_ulaw_remainder:
                        frame = self._outbound_ulaw_remainder.ljust(TWILIO_FRAME_SIZE, b"\xff")
                        self._outbound_ulaw_remainder = b""
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
        except Exception as e:
            logger.error("TTS streaming failed", error=str(e))
        finally:
            if self._state == PipelineState.SPEAKING:
                self._state = PipelineState.LISTENING
    
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
                
                silence_duration = time.time() - self._last_speech_time
                
                if silence_duration > self.config.silence_timeout_seconds * 3:
                    # Extended silence - prompt user
                    logger.debug("Extended silence detected")
                    prompt = (
                        "Vous êtes toujours là ? Est-ce que je peux vous aider avec autre chose ?"
                        if self._lang_state.current == "fr"
                        else "Are you still there? Is there anything else I can help you with?"
                    )
                    await self._speak_response(prompt)
                    self._last_speech_time = time.time()
                    
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
