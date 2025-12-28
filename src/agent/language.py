"""
Language utilities for the voice agent.

Provides a small, deterministic layer to detect explicit user requests to switch
the conversation language (English <-> French).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Literal, Optional

LanguageCode = Literal["en", "fr"]


def _normalize_for_matching(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("’", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


_FRENCH_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\bspeak\s+french\b",
        r"\btalk\s+in\s+french\b",
        r"\bin\s+french\b",
        r"\bswitch\s+to\s+french\b",
        r"\bparle(?:z|r)?\s+francais\b",
        r"\ben\s+francais\b",
        r"\bpasse\s+en\s+francais\b",
    )
)

_ENGLISH_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\bspeak\s+english\b",
        r"\btalk\s+in\s+english\b",
        r"\bin\s+english\b",
        r"\bswitch\s+to\s+english\b",
        r"\bparle(?:z|r)?\s+anglais\b",
        r"\ben\s+anglais\b",
        r"\bpasse\s+en\s+anglais\b",
    )
)


def detect_language_switch_command(text: str) -> Optional[LanguageCode]:
    """
    Detect an explicit user request to switch the conversation language.

    Returns:
        "fr" if the user asked to switch to French
        "en" if the user asked to switch to English
        None otherwise
    """
    normalized = _normalize_for_matching(text)

    for pattern in _FRENCH_PATTERNS:
        if pattern.search(normalized):
            return "fr"

    for pattern in _ENGLISH_PATTERNS:
        if pattern.search(normalized):
            return "en"

    return None


_FRENCH_OVERRIDE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\bparle\s+francais\b",
        r"\ben\s+francais\b",
        r"\bfrancais\s+s'?il\s+te\s+plait\b",
        r"\bfrancais\s+s'?il\s+vous\s+plait\b",
    )
)

_ENGLISH_OVERRIDE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\bspeak\s+english\b",
        r"\bin\s+english\s+please\b",
        r"\benglish\s+please\b",
    )
)


def detect_language_override(text: str) -> Optional[LanguageCode]:
    """
    Detect an explicit override phrase for language switching.

    These overrides should always win, regardless of STT language confidence.
    """
    normalized = _normalize_for_matching(text)

    for pattern in _FRENCH_OVERRIDE_PATTERNS:
        if pattern.search(normalized):
            return "fr"

    for pattern in _ENGLISH_OVERRIDE_PATTERNS:
        if pattern.search(normalized):
            return "en"

    return None


def normalize_detected_language(detected_language: Optional[str]) -> Optional[LanguageCode]:
    """
    Normalize Deepgram detected_language into our internal LanguageCode ("en"/"fr").
    """
    if not detected_language:
        return None
    norm = detected_language.strip().lower()
    if norm.startswith("fr"):
        return "fr"
    if norm.startswith("en"):
        return "en"
    return None


@dataclass
class LangState:
    """
    Per-call language state with stabilization to prevent flapping.

    Rules (Late-2025 best practice, tuned for PSTN):
    - confidence threshold: 0.75
    - require 2 consecutive eligible finals to switch
    - ignore very short transcripts (<12 chars)
    - hold time 20s after switch to prevent flapping
    """

    current: LanguageCode = "en"
    pending: Optional[LanguageCode] = None
    pending_count: int = 0
    last_switch_ts: float = 0.0

    CONFIDENCE_SWITCH: float = 0.75
    CONSECUTIVE_FINALS_REQUIRED: int = 2
    MIN_CHARS: int = 12
    HOLD_TIME_SECONDS: float = 20.0

    def reset_candidate(self) -> None:
        self.pending = None
        self.pending_count = 0

    def force(self, target: LanguageCode, now: float) -> None:
        if target == self.current:
            return
        self.current = target
        self.last_switch_ts = now
        self.reset_candidate()

    def update_from_detection(
        self,
        *,
        text: str,
        detected_language: Optional[str],
        language_confidence: Optional[float],
        now: float,
    ) -> tuple[bool, str]:
        """
        Update state from STT language detection with stabilization.

        Returns:
            (switched, reason) where reason is "stabilized" when switched, else "".
        """
        if not text:
            return False, ""

        if len(text.strip()) < self.MIN_CHARS:
            return False, ""

        normalized = normalize_detected_language(detected_language)
        if not normalized:
            return False, ""

        try:
            confidence = float(language_confidence) if language_confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        if confidence is None or confidence < self.CONFIDENCE_SWITCH:
            return False, ""

        if normalized == self.current:
            self.reset_candidate()
            return False, ""

        if self.last_switch_ts > 0 and (now - self.last_switch_ts) < self.HOLD_TIME_SECONDS:
            return False, ""

        if self.pending != normalized:
            self.pending = normalized
            self.pending_count = 1
            return False, ""

        self.pending_count += 1
        if self.pending_count >= self.CONSECUTIVE_FINALS_REQUIRED:
            self.current = normalized
            self.last_switch_ts = now
            self.reset_candidate()
            return True, "stabilized"

        return False, ""
