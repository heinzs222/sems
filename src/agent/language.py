"""
Language utilities for the voice agent.

Provides a small, deterministic layer to detect explicit user requests to switch
the conversation language (English <-> French).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal, Optional

LanguageCode = Literal["en", "fr"]


def _normalize_for_matching(text: str) -> str:
    text = (text or "").strip().lower()
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

