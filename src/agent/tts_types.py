from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TTSChunk:
    """
    A chunk of synthesized audio.

    `audio_bytes` is Twilio-ready mu-law (8kHz) unless otherwise noted.
    """

    audio_bytes: bytes
    is_final: bool = False
    timestamp: float = field(default_factory=time.time)

    # Optional: raw 24kHz mono WAV bytes (16-bit PCM) for contextual TTS history.
    audio_wav_24k: Optional[bytes] = None

    # Optional: structured metadata for debugging/metrics.
    meta: Optional[dict[str, Any]] = None

