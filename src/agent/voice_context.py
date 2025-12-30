from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Optional

from src.agent.audio import (
    read_wav_mono_pcm16,
    resample_pcm16,
    trim_wav_bytes,
    ulaw_8k_to_wav_24k,
    write_wav_mono_pcm16,
)


def _wav_duration_seconds(wav_bytes: bytes) -> float:
    if not wav_bytes:
        return 0.0
    sample_rate, pcm = read_wav_mono_pcm16(wav_bytes)
    samples = len(pcm) // 2
    if sample_rate <= 0:
        return 0.0
    return samples / float(sample_rate)


def _normalize_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in ("assistant", "agent", "bot", "0"):
        return "assistant"
    return "user"


def _role_to_speaker_id(role: str) -> int:
    return 0 if _normalize_role(role) == "assistant" else 1


@dataclass
class ContextTurn:
    role: str
    text: str
    audio_wav_24k: Optional[bytes] = None
    created_at: float = field(default_factory=time.time)
    duration_s: float = 0.0

    def to_payload(self) -> dict:
        payload: dict = {
            "role": str(_role_to_speaker_id(self.role)),
            "text": self.text or "",
        }
        if self.audio_wav_24k:
            payload["audio_base64_wav_24khz"] = base64.b64encode(self.audio_wav_24k).decode(
                "ascii"
            )
        return payload


class VoiceContextBuffer:
    """
    Rolling conversation context for contextual TTS.

    Stores short per-turn audio snippets as 24kHz mono PCM16 WAV (bytes) plus text.
    Drops old turns to keep total audio under `max_context_seconds`.
    """

    def __init__(self, *, max_context_seconds: float = 24.0):
        self.max_context_seconds = max(0.0, float(max_context_seconds or 0.0))
        self._turns: list[ContextTurn] = []

    @property
    def turns(self) -> list[ContextTurn]:
        return list(self._turns)

    def _snippet_cap_seconds(self) -> float:
        # Keep a per-snippet cap in the 3-6s range and derived from the total window.
        if self.max_context_seconds <= 0:
            return 0.0
        cap = self.max_context_seconds / 4.0
        return float(min(6.0, max(3.0, cap)))

    def add_turn(
        self,
        *,
        role: str,
        text: str,
        audio_ulaw_8k: Optional[bytes] = None,
        audio_wav_24k: Optional[bytes] = None,
    ) -> None:
        if not (text or "").strip() and not audio_ulaw_8k and not audio_wav_24k:
            return

        snippet_cap = self._snippet_cap_seconds()
        wav_bytes: Optional[bytes] = None

        if audio_wav_24k:
            sr, pcm = read_wav_mono_pcm16(audio_wav_24k)
            if sr != 24000:
                pcm = resample_pcm16(pcm, sr, 24000)
                audio_wav_24k = write_wav_mono_pcm16(pcm, 24000)
            wav_bytes = trim_wav_bytes(audio_wav_24k, max_seconds=snippet_cap)
        elif audio_ulaw_8k:
            wav_bytes = ulaw_8k_to_wav_24k(audio_ulaw_8k, max_seconds=snippet_cap)

        duration_s = _wav_duration_seconds(wav_bytes) if wav_bytes else 0.0
        turn = ContextTurn(role=_normalize_role(role), text=text or "", audio_wav_24k=wav_bytes)
        turn.duration_s = duration_s

        self._turns.append(turn)
        self._trim()

    def _trim(self) -> None:
        if self.max_context_seconds <= 0:
            self._turns = self._turns[-2:]
            return

        # Cap context length and total audio seconds.
        # Prefer keeping the most recent turns.
        max_turns = 12
        if len(self._turns) > max_turns:
            self._turns = self._turns[-max_turns:]

        total = 0.0
        kept: list[ContextTurn] = []
        for turn in reversed(self._turns):
            total += turn.duration_s
            kept.append(turn)
            if total >= self.max_context_seconds:
                break
        self._turns = list(reversed(kept))

    def build_payload(self) -> list[dict]:
        return [t.to_payload() for t in self._turns]
