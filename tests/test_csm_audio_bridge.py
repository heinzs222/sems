"""
Regression tests for the contextual TTS audio bridge:

24kHz PCM16 WAV -> 8kHz mu-law -> 20ms framing.
"""

from __future__ import annotations

from src.agent.audio import TWILIO_FRAME_SIZE, wav_bytes_to_twilio_ulaw, write_wav_mono_pcm16


def _chunk_20ms_frames(ulaw_8k: bytes) -> list[bytes]:
    frames: list[bytes] = []
    remainder = ulaw_8k or b""
    while len(remainder) >= TWILIO_FRAME_SIZE:
        frames.append(remainder[:TWILIO_FRAME_SIZE])
        remainder = remainder[TWILIO_FRAME_SIZE:]
    if remainder:
        frames.append(remainder.ljust(TWILIO_FRAME_SIZE, b"\xff"))
    return frames


def test_wav_24k_to_ulaw_8k_and_20ms_frames() -> None:
    # 1s of silence at 24kHz mono PCM16.
    pcm_24k = b"\x00\x00" * 24000
    wav_24k = write_wav_mono_pcm16(pcm_24k, 24000)

    ulaw = wav_bytes_to_twilio_ulaw(wav_24k)
    assert len(ulaw) == 8000  # 1 second at 8kHz, 1 byte/sample
    assert ulaw[:200] == b"\xff" * 200  # silence in mu-law

    frames = _chunk_20ms_frames(ulaw)
    assert len(frames) == 50  # 1000ms / 20ms
    assert all(len(f) == TWILIO_FRAME_SIZE for f in frames)

