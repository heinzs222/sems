"""
Audio conversion utilities for Twilio Voice Agent.

Late-2025 Best Practice:
- Twilio mu-law 8kHz sent DIRECTLY to Deepgram (no conversion needed)
- Cartesia outputs PCM 8kHz directly (no resampling)
- Only conversion: PCM 8kHz -> mu-law 8kHz (fast, uses audioop)

This eliminates CPU-heavy resampling that causes choppy audio.
"""

import audioop
import io
import wave
from typing import Generator, List, Optional
import struct

TWILIO_SAMPLE_RATE = 8000
STT_SAMPLE_RATE = 8000  # Deepgram accepts mulaw/8000 directly - no resampling needed!
TTS_SAMPLE_RATE = 8000  # Request Cartesia to output 8kHz directly
FRAME_DURATION_MS = 20
TWILIO_FRAME_SIZE = int(TWILIO_SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 160 bytes for 20ms
FLUX_FRAME_DURATION_MS = 80  # Deepgram Flux recommends ~80ms chunks
FLUX_FRAME_SIZE = int(TWILIO_SAMPLE_RATE * FLUX_FRAME_DURATION_MS / 1000)  # 640 bytes for 80ms


def ulaw_to_linear16(ulaw_bytes: bytes) -> bytes:
    """
    Convert mu-law 8kHz audio to linear PCM 16-bit.
    
    Args:
        ulaw_bytes: Raw mu-law encoded bytes at 8kHz
        
    Returns:
        Linear PCM 16-bit bytes at 8kHz
    """
    if not ulaw_bytes:
        return b""
    
    return audioop.ulaw2lin(ulaw_bytes, 2)


def linear16_to_ulaw(pcm_bytes: bytes) -> bytes:
    """
    Convert linear PCM 16-bit to mu-law.
    
    Args:
        pcm_bytes: Linear PCM 16-bit bytes
        
    Returns:
        Mu-law encoded bytes
    """
    if not pcm_bytes:
        return b""
    
    return audioop.lin2ulaw(pcm_bytes, 2)


def resample_8k_to_16k(pcm_8k: bytes) -> bytes:
    """
    Resample linear PCM from 8kHz to 16kHz using audioop.
    
    Args:
        pcm_8k: Linear PCM 16-bit bytes at 8kHz
        
    Returns:
        Linear PCM 16-bit bytes at 16kHz
    """
    if not pcm_8k:
        return b""
    
    # Use audioop.ratecv for resampling (much faster than scipy)
    # ratecv returns (converted_data, new_state)
    converted, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
    return converted


def resample_16k_to_8k(pcm_16k: bytes) -> bytes:
    """
    Resample linear PCM from 16kHz to 8kHz using audioop.
    
    Args:
        pcm_16k: Linear PCM 16-bit bytes at 16kHz
        
    Returns:
        Linear PCM 16-bit bytes at 8kHz
    """
    if not pcm_16k:
        return b""
    
    # Use audioop.ratecv for resampling
    converted, _ = audioop.ratecv(pcm_16k, 2, 1, 16000, 8000, None)
    return converted


def resample_to_8k(pcm_bytes: bytes, source_rate: int) -> bytes:
    """
    Resample linear PCM from any sample rate to 8kHz using audioop.
    
    Args:
        pcm_bytes: Linear PCM 16-bit bytes
        source_rate: Source sample rate in Hz
        
    Returns:
        Linear PCM 16-bit bytes at 8kHz
    """
    if not pcm_bytes or source_rate == TWILIO_SAMPLE_RATE:
        return pcm_bytes
    
    # Use audioop.ratecv for fast resampling
    converted, _ = audioop.ratecv(pcm_bytes, 2, 1, source_rate, TWILIO_SAMPLE_RATE, None)
    return converted


def twilio_ulaw_to_stt_pcm(ulaw_bytes: bytes) -> bytes:
    """
    DEPRECATED: Use twilio_ulaw_passthrough() instead.
    
    Late-2025 best practice: Send mu-law directly to Deepgram with encoding=mulaw.
    This function is kept for backwards compatibility only.
    """
    if not ulaw_bytes:
        return b""
    pcm_8k = ulaw_to_linear16(ulaw_bytes)
    pcm_16k = resample_8k_to_16k(pcm_8k)
    return pcm_16k


def twilio_ulaw_passthrough(ulaw_bytes: bytes) -> bytes:
    """
    Pass through Twilio mu-law audio unchanged for Deepgram.
    
    Late-2025 best practice: Deepgram accepts mulaw/8000 directly.
    No conversion needed - eliminates CPU-heavy resampling!
    
    Args:
        ulaw_bytes: Raw mu-law bytes from Twilio (8kHz)
        
    Returns:
        Same mu-law bytes (no conversion)
    """
    return ulaw_bytes


def tts_pcm_to_twilio_ulaw(
    pcm_bytes: bytes,
    source_rate: int = 8000,
    source_width: int = 2
) -> bytes:
    """
    Convert TTS PCM output to Twilio mu-law format.
    
    Late-2025 best practice: Request Cartesia to output 8kHz PCM directly.
    Then only PCM->mulaw conversion is needed (fast, uses audioop).
    
    Args:
        pcm_bytes: PCM bytes from TTS (should be 8kHz for optimal performance)
        source_rate: TTS output sample rate (8000 recommended to avoid resampling)
        source_width: Sample width in bytes (2 for 16-bit)
        
    Returns:
        Mu-law bytes at 8kHz for Twilio
    """
    if not pcm_bytes:
        return b""
    
    # Handle different bit depths
    if source_width == 1:
        pcm_bytes = audioop.lin2lin(pcm_bytes, 1, 2)
    elif source_width == 4:
        # 32-bit float to 16-bit int
        import numpy as np  # Local import (optional code path)
        samples = np.frombuffer(pcm_bytes, dtype=np.float32)
        samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        pcm_bytes = samples.tobytes()
    
    # Only resample if not already 8kHz (should be avoided in production)
    if source_rate != TWILIO_SAMPLE_RATE:
        # Log warning - this causes choppy audio!
        import structlog
        structlog.get_logger(__name__).warning(
            "Resampling TTS audio - causes choppy playback!",
            source_rate=source_rate,
            target_rate=TWILIO_SAMPLE_RATE
        )
        pcm_bytes = resample_to_8k(pcm_bytes, source_rate)
    
    # Fast PCM->mulaw conversion using audioop
    return audioop.lin2ulaw(pcm_bytes, 2)


def pcm_8k_to_ulaw(pcm_bytes: bytes) -> bytes:
    """
    Fast PCM 8kHz to mu-law conversion (no resampling).
    
    This is the ONLY conversion needed when Cartesia outputs 8kHz directly.
    Uses Python's audioop which is highly optimized.
    
    Args:
        pcm_bytes: Linear PCM 16-bit bytes at 8kHz
        
    Returns:
        Mu-law bytes at 8kHz
    """
    if not pcm_bytes:
        return b""
    return audioop.lin2ulaw(pcm_bytes, 2)


# Alias for compatibility with CSM TTS
pcm_to_ulaw = pcm_8k_to_ulaw


def resample_audio(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """
    Resample linear PCM audio between any two sample rates.
    
    Used by CSM TTS to convert 24kHz output to 8kHz for Twilio.
    
    Args:
        pcm_bytes: Linear PCM 16-bit bytes
        from_rate: Source sample rate in Hz
        to_rate: Target sample rate in Hz
        
    Returns:
        Resampled PCM 16-bit bytes
    """
    if not pcm_bytes or from_rate == to_rate:
        return pcm_bytes
    
    converted, _ = audioop.ratecv(pcm_bytes, 2, 1, from_rate, to_rate, None)
    return converted


def chunk_audio(audio_bytes: bytes, chunk_size: int = TWILIO_FRAME_SIZE) -> Generator[bytes, None, None]:
    """
    Chunk audio into fixed-size frames.
    
    For Twilio, we want 20ms frames = 160 bytes of mu-law at 8kHz.
    
    Args:
        audio_bytes: Raw audio bytes
        chunk_size: Size of each chunk in bytes (default: 160 for 20ms mu-law)
        
    Yields:
        Audio chunks of the specified size
    """
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        # Pad the last chunk if needed
        if len(chunk) < chunk_size:
            chunk = chunk + b'\xff' * (chunk_size - len(chunk))  # 0xFF is silence in mu-law
        yield chunk


def chunk_audio_list(audio_bytes: bytes, chunk_size: int = TWILIO_FRAME_SIZE) -> List[bytes]:
    """
    Chunk audio into fixed-size frames and return as a list.
    
    Args:
        audio_bytes: Raw audio bytes
        chunk_size: Size of each chunk in bytes
        
    Returns:
        List of audio chunks
    """
    return list(chunk_audio(audio_bytes, chunk_size))


def get_audio_duration_ms(audio_bytes: bytes, sample_rate: int = TWILIO_SAMPLE_RATE, is_ulaw: bool = True) -> float:
    """
    Calculate the duration of audio in milliseconds.
    
    Args:
        audio_bytes: Audio bytes
        sample_rate: Sample rate in Hz
        is_ulaw: Whether the audio is mu-law (1 byte per sample) or PCM (2 bytes per sample)
        
    Returns:
        Duration in milliseconds
    """
    if not audio_bytes:
        return 0.0
    
    bytes_per_sample = 1 if is_ulaw else 2
    num_samples = len(audio_bytes) // bytes_per_sample
    duration_seconds = num_samples / sample_rate
    
    return duration_seconds * 1000


def create_silence_ulaw(duration_ms: int) -> bytes:
    """
    Create silence in mu-law format.
    
    Args:
        duration_ms: Duration of silence in milliseconds
        
    Returns:
        Mu-law silence bytes
    """
    num_samples = int(TWILIO_SAMPLE_RATE * duration_ms / 1000)
    # 0xFF is the mu-law encoding for silence (0 amplitude)
    return b'\xff' * num_samples


def normalize_audio(pcm_bytes: bytes, target_db: float = -3.0) -> bytes:
    """
    Normalize audio to a target dB level.
    
    Args:
        pcm_bytes: Linear PCM 16-bit bytes
        target_db: Target dB level (relative to max)
        
    Returns:
        Normalized PCM bytes
    """
    if not pcm_bytes:
        return b""

    import numpy as np  # Local import (optional utility)
    
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    
    if len(samples) == 0:
        return b""
    
    # Calculate current max
    max_val = np.max(np.abs(samples))
    
    if max_val == 0:
        return pcm_bytes
    
    # Calculate target max
    target_max = 32767 * (10 ** (target_db / 20))
    
    # Scale
    scale = target_max / max_val
    normalized = samples * scale
    
    # Clip and convert back
    normalized = np.clip(normalized, -32768, 32767).astype(np.int16)
    
    return normalized.tobytes()


def resample_pcm16(pcm_bytes: bytes, source_rate: int, target_rate: int) -> bytes:
    """
    Resample mono 16-bit PCM from `source_rate` to `target_rate` using `audioop.ratecv`.
    """
    if not pcm_bytes or source_rate == target_rate:
        return pcm_bytes
    converted, _ = audioop.ratecv(pcm_bytes, 2, 1, int(source_rate), int(target_rate), None)
    return converted


def read_wav_mono_pcm16(wav_bytes: bytes) -> tuple[int, bytes]:
    """
    Read a WAV byte string and return (sample_rate, mono PCM16 bytes).

    - If input is stereo, it is downmixed to mono.
    - If sample width is not 16-bit, raises ValueError.
    """
    if not wav_bytes:
        raise ValueError("Empty WAV")

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width * 8} bits")

    if channels == 1:
        return int(sample_rate), frames

    if channels == 2:
        mono = audioop.tomono(frames, 2, 0.5, 0.5)
        return int(sample_rate), mono

    raise ValueError(f"Unsupported WAV channel count: {channels}")


def write_wav_mono_pcm16(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Create a mono 16-bit PCM WAV byte string from PCM bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_bytes or b"")
    return buf.getvalue()


def trim_wav_bytes(wav_bytes: bytes, *, max_seconds: float) -> bytes:
    """Trim a mono/stereo PCM16 WAV byte string to at most `max_seconds`."""
    if not wav_bytes:
        return wav_bytes
    if max_seconds <= 0:
        return write_wav_mono_pcm16(b"", 24000)

    sr, pcm = read_wav_mono_pcm16(wav_bytes)
    max_samples = int(sr * max_seconds)
    max_bytes = max_samples * 2
    if len(pcm) <= max_bytes:
        return write_wav_mono_pcm16(pcm, sr)
    return write_wav_mono_pcm16(pcm[:max_bytes], sr)


def ulaw_8k_to_wav_24k(ulaw_bytes: bytes, *, max_seconds: Optional[float] = None) -> bytes:
    """
    Convert Twilio 8kHz mu-law bytes to a 24kHz mono PCM16 WAV byte string.

    This is used for contextual TTS history; it is not played directly to Twilio.
    """
    if not ulaw_bytes:
        return write_wav_mono_pcm16(b"", 24000)

    if max_seconds is not None and max_seconds > 0:
        max_bytes = int(8000 * max_seconds)
        ulaw_bytes = ulaw_bytes[-max_bytes:]

    pcm_8k = ulaw_to_linear16(ulaw_bytes)
    pcm_24k = resample_pcm16(pcm_8k, 8000, 24000)
    return write_wav_mono_pcm16(pcm_24k, 24000)


def wav_bytes_to_twilio_ulaw(wav_bytes: bytes) -> bytes:
    """
    Convert a (24kHz recommended) PCM16 WAV byte string into Twilio 8kHz mu-law bytes.

    This is the bridge for contextual TTS models that output high-rate WAV audio.
    """
    sr, pcm = read_wav_mono_pcm16(wav_bytes)
    pcm_8k = resample_pcm16(pcm, sr, 8000)
    return linear16_to_ulaw(pcm_8k)
