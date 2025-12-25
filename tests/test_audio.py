"""
Tests for audio conversion utilities.
"""

import pytest
import numpy as np

from src.agent.audio import (
    ulaw_to_linear16,
    linear16_to_ulaw,
    resample_8k_to_16k,
    resample_16k_to_8k,
    twilio_ulaw_to_stt_pcm,
    tts_pcm_to_twilio_ulaw,
    chunk_audio,
    chunk_audio_list,
    get_audio_duration_ms,
    create_silence_ulaw,
    TWILIO_FRAME_SIZE,
    TWILIO_SAMPLE_RATE,
    STT_SAMPLE_RATE,
)


class TestUlawConversion:
    """Tests for mu-law conversion."""
    
    def test_ulaw_to_linear16_empty(self):
        """Test conversion of empty bytes."""
        result = ulaw_to_linear16(b"")
        assert result == b""
    
    def test_ulaw_to_linear16_basic(self):
        """Test basic mu-law to linear conversion."""
        # Create some mu-law bytes (silence = 0xFF)
        ulaw_bytes = b"\xff" * 100
        result = ulaw_to_linear16(ulaw_bytes)
        
        # Output should be 2x length (16-bit = 2 bytes per sample)
        assert len(result) == 200
        
        # Convert to samples and check they're near zero (silence)
        samples = np.frombuffer(result, dtype=np.int16)
        assert np.abs(samples).max() < 10  # Near zero
    
    def test_linear16_to_ulaw_empty(self):
        """Test conversion of empty bytes."""
        result = linear16_to_ulaw(b"")
        assert result == b""
    
    def test_linear16_to_ulaw_basic(self):
        """Test basic linear to mu-law conversion."""
        # Create silence in linear16
        pcm_bytes = b"\x00\x00" * 100
        result = linear16_to_ulaw(pcm_bytes)
        
        # Output should be half length (1 byte per sample)
        assert len(result) == 100
    
    def test_roundtrip_conversion(self):
        """Test that conversion is reversible (approximately)."""
        # Create a simple tone
        samples = np.sin(np.linspace(0, 4 * np.pi, 100)) * 16000
        original_pcm = samples.astype(np.int16).tobytes()
        
        # Convert to mu-law and back
        ulaw = linear16_to_ulaw(original_pcm)
        recovered_pcm = ulaw_to_linear16(ulaw)
        
        # Should be approximately equal (mu-law is lossy)
        original_samples = np.frombuffer(original_pcm, dtype=np.int16)
        recovered_samples = np.frombuffer(recovered_pcm, dtype=np.int16)
        
        # Check correlation is high
        correlation = np.corrcoef(original_samples, recovered_samples)[0, 1]
        assert correlation > 0.99


class TestResampling:
    """Tests for audio resampling."""
    
    def test_resample_8k_to_16k_empty(self):
        """Test resampling empty bytes."""
        result = resample_8k_to_16k(b"")
        assert result == b""
    
    def test_resample_8k_to_16k_length(self):
        """Test that resampling doubles the length."""
        # 100 samples at 8kHz
        samples_8k = np.zeros(100, dtype=np.int16)
        pcm_8k = samples_8k.tobytes()
        
        result = resample_8k_to_16k(pcm_8k)
        
        # Should be approximately 200 samples at 16kHz
        samples_16k = np.frombuffer(result, dtype=np.int16)
        assert abs(len(samples_16k) - 200) <= 2  # Allow small rounding
    
    def test_resample_16k_to_8k_empty(self):
        """Test resampling empty bytes."""
        result = resample_16k_to_8k(b"")
        assert result == b""
    
    def test_resample_16k_to_8k_length(self):
        """Test that resampling halves the length."""
        # 200 samples at 16kHz
        samples_16k = np.zeros(200, dtype=np.int16)
        pcm_16k = samples_16k.tobytes()
        
        result = resample_16k_to_8k(pcm_16k)
        
        # Should be approximately 100 samples at 8kHz
        samples_8k = np.frombuffer(result, dtype=np.int16)
        assert abs(len(samples_8k) - 100) <= 2
    
    def test_resample_roundtrip(self):
        """Test that resampling roundtrip preserves signal."""
        # Create a simple signal
        t = np.linspace(0, 0.1, 800)  # 100ms at 8kHz
        samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        original = samples.tobytes()
        
        # Upsample then downsample
        upsampled = resample_8k_to_16k(original)
        downsampled = resample_16k_to_8k(upsampled)
        
        # Should be similar length
        original_samples = np.frombuffer(original, dtype=np.int16)
        final_samples = np.frombuffer(downsampled, dtype=np.int16)
        
        assert abs(len(final_samples) - len(original_samples)) <= 2


class TestTwilioConversion:
    """Tests for Twilio-specific conversion functions."""
    
    def test_twilio_ulaw_to_stt_pcm_empty(self):
        """Test conversion of empty input."""
        result = twilio_ulaw_to_stt_pcm(b"")
        assert result == b""
    
    def test_twilio_ulaw_to_stt_pcm_produces_16k(self):
        """Test that output is 16kHz."""
        # 160 bytes of mu-law = 20ms at 8kHz = 160 samples
        ulaw_input = b"\xff" * 160
        
        result = twilio_ulaw_to_stt_pcm(ulaw_input)
        
        # Should produce ~320 samples at 16kHz (20ms)
        samples = np.frombuffer(result, dtype=np.int16)
        assert abs(len(samples) - 320) <= 4
    
    def test_tts_pcm_to_twilio_ulaw_empty(self):
        """Test conversion of empty input."""
        result = tts_pcm_to_twilio_ulaw(b"")
        assert result == b""
    
    def test_tts_pcm_to_twilio_ulaw_produces_8k_ulaw(self):
        """Test that output is 8kHz mu-law."""
        # Create 24kHz PCM (typical TTS output)
        samples_24k = np.zeros(2400, dtype=np.int16)  # 100ms at 24kHz
        pcm_24k = samples_24k.tobytes()
        
        result = tts_pcm_to_twilio_ulaw(pcm_24k, source_rate=24000)
        
        # Should produce ~800 samples at 8kHz (100ms)
        # mu-law is 1 byte per sample
        assert abs(len(result) - 800) <= 10


class TestChunking:
    """Tests for audio chunking."""
    
    def test_chunk_audio_empty(self):
        """Test chunking empty bytes."""
        chunks = list(chunk_audio(b""))
        assert chunks == []
    
    def test_chunk_audio_exact_multiple(self):
        """Test chunking when length is exact multiple of chunk size."""
        # 320 bytes = 2 chunks of 160
        audio = b"\xff" * 320
        chunks = list(chunk_audio(audio, TWILIO_FRAME_SIZE))
        
        assert len(chunks) == 2
        assert all(len(c) == TWILIO_FRAME_SIZE for c in chunks)
    
    def test_chunk_audio_with_remainder(self):
        """Test chunking with remainder (should pad)."""
        # 200 bytes = 1 full chunk + 1 padded chunk
        audio = b"\xaa" * 200
        chunks = list(chunk_audio(audio, TWILIO_FRAME_SIZE))
        
        assert len(chunks) == 2
        assert len(chunks[0]) == TWILIO_FRAME_SIZE
        assert len(chunks[1]) == TWILIO_FRAME_SIZE
        # Last chunk should be padded with 0xFF (silence)
        assert chunks[1].endswith(b"\xff" * (TWILIO_FRAME_SIZE - 40))
    
    def test_chunk_audio_list(self):
        """Test chunk_audio_list returns a list."""
        audio = b"\xff" * 480
        chunks = chunk_audio_list(audio, TWILIO_FRAME_SIZE)
        
        assert isinstance(chunks, list)
        assert len(chunks) == 3


class TestUtilities:
    """Tests for utility functions."""
    
    def test_get_audio_duration_ms_ulaw(self):
        """Test duration calculation for mu-law."""
        # 8000 bytes = 1 second at 8kHz mu-law
        duration = get_audio_duration_ms(b"\xff" * 8000, is_ulaw=True)
        assert duration == 1000.0
    
    def test_get_audio_duration_ms_pcm(self):
        """Test duration calculation for PCM."""
        # 16000 bytes = 1 second at 8kHz PCM (2 bytes per sample)
        duration = get_audio_duration_ms(b"\x00" * 16000, sample_rate=8000, is_ulaw=False)
        assert duration == 1000.0
    
    def test_get_audio_duration_ms_empty(self):
        """Test duration of empty audio."""
        duration = get_audio_duration_ms(b"")
        assert duration == 0.0
    
    def test_create_silence_ulaw(self):
        """Test silence generation."""
        silence = create_silence_ulaw(100)  # 100ms
        
        # 100ms at 8kHz = 800 samples
        assert len(silence) == 800
        # All bytes should be 0xFF (mu-law silence)
        assert all(b == 0xFF for b in silence)
    
    def test_frame_size_is_20ms(self):
        """Verify TWILIO_FRAME_SIZE is correct for 20ms."""
        # 20ms at 8kHz = 160 samples
        expected = int(TWILIO_SAMPLE_RATE * 20 / 1000)
        assert TWILIO_FRAME_SIZE == expected
        assert TWILIO_FRAME_SIZE == 160
