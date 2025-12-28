"""
Tests for Deepgram Flux (v2) audio batching behavior.
"""

from unittest.mock import AsyncMock

import pytest

from src.agent.audio import FLUX_FRAME_SIZE, TWILIO_FRAME_SIZE
from src.agent.stt import DeepgramSTT


@pytest.mark.asyncio
async def test_flux_batches_into_80ms_chunks():
    stt = DeepgramSTT()
    stt._is_connected = True
    stt._uses_flux = True
    stt._ws = AsyncMock()
    stt._flux_remainder = b""

    frame = b"\xff" * TWILIO_FRAME_SIZE  # 20ms ulaw frame

    # 1x20ms -> no send yet
    await stt.send_audio(frame)
    assert stt._ws.send.await_count == 0

    # +3x20ms -> 4 frames total -> 80ms -> exactly one send
    await stt.send_audio(frame * 3)
    assert stt._ws.send.await_count == 1
    sent = stt._ws.send.await_args.args[0]
    assert len(sent) == FLUX_FRAME_SIZE
    assert stt._flux_remainder == b""


@pytest.mark.asyncio
async def test_flux_keeps_remainder_after_full_chunks():
    stt = DeepgramSTT()
    stt._is_connected = True
    stt._uses_flux = True
    stt._ws = AsyncMock()
    stt._flux_remainder = b""

    frame = b"\xff" * TWILIO_FRAME_SIZE  # 20ms ulaw frame

    # 5 frames -> 80ms chunk + 20ms remainder
    await stt.send_audio(frame * 5)
    assert stt._ws.send.await_count == 1
    sent = stt._ws.send.await_args.args[0]
    assert len(sent) == FLUX_FRAME_SIZE
    assert stt._flux_remainder == frame

