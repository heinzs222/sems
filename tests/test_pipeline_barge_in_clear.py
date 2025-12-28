"""
Tests for barge-in clear behavior in the voice pipeline.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.agent.pipeline import VoicePipeline, PipelineState
from src.agent.twilio_protocol import TwilioStartEvent


@pytest.mark.asyncio
async def test_hard_interrupt_sends_twilio_clear_before_cleanup_tasks():
    send_message = AsyncMock()
    pipeline = VoicePipeline(send_message)

    # Make protocol state available so create_clear() returns a message.
    pipeline._protocol.handle_start(
        TwilioStartEvent(
            stream_sid="MZ123",
            call_sid="CA456",
            account_sid="AC789",
            tracks=["inbound"],
        )
    )

    # Pretend we're mid-TTS so hard interrupt path is active.
    pipeline._state = PipelineState.SPEAKING
    pipeline._is_running = True

    # Block stop_pacer so we can assert that clear was already sent.
    stop_started = asyncio.Event()
    stop_continue = asyncio.Event()

    async def stop_pacer_mock():
        stop_started.set()
        await stop_continue.wait()

    pipeline._stop_pacer = stop_pacer_mock  # type: ignore[method-assign]
    pipeline._tts.cancel_context = AsyncMock()

    task = asyncio.create_task(pipeline._handle_hard_interrupt(reason="test"))

    await asyncio.wait_for(stop_started.wait(), timeout=1.0)

    assert send_message.await_count >= 1
    first_msg = send_message.await_args_list[0].args[0]
    assert '"event":"clear"' in first_msg.replace(" ", "")
    assert pipeline._protocol.call_state.playback_generation_id == 1

    stop_continue.set()
    await asyncio.wait_for(task, timeout=1.0)

