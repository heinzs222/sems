"""
Tests for NAME/ADDRESS debounce in deterministic checkout capture.
"""

from unittest.mock import AsyncMock

import pytest

from src.agent.config import get_config
from src.agent.pipeline import CheckoutPhase, PipelineState, VoicePipeline
from src.agent.stt import TranscriptionResult


@pytest.mark.asyncio
async def test_name_prefix_debounce_combines_two_finals():
    pipeline = VoicePipeline(send_message=AsyncMock(), config=get_config())
    pipeline._is_running = True  # type: ignore[attr-defined]
    pipeline._state = PipelineState.LISTENING  # type: ignore[attr-defined]
    pipeline._checkout_phase = CheckoutPhase.NAME  # type: ignore[attr-defined]

    await pipeline._on_transcript(TranscriptionResult(text="my name is", is_final=True, confidence=0.9))  # type: ignore[attr-defined]
    assert pipeline._turn_queue.empty()  # type: ignore[attr-defined]

    await pipeline._on_transcript(TranscriptionResult(text="Hatem", is_final=True, confidence=0.9))  # type: ignore[attr-defined]
    queued = await pipeline._turn_queue.get()  # type: ignore[attr-defined]
    assert queued.text == "my name is Hatem"

