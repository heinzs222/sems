"""
Tests for automatic language switching in the pipeline.
"""

from unittest.mock import AsyncMock

import pytest

from src.agent.config import get_config
from src.agent.pipeline import VoicePipeline


@pytest.mark.asyncio
async def test_auto_language_switch_en_to_fr(monkeypatch):
    pipeline = VoicePipeline(send_message=AsyncMock(), config=get_config())

    # Avoid any external network work in this unit test.
    pipeline._stt.restart = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    pipeline._generate_llm_response = AsyncMock()  # type: ignore[method-assign]

    await pipeline._process_transcript("Bonjour, je voudrais parler en français. Merci.")
    assert pipeline._lang_state.current == "fr"


@pytest.mark.asyncio
async def test_auto_language_switch_fr_to_en(monkeypatch):
    monkeypatch.setenv("DEFAULT_LANGUAGE", "fr")
    get_config.cache_clear()
    config = get_config()

    pipeline = VoicePipeline(send_message=AsyncMock(), config=config)

    pipeline._stt.restart = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    pipeline._generate_llm_response = AsyncMock()  # type: ignore[method-assign]

    # Stabilization requires 2 consecutive final transcripts to switch.
    await pipeline._process_transcript("Hello, I need help with my order. Thanks.")
    await pipeline._process_transcript("I'm calling in English, can you help me please?")
    assert pipeline._lang_state.current == "en"
