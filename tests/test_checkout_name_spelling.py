"""
Tests for deterministic checkout name confirmation + spelling fallback.
"""

from unittest.mock import AsyncMock

import pytest

from src.agent.config import Config
from src.agent.pipeline import CheckoutPhase, VoicePipeline


@pytest.mark.asyncio
async def test_name_confirm_then_spell_on_disapproval():
    send_message = AsyncMock()
    config = Config(public_host="test", menu_only=True, router_enabled=False)
    pipeline = VoicePipeline(send_message, config=config)
    pipeline._lang_state.current = "en"

    pipeline._speak_response = AsyncMock()  # type: ignore[method-assign]

    pipeline._checkout_phase = CheckoutPhase.NAME
    handled = await pipeline._handle_checkout_flow("John Doe")
    assert handled is True
    assert pipeline._customer_name == "John Doe"
    assert pipeline._checkout_phase == CheckoutPhase.NAME_CONFIRM

    spoken = pipeline._speak_response.await_args_list[-1].args[0]
    assert "your name is John Doe" in spoken

    pipeline._speak_response.reset_mock()
    handled = await pipeline._handle_checkout_flow("no")
    assert handled is True
    assert pipeline._checkout_phase == CheckoutPhase.NAME_SPELL

    spoken = pipeline._speak_response.await_args_list[-1].args[0].lower()
    assert "spell" in spoken

    pipeline._speak_response.reset_mock()
    handled = await pipeline._handle_checkout_flow("J O H N space D O E")
    assert handled is True
    assert pipeline._customer_name == "John Doe"
    assert pipeline._checkout_phase == CheckoutPhase.NAME_CONFIRM

    spoken = pipeline._speak_response.await_args_list[-1].args[0]
    assert "John Doe" in spoken
    assert "SPACE" in spoken


def test_llm_prompt_syncs_checkout_phase_to_name():
    send_message = AsyncMock()
    config = Config(public_host="test", menu_only=True, router_enabled=False)
    pipeline = VoicePipeline(send_message, config=config)

    pipeline._checkout_phase = CheckoutPhase.ORDERING
    pipeline._maybe_sync_checkout_phase_from_assistant("Perfect! What's your name?")
    assert pipeline._checkout_phase == CheckoutPhase.NAME


def test_extract_name_ignores_prefix_only_phrases():
    send_message = AsyncMock()
    config = Config(public_host="test", menu_only=True, router_enabled=False)
    pipeline = VoicePipeline(send_message, config=config)

    assert pipeline._extract_name("my name is") == ""
    assert pipeline._extract_name("my name is John") == "John"
    assert pipeline._extract_name("je m'appelle") == ""
    assert pipeline._extract_name("je m'appelle Hatem") == "Hatem"


@pytest.mark.asyncio
async def test_name_confirm_accepts_correction_without_yes_no():
    send_message = AsyncMock()
    config = Config(public_host="test", menu_only=True, router_enabled=False)
    pipeline = VoicePipeline(send_message, config=config)
    pipeline._lang_state.current = "en"

    pipeline._speak_response = AsyncMock()  # type: ignore[method-assign]

    pipeline._customer_name = "John"
    pipeline._checkout_phase = CheckoutPhase.NAME_CONFIRM

    handled = await pipeline._handle_checkout_flow("it's Hatem")
    assert handled is True
    assert pipeline._customer_name == "Hatem"
    assert pipeline._checkout_phase == CheckoutPhase.NAME_CONFIRM
