import os
from unittest.mock import patch

import pytest


def test_config_validate_realtime_mode_allows_missing_pipeline_keys():
    from src.agent.config import get_config

    env = {
        "PUBLIC_HOST": "test.ngrok.io",
        "VOICE_MODE": "openai_realtime",
        "OPENAI_API_KEY": "test_openai_key",
        "OPENAI_REALTIME_MODEL": "gpt-4o-realtime-preview",
    }

    with patch.dict(os.environ, env, clear=True):
        get_config.cache_clear()
        config = get_config()
        config.validate()


def test_config_validate_pipeline_mode_requires_deepgram():
    from src.agent.config import get_config, ConfigError

    env = {
        "PUBLIC_HOST": "test.ngrok.io",
        "VOICE_MODE": "pipeline",
        "GROQ_API_KEY": "test_groq",
        "CARTESIA_API_KEY": "test_cartesia",
        # Intentionally missing DEEPGRAM_API_KEY
    }

    with patch.dict(os.environ, env, clear=True):
        get_config.cache_clear()
        config = get_config()
        with pytest.raises(ConfigError):
            config.validate()


@pytest.mark.asyncio
async def test_create_pipeline_selects_realtime_pipeline():
    from src.agent.config import get_config
    from src.agent.pipeline import create_pipeline

    env = {
        "PUBLIC_HOST": "test.ngrok.io",
        "VOICE_MODE": "openai_realtime",
        "OPENAI_API_KEY": "test_openai_key",
        "OPENAI_REALTIME_MODEL": "gpt-4o-realtime-preview",
    }

    async def _send(_message: str) -> None:
        return None

    with patch.dict(os.environ, env, clear=True):
        get_config.cache_clear()
        pipeline = await create_pipeline(_send)
        try:
            assert pipeline.__class__.__name__ == "OpenAIRealtimePipeline"
        finally:
            await pipeline.stop()
