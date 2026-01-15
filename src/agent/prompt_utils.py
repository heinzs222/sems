from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog

from src.agent.config import Config

logger = structlog.get_logger(__name__)

_DEFAULT_MAX_PROMPT_CHARS = 40_000


def _repo_root() -> Path:
    # src/agent/prompt_utils.py -> repo root is ../../
    return Path(__file__).resolve().parents[2]


def _read_text_file(path: str, *, max_chars: int) -> str:
    if not path:
        return ""

    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = _repo_root() / file_path

    try:
        content = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Prompt file not found", path=str(file_path))
        return ""
    except UnicodeDecodeError:
        try:
            content = file_path.read_text(encoding="utf-8-sig")
        except Exception:
            logger.warning("Prompt file decode failed", path=str(file_path))
            return ""
    except Exception:
        logger.exception("Prompt file read failed", path=str(file_path))
        return ""

    content = content.strip()
    if not content:
        return ""

    if len(content) > max_chars:
        logger.warning("Prompt truncated (too long)", path=str(file_path), max_chars=max_chars)
        content = content[:max_chars]

    return content


def _apply_placeholders(prompt: str, config: Config) -> str:
    if not prompt:
        return ""

    replacements = {
        "{AGENT_NAME}": config.agent_name,
        "{COMPANY_NAME}": config.company_name,
        "{agent_name}": config.agent_name,
        "{company_name}": config.company_name,
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)

    return prompt


def resolve_prompt(
    *,
    config: Config,
    inline_text: str,
    file_path: str,
    max_chars: int = _DEFAULT_MAX_PROMPT_CHARS,
) -> str:
    """
    Resolve a prompt from (1) inline text, else (2) file path, else "".

    - Applies simple placeholder substitution.
    - Truncates large prompts for safety.
    """
    prompt = (inline_text or "").strip()
    if not prompt:
        prompt = _read_text_file(file_path, max_chars=max_chars)

    return _apply_placeholders(prompt, config)

