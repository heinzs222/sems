"""
Agent package.

Keep imports lightweight so modules like `src.agent.language` can be used without
requiring the full runtime dependency set (e.g., dotenv) at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agent.config import Config

__all__ = ["Config", "get_config"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from src.agent.config import Config, get_config

        return {"Config": Config, "get_config": get_config}[name]
    raise AttributeError(name)
