from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

from src.agent.tts_types import TTSChunk


class TTSProvider(ABC):
    @abstractmethod
    async def synthesize_streaming(
        self,
        text: str,
        *,
        voice_id: Optional[str] = None,
        context: Optional[list[dict]] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        raise NotImplementedError

    async def cancel_context(self) -> None:
        return None

    def cancel(self) -> None:
        return None

    async def close(self) -> None:
        return None

