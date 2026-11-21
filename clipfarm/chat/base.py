from __future__ import annotations

import abc
from typing import AsyncIterator

from ..models import ChatMessage


class ChatSource(abc.ABC):
    """Async stream of chat messages for ONE live channel."""

    def __init__(self, channel: str):
        self.channel = channel

    @abc.abstractmethod
    def messages(self) -> AsyncIterator[ChatMessage]:
        ...

    async def is_live(self) -> bool:  # override where the platform can tell us
        return True

    def stream_url(self) -> str | None:
        """URL streamlink can open to record the live video."""
        return None
