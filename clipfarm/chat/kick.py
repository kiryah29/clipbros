"""Kick chat: placeholder.

Kick offers an official developer API (OAuth app + chat webhooks/events).
Implement `messages()` against it once you have API access; until then this
source refuses to start rather than scraping undocumented endpoints.
"""
from __future__ import annotations

from typing import AsyncIterator

from ..models import ChatMessage
from .base import ChatSource


class KickChat(ChatSource):
    def stream_url(self) -> str:
        return f"https://kick.com/{self.channel}"

    async def messages(self) -> AsyncIterator[ChatMessage]:
        raise NotImplementedError(
            "Kick chat needs the official Kick developer API. Register an app, subscribe to "
            "chat message events for the channel, and yield ChatMessage objects here.")
        yield  # pragma: no cover
