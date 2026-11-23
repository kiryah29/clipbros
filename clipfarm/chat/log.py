"""Rolling log of recent chat messages, so a clip can carry the chat that was on screen."""
from __future__ import annotations

from collections import deque

from ..models import ChatMessage


class ChatLog:
    def __init__(self, keep_s: float = 900):
        self.keep_s = float(keep_s)
        self.items: deque[ChatMessage] = deque()

    def add(self, msg: ChatMessage) -> None:
        self.items.append(msg)
        while self.items and self.items[0].ts < msg.ts - self.keep_s:
            self.items.popleft()

    def between(self, start: float, end: float, lead_s: float = 10,
                cap: int = 800) -> list[list]:
        """[[seconds into the clip, user, text, colour], ...] from `lead_s` before the
        clip start (so the strip isn't empty on frame one) up to its end."""
        out = [[round(m.ts - start, 2), m.user, m.text, m.color]
               for m in self.items if start - lead_s <= m.ts <= end]
        return out[-cap:]
