"""Synthetic chat: a noisy baseline with occasional hype bursts.

Used by demo mode and tests so the whole pipeline runs without API keys.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import AsyncIterator, Iterator

from ..models import ChatMessage
from .base import ChatSource

NORMAL = ["gg", "hi chat", "what rank is this", "lol", "nice", "true", "o7", "monkaS",
          "first time here", "which game is this", "ok", "hmm", "bruh", "sheesh"]
HYPE = ["KEKW", "LUL LUL LUL", "OMEGALUL", "CLIP IT", "POGGERS", "W", "NO WAY", "💀💀💀",
        "LMAO", "😂😂", "🔥🔥🔥", "CLIP THAT", "WHAT"]
DEMO_LINES = ["no way he just hit that shot", "chat did you see that",
              "that was actually insane", "okay okay we take those", "let's go"]


class SimulatedChat(ChatSource):
    def __init__(self, channel: str, base_rate: float = 1.5, minutes: float = 20,
                 bursts: list[tuple[float, float, float]] | None = None,
                 seed: int = 7, realtime: bool = False, start_ts: float | None = None):
        super().__init__(channel)
        self.base_rate = base_rate
        self.duration = minutes * 60
        self.rng = random.Random(seed)
        self.realtime = realtime
        self.start_ts = start_ts if start_ts is not None else time.time()
        # bursts: (offset_seconds, duration_seconds, multiplier)
        self.bursts = bursts if bursts is not None else self._random_bursts()

    def _random_bursts(self) -> list[tuple[float, float, float]]:
        out, t = [], 150.0
        while t < self.duration - 60:
            out.append((t, self.rng.uniform(6, 15), self.rng.uniform(4, 12)))
            t += self.rng.uniform(120, 260)
        return out

    def _rate_at(self, off: float) -> tuple[float, bool]:
        # slow drift so the baseline has to adapt
        drift = 1 + 0.3 * ((off / self.duration) - 0.5)
        for (b, dur, mult) in self.bursts:
            if b <= off < b + dur:
                return self.base_rate * drift * mult, True
        return self.base_rate * drift, False

    def generate(self) -> Iterator[ChatMessage]:
        """Deterministic, instant stream (virtual clock)."""
        off = 0.0
        while off < self.duration:
            rate, hype = self._rate_at(off)
            off += self.rng.expovariate(rate)
            pool = HYPE if hype and self.rng.random() < 0.8 else NORMAL
            yield ChatMessage(self.channel, self.start_ts + off,
                              f"viewer{self.rng.randint(1, 900)}", self.rng.choice(pool))

    async def messages(self) -> AsyncIterator[ChatMessage]:
        self.start_ts = time.time()
        for m in self.generate():
            if self.realtime:
                await asyncio.sleep(max(0.0, m.ts - time.time()))
            yield m
