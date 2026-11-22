"""Twitch chat via the official Twitch IRC interface (WebSocket).

Anonymous read-only login works for public channels. Set TWITCH_BOT_NICK and
TWITCH_OAUTH_TOKEN (scope: chat:read) to read as your own bot account.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from typing import AsyncIterator

from ..models import ChatMessage
from .base import ChatSource

WS_URL = "wss://irc-ws.chat.twitch.tv:443"


class TwitchChat(ChatSource):
    def __init__(self, channel: str):
        super().__init__(channel.lower().lstrip("#"))

    def stream_url(self) -> str:
        return f"https://www.twitch.tv/{self.channel}"

    async def messages(self) -> AsyncIterator[ChatMessage]:
        import websockets

        nick = os.getenv("TWITCH_BOT_NICK") or f"justinfan{random.randint(10000, 99999)}"
        token = os.getenv("TWITCH_OAUTH_TOKEN") or "SCHMOOPIIE"
        backoff = 1
        while True:
            try:
                async with websockets.connect(WS_URL, ping_interval=None) as ws:
                    await ws.send(f"PASS {token if token.startswith('oauth:') or token == 'SCHMOOPIIE' else 'oauth:' + token}")
                    await ws.send(f"NICK {nick}")
                    await ws.send("CAP REQ :twitch.tv/tags")   # tmi-sent-ts: when Twitch got it
                    await ws.send(f"JOIN #{self.channel}")
                    backoff = 1
                    async for raw in ws:
                        for line in raw.split("\r\n"):
                            if not line:
                                continue
                            if line.startswith("PING"):
                                await ws.send("PONG :tmi.twitch.tv")
                                continue
                            msg = self._parse(line)
                            if msg:
                                yield msg
            except (OSError, Exception) as e:  # reconnect on any drop
                print(f"[twitch:{self.channel}] disconnected ({e!r}); retry in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _parse(self, line: str) -> ChatMessage | None:
        # [@k=v;k=v ]:user!user@user.tmi.twitch.tv PRIVMSG #chan :text
        if " PRIVMSG #" not in line:
            return None
        tags = {}
        if line.startswith("@"):
            raw_tags, line = line[1:].split(" ", 1)
            tags = dict(t.split("=", 1) for t in raw_tags.split(";") if "=" in t)
        try:
            prefix, rest = line[1:].split(" PRIVMSG ", 1)
            text = rest.split(" :", 1)[1]
            user = prefix.split("!", 1)[0]
        except (IndexError, ValueError):
            return None
        try:
            sent = int(tags["tmi-sent-ts"]) / 1000
        except (KeyError, ValueError):
            sent = None
        return ChatMessage(self.channel, time.time(), user, text, sent)
