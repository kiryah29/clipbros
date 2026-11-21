"""YouTube live chat via the official YouTube Data API v3.

Needs YOUTUBE_API_KEY. Note: liveChatMessages.list costs quota on every poll;
the API tells us how long to wait between polls and we respect it.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncIterator

from ..models import ChatMessage
from .base import ChatSource

API = "https://www.googleapis.com/youtube/v3"


class YouTubeChat(ChatSource):
    def __init__(self, channel: str, video_id: str | None = None):
        super().__init__(channel)
        self.video_id = video_id
        self.key = os.getenv("YOUTUBE_API_KEY")

    def stream_url(self) -> str | None:
        return f"https://www.youtube.com/watch?v={self.video_id}" if self.video_id else None

    async def _get(self, session, path: str, **params):
        params["key"] = self.key
        async with session.get(f"{API}/{path}", params=params) as r:
            r.raise_for_status()
            return await r.json()

    async def _resolve_live_video(self, session) -> str | None:
        handle = self.channel.lstrip("@")
        ch = await self._get(session, "channels", part="id", forHandle=handle)
        if not ch.get("items"):
            return None
        cid = ch["items"][0]["id"]
        res = await self._get(session, "search", part="id", channelId=cid,
                              eventType="live", type="video", maxResults=1)
        items = res.get("items", [])
        return items[0]["id"]["videoId"] if items else None

    async def messages(self) -> AsyncIterator[ChatMessage]:
        import aiohttp

        if not self.key:
            raise RuntimeError("YOUTUBE_API_KEY is not set")
        async with aiohttp.ClientSession() as s:
            while not self.video_id:
                self.video_id = await self._resolve_live_video(s)
                if not self.video_id:
                    await asyncio.sleep(120)   # not live yet
            v = await self._get(s, "videos", part="liveStreamingDetails", id=self.video_id)
            chat_id = v["items"][0]["liveStreamingDetails"]["activeLiveChatId"]
            page = None
            while True:
                params = dict(liveChatId=chat_id, part="snippet,authorDetails", maxResults=2000)
                if page:
                    params["pageToken"] = page
                data = await self._get(s, "liveChat/messages", **params)
                for it in data.get("items", []):
                    text = it["snippet"].get("displayMessage", "")
                    yield ChatMessage(self.channel, time.time(),
                                      it["authorDetails"]["displayName"], text)
                page = data.get("nextPageToken")
                await asyncio.sleep(data.get("pollingIntervalMillis", 5000) / 1000)
