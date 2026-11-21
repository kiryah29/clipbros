"""Thin async client for TikTok Login Kit (OAuth v2) and the Content Posting API."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import aiohttp

AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
API = "https://open.tiktokapis.com"
DEFAULT_SCOPES = "user.info.basic,video.publish"

MIN_CHUNK = 5 * 1024 * 1024
MAX_CHUNK = 64 * 1024 * 1024


class TikTokError(RuntimeError):
    pass


@dataclass
class Config:
    client_key: str
    client_secret: str
    redirect_uri: str
    scopes: str = DEFAULT_SCOPES

    @classmethod
    def from_env(cls) -> "Config":
        key = os.getenv("TIKTOK_CLIENT_KEY", "")
        secret = os.getenv("TIKTOK_CLIENT_SECRET", "")
        redirect = os.getenv("TIKTOK_REDIRECT_URI", "")
        missing = [n for n, v in (("TIKTOK_CLIENT_KEY", key), ("TIKTOK_CLIENT_SECRET", secret),
                                  ("TIKTOK_REDIRECT_URI", redirect)) if not v]
        if missing:
            raise TikTokError("missing environment variables: " + ", ".join(missing))
        return cls(key, secret, redirect, os.getenv("TIKTOK_SCOPES", DEFAULT_SCOPES))


def authorize_url(cfg: Config, state: str) -> str:
    return AUTH_URL + "?" + urlencode({
        "client_key": cfg.client_key, "scope": cfg.scopes, "response_type": "code",
        "redirect_uri": cfg.redirect_uri, "state": state})


def _check(body: dict) -> dict:
    err = body.get("error")
    if isinstance(err, dict) and err.get("code") not in (None, "ok"):
        raise TikTokError(f"{err.get('code')}: {err.get('message')}")
    if body.get("error") and isinstance(body["error"], str):  # oauth endpoint style
        raise TikTokError(f"{body['error']}: {body.get('error_description', '')}")
    return body


class Token:
    def __init__(self, data: dict):
        self.access_token: str = data["access_token"]
        self.refresh_token: str = data.get("refresh_token", "")
        self.open_id: str = data.get("open_id", "")
        self.expires_at: float = data.get("expires_at") or time.time() + int(data.get("expires_in", 0))

    def to_dict(self) -> dict:
        return {"access_token": self.access_token, "refresh_token": self.refresh_token,
                "open_id": self.open_id, "expires_at": self.expires_at}

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at - 60


class TokenStore:
    """One token file on disk (chmod 600). Lives under data/, which is gitignored."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> Token | None:
        try:
            return Token(json.loads(self.path.read_text()))
        except (OSError, ValueError, KeyError):
            return None

    def save(self, token: Token) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(token.to_dict()))
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class TikTokClient:
    def __init__(self, cfg: Config, session: aiohttp.ClientSession):
        self.cfg, self.http = cfg, session

    async def _token_request(self, payload: dict) -> Token:
        payload = {"client_key": self.cfg.client_key, "client_secret": self.cfg.client_secret, **payload}
        async with self.http.post(f"{API}/v2/oauth/token/", data=payload) as r:
            body = _check(await r.json(content_type=None))
        return Token(body)

    async def exchange_code(self, code: str) -> Token:
        return await self._token_request({"code": code, "grant_type": "authorization_code",
                                          "redirect_uri": self.cfg.redirect_uri})

    async def refresh(self, token: Token) -> Token:
        return await self._token_request({"grant_type": "refresh_token",
                                          "refresh_token": token.refresh_token})

    async def user_info(self, token: Token) -> dict:
        async with self.http.get(f"{API}/v2/user/info/", params={"fields": "open_id,display_name,avatar_url"},
                                 headers=_auth(token)) as r:
            return _check(await r.json(content_type=None)).get("data", {}).get("user", {})

    async def creator_info(self, token: Token) -> dict:
        async with self.http.post(f"{API}/v2/post/publish/creator_info/query/", headers=_auth(token, json_body=True)) as r:
            return _check(await r.json(content_type=None)).get("data", {})

    async def publish_video(self, token: Token, path: Path, post_info: dict) -> str:
        """Direct post via FILE_UPLOAD. Returns the publish_id."""
        size = path.stat().st_size
        chunk = size if size <= MAX_CHUNK else MAX_CHUNK
        total_chunks = max(1, size // chunk)  # TikTok folds the remainder into the last chunk
        body = {"post_info": post_info,
                "source_info": {"source": "FILE_UPLOAD", "video_size": size,
                                "chunk_size": chunk, "total_chunk_count": total_chunks}}
        async with self.http.post(f"{API}/v2/post/publish/video/init/", json=body,
                                  headers=_auth(token, json_body=True)) as r:
            data = _check(await r.json(content_type=None)).get("data", {})
        upload_url, publish_id = data["upload_url"], data["publish_id"]
        with open(path, "rb") as f:
            for i in range(total_chunks):
                start = i * chunk
                end = size - 1 if i == total_chunks - 1 else start + chunk - 1
                f.seek(start)
                blob = f.read(end - start + 1)
                headers = {"Content-Type": "video/mp4", "Content-Range": f"bytes {start}-{end}/{size}"}
                async with self.http.put(upload_url, data=blob, headers=headers) as r:
                    if r.status not in (200, 201, 206):
                        raise TikTokError(f"upload chunk {i + 1}/{total_chunks} failed: HTTP {r.status} {await r.text()}")
        return publish_id

    async def publish_status(self, token: Token, publish_id: str) -> dict:
        async with self.http.post(f"{API}/v2/post/publish/status/fetch/", json={"publish_id": publish_id},
                                  headers=_auth(token, json_body=True)) as r:
            return _check(await r.json(content_type=None)).get("data", {})


def _auth(token: Token, json_body: bool = False) -> dict:
    h = {"Authorization": f"Bearer {token.access_token}"}
    if json_body:
        h["Content-Type"] = "application/json; charset=UTF-8"
    return h
