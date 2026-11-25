"""Thin async client for Business Login for Instagram and Reels publishing.

Instagram API with Instagram Login (no Facebook Page needed). For your own professional
(creator or business) account, Standard Access is enough: no Meta app review, and reels go
out public straight away.

  1. https://www.instagram.com/oauth/authorize            -> code
  2. POST https://api.instagram.com/oauth/access_token     -> short-lived token (1 h) + user id
  3. GET  https://graph.instagram.com/access_token          -> long-lived token (60 days)
     GET  https://graph.instagram.com/refresh_access_token  -> another 60 days
  4. POST /<IG_ID>/media  media_type=REELS, video_url=...   -> container (Meta fetches the video)
     GET  /<container>?fields=status_code                   -> IN_PROGRESS, FINISHED, ERROR, EXPIRED
     POST /<IG_ID>/media_publish creation_id=<container>    -> media id
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlparse

import aiohttp

AUTH_URL = "https://www.instagram.com/oauth/authorize"
TOKEN_URL = "https://api.instagram.com/oauth/access_token"
GRAPH = "https://graph.instagram.com"
VERSION = "v25.0"
SCOPES = "instagram_business_basic,instagram_business_content_publish"
REFRESH_BEFORE_S = 7 * 86400     # refresh a long-lived token when it has a week left


class InstagramError(RuntimeError):
    pass


@dataclass
class Config:
    app_id: str
    app_secret: str
    redirect_uri: str

    @classmethod
    def from_env(cls) -> "Config | None":
        """None when Instagram is not set up (the app still runs, the page says what's missing)."""
        vals = {k: os.getenv(k, "") for k in
                ("INSTAGRAM_APP_ID", "INSTAGRAM_APP_SECRET", "INSTAGRAM_REDIRECT_URI")}
        if not all(vals.values()):
            return None
        return cls(vals["INSTAGRAM_APP_ID"], vals["INSTAGRAM_APP_SECRET"], vals["INSTAGRAM_REDIRECT_URI"])

    @property
    def public_base(self) -> str:
        """https://clipbros.app: where Meta can fetch a reel from (same host as the redirect)."""
        u = urlparse(self.redirect_uri)
        return f"{u.scheme}://{u.netloc}"


def authorize_url(cfg: Config, state: str) -> str:
    return AUTH_URL + "?" + urlencode({
        "client_id": cfg.app_id, "redirect_uri": cfg.redirect_uri, "response_type": "code",
        "scope": SCOPES, "state": state})


def _check(body: dict) -> dict:
    err = body.get("error")
    if isinstance(err, dict):
        msg = err.get("error_user_msg") or err.get("message") or str(err)
        raise InstagramError(f"{msg} (code {err.get('code')}, subcode {err.get('error_subcode')})")
    if body.get("error_type") or body.get("error_message"):       # OAuth endpoint style
        raise InstagramError(f"{body.get('error_type', 'error')}: {body.get('error_message', '')}")
    return body


class Token:
    def __init__(self, data: dict):
        self.access_token = data["access_token"]
        self.user_id = str(data.get("user_id", ""))
        self.expires_at = float(data.get("expires_at") or time.time() + float(data.get("expires_in", 3600)))
        self.issued_at = float(data.get("issued_at") or time.time())

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at - 60

    @property
    def due_for_refresh(self) -> bool:     # Meta refreshes only tokens older than 24 h
        return self.expires_at - time.time() < REFRESH_BEFORE_S and time.time() - self.issued_at > 86400

    def to_dict(self) -> dict:
        return {"access_token": self.access_token, "user_id": self.user_id,
                "expires_at": self.expires_at, "issued_at": self.issued_at}


class TokenStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Token | None:
        try:
            return Token(json.loads(self.path.read_text()))
        except (OSError, ValueError, KeyError):
            return None

    def save(self, tok: Token) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(tok.to_dict()))

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class InstagramClient:
    def __init__(self, cfg: Config, http: aiohttp.ClientSession):
        self.cfg, self.http = cfg, http

    async def _json(self, resp: aiohttp.ClientResponse) -> dict:
        try:
            body = await resp.json(content_type=None)
        except ValueError:
            raise InstagramError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        return _check(body if isinstance(body, dict) else {"data": body})

    async def exchange_code(self, code: str) -> Token:
        form = {"client_id": self.cfg.app_id, "client_secret": self.cfg.app_secret,
                "grant_type": "authorization_code", "redirect_uri": self.cfg.redirect_uri,
                "code": code.split("#")[0]}
        async with self.http.post(TOKEN_URL, data=form) as r:
            short = await self._json(r)
        short = short.get("data", [short])[0] if isinstance(short.get("data"), list) else short
        params = {"grant_type": "ig_exchange_token", "client_secret": self.cfg.app_secret,
                  "access_token": short["access_token"]}
        async with self.http.get(f"{GRAPH}/access_token", params=params) as r:
            long = await self._json(r)
        return Token({**long, "user_id": short.get("user_id", "")})

    async def refresh(self, tok: Token) -> Token:
        params = {"grant_type": "ig_refresh_token", "access_token": tok.access_token}
        async with self.http.get(f"{GRAPH}/refresh_access_token", params=params) as r:
            new = await self._json(r)
        return Token({**new, "user_id": tok.user_id})

    async def me(self, tok: Token) -> dict:
        params = {"fields": "user_id,username,name,profile_picture_url,account_type",
                  "access_token": tok.access_token}
        async with self.http.get(f"{GRAPH}/{VERSION}/me", params=params) as r:
            return await self._json(r)

    async def create_reel(self, tok: Token, ig_id: str, video_url: str, caption: str,
                          share_to_feed: bool = True) -> str:
        form = {"media_type": "REELS", "video_url": video_url, "caption": caption[:2200],
                "share_to_feed": "true" if share_to_feed else "false", "access_token": tok.access_token}
        async with self.http.post(f"{GRAPH}/{VERSION}/{ig_id}/media", data=form) as r:
            return str((await self._json(r))["id"])

    async def container_status(self, tok: Token, container: str) -> dict:
        params = {"fields": "status_code,status", "access_token": tok.access_token}
        async with self.http.get(f"{GRAPH}/{VERSION}/{container}", params=params) as r:
            return await self._json(r)

    async def publish(self, tok: Token, ig_id: str, container: str) -> str:
        form = {"creation_id": container, "access_token": tok.access_token}
        async with self.http.post(f"{GRAPH}/{VERSION}/{ig_id}/media_publish", data=form) as r:
            return str((await self._json(r))["id"])

    async def permalink(self, tok: Token, media_id: str) -> str:
        params = {"fields": "permalink", "access_token": tok.access_token}
        async with self.http.get(f"{GRAPH}/{VERSION}/{media_id}", params=params) as r:
            return (await self._json(r)).get("permalink", "")
