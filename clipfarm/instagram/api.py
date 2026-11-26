"""Thin async client for Instagram API with Facebook Login and Reels publishing.

The Meta app uses Facebook Login: the Instagram professional account must be linked to a
Facebook Page the logged-in user manages. Your own accounts (app roles) need no app review.

  1. https://www.facebook.com/dialog/oauth                          -> code (+ state, checked)
  2. GET /<v>/oauth/access_token  code=...                           -> short-lived user token
     GET /<v>/oauth/access_token  grant_type=fb_exchange_token       -> long-lived token (~60 days)
  3. GET /<v>/me/accounts, then /<page>?fields=instagram_business_account,access_token
                                                                    -> IG user id + Page token
  4. POST /<IG_ID>/media  media_type=REELS, video_url=...           -> container (Meta fetches the video)
     GET  /<container>?fields=status_code                           -> IN_PROGRESS, FINISHED, ERROR, EXPIRED
     POST /<IG_ID>/media_publish creation_id=<container>            -> media id
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlparse

import aiohttp

VERSION = "v25.0"
AUTH_URL = "https://www.facebook.com/dialog/oauth"
GRAPH = "https://graph.facebook.com"
# business_management: without it /me/accounts misses Pages owned by a business portfolio
SCOPES = "instagram_basic,instagram_content_publish,pages_show_list,pages_read_engagement,business_management"
LONG_LIVED_S = 60 * 86400        # Meta often leaves expires_in out of long-lived user tokens


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
        self.user_id = str(data.get("user_id", ""))          # the Instagram account's id
        self.page_id = str(data.get("page_id", ""))          # the Facebook Page it is linked to
        self.expires_at = float(data.get("expires_at") or time.time() + float(data.get("expires_in") or LONG_LIVED_S))

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at - 60

    def to_dict(self) -> dict:
        return {"access_token": self.access_token, "user_id": self.user_id,
                "page_id": self.page_id, "expires_at": self.expires_at}


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

    async def _get(self, path: str, **params) -> dict:
        async with self.http.get(f"{GRAPH}/{VERSION}/{path}", params=params) as r:
            return await self._json(r)

    async def exchange_code(self, code: str) -> Token:
        """Code -> short-lived -> long-lived user token, then the Page's Instagram account."""
        app = {"client_id": self.cfg.app_id, "client_secret": self.cfg.app_secret}
        short = await self._get("oauth/access_token", redirect_uri=self.cfg.redirect_uri,
                                code=code.split("#")[0], **app)
        long = await self._get("oauth/access_token", grant_type="fb_exchange_token",
                               fb_exchange_token=short["access_token"], **app)
        pages = await self._get("me/accounts", fields="id", access_token=long["access_token"], limit="100")
        for p in pages.get("data", []):
            page = await self._get(p["id"], fields="instagram_business_account,access_token",
                                   access_token=long["access_token"])
            if page.get("instagram_business_account") and page.get("access_token"):
                # the Page token (made from a long-lived user token) is what publishing uses
                return Token({"access_token": page["access_token"], "expires_in": long.get("expires_in"),
                              "user_id": page["instagram_business_account"]["id"], "page_id": p["id"]})
        raise InstagramError("No Facebook Page linked to an Instagram Business or Creator account was "
                             "shared during login. Link the Instagram account to a Facebook Page you "
                             "manage, then connect again and tick that Page.")

    async def me(self, tok: Token) -> dict:
        me = await self._get(tok.user_id, fields="username,name,profile_picture_url",
                             access_token=tok.access_token)
        return {**me, "user_id": tok.user_id}

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
