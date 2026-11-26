"""Instagram pages of the web app: connect the account, review a clip, post it as a reel.

Same flow as TikTok: every reel is checked on a review screen and posted with an explicit
click. Meta fetches the video itself, so for the upload the reel is served from a public,
random, short-lived URL on this app (/ig/media/<token>.mp4) and the link is dropped after.
"""
from __future__ import annotations

import asyncio
import html
import os
import secrets
import time
from pathlib import Path
from urllib.parse import urlparse

from aiohttp import web

from .api import Config, InstagramClient, InstagramError, Token, TokenStore, authorize_url, public_base

e = html.escape
DEFAULT_CAPTION = "Clip from {channel} #gaming #streamer #clips #reels"
MEDIA_TTL_S = 2 * 3600
MEDIA_RE = r"^/ig/media/[A-Za-z0-9_-]{24,}\.mp4$"


def default_caption(clip) -> str:
    """The clip's hook, then INSTAGRAM_DEFAULT_CAPTION ({channel} is replaced)."""
    tail = (os.getenv("INSTAGRAM_DEFAULT_CAPTION") or DEFAULT_CAPTION).replace("{channel}", clip.channel)
    hook = str((clip.receipt or {}).get("hook") or "").strip()
    return (f"{hook}\n{tail}" if hook else tail)[:2200]


class InstagramPages:
    def __init__(self, cfg: Config | None, store: TokenStore, runs, page, http_getter):
        self.cfg, self.store, self.runs, self.page = cfg, store, runs, page
        self.http = http_getter                 # the app's aiohttp session (made at startup)
        self.states: dict[str, str] = {}       # OAuth state -> the redirect_uri the login used
        self.media: dict[str, tuple[Path, float]] = {}
        self.pending: dict[str, dict] = {}      # container id -> what to publish and mark
        self.locks: dict[str, asyncio.Lock] = {}

    @property
    def redirect_paths(self) -> list[str]:
        if not self.cfg:
            return ["/callback/instagram"]
        return list(dict.fromkeys(urlparse(u).path for u in self.cfg.redirect_uris))

    def redirect_for(self, request: web.Request) -> str:
        return self.cfg.redirect_for(request.headers.get("X-Forwarded-Host") or request.host)

    def client(self) -> InstagramClient:
        return InstagramClient(self.cfg, self.http())

    async def token(self) -> Token | None:
        t = self.store.load()
        if not t:
            return None
        if t.expired or not t.user_id:           # no user_id: a token from the old Instagram Login
            self.store.clear()
            return None
        return t

    def not_set_up(self) -> web.Response:
        return self.page("Instagram", (
            "<div class='card'><p>Instagram is not set up yet. Put <code>INSTAGRAM_APP_ID</code>, "
            "<code>INSTAGRAM_APP_SECRET</code> and <code>INSTAGRAM_REDIRECT_URIS</code> in <code>.env</code> "
            "and restart the app (see README, 'Post to Instagram').</p></div>"))

    # -- account --------------------------------------------------------------------
    async def index(self, request: web.Request) -> web.Response:
        if not self.cfg:
            return self.not_set_up()
        tok = await self.token()
        if not tok:
            return self.page("Instagram", "<div class='card'><p>Connect your Instagram professional "
                             "account (through the Facebook Page it is linked to) to post reels.</p><a class='btn' href='/instagram/login'>Connect "
                             "Instagram</a></div>")
        try:
            me = await self.client().me(tok)
        except InstagramError as ex:
            return self.page("Instagram", f"<p class='err'>{e(str(ex))}</p>"
                             "<a class='btn' href='/instagram/login'>Connect again</a>")
        return self.page("Instagram", (
            f"<div class='card row'><img class='av' src='{e(me.get('profile_picture_url', ''))}' alt=''>"
            f"<div><b>@{e(me.get('username', ''))}</b><br><span class='mut'>Instagram connected"
            f" &middot; login valid until {time.strftime('%d %b %Y', time.localtime(tok.expires_at))}</span></div>"
            "<form method='post' action='/instagram/logout' style='margin-left:auto'>"
            "<button class='ghost'>Disconnect</button></form></div>"
            "<p>Post reels from <a href='/clips'>Clips</a>.</p>"))

    async def login(self, request: web.Request) -> web.Response:
        if not self.cfg:
            return self.not_set_up()
        state, redirect = secrets.token_urlsafe(16), self.redirect_for(request)
        self.states[state] = redirect
        raise web.HTTPFound(authorize_url(self.cfg, state, redirect))

    async def callback(self, request: web.Request) -> web.Response:
        if not self.cfg:
            return self.not_set_up()
        q = request.query
        if q.get("error"):
            return self.page("Login cancelled", "<p class='err'>Instagram login was not completed: "
                             f"{e(q.get('error_description') or q['error'])}</p>")
        if q.get("state", "") not in self.states:
            return self.page("Login failed", "<p class='err'>Invalid login state. Start again from "
                             "<a href='/instagram'>Instagram</a>.</p>")
        redirect = self.states.pop(q["state"])
        try:
            self.store.save(await self.client().exchange_code(q.get("code", ""), redirect))
        except (InstagramError, KeyError) as ex:
            return self.page("Login failed", f"<p class='err'>{e(str(ex))}</p>")
        raise web.HTTPFound("/instagram")

    async def logout(self, request: web.Request) -> web.Response:
        self.store.clear()
        raise web.HTTPFound("/instagram")

    # -- review and post ------------------------------------------------------------
    def _clip(self, request: web.Request):
        rid, cid = request.match_info["rid"], int(request.match_info["cid"])
        if rid not in self.runs.runs:
            raise web.HTTPNotFound()
        c, p = self.runs.clip(rid, cid), self.runs.clip_file(rid, cid)
        if not c or not p:
            raise web.HTTPNotFound()
        return rid, c, p

    async def review(self, request: web.Request) -> web.Response:
        if not self.cfg:
            return self.not_set_up()
        rid, c, _ = self._clip(request)
        tok = await self.token()
        if not tok:
            raise web.HTTPFound("/instagram/login")
        try:
            me = await self.client().me(tok)
        except InstagramError as ex:
            return self.page("Review", f"<p class='err'>Could not load your Instagram account: {e(str(ex))}</p>")
        base = f"/runs/{rid}/clip/{c.id}"
        done = (c.receipt.get("posted_to") or {}).get("instagram")
        held = ("<p class='err'>Held by the nudity check: watch the whole clip before posting.</p>"
                if c.status == "held" else "")
        again = f"<p class='err'>Already posted to Instagram: {e(done)}</p>" if done else ""
        body = f"""
<div class='card row'><div>Posting as <b>@{e(me.get('username', ''))}</b> on Instagram (public reel)</div></div>
{held}{again}<video controls playsinline src='{base}.mp4'></video>
<form id='f' class='card' method='post' action='{base}/instagram/post'>
 <label>Caption<textarea name='caption' rows='4' maxlength='2200'>{e(default_caption(c))}</textarea></label>
 <label class='opt'><input type='checkbox' name='share_to_feed' checked> Also show on my profile grid</label>
 <label class='opt'><input type='checkbox' id='consent' name='consent' required> I have reviewed this video and want to post it publicly to my Instagram account.</label>
 <button id='go' disabled>Post to Instagram</button></form>
<script>const c=document.getElementById('consent'),g=document.getElementById('go');
c.addEventListener('change',()=>{{g.disabled=!c.checked;}});</script>"""
        return self.page(f"Instagram: clip {c.id}", body)

    async def post(self, request: web.Request) -> web.Response:
        if not self.cfg:
            return self.not_set_up()
        rid, c, path = self._clip(request)
        tok = await self.token()
        if not tok:
            raise web.HTTPFound("/instagram/login")
        form = await request.post()
        if "consent" not in form:
            return self.page("Not posted", "<p class='err'>Tick the box to confirm you reviewed the clip.</p>")
        ig_id = tok.user_id
        link = secrets.token_urlsafe(24)
        self.media[link] = (path, time.time() + MEDIA_TTL_S)
        url = f"{public_base(self.redirect_for(request))}/ig/media/{link}.mp4"
        try:
            container = await self.client().create_reel(
                tok, ig_id, url, str(form.get("caption", "")), "share_to_feed" in form)
        except InstagramError as ex:
            self.media.pop(link, None)
            return self.page("Not posted", f"<p class='err'>{e(str(ex))}</p>"
                             f"<a href='/runs/{rid}/clip/{c.id}/instagram'>Back</a>")
        self.pending[container] = {"rid": rid, "cid": c.id, "link": link, "media": None, "ig_id": ig_id}
        body = f"""<div class='card'><p>Instagram is processing the reel. This takes up to a few minutes;
keep this page open.</p><p id='s' class='mut'>IN_PROGRESS</p></div>
<script>
async function poll(){{const r=await fetch('/instagram/status/{e(container)}');const j=await r.json();
 const s=document.getElementById('s');
 if(j.permalink){{s.innerHTML="Posted: <a href='"+j.permalink+"' target='_blank' rel='noopener'>"+j.permalink+"</a>";return;}}
 s.textContent=j.status||j.error||'unknown';
 if(!j.error&&!['ERROR','EXPIRED'].includes(j.status))setTimeout(poll,4000);}}poll();
</script>"""
        return self.page("Posting to Instagram", body)

    async def status(self, request: web.Request) -> web.Response:
        container = request.match_info["container"]
        job = self.pending.get(container)
        tok = await self.token()
        if not job or not tok:
            return web.json_response({"error": "unknown upload or not connected"}, status=404)
        lock = self.locks.setdefault(container, asyncio.Lock())
        async with lock:                         # two polls must not publish twice
            if job["media"]:
                return web.json_response({"status": "PUBLISHED", "permalink": job.get("permalink", "")})
            try:
                st = await self.client().container_status(tok, container)
                code = st.get("status_code", "")
                if code == "FINISHED":
                    job["media"] = await self.client().publish(tok, job["ig_id"], container)
                    job["permalink"] = await self.client().permalink(tok, job["media"])
                    self.runs.update_receipt(job["rid"], job["cid"], {"posted_to": {
                        **(self.runs.clip(job["rid"], job["cid"]).receipt.get("posted_to") or {}),
                        "instagram": job["permalink"] or job["media"]}})
                    self.media.pop(job["link"], None)
                    return web.json_response({"status": "PUBLISHED", "permalink": job["permalink"]})
                if code in ("ERROR", "EXPIRED"):
                    self.media.pop(job["link"], None)
                return web.json_response({"status": code, "detail": st.get("status", "")})
            except InstagramError as ex:
                return web.json_response({"error": str(ex)})

    async def media_file(self, request: web.Request) -> web.StreamResponse:
        """Public on purpose: Meta downloads the reel from here. Random, short-lived links only."""
        now = time.time()
        for k in [k for k, (_, exp) in self.media.items() if exp < now]:
            del self.media[k]
        hit = self.media.get(request.match_info["link"])
        if not hit:
            raise web.HTTPNotFound()
        return web.FileResponse(hit[0], headers={"Cache-Control": "no-store"})

    def routes(self) -> list:
        clip = "/runs/{rid}/clip/{cid:\\d+}/instagram"
        return [
            web.get("/instagram", self.index), web.get("/instagram/login", self.login),
            web.post("/instagram/logout", self.logout),
            *(web.get(p, self.callback) for p in self.redirect_paths),
            web.get(clip, self.review), web.post(clip + "/post", self.post),
            web.get("/instagram/status/{container}", self.status),
            web.get("/ig/media/{link:[A-Za-z0-9_-]{24,}}.mp4", self.media_file),
        ]
