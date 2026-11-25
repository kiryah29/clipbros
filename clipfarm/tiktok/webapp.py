"""Small review-and-post web app: connect TikTok, review a clip, post it to your own account.

Follows TikTok's Content Posting API UX rules: show the creator's account, no default
privacy choice, interaction toggles off by default, commercial-content disclosure, the
consent declaration, and an explicit Post click for every video.
"""
from __future__ import annotations

import html
import json
import re
import os
import secrets
from urllib.parse import urlparse
from pathlib import Path

import aiohttp
from aiohttp import web

from .. import brand
from ..control import COOKIE, ControlPages, RunManager, Sessions, auth_middleware
from ..instagram import api as ig_api
from ..instagram.webapp import MEDIA_RE, InstagramPages
from ..queue import ClipQueue
from .api import Config, TikTokClient, TikTokError, Token, TokenStore, authorize_url

CSS = brand.WEB_CSS + """
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.mut{color:var(--mut);font-size:.9rem}
button,.btn{background:var(--acc);color:var(--on-acc);border:0;border-radius:10px;padding:10px 16px;font:600 1rem Inter,system-ui,sans-serif;cursor:pointer;text-decoration:none;display:inline-block}
button[disabled]{opacity:.45;cursor:not-allowed}.ghost{background:transparent;color:var(--fg);border:1px solid var(--line)}
input[type=text],input[type=password],input[type=number],select,textarea{width:100%;box-sizing:border-box;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--fg);font:inherit}
video{width:100%;max-height:420px;border-radius:10px;background:#000}img.av{width:44px;height:44px;border-radius:50%}
label.opt{display:block;margin:6px 0}.err{color:#FF6B6B}.ok{color:var(--acc)}
"""


def page(title: str, body: str, status: int = 200) -> web.Response:
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
           f"{brand.FONTS_LINK}<style>{CSS}</style></head><body>"
           f"<header>{brand.logo_html().replace('{home}', '/')}<span class='label'>Clip bros app</span></header>{body}"
           f"<p class='mut'><a href='/control'>Control</a> &middot; <a href='/clips'>Clips</a> &middot; "
           f"<a href='{LEGAL}terms.html'>Terms</a> &middot; "
           f"<a href='{LEGAL}privacy.html'>Privacy</a></p></body></html>")
    return web.Response(text=doc, content_type="text/html", status=status)


LEGAL = "/"
DOCS = Path(__file__).resolve().parents[2] / "docs"
e = html.escape
DEFAULT_CAPTION = "Clip from {channel} #fyp #streamer #clips"


def default_caption(channel: str) -> str:
    """Caption prefilled on the review screen. Override with TIKTOK_DEFAULT_CAPTION ({channel} is replaced)."""
    template = os.getenv("TIKTOK_DEFAULT_CAPTION") or DEFAULT_CAPTION
    return template.replace("{channel}", channel)[:2200]


def caption(clip) -> str:
    """The hook card's text on the first line, then the default caption with hashtags."""
    hook = str((clip.receipt or {}).get("hook") or "").strip()
    return (f"{hook}\n{default_caption(clip.channel)}" if hook else default_caption(clip.channel))[:2200]


FRIENDLY_ERRORS = {
    "unaudited_client_can_only_post_to_private_accounts":
        "TikTok only lets an app that has not been audited post to a PRIVATE TikTok account. Set your TikTok account "
        "to private in the TikTok app (Settings and privacy, Privacy, Private account), then post again with "
        "'Only me'. You can make it public again after posting, or after TikTok approves the app.",
    "privacy_level_option_mismatch":
        "The privacy choice is not allowed for your account. Pick one of the options in the list.",
    "access_token_invalid": "Your TikTok login has expired. Disconnect and connect TikTok again.",
    "scope_not_authorized": "The app is missing the video.publish permission. Disconnect, then connect TikTok again "
                            "and approve all permissions.",
    "spam_risk_too_many_posts": "TikTok's daily posting limit for this account was reached. Try again tomorrow.",
}


def friendly(ex: Exception) -> str:
    text = str(ex)
    for code, msg in FRIENDLY_ERRORS.items():
        if code in text:
            return f"{msg} <span class='mut'>(TikTok error: {e(code)})</span>"
    return e(text)


PRIVACY_LABELS = {"PUBLIC_TO_EVERYONE": "Everyone", "MUTUAL_FOLLOW_FRIENDS": "Friends",
                  "FOLLOWER_OF_CREATOR": "Followers", "SELF_ONLY": "Only me"}


class App:
    def __init__(self, cfg: Config, queue: ClipQueue, store: TokenStore):
        self.cfg, self.queue, self.store = cfg, queue, store
        self.runs: RunManager | None = None    # control runs, where the live clips are
        self.states: set[str] = set()
        self.http: aiohttp.ClientSession | None = None

    # -- helpers --------------------------------------------------------------------
    def client(self) -> TikTokClient:
        return TikTokClient(self.cfg, self.http)

    async def token(self) -> Token | None:
        t = self.store.load()
        if t and t.expired and t.refresh_token:
            try:
                t = await self.client().refresh(t)
                self.store.save(t)
            except TikTokError:
                self.store.clear()
                return None
        return t

    def clip_path(self, clip_id: int) -> Path | None:
        c = self.queue.get(clip_id)
        p = Path(c.reel_path) if c and c.reel_path else None
        return p if p and p.is_file() else None

    def target(self, request: web.Request) -> dict:
        """The clip a review/post URL points at: one in the own queue (/clip/7) or one from a
        control run (/runs/<run>/clip/7/tiktok)."""
        cid = int(request.match_info["id"] if "id" in request.match_info else request.match_info["cid"])
        rid = request.match_info.get("rid")
        if rid is None:
            c, p = self.queue.get(cid), self.clip_path(cid)
            return {"clip": c, "path": p, "video": f"/video/{cid}", "post": f"/clip/{cid}/post",
                    "back": f"/clip/{cid}", "mark": lambda st: self.queue.update(cid, status=st),
                    "posted_status": "approved"}
        if not self.runs or rid not in self.runs.runs:
            raise web.HTTPNotFound()
        base = f"/runs/{rid}/clip/{cid}"
        return {"clip": self.runs.clip(rid, cid), "path": self.runs.clip_file(rid, cid),
                "video": f"{base}.mp4", "post": f"{base}/tiktok/post", "back": f"{base}/tiktok",
                "mark": lambda st: self.runs.set_status(rid, cid, st), "posted_status": "posted"}

    def run_clips(self) -> list[tuple[dict, object]]:
        """Rendered clips from the control runs, newest run first, rejected and posted ones left out."""
        if not self.runs:
            return []
        out = []
        for r in sorted(self.runs.runs.values(), key=lambda r: r["created"], reverse=True):
            out += [(r, c) for c in self.runs.clips(r["id"])
                    if c.reel_path and c.status not in ("rejected", "posted")]
        return out

    # -- pages ----------------------------------------------------------------------
    async def index(self, request: web.Request) -> web.Response:
        tok = await self.token()
        if tok:
            try:
                u = await self.client().user_info(tok)
            except TikTokError as ex:
                u = {}
                note = f"<p class='err'>{e(str(ex))}</p>"
            else:
                note = ""
            acct = (f"<div class='card row'><img class='av' src='{e(u.get('avatar_url', ''))}' alt=''>"
                    f"<div><b>{e(u.get('display_name', 'Connected account'))}</b><br>"
                    f"<span class='mut'>TikTok account connected</span></div>"
                    f"<form method='post' action='/logout' style='margin-left:auto'><button class='ghost'>Disconnect</button></form></div>{note}")
        else:
            acct = ("<div class='card'><p>Connect your TikTok account to post your clips.</p>"
                    "<a class='btn' href='/login'>Connect TikTok</a></div>")
        clips = [c for c in self.queue.ranked(None, 50) if c.reel_path and Path(c.reel_path).is_file()]
        runs = "".join(
            f"<div class='card row'><div><b>{e(c.channel)}</b> &middot; clip #{c.id} &middot; {e(c.status)}<br>"
            f"<span class='mut'>{e(str(c.receipt.get('hook', '')))}</span>"
            + ("<br><span class='err'>Held by the nudity check: watch it before posting</span>"
               if c.status == "held" else "") + "</div>"
            f"<a class='btn' style='margin-left:auto' href='/runs/{r['id']}/clip/{c.id}/tiktok'>Review</a></div>"
            for r, c in self.run_clips())
        if runs:
            acct += "<h2>Clips from test runs</h2>" + runs
        if clips:
            items = "".join(
                f"<div class='card row'><div><b>Clip #{c.id}</b> &middot; {e(c.channel)}<br>"
                f"<span class='mut'>score {c.score:.2f} &middot; {e(str(c.receipt.get('why', '')))}</span>"
                + ("<br><span class='err'>Held by the nudity check: watch it before posting</span>"
                   if c.status == "held" else "") + "</div>"
                f"<a class='btn' style='margin-left:auto' href='/clip/{c.id}'>Review</a></div>" for c in clips)
        else:
            items = "<p class='mut'>No rendered clips in your own queue.</p>"
        return page("Clips", acct + "<h2>Your clips</h2>" + items)

    async def login(self, request: web.Request) -> web.Response:
        state = secrets.token_urlsafe(16)
        self.states.add(state)
        raise web.HTTPFound(authorize_url(self.cfg, state))

    async def callback(self, request: web.Request) -> web.Response:
        q = request.query
        if q.get("error"):
            return page("Login cancelled", f"<p class='err'>TikTok login was not completed: {e(q.get('error_description') or q['error'])}</p>")
        state = q.get("state", "")
        if state not in self.states:
            return page("Login failed", "<p class='err'>Invalid login state. Please start again from the home page.</p>")
        self.states.discard(state)
        try:
            tok = await self.client().exchange_code(q["code"])
        except (TikTokError, KeyError) as ex:
            return page("Login failed", f"<p class='err'>{e(str(ex))}</p>")
        self.store.save(tok)
        raise web.HTTPFound("/tiktok")

    async def logout(self, request: web.Request) -> web.Response:
        self.store.clear()
        raise web.HTTPFound("/tiktok")

    async def static_doc(self, request: web.Request) -> web.StreamResponse:
        """Serve the terms/privacy/home pages and TikTok verification files from docs/ on the app's own domain."""
        p = DOCS / request.match_info["name"]
        if not p.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(p)

    async def video(self, request: web.Request) -> web.StreamResponse:
        p = self.clip_path(int(request.match_info["id"]))
        if not p:
            raise web.HTTPNotFound()
        return web.FileResponse(p)

    async def review(self, request: web.Request) -> web.Response:
        t = self.target(request)
        c = t["clip"]
        if not c or not t["path"]:
            raise web.HTTPNotFound()
        cid = c.id
        tok = await self.token()
        if not tok:
            raise web.HTTPFound("/login")
        try:
            info = await self.client().creator_info(tok)
        except TikTokError as ex:
            return page("Review", f"<p class='err'>Could not load your TikTok posting options: {e(str(ex))}</p>")
        opts = info.get("privacy_level_options", [])
        options = "<option value='' selected disabled>Select who can view this video</option>" + "".join(
            f"<option value='{e(o)}'>{e(PRIVACY_LABELS.get(o, o))}</option>" for o in opts)
        held = ("<p class='err'>Held by the nudity check: watch the whole clip before posting.</p>"
                if c.status == "held" else "")
        dis = {k: info.get(f"{k}_disabled", False) for k in ("comment", "duet", "stitch")}
        toggles = "".join(
            f"<label class='opt'><input type='checkbox' name='allow_{k}' {'disabled' if dis[k] else ''}> Allow {k}s"
            f"{' (turned off in your TikTok settings)' if dis[k] else ''}</label>" for k in ("comment", "duet", "stitch"))
        body = f"""
<div class='card row'><div>Posting as <b>@{e(info.get('creator_username', ''))}</b>
 ({e(info.get('creator_nickname', ''))})</div></div>
{held}<video controls playsinline src='{t["video"]}'></video>
<form id='f' class='card' method='post' action='{t["post"]}'>
 <label>Caption<textarea name='title' rows='3' maxlength='2200'>{e(caption(c))}</textarea></label>
 <label>Who can view this video<select name='privacy' id='privacy' required>{options}</select></label>
 <p class='mut' style='margin:6px 0 0'>Unaudited apps can only post as "Only me" until TikTok approves the app.</p>
 <div style='margin-top:10px'><b>Allow users to</b>{toggles}</div>
 <div style='margin-top:10px'><label class='opt'><input type='checkbox' id='disclose' name='disclose'> <b>Disclose commercial content</b></label>
  <div id='brand' hidden style='margin-left:24px'>
   <label class='opt'><input type='checkbox' name='your_brand'> Your brand (promoting yourself or your business)</label>
   <label class='opt'><input type='checkbox' name='branded'> Branded content (promoting a third party)</label>
   <p class='mut' id='bnote'></p></div></div>
 <p class='mut' id='declare'>By posting, you agree to TikTok's <a href='https://www.tiktok.com/legal/page/global/music-usage-confirmation/en' target='_blank' rel='noopener'>Music Usage Confirmation</a>.</p>
 <label class='opt'><input type='checkbox' id='consent' name='consent' required> I have reviewed this video and agree to post it to my TikTok account.</label>
 <button id='go' disabled>Post to TikTok</button></form>
<script>
const f=document.getElementById('f'),go=document.getElementById('go');
function sync(){{const d=document.getElementById('disclose').checked,b=document.getElementById('brand');b.hidden=!d;
 const p=document.getElementById('privacy').value,br=f.branded.checked,yb=f.your_brand.checked;
 const sel=(!d)||br||yb;let note='';
 if(d&&br&&p==='SELF_ONLY'){{note='Branded content visibility cannot be set to private.';}}
 else if(d&&!yb&&!br){{note='Select at least one option to disclose.';}}
 else if(d&&br){{note="Your video will be labeled 'Paid partnership'.";}} else if(d&&yb){{note="Your video will be labeled 'Promotional content'.";}}
 document.getElementById('bnote').textContent=note;
 go.disabled=!(p&&document.getElementById('consent').checked&&sel&&!(d&&br&&p==='SELF_ONLY'));}}
f.addEventListener('input',sync);f.addEventListener('change',sync);sync();
</script>"""
        return page(f"Review clip {cid}", body)

    async def post(self, request: web.Request) -> web.Response:
        t = self.target(request)
        p = t["path"]
        if not t["clip"] or not p:
            raise web.HTTPNotFound()
        tok = await self.token()
        if not tok:
            raise web.HTTPFound("/login")
        form = await request.post()
        privacy = form.get("privacy", "")
        disclose = "disclose" in form
        yb, br = "your_brand" in form, "branded" in form
        if not privacy or "consent" not in form:
            return page("Not posted", "<p class='err'>Choose who can view the video and confirm the consent box.</p>")
        if disclose and not (yb or br):
            return page("Not posted", "<p class='err'>Select a commercial-content option or turn the disclosure off.</p>")
        if disclose and br and privacy == "SELF_ONLY":
            return page("Not posted", "<p class='err'>Branded content cannot be private.</p>")
        post_info = {
            "title": str(form.get("title", ""))[:2200],
            "privacy_level": privacy,
            "disable_comment": "allow_comment" not in form,
            "disable_duet": "allow_duet" not in form,
            "disable_stitch": "allow_stitch" not in form,
            "video_cover_timestamp_ms": 1000,
        }
        if disclose:
            post_info["brand_content_toggle"] = br
            post_info["brand_organic_toggle"] = yb
        try:
            pid = await self.client().publish_video(tok, p, post_info)
        except TikTokError as ex:
            return page("Not posted", f"<p class='err'>{friendly(ex)}</p><a href='{t['back']}'>Back</a>")
        t["mark"](t["posted_status"])
        body = f"""<div class='card'><p>Upload sent to TikTok. Checking status&hellip;</p><p id='s' class='mut'>PROCESSING</p></div>
<script>
async function poll(){{const r=await fetch('/status/{e(pid)}');const j=await r.json();
 document.getElementById('s').textContent=j.status||j.error||'unknown';
 if(!['PUBLISH_COMPLETE','FAILED'].includes(j.status)&&!j.error)setTimeout(poll,3000);}}poll();
</script>"""
        return page("Posting", body)

    async def status(self, request: web.Request) -> web.Response:
        tok = await self.token()
        if not tok:
            return web.json_response({"error": "not connected"}, status=401)
        try:
            data = await self.client().publish_status(tok, request.match_info["pid"])
        except TikTokError as ex:
            return web.json_response({"error": str(ex)})
        return web.json_response({"status": data.get("status"), "reason": data.get("fail_reason")})


def build_app(cfg: Config, db_path: Path, token_path: Path, password_hash: str | None = None,
              allow_posting: bool = False, control_dir: Path | None = None, keep_days: float = 3,
              ig_cfg: "ig_api.Config | None" = None) -> web.Application:
    """Public: home, terms, privacy, TikTok files, the OAuth callback, sign-in. Everything else needs
    the shared password. The TikTok posting pages exist only when the owner sets allow_posting."""
    app_state = App(cfg, ClipQueue(db_path), TokenStore(token_path))
    sessions = Sessions()
    runs = RunManager(control_dir or db_path.parent / "control", keep_days)
    app_state.runs = runs
    pages = ControlPages(runs, sessions, password_hash, page, posting=allow_posting)
    redirect_path = urlparse(cfg.redirect_uri).path or "/callback"
    insta = InstagramPages(ig_cfg, ig_api.TokenStore(token_path.with_name("instagram_token.json")),
                           runs, page, lambda: app_state.http) if allow_posting else None
    public = re.compile(r"^/(|index\.html|terms\.html|privacy\.html|tiktok[A-Za-z0-9]+\.txt|signin)$")
    media = re.compile(MEDIA_RE)

    def is_public(path: str) -> bool:
        if public.match(path) or path == redirect_path:
            return True     # Meta's servers fetch the reel and Instagram calls back, both signed out
        return bool(insta and (path == insta.redirect_path or media.match(path)))

    app = web.Application(middlewares=[auth_middleware(sessions, is_public)])

    async def home(request: web.Request) -> web.StreamResponse:
        if sessions.valid(request.cookies.get(COOKIE)):
            raise web.HTTPFound("/control")
        return web.FileResponse(DOCS / "index.html")

    async def on_start(_: web.Application) -> None:
        app_state.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))

    async def on_stop(_: web.Application) -> None:
        await app_state.http.close()
        for rid in list(runs.procs):   # don't leave test runs behind when the app stops
            await runs.stop(rid)

    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.add_routes([
        web.get("/", home), web.get(redirect_path, app_state.callback),
        web.get(r"/{name:(?:index|terms|privacy)\.html}", app_state.static_doc),
        web.get(r"/{name:tiktok[A-Za-z0-9]+\.txt}", app_state.static_doc),
        *pages.routes(),
    ])
    if allow_posting:   # owner-only switch (CONTROL_ALLOW_POSTING=1 in .env)
        app.add_routes([
            web.get("/tiktok", app_state.index), web.get("/login", app_state.login),
            web.post("/logout", app_state.logout),
            web.get(r"/video/{id:\d+}", app_state.video), web.get(r"/clip/{id:\d+}", app_state.review),
            web.post(r"/clip/{id:\d+}/post", app_state.post), web.get("/status/{pid}", app_state.status),
            web.get(r"/runs/{rid}/clip/{cid:\d+}/tiktok", app_state.review),
            web.post(r"/runs/{rid}/clip/{cid:\d+}/tiktok/post", app_state.post),
            *insta.routes(),
        ])
    return app
