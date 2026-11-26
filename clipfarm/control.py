"""Password-protected control page: start/stop test clipping runs and review their clips.

Each run gets its own folder data/control/<run id>/ (own queue.db, clips, buffer and log),
so it never touches the owner's own queue. At most MAX_RUNNING runs at once; extra runs
wait in line. Old run folders are deleted after CONTROL_KEEP_DAYS.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from aiohttp import web

from .config import ROOT, load_channels
from .feedback import FILE as FEEDBACK_FILE, FeedbackStore
from .feedback_web import ReviewPages
from .queue import ClipQueue
from .streamers import FILE as STREAMERS_FILE, StreamerRegistry

CHANNEL_RE = re.compile(r"^[A-Za-z0-9_]{3,25}$")   # Twitch login names; no '-', so never a flag
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
MAX_MINUTES = 120
MAX_RUNNING = 2
MAX_WAITING = 5
SESSION_S = 12 * 3600
COOKIE = "cb_session"
ACTIVE = ("waiting", "running")
e = html.escape


# --------------------------------------------------------------------------- auth
def hash_password(pw: str, iterations: int = 600_000) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${dk.hex()}"


def check_password(pw: str, stored: str) -> bool:
    try:
        algo, iterations, salt, want = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(iterations))
    except ValueError:
        return False
    return algo == "pbkdf2_sha256" and hmac.compare_digest(dk.hex(), want)


class LoginLimiter:
    """Blocks logins after `per_ip` failures from one IP, or `total` from everyone, in `window_s`."""

    def __init__(self, per_ip: int = 5, total: int = 30, window_s: float = 900):
        self.per_ip, self.total, self.window = per_ip, total, window_s
        self.fails: list[tuple[float, str]] = []

    def blocked(self, ip: str, now: float | None = None) -> bool:
        now = now or time.time()
        self.fails = [f for f in self.fails if now - f[0] < self.window]
        return len(self.fails) >= self.total or sum(i == ip for _, i in self.fails) >= self.per_ip

    def fail(self, ip: str, now: float | None = None) -> None:
        self.fails.append((now or time.time(), ip))


class Sessions:
    """Random session tokens kept in memory (an app restart signs everyone out)."""

    def __init__(self):
        self.live: dict[str, float] = {}

    def new(self) -> str:
        tok = secrets.token_urlsafe(32)
        self.live[tok] = time.time() + SESSION_S
        return tok

    def valid(self, tok: str | None) -> bool:
        exp = self.live.get(tok or "")
        if exp and exp > time.time():
            return True
        self.live.pop(tok or "", None)
        return False

    def end(self, tok: str | None) -> None:
        self.live.pop(tok or "", None)


def client_ip(request: web.Request) -> str:
    # the app listens on 127.0.0.1 behind the Cloudflare tunnel, which sets this header
    return request.headers.get("CF-Connecting-IP") or request.remote or "?"


def auth_middleware(sessions: Sessions, is_public: Callable[[str], bool]):
    @web.middleware
    async def auth(request: web.Request, handler):
        if is_public(request.path) or sessions.valid(request.cookies.get(COOKIE)):
            return await handler(request)
        if request.method == "GET":
            raise web.HTTPFound("/signin")
        raise web.HTTPUnauthorized(text="Sign in first.")
    return auth


# --------------------------------------------------------------------------- runs
def games() -> list[str]:
    return sorted(p.stem for p in (ROOT / "config" / "games").glob("*.yaml")
                  if not p.stem.startswith("_"))     # _template.yaml is not a game


def known_channels(registry: StreamerRegistry | None = None) -> list[str]:
    """Twitch names from channels.yaml plus every Twitch streamer the app has saved."""
    saved = registry.names("twitch") if registry else []
    try:
        cfg = [c["name"] for c in load_channels() if c.get("platform") == "twitch"]
    except (OSError, KeyError, TypeError, AttributeError):
        cfg = []
    return sorted(set(cfg) | set(saved))


def default_cmd(run_dir: Path, channel: str, game: str, minutes: int) -> list[str]:
    cmd = [sys.executable, "-u", "-m", "clipfarm", "--data-dir", str(run_dir),
           "run", "twitch", channel, "--minutes", str(minutes)]
    return cmd + (["--game", game] if game else [])


def stream_state(log: str) -> str:
    if "stream offline" in log:
        return "offline"
    return "live" if "monitoring" in log else "starting"


class RunManager:
    def __init__(self, root: Path, keep_days: float = 3, cmd=default_cmd, max_running: int = MAX_RUNNING,
                 streamers: StreamerRegistry | None = None):
        self.root, self.keep_s, self.cmd, self.max_running = root, keep_days * 86400, cmd, max_running
        # kept next to the run folders' parent (data/streamers.json), so it outlives cleanup
        self.streamers = streamers or StreamerRegistry(root.parent / STREAMERS_FILE)
        # one review store for all runs, next to data/control/, so run cleanup never deletes it
        self.feedback = FeedbackStore(root.parent / FEEDBACK_FILE)
        root.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, dict] = {}
        self.procs: dict[str, asyncio.subprocess.Process] = {}
        self.tasks: set[asyncio.Task] = set()
        for f in root.glob("*/run.json"):
            try:
                r = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            self.runs[r["id"]] = r
            if r["status"] in ACTIVE:   # the app restarted; its runs died with it
                r["status"], r["ended"] = "interrupted", r.get("ended") or time.time()
                self._save(r)
        self.cleanup()

    def dir(self, rid: str) -> Path:
        return self.root / rid

    def _save(self, r: dict) -> None:
        (self.dir(r["id"]) / "run.json").write_text(json.dumps(r))

    def count(self, status: str) -> int:
        return sum(r["status"] == status for r in self.runs.values())

    def start(self, channel: str, game: str, minutes: int) -> dict:
        if not CHANNEL_RE.match(channel):
            raise ValueError("Channel names are 3-25 letters, digits or _.")
        if game and game not in games():
            raise ValueError("Unknown game profile.")
        if not 1 <= minutes <= MAX_MINUTES:
            raise ValueError(f"Pick 1 to {MAX_MINUTES} minutes.")
        if self.count("waiting") >= MAX_WAITING:
            raise ValueError(f"{MAX_WAITING} runs are already waiting. Stop one or wait.")
        rid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        self.dir(rid).mkdir(parents=True)
        r = {"id": rid, "channel": channel.lower(), "game": game, "minutes": minutes, "status": "waiting",
             "created": time.time(), "started": None, "ended": None}
        self.runs[rid] = r
        self._save(r)
        self.streamers.record(r["channel"], "twitch", game, source="control")
        self._pump()
        return r

    def _pump(self) -> None:
        waiting = sorted((r for r in self.runs.values() if r["status"] == "waiting"), key=lambda r: r["created"])
        for r in waiting[:max(0, self.max_running - self.count("running"))]:
            r["status"], r["started"] = "running", time.time()
            self._save(r)
            t = asyncio.get_running_loop().create_task(self._run(r))
            self.tasks.add(t)
            t.add_done_callback(self.tasks.discard)

    async def _run(self, r: dict) -> None:
        d = self.dir(r["id"])
        # streamlink lives next to the venv's python; the run must find it on PATH
        env = {**os.environ, "PYTHONIOENCODING": "utf-8",
               "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
        extra = {} if sys.platform == "win32" else {"start_new_session": True}
        try:
            with open(d / "run.log", "ab") as log:
                proc = await asyncio.create_subprocess_exec(
                    *self.cmd(d, r["channel"], r["game"], r["minutes"]), cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **extra)
            self.procs[r["id"]] = proc
            if r.get("stop"):   # Stop was pressed while it was starting
                await self._kill(proc)
            try:   # the run stops itself at its time limit; this is only the backstop
                await asyncio.wait_for(proc.wait(), r["minutes"] * 60 + 120)
            except asyncio.TimeoutError:
                await self._kill(proc)
            r["status"] = "stopped" if r.get("stop") else ("finished" if proc.returncode == 0 else "failed")
        except OSError as ex:
            r["status"] = "failed"
            with open(d / "run.log", "a", encoding="utf-8") as log:
                log.write(f"could not start: {ex}\n")
        finally:
            self.procs.pop(r["id"], None)
            r["ended"] = time.time()
            self._save(r)
            self._pump()

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process) -> None:
        """Kill the run and its streamlink/ffmpeg children (whole tree / process group)."""
        if proc.returncode is not None:
            return
        if sys.platform == "win32":
            k = await asyncio.create_subprocess_exec("taskkill", "/PID", str(proc.pid), "/T", "/F",
                                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            await k.wait()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    async def stop(self, rid: str) -> None:
        r = self.runs.get(rid)
        if not r or r["status"] not in ACTIVE:
            return
        r["stop"] = True
        if r["status"] == "waiting":
            r["status"], r["ended"] = "stopped", time.time()
            self._save(r)
        elif rid in self.procs:
            await self._kill(self.procs[rid])

    def cleanup(self, now: float | None = None) -> None:
        now = now or time.time()
        for rid, r in list(self.runs.items()):
            if r["status"] not in ACTIVE and now - (r.get("ended") or r["created"]) > self.keep_s:
                shutil.rmtree(self.dir(rid), ignore_errors=True)
                del self.runs[rid]

    # -- clips -------------------------------------------------------------------------
    def _queue(self, rid: str) -> ClipQueue | None:
        db = self.dir(rid) / "queue.db"
        return ClipQueue(db) if rid in self.runs and db.is_file() else None

    def clips(self, rid: str) -> list:
        q = self._queue(rid)
        if not q:
            return []
        try:
            return q.ranked(None, 200)
        finally:
            q.db.close()

    def clip(self, rid: str, cid: int):
        q = self._queue(rid)
        if not q:
            return None
        try:
            return q.get(cid)
        finally:
            q.db.close()

    def clip_file(self, rid: str, cid: int) -> Path | None:
        q = self._queue(rid)
        if not q:
            return None
        try:
            c = q.get(cid)
        finally:
            q.db.close()
        p = Path(c.reel_path).resolve() if c and c.reel_path else None
        return p if p and p.is_file() and p.is_relative_to(self.dir(rid).resolve()) else None

    def update_receipt(self, rid: str, cid: int, patch: dict) -> None:
        q = self._queue(rid)
        if q:
            try:
                c = q.get(cid)
                if c:
                    q.update(cid, receipt={**c.receipt, **patch})
            finally:
                q.db.close()

    def set_status(self, rid: str, cid: int, status: str) -> None:
        q = self._queue(rid)
        if q:
            try:
                if q.get(cid):
                    q.update(cid, status=status)
            finally:
                q.db.close()

    def log_tail(self, rid: str, size: int = 4000) -> str:
        p = self.dir(rid) / "run.log"
        if not p.is_file():
            return ""
        with open(p, "rb") as f:
            f.seek(max(0, p.stat().st_size - size))
            return f.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------- pages
class ControlPages(ReviewPages):
    """The /clips review page, approve/reject and the review API come from ReviewPages
    (feedback_web.py)."""

    def __init__(self, runs: RunManager, sessions: Sessions, password_hash: str | None, page,
                 posting: bool = False):
        self.runs, self.sessions, self.hash, self.page = runs, sessions, password_hash, page
        self.limiter = LoginLimiter()
        self.posting = posting   # the owner turned TikTok posting on (CONTROL_ALLOW_POSTING=1)

    def _signin_form(self, msg: str = "") -> str:
        return (f"{msg}<form class='card' method='post' action='/signin'>"
                "<label>Password<input type='password' name='password' autocomplete='current-password' "
                "required autofocus></label><p><button>Sign in</button></p></form>")

    async def signin(self, request: web.Request) -> web.Response:
        if self.sessions.valid(request.cookies.get(COOKIE)):
            raise web.HTTPFound("/control")
        return self.page("Sign in", "<h2>Sign in</h2>" + self._signin_form())

    async def signin_post(self, request: web.Request) -> web.Response:
        if not self.hash:
            return self.page("Sign in", "<p class='err'>Sign-in is not set up yet (no CONTROL_PASSWORD_HASH).</p>",
                             status=503)
        ip = client_ip(request)
        if self.limiter.blocked(ip):
            return self.page("Sign in", "<p class='err'>Too many wrong passwords. Try again in 15 minutes.</p>",
                             status=429)
        form = await request.post()
        pw = str(form.get("password", ""))[:256]
        if not await asyncio.to_thread(check_password, pw, self.hash):
            self.limiter.fail(ip)
            return self.page("Sign in", "<h2>Sign in</h2>" + self._signin_form("<p class='err'>Wrong password.</p>"),
                             status=401)
        resp = web.HTTPFound("/control")
        https = request.secure or request.headers.get("X-Forwarded-Proto") == "https"
        resp.set_cookie(COOKIE, self.sessions.new(), max_age=SESSION_S, httponly=True, samesite="Lax",
                        secure=https, path="/")
        raise resp

    async def signout(self, request: web.Request) -> web.Response:
        self.sessions.end(request.cookies.get(COOKIE))
        resp = web.HTTPFound("/")
        resp.del_cookie(COOKIE, path="/")
        raise resp

    def runs_html(self) -> str:
        rows = sorted(self.runs.runs.values(), key=lambda r: r["created"], reverse=True)[:20]
        if not rows:
            return "<p class='mut'>No runs yet.</p>"
        out = []
        for r in rows:
            n = len(self.runs.clips(r["id"]))
            log = self.runs.log_tail(r["id"]) if r["status"] == "running" else ""
            state = f" &middot; stream {stream_state(log)}" if r["status"] == "running" else ""
            last = log.strip().splitlines()[-1][:160] if log.strip() else ""
            when = time.strftime("%d %b %H:%M", time.localtime(r["started"] or r["created"]))
            stop = (f"<form method='post' action='/control/runs/{r['id']}/stop'><button class='ghost'>Stop</button></form>"
                    if r["status"] in ACTIVE else "")
            out.append(
                f"<div class='card row'><div><b>{e(r['channel'])}</b> &middot; {e(r['game'] or 'default')} &middot; "
                f"{r['minutes']} min<br><span class='mut'>{when} &middot; <b>{e(r['status'])}</b>{state} &middot; "
                f"{n} clip{'s' if n != 1 else ''}</span>{f'<br><code class=mut>{e(last)}</code>' if last else ''}</div>"
                f"<div class='row' style='margin-left:auto'><a class='btn ghost' href='/clips?run={r['id']}'>Clips</a>{stop}</div></div>")
        return "".join(out)

    def streamers_html(self) -> str:
        rows = self.runs.streamers.all()
        if not rows:
            return "<p class='mut'>Streamers are saved here when you start their first run.</p>"
        day = lambda t: time.strftime("%d %b %Y", time.localtime(t))  # noqa: E731
        return "".join(
            f"<div class='card'><b>{e(s['name'])}</b> &middot; {e(s.get('platform', ''))} &middot; "
            f"{e(s.get('game') or 'default')}<br><span class='mut'>first {day(s.get('first_seen', 0))} &middot; "
            f"last {day(s.get('last_seen', 0))} &middot; {s.get('runs', 0)} run{'s' if s.get('runs') != 1 else ''}"
            "</span></div>" for s in rows)

    async def control(self, request: web.Request) -> web.Response:
        self.runs.cleanup()
        chans = "".join(f"<option value='{e(c)}'>" for c in known_channels(self.runs.streamers))
        gopts = "".join(f"<option value='{e(g)}'>{e(g)}</option>" for g in games())
        body = f"""
<div class='row'><h2 style='margin-right:auto'>Start a test run</h2>
 <form method='post' action='/signout'><button class='ghost'>Sign out</button></form></div>
<form class='card' method='post' action='/control/start'>
 <label>Twitch channel<input type='text' name='channel' list='chans' required maxlength='25'
  pattern='[A-Za-z0-9_]{{3,25}}' autocomplete='off' autocapitalize='none' placeholder='e.g. ishowspeed'></label>
 <datalist id='chans'>{chans}</datalist>
 <label>Game profile<select name='game'><option value=''>Channel default</option>{gopts}</select></label>
 <label>Minutes (1-{MAX_MINUTES})<input type='number' name='minutes' min='1' max='{MAX_MINUTES}' value='30' required></label>
 <p><button>Start run</button></p>
 <p class='mut'>Up to {MAX_RUNNING} runs at once; extra runs wait in line. Nothing is posted anywhere.</p></form>
<h2>Runs</h2><div id='runs'>{self.runs_html()}</div>
<h2>Saved streamers</h2>{self.streamers_html()}
<script>setInterval(async()=>{{const r=await fetch('/control/runs');
 if(r.ok)document.getElementById('runs').innerHTML=await r.text();}},5000);</script>"""
        return self.page("Control", body)

    async def runs_fragment(self, request: web.Request) -> web.Response:
        return web.Response(text=self.runs_html(), content_type="text/html")

    async def start(self, request: web.Request) -> web.Response:
        form = await request.post()
        try:
            minutes = int(str(form.get("minutes", "")))
        except ValueError:
            minutes = 0   # start() rejects it with the allowed range
        try:
            self.runs.start(str(form.get("channel", "")).strip(), str(form.get("game", "")), minutes)
        except ValueError as ex:
            return self.page("Not started", f"<p class='err'>{e(str(ex))}</p><a href='/control'>Back</a>", status=400)
        raise web.HTTPFound("/control")

    def _rid(self, request: web.Request) -> str:
        rid = request.match_info["rid"]
        if not RUN_ID_RE.match(rid) or rid not in self.runs.runs:
            raise web.HTTPNotFound()
        return rid

    async def stop(self, request: web.Request) -> web.Response:
        await self.runs.stop(self._rid(request))
        raise web.HTTPFound("/control")

    async def video(self, request: web.Request) -> web.StreamResponse:
        rid = self._rid(request)
        cid = int(request.match_info["cid"])
        p = self.runs.clip_file(rid, cid)
        if not p:
            raise web.HTTPNotFound()
        headers = ({"Content-Disposition": f'attachment; filename="clipbros-{self.runs.runs[rid]["channel"]}-{cid}.mp4"'}
                   if request.query.get("dl") else None)
        return web.FileResponse(p, headers=headers)

    def routes(self) -> list:
        clip = "/runs/{rid}/clip/{cid:\\d+}"
        return [
            web.get("/signin", self.signin), web.post("/signin", self.signin_post),
            web.post("/signout", self.signout),
            web.get("/control", self.control), web.get("/control/runs", self.runs_fragment),
            web.post("/control/start", self.start), web.post("/control/runs/{rid}/stop", self.stop),
            web.get("/clips", self.clips), web.get(clip + ".mp4", self.video),
            web.post(clip + "/{action:approve|reject}", self.mark),
            *self.review_routes(),
        ]
