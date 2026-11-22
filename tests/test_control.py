import asyncio
import sys
import time

import aiohttp
from aiohttp import web

from clipfarm import control
from clipfarm.control import LoginLimiter, RunManager, check_password, hash_password, stream_state
from clipfarm.queue import ClipQueue
from clipfarm.tiktok import api
from clipfarm.tiktok.webapp import build_app

CFG = api.Config("ck", "cs", "https://example.test/callback")


def test_password_hash_roundtrip():
    h = hash_password("secret pw", 1000)
    assert "secret" not in h and check_password("secret pw", h)
    assert not check_password("wrong", h) and not check_password("x", "garbage")


def test_login_limiter():
    lim = LoginLimiter(per_ip=2, total=3, window_s=60)
    lim.fail("a", 0), lim.fail("a", 1)
    assert lim.blocked("a", 2) and not lim.blocked("b", 2)
    lim.fail("b", 3)
    assert lim.blocked("c", 4)            # global cap
    assert not lim.blocked("a", 100)      # window passed


def test_stream_state():
    assert stream_state("") == "starting"
    assert stream_state("[x] monitoring twitch chat") == "live"
    assert stream_state("monitoring\n[x] stream recorder exited (stream offline?)") == "offline"


def fake_cmd(run_dir, channel, game, minutes):
    # stands in for `clipfarm run`: makes one clip, then waits to be stopped
    code = ("import sys,time,pathlib;from clipfarm.queue import ClipQueue;d=pathlib.Path(sys.argv[1]);"
            "r=d/'r.mp4';r.write_bytes(b'0'*10);q=ClipQueue(d/'queue.db');"
            "i=q.add('c','twitch',0,1,42.0,{'why':'spike','signals':{'audio':0.5}});"
            "q.update(i,reel_path=str(r));print('monitoring',flush=True);time.sleep(60)")
    return [sys.executable, "-c", code, str(run_dir)]


def test_rejects_bad_input(tmp_path):
    m = RunManager(tmp_path, cmd=fake_cmd)
    for ch, game, mins in [("bad;rm -rf", "", 5), ("-flag", "", 5), ("ok_name", "nope", 5),
                           ("ok_name", "", 0), ("ok_name", "", control.MAX_MINUTES + 1)]:
        try:
            m.start(ch, game, mins)
        except ValueError:
            continue
        raise AssertionError(f"accepted {ch!r} {game!r} {mins}")


def test_queue_stop_and_cleanup(tmp_path):
    async def run():
        m = RunManager(tmp_path, cmd=fake_cmd, max_running=1)
        a = m.start("chan_one", "", 5)
        b = m.start("chan_two", "", 5)
        assert a["status"] == "running" and b["status"] == "waiting"
        for _ in range(100):
            if m.clips(a["id"]):
                break
            await asyncio.sleep(0.1)
        assert m.clips(a["id"])[0].score == 42.0
        assert m.clip_file(a["id"], 1).name == "r.mp4"
        await m.stop(a["id"])
        for _ in range(50):
            if a["status"] == "stopped" and b["status"] == "running":
                break
            await asyncio.sleep(0.1)
        assert a["status"] == "stopped" and b["status"] == "running"   # next in line started
        await m.stop(b["id"])
        await asyncio.gather(*m.tasks)
        m.cleanup(now=time.time() + 4 * 86400)
        return m, a

    m, a = asyncio.run(run())
    assert not m.runs and not (tmp_path / a["id"]).exists()


def test_web_auth_and_control(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "default_cmd", fake_cmd)

    async def run():
        app = build_app(CFG, tmp_path / "q.db", tmp_path / "tok.json", password_hash=hash_password("pw", 1000),
                        control_dir=tmp_path / "control")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        out = {}
        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as s:
            out["anon_control"] = (await s.get(base + "/control", allow_redirects=False)).status
            out["anon_start"] = (await s.post(base + "/control/start", data={"channel": "x"})).status
            out["home"] = (await s.get(base + "/")).status
            out["callback"] = (await s.get(base + "/callback?error=x")).status
            out["tiktok_off"] = (await s.get(base + "/clip/1", allow_redirects=False)).status
            out["wrong"] = (await s.post(base + "/signin", data={"password": "no"})).status
            await s.post(base + "/signin", data={"password": "pw"})
            out["control"] = await (await s.get(base + "/control")).text()
            out["bad"] = (await s.post(base + "/control/start",
                                       data={"channel": "a b", "minutes": "5"})).status
            await s.post(base + "/control/start", data={"channel": "chan_one", "minutes": "5"})
            out["posting_hidden"] = (await s.get(base + "/tiktok")).status
        await runner.cleanup()   # app shutdown; the fake run is killed below
        return out

    out = asyncio.run(run())
    assert out["anon_control"] == 302 and out["anon_start"] == 401
    assert out["home"] == 200 and out["callback"] == 200
    assert out["tiktok_off"] == 302 and out["wrong"] == 401
    assert "Start a test run" in out["control"] and out["bad"] == 400
    assert out["posting_hidden"] == 404
