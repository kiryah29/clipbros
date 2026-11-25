import asyncio
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import aiohttp
from aiohttp import web

from clipfarm.control import hash_password
from clipfarm.queue import ClipQueue
from clipfarm.tiktok import api
from clipfarm.tiktok.webapp import build_app

CFG = api.Config("ck", "cs", "https://example.test/callback")


def test_authorize_url():
    q = parse_qs(urlparse(api.authorize_url(CFG, "st")).query)
    assert q["client_key"] == ["ck"] and q["state"] == ["st"]
    assert q["scope"] == ["user.info.basic,video.publish"] and q["response_type"] == ["code"]


def fake_tiktok(calls):
    async def init(req):
        body = await req.json()
        calls["init"] = body
        return web.json_response({"data": {"publish_id": "p1", "upload_url": f"http://{req.host}/up"},
                                  "error": {"code": "ok"}})

    async def up(req):
        calls.setdefault("chunks", []).append((req.headers["Content-Range"], len(await req.read())))
        return web.Response(status=201)

    async def creator(req):
        return web.json_response({"data": {"creator_username": "bob", "creator_nickname": "Bob",
                                           "privacy_level_options": ["SELF_ONLY"], "duet_disabled": True},
                                  "error": {"code": "ok"}})

    async def status(req):
        return web.json_response({"data": {"status": "PUBLISH_COMPLETE"}, "error": {"code": "ok"}})

    async def user(req):
        return web.json_response({"data": {"user": {"display_name": "Bob", "avatar_url": ""}},
                                  "error": {"code": "ok"}})

    app = web.Application(client_max_size=10 * 1024 * 1024)
    app.add_routes([web.get("/v2/user/info/", user),
                    web.post("/v2/post/publish/video/init/", init), web.put("/up", up),
                    web.post("/v2/post/publish/creator_info/query/", creator),
                    web.post("/v2/post/publish/status/fetch/", status)])
    return app


def test_publish_flow(tmp_path, monkeypatch):
    calls = {}
    video = tmp_path / "v.mp4"
    monkeypatch.setattr(api, "MAX_CHUNK", 2000)
    video.write_bytes(b"x" * 5000)  # 2 chunks: 0-1999, then the remainder 2000-4999

    async def run():
        runner = web.AppRunner(fake_tiktok(calls))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        monkeypatch.setattr(api, "API", f"http://127.0.0.1:{port}")
        async with aiohttp.ClientSession() as s:
            c = api.TikTokClient(CFG, s)
            tok = api.Token({"access_token": "t", "expires_in": 3600})
            info = await c.creator_info(tok)
            pid = await c.publish_video(tok, video, {"title": "hi", "privacy_level": "SELF_ONLY"})
            st = await c.publish_status(tok, pid)
        await runner.cleanup()
        return info, pid, st

    info, pid, st = asyncio.run(run())
    assert info["creator_username"] == "bob" and pid == "p1" and st["status"] == "PUBLISH_COMPLETE"
    assert calls["init"]["source_info"]["total_chunk_count"] == 2
    assert calls["chunks"] == [("bytes 0-1999/5000", 2000), ("bytes 2000-4999/5000", 3000)]


def test_store_roundtrip(tmp_path):
    st = api.TokenStore(tmp_path / "t.json")
    assert st.load() is None
    st.save(api.Token({"access_token": "a", "refresh_token": "r", "open_id": "o", "expires_in": 100}))
    t = st.load()
    assert t.access_token == "a" and t.refresh_token == "r" and not t.expired
    st.clear()
    assert st.load() is None


def test_error_mapping():
    try:
        api._check({"error": {"code": "access_token_invalid", "message": "bad"}})
    except api.TikTokError as ex:
        assert "access_token_invalid" in str(ex)
    else:
        raise AssertionError


def test_review_page_enforces_ux(tmp_path, monkeypatch):
    calls = {}
    reel = tmp_path / "r.mp4"
    reel.write_bytes(b"0" * 100)
    q = ClipQueue(tmp_path / "q.db")
    cid = q.add("ch", "twitch", 0, 10, 5.0, {"why": "spike"})
    q.update(cid, reel_path=str(reel))
    tokfile = tmp_path / "tok.json"
    api.TokenStore(tokfile).save(api.Token({"access_token": "t", "expires_in": 3600}))

    async def run():
        runner = web.AppRunner(fake_tiktok(calls))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        monkeypatch.setattr(api, "API", f"http://127.0.0.1:{port}")
        app = build_app(CFG, tmp_path / "q.db", tokfile, password_hash=hash_password("pw", 1000),
                        allow_posting=True)
        r2 = web.AppRunner(app)
        await r2.setup()
        s2 = web.TCPSite(r2, "127.0.0.1", 0)
        await s2.start()
        p2 = s2._server.sockets[0].getsockname()[1]
        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as s:
            await s.post(f"http://127.0.0.1:{p2}/signin", data={"password": "pw"})
            review = await (await s.get(f"http://127.0.0.1:{p2}/clip/{cid}")).text()
            no_consent = await (await s.post(f"http://127.0.0.1:{p2}/clip/{cid}/post",
                                             data={"title": "t", "privacy": "SELF_ONLY"})).text()
            ok = await (await s.post(f"http://127.0.0.1:{p2}/clip/{cid}/post",
                                     data={"title": "t", "privacy": "SELF_ONLY", "consent": "on"})).text()
        await r2.cleanup()
        await runner.cleanup()
        return review, no_consent, ok

    review, no_consent, ok = asyncio.run(run())
    assert "name='consent'" in review  # the checkbox must be submitted with the form
    assert "Clip from ch #fyp" in review
    assert "@bob" in review and "selected disabled" in review  # no default privacy
    assert "Music Usage Confirmation" in review and "turned off in your TikTok settings" in review
    assert "Not posted" in no_consent and "Upload sent" in ok
    assert calls["init"]["post_info"]["disable_comment"] is True  # toggles off by default
    assert calls["init"]["post_info"]["privacy_level"] == "SELF_ONLY"


def test_serves_legal_pages_and_verification_files(tmp_path):
    from clipfarm.tiktok import webapp
    app = build_app(CFG, tmp_path / "q.db", tmp_path / "tok.json")

    async def run():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        out = {}
        async with aiohttp.ClientSession() as s:
            for path in ("/terms.html", "/privacy.html", "/index.html", "/tiktokMCbyaXKMPCaRybGqBwI6Uulk30HpaWTS.txt",
                         "/tiktokNope.txt", "/..%2FREADME.md", "/tiktok..%2f..%2fREADME.txt"):
                r = await s.get(f"http://127.0.0.1:{port}{path}", allow_redirects=False)
                out[path] = (r.status, (await r.text())[:80])
        await runner.cleanup()
        return out

    out = asyncio.run(run())
    assert out["/terms.html"][0] == 200 and "Terms of Service" in out["/terms.html"][1] + "Terms of Service"
    assert out["/privacy.html"][0] == 200 and out["/index.html"][0] == 200
    assert out["/tiktokMCbyaXKMPCaRybGqBwI6Uulk30HpaWTS.txt"][0] == 200
    assert "tiktok-developers-site-verification=" in out["/tiktokMCbyaXKMPCaRybGqBwI6Uulk30HpaWTS.txt"][1]
    assert out["/tiktokNope.txt"][0] == 404
    # not public: sent to sign-in (302), never served
    assert out["/..%2FREADME.md"][0] in (302, 404) and out["/tiktok..%2f..%2fREADME.txt"][0] in (302, 404)


def test_friendly_error_for_unaudited_private_rule():
    from clipfarm.tiktok.webapp import friendly
    msg = friendly(api.TikTokError("unaudited_client_can_only_post_to_private_accounts: blocked"))
    assert "PRIVATE TikTok account" in msg and "Private account" in msg
    assert friendly(api.TikTokError("some_other: <b>x</b>")) == "some_other: &lt;b&gt;x&lt;/b&gt;"


def test_default_caption(monkeypatch):
    from clipfarm.tiktok.webapp import default_caption
    monkeypatch.delenv("TIKTOK_DEFAULT_CAPTION", raising=False)
    assert default_caption("xqc") == "Clip from xqc #fyp #streamer #clips"
    monkeypatch.setenv("TIKTOK_DEFAULT_CAPTION", "{channel} highlights #twitch")
    assert default_caption("xqc") == "xqc highlights #twitch"


def test_clips_from_test_runs_can_be_reviewed_and_posted(tmp_path, monkeypatch):
    """The live clips sit in control run folders, not in the own queue: they must be listed
    on /tiktok, get a Post button on /clips, and be marked 'posted' afterwards."""
    import time as _time
    calls = {}
    rid = "20261125-173803-7542"
    run_dir = tmp_path / "control" / rid
    (run_dir / "clips").mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(
        {"id": rid, "channel": "nickmercs", "game": "gta6", "minutes": 30, "status": "done",
         "created": _time.time(), "started": _time.time(), "ended": _time.time()}))
    reel = run_dir / "clips" / "00003_reel.mp4"
    reel.write_bytes(b"0" * 100)
    rq = ClipQueue(run_dir / "queue.db")
    cid = rq.add("nickmercs", "twitch", 0, 10, 5.0, {"hook": 'Nick: "DUDE DUDE DUDE"'})
    rq.update(cid, reel_path=str(reel), status="crafted")
    rq.db.close()
    tokfile = tmp_path / "tok.json"
    api.TokenStore(tokfile).save(api.Token({"access_token": "t", "expires_in": 3600}))

    async def run():
        runner = web.AppRunner(fake_tiktok(calls))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        monkeypatch.setattr(api, "API", f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}")
        app = build_app(CFG, tmp_path / "q.db", tokfile, password_hash=hash_password("pw", 1000),
                        allow_posting=True, control_dir=tmp_path / "control")
        r2 = web.AppRunner(app)
        await r2.setup()
        s2 = web.TCPSite(r2, "127.0.0.1", 0)
        await s2.start()
        base = f"http://127.0.0.1:{s2._server.sockets[0].getsockname()[1]}"
        url = f"{base}/runs/{rid}/clip/{cid}/tiktok"
        out = {}
        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as s:
            await s.post(base + "/signin", data={"password": "pw"})
            out["index"] = await (await s.get(base + "/tiktok")).text()
            out["clips"] = await (await s.get(base + "/clips")).text()
            out["review"] = await (await s.get(url)).text()
            out["bad_run"] = (await s.get(f"{base}/runs/nope/clip/{cid}/tiktok")).status
            out["ok"] = await (await s.post(url + "/post", data={
                "title": "t", "privacy": "SELF_ONLY", "consent": "on"})).text()
            out["index_after"] = await (await s.get(base + "/tiktok")).text()
        await r2.cleanup()
        await runner.cleanup()
        return out

    out = asyncio.run(run())
    assert f"/runs/{rid}/clip/{cid}/tiktok" in out["index"] and "Clips from test runs" in out["index"]
    assert f"/runs/{rid}/clip/{cid}/tiktok'>Post to TikTok" in out["clips"]
    assert f"src='/runs/{rid}/clip/{cid}.mp4'" in out["review"]
    assert "Nick: &quot;DUDE DUDE DUDE&quot;" in out["review"]          # hook prefilled in the caption
    assert out["bad_run"] == 404 and "Upload sent" in out["ok"]
    assert ClipQueue(run_dir / "queue.db").get(cid).status == "posted"
    assert f"/runs/{rid}/clip/{cid}/tiktok" not in out["index_after"]    # posted: off the list


def test_post_buttons_hidden_without_the_switch(tmp_path):
    from clipfarm.control import ControlPages, RunManager, Sessions
    pages = ControlPages(RunManager(tmp_path / "control"), Sessions(), None, None)
    assert pages.posting is False
