import asyncio
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import aiohttp
from aiohttp import web

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

    app = web.Application(client_max_size=10 * 1024 * 1024)
    app.add_routes([web.post("/v2/post/publish/video/init/", init), web.put("/up", up),
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
        app = build_app(CFG, tmp_path / "q.db", tokfile)
        r2 = web.AppRunner(app)
        await r2.setup()
        s2 = web.TCPSite(r2, "127.0.0.1", 0)
        await s2.start()
        p2 = s2._server.sockets[0].getsockname()[1]
        async with aiohttp.ClientSession() as s:
            review = await (await s.get(f"http://127.0.0.1:{p2}/clip/{cid}")).text()
            no_consent = await (await s.post(f"http://127.0.0.1:{p2}/clip/{cid}/post",
                                             data={"title": "t", "privacy": "SELF_ONLY"})).text()
            ok = await (await s.post(f"http://127.0.0.1:{p2}/clip/{cid}/post",
                                     data={"title": "t", "privacy": "SELF_ONLY", "consent": "on"})).text()
        await r2.cleanup()
        await runner.cleanup()
        return review, no_consent, ok

    review, no_consent, ok = asyncio.run(run())
    assert "@bob" in review and "selected disabled" in review  # no default privacy
    assert "Music Usage Confirmation" in review and "turned off in your TikTok settings" in review
    assert "Not posted" in no_consent and "Upload sent" in ok
    assert calls["init"]["post_info"]["disable_comment"] is True  # toggles off by default
    assert calls["init"]["post_info"]["privacy_level"] == "SELF_ONLY"
