"""Instagram: connect, review a run clip, post it as a reel through a fake Graph API."""
import asyncio
import json
import time
from urllib.parse import parse_qs, urlparse

import aiohttp
from aiohttp import web

from clipfarm.control import hash_password
from clipfarm.instagram import api as ig
from clipfarm.queue import ClipQueue
from clipfarm.tiktok import api as tt
from clipfarm.tiktok.webapp import build_app

TT = tt.Config("ck", "cs", "https://clipbros.test/callback")
IG = ig.Config("app", "secret", "https://clipbros.test/callback/instagram")


def fake_graph(calls, pages=({"id": "p1"}, {"id": "p2"})):
    async def oauth(req):
        q = dict(req.query)
        if q.get("grant_type") == "fb_exchange_token":
            calls["exchange"] = q
            return web.json_response({"access_token": "long", "token_type": "bearer"})
        calls["code"] = q
        return web.json_response({"access_token": "short", "token_type": "bearer", "expires_in": 5000})

    async def accounts(req):
        return web.json_response({"data": list(pages)})

    async def node(req):
        i = req.match_info["id"]
        if i == "p1":                            # a Page without an Instagram account
            return web.json_response({"id": "p1", "access_token": "pt1"})
        if i == "p2":
            calls["page_token_with"] = req.query["access_token"]
            return web.json_response({"id": "p2", "access_token": "page-token",
                                      "instagram_business_account": {"id": "1784"}})
        if i == "1784":
            calls["me_token"] = req.query["access_token"]
            return web.json_response({"id": "1784", "username": "clipbros.cb", "profile_picture_url": ""})
        if i == "c1":
            calls["polls"] = calls.get("polls", 0) + 1
            return web.json_response({"status_code": "FINISHED" if calls["polls"] > 1 else "IN_PROGRESS"})
        return web.json_response({"permalink": "https://www.instagram.com/reel/abc/"})

    async def media(req):
        calls["media"] = dict(await req.post())
        return web.json_response({"id": "c1"})

    async def publish(req):
        calls.setdefault("publish", []).append(dict(await req.post()))
        return web.json_response({"id": "m1"})

    app = web.Application()
    v = ig.VERSION
    app.add_routes([web.get(f"/{v}/oauth/access_token", oauth), web.get(f"/{v}/me/accounts", accounts),
                    web.post(f"/{v}/{{ig}}/media", media), web.post(f"/{v}/{{ig}}/media_publish", publish),
                    web.get(f"/{v}/{{id}}", node)])
    return app


def make_run(tmp_path, rid="20261125-173803-7542"):
    run_dir = tmp_path / "control" / rid
    (run_dir / "clips").mkdir(parents=True)
    now = time.time()
    (run_dir / "run.json").write_text(json.dumps(
        {"id": rid, "channel": "nickmercs", "game": "gta6", "minutes": 30, "status": "done",
         "created": now, "started": now, "ended": now}))
    reel = run_dir / "clips" / "00003_reel.mp4"
    reel.write_bytes(b"reel-bytes")
    q = ClipQueue(run_dir / "queue.db")
    cid = q.add("nickmercs", "twitch", 0, 10, 5.0, {"hook": 'Nick: "DUDE DUDE DUDE"'})
    q.update(cid, reel_path=str(reel), status="crafted")
    q.db.close()
    return rid, cid, run_dir


def test_connect_review_and_post_a_reel(tmp_path, monkeypatch):
    calls = {}
    rid, cid, run_dir = make_run(tmp_path)

    async def run():
        g = web.AppRunner(fake_graph(calls))
        await g.setup()
        gs = web.TCPSite(g, "127.0.0.1", 0)
        await gs.start()
        gbase = f"http://127.0.0.1:{gs._server.sockets[0].getsockname()[1]}"
        monkeypatch.setattr(ig, "GRAPH", gbase)
        app = build_app(TT, tmp_path / "q.db", tmp_path / "tiktok_token.json",
                        password_hash=hash_password("pw", 1000), allow_posting=True,
                        control_dir=tmp_path / "control", ig_cfg=IG)
        r = web.AppRunner(app)
        await r.setup()
        s = web.TCPSite(r, "127.0.0.1", 0)
        await s.start()
        base = f"http://127.0.0.1:{s._server.sockets[0].getsockname()[1]}"
        clip = f"{base}/runs/{rid}/clip/{cid}/instagram"
        out = {}
        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as anon:
            out["anon_review"] = (await anon.get(clip, allow_redirects=False)).status
        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as c:
            await c.post(base + "/signin", data={"password": "pw"})
            login = await c.get(base + "/instagram/login", allow_redirects=False)
            q = parse_qs(urlparse(login.headers["Location"]).query)
            out["auth"] = login.headers["Location"].split("?")[0]
            out["scope"] = q["scope"][0]
            out["bad_state"] = await (await c.get(f"{base}/callback/instagram?code=abc&state=nope")).text()
            await c.get(f"{base}/callback/instagram?code=abc%23_&state={q['state'][0]}")
            out["account"] = await (await c.get(base + "/instagram")).text()
            out["clips"] = await (await c.get(base + "/clips")).text()
            out["review"] = await (await c.get(clip)).text()
            out["no_consent"] = await (await c.post(clip + "/post", data={"caption": "x"})).text()
            out["posting"] = await (await c.post(clip + "/post", data={
                "caption": "hi", "share_to_feed": "on", "consent": "on"})).text()
            video = urlparse(calls["media"]["video_url"]).path
            async with aiohttp.ClientSession() as meta:      # Meta's servers: not signed in
                out["meta_fetch"] = await (await meta.get(base + video)).read()
            out["s1"] = await (await c.get(base + "/instagram/status/c1")).json()
            out["s2"] = await (await c.get(base + "/instagram/status/c1")).json()
            out["s3"] = await (await c.get(base + "/instagram/status/c1")).json()
            async with aiohttp.ClientSession() as meta:
                out["after"] = (await meta.get(base + video)).status
            out["clips_after"] = await (await c.get(base + "/clips")).text()
        await r.cleanup()
        await g.cleanup()
        return out

    out = asyncio.run(run())
    assert out["anon_review"] == 302
    assert out["auth"] == "https://www.facebook.com/dialog/oauth"
    assert out["scope"] == ("instagram_basic,instagram_content_publish,pages_show_list,"
                            "pages_read_engagement,business_management")
    assert "Invalid login state" in out["bad_state"] and "exchange" in calls   # only the good state got through
    assert calls["code"]["code"] == "abc" and calls["code"]["client_secret"] == "secret"
    assert calls["code"]["redirect_uri"] == "https://clipbros.test/callback/instagram"
    assert calls["exchange"]["fb_exchange_token"] == "short" and calls["page_token_with"] == "long"
    assert "@clipbros.cb" in out["account"] and calls["me_token"] == "page-token"
    assert calls["media"]["access_token"] == "page-token"
    assert f"/runs/{rid}/clip/{cid}/instagram'>Post to Instagram" in out["clips"]
    assert "Nick: &quot;DUDE DUDE DUDE&quot;" in out["review"] and "name='consent'" in out["review"]
    assert "Not posted" in out["no_consent"] and "processing" in out["posting"]
    assert calls["media"]["media_type"] == "REELS" and calls["media"]["share_to_feed"] == "true"
    assert calls["media"]["video_url"].startswith("https://clipbros.test/ig/media/")
    assert out["meta_fetch"] == b"reel-bytes"
    assert out["s1"]["status"] == "IN_PROGRESS" and out["s2"]["permalink"].endswith("/reel/abc/")
    assert out["s3"]["status"] == "PUBLISHED" and len(calls["publish"]) == 1   # published once
    assert out["after"] == 404                                   # the public link is gone
    r = ClipQueue(run_dir / "queue.db").get(cid).receipt
    assert r["posted_to"]["instagram"] == "https://www.instagram.com/reel/abc/"
    assert "Post to Instagram" not in out["clips_after"] and "On Instagram" in out["clips_after"]


def test_without_setup_the_page_says_what_is_missing(tmp_path):
    async def run():
        app = build_app(TT, tmp_path / "q.db", tmp_path / "t.json", password_hash=hash_password("pw", 1000),
                        allow_posting=True, control_dir=tmp_path / "control")
        r = web.AppRunner(app)
        await r.setup()
        s = web.TCPSite(r, "127.0.0.1", 0)
        await s.start()
        base = f"http://127.0.0.1:{s._server.sockets[0].getsockname()[1]}"
        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as c:
            await c.post(base + "/signin", data={"password": "pw"})
            text = await (await c.get(base + "/instagram")).text()
            guessed = (await c.get(base + "/ig/media/" + "a" * 30 + ".mp4")).status
        await r.cleanup()
        return text, guessed

    text, guessed = asyncio.run(run())
    assert "INSTAGRAM_APP_ID" in text and guessed == 404


def test_posting_off_means_no_instagram_pages(tmp_path):
    app = build_app(TT, tmp_path / "q.db", tmp_path / "t.json", control_dir=tmp_path / "control", ig_cfg=IG)
    paths = {r.resource.canonical for r in app.router.routes() if r.resource}
    assert not any("instagram" in p or "/ig/" in p for p in paths)


def test_no_page_with_instagram_gives_a_clear_error(tmp_path, monkeypatch):
    async def run():
        g = web.AppRunner(fake_graph({}, pages=({"id": "p1"},)))
        await g.setup()
        gs = web.TCPSite(g, "127.0.0.1", 0)
        await gs.start()
        monkeypatch.setattr(ig, "GRAPH", f"http://127.0.0.1:{gs._server.sockets[0].getsockname()[1]}")
        async with aiohttp.ClientSession() as http:
            try:
                await ig.InstagramClient(IG, http).exchange_code("abc")
            except ig.InstagramError as ex:
                return str(ex)
            finally:
                await g.cleanup()

    assert "No Facebook Page linked to an Instagram" in asyncio.run(run())


def test_graph_errors_are_readable():
    try:
        ig._check({"error": {"message": "Invalid OAuth access token.", "code": 190}})
    except ig.InstagramError as ex:
        assert "Invalid OAuth access token." in str(ex) and "190" in str(ex)
