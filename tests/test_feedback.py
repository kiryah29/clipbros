import asyncio
import csv
import json
import io
from dataclasses import dataclass, field

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from multidict import MultiDict

from clipfarm.feedback import (FeedbackStore, backfill, clip_uid, features, parse_review,
                               selection_label)
from clipfarm.feedback_web import ReviewPages
from clipfarm.queue import ClipQueue

RECEIPT = {
    "why": "chat hit 9 msg/s", "triggers": ["chat", "audio"], "chat_ratio": 4.2, "audio_ratio": 3.1,
    "signals": {"audio": 0.6, "speech": 0.3, "chat_speed": 0.8, "chat_words": 0.4, "game_events": 0.0},
    "ratio": 5.0, "z_score": 6.1, "hype_share": 0.35, "peak_rate": 9.0, "baseline_rate": 1.8,
    "start_ts": 1000.0, "peak_ts": 1020.0, "moment": {"label": "cops/chase", "hits": 4},
    "speech": {"phrases": ["no way"], "pace_ratio": 1.7}, "transcript": [["no", 0, 1], ["way", 1, 2]],
    "chat": [[1001, "a", "LUL"]] * 5,
    "clip_agent": {"clip_type": "funny", "confidence_score": 0.81, "recommended": True},
    "hook": "DO NOT steal the cop car", "hook_kind": "warning", "caption_source": "clip",
    "facecam_source": "auto", "nudity": {"decision": "clean"}, "anchor": {"length_s": 24.0, "peak_s": 9.5},
}


@dataclass
class Clip:
    id: int
    channel: str = "speed"
    score: float = 72.5
    status: str = "crafted"
    receipt: dict = field(default_factory=lambda: dict(RECEIPT))
    clip_start_ts: float = 995.0
    clip_end_ts: float = 1030.0
    raw_path: str | None = None


def test_features_from_rich_and_empty_receipts():
    f = features(RECEIPT, 72.5, 995.0, 1030.0)
    s, r = f["selection"], f["render"]
    assert s["sig_chat_speed"] == 0.8 and s["clip_type"] == "funny" and s["confidence"] == 0.81
    assert s["moment"] == "cops/chase" and s["triggers"] == "audio+chat"
    assert s["length_s"] == 35.0 and s["peak_offset_s"] == 25.0
    assert s["n_phrases"] == 1 and s["chat_msgs"] == 5 and s["transcript_words"] == 2
    assert r["hook_kind"] == "warning" and r["nudity"] == "clean" and r["reel_length_s"] == 24.0
    assert "hook" not in s  # render facts never leak into selection features
    empty = features({}, None)
    assert empty["selection"]["clip_type"] is None and empty["selection"]["n_phrases"] == 0


def test_parse_review_form_and_json():
    form = MultiDict([("reason", "boring"), ("reason", "bad_hook"), ("reason", "boring"),
                      ("note", "  meh  "), ("reviewer", "timo"), ("watch_ms", "8123"), ("rank", "3"),
                      ("blind", "1")])
    r = parse_review("reject", form)
    assert r["reasons"] == ["boring", "bad_hook"] and r["note"] == "meh" and r["watch_ms"] == 8123
    assert r["rank_position"] == 3 and r["blind"] and not r["star"]
    a = parse_review("approve", {"star": True, "reasons": ["boring"]})
    assert a["star"] and a["reasons"] == [] and a["reviewer"] == "unknown"
    assert not parse_review("reject", {"star": "1"})["star"]  # a reject is never a top clip
    with pytest.raises(ValueError):
        parse_review("reject", {"reasons": ["nope"]})
    with pytest.raises(ValueError):
        parse_review("maybe", {})


@pytest.mark.parametrize("row, want", [
    ({"decision": "approve", "star": 0, "reasons": [], "watch_ms": 9000}, (1, 1.0)),
    ({"decision": "approve", "star": 1, "reasons": [], "watch_ms": None}, (1, 2.0)),
    ({"decision": "approve", "star": 0, "reasons": [], "watch_ms": 1500}, (1, 0.5)),
    ({"decision": "reject", "star": 0, "reasons": ["boring", "bad_hook"], "watch_ms": 9000}, (0, 1.0)),
    ({"decision": "reject", "star": 0, "reasons": [], "watch_ms": 9000}, (0, 0.5)),
    ({"decision": "reject", "star": 0, "reasons": ["bad_captions"], "watch_ms": 9000}, None),
    ({"decision": "reject", "star": 0, "reasons": '["duplicate"]', "watch_ms": None}, None),
])
def test_selection_label(row, want):
    assert selection_label(row) == want


def test_store_newest_review_wins_and_export(tmp_path):
    fb = FeedbackStore(tmp_path / "feedback.db")
    c = Clip(7)
    uid = clip_uid("speed", "20261126-001500-abcd", 7)
    fb.record(uid, parse_review("reject", {"reasons": ["boring"], "reviewer": "timo"}), c, run_id="r", now=1)
    fb.record(uid, parse_review("approve", {"reviewer": "timo"}), c, run_id="r", now=2)
    fb.record(uid, parse_review("reject", {"reasons": ["bad_hook"], "reviewer": "hayrik"}), c, run_id="r", now=3)
    assert fb.latest(uid)["reviewer"] == "hayrik"
    cur = {r["reviewer"]: r["decision"] for r in fb.current()}
    assert cur == {"timo": "approve", "hayrik": "reject"}
    stored = fb.latest(uid)
    assert stored["features"]["selection"]["clip_type"] == "funny" and stored["confidence"] == 0.81
    rows = list(csv.DictReader(io.StringIO(fb.export_csv())))
    assert len(rows) == 2 and {r["label"] for r in rows} == {"1", ""}
    assert rows[0]["f_sig_chat_speed"] == "0.8" and rows[0]["f_clip_type"] == "funny"
    s = fb.stats()
    assert s["reviews"] == 2 and s["approved"] == 1 and s["reasons"]["bad_hook"] == 1
    assert s["selection_labels"] == 1


def test_backfill_copies_decided_clips_once(tmp_path):
    q = ClipQueue(tmp_path / "queue.db")
    a = q.add("speed", "twitch", 0, 30, 60.0, RECEIPT, status="approved")
    old = q.add("speed", "twitch", 40, 70, 40.0, RECEIPT, status="rejectd")  # old action + "d" bug
    q.add("speed", "twitch", 60, 90, 45.0, RECEIPT, status="rejected")
    q.add("speed", "twitch", 80, 110, 50.0, RECEIPT, status="crafted")
    q.db.close()
    fb = FeedbackStore(tmp_path / "feedback.db")
    assert backfill(fb, [(None, tmp_path / "queue.db"), ("gone", tmp_path / "missing.db")]) == 3
    assert backfill(fb, [(None, tmp_path / "queue.db")]) == 0
    assert fb.latest(clip_uid("speed", None, a))["reviewer"] == "backfill"
    assert fb.latest(clip_uid("speed", None, old))["decision"] == "reject"
    q = ClipQueue(tmp_path / "queue.db")
    assert q.get(old).status == "rejected"
    q.db.close()


# --------------------------------------------------------------------------- web
class FakeRuns:
    def __init__(self, tmp_path):
        self.rid = "20261126-001500-abcd"
        self.runs = {self.rid: {"id": self.rid, "channel": "speed", "game": "gta6", "status": "finished",
                                "created": 1.0, "started": 1.0}}
        self.feedback = FeedbackStore(tmp_path / "feedback.db")
        self.q = ClipQueue(tmp_path / "queue.db")
        self.low = self.q.add("speed", "twitch", 0, 30, 30.0, RECEIPT, status="crafted")
        self.high = self.q.add("speed", "twitch", 40, 70, 90.0, RECEIPT, status="crafted")
        for cid in (self.low, self.high):
            self.q.update(cid, reel_path=f"/x/{cid}_reel.mp4", raw_path=f"/x/{cid:05d}_raw.mp4")

    def clips(self, rid):
        return self.q.ranked(None, 200)

    def clip(self, rid, cid):
        return self.q.get(cid)

    def set_status(self, rid, cid, status):
        self.q.update(cid, status=status)


class Pages(ReviewPages):
    def __init__(self, runs):
        self.runs, self.posting = runs, False

    def page(self, title, body, status=200):
        return web.Response(text=body, content_type="text/html", status=status)

    def _rid(self, request):
        rid = request.match_info["rid"]
        if rid not in self.runs.runs:
            raise web.HTTPNotFound()
        return rid

    def routes(self):
        clip = "/runs/{rid}/clip/{cid:\\d+}"
        return [web.get("/clips", self.clips), web.post(clip + "/{action:approve|reject}", self.mark),
                *self.review_routes()]


def run_web(tmp_path, scenario):
    runs = FakeRuns(tmp_path)
    app = web.Application()
    app.add_routes(Pages(runs).routes())

    async def go():
        async with TestClient(TestServer(app)) as client:
            await scenario(client, runs)
    asyncio.run(go())


def test_clips_page_blind_by_default(tmp_path):
    async def scenario(client, runs):
        blind = await (await client.get("/clips")).text()
        assert "score 90.0" not in blind and "Reject&hellip;" in blind and "Starts too late" in blind
        shown = await (await client.get("/clips?blind=0")).text()
        assert "score 90.0" in shown
    run_web(tmp_path, scenario)


def test_form_review_saves_and_sets_status(tmp_path):
    async def scenario(client, runs):
        url = f"/runs/{runs.rid}/clip/{runs.high}/reject"
        r = await client.post(url, data=[("reason", "late_start"), ("reason", "bad_hook"), ("note", "slow"),
                                         ("reviewer", "timo"), ("watch_ms", "12000"), ("rank", "1"),
                                         ("blind", "1")], allow_redirects=False)
        assert r.status == 302 and r.headers["Location"].endswith(f"#clip-{runs.high}")
        row = runs.feedback.latest(clip_uid("speed", runs.rid, runs.high))
        assert row["reasons"] == ["late_start", "bad_hook"] and row["game"] == "gta6" and row["blind"] == 1
        assert row["receipt_path"].endswith(f"{runs.high:05d}_receipt.json")
        assert runs.q.get(runs.high).status == "rejected"
        page = await (await client.get("/clips")).text()
        assert "score 90.0" in page and "Last review: <b>reject</b> by timo" in page  # revealed after deciding
        bad = await client.post(url, data={"reason": "nope"})
        assert bad.status == 400
    run_web(tmp_path, scenario)


def test_json_api_and_export(tmp_path):
    async def scenario(client, runs):
        url = f"/api/runs/{runs.rid}/clip/{runs.low}/review"
        r = await client.post(url, json={"decision": "approve", "star": True, "reviewer": "hayrik"})
        assert r.status == 200 and (await r.json())["ok"]
        assert runs.q.get(runs.low).status == "approved"
        bad = await client.post(url, json={"decision": "reject", "reasons": ["what"]})
        assert bad.status == 400 and not (await bad.json())["ok"]
        assert (await client.post(url, data="not json")).status == 400
        assert (await client.post("/api/runs/nope/clip/1/review", json={})).status == 404
        text = await (await client.get("/api/feedback/export.csv?days=7")).text()
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 1 and rows[0]["star"] == "1" and rows[0]["label_weight"] == "2.0"
    run_web(tmp_path, scenario)


def test_posted_clip_stays_posted(tmp_path):
    async def scenario(client, runs):
        runs.q.update(runs.high, status="posted")
        await client.post(f"/runs/{runs.rid}/clip/{runs.high}/approve", data={"reviewer": "timo"},
                          allow_redirects=False)
        assert runs.q.get(runs.high).status == "posted"
        assert runs.feedback.latest(clip_uid("speed", runs.rid, runs.high))["decision"] == "approve"
    run_web(tmp_path, scenario)


# --------------------------------------------------------------------------- the real app
def test_real_app_review_flow(tmp_path):
    """build_app + sign-in + a finished control run: review on /clips lands in data/feedback.db."""
    import aiohttp as aio

    from clipfarm.control import hash_password
    from clipfarm.tiktok import api
    from clipfarm.tiktok.webapp import build_app

    rid = "20261126-001500-abcd"
    run_dir = tmp_path / "control" / rid
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"id": rid, "channel": "speed", "game": "gta6", "minutes": 5,
                                                  "status": "finished", "created": 1e10, "started": 1e10,
                                                  "ended": 1e10}))
    q = ClipQueue(run_dir / "queue.db")
    cid = q.add("speed", "twitch", 0, 30, 88.0, RECEIPT, status="crafted")
    reel = run_dir / "00001_reel.mp4"
    reel.write_bytes(b"0")
    q.update(cid, reel_path=str(reel), raw_path=str(run_dir / "00001_raw.mp4"))
    q.db.close()

    async def go():
        app = build_app(api.Config("ck", "cs", "https://example.test/callback"), tmp_path / "q.db",
                        tmp_path / "tok.json", password_hash=hash_password("pw", 1000),
                        control_dir=tmp_path / "control")
        async with TestClient(TestServer(app), cookie_jar=aio.CookieJar(unsafe=True)) as c:
            assert (await c.post(f"/api/runs/{rid}/clip/{cid}/review", json={"decision": "approve"})).status == 401
            await c.post("/signin", data={"password": "pw"})
            page = await (await c.get(f"/clips?run={rid}")).text()
            assert "Reject&hellip;" in page and "score 88.0" not in page
            r = await c.post(f"/runs/{rid}/clip/{cid}/reject", allow_redirects=False,
                             data=[("reason", "boring"), ("reviewer", "timo"), ("watch_ms", "9000")])
            assert r.status == 302
            csv_text = await (await c.get("/api/feedback/export.csv")).text()
            return csv_text

    csv_text = asyncio.run(go())
    assert "speed/20261126-001500-abcd/1" in csv_text
    fb = FeedbackStore(tmp_path / "feedback.db")      # next to data/control/, outside the run folder
    row = fb.latest(clip_uid("speed", rid, cid))
    assert row["decision"] == "reject" and row["reasons"] == ["boring"] and row["game"] == "gta6"
    q = ClipQueue(run_dir / "queue.db")
    assert q.get(cid).status == "rejected"
