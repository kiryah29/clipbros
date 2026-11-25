"""Rule-based hooks: facts -> patterns -> scored candidates, and the hook lab."""
import asyncio
import json

import aiohttp
from aiohttp.test_utils import TestServer

from clipfarm.config import channel_settings, load_settings
from clipfarm.control import hash_password
from clipfarm.hooklab import Picks, ReadOnlyRuns, coverage, lab_items, reel_view, review
from clipfarm.hooks import HookHistory
from clipfarm.hookwriter import (best_chat_message, candidates, collect_facts, pick_hook,
                                 plan_line)
from clipfarm.queue import ClipQueue
from clipfarm.tiktok.api import Config
from clipfarm.tiktok.webapp import build_app

S = load_settings()
GTA = channel_settings(S, {"name": "ishowspeed", "platform": "twitch", "streamer": "Speed",
                           "game": "gta6"})
CHATTING = channel_settings(S, {"name": "lydiaviolet", "platform": "twitch", "streamer": "Lydia",
                                "pronoun": "she"})


def said(text, start=1.0, step=0.3):
    out, t = [], start
    for w in text.split():
        out.append([w, round(t, 2), round(t + step - 0.05, 2)])
        t += step
    return out


def receipt(**kw):
    return {"triggers": ["chat"], "chat_ratio": 4.0, "audio_ratio": 1.0, "speech": {},
            "top_tokens": [], **kw}


def test_plan_line_rewrites_the_plan_for_a_stranger():
    p = plan_line(said("Okay. I'm gonna rob my neighbor's house."), 10.0, ["i'm gonna"])
    assert p["action"] == "rob his neighbor's house"
    assert p["action_you"] == "rob your neighbor's house"
    she = plan_line(said("I'm gonna steal my mom's car."), 10.0, ["i'm gonna"], pronoun="she")
    assert she["action"] == "steal her mom's car"
    # long plans are cut before a preposition, "go" and "try to" are dropped
    long = plan_line(said("I'm gonna go steal the tank from the military base over there"), 10.0,
                     ["i'm gonna"])
    assert long["action"] == "steal the tank"
    assert plan_line(said("I'm gonna try to clutch the round."), 10.0, ["i'm gonna"])["action"] == \
        "clutch the round"


def test_talk_is_not_a_plan():
    for line in ["Let's go!", "I'm gonna be honest with you.", "I'm gonna run this side.",
                 "Let's get it.", "I'm gonna show you something.", "I'm gonna go to the bank",
                 "I'm gonna die."]:
        assert plan_line(said(line), 10.0, ["i'm gonna", "let's"]) is None, line
    # said long after the peak: not the setup of this moment
    assert plan_line(said("I'm gonna rob the bank.", start=30), 10.0, ["i'm gonna"]) is None


def test_story_and_warning_from_the_plan():
    words = said("Watch this. I'm gonna steal the cop car.")
    laugh = [["KEKW", 40], ["LUL", 30]]
    cands, facts = candidates(receipt(top_tokens=laugh), GTA, words=words, peak=6.0, seed=1)
    texts = {c.style: c.text for c in cands}
    assert texts["story"] == "Speed tried to steal the cop car"
    assert texts["warning"] == "DO NOT steal the cop car in GTA 6"      # chat laughed: it went wrong
    assert texts["prediction"] in ('Speed said "Watch this"', "Speed really thought this would work")
    assert cands[0].style in ("warning", "story")                     # GTA weights these highest
    # nothing went wrong: no warning, no prediction
    calm, _ = candidates(receipt(), GTA, words=words, peak=6.0, seed=1)
    assert {c.style for c in calm} >= {"story"} and not {"warning", "prediction"} & {c.style for c in calm}
    # Just Chatting has no game name: no warning
    jc, _ = candidates(receipt(top_tokens=laugh), CHATTING, words=words, peak=6.0, seed=1)
    assert "warning" not in {c.style for c in jc}


def test_chat_voice_needs_other_chatters_and_a_reaction():
    def msg(t, user, text):
        return [t, user, text, None]
    chat = [msg(5, "a", "he really thought that would work"),
            msg(6, "b", "he really thought that"), msg(6.5, "c", "bro really thought it would work"),
            msg(5, "d", "switch to the other car bro"), msg(5.1, "e", "switch to the other car"),
            msg(5.2, "f", "you need a winch"), msg(5.3, "g", "you need a winch man"),
            msg(5.4, "h", "kai meat leaked lol"), msg(5.5, "i", "kai meat leaked"),
            msg(5.6, "j", "CALL NOW FOR 100K+ JOB"), msg(5.7, "k", "CALL NOW FOR 100K+ JOB")]
    best = best_chat_message(chat, [], 4.0, "Speed", set(), set(), 34)
    assert best["message"] == "he really thought that would work" and best["support"] == 2
    # one person saying it is not a consensus
    assert best_chat_message(chat[:1], [], 4.0, "Speed", set(), set(), 34) is None
    # orders, talking to the streamer, the stricter chat blocklist and spam never qualify
    assert best_chat_message(chat[3:], [], 4.0, "Speed", set(), set(), 34) is None
    # stretched spelling is cleaned up
    soldd = [msg(5, "a", "omg he soldd"), msg(5, "b", "he sold it"), msg(5, "c", "nah he sold")]
    assert best_chat_message(soldd, [], 4.0, "Kai", set(), set(), 34)["message"] == "omg he sold"


def test_sound_and_hud_facts_only_when_confident():
    r = receipt(sound_events=[{"label": "Siren", "t": 8.0, "score": 0.92}],
                hud={"events": [{"name": "wasted", "t": 9.0, "confidence": 0.9, "outcome": "fail"}]})
    words = said("I'm gonna outrun the cops.")
    cands, facts = candidates(r, GTA, words=words, peak=8.5, seed=2)
    styles = {c.style: c.text for c in cands}
    assert facts["thing"] == "cops" and facts["failed"] == "1"
    assert styles["payoff"] == "Wait for the cops"
    assert styles["story_outcome"] == "Speed tried to outrun the cops. It failed"
    assert styles["game"] == "Speed vs the cops in GTA 6"      # profile pattern, sirens heard
    unsure = receipt(sound_events=[{"label": "Siren", "t": 8.0, "score": 0.4}],
                     hud={"events": [{"name": "wasted", "t": 9.0, "confidence": 0.5, "outcome": "fail"}]})
    f = collect_facts(unsure, GTA, words, peak=8.5)
    assert "thing" not in f and "failed" not in f          # no fact beats a wrong one
    # payoff only when the peak is a few seconds in
    early = collect_facts(r, GTA, words, peak=1.0)
    assert "late_peak" not in early


def test_pick_hook_writers_and_fallback():
    words = said("I'm gonna steal the cop car.")
    rules = pick_hook(receipt(), GTA, words=words, peak=6.0, seed=1, writer="rules")
    assert rules["hook_writer"] == "rules" and rules["hook"] == "Speed tried to steal the cop car"
    assert rules["hook_facts"]["plan_said"].startswith("I'm gonna steal")
    assert rules["hook_candidates"][0][0] == rules["hook"]
    # no usable facts: templates, with phrase/quote demoted and chat speed gone
    r = receipt(top_tokens=[["KEKW", 40]], speech={"phrases": ["no way"]})
    fb = pick_hook(r, GTA, words=said("uh what"), peak=2.0, seed=1, writer="rules")
    assert fb["hook_writer"] == "templates" and fb["hook_kind"] == "laugh"
    # the default writer is still today's templates: the phrase wins there
    assert pick_hook(r, GTA, words=said("uh what"), peak=2.0, seed=1)["hook_kind"] == "phrase"


def test_history_avoids_the_same_style_twice():
    hist = HookHistory()
    laugh = receipt(top_tokens=[["KEKW", 40]])
    words = said("Watch this. I'm gonna steal the cop car.")
    first = pick_hook(laugh, GTA, hist, words=words, peak=6.0, seed=1, writer="rules")
    second = pick_hook(laugh, GTA, hist, words=words, peak=6.0, seed=1, writer="rules")
    assert first["hook"] != second["hook"] and first["hook_kind"] != second["hook_kind"]


def test_first_spoken_line_is_not_repeated():
    words = said("I'm gonna steal the cop car.", start=0.2)   # the viewer hears it right away
    cands, _ = candidates(receipt(), GTA, words=words, peak=6.0, seed=1)
    story = next(c for c in cands if c.style == "story")
    late, _ = candidates(receipt(), GTA, words=said("Okay chat. Yeah. I'm gonna steal the cop car.",
                                                     start=4.0), peak=9.0, seed=1)
    assert story.score < next(c for c in late if c.style == "story").score


def _run(tmp_path, receipts, channel="ishowspeed", game="gta6"):
    rid = "20261125-120000-abcd"
    d = tmp_path / "control" / rid
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"id": rid, "channel": channel, "game": game,
                                            "minutes": 5, "status": "stopped", "created": 1.0,
                                            "started": 1.0}))
    q = ClipQueue(d / "queue.db")
    for r in receipts:
        cid = q.add(channel, "twitch", 0, 30, 50, r)
        reel = d / f"{cid:05d}_reel.mp4"
        reel.write_bytes(b"\0")
        q.update(cid, reel_path=str(reel), status="crafted")
    q.db.close()
    return rid


def test_reel_view_uses_the_clip_transcript_when_there_is_one():
    live = receipt(transcript=said("hi"), chat=[[3.0, "a", "x", None]], peak_ts=110, start_ts=100)
    assert reel_view(live)[2:] == (10, "live")
    crafted = {**live, "caption_transcript": said("hi there"), "anchor": {"shift_s": 2.0, "peak_s": 6.5}}
    words, chat, peak, src = reel_view(crafted)
    assert src == "clip" and peak == 6.5 and chat[0][0] == 1.0


def test_hook_lab_cli_and_page(tmp_path):
    story = receipt(hook="Speed: \"OH MY GOD\"", hook_kind="phrase", peak_ts=106, start_ts=100,
                    transcript=said("I'm gonna steal the cop car."))
    rid = _run(tmp_path, [story, receipt(hook="WAIT FOR IT...", hook_kind="fallback")])
    items = lab_items(ReadOnlyRuns(tmp_path / "control"))
    assert [rv["rules_writer"] for _, _, rv in items] == ["rules", "templates"]
    assert coverage(items)["rules"] == 1
    assert review(story, GTA)["burned"] == 'Speed: "OH MY GOD"'

    async def go():
        app = build_app(Config("k", "s", "https://x.test/callback"), tmp_path / "q.db",
                        tmp_path / "tok.json", password_hash=hash_password("pw", 1000),
                        control_dir=tmp_path / "control", keep_days=100000)
        async with TestServer(app) as srv, \
                aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as s:
            base = str(srv.make_url(""))
            out = {"signed_out": (await s.get(base + "/hooklab", allow_redirects=False)).status}
            await s.post(base + "/signin", data={"password": "pw"})
            page = await (await s.get(base + "/hooklab")).text()
            r = await s.post(base + "/hooklab/pick", data={"rid": rid, "cid": "1", "pick": "c0",
                                                            "bad": "c1", "view": rid},
                             allow_redirects=False)
            out["pick"] = (r.status, r.headers["Location"])
            await s.post(base + "/hooklab/pick", data={"rid": rid, "cid": "2", "pick": "none"})
            out["bad_clip"] = (await s.post(base + "/hooklab/pick",
                                            data={"rid": rid, "cid": "99", "pick": "old"})).status
            out["after"] = await (await s.get(base + "/hooklab")).text()
            return page, out
    page, out = asyncio.run(go())
    assert out["signed_out"] == 302
    assert "Speed tried to steal the cop car" in page and "Rule writer found a hook on 1 of 2" in page
    assert out["pick"] == (302, f"/hooklab?run={rid}#c-{rid}-1")
    assert out["bad_clip"] == 404
    picks = Picks(tmp_path / "hooklab" / "picks.json")
    assert picks.items[f"{rid}/1"]["choice"] == "new"
    assert picks.items[f"{rid}/1"]["style"] in ("story", "warning")
    assert picks.summary()["none"] == 1 and "2 reviewed" in out["after"]


def test_craft_writes_the_rule_hook_onto_the_reel(tmp_path, monkeypatch):
    from clipfarm import engine
    from clipfarm.craft.captions import Word
    seen = {}
    monkeypatch.setattr(engine, "render_reel", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(engine, "pick_facecam", lambda *a, **k: (None, "none"))
    monkeypatch.setattr(engine, "decode_pcm", lambda p: b"")
    monkeypatch.setattr(engine, "clip_words", lambda raw, cfg, lead: [
        Word(t, s, e) for t, s, e in said("Okay chat. I'm gonna steal the cop car.", start=2.0)])
    cfg = {**GTA, "reel": {**GTA["reel"], "hooks": {**GTA["reel"]["hooks"], "writer": "rules"}}}
    q = ClipQueue(tmp_path / "q.db")
    cid = q.add("ishowspeed", "twitch", 0, 30, 50,
                receipt(hook='Speed: "OH MY GOD"', hook_kind="phrase", peak_ts=112, start_ts=100,
                        top_tokens=[["LUL", 40]]),
                raw_path=str(tmp_path / "r.mp4"))
    clean = lambda p, c, **k: {"decision": "clean", "ranges": [], "why": "t"}   # noqa: E731
    engine.craft_clip(q, cid, cfg, None, nudity_check=clean)
    r = q.get(cid).receipt
    assert r["hook_writer"] == "rules" and r["hook_kind"] in ("warning", "story")
    assert seen["hook_text"] == r["hook"] and "steal the cop car" in r["hook"]
    assert r["hook_live"] == 'Speed: "OH MY GOD"' and r["hook_facts"]["action"] == "steal the cop car"
    assert len(r["hook_candidates"]) >= 2
