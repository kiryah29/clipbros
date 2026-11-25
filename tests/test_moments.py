from clipfarm.config import channel_settings, load_settings
from clipfarm.craft.overlay import draw_base, draw_hook, strip_emoji
from clipfarm.engine import build_receipt, clip_window
from clipfarm.models import Spike
from clipfarm.moments import classify

CFG = channel_settings(load_settings(), {"name": "jasontheween", "game": "gta6",
                                         "streamer": "Jason"})


def _spike(moment=None):
    return Spike("c", start_ts=1000, peak_ts=1005, end_ts=1010, peak_rate=9, baseline_rate=2,
                 baseline_std=1, z_score=5, ratio=4.5, moment=moment)


def test_game_profile_merges_under_channel_overrides():
    assert CFG["reel"]["caption_style"] == "clipbros"
    assert CFG["reel"]["cta_text"] == ""
    assert CFG["clip"]["post_roll_s"] == 16 and CFG["clip"]["max_length_s"] == 30
    assert CFG["streamer"] == "Jason"
    assert {m["name"] for m in CFG["moments"]} >= {"police_escalation", "graphic_shock"}
    over = channel_settings(load_settings(), {"name": "x", "game": "gta6",
                                              "clip": {"post_roll_s": 9}})
    assert over["clip"]["post_roll_s"] == 9


def test_no_game_profile_keeps_old_behaviour():
    cfg = channel_settings(load_settings(), {"name": "x"})
    assert cfg["moments"] == []


def test_classify_picks_police_and_stretched_words():
    msgs = ["COPSSS 💀", "5 star lmao", "the police 😭", "bro", "RUN", "lol", "gg", "chase him"]
    m = classify(msgs, CFG["moments"])
    assert m["name"] == "police_escalation"
    assert m["hits"] >= 4


def test_classify_none_on_unrelated_chat():
    assert classify(["gg", "hi chat", "o7", "true"] * 5, CFG["moments"]) is None


def test_escalation_window_and_hook():
    m = classify(["cops!!", "police", "wanted lol", "RUN"], CFG["moments"])
    sp = _spike(m)
    start, end = clip_window(sp, CFG["clip"], CFG["moments"])
    assert sp.peak_ts - start == 12 and end - start <= CFG["clip"]["raw_max_s"]
    r = build_receipt(sp, CFG)
    assert r["hook"] and "{" not in r["hook"] and r["safety_review"] is False


def test_graphic_moment_is_held_for_review():
    m = classify(["his HEAD", "blood everywhere", "gore", "headless 💀"], CFG["moments"])
    assert m["name"] == "graphic_shock"
    assert build_receipt(_spike(m), CFG)["safety_review"] is True


def test_min_length_is_enforced():
    start, end = clip_window(_spike(), {**CFG["clip"], "pre_roll_s": 2}, [])
    assert end - start >= CFG["clip"]["min_length_s"]


def test_overlays_render(tmp_path):
    reel = CFG["reel"]
    assert strip_emoji("DO NOT STEAL A TESLA 😭") == "DO NOT STEAL A TESLA"
    b = draw_base(reel, 576, tmp_path / "b.png")
    h = draw_hook(reel, "He lost 11 hours in one second", 576, tmp_path / "h.png")
    from PIL import Image
    hb = Image.open(h).getbbox()
    assert hb[1] < 576 < hb[3]            # card straddles the seam
    assert hb[2] <= reel["width"] - reel["safe_zone"]["right"] + 8   # clear of TikTok buttons
    assert Image.open(b).getpixel((reel["width"] - 5, 576))[3] == 255  # lime line full width
