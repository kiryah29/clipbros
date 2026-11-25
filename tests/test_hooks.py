"""Hook text depends on what happens in the clip, and consecutive clips don't repeat."""
from clipfarm.config import channel_settings, load_settings
from clipfarm.hooks import HookHistory, quote_near, reaction_of, write_hook

S = load_settings()
CFG = channel_settings(S, {"name": "jasontheween", "platform": "twitch", "streamer": "Jason"})


def chat(tokens, ratio=4.0, **kw):
    return {"triggers": ["chat"], "chat_ratio": ratio, "ratio": ratio, "audio_ratio": 1.0,
            "top_tokens": tokens, "speech": {}, **kw}


def test_reaction_from_chat_tokens():
    assert reaction_of([("KEKW", 40), ("the", 10), ("LMAOOOO", 12)])[0] == "laugh"
    assert reaction_of([("W", 50), ("lets", 5)]) == ("win", "W")
    assert reaction_of([("L", 30), ("bro", 20)])[0] == "loss"
    assert reaction_of([("???", 25), ("what", 10)])[0] == "confused"
    assert reaction_of([("the", 50), ("is", 40), ("KEKW", 2)]) is None   # too small a share
    assert reaction_of([("💀💀💀", 20), ("is", 5)])[0] == "dead"
    assert reaction_of([]) is None


def test_different_moments_get_different_hooks():
    laugh = write_hook(chat([("KEKW", 40), ("LUL", 20)]), CFG, seed=1)
    win = write_hook(chat([("W", 60)]), CFG, seed=1)
    rip = write_hook(chat([("RIP", 30), ("F", 30)]), CFG, seed=1)
    said = write_hook(chat([], speech={"phrases": ["no way"]}), CFG, seed=1)
    assert [k for _, k in (laugh, win, rip, said)] == ["laugh", "win", "rip", "phrase"]
    assert len({t for t, _ in (laugh, win, rip, said)}) == 4
    assert "NO WAY" in said[0]
    filler = write_hook(chat([], speech={"phrases": ["the x3", "help"]}), CFG, seed=1)
    assert "THE THE" not in filler[0] and "HELP" in filler[0]


def test_quote_from_transcript_before_peak():
    words = [["okay", 1.0, 1.3], ["watch", 9.0, 9.3], ["me", 9.3, 9.5], ["steal", 9.5, 9.9],
             ["this", 9.9, 10.1], ["car", 10.1, 10.5], ["later", 30.0, 30.4]]
    assert quote_near(words, 12.0) == "watch me steal this car"
    assert quote_near([["uh", 1, 1.1], ["um", 1.1, 1.2], ["so", 1.2, 1.3]], 2.0) == ""
    text, kind = write_hook(chat([], transcript=words), CFG, peak_rel=12.0, seed=3)
    assert kind == "quote" and "watch me steal this car" in text


def test_consecutive_clips_do_not_repeat():
    hist = HookHistory()
    r = chat([("hello", 5)], ratio=4.0)
    # chat-speed lines are off by default (a stranger can't read "chat went 4x faster")
    assert write_hook(r, CFG, seed=7) == ("WAIT FOR IT...", "fallback")
    cfg = {**CFG, "reel": {**CFG["reel"], "hooks": {**CFG["reel"]["hooks"], "drop_kinds": []}}}
    hooks = [write_hook(r, cfg, hist, seed=7)[0] for _ in range(4)]
    assert len(set(hooks)) == 4                       # the chat pool has 4 lines
    assert "Chat went 4x faster for this" in hooks
    # once the pool is exhausted it still returns a usable line, never crashes
    assert write_hook(r, cfg, hist, seed=7)[0] in hooks


def test_same_clip_same_hook():
    r = chat([("KEKW", 40)])
    assert write_hook(r, CFG, seed=42) == write_hook(r, CFG, seed=42)


def test_unsafe_words_never_reach_the_card():
    r = chat([], speech={"phrases": ["shit"]},
             transcript=[["what", 1, 1.2], ["the", 1.2, 1.3], ["fuck", 1.3, 1.6], ["bro", 1.6, 2]])
    text, kind = write_hook(r, CFG, peak_rel=2.0, seed=1)
    assert kind == "fallback" and "SHIT" not in text.upper() and "FUCK" not in text.upper()
    custom = {**CFG, "reel": {**CFG["reel"], "hooks": {"blocklist": ["kekw"]}}}
    assert "KEKW" not in write_hook(chat([("KEKW", 50)]), custom, seed=0)[0]


def test_long_quotes_are_skipped_and_fallback_is_safe():
    r = chat([], speech={"phrases": ["absolutely unbelievable incredible ridiculous"]})
    text, _ = write_hook(r, CFG, seed=1)
    assert len(text) <= 42
    bare = {"triggers": [], "chat_ratio": 1.0, "audio_ratio": 1.0}
    assert write_hook(bare, CFG) == ("WAIT FOR IT...", "fallback")


def test_game_moment_templates_come_first():
    gta = channel_settings(S, {"name": "jasontheween", "platform": "twitch",
                               "streamer": "Jason", "game": "gta6"})
    m = gta["moments"][0]
    r = chat([("KEKW", 40)], moment={"name": m["name"], "label": m.get("label", "")})
    text, kind = write_hook(r, gta, seed=1)
    assert kind == "moment" and "{" not in text
