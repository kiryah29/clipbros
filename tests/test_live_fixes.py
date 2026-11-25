"""xQc/CaseOh live test fixes: late chat, caption timing, game dialogue, hooks."""
from clipfarm.chat.twitch import TwitchChat
from clipfarm.config import channel_settings, load_settings
from clipfarm.craft.captions import Word, build_ass, drop_dialogue, group_words, looks_like_subtitle
from clipfarm.detector import LagFilter
from clipfarm.hooks import write_hook
from clipfarm.models import ChatMessage

S = load_settings()


def test_twitch_parses_tags_and_send_time():
    line = ("@badge-info=;color=#FF0000;display-name=Bob;tmi-sent-ts=1700000000123 "
            ":bob!bob@bob.tmi.twitch.tv PRIVMSG #xqc :LMAO no way")
    m = TwitchChat("xqc")._parse(line)
    assert (m.user, m.text, m.sent_ts) == ("bob", "LMAO no way", 1700000000.123)
    plain = TwitchChat("xqc")._parse(":bob!bob@bob.tmi.twitch.tv PRIVMSG #xqc :hi")
    assert plain.user == "bob" and plain.sent_ts is None


def test_backlog_after_stall_is_late():
    f = LagFilter(max_lag_s=5)
    for i in range(50):                       # normal: ~0.4 s behind (clock offset included)
        assert not f.is_late(ChatMessage("c", 0, "u", "x", sent_ts=i), i + 0.4)
    # stall: messages sent at 60-90 s all arrive at 95 s
    late = [f.is_late(ChatMessage("c", 0, "u", "x", sent_ts=t), 95) for t in range(60, 91)]
    assert sum(late) == 30 and not late[-1]   # only the one sent within 5 s of usual counts
    assert not f.is_late(ChatMessage("c", 0, "u", "x"), 95)   # no send time: keep


def test_caption_line_breaks_at_pause_and_word_time_is_capped(tmp_path):
    words = [Word("what", 8.7, 12.3), Word("for", 12.3, 12.5), Word("never", 24.9, 25.6)]
    assert [len(g) for g in group_words(words, 4, 1.0)] == [2, 1]
    reel = {**S["reel"], "width": 1080, "height": 1920}
    out = build_ass(words, 30, reel, 1000, tmp_path / "c.ass").read_text(encoding="utf-8")
    ends = [l.split(",")[2] for l in out.splitlines() if l.startswith("Dialogue: 1")]
    assert ends[0] == "0:00:10.20"            # 8.7 + 1.5 s max, not held to 24.9


def test_subtitle_frame_detection():
    w, h = 480, 80
    dark = bytes(w * h)
    text = bytearray(dark)                    # thin bright strokes, like subtitle letters
    for r in range(40, 55):
        for c in range(100, 380, 4):
            text[r * w + c] = 230
    sky = bytearray(dark)                     # a big bright blob (headlights, sky)
    for r in range(10, 70):
        for c in range(50, 300):
            sky[r * w + c] = 220
    assert looks_like_subtitle(bytes(text), w)
    assert not looks_like_subtitle(dark, w) and not looks_like_subtitle(bytes(sky), w)
    words = [Word("okay", 1.0, 1.4), Word("help", 6.0, 6.3)]
    assert [x.text for x in drop_dialogue(words, [1.5, 2.0])] == ["help"]


def _cfg(**audio):
    ch = {"name": "caseoh_", "platform": "twitch", "streamer": "CaseOh", "audio": audio}
    return channel_settings(S, ch)


def test_hook_kind_follows_trigger():
    cfg = _cfg(min_ratio=3.5)
    hook = lambda r: write_hook(r, cfg, seed=1)
    big = {"triggers": ["chat"], "audio_ratio": 3.6, "chat_ratio": 6.0, "speech": {"phrases": []}}
    assert hook(big)[1] == "big"
    # clip 3 from the test: chat 3.6x, audio 2.4x, said "help"
    c3 = {"triggers": ["chat"], "audio_ratio": 2.4, "chat_ratio": 3.6,
          "speech": {"phrases": ["help"]}}
    text, kind = hook(c3)
    assert kind == "phrase" and '"HELP"' in text
    assert '"NO NO NO"' in hook({**c3, "speech": {"phrases": ["no x3"]}})[0]
    loud = {"triggers": ["audio"], "audio_ratio": 7, "chat_ratio": 1.6, "speech": {}}
    assert hook(loud)[1] == "audio"
    chat = {"triggers": ["chat"], "audio_ratio": 1.2, "chat_ratio": 4, "speech": {}}
    assert hook(chat)[1] == "fallback"        # chat-speed lines are off by default
    cfg["reel"]["hooks"] = {**cfg["reel"].get("hooks", {}), "drop_kinds": []}
    assert hook(chat)[1] == "chat"
