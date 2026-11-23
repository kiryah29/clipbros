from clipfarm import brand
from clipfarm.chat.log import ChatLog
from clipfarm.chat.twitch import TwitchChat
from clipfarm.craft.chatstrip import _contrast, chat_ass, fit, pick_messages, user_colour
from clipfarm.models import ChatMessage

CFG = {"chat_lines": 3, "chat_font_size": 44, "chat_min_gap_s": 0.7}
BOX = (80, 1354, 956, 1578)


def test_twitch_keeps_name_colour():
    line = ("@color=#1E90FF;display-name=LottePlays;tmi-sent-ts=1700000000000 "
            ":lotteplays!lotteplays@lotteplays.tmi.twitch.tv PRIVMSG #x :KEKW")
    m = TwitchChat("x")._parse(line)
    assert (m.user, m.text, m.color) == ("lotteplays", "KEKW", "#1E90FF")


def test_dark_names_are_lightened_to_read_on_the_strip():
    for c in ("#0000FF", "#000000", "#1E90FF", None):
        assert _contrast(user_colour("u", c), brand.NIGHT) >= 4.5


def test_bots_commands_and_links_are_left_out():
    chat = [[1, "Nightbot", "follow!", None], [2, "a", "!discord", None],
            [3, "b", "go to www.spam.com", None], [4, "c", "KEKW", None]]
    assert [m[1] for m in pick_messages(chat, 10, 3, 0.7)] == ["c"]


def test_fast_chat_is_throttled_and_preroll_shows_from_frame_one():
    chat = [[-5, "old1", "a", None], [-1, "old2", "b", None]] + \
        [[1 + i * 0.1, f"u{i}", "W", None] for i in range(20)]
    picked = pick_messages(chat, 10, 3, 0.7)
    assert [m[0] for m in picked[:2]] == [0.0, 0.0]
    times = [m[0] for m in picked[2:]]
    assert all(b - a >= 0.7 - 1e-9 for a, b in zip(times, times[1:]))


def test_long_messages_are_cut():
    assert fit("a" * 100, 10) == "a" * 9 + "…"


def test_chat_ass_shows_last_lines_and_label():
    chat = [[i, f"user{i}", f"msg {i}", None] for i in range(5)]
    styles, events = chat_ass(chat, 6, CFG, BOX)
    assert styles and "TWITCH CHAT" in events[0]
    assert events[-1].count("\\N") == 2 and "user4" in events[-1] and "user1" not in events[-1]
    assert chat_ass([], 6, CFG, BOX) == ([], [])


def test_chatlog_window_is_relative_to_clip_start():
    log = ChatLog()
    for t in (80, 95, 105, 130, 200):
        log.add(ChatMessage("c", t, "u", str(t)))
    assert [m[0] for m in log.between(100, 150)] == [-5, 5, 30]
