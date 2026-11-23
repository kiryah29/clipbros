"""Twitch chat strip: the chat that was live during the clip, burned in as ASS events.

Layout (design option C): a solid Night band with a lime line on top, just above the
TikTok description bar. A "TWITCH CHAT" label, then the last few messages, newest at the
bottom, each name in the user's own Twitch colour. Messages appear at the moment they
were sent; in a fast chat at most one new line every `chat_min_gap_s`, so it stays
readable. Bots, !commands and links are left out; swear words are starred like captions.
"""
from __future__ import annotations

import re

from .. import brand
from .captions import CENSOR_ROOTS, CENSOR_WORDS, _esc, _t, censor

BOTS = {"nightbot", "streamelements", "moobot", "fossabot", "streamlabs", "wizebot",
        "sery_bot", "soundalerts", "botrixoficial", "kofistreambot", "pokemoncommunitygame"}
# Twitch's default name colours, for users who never picked one
DEFAULT_COLOURS = ["#FF0000", "#0000FF", "#008000", "#B22222", "#FF7F50", "#9ACD32",
                   "#FF4500", "#2E8B57", "#DAA520", "#D2691E", "#5F9EA0", "#1E90FF",
                   "#FF69B4", "#8A2BE2", "#00FF7F"]
LINK = re.compile(r"https?://|www\.|\.(com|gg|tv|net|org)\b", re.I)


def _lum(h: str) -> float:
    def ch(c: int) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = brand.rgb(h)
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def user_colour(user: str, colour: str | None, ground: str = brand.NIGHT) -> str:
    """The user's Twitch colour (or Twitch's default pick for them), lightened until it
    reads on the strip (4.5:1). Dark blue names would vanish on Night otherwise."""
    c = colour if colour and re.fullmatch(r"#[0-9A-Fa-f]{6}", colour) else \
        DEFAULT_COLOURS[sum(map(ord, user.lower())) % len(DEFAULT_COLOURS)]
    r, g, b = brand.rgb(c)
    for _ in range(12):
        c = f"#{r:02X}{g:02X}{b:02X}"
        if _contrast(c, ground) >= 4.5:
            break
        r, g, b = (int(x + (255 - x) * 0.25) for x in (r, g, b))
    return c


def keep(user: str, text: str) -> bool:
    t = text.strip()
    return bool(t) and user.lower() not in BOTS and not t.startswith("!") and not LINK.search(t)


def pick_messages(chat: list, duration: float, lines: int, min_gap: float) -> list:
    """Messages to show: the last `lines` sent before the clip starts (shown from frame
    one), then at most one new message per `min_gap` seconds during the clip."""
    msgs = sorted((m for m in chat if keep(m[1], m[2]) and m[0] <= duration),
                  key=lambda m: m[0])
    before = [m for m in msgs if m[0] < 0][-lines:]
    out, last = [[0.0, *m[1:]] for m in before], -min_gap
    for m in msgs:
        if m[0] >= 0 and m[0] - last >= min_gap:
            out.append(m)
            last = m[0]
    return out


def fit(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max(1, max_chars - 1)].rstrip() + "…"


def chat_ass(chat: list, duration: float, cfg: dict,
             box: tuple[int, int, int, int]) -> tuple[list[str], list[str]]:
    """(ASS style lines, ASS event lines) for the strip inside box = (left, top, right,
    bottom) in reel pixels. Empty when there is nothing to show."""
    lines = int(cfg.get("chat_lines", 3))
    fs = int(cfg.get("chat_font_size", 44))
    msgs = pick_messages(chat, duration, lines, float(cfg.get("chat_min_gap_s", 0.7)))
    if not msgs:
        return [], []
    left, top, right, bottom = box
    font = cfg.get("caption_font", brand.ONCLIP_FONT_NAME)
    paper, fog = brand.ass_colour(brand.PAPER), brand.ass_colour(brand.FOG)
    styles = [f"Style: Chat,{font},{fs},{paper},{paper},{paper},{paper},0,0,0,0,100,100,0,0,"
              f"1,0,0,1,0,0,0,1"]
    label = cfg.get("chat_label", "TWITCH CHAT")
    events = [f"Dialogue: 3,{_t(0)},{_t(duration)},Chat,,0,0,0,,"
              f"{{\\an7\\pos({left},{top})\\fs{int(fs * 0.7)}\\fsp3\\c{fog}}}{_esc(label)}"]
    max_chars = int((right - left) / (fs * 0.58))   # Inter ExtraBold averages ~0.58 em
    censor_on = cfg.get("caption_censor", True)

    def line(m: list, faded: bool) -> str:
        user, text = _esc(m[1]), m[2]
        if censor_on:
            text = " ".join(censor(w, cfg.get("censor_roots", CENSOR_ROOTS),
                                   cfg.get("censor_words", CENSOR_WORDS)) for w in text.split())
        text = _esc(fit(text, max(4, max_chars - len(user) - 1)))
        name = brand.ass_colour(user_colour(m[1], m[3]))
        body = fog if faded else paper
        return f"{{\\c{name}}}{user}{{\\c{body}}} {text}"

    for i, m in enumerate(msgs):
        start = m[0]
        end = msgs[i + 1][0] if i + 1 < len(msgs) else duration
        if end <= start:
            continue
        shown = msgs[max(0, i - lines + 1):i + 1]
        text = "\\N".join(line(s, faded=(k == 0 and len(shown) == lines))
                          for k, s in enumerate(shown))
        events.append(f"Dialogue: 3,{_t(start)},{_t(end)},Chat,,0,0,0,,"
                      f"{{\\an1\\pos({left},{bottom})}}{text}")
    return styles, events
