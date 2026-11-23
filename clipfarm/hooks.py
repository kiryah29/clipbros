"""Hook card text written from what actually happens in the clip.

Every clip used to get one fixed line per trigger type, so a stream full of chat spikes
produced the same "Chat could NOT believe this" card over and over. Now each clip's
receipt is read for concrete details, most specific first:

  moment     the game profile's moment type (config/games/<game>.yaml hook_templates)
  phrase     an excited phrase the streamer said ("no way", "no x3" -> "NO NO NO")
  quote      a short line the streamer said right before the peak (transcript)
  reaction   what chat spammed: laughing, W, L, shock, confusion, fear, RIP
  big        stream got very loud AND chat went far past a normal spike
  audio      loud on its own (screaming)
  chat       chat sped up (uses the real "6x faster" number)
  fallback   reel.hook_text

Each kind has several templates (config: reel.hooks.templates). A template whose
details are missing, unsafe, or too long is skipped, and the last few hooks of the
channel are skipped too, so consecutive clips don't share a card. Choice is seeded by
the clip's peak time, so the same clip always gets the same hook.
"""
from __future__ import annotations

import random
import re
import string
from collections import deque

from .moments import moment_def

# Chat token -> reaction. Matched case-insensitively on whole tokens; stretched letters
# ("LMAOOOO") are collapsed first.
REACTIONS: dict[str, set[str]] = {
    "laugh": {"lmao", "lmfao", "lol", "lul", "kekw", "omegalul", "icant", "😂", "🤣",
              "haha", "hahaha", "kek", "lulw", "pepelaugh"},
    "dead": {"💀", "dead", "ded", "☠️"},
    "win": {"w", "ww", "www", "dub", "pog", "poggers", "pogchamp", "🔥", "goat", "clutch"},
    "loss": {"l", "ll", "lll", "ratio", "unlucky", "noooo"},
    "shock": {"wtf", "omg", "😱", "😳", "nah", "nahh", "insane", "wow"},
    "confused": {"?", "??", "???", "huh", "what", "wut", "🤨", "🤔"},
    "scared": {"monkas", "monkaw", "😨", "😰", "scary", "run"},
    "rip": {"rip", "f", "o7", "🪦", "sadge", "😭"},
}

DEFAULT_TEMPLATES: dict[str, list[str]] = {
    "phrase": ['{streamer}: "{phrase}"', 'The moment {streamer} said "{phrase}"',
               '"{phrase}" - {streamer}', '{streamer} could only say "{phrase}"'],
    "quote": ['{streamer}: "{quote}"', '"{quote}"', 'Then {streamer} said "{quote}"'],
    "laugh": ["Chat could NOT stop laughing", "{streamer} had chat in tears",
              "Chat lost it at this", 'Chat spammed "{chat_word}" for this'],
    "dead": ["Chat was DEAD after this", "{streamer} ended chat with this",
             "Nobody survived this one"],
    "win": ['Chat spammed "W" for this', "Biggest W of the stream",
            "{streamer} actually pulled this off"],
    "loss": ['Chat spammed "L" for this', "This was a massive L",
             "{streamer} will never live this down"],
    "shock": ["Chat was speechless", "Nobody saw this coming",
              "{streamer} did NOT see this coming"],
    "confused": ["Chat had questions", "What was {streamer} thinking?",
                 "Nobody knows what just happened"],
    "scared": ["Chat was scared for {streamer}", "{streamer} was NOT safe here"],
    "rip": ["RIP {streamer}", "It was over for {streamer}", "{streamer} never stood a chance"],
    "big": ["WAIT FOR IT...", "Watch till the end", "The ending is insane"],
    "audio": ["{streamer} was NOT ready for this", "{streamer} completely lost it",
              "Turn your sound up for this", "{streamer} screamed so loud"],
    "chat": ["Chat could NOT believe this", "Chat went {chat_x}x faster for this",
             "Chat exploded at this", "Chat went crazy for this"],
}

# never burned onto a clip, not even inside a quote (extend with reel.hooks.blocklist)
DEFAULT_BLOCKLIST = {
    "fuck", "fucking", "fucked", "shit", "bitch", "cunt", "dick", "pussy", "cock",
    "whore", "slut", "rape", "raped", "nazi", "hitler", "retard", "retarded", "fag",
    "faggot", "kys", "suicide", "porn", "nigga", "nigger",
}

FILLER = {"uh", "um", "uhh", "umm", "like", "the", "a", "an", "and", "so", "i", "you", "it",
          "oh", "to", "is", "of", "in", "on", "that", "this", "but", "or", "yeah", "okay"}
STRETCH_RE = re.compile(r"(.)\1{2,}")
FIELD_RE = re.compile(r"{(\w+)}")


def _norm(tok: str) -> str:
    return STRETCH_RE.sub(r"\1\1", tok.lower().strip(string.punctuation.replace("?", "")) or tok)


class HookHistory:
    """The channel's last few hooks, so the next clip picks something else."""

    def __init__(self, size: int = 6):
        self.texts: deque[str] = deque(maxlen=size)
        self.templates: deque[str] = deque(maxlen=size)

    def seen(self, text: str, template: str) -> bool:
        return text.lower() in (t.lower() for t in self.texts) or template in self.templates

    def add(self, text: str, template: str) -> None:
        self.texts.append(text)
        self.templates.append(template)


def reaction_of(top_tokens, min_share: float = 0.15) -> tuple[str, str] | None:
    """(reaction, the token chat spammed) from a spike's top tokens, if one dominates."""
    toks = [(_norm(str(t)), str(t), int(n)) for t, n in (top_tokens or [])]
    total = sum(n for _, _, n in toks)
    if not total:
        return None
    score: dict[str, int] = {}
    best_tok: dict[str, tuple[int, str]] = {}
    for low, raw, n in toks:
        single = re.sub(r"(.)\1+", r"\1", low)          # "💀💀💀" -> "💀", "lmaoo" -> "lmao"
        for kind, words in REACTIONS.items():
            if low in words or single in words:
                score[kind] = score.get(kind, 0) + n
                if n > best_tok.get(kind, (0, ""))[0]:
                    best_tok[kind] = (n, raw)
    if not score:
        return None
    kind = max(score, key=score.get)
    if score[kind] / total < min_share:
        return None
    return kind, best_tok[kind][1]


def _said(phrase: str) -> str:
    word, _, n = phrase.partition(" x")          # "no x3" -> "no no no"
    if not n.isdigit():
        return phrase
    return "" if word in FILLER else " ".join([word] * min(int(n), 4))   # not "THE THE THE"


def quote_near(transcript, peak_rel: float | None, before_s: float = 8.0,
               min_words: int = 3, max_words: int = 6) -> str:
    """The streamer's last few words before the peak (times relative to clip start)."""
    if not transcript or peak_rel is None:
        return ""
    words = [str(w[0]).strip() for w in transcript
             if peak_rel - before_s <= float(w[1]) <= peak_rel + 1.0 and str(w[0]).strip()]
    words = words[-max_words:]
    content = [w for w in words if _norm(w) not in FILLER]
    if len(words) < min_words or len(content) < 3:
        return ""
    return " ".join(words).strip(" ,.")


def _safe(value: str, blocklist: set[str]) -> bool:
    return not any(_norm(t) in blocklist for t in re.findall(r"[\w']+", value.lower()))


def _facts(receipt: dict, cfg: dict, peak_rel: float | None, hcfg: dict) -> dict[str, str]:
    phrases = (receipt.get("speech") or {}).get("phrases") or []
    react = reaction_of(receipt.get("top_tokens"), float(hcfg.get("reaction_share", 0.15)))
    chat_x = receipt.get("ratio") or receipt.get("chat_ratio") or 0
    f = {
        "streamer": cfg.get("streamer") or cfg.get("name") or "",
        "game": cfg.get("game_name") or "",
        "phrase": next((_said(p).upper() for p in phrases if _said(p)), ""),
        "quote": quote_near(receipt.get("transcript"), peak_rel,
                            max_words=int(hcfg.get("max_quote_words", 6))),
        "chat_word": react[1] if react else "",
        "reaction": react[0] if react else "",
        "chat_x": str(int(chat_x)) if chat_x and float(chat_x) >= 2 else "",
        "moment": (receipt.get("moment") or {}).get("label", ""),
    }
    return {k: v for k, v in f.items() if v}


def _kinds(receipt: dict, cfg: dict, facts: dict) -> list[str]:
    """Moment kinds that apply to this clip, most specific first."""
    reel = cfg["reel"]
    trig = receipt.get("triggers") or []
    a, c = float(receipt.get("audio_ratio") or 0), float(receipt.get("chat_ratio") or 1)
    need_a = float(reel.get("wait_hook_audio_ratio") or cfg.get("audio", {}).get("min_ratio", 6.0))
    need_c = float(reel.get("wait_hook_chat_ratio", 5.0))
    kinds = []
    if receipt.get("moment"):
        kinds.append("moment")
    if "phrase" in facts:
        kinds.append("phrase")
    if "quote" in facts:
        kinds.append("quote")
    if "reaction" in facts:
        kinds.append(facts["reaction"])
    if a >= need_a and c >= need_c:
        kinds.append("big")
    if a >= need_a or "audio" in trig:
        kinds.append("audio")
    if "chat" in trig or c >= 2:
        kinds.append("chat")
    return kinds


def write_hook(receipt: dict, cfg: dict, history: HookHistory | None = None,
               peak_rel: float | None = None, seed: int | None = None) -> tuple[str, str]:
    """(hook text, kind) for a clip. Records the pick in `history` if given."""
    reel = cfg["reel"]
    hcfg = reel.get("hooks") or {}
    templates = {**DEFAULT_TEMPLATES, **(hcfg.get("templates") or {})}
    moment = receipt.get("moment") or {}
    templates["moment"] = moment_def(cfg.get("moments") or [], moment.get("name")).get(
        "hook_templates") or []
    block = DEFAULT_BLOCKLIST | {w.lower() for w in hcfg.get("blocklist") or []}
    max_chars = int(hcfg.get("max_chars", 42))
    facts = {k: v for k, v in _facts(receipt, cfg, peak_rel, hcfg).items() if _safe(v, block)}
    rng = random.Random(seed)

    fallback = None
    for kind in _kinds(receipt, cfg, facts):
        pool = list(templates.get(kind) or [])
        rng.shuffle(pool)
        for t in pool:
            need = set(FIELD_RE.findall(t))
            if not need <= facts.keys():
                continue
            text = t.format(**facts).strip()
            if len(text) > max_chars or not _safe(text, block):
                continue
            if history and history.seen(text, t):
                fallback = fallback or (text, kind, t)   # usable, just used recently
                continue
            if history:
                history.add(text, t)
            return text, kind
    if fallback:                       # everything fitting was used lately: reuse the first
        text, kind, t = fallback
        if history:
            history.add(text, t)
        return text, kind
    return reel.get("hook_text") or "WAIT FOR IT...", "fallback"
