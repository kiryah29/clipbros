"""Rule-based hook writer: facts about the clip -> a few hook patterns -> scored candidates.

The template writer (hooks.py) mostly put a few words the streamer said on the card. A
viewer hears those a second later anyway, and a quote without context means nothing to a
stranger. This writer collects facts first and only uses a pattern when every slot is
filled from a real fact, so a hook never invents anything:

  core facts (every game)   streamer, game name, plan line ("I'm gonna rob the bank"),
                            setup line ("watch this"), best real chat message, Twitch clip
                            title, peak time, chat reaction, ClipAgent clip type
  profile facts (optional)  game nouns, HUD outcome / counters, sound events mapped to
                            words, the game's own extra patterns (config/games/<game>.yaml)

Patterns (core, every game):

  story          {streamer} tried to {action}
  story_outcome  {streamer} tried to {action}. It failed             (HUD reports a fail)
  warning        DO NOT {action} in {game}                          (something went wrong)
  prediction     {streamer} said "{setup}"                          (and it went wrong)
  stakes         {stakes} and {streamer} still went for it         (HUD counters)
  payoff         Wait for the {thing}                               (peak 3 s+ in)
  chat_voice     Chat: "{message}"                                  (echoed by other chatters)
  viewer_title   {title}                                            (Twitch clip title)
  quote          {streamer}: "{quote}"                              (demoted)
  game           the profile's extra patterns and moment templates

Every candidate is scored (style weight, 4-8 words, concrete nouns, known outcome, minus
generic words and minus repeating the reel's first spoken line) and the channel's last
few hooks are skipped by text and by style. Nothing usable means today's templates
(hooks.write_hook), minus the chat-speed lines.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from .hooks import (DEFAULT_BLOCKLIST, FIELD_RE, FILLER, HookHistory, _norm, _safe,
                    quote_near, reaction_of, sentences, write_hook)
from .moments import moment_def

STYLE_WEIGHTS = {"story": 1.2, "story_outcome": 1.3, "warning": 1.2, "prediction": 1.1,
                 "stakes": 1.0, "payoff": 1.0, "viewer_title": 1.0, "game": 0.9,
                 "chat_voice": 0.8, "quote": 0.6}

# (style, template, extra facts that must be present besides the template's own slots)
CORE_PATTERNS: list[tuple[str, str, tuple[str, ...]]] = [
    ("story_outcome", "{streamer} tried to {action}. It failed", ("failed",)),
    ("story", "{streamer} tried to {action}", ()),
    ("warning", "DO NOT {action_you} in {game}", ("went_wrong",)),
    ("prediction", '{streamer} said "{setup}"', ("went_wrong",)),
    ("prediction", "{streamer} really thought this would work", ("setup", "went_wrong")),
    ("stakes", "{stakes} and {streamer} still went for it", ()),
    ("payoff", "Wait for the {thing}", ("late_peak",)),
    ("chat_voice", 'Chat: "{message}"', ()),
    ("viewer_title", "{title}", ()),
    ("quote", '{streamer}: "{quote}"', ()),
]

PLAN_PHRASES = ["i'm gonna", "im gonna", "i am gonna", "i'm going to", "im going to",
                "let's", "lets", "watch me", "bet i can", "i'm about to", "im about to",
                "imma", "i'ma", "i'm finna", "i will", "i'll"]
SETUP_PHRASES = ["watch this", "watch me", "trust me", "i got this", "too easy", "light work",
                 "easy money", "this is gonna work", "this is going to work", "i can't lose",
                 "no way this fails", "it's over for"]

# a plan line that starts with these is talk, not a plan ("I'm gonna be honest")
NOT_A_PLAN = {"be", "say", "tell", "have", "need", "want", "think", "cry", "die", "lose", "leave",
              "stop", "see", "check", "look", "talk", "explain", "show", "start", "end", "sleep",
              "read", "reload", "level", "play", "call", "level", "pee", "eat", "drink", "not",
              "never", "just", "actually", "really", "probably", "the", "a", "an", "it", "this",
              "that", "to", "um", "uh", "so", "and", "you", "we", "he", "she", "they", "i",
              "gonna", "go"}
# words that cut a plan line short ("rob the bank and then ..." -> "rob the bank")
PLAN_STOP = {"and", "but", "because", "cause", "cuz", "so", "if", "when", "then", "bro", "chat",
             "dude", "man", "yo", "right", "okay", "ok", "like", "uh", "um", "now", "guys", "real",
             "quick", "please", "lol", "bruh", "while", "until", "before", "after", "or"}
PREPOSITIONS = {"from", "to", "with", "in", "at", "on", "over", "into", "for", "by", "near", "under"}
ADDRESSING = {"you", "your", "you're", "y'all", "ya", "u", "yall", "yourself"}

THIRD = {
    "he": {"my": "his", "me": "him", "myself": "himself", "mine": "his", "i": "he",
           "i'm": "he's", "i've": "he's", "i'll": "he'll", "we": "they", "us": "them",
           "our": "their", "ours": "theirs"},
    "she": {"my": "her", "me": "her", "myself": "herself", "mine": "hers", "i": "she",
            "i'm": "she's", "i've": "she's", "i'll": "she'll", "we": "they", "us": "them",
            "our": "their", "ours": "theirs"},
    "they": {"my": "their", "me": "them", "myself": "themself", "mine": "theirs", "i": "they",
             "i'm": "they're", "i've": "they've", "i'll": "they'll", "we": "they", "us": "them",
             "our": "their", "ours": "theirs"},
}
SECOND = {"my": "your", "me": "yourself", "myself": "yourself", "mine": "yours", "i": "you",
          "i'm": "you're", "we": "you", "us": "you", "our": "your", "ours": "yours"}

# a plan needs one concrete word: "run this side" says nothing to a stranger
VAGUE = {"this", "that", "here", "there", "side", "way", "thing", "things", "it", "one", "ones",
         "stuff", "them", "these", "those", "something", "everything", "anything", "him", "her",
         "his", "their", "your", "yourself", "himself", "herself", "real", "quick", "again"}
# chat lines that are orders to the streamer, not a reaction anyone else understands
ORDERS = {"go", "switch", "just", "use", "try", "get", "turn", "press", "run", "do", "don't",
          "dont", "buy", "sell", "calm", "lock", "swim", "kill", "stop", "shoot", "drive", "take",
          "put", "check", "look", "read", "play", "jump", "hit", "open", "close", "leave", "wait",
          "type", "say", "pick", "grab", "save", "follow", "sub", "watch"}
# chat lines must react to the streamer or the moment, not start a side conversation
ABOUT = {"he", "his", "him", "she", "her", "bro", "bros", "cooked", "thought", "really", "never",
         "always", "actually", "accidentally", "how", "why", "nah", "bait", "fake", "scripted",
         "glitched", "fumbled", "fell", "died", "dead", "lost", "won", "clutch", "clutched",
         "trolled", "robbed", "caught", "missed", "nobody", "everyone"}
# stricter than the card blocklist: a stranger's message goes on the card word for word
CHAT_BLOCK = {"ass", "tf", "stfu", "wtf", "epstein", "diddy", "meat", "sex", "sexy", "gay",
              "racist", "kill", "killed", "dumb", "stupid", "idiot", "ugly", "fat", "trash",
              "hoe", "simp", "mom", "moms", "mother", "nuts", "drugs", "weed", "bomb", "terrorist"}

GENERIC = {"this", "insane", "crazy", "chat", "wait", "moment", "literally", "thing", "stuff",
           "something", "omg", "wow", "bro"}
EMOTE_RE = re.compile(r"^[a-z0-9]{2,}[A-Z][A-Za-z0-9]*$")      # cohhLUL, kaiW, xqcL
SPAM_RE = re.compile(r"https?://|www\.|\.com\b|\b\d+k\+?\b|\$\d|\b(call|dm|follow|subscribe|"
                     r"sub to|promo|giveaway|discount|code)\b", re.I)
WORD_RE = re.compile(r"[\w']+")
EMOTES = {"kekw", "lul", "lulw", "omegalul", "pog", "poggers", "pogchamp", "kappa", "monkas",
          "monkaw", "pepelaugh", "sadge", "icant", "aware", "clueless", "copium", "o7", "ez",
          "gg", "w", "l", "lmao", "lmfao", "lol", "xd", "f"}


def _clean(tok: str) -> str:
    return tok.replace("’", "'").lower().strip(".,!?;:\"()[]-—…")


def _nouns(cfg: dict) -> set[str]:
    """The profile's nouns as single lower-case words ("Vice City" -> vice, city)."""
    return {_clean(t) for n in cfg.get("nouns") or [] for t in WORD_RE.findall(str(n))}


def hook_cfg(cfg: dict) -> dict:
    """reel.hooks with the game profile's top-level `hooks:` block merged over it."""
    base = dict((cfg.get("reel") or {}).get("hooks") or {})
    prof = cfg.get("hooks") or {}
    out = {**base, **{k: v for k, v in prof.items() if k not in ("style_weights", "extra_patterns")}}
    out["style_weights"] = {**STYLE_WEIGHTS, **(base.get("style_weights") or {}),
                            **(prof.get("style_weights") or {})}
    out["extra_patterns"] = {**(base.get("extra_patterns") or {}),
                             **(prof.get("extra_patterns") or {})}
    return out


# --------------------------------------------------------------------------- facts
def _words(transcript, min_voice: float | None = None) -> list[tuple[str, float, float]]:
    out = []
    for w in transcript or []:
        voice = w[3] if len(w) > 3 else None
        if min_voice is not None and voice is not None and voice < min_voice:
            continue                     # not the streamer's voice (game character, radio)
        text = str(w[0]).strip()
        if text:
            out.append((text, float(w[1]), float(w[2])))
    return out


def _match(toks: list[str], i: int, phrases: list[list[str]]) -> int:
    """Length of the phrase starting at toks[i], 0 if none."""
    for p in phrases:
        if toks[i:i + len(p)] == p:
            return len(p)
    return 0


def _action(words: list[tuple[str, float, float]], start: int, max_words: int,
            pause_s: float = 0.6) -> list[str] | None:
    """The plan words after a plan phrase, cut at a sentence end, a pause or a filler word."""
    out: list[str] = []
    prev_end = words[start - 1][2] if start else None
    for text, s, e in words[start:]:
        if prev_end is not None and out and s - prev_end >= pause_s:
            break
        low = _clean(text)
        if not low or low in PLAN_STOP:
            break
        out.append(text.replace("’", "'").strip(",;:\"()"))
        prev_end = e
        if text.rstrip().endswith((".", "!", "?")) or len(out) > max_words + 4:
            break
    out = [w.rstrip(".!?") for w in out]
    keep_go = PREPOSITIONS | {"back", "home", "up", "down", "out", "in", "off", "away"}
    while len(out) > 2 and _clean(out[0]) in ("try", "to", "and", "go"):
        if _clean(out[0]) == "go" and _clean(out[1]) in keep_go:
            break                          # "go to the bank" stays (and is rejected as talk)
        out = out[1:]   # "try to clutch" -> "clutch", "go rob the bank" -> "rob the bank"
    if len(out) > max_words:               # too long: cut before a preposition
        cut = next((i for i, w in enumerate(out) if i >= 2 and _clean(w) in PREPOSITIONS), None)
        out = out[:cut] if cut and cut <= max_words else []
    while out and _clean(out[-1]) in FILLER | PREPOSITIONS:
        out = out[:-1]
    lows = [_clean(w) for w in out]
    if len(out) < 2 or lows[0] in NOT_A_PLAN or any(w in ADDRESSING for w in lows):
        return None
    if not any(w not in FILLER | NOT_A_PLAN | VAGUE and len(w) >= 3 for w in lows[1:]):
        return None                        # "get it", "do this", "run this side": no content
    return out


def _person(words: list[str], table: dict, nouns: set[str]) -> str:
    out = []
    for i, w in enumerate(words):
        low = _clean(w)
        if low in table:
            out.append(table[low])
        elif i == 0 and low not in nouns and w[:1].isupper() and w != "I":
            out.append(w[:1].lower() + w[1:])
        else:
            out.append(w)
    return " ".join(out)


def plan_line(words, peak: float | None, phrases: list[str], pronoun: str = "he",
              nouns: set[str] | None = None, max_words: int = 6, before_s: float = 20.0
              ) -> dict | None:
    """The streamer's last plan before the peak: "I'm gonna rob the bank" ->
    {"action": "rob the bank", "action_you": "rob the bank", "said": "I'm gonna rob the bank"}.
    my -> his (story) and my -> your (warning)."""
    nouns = nouns or set()
    toks = [_clean(w[0]) for w in words]
    plist = sorted((p.split() for p in phrases), key=len, reverse=True)
    best = None
    for i in range(len(toks)):
        n = _match(toks, i, plist)
        if not n:
            continue
        t = words[i][1]
        if peak is not None and not (peak - before_s <= t <= peak + 1.0):
            continue
        act = _action(words, i + n, max_words)
        if act:
            best = {"action": _person(act, THIRD.get(pronoun, THIRD["he"]), nouns),
                    "action_you": _person(act, SECOND, nouns),
                    "said": " ".join(w[0] for w in words[i:i + n]) + " " + " ".join(act),
                    "at": round(t, 2)}
    return best


def setup_line(words, peak: float | None, phrases: list[str], before_s: float = 20.0) -> dict | None:
    """A confident line said before the peak ("watch this", "trust me")."""
    toks = [_clean(w[0]) for w in words]
    plist = sorted((p.split() for p in phrases), key=len, reverse=True)
    found = None
    for i in range(len(toks)):
        n = _match(toks, i, plist)
        t = words[i][1] if n else None
        if n and (peak is None or peak - before_s <= t <= peak):
            text = " ".join(w[0] for w in words[i:i + n]).strip(" ,.!?")
            found = {"setup": text[:1].upper() + text[1:], "at": round(t, 2)}
    return found


def best_chat_message(chat, sample, peak: float | None, streamer: str, nouns: set[str],
                      block: set[str], max_chars: int, min_support: int = 2) -> dict | None:
    """A short, readable chat message that other chatters echoed (shared words, different
    users), from the seconds after the peak, that reacts to the streamer ("he really thought
    that would work"). Out: emotes, links, @replies, promo spam, orders to the streamer
    ("switch to ..."), lines talking to the streamer ("you need a winch") and anything on
    the stricter chat blocklist, since a stranger's words go on the card unchanged."""
    msgs = []
    for m in chat or []:
        t, user, text = float(m[0]), str(m[1]), str(m[2])
        if peak is None or peak - 3 <= t <= peak + 15:
            msgs.append((user, text))
    if not msgs:
        msgs = [(f"s{i}", str(t)) for i, t in enumerate(sample or [])]
    cleaned = []
    for user, text in msgs:
        text = text.strip()
        if not text or text.startswith(("@", "!")) or SPAM_RE.search(text):
            continue
        toks = [t for t in text.split() if not EMOTE_RE.match(t) and _norm(t) not in EMOTES]
        toks = [re.sub(r"(\w)\1{2,}", r"\1\1", t) for t in toks if WORD_RE.search(t)]
        toks = [re.sub(r"([^lsfzeo\W\d])\1$", r"\1", t) for t in toks]   # "soldd" -> "sold"
        if not 3 <= len(toks) <= 8:
            continue
        line = " ".join(toks)
        if line.isupper() or sum(c.isupper() for c in line) > len(line) * 0.6:
            line = line.lower()
        if len(line) > max_chars or not _safe(line, block):
            continue
        lows = [_clean(t) for t in toks]
        content = set(lows) - FILLER - EMOTES
        if len(content) < 2 or lows[0] in ORDERS or set(lows) & (ADDRESSING | CHAT_BLOCK):
            continue
        names = {_clean(streamer), *nouns} | {n[:4] for n in [_clean(streamer)] if len(n) > 4}
        if not set(lows) & (ABOUT | names) and not any(t.startswith(tuple(names)) for t in lows):
            continue
        cleaned.append((user, line, content))
    best, best_score = None, 0.0
    for user, line, content in cleaned:
        others = {u for u, _, c in cleaned if u != user and len(content & c) >= max(1, len(content) // 2)}
        support = len(others)
        if support < min_support:
            continue
        words = line.split()
        score = support + (0.5 if 4 <= len(words) <= 7 else 0)
        if _clean(streamer) in content or content & nouns:
            score += 0.5
        if score > best_score:
            best, best_score = {"message": line, "support": support}, score
    return best


def clip_title(receipt: dict, block: set[str], max_chars: int) -> str:
    """The Twitch clip a viewer made of this moment, if its title reads like a hook. A clip
    nobody renamed carries the stream title, which says nothing about the moment."""
    tc = receipt.get("twitch_clip") or {}
    title = str(tc.get("title") or "").strip()
    if not title or title == str(tc.get("stream_title") or receipt.get("stream_title") or ""):
        return ""
    if len(title) > max_chars or len(title.split()) < 2 or not _safe(title, block) or SPAM_RE.search(title):
        return ""
    return title


def _hud(receipt: dict, cfg: dict, min_conf: float) -> dict:
    """HUD outcome and stakes from the receipt's `hud` entry (filled by HUD matching)."""
    hud = receipt.get("hud") or {}
    events = [e for e in hud.get("events") or [] if float(e.get("confidence", 0)) >= min_conf]
    out = {}
    outcomes = {e.get("outcome") for e in events if e.get("outcome")}
    if "fail" in outcomes:
        out["failed"] = "1"
    elif "win" in outcomes:
        out["won"] = "1"
    if hud.get("stakes"):
        out["stakes"] = str(hud["stakes"])
    words = [(float(e.get("t", 0)), e.get("word")) for e in events if e.get("word")]
    if words:
        out["_hud_words"] = words
    return out


def _sound_thing(receipt: dict, cfg: dict, peak: float | None, min_conf: float) -> str:
    """The game word for the sound nearest the peak (profile sound_words: siren -> cops)."""
    mapping = {str(k).lower(): v for k, v in (cfg.get("sound_words") or {}).items()}
    best = None
    for ev in receipt.get("sound_events") or []:
        word = mapping.get(str(ev.get("label", "")).lower())
        if not word or float(ev.get("score", 0)) < min_conf:
            continue
        d = abs(float(ev.get("t", 0)) - (peak or 0))
        if best is None or d < best[0]:
            best = (d, word)
    return best[1] if best else ""


def collect_facts(receipt: dict, cfg: dict, words=None, chat=None, peak: float | None = None
                  ) -> dict:
    """Every fact a hook may use. `words`/`chat`/`peak` are timed from the same zero (the
    reel start at craft time, the clip window start live); defaults read the receipt."""
    hc = hook_cfg(cfg)
    block = DEFAULT_BLOCKLIST | {w.lower() for w in hc.get("blocklist") or []}
    max_chars = int(hc.get("max_chars", 42))
    pronoun = str(cfg.get("pronoun") or "he")
    nouns = _nouns(cfg)
    qmin = float((cfg.get("voice") or {}).get("quote_min_similarity", 0.3))
    ws = _words(words if words is not None else receipt.get("transcript"), qmin)
    min_conf = float(hc.get("min_confidence", 0.8))
    streamer = cfg.get("streamer") or cfg.get("name") or ""

    f: dict = {"streamer": streamer, "game": cfg.get("game_name") or ""}
    if ws:
        plan = plan_line(ws, peak, hc.get("plan_phrases") or PLAN_PHRASES, pronoun, nouns,
                         int(hc.get("max_action_words", 6)))
        if plan:
            f.update(action=plan["action"], action_you=plan["action_you"], _plan=plan)
        setup = setup_line(ws, peak, hc.get("setup_phrases") or SETUP_PHRASES)
        if setup:
            f.update(setup=setup["setup"], _setup=setup)
        q = quote_near([list(w) for w in ws], peak, 8.0, max_words=int(hc.get("max_quote_words", 6)))
        if q:
            f["quote"] = q
        first = next((s for s in sentences([list(w) for w in ws]) if s[0][1] >= -0.5), None)
        if first and first[0][1] <= 3.0:
            f["_first_line"] = " ".join(w[0] for w in first)
    msg = best_chat_message(chat if chat is not None else receipt.get("chat"),
                            receipt.get("sample_messages"), peak, streamer, nouns, block,
                            max_chars - 8, int(hc.get("chat_min_support", 2)))
    if msg:
        f.update(message=msg["message"], _chat_support=msg["support"])
    title = clip_title(receipt, block, max_chars)
    if title:
        f["title"] = title

    hud = _hud(receipt, cfg, min_conf)
    f.update({k: v for k, v in hud.items() if not k.startswith("_hud")})
    thing = _sound_thing(receipt, cfg, peak, min_conf)
    if not thing and hud.get("_hud_words"):
        thing = min(hud["_hud_words"], key=lambda tw: abs(tw[0] - (peak or 0)))[1]
    if thing:
        f["thing"] = thing
    if peak is not None and peak >= float(hc.get("payoff_min_s", 3.0)):
        f["late_peak"] = "1"

    react = reaction_of(receipt.get("top_tokens"), float(hc.get("reaction_share", 0.15)))
    agent = receipt.get("clip_agent") or {}
    ctype = agent.get("clip_type")
    moment = receipt.get("moment") or {}
    mdef = moment_def(cfg.get("moments") or [], moment.get("name")) if moment else {}
    wrong = ("failed" in f or (react and react[0] in ("loss", "rip", "dead", "laugh"))
             or (ctype in ("fail", "rage") and float(agent.get("confidence") or 0) >= 0.5)
             or mdef.get("clip_type") == "fail")
    if wrong:
        f["went_wrong"] = "1"
    if moment.get("name"):
        f["moment"] = moment["name"]
    return {k: v for k, v in f.items() if v and (k.startswith("_") or _safe(str(v), block))}


# --------------------------------------------------------------------------- patterns
@dataclass
class Candidate:
    text: str
    style: str
    score: float
    template: str
    facts: list[str] = field(default_factory=list)

    def to_list(self) -> list:
        return [self.text, self.style, round(self.score, 2)]


def _patterns(cfg: dict, hc: dict, facts: dict) -> list[tuple[str, str, tuple, dict]]:
    out = [(s, t, need, {}) for s, t, need in CORE_PATTERNS]
    for style, items in (hc.get("extra_patterns") or {}).items():
        for it in items or []:
            if isinstance(it, str):
                out.append((style, it, (), {}))
            elif isinstance(it, dict) and it.get("text"):
                out.append((it.get("style", style), it["text"], (), it.get("when") or {}))
    if facts.get("moment"):     # the game moment's own templates count as game patterns
        for t in moment_def(cfg.get("moments") or [], facts["moment"]).get("hook_templates") or []:
            out.append(("game", t, (), {}))
    return out


def _content(text: str) -> set[str]:
    return {_clean(t) for t in WORD_RE.findall(text)} - FILLER - GENERIC


def score(text: str, style: str, facts: dict, hc: dict, nouns: set[str],
          history: HookHistory | None = None, slots: set[str] | None = None) -> float:
    """Style weight + length + concrete words from the facts used (not the template's own
    words or the streamer/game name, which every clip has) - generic words - repeating the
    first spoken line - styles and texts the channel used lately."""
    s = float(hc["style_weights"].get(style, 1.0))
    words = text.split()
    n = len(words)
    s += 0.3 if 4 <= n <= 8 else 0.0 if n in (3, 9) else -0.4
    toks = {_clean(t) for t in WORD_RE.findall(text)}
    filled = {_clean(t) for k in (slots or ()) if k not in ("streamer", "game")
              for t in WORD_RE.findall(str(facts.get(k, "")))}
    s += min(0.4, 0.2 * len(filled & nouns))
    s += min(0.3, 0.1 * len({t for t in filled - FILLER - GENERIC - VAGUE if len(t) >= 4}))
    if style == "story_outcome" or (style == "prediction" and "failed" in facts):
        s += 0.2
    s -= 0.1 * len(toks & GENERIC)
    first = facts.get("_first_line")
    if first:
        c, f = _content(text), _content(first)
        if c and len(c & f) / len(c) >= 0.6:
            s -= 0.6                      # the viewer hears this line a second later anyway
    if history:
        if text.lower() in (t.lower() for t in history.texts):
            s -= 1.0
        if style in list(history.styles)[-2:]:
            s -= 0.3
    return s


def candidates(receipt: dict, cfg: dict, history: HookHistory | None = None, words=None,
               chat=None, peak: float | None = None, seed: int | None = None
               ) -> tuple[list[Candidate], dict]:
    """All filled patterns, best first, one text per style. Returns (candidates, facts)."""
    hc = hook_cfg(cfg)
    facts = collect_facts(receipt, cfg, words, chat, peak)
    block = DEFAULT_BLOCKLIST | {w.lower() for w in hc.get("blocklist") or []}
    max_chars = int(hc.get("max_chars", 42))
    nouns = _nouns(cfg)
    pool = _patterns(cfg, hc, facts)
    random.Random(seed).shuffle(pool)       # ties between templates of one style vary by clip
    seen, out = set(), []
    for style, t, need, when in pool:
        slots = set(FIELD_RE.findall(t)) | set(need)
        if not slots <= facts.keys() or any(str(facts.get(k)) != str(v) for k, v in when.items()):
            continue
        text = t.format(**{k: v for k, v in facts.items() if not k.startswith("_")}).strip()
        if not text or len(text) > max_chars or not _safe(text, block) or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(Candidate(text, style, score(text, style, facts, hc, nouns, history, slots),
                             t, sorted(slots)))
    out.sort(key=lambda c: -c.score)
    best_per_style: dict[str, Candidate] = {}
    for c in out:
        best_per_style.setdefault(c.style, c)
    ranked = sorted(best_per_style.values(), key=lambda c: -c.score)
    return ranked, facts


def public_facts(facts: dict) -> dict:
    """The facts worth keeping in a receipt (plain values, no helper keys)."""
    out = {k: v for k, v in facts.items() if not k.startswith("_")}
    for k in ("_plan", "_setup"):
        if k in facts:
            out[k[1:] + "_said"] = facts[k].get("said") or facts[k].get("setup")
            out[k[1:] + "_at"] = facts[k].get("at")
    if "_chat_support" in facts:
        out["chat_support"] = facts["_chat_support"]
    return out


def pick_hook(receipt: dict, cfg: dict, history: HookHistory | None = None, words=None,
              chat=None, peak: float | None = None, seed: int | None = None,
              writer: str | None = None) -> dict:
    """The hook for a clip and why: {hook, hook_kind, hook_writer, hook_facts,
    hook_candidates}. writer: rules | templates (default reel.hooks.writer)."""
    hc = hook_cfg(cfg)
    writer = writer or hc.get("writer", "templates")
    out: dict = {"hook_writer": "templates"}
    if writer == "rules":
        cands, facts = candidates(receipt, cfg, history, words, chat, peak, seed)
        out["hook_facts"] = public_facts(facts)
        out["hook_candidates"] = [c.to_list() for c in cands[:int(hc.get("candidates", 3))]]
        good = [c for c in cands if c.score >= float(hc.get("min_score", 1.0))]
        if good:
            c = good[0]
            if history:
                history.add(c.text, "rules:" + c.style)
                history.styles.append(c.style)
            return {**out, "hook": c.text, "hook_kind": c.style, "hook_writer": "rules"}
    r = receipt
    if words is not None:
        r = {**receipt, "transcript": [list(w[:3]) for w in words]}
    demote = set(hc.get("demote_kinds", ["phrase", "quote"]) or []) if writer == "rules" else None
    text, kind = write_hook(r, cfg, history, peak_rel=peak, seed=seed, demote=demote)
    if history:
        history.styles.append(kind)
    return {**out, "hook": text, "hook_kind": kind}
