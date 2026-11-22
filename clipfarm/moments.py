"""Game-profile moment hints: label a chat spike with the kind of moment it is.

A game profile (config/games/<game>.yaml) lists moment types, each with the
chat words viewers type when it happens. When a spike closes, its messages are
matched against those words. The best match:
  - feeds the game_events signal (share of messages that matched * weight, see signals.py)
  - picks the clip window (setup-heavy moments get a longer pre-roll)
  - picks a hook line from the moment's templates
  - flags moments that need a human safety look before posting
Everything here is optional: a channel without a game profile behaves as before.
"""
from __future__ import annotations

import random
import re
from typing import Iterable

WORD_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)


def _norm(text: str) -> str:
    # collapse stretched letters so "NAHHHH" matches "nahh" and "COPSSS" matches "cops"
    return re.sub(r"(.)\1{2,}", r"\1\1", text.lower())


def classify(messages: Iterable[str], moments: list[dict]) -> dict | None:
    msgs = [_norm(m) for m in messages if m]
    if not msgs or not moments:
        return None
    best = None
    for m in moments:
        keys = [_norm(k) for k in m.get("chat_keywords", [])]
        if not keys:
            continue
        hits = sum(1 for msg in msgs if any(_hit(k, msg) for k in keys))
        share = hits / len(msgs)
        if share < float(m.get("min_share", 0.08)):
            continue
        rank = share * float(m.get("weight", 1.0))
        if best is None or rank > best["rank"]:
            best = {"name": m["name"], "label": m.get("label", m["name"]), "hits": hits,
                    "share": round(share, 3), "weight": float(m.get("weight", 1.0)),
                    "rank": rank, "safety": m.get("safety", "ok")}
    if best:
        best.pop("rank")
    return best


def _hit(key: str, msg: str) -> bool:
    if " " in key or not key.isalnum():   # phrases and emoji: substring match
        return key in msg
    return key in WORD_RE.findall(msg)


def moment_def(moments: list[dict], name: str | None) -> dict:
    return next((m for m in moments if m["name"] == name), {})


def pick_hook(moment: dict | None, moments: list[dict], fallback: str, streamer: str,
              game: str, seed: int | None = None) -> str:
    templates = moment_def(moments, (moment or {}).get("name")).get("hook_templates") or []
    if not templates:
        return fallback
    t = random.Random(seed).choice(templates)
    return t.format(streamer=streamer, game=game)


def clip_window_for(moment: dict | None, moments: list[dict], clip_cfg: dict) -> dict:
    """Clip settings with the moment's own pre/post-roll applied, if it has any."""
    m = moment_def(moments, (moment or {}).get("name"))
    return {**clip_cfg, **{k: m[k] for k in ("pre_roll_s", "post_roll_s") if k in m}}
