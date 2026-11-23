"""ClipAgent: turn a scored clip window into a ranked, explained clip candidate.

The moment score (signals.py) says HOW STRONG a moment is. ClipAgent adds the checks
from the ClipAgent spec on top of it:

  - keywords are weighted evidence, never a trigger on their own: a clip needs at
    least `min_signals` independent signals AND one of them must be activity
    (chat speeding up or the stream getting loud)
  - six signal checks: chat keywords/emotes, chat velocity spike, chat intensity
    (!/? density or an emote wall), audio spike, spoken trigger phrase, and a
    context event (game-profile moment, donation read, streamer reading chat, rage quit)
  - clip_type: hype | funny | clutch | fail | rage | donation | chat_interaction | other,
    voted on by the keyword buckets, phrases, context and the game moment
  - confidence 0..1 = corroboration (how many checks agree) and intensity (the moment
    score), minus a penalty when signals disagree; only clips at or above
    `min_confidence` (0.65) are recommended
  - a JSON candidate in the spec's format, saved next to each clip

Keyword lists live in settings.yaml `clip_agent` and can be replaced per game profile
or channel, or extended with `extra_chat_keywords` / `extra_speech_phrases`.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from typing import Iterable

CLIP_TYPES = ("hype", "funny", "clutch", "fail", "rage", "donation", "chat_interaction", "other")

# bucket name -> clip-type votes. A custom bucket named after a clip type votes for it.
BUCKET_VOTES: dict[str, dict[str, float]] = {
    # chat
    "hype": {"hype": 1.0},
    "funny": {"funny": 1.0},
    "tense": {"clutch": 0.5},
    "outcome_win": {"clutch": 1.0},
    "outcome_loss": {"fail": 1.0},
    "clip_command": {},
    # speech
    "shock": {"hype": 0.4},
    "celebration": {"hype": 0.5, "clutch": 0.3},
    "frustration": {"rage": 0.8, "fail": 0.3},
    # context (chat or speech)
    "donation": {"donation": 1.5},
    "chat_interaction": {"chat_interaction": 1.5},
    "rage_quit": {"rage": 1.5},
}
CONTEXT_TYPES = {"donation", "chat_interaction", "rage_quit"}
# types that can share a moment without contradicting each other
COMPATIBLE = ({"hype", "clutch"}, {"hype", "funny"}, {"rage", "fail"}, {"funny", "fail"})
MOMENT_VOTE = 0.6      # a game-profile moment is a hint about the scene, not chat's emotion

STRETCH_RE = re.compile(r"(.)\1{2,}")
REPEAT_RE = re.compile(r"(.)\1+")
SPOKEN_RE = re.compile(r"[a-z0-9']+")


def _norm(text: str) -> str:
    """Lower-case and collapse stretched letters: "LETSGOOOO" -> "letsgoo"."""
    return STRETCH_RE.sub(r"\1\1", text.lower())


def _tokens(msg: str) -> list[str]:
    # keep a leading "!" (commands like !clip), strip other punctuation around words
    out = []
    for t in msg.split():
        core = t.strip(".,;:\"'()[]")
        cmd = core.startswith("!")
        core = core.strip("!?")
        if core:
            out.append(("!" + core) if cmd else core)
    return out


def _single(text: str) -> str:
    return REPEAT_RE.sub(r"\1", text)


def _hit(key: str, norm_msg: str, toks: set[str]) -> bool:
    if " " in key:                                  # phrase: substring match
        return key in norm_msg or _single(key) in _single(norm_msg)
    if key in toks:
        return True
    # stretched emote: "LETSGOOOO" is "letsgo", but "g" is not "GG"
    one = _single(key)
    return any(len(t) >= len(key) and _single(t) == one for t in toks)


def merged_buckets(cfg: dict, key: str) -> dict[str, dict[str, str]]:
    """A bucket set with its `extra_*` additions: {bucket: {normalised: as written}}."""
    base = {k: list(v or []) for k, v in (cfg.get(key) or {}).items()}
    for k, v in (cfg.get(f"extra_{key}") or {}).items():
        base.setdefault(k, []).extend(v or [])
    return {k: {_norm(str(w)): str(w) for w in v} for k, v in base.items() if v}


# ---------------------------------------------------------------------- chat
def chat_evidence(messages: Iterable[str], cfg: dict) -> dict:
    """Keyword/emote hits per bucket plus !/? density and emote-wall share, over ALL of
    a spike's messages (run by the detector when the spike closes)."""
    msgs = [m for m in messages if m]
    n = len(msgs)
    buckets = {**merged_buckets(cfg, "chat_keywords"), **{
        k: v for k, v in merged_buckets(cfg, "context_phrases").items()}}
    hits: dict[str, dict] = {}
    per_tok: Counter = Counter()
    punct = 0
    for raw in msgs:
        norm = _norm(raw)
        toks = [_norm(t) for t in _tokens(raw)]
        tset = set(toks)
        per_tok.update(tset)                       # one count per message, not per repeat
        if re.search(r"[!?]", re.sub(r"(^|\s)!\w", " ", raw)):   # "!clip" isn't excitement
            punct += 1
        for name, keys in buckets.items():
            found = [orig for k, orig in keys.items() if _hit(k, norm, tset)]
            if found:
                b = hits.setdefault(name, {"msgs": 0, "words": Counter()})
                b["msgs"] += 1
                b["words"].update(found)
    top = per_tok.most_common(1)
    return {
        "messages": n,
        "buckets": {k: {"msgs": v["msgs"], "share": round(v["msgs"] / max(n, 1), 3),
                        "words": [w for w, _ in v["words"].most_common(4)]}
                    for k, v in hits.items()},
        "punct_share": round(punct / max(n, 1), 3),
        "emote_wall": {"token": top[0][0], "share": round(top[0][1] / max(n, 1), 3)} if top
        else {"token": "", "share": 0.0},
    }


# ---------------------------------------------------------------------- speech
def speech_evidence(transcript, cfg: dict) -> dict[str, list[str]]:
    """Spoken trigger phrases per bucket, from the clip's transcript words."""
    words = [str(w[0]) for w in (transcript or [])]
    text = " " + " ".join(SPOKEN_RE.findall(_single(" ".join(words).lower()))) + " "
    out: dict[str, list[str]] = {}
    buckets = {**merged_buckets(cfg, "speech_phrases"), **merged_buckets(cfg, "context_phrases")}
    for name, phrases in buckets.items():
        found, keys = [], set()
        for p, orig in phrases.items():
            key = " ".join(SPOKEN_RE.findall(_single(p)))
            if key and f" {key} " in text and key not in keys:
                keys.add(key)
                found.append(orig.lower())
        if found:
            out[name] = found
    return out


# ---------------------------------------------------------------------- verdict
def _clock(ts: float, origin: float | None) -> str:
    if origin is not None:
        s = max(0, int(round(ts - origin)))
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return time.strftime("%H:%M:%S", time.localtime(ts))


def evaluate(receipt: dict, cfg: dict, score: float, start: float, end: float,
             origin_ts: float | None = None) -> dict:
    """The spec's candidate JSON (plus a few extra fields) for one scored clip window.

    `cfg` is the channel config; settings come from cfg["clip_agent"]. Timestamps are
    HH:MM:SS since `origin_ts` when given (demo: stream start), else local clock time."""
    a = cfg.get("clip_agent") or {}
    trig = float(cfg.get("signals", {}).get("trigger_ratio", 3.0))
    kw_share = float(a.get("chat_keyword_share", 0.10))
    ce = receipt.get("chat_evidence") or {}
    se = speech_evidence(receipt.get("transcript"), a)
    moment = receipt.get("moment") or {}

    # chat keyword buckets that count (a single Pog in a big spike doesn't)
    chat_b = {k: v for k, v in (ce.get("buckets") or {}).items()
              if v["share"] >= kw_share or (k == "clip_command" and v["msgs"] >= 3)
              or (k in CONTEXT_TYPES and (v["msgs"] >= 3 or v["share"] >= kw_share / 2))}
    speech_b = {k: v for k, v in se.items() if k not in CONTEXT_TYPES}
    context_b = sorted({k for k in chat_b if k in CONTEXT_TYPES} |
                       {k for k in se if k in CONTEXT_TYPES})
    chat_ratio = max(float(receipt.get("chat_ratio") or 1.0), float(receipt.get("ratio") or 1.0))
    audio_ratio = float(receipt.get("audio_ratio") or 1.0)
    spike = receipt.get("peak_rate") is not None
    wall = ce.get("emote_wall") or {}

    checks = {
        "chat_keywords": any(k not in CONTEXT_TYPES for k in chat_b),
        "chat_velocity": spike or chat_ratio >= trig,
        "chat_intensity": ce.get("messages", 0) >= 5 and (
            ce.get("punct_share", 0) >= float(a.get("punct_share", 0.30))
            or wall.get("share", 0) >= float(a.get("emote_wall_share", 0.30))),
        "audio_spike": audio_ratio >= trig,
        "transcript_trigger": bool(speech_b),
        "context_event": bool(moment) or bool(context_b),
    }
    present = [k for k, v in checks.items() if v]
    activity = checks["chat_velocity"] or checks["audio_spike"]

    # clip type votes
    votes: Counter = Counter()
    for name, b in chat_b.items():
        for t, w in _votes(name).items():
            votes[t] += w * min(1.5, b["share"] / 0.2) if name not in CONTEXT_TYPES else w
    for name, found in se.items():
        for t, w in _votes(name).items():
            votes[t] += w * min(1.0, len(found) / 2) if name not in CONTEXT_TYPES else w
    if moment.get("clip_type") in CLIP_TYPES:
        votes[moment["clip_type"]] += MOMENT_VOTE * float(moment.get("weight", 1.0))
    ranked = [(t, round(v, 3)) for t, v in votes.most_common() if v > 0]
    clip_type = ranked[0][0] if ranked and ranked[0][1] >= 0.2 else "other"

    conflicts = []
    if len(ranked) >= 2 and ranked[1][1] >= 0.75 * ranked[0][1] and \
            {ranked[0][0], ranked[1][0]} not in COMPATIBLE:
        conflicts.append(f"mixed signals: {ranked[0][0]} vs {ranked[1][0]}")
    if checks["audio_spike"] and not spike and chat_ratio < 1.2:
        conflicts.append(f"loud ({audio_ratio}x) but chat stayed at {chat_ratio}x normal")

    corroboration = min(1.0, len(present) / float(a.get("full_signals", 4)))
    intensity = min(1.0, max(0.0, score) / float(a.get("intensity_full_score", 50)))
    conf = 0.55 * corroboration + 0.45 * intensity
    conf -= min(0.2, 0.1 * len(conflicts))
    floor = float(a.get("min_confidence", 0.65))
    blocked = None
    if not activity:
        blocked = "no chat or audio spike: keywords alone never make a clip"
    elif len(present) < int(a.get("min_signals", 2)):
        blocked = f"only one signal ({present[0] if present else 'none'})"
    if blocked:
        conf = min(conf, floor - 0.25)
    conf = round(max(0.0, min(1.0, conf)), 2)

    keywords = []
    for b in chat_b.values():
        keywords += [w for w in b["words"] if w not in keywords]
    transcript_trigger = next((f[0] for f in speech_b.values()), "") or \
        next((p for p in (receipt.get("speech") or {}).get("phrases") or []), "")
    note = []
    if moment:
        note.append(f"game moment: {moment.get('label', moment.get('name'))}")
    note += [f"{c.replace('_', ' ')} detected" for c in context_b]
    if checks["chat_intensity"]:
        note.append(f"{int(100 * ce.get('punct_share', 0))}% of chat used !/?, "
                    f"'{wall.get('token', '')}' in {int(100 * wall.get('share', 0))}% of messages")
    return {
        "timestamp_start": _clock(start, origin_ts),
        "timestamp_end": _clock(end, origin_ts),
        "clip_type": clip_type,
        "confidence_score": conf,
        "evidence": {
            "keywords_detected": keywords[:8],
            "chat_velocity_spike": checks["chat_velocity"],
            "audio_spike": checks["audio_spike"],
            "transcript_trigger": transcript_trigger,
            "context_note": "; ".join(note),
        },
        "reason": _reason(clip_type, present, blocked, conflicts, chat_ratio, audio_ratio,
                          transcript_trigger, keywords),
        # extras beyond the spec, for tuning
        "recommended": conf >= floor,
        "signals_present": present,
        "type_votes": dict(ranked),
        "conflicts": conflicts,
        "start_ts": start,
        "end_ts": end,
    }


def _votes(name: str) -> dict[str, float]:
    if name in BUCKET_VOTES:
        return BUCKET_VOTES[name]
    return {name: 1.0} if name in CLIP_TYPES else {}


def _reason(clip_type, present, blocked, conflicts, chat_x, audio_x, said, keywords) -> str:
    if blocked:
        return f"Not clip-worthy: {blocked}."
    bits = []
    if "chat_velocity" in present:
        bits.append(f"chat {chat_x:g}x normal")
    if "audio_spike" in present:
        bits.append(f"audio {audio_x:g}x normal")
    if said:
        bits.append(f'streamer said "{said}"')
    if keywords:
        bits.append("chat spammed " + ", ".join(keywords[:3]))
    text = f"{clip_type.replace('_', ' ').capitalize()} moment: " + ", ".join(bits) + \
        f" ({len(present)} signals agree)."
    if conflicts:
        text += " Lower confidence: " + "; ".join(conflicts) + "."
    return text


def below_floor(receipt: dict, cfg: dict) -> str | None:
    """Why a clip is dropped by the confidence floor, or None to keep it."""
    a = cfg.get("clip_agent") or {}
    c = receipt.get("clip_agent")
    if not c or not a.get("enabled", True) or not a.get("gate", True) or c["recommended"]:
        return None
    return f"confidence {c['confidence_score']} < {a.get('min_confidence', 0.65)}: {c['reason']}"


SPEC_KEYS = ("timestamp_start", "timestamp_end", "clip_type", "confidence_score",
             "evidence", "reason")


def candidate_json(c: dict) -> dict:
    """Only the spec's fields, in the spec's order."""
    return {k: c[k] for k in SPEC_KEYS}
