"""Multi-signal moment scoring.

Every signal is measured against the streamer's OWN normal level (an EWMA
baseline), so a loud streamer and a quiet one are judged fairly. A signal's
strength is mapped to 0..1 with `strength()`: 0 at or below normal, 0.5 at the
trigger ratio (3x by default), 1.0 at twice that distance. The final score is
the weighted sum of strengths on a 0-100 scale.

Signals (weights in settings.yaml `signals.weights`, they must add up to 1.0):
  chat_speed   chat messages/s vs normal (detector.py)
  chat_words   share of hype words in chat (KEKW, LUL, 💀...)
  audio        stream loudness vs normal (audio.py) - screaming, explosions
  speech       what the streamer says: pace vs normal + excited phrases (speech.py)
  game_events  game-profile moment match (moments.py)
  twitch_clips viewers clipping the moment on Twitch (not wired up yet, scores 0)
"""
from __future__ import annotations

import math
from collections import deque

SIGNALS = ("chat_speed", "chat_words", "audio", "speech", "game_events", "twitch_clips")


def strength(ratio: float, trigger: float = 3.0) -> float:
    """Ratio-to-normal -> 0..1. 1x = 0, trigger = 0.5, 2*trigger-1 = 1."""
    if ratio <= 1 or trigger <= 1:
        return 0.0
    return round(min(1.0, (ratio - 1) / (2 * (trigger - 1))), 3)


def check_weights(weights: dict) -> dict:
    unknown = set(weights) - set(SIGNALS)
    if unknown:
        raise ValueError(f"unknown signal weights: {sorted(unknown)} (known: {SIGNALS})")
    total = sum(float(v) for v in weights.values())
    if not math.isclose(total, 1.0, abs_tol=0.005):
        raise ValueError(f"signals.weights must add up to 1.0, they add up to {total:.3f}")
    return {k: float(weights.get(k, 0.0)) for k in SIGNALS}


def score_signals(values: dict, weights: dict) -> tuple[float, dict]:
    """Weighted score 0-100 and the per-signal points (for the receipt)."""
    w = check_weights(weights)
    points = {k: round(100 * w[k] * max(0.0, min(1.0, float(values.get(k, 0.0)))), 2)
              for k in SIGNALS}
    return round(sum(points.values()), 2), points


class BaselineTrigger:
    """Per-second level vs an EWMA baseline; opens an event when the level stays
    `min_ratio` x normal for `min_sustain_s`. The baseline is frozen during an event
    so the hype doesn't teach it that hype is normal (same idea as the chat detector)."""

    def __init__(self, halflife_s: float = 300, warmup_s: float = 90, min_ratio: float = 3.0,
                 min_sustain_s: float = 1, cooldown_s: float = 30, floor: float = 0.0,
                 history_s: float = 3600):
        self.alpha = 1 - 0.5 ** (1.0 / halflife_s)
        self.warmup = warmup_s
        self.min_ratio = min_ratio
        self.sustain = max(1, round(min_sustain_s))
        self.cooldown = cooldown_s
        self.floor = floor                  # baseline never counts as lower than this
        self.mean: float | None = None
        self.n = 0
        self.t0: float | None = None
        self.open: dict | None = None
        self.hot = 0
        self.last_end = -math.inf
        self.history: deque = deque(maxlen=int(history_s))   # (ts, value, baseline)
        self.events: list[dict] = []

    def baseline(self) -> float:
        return max(self.mean or 0.0, self.floor)

    def feed(self, ts: float, value: float) -> dict | None:
        """Add one second. Returns an event when one closes."""
        if self.t0 is None:
            self.t0 = ts
        if self.mean is None:
            self.mean = value
            self.n = 1
            self.history.append((ts, value, self.baseline()))
            return None
        base = self.baseline()
        self.history.append((ts, value, base))
        ratio = value / base if base > 0 else 0.0
        warm = ts - self.t0 >= self.warmup
        hot = warm and ratio >= self.min_ratio

        if self.open is not None:
            o = self.open
            if ratio > o["ratio"]:
                o.update(ratio=round(ratio, 2), peak_ts=ts, peak=value)
            if ratio < self.min_ratio / 2:      # closes once it's back under half the trigger
                o["end_ts"] = ts
                self.open, self.last_end = None, ts
                self.events.append(o)
                return o
            return None

        if hot and ts - self.last_end >= self.cooldown:
            self.hot += 1
            if self.hot == 1:
                self._cand = dict(start_ts=ts, peak_ts=ts, peak=value, ratio=round(ratio, 2),
                                  baseline=round(base, 4))
            elif ratio > self._cand["ratio"]:
                self._cand.update(ratio=round(ratio, 2), peak_ts=ts, peak=value)
            if self.hot >= self.sustain:
                self.open, self.hot = self._cand, 0
            return None
        self.hot = 0

        self.n += 1
        a = max(self.alpha, 1.0 / self.n)      # plain average early, EWMA later
        self.mean += a * (value - self.mean)
        return None

    def peak_ratio(self, start: float, end: float) -> float:
        """Highest level / baseline inside [start, end] (1.0 if no data)."""
        vals = [v / b for t, v, b in self.history if start <= t <= end and b > 0]
        return round(max(vals), 2) if vals else 1.0
