"""Chat-velocity spike detector.

Counts messages per bucket, smooths them, and keeps an exponentially weighted
baseline (mean + variance) of each channel's normal pace. A spike fires when
the smoothed rate is both `z_threshold` std-devs above the baseline and
`min_ratio` times the baseline. The baseline is frozen while a spike is open
so the hype doesn't teach the detector that hype is normal.
"""
from __future__ import annotations

import math
import re
from collections import Counter, deque
from typing import Callable, Iterable

from .models import ChatMessage, Spike

TOKEN_RE = re.compile(r"\S+")


class SpikeDetector:
    def __init__(self, channel: str, cfg: dict, hype_tokens: Iterable[str] = (),
                 on_spike: Callable[[Spike], None] | None = None,
                 classifier: Callable[[list[str]], dict | None] | None = None):
        d = cfg
        self.channel = channel
        self.bucket = float(d["bucket_seconds"])
        self.alpha = 1 - 0.5 ** (self.bucket / float(d["baseline_halflife_s"]))
        self.warmup = float(d["warmup_s"])
        self.z_th = float(d["z_threshold"])
        self.ratio_th = float(d["min_ratio"])
        self.min_rate = float(d["min_msgs_per_s"])
        self.cooldown = float(d["cooldown_s"])
        self.smooth_n = int(d.get("smoothing_buckets", 1))
        self.sustain_n = max(1, round(float(d.get("min_sustain_s", 3)) / self.bucket))
        self.hype = {t.lower() for t in hype_tokens}
        self.on_spike = on_spike
        self.classifier = classifier

        self.mean: float | None = None
        self.var = 0.0
        self.t0: float | None = None
        self.cur_bucket: int | None = None
        self.cur_count = 0
        self.cur_msgs: list[str] = []
        self.recent = deque(maxlen=self.smooth_n)
        self.history = deque(maxlen=int(3600 / self.bucket))  # (ts, rate) for receipts/scoring
        self.open: dict | None = None
        self.candidate: dict | None = None
        self.last_end = -math.inf
        self.spikes: list[Spike] = []

    # -- feeding -----------------------------------------------------------
    def feed(self, msg: ChatMessage) -> list[Spike]:
        """Add a message. Returns spikes that closed as a result (usually none)."""
        b = int(msg.ts // self.bucket)
        fired = self._advance_to(b)
        self.cur_count += 1
        self.cur_msgs.append(msg.text)
        return fired

    def tick(self, now: float) -> list[Spike]:
        """Call periodically so silence also closes buckets."""
        return self._advance_to(int(now // self.bucket))

    def _advance_to(self, b: int) -> list[Spike]:
        fired: list[Spike] = []
        if self.cur_bucket is None:
            self.cur_bucket = b
            self.t0 = b * self.bucket
            return fired
        while self.cur_bucket < b:
            s = self._close_bucket(self.cur_bucket, self.cur_count, self.cur_msgs)
            if s:
                fired.append(s)
            self.cur_bucket += 1
            self.cur_count, self.cur_msgs = 0, []
        return fired

    # -- core --------------------------------------------------------------
    def _close_bucket(self, b: int, count: int, msgs: list[str]) -> Spike | None:
        ts = b * self.bucket
        rate = count / self.bucket
        self.recent.append(rate)
        smooth = sum(self.recent) / len(self.recent)
        self.history.append((ts, smooth))

        if self.mean is None:
            self.mean = smooth
            self.var = max(smooth, 1.0) / (self.bucket * self.smooth_n)  # Poisson guess
            return None

        # chat is bursty count data: never trust a std below the Poisson noise floor
        poisson_std = math.sqrt(max(self.mean, 0.5) / (self.bucket * self.smooth_n))
        std = max(math.sqrt(max(self.var, 0.0)), poisson_std)
        z = (smooth - self.mean) / std
        ratio = smooth / max(self.mean, 0.1)
        warm = ts - self.t0 >= self.warmup
        hot = warm and z >= self.z_th and ratio >= self.ratio_th and smooth >= self.min_rate

        if self.open is not None:
            o = self.open
            o["msgs"].extend(msgs)
            if smooth > o["peak_rate"]:
                o.update(peak_rate=smooth, peak_ts=ts, z=z, ratio=ratio)
            # close when the rate drops back under half the way to threshold
            if not (z >= self.z_th / 2 and smooth >= self.min_rate):
                return self._close_spike(ts)
            return None

        if hot and ts - self.last_end >= self.cooldown:
            c = self.candidate
            if c is None:
                c = self.candidate = dict(start_ts=ts, peak_ts=ts, peak_rate=smooth, z=z,
                                          ratio=ratio, base=self.mean, std=std, msgs=[], n=0)
            c["n"] += 1
            c["msgs"].extend(msgs)
            if smooth > c["peak_rate"]:
                c.update(peak_rate=smooth, peak_ts=ts, z=z, ratio=ratio)
            if c["n"] >= self.sustain_n:   # must stay hot a few seconds, not a one-off blip
                self.open, self.candidate = c, None
            return None
        self.candidate = None

        # only learn the baseline from non-spike buckets; early on use a plain running
        # average (alpha = 1/n) so one odd first bucket can't skew the baseline for minutes
        self.n_learned = getattr(self, "n_learned", 1) + 1
        a = max(self.alpha, 1.0 / self.n_learned)
        diff = smooth - self.mean
        self.mean += a * diff
        self.var = (1 - a) * (self.var + a * diff * diff)
        return None

    def _close_spike(self, end_ts: float) -> Spike:
        o, self.open = self.open, None
        self.last_end = end_ts
        tokens = Counter(t for m in o["msgs"] for t in TOKEN_RE.findall(m))
        hype_hits = sum(1 for m in o["msgs"]
                        if any(t.lower() in self.hype for t in TOKEN_RE.findall(m)))
        curve = [r for (t, r) in self.history if o["start_ts"] - 30 <= t <= end_ts]
        spike = Spike(
            channel=self.channel, start_ts=o["start_ts"], peak_ts=o["peak_ts"], end_ts=end_ts,
            peak_rate=round(o["peak_rate"], 2), baseline_rate=round(o["base"], 2),
            baseline_std=round(o["std"], 2), z_score=round(o["z"], 2), ratio=round(o["ratio"], 2),
            hype_share=round(hype_hits / max(len(o["msgs"]), 1), 3),
            top_tokens=tokens.most_common(8), sample_messages=o["msgs"][:10],
            rate_curve=[round(r, 2) for r in curve],
            moment=self.classifier(o["msgs"]) if self.classifier else None,
        )
        self.spikes.append(spike)
        if self.on_spike:
            self.on_spike(spike)
        return spike


    def ratio_between(self, start: float, end: float) -> float:
        """Peak chat rate inside [start, end] vs the learned normal pace (1.0 if unknown)."""
        rates = [r for t, r in self.history if start <= t <= end]
        if not rates or not self.mean:
            return 1.0
        return round(max(rates) / max(self.mean, 0.1), 2)


def chat_values(spike: Spike | None, chat_ratio: float, trigger: float = 3.0) -> dict:
    """chat_speed / chat_words / game_events strengths (0..1) for signals.score_signals."""
    from .signals import strength
    m = (spike.moment if spike else None) or {}
    return {
        "chat_speed": strength(spike.ratio if spike else chat_ratio, trigger),
        # half the spike's messages being hype words (KEKW, LUL, 💀...) = full points
        "chat_words": round(min(1.0, 2 * spike.hype_share), 3) if spike else 0.0,
        # game-profile moment: matching share x weight of 0.4 or more = full points
        "game_events": round(min(1.0, m.get("share", 0.0) * m.get("weight", 1.0) / 0.4), 3),
    }


def score_spike(spike: Spike, weights: dict, trigger: float = 3.0) -> float:
    """Chat-only score (0-100) for a spike, e.g. in `watch` mode where there is no audio."""
    from .signals import score_signals
    return score_signals(chat_values(spike, spike.ratio, trigger), weights)[0]


class LagFilter:
    """Spots chat that arrives late. When the chat connection stalls, Twitch holds the
    messages and then delivers the whole backlog at once; counted on arrival that burst
    looks like a huge spike. Each message's own send time (`sent_ts`) shows it is late.
    The usual delay (network + our clock vs Twitch's) is learned as the median lag, and a
    message more than `max_lag_s` behind that is late: its bucket is already closed, so
    it is left out of the pace instead of piling into the current second."""

    def __init__(self, max_lag_s: float = 5.0, keep: int = 300):
        self.max_lag = float(max_lag_s)
        self.lags: deque[float] = deque(maxlen=keep)

    def is_late(self, msg: ChatMessage, now: float) -> bool:
        if msg.sent_ts is None:
            return False
        lag = now - msg.sent_ts
        typical = sorted(self.lags)[len(self.lags) // 2] if self.lags else lag
        late = lag > typical + self.max_lag
        if not late:
            self.lags.append(lag)
        return late
