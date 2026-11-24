"""Stream loudness signal.

Each finished buffer segment is decoded once to 16 kHz mono PCM (shared with the
speech transcriber). Every second gets a loudness value: the loudest 250 ms RMS
inside it, so a short scream isn't averaged away. When a clip is crafted,
`anchor_on_peak` re-times it around the loudest 250 ms window. A BaselineTrigger learns the
streamer's normal loudness and fires when it jumps `min_ratio` x (3x amplitude is
about +10 dB: a scream, an explosion, a sudden "NO WAY").
"""
from __future__ import annotations

import array
import math
import subprocess
from pathlib import Path

from .signals import BaselineTrigger

SAMPLE_RATE = 16000


def decode_pcm(media: Path, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Mono 16-bit PCM of a media file's audio (empty if it has none)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(media), "-vn", "-ac", "1",
                        "-ar", str(sample_rate), "-f", "s16le", "-"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return r.stdout


def samples(pcm: bytes) -> array.array:
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    return a


def loudness_per_second(pcm: bytes, sample_rate: int = SAMPLE_RATE,
                        window_s: float = 0.25) -> list[float]:
    """One value (0..1) per full or partial second: the loudest window's RMS."""
    s = samples(pcm)
    win = max(1, int(sample_rate * window_s))
    out = []
    for sec in range(0, len(s), sample_rate):
        best = 0.0
        for w in range(sec, min(sec + sample_rate, len(s)), win):
            chunk = s[w:w + win]
            if chunk:
                best = max(best, math.sqrt(sum(x * x for x in chunk) / len(chunk)) / 32768)
        out.append(round(best, 5))
    return out


def window_levels(pcm: bytes, sample_rate: int = SAMPLE_RATE, window_s: float = 0.25) -> list[float]:
    """RMS (0..1) of each consecutive `window_s` window."""
    s = samples(pcm)
    win = max(1, int(sample_rate * window_s))
    return [math.sqrt(sum(x * x for x in s[w:w + win]) / len(s[w:w + win])) / 32768
            for w in range(0, len(s), win) if s[w:w + win]]


def anchor_on_peak(levels: list[float], window_s: float, lo: float, c: dict
                   ) -> tuple[float, float, float] | None:
    """(start, end, peak) in seconds for a clip anchored on the loudest window at or after
    `lo`: start `anchor.pre_s` before the peak, end `anchor.post_s` after the reaction fades
    (the last window still above median + fade_ratio x (peak - median) with no quiet gap
    longer than fade_gap_s), then fitted to min/max_length_s without moving the start."""
    a = {"pre_s": 4, "post_s": 7, "fade_ratio": 0.5, "fade_gap_s": 2.0, **(c.get("anchor") or {})}
    first = int(lo / window_s)
    if len(levels) - first < 2:
        return None
    peak_i = max(range(first, len(levels)), key=levels.__getitem__)
    med = sorted(levels)[len(levels) // 2]
    thr = med + float(a["fade_ratio"]) * (levels[peak_i] - med)
    fade_i, gap = peak_i, int(float(a["fade_gap_s"]) / window_s)
    for i in range(peak_i + 1, len(levels)):
        if levels[i] >= thr:
            fade_i = i
        elif i - fade_i > gap:
            break
    total = len(levels) * window_s
    peak = peak_i * window_s
    start = max(0.0, peak - float(a["pre_s"]))
    end = min(total, (fade_i + 1) * window_s + float(a["post_s"]))
    end = min(max(end, start + float(c.get("min_length_s", 15))),
              start + float(c.get("max_length_s", 30)), total)
    return round(start, 2), round(end, 2), round(peak, 2)


class AudioTracker:
    def __init__(self, cfg: dict, on_event=None):
        self.cfg = cfg
        self.trigger = BaselineTrigger(
            halflife_s=float(cfg.get("baseline_halflife_s", 300)),
            warmup_s=float(cfg.get("warmup_s", 90)),
            min_ratio=float(cfg.get("min_ratio", 3.0)),
            min_sustain_s=float(cfg.get("min_sustain_s", 1)),
            cooldown_s=float(cfg.get("cooldown_s", 30)),
            floor=float(cfg.get("silence_floor", 0.01)),
        )
        self.on_event = on_event
        self.last_ts: float | None = None   # newest second fed (how far audio is analysed)

    def feed_levels(self, start_ts: float, levels: list[float]) -> list[dict]:
        fired = []
        for i, v in enumerate(levels):
            ev = self.trigger.feed(start_ts + i, v)
            self.last_ts = start_ts + i
            if ev:
                fired.append(ev)
                if self.on_event:
                    self.on_event(ev)
        return fired

    def feed_pcm(self, start_ts: float, pcm: bytes) -> list[dict]:
        return self.feed_levels(start_ts, loudness_per_second(pcm))

    def ratio_between(self, start: float, end: float) -> float:
        return self.trigger.peak_ratio(start, end)

    @property
    def baseline(self) -> float:
        return self.trigger.baseline()
