from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ChatMessage:
    channel: str
    ts: float          # unix seconds
    user: str
    text: str
    sent_ts: float | None = None   # platform's own send time (Twitch tmi-sent-ts), if known


@dataclass
class Spike:
    """A detected chat-velocity peak. Becomes the clip's 'receipt'."""
    channel: str
    start_ts: float            # first bucket above threshold
    peak_ts: float             # bucket with the highest rate
    end_ts: float              # first bucket back below threshold
    peak_rate: float           # msgs/s at peak (smoothed)
    baseline_rate: float       # learned normal pace at trigger time
    baseline_std: float
    z_score: float
    ratio: float               # peak_rate / baseline_rate
    hype_share: float = 0.0    # share of spike messages containing hype tokens
    top_tokens: list[tuple[str, int]] = field(default_factory=list)
    sample_messages: list[str] = field(default_factory=list)
    rate_curve: list[float] = field(default_factory=list)  # msgs/s around the spike
    moment: dict[str, Any] | None = None  # game-profile label, see moments.py
    chat_evidence: dict[str, Any] | None = None  # keyword buckets, !/? density (agent.py)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Clip:
    id: int | None
    channel: str
    platform: str
    clip_start_ts: float
    clip_end_ts: float
    score: float
    status: str                # recorded | crafted | approved | rejected | failed
    raw_path: str | None
    reel_path: str | None
    receipt: dict[str, Any]
    created_at: float
