"""Smart scheduling: only monitor a channel inside its configured windows."""
from __future__ import annotations

from datetime import datetime

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def in_window(schedule: list[dict], now: datetime | None = None) -> bool:
    if not schedule:
        return True
    now = now or datetime.now()
    day, hm = DAYS[now.weekday()], now.strftime("%H:%M")
    for w in schedule:
        days = w.get("days") or DAYS
        start, end = w["start"], w["end"]
        if start <= end:
            if day in days and start <= hm < end:
                return True
        else:  # window crosses midnight
            prev = DAYS[(now.weekday() - 1) % 7]
            if (day in days and hm >= start) or (prev in days and hm < end):
                return True
    return False
