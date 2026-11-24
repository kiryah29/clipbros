"""Every streamer the app has clipped, saved in data/streamers.json.

A streamer is saved the first time a run starts for them (control page or `clipfarm run`),
and updated on later runs: first/last seen, number of runs, last game profile. Control-page
run folders are deleted after a few days; this list is kept, and its Twitch names show up
as suggestions in the control page's channel box.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

FILE = "streamers.json"


class StreamerRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    @staticmethod
    def key(name: str, platform: str) -> str:
        return f"{platform.lower()}:{name.lower().lstrip('@')}"

    def _load(self) -> dict[str, dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)   # a crash mid-write never leaves half a file

    def record(self, name: str, platform: str, game: str | None = None, source: str = "",
               now: float | None = None) -> dict:
        """Save the streamer (or update them if already saved). Returns their entry."""
        now = now or time.time()
        k = self.key(name, platform)
        with self._lock:
            data = self._load()
            s = data.get(k) or {"name": name.lower().lstrip("@"), "platform": platform.lower(),
                                "first_seen": now, "runs": 0}
            s["last_seen"] = now
            s["runs"] = int(s.get("runs", 0)) + 1
            if game:
                s["game"] = game
            if source:
                s["source"] = source
            data[k] = s
            self._write(data)
        return s

    def all(self) -> list[dict]:
        """Saved streamers, most recently seen first."""
        return sorted(self._load().values(), key=lambda s: s.get("last_seen", 0), reverse=True)

    def names(self, platform: str) -> list[str]:
        return [s["name"] for s in self.all() if s.get("platform") == platform.lower()]
