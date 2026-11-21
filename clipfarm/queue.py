"""Unified, ranked clip queue backed by SQLite. Each clip keeps its receipt."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .models import Clip

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT NOT NULL,
    platform TEXT NOT NULL,
    clip_start_ts REAL NOT NULL,
    clip_end_ts REAL NOT NULL,
    score REAL NOT NULL,
    status TEXT NOT NULL,
    raw_path TEXT,
    reel_path TEXT,
    receipt TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_rank ON clips(status, score DESC);
"""


class ClipQueue:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def add(self, channel: str, platform: str, start: float, end: float, score: float,
            receipt: dict, raw_path: str | None = None, status: str = "recorded") -> int:
        cur = self.db.execute(
            "INSERT INTO clips (channel, platform, clip_start_ts, clip_end_ts, score, status,"
            " raw_path, reel_path, receipt, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (channel, platform, start, end, score, status, raw_path, None,
             json.dumps(receipt), time.time()))
        self.db.commit()
        return cur.lastrowid

    def update(self, clip_id: int, **fields) -> None:
        if "receipt" in fields:
            fields["receipt"] = json.dumps(fields["receipt"])
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE clips SET {cols} WHERE id=?", (*fields.values(), clip_id))
        self.db.commit()

    def get(self, clip_id: int) -> Clip | None:
        r = self.db.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
        return self._row(r) if r else None

    def ranked(self, status: str | None = None, limit: int = 50) -> list[Clip]:
        q, args = "SELECT * FROM clips", []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY score DESC LIMIT ?"
        args.append(limit)
        return [self._row(r) for r in self.db.execute(q, args)]

    @staticmethod
    def _row(r: sqlite3.Row) -> Clip:
        d = dict(r)
        d["receipt"] = json.loads(d["receipt"])
        return Clip(**d)
