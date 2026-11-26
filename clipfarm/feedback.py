"""Review feedback: every approve/reject, why, and a frozen copy of the clip's features.

All reviews go into ONE central SQLite file, data/feedback.db, next to the control run
folders but outside them, so the 3-day run cleanup never deletes them. Rows are only
appended: changing your mind adds a new row, and the newest row per clip and reviewer
counts. This is Phase 0 of the learning loop (docs: "ClipBros learning loop").
"""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

FILE = "feedback.db"
FEATURES_VERSION = 1

# code, button label, pipeline stage. The stage decides what a reject teaches later:
# only "selection" rejects count against the moment itself.
REASONS: list[tuple[str, str, str]] = [
    ("boring", "Boring, nothing happens", "selection"),
    ("needs_context", "Needs context", "selection"),
    ("late_start", "Starts too late", "window"),
    ("too_long", "Too long", "window"),
    ("bad_hook", "Bad hook", "hook"),
    ("bad_captions", "Bad captions / wrong speaker", "captions"),
    ("duplicate", "Duplicate", "dedupe"),
    ("unsafe", "Unsafe / not allowed", "safety"),
]
REASON_CODES = {code for code, _, _ in REASONS}
STAGE = {code: stage for code, _, stage in REASONS}
MAX_NOTE = 500

# queue status after a review. The old code did action + "d", which gave "rejectd"; those
# clips are still read as rejected.
STATUS_OF = {"approve": "approved", "reject": "rejected"}
DECISION_OF = {"approved": "approve", "rejected": "reject", "rejectd": "reject"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS clip_feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_uid      TEXT NOT NULL,
    channel       TEXT,
    game          TEXT,
    run_id        TEXT,
    clip_id       INTEGER,
    decision      TEXT NOT NULL CHECK (decision IN ('approve', 'reject')),
    star          INTEGER NOT NULL DEFAULT 0,
    reasons       TEXT NOT NULL DEFAULT '[]',
    note          TEXT,
    reviewer      TEXT,
    reviewed_at   REAL NOT NULL,
    watch_ms      INTEGER,
    blind         INTEGER,
    was_explore   INTEGER NOT NULL DEFAULT 0,
    rank_position INTEGER,
    score         REAL,
    confidence    REAL,
    p_approve     REAL,
    model_version TEXT,
    features      TEXT NOT NULL,
    receipt_path  TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_clip ON clip_feedback(clip_uid, reviewed_at);
CREATE INDEX IF NOT EXISTS idx_feedback_time ON clip_feedback(reviewed_at);
"""


def clip_uid(channel: str, run_id: str | None, clip_id: int) -> str:
    """Unique across the main queue and every control run: channel/run/clip."""
    return f"{channel}/{run_id or 'main'}/{clip_id}"


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _count(v: Any) -> int | None:
    if isinstance(v, (list, tuple, dict)):
        return len(v)
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else None


def features(receipt: Mapping[str, Any], score: float | None = None,
             start: float | None = None, end: float | None = None) -> dict:
    """Snapshot of what the clip looked like when it was reviewed.

    `selection` holds only what is known when the clip is cut (safe to learn the gate
    from). `render` holds what the reel craft added (hook, captions, layout), used only
    for generation feedback. Missing receipt keys become None, never errors."""
    r = receipt or {}
    sig = r.get("signals") or {}
    ca = r.get("clip_agent") or {}
    sp = r.get("speech") or {}
    moment = r.get("moment")
    anchor = r.get("anchor") or {}
    nud = r.get("nudity") or {}
    start_ts = _num(r.get("start_ts"))
    peak_ts = _num(r.get("peak_ts"))
    selection = {
        "score": _num(score),
        **{f"sig_{k}": _num(v) for k, v in sig.items() if _num(v) is not None},
        "chat_ratio": _num(r.get("chat_ratio")),
        "audio_ratio": _num(r.get("audio_ratio")),
        "spike_ratio": _num(r.get("ratio")),
        "z_score": _num(r.get("z_score")),
        "hype_share": _num(r.get("hype_share")),
        "peak_rate": _num(r.get("peak_rate")),
        "baseline_rate": _num(r.get("baseline_rate")),
        "pace_ratio": _num(sp.get("pace_ratio")),
        "n_phrases": _count(sp.get("phrases")) or 0,
        "transcript_words": _count(r.get("transcript")) or 0,
        "chat_msgs": _count(r.get("chat")) or 0,
        "triggers": "+".join(sorted(set(r.get("triggers") or []))) or None,
        "clip_type": ca.get("clip_type"),
        "confidence": _num(ca.get("confidence_score")),
        "recommended": ca.get("recommended"),
        "n_checks": _count(ca.get("checks_passed", ca.get("checks"))),
        "n_conflicts": _count(ca.get("conflicts")),
        "moment": moment.get("label") if isinstance(moment, dict) else moment,
        "safety_review": bool(r.get("safety_review")),
        "length_s": round(end - start, 2) if start is not None and end is not None else None,
        "peak_offset_s": round(peak_ts - start, 2) if peak_ts is not None and start is not None
        else (round(peak_ts - start_ts, 2) if peak_ts is not None and start_ts is not None else None),
    }
    render = {
        "hook": r.get("hook"),
        "hook_kind": r.get("hook_kind"),
        "hook_style": r.get("hook_style"),
        "hook_writer": r.get("hook_writer") if isinstance(r.get("hook_writer"), str) else None,
        "caption_source": r.get("caption_source"),
        "facecam_source": r.get("facecam_source"),
        "nudity": nud.get("decision"),
        "reel_length_s": _num(anchor.get("length_s")),
        "peak_in_reel_s": _num(anchor.get("peak_s")),
    }
    return {"v": FEATURES_VERSION, "selection": selection, "render": render}


def parse_review(action: str, data: Any) -> dict:
    """A review from a form post (MultiDict) or a JSON body (dict). Raises ValueError."""
    if action not in ("approve", "reject"):
        raise ValueError("Action must be approve or reject.")

    def get(key: str, default: Any = None) -> Any:
        v = data.get(key, default) if data is not None else default
        return v if v not in ("", None) else default

    if hasattr(data, "getall"):
        reasons = [str(x) for x in data.getall("reason", [])]
    else:
        reasons = [str(x) for x in (get("reasons") or [])]
    reasons = list(dict.fromkeys(reasons))  # keep order, drop repeats
    unknown = [x for x in reasons if x not in REASON_CODES]
    if unknown:
        raise ValueError(f"Unknown reason: {', '.join(unknown)}")
    if action == "approve":
        reasons = []  # reasons only explain rejects

    def as_int(key: str) -> int | None:
        v = get(key)
        try:
            return int(float(v)) if v is not None else None
        except (TypeError, ValueError):
            return None

    star = str(get("star", "0")).lower() in ("1", "true", "on", "yes")
    watch = as_int("watch_ms")
    return {
        "decision": action,
        "star": star and action == "approve",
        "reasons": reasons,
        "note": (str(get("note", "")).strip()[:MAX_NOTE] or None),
        "reviewer": (str(get("reviewer", "")).strip()[:40] or "unknown"),
        "watch_ms": max(0, min(watch, 3_600_000)) if watch is not None else None,
        "blind": str(get("blind", "")).lower() in ("1", "true", "on", "yes"),
        "rank_position": as_int("rank"),
    }


def selection_label(row: Mapping[str, Any]) -> tuple[int, float] | None:
    """(label, weight) for learning which moments to clip, or None to leave the row out.

    Approve = 1 (star counts double). A reject for boring/needs context = 0. A reject
    without a reason = 0 at half weight. Rejects only for the window, hook, captions,
    a duplicate or safety say nothing about the moment, so they are left out. Under
    3 s of watching halves the weight."""
    reasons = row["reasons"]
    if isinstance(reasons, str):
        reasons = json.loads(reasons or "[]")
    quick = row.get("watch_ms") is not None and row["watch_ms"] < 3000
    factor = 0.5 if quick else 1.0
    if row["decision"] == "approve":
        return 1, (2.0 if row.get("star") else 1.0) * factor
    stages = {STAGE.get(x) for x in reasons}
    if "selection" in stages:
        return 0, 1.0 * factor
    if not reasons:
        return 0, 0.5 * factor
    return None


class FeedbackStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")  # web app and CLI may write at once
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def record(self, uid: str, review: Mapping[str, Any], clip: Any, *, run_id: str | None = None,
               game: str | None = None, receipt_path: str | None = None,
               now: float | None = None) -> int:
        """Store one review. `clip` is a queue Clip (channel, id, score, receipt, ...)."""
        rc = clip.receipt or {}
        ca = rc.get("clip_agent") or {}
        feats = features(rc, clip.score, getattr(clip, "clip_start_ts", None),
                         getattr(clip, "clip_end_ts", None))
        cur = self.db.execute(
            "INSERT INTO clip_feedback (clip_uid, channel, game, run_id, clip_id, decision, star,"
            " reasons, note, reviewer, reviewed_at, watch_ms, blind, was_explore, rank_position,"
            " score, confidence, p_approve, model_version, features, receipt_path)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (uid, clip.channel, game or rc.get("game") or None, run_id, clip.id,
             review["decision"], int(bool(review.get("star"))), json.dumps(review.get("reasons") or []),
             review.get("note"), review.get("reviewer") or "unknown", now or time.time(),
             review.get("watch_ms"), int(bool(review.get("blind"))), int(bool(rc.get("explore"))),
             review.get("rank_position"), clip.score, _num(ca.get("confidence_score")),
             _num(rc.get("p_approve")), rc.get("model_version"), json.dumps(feats), receipt_path))
        self.db.commit()
        return cur.lastrowid

    def latest(self, uid: str) -> dict | None:
        r = self.db.execute("SELECT * FROM clip_feedback WHERE clip_uid=? ORDER BY reviewed_at DESC, id DESC"
                            " LIMIT 1", (uid,)).fetchone()
        return self._row(r) if r else None

    def has(self, uid: str) -> bool:
        return self.db.execute("SELECT 1 FROM clip_feedback WHERE clip_uid=? LIMIT 1", (uid,)).fetchone() is not None

    def current(self, since: float | None = None) -> list[dict]:
        """The newest review per clip and reviewer: what learning should use."""
        rows = self.db.execute(
            "SELECT f.* FROM clip_feedback f JOIN (SELECT clip_uid, reviewer, MAX(id) AS id FROM clip_feedback"
            " GROUP BY clip_uid, reviewer) last ON f.id = last.id WHERE f.reviewed_at >= ?"
            " ORDER BY f.reviewed_at", (since or 0,)).fetchall()
        return [self._row(r) for r in rows]

    def export_csv(self, since: float | None = None) -> str:
        """Current reviews as CSV, selection features flattened into columns."""
        rows = self.current(since)
        feat_cols: list[str] = []
        for r in rows:
            for k in r["features"].get("selection", {}):
                if k not in feat_cols:
                    feat_cols.append(k)
        base = ["clip_uid", "channel", "game", "run_id", "clip_id", "decision", "star", "reasons", "note",
                "reviewer", "reviewed_at", "watch_ms", "blind", "was_explore", "rank_position", "score",
                "confidence", "label", "label_weight"]
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(base + [f"f_{k}" for k in feat_cols])  # f_ = feature snapshot at review time
        for r in rows:
            lab = selection_label(r)
            w.writerow([r["clip_uid"], r["channel"], r["game"], r["run_id"], r["clip_id"], r["decision"],
                        int(r["star"]), "|".join(r["reasons"]), r["note"] or "", r["reviewer"],
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["reviewed_at"])),
                        r["watch_ms"], int(r["blind"] or 0), r["was_explore"], r["rank_position"],
                        r["score"], r["confidence"], "" if lab is None else lab[0],
                        "" if lab is None else lab[1]]
                       + [r["features"]["selection"].get(k) for k in feat_cols])
        return buf.getvalue()

    def stats(self, since: float | None = None) -> dict:
        rows = self.current(since)
        out: dict = {"reviews": len(rows), "approved": 0, "starred": 0, "rejected": 0,
                     "reasons": {code: 0 for code in REASON_CODES}, "by_clip_type": {},
                     "by_channel": {}, "reviewers": {}, "selection_labels": 0}
        for r in rows:
            ok = r["decision"] == "approve"
            out["approved" if ok else "rejected"] += 1
            out["starred"] += int(bool(r["star"]))
            for x in r["reasons"]:
                out["reasons"][x] = out["reasons"].get(x, 0) + 1
            ct = r["features"].get("selection", {}).get("clip_type") or "unknown"
            for key, name in (("by_clip_type", ct), ("by_channel", r["channel"] or "?"),
                              ("reviewers", r["reviewer"] or "unknown")):
                b = out[key].setdefault(name, {"reviews": 0, "approved": 0})
                b["reviews"] += 1
                b["approved"] += int(ok)
            out["selection_labels"] += int(selection_label(r) is not None)
        return out

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["reasons"] = json.loads(d["reasons"] or "[]")
        d["features"] = json.loads(d["features"] or "{}")
        return d


def backfill(store: FeedbackStore, queues: Iterable[tuple[str | None, Path]], game_of=None) -> int:
    """Copy approved/rejected clips that have no review row yet (status only, no reasons),
    and repair the old "rejectd" status to "rejected" so the posting pages skip them.

    `queues` = (run_id or None for the main queue, path to its queue.db)."""
    from .queue import ClipQueue
    added = 0
    for run_id, db in queues:
        if not Path(db).is_file():
            continue
        q = ClipQueue(db)
        try:
            for c in q.ranked(None, 100_000):
                if c.status not in DECISION_OF:
                    continue
                if c.status == "rejectd":
                    q.update(c.id, status="rejected")
                uid = clip_uid(c.channel, run_id, c.id)
                if store.has(uid):
                    continue
                review = {"decision": DECISION_OF[c.status], "star": False, "reasons": [], "note": "backfilled",
                          "reviewer": "backfill", "watch_ms": None, "blind": False, "rank_position": None}
                store.record(uid, review, c, run_id=run_id,
                             game=game_of(run_id) if game_of else None, now=c.created_at)
                added += 1
        finally:
            q.db.close()
    return added


def print_stats(s: dict) -> None:
    n = s["reviews"]
    if not n:
        print("No reviews yet.")
        return
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"  # noqa: E731
    print(f"{n} clips reviewed: {s['approved']} approved ({pct(s['approved'], n)}), "
          f"{s['starred']} starred, {s['rejected']} rejected")
    print(f"usable for selection learning: {s['selection_labels']} (phase 2 needs 300+)")
    print("\nreject reasons:")
    for code, label, stage in REASONS:
        if s["reasons"].get(code):
            print(f"  {s['reasons'][code]:>4}  {label} [{stage}]")
    for key, title in (("by_clip_type", "clip type"), ("by_channel", "channel"), ("reviewers", "reviewer")):
        print(f"\nby {title}:")
        for name, b in sorted(s[key].items(), key=lambda kv: -kv[1]["reviews"]):
            print(f"  {name:<20} {b['reviews']:>4} reviews  {pct(b['approved'], b['reviews']):>4} approved")


# --------------------------------------------------------------------------- web UI bits
def stable_order(uid: str) -> str:
    """Blind review order: shuffled, but the same on every page load."""
    return hashlib.sha1(uid.encode()).hexdigest()


def review_bar(blind: bool, toggle_url: str) -> str:
    """Reviewer name + blind toggle, shown once at the top of /clips. The script fills
    reviewer and watch time into every review form before it is sent."""
    e = html.escape
    return f"""
<div class='card row' id='reviewbar'>
<label style='margin:0'>Reviewer <input id='reviewer' maxlength='40' placeholder='your name' style='width:12em'></label>
<span class='mut'>{'Blind review: scores hidden until you decide.' if blind else 'Scores visible.'}</span>
<a class='btn ghost' style='margin-left:auto' href='{e(toggle_url)}'>{'Show scores' if blind else 'Blind review'}</a>
</div>
<script>
(()=>{{const k='cb_reviewer',inp=document.getElementById('reviewer');
try{{inp.value=localStorage.getItem(k)||''}}catch(_){{}}
inp.addEventListener('change',()=>{{try{{localStorage.setItem(k,inp.value.trim())}}catch(_){{}}}});
const seen={{}};
document.querySelectorAll('video[data-uid]').forEach(v=>v.addEventListener('play',()=>{{
  if(!seen[v.dataset.uid])seen[v.dataset.uid]=Date.now();}}));
document.querySelectorAll('form.review').forEach(f=>f.addEventListener('submit',ev=>{{
  if(!inp.value.trim()){{ev.preventDefault();inp.focus();inp.placeholder='fill in your name first';
    inp.style.outline='2px solid #C8F03C';inp.scrollIntoView({{block:'center'}});return;}}
  f.querySelector('[name=reviewer]').value=inp.value.trim();
  const t=seen[f.dataset.uid];f.querySelector('[name=watch_ms]').value=t?Date.now()-t:'';}}));
}})();
</script>"""


def review_form(url: str, uid: str, rank: int, blind: bool, last: dict | None) -> str:
    """Approve / top clip / reject-with-reasons for one clip card."""
    e = html.escape
    hidden = (f"<input type='hidden' name='reviewer'><input type='hidden' name='watch_ms'>"
              f"<input type='hidden' name='rank' value='{rank}'>"
              f"<input type='hidden' name='blind' value='{1 if blind else 0}'>")
    prev = ""
    if last:
        why = ", ".join(lbl for code, lbl, _ in REASONS if code in last["reasons"])
        prev = (f"<p class='mut' style='margin:6px 0'>Last review: <b>{e(last['decision'])}"
                f"{' (top clip)' if last['star'] else ''}</b> by {e(last['reviewer'] or '?')}"
                f"{': ' + e(why) if why else ''}{' &middot; ' + e(last['note']) if last.get('note') else ''}</p>")
    chips = "".join(f"<label style='display:inline-block;margin:2px 8px 2px 0'><input type='checkbox' "
                    f"name='reason' value='{code}'> {e(lbl)}</label>" for code, lbl, _ in REASONS)
    return (prev +
            f"<div class='row'>"
            f"<form class='review' data-uid='{e(uid)}' method='post' action='{url}/approve'>{hidden}"
            f"<button>Approve</button></form>"
            f"<form class='review' data-uid='{e(uid)}' method='post' action='{url}/approve'>{hidden}"
            f"<input type='hidden' name='star' value='1'><button>Top clip</button></form></div>"
            f"<details style='margin-top:6px'><summary>Reject&hellip;</summary>"
            f"<form class='review' data-uid='{e(uid)}' method='post' action='{url}/reject'>{hidden}"
            f"<p style='margin:6px 0'>{chips}</p>"
            f"<label>Note (optional)<input name='note' maxlength='{MAX_NOTE}'></label>"
            f"<p><button class='ghost'>Reject</button></p></form></details>")
