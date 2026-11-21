from datetime import datetime

from clipfarm.queue import ClipQueue
from clipfarm.scheduler import in_window


def test_queue_ranks_by_score_and_keeps_receipt(tmp_path):
    q = ClipQueue(tmp_path / "q.db")
    a = q.add("c", "twitch", 0, 30, 5.0, {"why": "a"})
    b = q.add("c", "twitch", 0, 30, 9.0, {"why": "b", "peak_rate": 12})
    assert [c.id for c in q.ranked()] == [b, a]
    assert q.get(b).receipt["peak_rate"] == 12
    q.update(a, status="approved")
    assert [c.id for c in q.ranked("approved")] == [a]


def test_schedule_windows():
    sched = [{"days": ["fri"], "start": "22:00", "end": "02:00"}]
    assert in_window(sched, datetime(2026, 11, 20, 23, 0))   # Fri 23:00
    assert in_window(sched, datetime(2026, 11, 21, 1, 0))    # Sat 01:00 (spillover)
    assert not in_window(sched, datetime(2026, 11, 21, 23, 0))
    assert in_window([], datetime.now())
