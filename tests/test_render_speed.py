"""Stream-copy cut (lead-in trimmed by the render), one craft at a time, cheap bg blur."""
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from clipfarm import engine
from clipfarm.config import channel_settings, load_settings
from clipfarm.craft.captions import Word
from clipfarm.craft.reel import render_reel
from clipfarm.queue import ClipQueue
from clipfarm.recorder import LiveRecorder

CFG = channel_settings(load_settings(), {"name": "x", "platform": "twitch"})
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
GRAY = 30   # second i of the test stream is a flat frame of grey level GRAY * i


def _buffer(tmp_path: Path, seconds: int = 8) -> LiveRecorder:
    """A recorder buffer like the live one: 2 s .ts segments, keyframe every 2 s."""
    src = tmp_path / "src.mp4"
    ins = []
    for i in range(seconds):
        ins += ["-f", "lavfi", "-i", f"color=c=0x{GRAY * i:02x}{GRAY * i:02x}{GRAY * i:02x}"
                                     f":s=320x180:r=30:d=1"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-f", "lavfi", "-i",
                    "sine=frequency=440:r=48000", "-filter_complex",
                    "".join(f"[{i}:v]" for i in range(seconds)) +
                    f"concat=n={seconds}:v=1:a=0[v]", "-map", "[v]", "-map", f"{seconds}:a",
                    "-t", str(seconds), "-c:v", "libx264", "-g", "60", "-keyint_min", "60",
                    "-sc_threshold", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)],
                   check=True)
    buf = tmp_path / "buffer"
    buf.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-c", "copy", "-f", "segment",
                    "-segment_time", "2", "-reset_timestamps", "1", "-segment_list",
                    str(buf / "segments.csv"), "-segment_list_type", "csv",
                    str(buf / "seg_%06d.ts")], check=True)
    rec = LiveRecorder.__new__(LiveRecorder)
    rec.dir, rec.list_path, rec.t0 = buf, buf / "segments.csv", 1000.0
    return rec


def _grey_at(path: Path, t: float) -> float:
    """Mean grey of the centre of the frame at t."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v",
                          "1", "-vf", "crop=iw/4:ih/8:iw*3/8:ih*7/16,format=gray", "-f",
                          "rawvideo", "pipe:1"], capture_output=True, check=True).stdout
    return float(np.frombuffer(out, np.uint8).mean())


def _second_shown(path: Path, t: float) -> int:
    return round(_grey_at(path, t) / GRAY)


@needs_ffmpeg
def test_cut_copies_from_the_keyframe_before_and_reports_the_lead(tmp_path):
    rec = _buffer(tmp_path)
    raw = tmp_path / "raw.mp4"
    lead = rec.cut(1002.5, 1006.5, raw)
    seg_start = max(a for _, a, _ in rec.segments() if a <= 1002.5) - 1000
    assert 0 < lead <= 2.5 and lead == pytest.approx(2.5 - seg_start, abs=0.01)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
                            "-of", "csv=p=0", str(raw)], capture_output=True, text=True).stdout
    assert "h264" in probe and "aac" in probe      # copied, audio kept
    assert _second_shown(raw, 0.1) == int(seg_start)   # raw starts at that keyframe
    assert _second_shown(raw, lead + 0.6) == 3      # 2.5 + 0.6 = 3.1 s


@needs_ffmpeg
def test_render_trims_the_lead_so_the_reel_starts_on_the_moment(tmp_path):
    rec = _buffer(tmp_path)
    raw = tmp_path / "raw.mp4"
    lead = rec.cut(1002.7, 1006.7, raw)
    reel = render_reel(raw, tmp_path / "reel.mp4", {**CFG["reel"], "layout": "gameplay_only",
                                                    "width": 360, "height": 640,
                                                    "chat_strip": False},
                       None, [Word("hi", 0.1, 0.5)], hook_text="", lead=lead)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(reel)], capture_output=True,
                               text=True).stdout)
    assert dur == pytest.approx(4.0, abs=0.1)       # the window, not window + lead
    assert _second_shown(reel, 0.1) == 2            # 2.7 + 0.1
    assert _second_shown(reel, 0.5) == 3            # 2.7 + 0.5 = 3.2: no lead left in
    assert _second_shown(reel, 3.5) == 6


def test_craft_passes_the_lead_to_render_and_nudity_and_retimes_raw_words(tmp_path,
                                                                          monkeypatch):
    seen, scanned = {}, {}
    monkeypatch.setattr(engine, "render_reel", lambda *a, **k: seen.update(k, words=a[4]))
    monkeypatch.setattr(engine, "pick_facecam", lambda *a, **k: (None, "none"))
    monkeypatch.setattr(engine, "transcribe", lambda *a: [Word("early", 0.2, 0.6),
                                                          Word("late", 1.9, 2.3)])
    q = ClipQueue(tmp_path / "q.db")
    cid = q.add("x", "twitch", 0, 30, 50, {"raw_lead_s": 1.5}, raw_path=str(tmp_path / "r.mp4"))

    def nud(path, cfg, start=0.0, **k):
        scanned["start"] = start
        return {"decision": "clean", "ranges": [], "why": "t"}

    engine.craft_clip(q, cid, {**CFG, "reel": {**CFG["reel"], "transcriber": "auto"}}, None,
                      nudity_check=nud)
    assert seen["lead"] == 1.5 and scanned["start"] == 1.5
    assert [(w.text, round(w.start, 2)) for w in seen["words"]] == [("late", 0.4)]


def test_clips_are_crafted_one_at_a_time(tmp_path, monkeypatch):
    running, peak = [0], [0]
    lock = threading.Lock()

    def slow_render(*a, **k):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.2)
        with lock:
            running[0] -= 1

    monkeypatch.setattr(engine, "render_reel", slow_render)
    monkeypatch.setattr(engine, "pick_facecam", lambda *a, **k: (None, "none"))
    q = ClipQueue(tmp_path / "q.db")
    ids = [q.add("x", "twitch", i, i + 30, 50, {}, raw_path=str(tmp_path / f"{i}.mp4"))
           for i in range(3)]
    cfg = {**CFG, "reel": {**CFG["reel"], "transcriber": "none"}}
    clean = lambda p, c, **k: {"decision": "clean", "ranges": [], "why": "t"}   # noqa: E731
    threads = [threading.Thread(target=engine.craft_clip, args=(q, i, cfg, None),
                                kwargs={"words": [], "nudity_check": clean}) for i in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1
    assert all(q.get(i).status == "crafted" for i in ids)
