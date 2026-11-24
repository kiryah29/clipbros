"""Nudity check: blur ranges, hold-for-review decisions, the reel blur and the pipeline step."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipfarm import engine, nudity
from clipfarm.config import channel_settings, load_settings
from clipfarm.craft.reel import render_reel
from clipfarm.queue import ClipQueue

CFG = channel_settings(load_settings(), {"name": "x", "platform": "twitch"})
NCFG = CFG["nudity"]
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def test_settings_section_is_loaded_and_overridable_per_channel():
    assert NCFG["enabled"] and NCFG["blur_score"] > NCFG["review_score"]
    ch = channel_settings(load_settings(), {"name": "x", "platform": "twitch",
                                            "nudity": {"blur_score": 0.5}})
    assert ch["nudity"]["blur_score"] == 0.5 and ch["nudity"]["sizes"] == NCFG["sizes"]


def test_ranges_are_padded_merged_and_clamped():
    assert nudity.merge_ranges([0.2, 0.5, 1.0], 0.5, 1.0, 10) == [[0.0, 1.5]]
    assert nudity.merge_ranges([2.0, 6.0], 0.5, 1.0, 10) == [[1.5, 2.5], [5.5, 6.5]]
    assert nudity.merge_ranges([9.9], 0.5, 1.0, 10) == [[9.4, 10.0]]


def frames(pairs, step=1 / 3):
    """[(n_frames, score)] -> per-frame (time, score, label) rows."""
    out, t = [], 0.0
    for n, s in pairs:
        for _ in range(n):
            out.append((t, s, "BUTTOCKS_EXPOSED"))
            t += step
    return out


def test_clean_clip():
    r = nudity.decide(frames([(90, 0.05)]), 30, NCFG)
    assert (r["decision"], r["ranges"]) == ("clean", [])


def test_clear_nudity_is_blurred_with_padding():
    # 20 s clean, 3.3 s of nudity, then clean (the caseoh clip's shape)
    r = nudity.decide(frames([(60, 0.05), (10, 0.6), (30, 0.05)]), 33.3, NCFG)
    assert r["decision"] == "blur"
    (a, b, _), = r["ranges"]
    assert a == pytest.approx(20 - NCFG["pad_s"], abs=0.01)
    assert b == pytest.approx(23 + NCFG["pad_s"], abs=0.01)
    assert r["labels"] == ["BUTTOCKS_EXPOSED"]


def test_unsure_frames_next_to_a_blurred_part_are_covered():
    r = nudity.decide(frames([(60, 0.05), (10, 0.6), (1, 0.27), (29, 0.05)]), 33.3, NCFG)
    assert r["decision"] == "blur" and r["unsure"] == []


CAM = {"x": 0.0, "y": 0.0, "w": 0.25, "h": 0.3}


def boxed(rows, box):
    return [(*r, box) for r in rows]


def test_unsure_frame_on_the_facecam_holds_the_clip():
    rows = boxed(frames([(30, 0.05), (1, 0.28), (60, 0.05)]), [0.05, 0.05, 0.1, 0.1])
    r = nudity.decide(rows, 30, NCFG, facecam=CAM)
    assert r["decision"] == "review" and r["ranges"] == []
    assert "facecam" in r["why"]


def test_game_character_below_the_game_threshold_is_left_alone():
    # kaicenat 19:51 run #2: GTA characters at 0.3-0.45 blurred the whole band for ~16 s
    rows = boxed(frames([(30, 0.05), (48, 0.4), (12, 0.05)]), [0.5, 0.4, 0.1, 0.2])
    r = nudity.decide(rows, 30, NCFG, facecam=CAM)
    assert (r["decision"], r["ranges"]) == ("clean", [])


def test_game_nudity_blurs_only_its_box_and_never_holds():
    rows = boxed(frames([(80, 0.7), (10, 0.05)]), [0.5, 0.4, 0.1, 0.2])
    r = nudity.decide(rows, 30, NCFG, facecam=CAM)
    assert r["decision"] == "blur"
    (a, b, box), = r["ranges"]
    assert box[0] < 0.5 < box[0] + box[2] and box[2] < 0.2   # padded box, not the frame


def test_full_camera_stream_counts_as_a_real_person():
    rows = boxed(frames([(30, 0.05), (1, 0.28), (60, 0.05)]), [0.5, 0.4, 0.1, 0.2])
    assert nudity.decide(rows, 30, NCFG, full_cam=True)["decision"] == "review"


def test_no_model_holds_or_skips_as_configured():
    assert nudity.unavailable(NCFG, "x")["decision"] == "review"
    assert nudity.unavailable({**NCFG, "on_unavailable": "skip"}, "x")["decision"] == "skipped"
    assert nudity.check_clip("x.mp4", {**NCFG, "enabled": False})["decision"] == "skipped"


def test_blur_graph():
    assert nudity.blur_graph([], "[a]", "[b]") == ["[a]null[b]"]
    g = ";".join(nudity.blur_graph([[1.5, 3.5, [0.5, 0.4, 0.1, 0.2]], [6, 7, None]], "[a]", "[b]"))
    assert "crop=trunc(iw*0.1/2)*2" in g and "overlay=main_w*0.5:main_h*0.4" in g
    assert "between(t\\,1.50\\,3.50)" in g and "between(t\\,6.00\\,7.00)'[b]" in g


def _video(path: Path, colors: list[tuple[str, float]]) -> Path:
    """Test video: testsrc pattern in black parts, plain white for 'nude' parts."""
    args, parts = [], []
    for i, (c, d) in enumerate(colors):
        src = f"color=c={c}:s=320x180:r=30:d={d}" if c != "pattern" else f"testsrc=s=320x180:r=30:d={d}"
        args += ["-f", "lavfi", "-i", src]
        parts.append(f"[{i}:v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-f", "lavfi", "-i",
                    f"anullsrc=r=48000:cl=mono", "-filter_complex",
                    "".join(parts) + f"concat=n={len(colors)}:v=1:a=0[v]", "-map", "[v]",
                    "-map", f"{len(colors)}:a", "-shortest", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", str(path)], check=True)
    return path


class BrightIsNude:
    """Fake detector: a bright frame 'shows nudity'."""
    engine = "fake"

    def scores(self, img, size):
        s = np.zeros((1, len(nudity.LABELS)), np.float32)
        s[0, nudity.LABELS.index("MALE_GENITALIA_EXPOSED")] = 0.9 if img.mean() > 200 else 0.0
        return s


@needs_ffmpeg
def test_check_clip_finds_the_window_in_a_real_video(tmp_path):
    v = _video(tmp_path / "raw.mp4", [("pattern", 4), ("white", 2), ("pattern", 4)])
    r = nudity.check_clip(v, NCFG, detector=BrightIsNude())
    assert r["decision"] == "blur" and r["labels"] == ["MALE_GENITALIA_EXPOSED"]
    (a, b, _), = r["ranges"]
    assert 3.3 <= a <= 4.0 and 5.6 <= b <= 6.7


def _sharpness(path: Path, t: float) -> float:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-vf", "crop=600:300:240:800,format=gray", "-f", "rawvideo", "pipe:1"],
                         capture_output=True, check=True).stdout
    img = np.frombuffer(raw, np.uint8).reshape(300, 600).astype(np.float32)
    return float(np.abs(img[:, 12:] - img[:, :-12]).mean())   # local contrast


@needs_ffmpeg
def test_reel_blurs_only_the_censored_range(tmp_path):
    raw = _video(tmp_path / "raw.mp4", [("pattern", 3)])
    reel = render_reel(raw, tmp_path / "reel.mp4", {**CFG["reel"], "layout": "gameplay_only",
                                                    "chat_strip": False},
                       None, [], hook_text="", censor=[[1.0, 2.0]])
    assert _sharpness(reel, 1.5) < 0.65 * _sharpness(reel, 0.5)
    assert _sharpness(reel, 2.5) > 0.8 * _sharpness(reel, 0.5)


@pytest.mark.parametrize("decision,status", [("review", "held"), ("blur", "crafted"),
                                             ("clean", "crafted")])
def test_pipeline_holds_or_blurs(tmp_path, monkeypatch, decision, status):
    seen = {}
    monkeypatch.setattr(engine, "render_reel", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(engine, "pick_facecam", lambda *a, **k: (None, "none"))
    q = ClipQueue(tmp_path / "q.db")
    raw = tmp_path / "1_raw.mp4"
    cid = q.add("x", "twitch", 0, 30, 50, {"hook": "h"}, raw_path=str(raw))
    result = {"decision": decision, "ranges": [[20.0, 23.5]] if decision != "clean" else [],
              "why": "test"}
    engine.craft_clip(q, cid, {**CFG, "reel": {**CFG["reel"], "transcriber": "none"}}, None,
                      words=[], nudity_check=lambda path, cfg, **k: result)
    c = q.get(cid)
    assert c.status == status
    assert c.receipt["nudity"] == result and c.receipt["hook"] == "h"
    assert seen["censor"] == result["ranges"]


MODEL = nudity.model_path(NCFG)


@pytest.mark.skipif(not MODEL.exists(), reason="nudity model not downloaded (setup-nudity)")
def test_real_model_loads_and_scores_a_blank_frame_low():
    det = nudity.Detector(MODEL)
    score, _, _ = nudity.frame_score(det, np.zeros((540, 960, 3), np.uint8), NCFG)
    assert score < NCFG["review_score"]
