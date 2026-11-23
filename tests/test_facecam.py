"""Facecam auto-detect for channels with no facecam box in channels.yaml."""
import pytest

from clipfarm.config import channel_settings, load_settings
from clipfarm.craft import facecam as fc
from clipfarm import engine

REEL = channel_settings(load_settings(), {"name": "x", "platform": "twitch"})["reel"]
BOX = {"x": 0.0, "y": 0.12, "w": 0.21, "h": 0.2}


@pytest.fixture(autouse=True)
def _fresh_memory():
    engine._found_facecams.clear()
    yield
    engine._found_facecams.clear()


def test_configured_box_wins_and_skips_detection():
    called = []
    box, src = engine.pick_facecam("raw.mp4", "xqc", REEL, BOX, detect=lambda *a, **k: called.append(1))
    assert (box, src, called) == (BOX, "config", [])


def test_unconfigured_channel_gets_detected_box_shaped_for_the_band():
    seen = {}

    def detect(raw, aspect):
        seen["aspect"] = aspect
        return BOX
    box, src = engine.pick_facecam("raw.mp4", "caseoh_", REEL, None, detect=detect)
    assert (box, src) == (BOX, "auto")
    assert seen["aspect"] == pytest.approx(1080 / (1920 * 0.30))


def test_reuses_last_found_box_when_streamer_looks_away():
    engine.pick_facecam("a.mp4", "caseoh_", REEL, None, detect=lambda *a, **k: BOX)
    box, src = engine.pick_facecam("b.mp4", "caseoh_", REEL, None, detect=lambda *a, **k: None)
    assert (box, src) == (BOX, "auto-last")
    box, src = engine.pick_facecam("c.mp4", "other", REEL, None, detect=lambda *a, **k: None)
    assert (box, src) == (None, "none")   # never borrowed from another channel


def test_auto_off_or_gameplay_only_layout_means_no_facecam():
    hit = lambda *a, **k: BOX   # noqa: E731
    assert engine.pick_facecam("r", "c", {**REEL, "facecam_auto": False}, None, detect=hit) == (None, "none")
    assert engine.pick_facecam("r", "c", {**REEL, "layout": "gameplay_only"}, None, detect=hit) == (None, "none")


def test_steady_face_beats_faces_that_move():
    cam = (40, 90, 30, 34)
    frames = [[cam, (300 + 40 * i, 200, 50, 50)] for i in range(6)]   # NPC face walks across
    frames[2] = []                                                    # one miss is fine
    assert fc.steady_face(frames, min_hits=3) == cam
    assert fc.steady_face([[(10, 10, 30, 30)], [], [], []], min_hits=2) is None


def test_fit_aspect_keeps_face_in_view():
    face = (40, 90, 30, 34)
    x0, y0, x1, y1 = fc.fit_aspect((0, 45, 135, 146), face, 1.875)
    assert (x0, x1) == (0, 135) and (x1 - x0) / (y1 - y0) == pytest.approx(1.875, abs=0.02)
    assert y0 <= face[1] and y1 >= face[1] + face[3] and y0 >= 45
    x0, y0, x1, y1 = fc.fit_aspect((0, 0, 400, 100), face, 1.875)   # too wide: trim sides
    assert (y0, y1) == (0, 100) and x0 <= face[0] and x1 >= face[0] + face[2]


cv2, np = fc.cv2, fc.np
needs_cv = pytest.mark.skipif(cv2 is None, reason="opencv-python-headless not installed")


def _frames(n=8, W=640, H=360, cam=(0, 45, 135, 146)):
    """Gameplay noise that changes every frame, with a still webcam overlay (bright border,
    a room edge inside it) in the top-left."""
    rng = np.random.default_rng(1)
    x0, y0, x1, y1 = cam
    out = []
    for _ in range(n):
        img = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 3)
        img[y0:y1, x0:x1] = 60
        img[y0 + 10:y1, x0 + 70:x1] = 90               # wall edge inside the room
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 2)
        out.append(img)
    return out


@needs_cv
def test_overlay_box_finds_the_webcam_border():
    face = (40, 90, 30, 34)
    x0, y0, x1, y1 = fc.overlay_box(_frames(), face)
    assert x0 == 0 and abs(y0 - 45) <= 2 and abs(x1 - 135) <= 2 and abs(y1 - 146) <= 2


@needs_cv
def test_overlay_box_falls_back_to_face_sized_box_without_border():
    rng = np.random.default_rng(2)
    frames = [cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (0, 0), 3)
              for _ in range(6)]
    x0, y0, x1, y1 = fc.overlay_box(frames, (300, 150, 30, 34))
    assert x0 < 300 and x1 > 330 and y0 < 150 and y1 > 184
    assert x1 - x0 < 200 and y1 - y0 < 200


@needs_cv
def test_no_face_in_clip_returns_none(tmp_path):
    if not fc.available():
        pytest.skip("face model or OpenCV missing")
    path = tmp_path / "noface.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    for f in _frames(20):
        vw.write(f)
    vw.release()
    assert fc.detect_facecam(path) is None
