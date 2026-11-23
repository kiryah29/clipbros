"""Find the streamer's facecam in a 16:9 clip when the channel has no facecam box configured.

1. Sample a few frames across the clip and look for faces (OpenCV's YuNet detector, model
   bundled in assets/models, MIT licence from opencv_zoo).
2. Keep the face that stays in the same spot in most frames: that is the webcam overlay
   (faces in the game move around or come and go).
3. Grow a box around that face to the overlay's border: overlay borders are edges that stay
   put in every frame, while gameplay edges move and average away. Where no clear border
   shows, a box a few face-sizes wide is used.

Returns the box as fractions of the frame, the same shape as `facecam:` in channels.yaml,
or None when no steady face is found (then the reel falls back to gameplay_only).
OpenCV is optional: without it this returns None and prints why once.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:          # pragma: no cover - depends on the machine
    cv2 = None
    np = None

MODEL = Path(__file__).resolve().parents[2] / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
_warned = False
_detector = None


def available() -> bool:
    global _warned
    ok = cv2 is not None and hasattr(cv2, "FaceDetectorYN") and MODEL.exists()
    if not ok and not _warned:
        print("  ! facecam auto-detect off: needs opencv-python-headless and " + str(MODEL))
        _warned = True
    return ok


def _face_detector(w: int, h: int):
    global _detector
    if _detector is None:
        try:   # OpenCV 5 warns about DNN targets on every load; nothing to act on
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            pass
        _detector = cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), 0.7, 0.3, 50)
    _detector.setInputSize((w, h))
    return _detector


def _probe(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    st = info["streams"][0]
    return int(st["width"]), int(st["height"]), float(info["format"].get("duration") or 0)


def sample_frames(path: Path, n: int = 8, width: int = 640) -> list:
    """n BGR frames spread over 10-90 % of the clip, resized to `width`.

    Frames come from ffmpeg, not cv2.VideoCapture: the Windows ARM64 OpenCV wheel ships
    without its FFmpeg video plugin, so VideoCapture opens nothing there and the facecam
    was silently never found."""
    try:
        W, H, dur = _probe(path)
    except (OSError, subprocess.CalledProcessError, KeyError, IndexError, ValueError):
        return []
    h = max(2, int(H * width / W) // 2 * 2)
    frames = []
    for i in range(n):
        t = dur * (0.1 + 0.8 * i / max(n - 1, 1))
        out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path),
                              "-frames:v", "1", "-vf", f"scale={width}:{h}",
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                             capture_output=True)
        if out.returncode == 0 and len(out.stdout) == width * h * 3:
            frames.append(np.frombuffer(out.stdout, np.uint8).reshape(h, width, 3).copy())
    return frames


def find_faces(img) -> list[tuple[int, int, int, int]]:
    """Face boxes (x, y, w, h) in pixels; tiny ones (crowds, posters) are ignored."""
    H, W = img.shape[:2]
    _, faces = _face_detector(W, H).detect(img)
    if faces is None:
        return []
    return [tuple(int(v) for v in f[:4]) for f in faces if f[3] >= H * 0.04]


def steady_face(per_frame: list[list[tuple]], min_hits: int) -> tuple | None:
    """The face seen near the same spot in the most frames (at least min_hits), as the
    median box of those sightings."""
    best: list[tuple] = []
    for faces in per_frame:
        for f in faces:
            cx, cy, s = f[0] + f[2] / 2, f[1] + f[3] / 2, f[2]
            near = []
            for other in per_frame:
                hit = [g for g in other
                       if abs(g[0] + g[2] / 2 - cx) < 0.5 * s and abs(g[1] + g[3] / 2 - cy) < 0.5 * s]
                if hit:
                    near.append(min(hit, key=lambda g: abs(g[0] - f[0]) + abs(g[1] - f[1])))
            if len(near) > len(best):
                best = near
    if len(best) < min_hits:
        return None
    return tuple(int(np.median([b[i] for b in best])) for i in range(4))


def _line_score(grad, idx: int, span: slice, thr: float) -> float:
    """How much of a straight line runs through row/col `idx` of `grad` (already oriented so
    idx is the first axis): the share of `span` where the edge is strong, times how much it
    stands out against its neighbours. Overlay borders score ~1 x high; heads, shelves and
    chat text are short or broken."""
    n = grad.shape[0]
    band = grad[max(idx - 1, 0):idx + 2, span].max(axis=0)
    if band.size == 0:
        return 0.0
    cover = float((band > thr).mean())
    near = grad[max(idx - 6, 0):min(idx + 7, n), span].mean()
    sharp = float(grad[idx, span].mean() - near * 0.5)
    return cover * max(sharp, 0.0) if cover >= 0.6 else 0.0


def _border(grad, lo: int, hi: int, span: slice, default: int, outward: int,
            snap: int = 0) -> int:
    """Strongest straight steady line between lo and hi; else the frame edge when the search
    reaches it (overlays usually sit flush in a corner); else `default`. A line within `snap`
    px of the frame edge is taken to be inside a flush overlay, so the frame edge wins."""
    n = grad.shape[0]
    lo, hi = max(lo, 0), min(hi, n)
    thr = 3.0 * float(np.median(grad)) + 1e-6
    best, best_i = 0.0, None
    for i in range(lo, hi):
        sc = _line_score(grad, i, span, thr)
        if sc > best:
            best, best_i = sc, i
    if best_i is not None:
        if outward < 0 and best_i < snap:
            return 0
        if outward > 0 and n - best_i <= snap:
            return n
        return best_i + (1 if outward > 0 else 0)
    if (outward < 0 and lo == 0) or (outward > 0 and hi == n):
        return 0 if outward < 0 else n
    return min(max(default, 0), n)


def overlay_box(frames, face: tuple) -> tuple[int, int, int, int]:
    """Box (x0, y0, x1, y1) in frame pixels around `face`, out to the overlay border."""
    stack = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    # an edge that moves between frames averages out; only the steady ones stay strong
    gx = np.mean([np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)) for f in stack], axis=0)
    gy = np.mean([np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)) for f in stack], axis=0)
    H, W = gx.shape
    fx, fy, fw, fh = face
    cx = fx + fw // 2
    rows = slice(max(fy - fh // 2, 0), min(fy + fh * 3 // 2, H))
    x0 = _border(gx.T, fx - 3 * fw, fx - fw // 4, rows, cx - int(1.6 * fw), -1, fw // 2)
    x1 = _border(gx.T, fx + fw + fw // 4, fx + 4 * fw, rows, cx + int(1.6 * fw), 1, fw // 2)
    cols = slice(x0 + 2, max(x1 - 2, x0 + 3))
    y0 = _border(gy, fy - 2 * fh, fy - fh // 4, cols, fy - fh, -1, fh // 2)
    y1 = _border(gy, fy + fh + fh // 4, fy + 3 * fh, cols, fy + int(2.2 * fh), 1, fh // 2)
    return x0, y0, x1, y1


def fit_aspect(box: tuple, face: tuple, aspect: float, face_at: float = 0.5,
               frame: tuple[int, int] | None = None, max_face: float = 0.4) -> tuple:
    """Trim box (x0, y0, x1, y1) to width/height = aspect, keeping the face in view: centred
    across, and `face_at` of the way down (the hook card covers the bottom of the band).

    With `frame` (W, H), a box so tight that the face would fill more than `max_face` of the
    band's height is grown around the face instead (borderless green-screen cams have no
    overlay border, so the border search hugs the head and the band became a forehead)."""
    x0, y0, x1, y1 = box
    fx, fy, fw, fh = face
    w, h = x1 - x0, y1 - y0
    if w / h > aspect:
        nw = int(round(h * aspect))
        nx = min(max(int(fx + fw / 2 - nw / 2), x0), x1 - nw)
        x0, x1 = nx, nx + nw
    else:
        nh = int(round(w / aspect))
        ny = min(max(int(fy + fh / 2 - nh * face_at), y0), y1 - nh)
        y0, y1 = ny, ny + nh
    if frame and fh > max_face * (y1 - y0):
        W, H = frame
        nh = min(int(round(fh / max_face)), H, int(W / aspect))
        nw = int(round(nh * aspect))
        nx = min(max(int(fx + fw / 2 - nw / 2), 0), W - nw)
        ny = min(max(int(fy + fh / 2 - nh * face_at), 0), H - nh)
        return nx, ny, nx + nw, ny + nh
    return x0, y0, x1, y1


def detect_facecam(path: Path, samples: int = 8, min_share: float = 0.4,
                   aspect: float | None = None) -> dict | None:
    """Facecam box as fractions of the frame. With `aspect` (band width / height in pixels)
    the box is trimmed to that shape around the face, so the reel's cover-crop never cuts
    the face off."""
    if not available():
        return None
    frames = sample_frames(Path(path), samples)
    if len(frames) < 2:
        return None
    face = steady_face([find_faces(f) for f in frames], max(2, round(len(frames) * min_share)))
    if face is None:
        return None
    x0, y0, x1, y1 = overlay_box(frames, face)
    H, W = frames[0].shape[:2]
    if x1 - x0 < face[2] or y1 - y0 < face[3]:
        return None
    if aspect:
        x0, y0, x1, y1 = fit_aspect((x0, y0, x1, y1), face, aspect, face_at=0.4, frame=(W, H))
    return {"x": round(x0 / W, 4), "y": round(y0 / H, 4),
            "w": round((x1 - x0) / W, 4), "h": round((y1 - y0) / H, 4)}
