"""Nudity check: scan a clip before it is rendered, blur what shows nudity, hold what is unsure.

1. Sample frames across the raw clip (ffmpeg, `sample_fps` a second).
2. Run NudeNet's exposed-body-part detector on each frame (ONNX; onnxruntime when installed,
   else OpenCV's DNN module, both run on the Windows ARM64 PC). Small figures in wide shots
   are missed at 320 px, so each frame runs at every size in `sizes` and the highest score wins.
3. Only the detected box is blurred, over padded, merged time ranges (craft/reel.py).
   Game footage (anything outside the facecam) needs game_blur_score: GTA characters score
   0.3-0.5 and used to blur the whole band. A hit on the facecam (a real person; the whole
   frame on a full-camera stream) uses blur_score, and any facecam frame at review_score or
   more, or no model at all, holds the clip for a human look (status "held").

The model (NudeNet 320n, ~12 MB) is downloaded into models/ on first use, or with
`python -m clipfarm setup-nudity`.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path

try:
    import numpy as np
except ImportError:          # pragma: no cover - depends on the machine
    np = None

from .config import ROOT

# the model ships inside NudeNet's PyPI wheel; only 320n.onnx is taken out of it
MODEL_WHEEL = ("https://files.pythonhosted.org/packages/1c/ee/1aa02d44ba958cc77e16ff1e41a0aac5"
               "e721037db7bf62b9c9d124917f87/nudenet-3.4.2-py3-none-any.whl")
MODEL_SHA256 = "c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f"
LABELS = ["FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
          "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
          "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE",
          "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
          "BUTTOCKS_COVERED"]
DEFAULTS = {
    "enabled": True,
    "model": "models/nudenet_320n.onnx",
    "auto_download": True,
    "classes": ["BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
                "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED"],
    "sizes": [320, 640],
    "sample_fps": 3.0,
    "scan_width": 960,
    "blur_score": 0.30,
    "review_score": 0.26,
    "game_blur_score": 0.50,
    "box_pad": 0.15,
    "pad_s": 0.5,
    "merge_gap_s": 1.0,
    "on_unavailable": "review",
}

_model = None
_model_err: str | None = None


def settings(cfg: dict | None) -> dict:
    return {**DEFAULTS, **(cfg or {})}


# ------------------------------------------------------------------ decision (pure, tested)
def merge_ranges(times: list[float], pad: float, gap: float, dur: float) -> list[list[float]]:
    """Padded [start, end] ranges around flagged sample times; ranges closer than `gap` join."""
    out: list[list[float]] = []
    for t in sorted(times):
        a, b = max(0.0, t - pad), min(dur, t + pad) if dur else t + pad
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[round(a, 2), round(b, 2)] for a, b in out]


def _overlap(box: list[float], cam: dict) -> float:
    """Share of `box` [x, y, w, h] (fractions) that lies inside the facecam box."""
    x, y, w, h = box
    ix = max(0.0, min(x + w, cam["x"] + cam["w"]) - max(x, cam["x"]))
    iy = max(0.0, min(y + h, cam["y"] + cam["h"]) - max(y, cam["y"]))
    return ix * iy / (w * h) if w * h > 0 else 0.0


def _union(boxes: list, pad: float) -> list[float] | None:
    """Padded box around all `boxes`; None (= whole frame) if any hit has no box."""
    if not boxes or any(b is None for b in boxes):
        return None
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0, x1, y1 = max(0.0, x0 - px), max(0.0, y0 - py), min(1.0, x1 + px), min(1.0, y1 + py)
    return [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]


def decide(scores: list[tuple], dur: float, cfg: dict, facecam: dict | None = None,
           full_cam: bool = False) -> dict:
    """scores: (time s, best score, label[, box]) per sampled frame, box = [x, y, w, h] as
    fractions of the frame. Returns the check result: decision clean | blur | review, blur
    ranges [start, end, box or None = whole frame], and why."""
    c = settings(cfg)
    rows = [(r[0], r[1], r[2], r[3] if len(r) > 3 else None) for r in scores]

    def on_cam(box) -> bool:   # a real person, not a game character
        return full_cam or bool(facecam and box and _overlap(box, facecam) >= 0.3)

    hits = [r for r in rows
            if r[1] >= (c["blur_score"] if on_cam(r[3]) else c["game_blur_score"])]
    ranges = [[a, b, _union([r[3] for r in hits if a <= r[0] <= b], float(c["box_pad"]))]
              for a, b in merge_ranges([r[0] for r in hits], c["pad_s"], c["merge_gap_s"], dur)]
    # only the facecam holds a clip: game footage is blurred or left alone
    unsure = [r for r in rows if r[1] >= c["review_score"] and on_cam(r[3])]
    reasons = []
    if unsure:
        reasons.append(f"{len(unsure)} facecam frame(s) might show nudity "
                       f"({', '.join(sorted({r[2] for r in unsure}))})")
    decision = "review" if reasons else ("blur" if ranges else "clean")
    return {
        "decision": decision,
        "ranges": ranges,
        "labels": sorted({r[2] for r in hits}),
        "max_score": round(max((r[1] for r in rows), default=0.0), 3),
        "frames": len(rows),
        "unsure": [[round(t, 2), round(s, 3), lab] for t, s, lab, _ in unsure],
        "why": "; ".join(reasons) or (f"blurred {len(ranges)} part(s): " + ", ".join(
            f"{a:.1f}-{b:.1f}s" for a, b, _ in ranges) if ranges else "no nudity found"),
    }


def unavailable(cfg: dict, why: str) -> dict:
    hold = settings(cfg)["on_unavailable"] == "review"
    return {"decision": "review" if hold else "skipped", "ranges": [], "labels": [],
            "max_score": 0.0, "frames": 0, "why": f"nudity check could not run: {why}"}


# ------------------------------------------------------------------ model
def model_path(cfg: dict) -> Path:
    p = Path(settings(cfg)["model"])
    return p if p.is_absolute() else ROOT / p


def download_model(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MODEL_WHEEL, timeout=120) as r:
        wheel = r.read()
    data = zipfile.ZipFile(io.BytesIO(wheel)).read("nudenet/320n.onnx")
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("nudity model download does not match the expected checksum")
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


class Detector:
    """NudeNet exposed-part detector. run(img_bgr, size) -> (n, 18) class scores."""

    def __init__(self, path: Path):
        try:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._name = self._sess.get_inputs()[0].name
            self._net = None
            self.engine = "onnxruntime"
        except ImportError:
            import cv2
            self._sess = None
            self._net = cv2.dnn.readNetFromONNX(str(path))
            self.engine = "opencv"

    def scores(self, img, size: int):
        return self.detect(img, size)[0]

    def detect(self, img, size: int):
        """(class scores (n, 18), boxes (n, 4) as [x, y, w, h] fractions of img)."""
        H, W = img.shape[:2]
        m = max(H, W)
        sq = np.zeros((m, m, 3), np.uint8)       # pad to a square, bottom/right (as NudeNet)
        sq[:H, :W] = img
        try:
            import cv2
            sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_LINEAR)
        except ImportError:   # nearest-neighbour fallback
            idx = np.linspace(0, m - 1, size).astype(np.int32)
            sq = sq[idx][:, idx]
        blob = sq[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        if self._sess is not None:
            out = self._sess.run(None, {self._name: np.ascontiguousarray(blob)})[0]
        else:
            self._net.setInput(np.ascontiguousarray(blob))
            out = self._net.forward()
        rows = out[0].T
        cx, cy, bw, bh = (rows[:, i] * m / size for i in range(4))
        boxes = np.stack([(cx - bw / 2) / W, (cy - bh / 2) / H, bw / W, bh / H], axis=1)
        return rows[:, 4:], boxes


def load_model(cfg: dict) -> Detector | None:
    global _model, _model_err
    if _model is not None or _model_err:
        return _model
    if np is None:
        _model_err = "numpy is not installed"
        return None
    p = model_path(cfg)
    try:
        if not p.exists():
            if not settings(cfg)["auto_download"]:
                raise FileNotFoundError(f"{p} missing (python -m clipfarm setup-nudity)")
            print(f"  downloading nudity model to {p} ...")
            download_model(p)
        _model = Detector(p)
    except Exception as e:  # noqa: BLE001 - any failure means: no check, hold the clip
        _model_err = f"{type(e).__name__}: {e}"
        print(f"  ! nudity check off: {_model_err}")
    return _model


def frame_score(det: Detector, img, cfg: dict) -> tuple[float, str, list | None]:
    """Best score over the configured classes and sizes, its label and its box (None when
    the detector gives no boxes)."""
    c = settings(cfg)
    cls = [LABELS.index(k) for k in c["classes"] if k in LABELS]
    best, label, box = 0.0, "", None
    for size in c["sizes"]:
        if hasattr(det, "detect"):
            sc, boxes = det.detect(img, int(size))
        else:
            sc, boxes = det.scores(img, int(size)), None
        sc = sc[:, cls]
        a, k = np.unravel_index(int(sc.argmax()), sc.shape)
        if sc[a, k] > best:
            best, label = float(sc[a, k]), c["classes"][k]
            box = None if boxes is None else [round(float(v), 4) for v in boxes[a]]
    return best, label, box


# ------------------------------------------------------------------ clip scan
def _probe(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    st = info["streams"][0]
    return int(st["width"]), int(st["height"]), float(info["format"].get("duration") or 0)


def frames(path: Path, fps: float, width: int, start: float = 0.0):
    """(time, BGR frame) pairs, `fps` a second, from one ffmpeg pass; times count from
    `start` seconds into the file."""
    W, H, dur = _probe(path)
    w = min(width, W) // 2 * 2
    h = max(2, int(H * w / W) // 2 * 2)
    seek = ["-ss", f"{start:.3f}"] if start else []
    proc = subprocess.Popen(["ffmpeg", "-v", "error", *seek, "-i", str(path), "-vf",
                             f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "pipe:1"], stdout=subprocess.PIPE)
    n, size = 0, w * h * 3
    try:
        while len(buf := proc.stdout.read(size)) == size:
            yield n / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            n += 1
    finally:
        proc.stdout.close()
        proc.wait()


def check_clip(path: Path, cfg: dict | None, detector: Detector | None = None,
               start: float = 0.0, facecam: dict | None = None, full_cam: bool = False) -> dict:
    """Scan the clip and decide: clean, blur these ranges, or hold for review.
    start: skip this many seconds of lead-in (ranges are then timed from there).
    facecam: the reel's facecam box; hits there are a real person. full_cam: the whole
    frame is the camera (just chatting)."""
    c = settings(cfg)
    if not c["enabled"]:
        return {"decision": "skipped", "ranges": [], "why": "nudity check disabled"}
    det = detector or load_model(c)
    if det is None:
        return unavailable(c, _model_err or "no model")
    try:
        dur = max(0.0, _probe(Path(path))[2] - start)
        scores = [(t, *frame_score(det, img, c))
                  for t, img in frames(Path(path), float(c["sample_fps"]), int(c["scan_width"]),
                                       start)]
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError) as e:
        return unavailable(c, f"{type(e).__name__}: {e}")
    if not scores:
        return unavailable(c, "no frames could be read")
    return decide(scores, dur, c, facecam, full_cam)


def blur_graph(ranges: list, src: str, dst: str) -> list[str]:
    """ffmpeg filtergraph lines from `src` to `dst` that strongly blur each range's box
    [start, end, [x, y, w, h] or None = whole frame] while the range is on screen."""
    if not ranges:
        return [f"{src}null{dst}"]
    out = [f"{src}split={len(ranges) + 1}[nz0]" + "".join(f"[nc{i}]" for i in range(len(ranges)))]
    for i, (a, b, *rest) in enumerate(ranges):
        x, y, w, h = rest[0] if rest and rest[0] else [0, 0, 1, 1]
        nxt = dst if i == len(ranges) - 1 else f"[nz{i + 1}]"
        out.append(f"[nc{i}]crop=trunc(iw*{w}/2)*2:trunc(ih*{h}/2)*2:iw*{x}:ih*{y},"
                   f"boxblur=luma_radius='min(w\\,h)/6':luma_power=3:"
                   f"chroma_radius='min(cw\\,ch)/6':chroma_power=3[nb{i}]")
        out.append(f"[nz{i}][nb{i}]overlay=main_w*{x}:main_h*{y}:"
                   f"enable='between(t\\,{a:.2f}\\,{b:.2f})'{nxt}")
    return out
