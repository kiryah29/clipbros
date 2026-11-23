"""Nudity check: scan a clip before it is rendered, blur what shows nudity, hold what is unsure.

1. Sample frames across the raw clip (ffmpeg, `sample_fps` a second).
2. Run NudeNet's exposed-body-part detector on each frame (ONNX; onnxruntime when installed,
   else OpenCV's DNN module, both run on the Windows ARM64 PC). Small figures in wide shots
   are missed at 320 px, so each frame runs at every size in `sizes` and the highest score wins.
3. Frames scoring >= blur_score become blur ranges (padded, merged); the reel blurs the
   gameplay in those ranges (craft/reel.py), the facecam, captions and hook stay sharp.
   Frames between review_score and blur_score, too much of the clip blurred, or no model
   at all hold the clip for a human look (status "held") instead of guessing.

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
    "pad_s": 0.5,
    "merge_gap_s": 1.0,
    "max_blur_share": 0.5,
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


def decide(scores: list[tuple[float, float, str]], dur: float, cfg: dict) -> dict:
    """scores: (time s, best score, label) per sampled frame. Returns the check result:
    decision clean | blur | review, blur ranges, and why."""
    c = settings(cfg)
    hits = [(t, s, lab) for t, s, lab in scores if s >= c["blur_score"]]
    ranges = merge_ranges([t for t, _, _ in hits], c["pad_s"], c["merge_gap_s"], dur)
    # unsure frames already inside a blur range are covered; the rest need a human look
    unsure = [(t, s, lab) for t, s, lab in scores if c["review_score"] <= s < c["blur_score"]
              and not any(a <= t <= b for a, b in ranges)]
    blurred = sum(b - a for a, b in ranges)
    reasons = []
    if unsure:
        reasons.append(f"{len(unsure)} frame(s) might show nudity "
                       f"({', '.join(sorted({lab for _, _, lab in unsure}))})")
    if dur and blurred / dur > c["max_blur_share"]:
        reasons.append(f"{blurred:.1f}s of {dur:.1f}s would be blurred")
    decision = "review" if reasons else ("blur" if ranges else "clean")
    return {
        "decision": decision,
        "ranges": ranges,
        "labels": sorted({lab for _, _, lab in hits}),
        "max_score": round(max((s for _, s, _ in scores), default=0.0), 3),
        "frames": len(scores),
        "unsure": [[round(t, 2), round(s, 3), lab] for t, s, lab in unsure],
        "why": "; ".join(reasons) or (f"blurred {len(ranges)} part(s): " + ", ".join(
            f"{a:.1f}-{b:.1f}s" for a, b in ranges) if ranges else "no nudity found"),
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
        return out[0].T[:, 4:]


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


def frame_score(det: Detector, img, cfg: dict) -> tuple[float, str]:
    c = settings(cfg)
    cls = [LABELS.index(k) for k in c["classes"] if k in LABELS]
    best, label = 0.0, ""
    for size in c["sizes"]:
        sc = det.scores(img, int(size))[:, cls].max(axis=0)
        i = int(sc.argmax())
        if sc[i] > best:
            best, label = float(sc[i]), c["classes"][i]
    return best, label


# ------------------------------------------------------------------ clip scan
def _probe(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    st = info["streams"][0]
    return int(st["width"]), int(st["height"]), float(info["format"].get("duration") or 0)


def frames(path: Path, fps: float, width: int):
    """(time, BGR frame) pairs, `fps` a second, from one ffmpeg pass."""
    W, H, dur = _probe(path)
    w = min(width, W) // 2 * 2
    h = max(2, int(H * w / W) // 2 * 2)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
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


def check_clip(path: Path, cfg: dict | None, detector: Detector | None = None) -> dict:
    """Scan the clip and decide: clean, blur these ranges, or hold for review."""
    c = settings(cfg)
    if not c["enabled"]:
        return {"decision": "skipped", "ranges": [], "why": "nudity check disabled"}
    det = detector or load_model(c)
    if det is None:
        return unavailable(c, _model_err or "no model")
    try:
        dur = _probe(Path(path))[2]
        scores = [(t, *frame_score(det, img, c))
                  for t, img in frames(Path(path), float(c["sample_fps"]), int(c["scan_width"]))]
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError) as e:
        return unavailable(c, f"{type(e).__name__}: {e}")
    if not scores:
        return unavailable(c, "no frames could be read")
    return decide(scores, dur, c)


def blur_filter(ranges: list[list[float]]) -> str:
    """ffmpeg filter that strongly blurs the whole frame inside `ranges` (empty = none)."""
    if not ranges:
        return ""
    when = "+".join(f"between(t\\,{a:.2f}\\,{b:.2f})" for a, b in ranges)
    return (f"boxblur=luma_radius='min(w\\,h)/10':luma_power=3:"
            f"chroma_radius='min(cw\\,ch)/10':chroma_power=3:enable='{when}'")
