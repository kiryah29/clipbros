"""Turn a 16:9 raw clip into a 9:16 reel with ffmpeg.

facecam_top layout (top to bottom):
    [ facecam band ]  cropped from the source using the channel's facecam box
    [ gameplay     ]  center crop of the source
    [ backdrop     ]  optional looping filler footage (else gameplay grows)
Captions/hook/CTA are burned in from an ASS file; optional music is ducked under.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .captions import Word, build_ass


def probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def _even(x: float) -> int:
    return int(x) // 2 * 2


def render_reel(raw: Path, out: Path, cfg: dict, facecam: dict | None,
                words: list[Word]) -> Path:
    W, H = cfg["width"], cfg["height"]
    dur = probe_duration(raw)
    backdrop = cfg.get("backdrop")
    music = cfg.get("music")
    has_back = bool(backdrop) and Path(backdrop).exists()
    has_music = bool(music) and Path(music).exists()

    inputs = ["-i", str(raw)]
    if has_back:
        inputs += ["-stream_loop", "-1", "-i", str(backdrop)]
    if has_music:
        inputs += ["-stream_loop", "-1", "-i", str(music)]

    f = []
    if cfg["layout"] == "facecam_top" and facecam:
        fh = _even(H * cfg["facecam_band_ratio"])
        gh = _even(H * cfg["gameplay_band_ratio"]) if has_back else H - fh
        bh = H - fh - gh
        fc = facecam
        f.append("[0:v]split=2[src1][src2]")
        f.append(f"[src1]crop=iw*{fc['w']}:ih*{fc['h']}:iw*{fc['x']}:ih*{fc['y']},"
                 f"scale={W}:{fh}:force_original_aspect_ratio=increase,crop={W}:{fh},setsar=1[face]")
        f.append(f"[src2]scale=-2:{gh},crop={W}:{gh},setsar=1[game]")
        stack = ["[face]", "[game]"]
        if has_back:
            f.append(f"[1:v]scale={W}:{bh}:force_original_aspect_ratio=increase,"
                     f"crop={W}:{bh},setsar=1[back]")
            stack.append("[back]")
        f.append("".join(stack) + f"vstack=inputs={len(stack)}[stacked]")
        caption_y = fh + gh - 260 if has_back else fh + int(gh * 0.62)
    else:
        # gameplay_only: blurred fill behind a centered 16:9 frame
        f.append("[0:v]split=2[bg][fg]")
        f.append(f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=20:2[bgb]")
        f.append(f"[fg]scale={W}:-2[fgs]")
        f.append("[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[stacked]")
        caption_y = int(H * 0.68)

    ass = out.with_suffix(".ass")
    build_ass(words, dur, cfg, caption_y, ass)
    ass_arg = str(ass.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    f.append(f"[stacked]ass='{ass_arg}'[v]")

    if has_music:
        mi = 2 if has_back else 1
        f.append(f"[{mi}:a]volume={cfg['music_volume']}[m];[0:a][m]amix=inputs=2:duration=first[a]")
        amap = "[a]"
    else:
        amap = "0:a?"

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(f),
           "-map", "[v]", "-map", amap, "-t", f"{dur:.2f}", "-r", "30",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out
