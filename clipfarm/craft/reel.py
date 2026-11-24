"""Turn a 16:9 raw clip into a 9:16 reel with ffmpeg.

facecam_top layout (top to bottom):
    [ facecam band ]  cropped from the source using the channel's facecam box
    [ gameplay     ]  tight centre crop of the source, slid off the facecam (never twice)
    [ backdrop     ]  optional looping filler footage (else gameplay grows)
    With chat_strip on and chat in the receipt, the backdrop is replaced by a solid
    Twitch chat strip that ends where TikTok's description bar starts (chatstrip.py).
full_cam layout (just chatting, the camera is the stream): one full-height crop on the
streamer. gameplay_only: the whole 16:9 frame over a blurred fill.
Audio: light compressor + two-pass loudnorm to -14 LUFS.
Brand layer (logo, lime seam line, hook card) is drawn by overlay.py as PNGs;
word captions/CTA are burned in from an ASS file; optional music is ducked under.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .captions import Word, build_ass
from .chatstrip import chat_ass
from ..nudity import blur_graph
from .overlay import draw_base, draw_hook, font_path, safe_box


def probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def _even(x: float) -> int:
    return int(x) // 2 * 2


def _ff_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def has_audio(path: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "csv=p=0", str(path)], capture_output=True,
                         text=True)
    return bool(out.stdout.strip())


def loudnorm_filter(raw: Path, lead: float, dur: float, cfg: dict) -> str:
    """Light compressor (screams don't clip) + loudnorm to `loudness` (-14 LUFS like most
    TikToks). Two-pass: measure the clip, then normalise it linearly; if the measurement
    fails, one-pass loudnorm still lands close."""
    comp = cfg.get("compressor") or "acompressor=threshold=-12dB:ratio=2:attack=5:release=150"
    ln = {"I": -14, "TP": -1.5, "LRA": 11, **(cfg.get("loudness") or {})}
    base = f"loudnorm=I={ln['I']}:TP={ln['TP']}:LRA={ln['LRA']}"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{lead:.3f}", "-t",
                        f"{dur:.2f}", "-i", str(raw), "-vn", "-af",
                        f"{comp},{base}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    try:
        m = json.loads(r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1])
        base += (f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
                 f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
                 f":offset={m['target_offset']}:linear=true")
    except (ValueError, KeyError):
        pass
    return f"{comp},{base},aresample=48000"


def gameplay_crop(fc: dict | None, W: int, gh: int, zoom: float, aspect: float = 16 / 9
                  ) -> tuple[str, list | None]:
    """(crop/scale filter for the gameplay band, facecam box still inside that crop or None).
    The crop is `zoom` x tighter than filling the band (HUD, minimap and the streamer's chat
    overlay sit at the frame edges) and slides sideways off the facecam, so the face doesn't
    show twice; a facecam it can't get away from is returned so it can be masked."""
    ch = 1 / max(float(zoom), 1.0)
    cw = ch * (W / gh) / aspect
    if cw > 1:
        cw, ch = 1.0, aspect * gh / W
    cx, cy = (1 - cw) / 2, (1 - ch) / 2

    def overlaps(x: float) -> bool:
        return bool(fc) and fc["x"] < x + cw and fc["x"] + fc["w"] > x and \
            fc["y"] < cy + ch and fc["y"] + fc["h"] > cy

    if overlaps(cx):
        if fc["x"] + fc["w"] / 2 < 0.5:
            cx = min(max(cx, fc["x"] + fc["w"]), 1 - cw)
        else:
            cx = max(min(cx, fc["x"] - cw), 0.0)
    crop = (f"crop=iw*{cw:.4f}:ih*{ch:.4f}:iw*{cx:.4f}:ih*{cy:.4f},"
            f"scale={W}:{gh}:force_original_aspect_ratio=increase,crop={W}:{gh}")
    return crop, ([fc["x"], fc["y"], fc["w"], fc["h"]] if overlaps(cx) else None)


def render_reel(raw: Path, out: Path, cfg: dict, facecam: dict | None,
                words: list[Word], hook_text: str | None = None,
                chat: list | None = None, censor: list | None = None,
                lead: float = 0.0, length: float | None = None) -> Path:
    """censor: [start, end, box] parts blurred by the nudity check (box = [x, y, w, h]
    fractions of the source, None = whole frame); hook card and captions stay sharp.
    lead: seconds at the start of `raw` before the clip window (stream-copied cuts start
    on the keyframe before the moment). They are trimmed frame-accurately here, so all
    times passed in (words, chat, censor) are relative to the clip window, not to raw.
    length: clip length in seconds (default: the rest of raw)."""
    W, H = cfg["width"], cfg["height"]
    lead = max(0.0, float(lead or 0))
    dur = probe_duration(raw) - lead
    if length:
        dur = min(dur, float(length))
    backdrop = cfg.get("backdrop")
    music = cfg.get("music")
    strip = bool(chat) and cfg.get("chat_strip", False) and \
        cfg["layout"] == "facecam_top" and bool(facecam)
    has_back = bool(backdrop) and Path(backdrop).exists() and not strip
    has_music = bool(music) and Path(music).exists()

    inputs = (["-ss", f"{lead:.3f}"] if lead else []) + ["-i", str(raw)]
    if has_back:
        inputs += ["-stream_loop", "-1", "-i", str(backdrop)]
    if has_music:
        inputs += ["-stream_loop", "-1", "-i", str(music)]

    f = blur_graph(censor or [], "[0:v]", "[src]")
    f[0] = "[0:v]fps=30," + f[0][len("[0:v]"):]
    if cfg["layout"] == "full_cam" and facecam:
        # one full-height 9:16 crop, centred on the streamer
        cx = facecam["x"] + facecam["w"] / 2
        f.append(f"[src]crop=trunc(ih*{W}/{H}/2)*2:ih:'min(max(iw*{cx:.4f}-ow/2,0),iw-ow)':0,"
                 f"scale={W}:{H},setsar=1[stacked]")
        seam_y = None
        frame_top = safe_box(cfg)[1] + 250   # hook card sits at the top, under TikTok's header
        logo_bottom = safe_box(cfg)[3]
        caption_y = int(H * cfg.get("caption_y_ratio", 0.60))
    elif cfg["layout"] == "facecam_top" and facecam:
        fh = _even(H * cfg["facecam_band_ratio"])
        gh = _even(H * cfg["gameplay_band_ratio"]) if has_back else H - fh
        if strip:   # chat strip sits right above TikTok's description bar
            strip_top = _even(safe_box(cfg)[3] - int(cfg.get("chat_strip_height", 272)))
            gh = strip_top - fh
        bh = H - fh - gh
        fc = facecam
        f.append("[src]split=2[src1][src2]")
        f.append(f"[src1]crop=iw*{fc['w']}:ih*{fc['h']}:iw*{fc['x']}:ih*{fc['y']},"
                 f"scale={W}:{fh}:force_original_aspect_ratio=increase,"
                 f"crop={W}:{fh}:(iw-ow)/2:(ih-oh)*0.3,setsar=1[face]")   # faces sit high: keep headroom
        crop, mask = gameplay_crop(fc, W, gh, cfg.get("gameplay_zoom", 1.0))
        if mask:   # facecam inside the gameplay crop: blur it out so the face shows once
            x, y, w, h = mask
            f.append(f"[src2]split[gm0][gm1];[gm1]crop=trunc(iw*{w}/2)*2:trunc(ih*{h}/2)*2:"
                     f"iw*{x}:ih*{y},boxblur=20:3[gmb];[gm0][gmb]overlay=main_w*{x}:main_h*{y}[gsrc]")
            f.append(f"[gsrc]{crop},setsar=1[game]")
        else:
            f.append(f"[src2]{crop},setsar=1[game]")
        stack = ["[face]", "[game]"]
        if has_back:
            f.append(f"[1:v]scale={W}:{bh}:force_original_aspect_ratio=increase,"
                     f"crop={W}:{bh},setsar=1[back]")
            stack.append("[back]")
        if strip:
            accent = cfg.get("accent_color", "#C8F03C").lstrip("#")
            f.append("".join(stack) + f"vstack=inputs={len(stack)},"
                     f"pad={W}:{H}:0:0:color=0x{cfg.get('chat_strip_bg', '#1D1A26').lstrip('#')},"
                     f"drawbox=x=0:y={strip_top}:w={W}:h=4:color=0x{accent}:t=fill[stacked]")
        else:
            f.append("".join(stack) + f"vstack=inputs={len(stack)}[stacked]")
        seam_y = fh
        logo_bottom = fh + gh
        caption_y = fh + gh - 260 if has_back else int(H * cfg.get("caption_y_ratio", 0.60))
        if strip:
            caption_y = min(caption_y, strip_top - 110)
    else:
        # gameplay_only: blurred fill behind a centered 16:9 frame
        f.append("[src]split=2[bg][fg]")
        # blur a quarter-size copy and scale it back up: same look as boxblur=20:2 at full
        # size for ~1/16 of the pixels (the full-size blur was half the render time)
        qw, qh = _even(W / 4), _even(H / 4)
        f.append(f"[bg]scale={qw}:{qh}:force_original_aspect_ratio=increase,crop={qw}:{qh},"
                 f"boxblur=5:2,scale={W}:{H}:flags=bilinear[bgb]")
        f.append(f"[fg]scale={W}:-2[fgs]")
        f.append("[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[stacked]")
        seam_y = None
        frame_top = (H - _even(W * 9 / 16)) // 2   # sources are 16:9
        logo_bottom = frame_top + _even(W * 9 / 16)
        caption_y = int(H * 0.68)

    # brand layer: logo + seam line always, hook card from frame one
    left, _, right, _ = safe_box(cfg)
    n_in = 1 + has_back + has_music
    base_png = draw_base(cfg, seam_y, out.with_name(out.stem + "_base.png"),
                         min(logo_bottom, safe_box(cfg)[3]))
    inputs += ["-loop", "1", "-i", str(base_png)]
    f.append(f"[stacked][{n_in}:v]overlay=0:0:shortest=1[b1]")
    last = "[b1]"
    hook = cfg.get("hook_text") if hook_text is None else hook_text
    if hook:
        # facecam_top: card straddles the seam; gameplay_only: card sits just above the frame
        hook_png = draw_hook(cfg, hook, seam_y or int(H * 0.30),
                             out.with_name(out.stem + "_hook.png"),
                             bottom_y=None if seam_y else frame_top - 24)
        inputs += ["-loop", "1", "-i", str(hook_png)]
        hs = float(cfg.get("hook_seconds") or 0)   # 0 = keep the hook card up the whole clip
        enable = f":enable='lt(t,{hs})'" if hs > 0 else ""
        f.append(f"{last}[{n_in + 1}:v]overlay=0:0:shortest=1{enable}[b2]")
        last = "[b2]"

    chat_styles, chat_events = [], []
    if strip:
        chat_styles, chat_events = chat_ass(
            chat, dur, cfg, (left + 18, strip_top + 26, right, strip_top +
                             int(cfg.get("chat_strip_height", 272)) - 22))
    ass = out.with_suffix(".ass")
    build_ass(words, dur, cfg, caption_y, ass, caption_x=(left + right) // 2,
              max_w=right - left, extra_styles=chat_styles, extra_events=chat_events)
    fp = font_path(cfg)
    fonts = f":fontsdir='{_ff_path(fp.parent)}'" if fp else ""
    f.append(f"{last}ass='{_ff_path(ass)}'{fonts}[v]")

    if has_music:
        mi = 2 if has_back else 1
        f.append(f"[{mi}:a]volume={cfg['music_volume']}[m];[0:a][m]amix=inputs=2:duration=first,"
                 f"{loudnorm_filter(raw, lead, dur, cfg)}[a]")
        amap = "[a]"
    elif has_audio(raw):
        f.append(f"[0:a]{loudnorm_filter(raw, lead, dur, cfg)}[a]")
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
