"""ClipBros brand overlays, drawn once per reel as transparent PNGs.

Layout (1080x1920, follows the ClipBros Brand Kit on-clip page; tokens in clipfarm/brand.py):
    - primary lockup (mark + "Clip"/"Bros" wordmark) top-left, with bracket-length clear space
    - lime accent line across the facecam/gameplay seam
    - hook card: off-white rounded box, dark bold text, centred on the seam
Safe zone keeps everything clear of TikTok's right-side buttons and the
description/sound bar at the bottom.
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .. import brand

ROOT = Path(__file__).resolve().parents[2]
FONT_CANDIDATES = [
    brand.ONCLIP_FONT,
    Path("C:/Windows/Fonts/seguibl.ttf"),
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]
# the brand font has no emoji glyphs; drop them from burned-in text (keep them in the post caption)
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+")


def strip_emoji(text: str) -> str:
    return re.sub(r"\s{2,}", " ", EMOJI_RE.sub("", text)).strip()


def font_path(cfg: dict) -> Path | None:
    for p in [cfg.get("font_file"), *FONT_CANDIDATES]:
        if p and Path(p).exists():
            return Path(p)
    return None


def _font(cfg: dict, size: int) -> ImageFont.ImageFont:
    p = font_path(cfg)
    return ImageFont.truetype(str(p), size) if p else ImageFont.load_default()


hex_rgb = brand.rgb


def safe_box(cfg: dict) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) pixel box that TikTok's UI never covers."""
    W, H = cfg["width"], cfg["height"]
    s = {"left": 62, "top": 112, "right": 124, "bottom": 320, **(cfg.get("safe_zone") or {})}
    return s["left"], s["top"], W - s["right"], H - s["bottom"]


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_base(cfg: dict, seam_y: int | None, out: Path) -> Path:
    """Always-on layer: logo + lime seam line."""
    W, H = cfg["width"], cfg["height"]
    accent = hex_rgb(cfg.get("accent_color", brand.LIME))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if seam_y is not None and cfg.get("seam_line", True):
        t = int(cfg.get("seam_thickness", 6))
        d.rectangle([0, seam_y - t // 2, W, seam_y + t - t // 2], fill=(*accent, 255))
    if cfg.get("show_logo", True):
        _place_logo(img, cfg)
    img.save(out)
    return out


def _place_logo(img: Image.Image, cfg: dict) -> None:
    """Primary lockup top-left. A custom `logo` PNG replaces it untouched (never recoloured).

    The kit forbids outlining the wordmark, so legibility over busy facecams comes from a
    soft Ink shadow only."""
    left, top, _, _ = safe_box(cfg)
    h = int(cfg.get("logo_height", 44))           # mark height; lockup is ~5x wider
    custom = cfg.get("logo")
    if custom and Path(custom).exists():
        logo = Image.open(custom).convert("RGBA")
        logo = logo.resize((max(1, round(logo.width * h / logo.height)), h), Image.LANCZOS)
    else:
        logo = brand.lockup(h, "ink")
    pad = brand.clear_space(h)
    x, y = left + pad, top + pad
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ink = (*hex_rgb(brand.INK), 0)
    sh = Image.new("RGBA", logo.size, ink)
    sh.putalpha(logo.getchannel("A").point(lambda a: a * 0.55))
    shadow.alpha_composite(sh, (x + 2, y + 3))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(6)))
    img.alpha_composite(logo, (x, y))


def draw_hook(cfg: dict, text: str, center_y: int, out: Path) -> Path:
    """Hook card: Paper rounded box with Ink Inter 800 text, centred on `center_y`."""
    W, H = cfg["width"], cfg["height"]
    left, top, right, _ = safe_box(cfg)
    text = strip_emoji(text)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = W // 2   # centred on screen (the safe box is lopsided), still clear of the side buttons
    card_w = min(int((right - left) * 0.90), 2 * min(right - cx, cx - left))
    pad_x, pad_y = 44, 30
    size = 68
    while True:  # shrink until it fits in 3 lines
        f = _font(cfg, size)
        lines = _wrap(d, text, f, card_w - 2 * pad_x)
        if len(lines) <= 3 or size <= 44:
            break
        size -= 4
    line_h = int(size * 1.18)
    card_h = line_h * len(lines) + 2 * pad_y
    x0, y0 = cx - card_w // 2, max(top, center_y - card_h // 2)
    # soft shadow, then card
    d.rounded_rectangle([x0, y0 + 6, x0 + card_w, y0 + card_h + 6], radius=20,
                        fill=(*hex_rgb(brand.INK), 80))
    d.rounded_rectangle([x0, y0, x0 + card_w, y0 + card_h], radius=20,
                        fill=(*hex_rgb(cfg.get("hook_bg", brand.PAPER)), 255))
    ink = (*hex_rgb(cfg.get("hook_color", brand.INK)), 255)
    for i, line in enumerate(lines):
        d.text((cx, y0 + pad_y + line_h * i + line_h // 2), line, font=f, fill=ink, anchor="mm")
    img.save(out)
    return out
