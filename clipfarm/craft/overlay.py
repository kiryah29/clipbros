"""ClipBros brand overlays, drawn once per reel as transparent PNGs.

Layout (1080x1920, matches the ClipBros mockup):
    - logo (icon + "ClipBros") top-left inside the safe zone
    - lime accent line across the facecam/gameplay seam
    - hook card: off-white rounded box, dark bold text, centred on the seam
Safe zone keeps everything clear of TikTok's right-side buttons and the
description/sound bar at the bottom.
"""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONT_CANDIDATES = [
    ROOT / "assets" / "fonts" / "Inter-ExtraBold.otf",
    Path("C:/Windows/Fonts/seguibl.ttf"),
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]
DEFAULT_LOGO = ROOT / "assets" / "brand" / "clipbros_icon.png"
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


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


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
    accent = hex_rgb(cfg.get("accent_color", "#C8F032"))
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if seam_y is not None and cfg.get("seam_line", True):
        t = int(cfg.get("seam_thickness", 6))
        d.rectangle([0, seam_y - t // 2, W, seam_y + t - t // 2], fill=(*accent, 255))
    logo = Path(cfg.get("logo") or DEFAULT_LOGO)
    if cfg.get("show_logo", True) and logo.exists():
        left, top, _, _ = safe_box(cfg)
        sz = 60
        icon = Image.open(logo).convert("RGBA").resize((sz, sz), Image.LANCZOS)
        mask = Image.new("L", (sz, sz), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, sz - 1, sz - 1], radius=14, fill=255)
        icon.putalpha(mask)
        x, y = left + 12, top + 12
        img.alpha_composite(icon, (x, y))
        f = _font(cfg, 40)
        d.text((x + sz + 16, y + sz // 2), cfg.get("brand_name", "ClipBros"), font=f,
               fill=(255, 255, 255, 235), anchor="lm", stroke_width=2, stroke_fill=(0, 0, 0, 120))
    img.save(out)
    return out


def draw_hook(cfg: dict, text: str, center_y: int, out: Path) -> Path:
    """Hook card: off-white rounded box with dark bold text, centred on `center_y`."""
    W, H = cfg["width"], cfg["height"]
    left, top, right, _ = safe_box(cfg)
    text = strip_emoji(text)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    card_w = int((right - left) * 0.90)
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
    cx = (left + right) // 2
    x0, y0 = cx - card_w // 2, max(top, center_y - card_h // 2)
    # soft shadow, then card
    d.rounded_rectangle([x0 + 4, y0 + 8, x0 + card_w + 4, y0 + card_h + 8], radius=26,
                        fill=(0, 0, 0, 70))
    d.rounded_rectangle([x0, y0, x0 + card_w, y0 + card_h], radius=26,
                        fill=(*hex_rgb(cfg.get("hook_bg", "#F3F1EA")), 255))
    ink = (*hex_rgb(cfg.get("hook_color", "#111111")), 255)
    for i, line in enumerate(lines):
        d.text((cx, y0 + pad_y + line_h * i + line_h // 2), line, font=f, fill=ink, anchor="mm")
    img.save(out)
    return out
