"""ClipBros brand kit as code: palette, fonts, and the logo drawn from geometry.

Source: ClipBros Brand Kit (logo, colour & type, on-clip layout).
    Ink #131119 backgrounds · Night #1D1A26 cards · Paper #F3F0E8 text/light ground
    Clip Lime #C8F03C main accent · Stream Violet #7C5CFF accent on light · Fog #A49FB2 muted
    Display/heading: Space Grotesk 700 · body: Inter 400 · labels: Inter 600 caps
    On-clip: hook Inter 800 Ink on a Paper box; captions Inter 800 caps, Paper with Ink
    outline, the spoken word in Clip Lime.
Logo rules: corner brackets frame a play button. Keep one bracket length clear on every
side; never stretch, outline or recolour the wordmark; lockup >= 120 px wide, else the
mark alone (>= 24 px). On Paper the play turns Violet.

`python -m clipfarm.brand [out_dir]` exports the logo PNGs to assets/brand/.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "assets" / "fonts"

INK = "#131119"
NIGHT = "#1D1A26"
PAPER = "#F3F0E8"
LIME = "#C8F03C"
VIOLET = "#7C5CFF"
FOG = "#A49FB2"

DISPLAY_FONT = FONTS / "SpaceGrotesk-Bold.ttf"     # wordmark, display, headings
ONCLIP_FONT = FONTS / "Inter-ExtraBold.otf"        # hook card + captions
ONCLIP_FONT_NAME = "Inter ExtraBold"

LOCKUP_MIN_W = 120
MARK_MIN = 24


def rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def ass_colour(h: str, alpha: int = 0) -> str:
    """#RRGGBB -> ASS &HAABBGGRR."""
    r, g, b = rgb(h)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _rgba(h: str) -> tuple[int, int, int, int]:
    return (*rgb(h), 255)


# ---------------------------------------------------------------------------- the mark
def draw_mark(d: ImageDraw.ImageDraw, x: float, y: float, s: float,
              bracket: str = PAPER, play: str = LIME) -> None:
    """Corner brackets + play button in an s x s box at (x, y)."""
    t = s * 0.092           # bracket stroke
    arm = s * 0.33          # bracket arm length
    r = t / 2
    col = _rgba(bracket)
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + s, y, -1, 1), (x, y + s, 1, -1), (x + s, y + s, -1, -1)):
        hx0, hx1 = sorted((cx, cx + dx * arm))
        vy0, vy1 = sorted((cy, cy + dy * arm))
        sx0, sx1 = sorted((cx, cx + dx * t))
        sy0, sy1 = sorted((cy, cy + dy * t))
        d.rounded_rectangle([hx0, sy0, hx1, sy1], radius=r, fill=col)   # horizontal arm
        d.rounded_rectangle([sx0, vy0, sx1, vy1], radius=r, fill=col)   # vertical arm
    # play triangle, slightly right of centre (optical centring)
    px0, px1 = x + s * 0.375, x + s * 0.79
    py0, py1 = y + s * 0.265, y + s * 0.735
    d.polygon([(px0, py0), (px1, (py0 + py1) / 2), (px0, py1)], fill=_rgba(play))
    d.line([(px0, py0), (px1, (py0 + py1) / 2), (px0, py1), (px0, py0)], fill=_rgba(play),
           width=max(1, round(s * 0.03)), joint="curve")


def mark(size: int, bracket: str = PAPER, play: str = LIME, ss: int = 4) -> Image.Image:
    """The mark alone on a transparent square (rendered supersampled, then downscaled)."""
    size = max(size, MARK_MIN)
    big = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    draw_mark(ImageDraw.Draw(big), 0, 0, size * ss, bracket, play)
    return big.resize((size, size), Image.LANCZOS)


def profile_picture(size: int = 1024, ss: int = 2) -> Image.Image:
    """TikTok profile picture: Lime disc, Ink mark."""
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 0, S - 1, S - 1], fill=_rgba(LIME))
    m = S * 0.54
    draw_mark(d, (S - m) / 2, (S - m) / 2, m, bracket=INK, play=INK)
    return img.resize((size, size), Image.LANCZOS)


# -------------------------------------------------------------------------- the lockup
def lockup(height: int, on: str = "ink", ss: int = 3) -> Image.Image:
    """Mark + two-tone wordmark on a transparent background.

    on="ink":   Paper brackets, Lime play, "Clip" Paper, "Bros" Lime   (primary)
    on="paper": Ink brackets, Violet play, "Clip" Ink, "Bros" Violet
    `height` is the mark height in px; the image width follows from it.
    """
    if on == "paper":
        bracket, play, clip_c, bros_c = INK, VIOLET, INK, VIOLET
    else:
        bracket, play, clip_c, bros_c = PAPER, LIME, PAPER, LIME
    s = height * ss
    font = ImageFont.truetype(str(DISPLAY_FONT), int(s * 1.17))
    gap = s * 0.34
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    clip_w = probe.textlength("Clip", font=font)
    bros_w = probe.textlength("Bros", font=font)
    kern = -s * 0.015                     # the kit sets "Clip" and "Bros" tight
    W = int(s + gap + clip_w + kern + bros_w + s * 0.02)
    H = int(s * 1.2)                     # room for the "p" descender
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    draw_mark(d, 0, 0, s, bracket, play)
    base = s * 0.875                      # text baseline: cap height sits inside the mark
    tx = s + gap
    d.text((tx, base), "Clip", font=font, fill=_rgba(clip_c), anchor="ls")
    d.text((tx + clip_w + kern, base), "Bros", font=font, fill=_rgba(bros_c), anchor="ls")
    out = img.resize((max(1, W // ss), max(1, H // ss)), Image.LANCZOS)
    if out.width < LOCKUP_MIN_W:          # kit: below 120 px wide use the mark alone
        return mark(height, bracket, play)
    return out


def clear_space(height: int) -> int:
    """One corner-bracket length, the minimum free space around the logo."""
    return round(height * 0.33)


# ---------------------------------------------------------------------------- web
def mark_svg(size: int = 28, bracket: str = "currentColor", play: str = LIME) -> str:
    """The mark as inline SVG (same geometry as draw_mark)."""
    h, a = 4.6, 33
    paths = (f"M{h} {a}V{h}H{a}", f"M{100 - a} {h}H{100 - h}V{a}",
             f"M{h} {100 - a}V{100 - h}H{a}", f"M{100 - a} {100 - h}H{100 - h}V{100 - a}")
    d = "".join(paths)
    return (f"<svg width='{size}' height='{size}' viewBox='0 0 100 100' aria-hidden='true'>"
            f"<path d='{d}' fill='none' stroke='{bracket}' stroke-width='9.2' "
            f"stroke-linecap='round' stroke-linejoin='round'/>"
            f"<path d='M37.5 26.5L79 50L37.5 73.5Z' fill='{play}' stroke='{play}' "
            f"stroke-width='3' stroke-linejoin='round'/></svg>")


def logo_html(size: int = 28) -> str:
    """Header lockup for web pages; colours follow the page theme via CSS (.logo)."""
    return (f"<a class='logo' href='{{home}}'>{mark_svg(size, 'var(--fg)', 'var(--acc)')}"
            f"<span>Clip<b>Bros</b></span></a>")


FONTS_LINK = ("<link rel='preconnect' href='https://fonts.googleapis.com'>"
              "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
              "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800"
              "&family=Space+Grotesk:wght@700&display=swap' rel='stylesheet'>")

# Ink ground with Paper text and Lime accents by default (the primary look);
# a light theme on Paper with Violet accents for viewers who prefer light.
WEB_CSS = f"""
:root{{color-scheme:dark;--bg:{INK};--card:{NIGHT};--fg:{PAPER};--mut:{FOG};--line:#2E2A3A;--acc:{LIME};--on-acc:{INK}}}
@media(prefers-color-scheme:light){{:root{{color-scheme:light;--bg:{PAPER};--card:#FFFFFF;--fg:{INK};--mut:#5E5970;--line:#DDD8CC;--acc:{VIOLET};--on-acc:#FFFFFF}}}}
body{{font:400 16px/1.6 Inter,system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:760px;margin:0 auto;padding:16px}}
h1,h2,h3{{font-family:'Space Grotesk',Inter,system-ui,sans-serif;font-weight:700;line-height:1.2;letter-spacing:-.01em}}
a{{color:var(--acc)}}
.logo{{display:inline-flex;align-items:center;gap:10px;text-decoration:none;color:var(--fg);font:700 22px/1 'Space Grotesk',Inter,sans-serif;letter-spacing:-.01em}}
.logo b{{color:var(--acc);font-weight:700}}
.label{{font:600 12px/1.4 Inter,sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--acc)}}
header{{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:8px 0 16px;border-bottom:1px solid var(--line);margin-bottom:16px}}
header nav{{margin-left:auto}}nav a{{margin-left:1rem;color:var(--mut);text-decoration:none}}nav a:hover{{color:var(--fg)}}
footer{{margin-top:3rem;color:var(--mut);font-size:14px}}
"""


def export(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "clipbros_mark.png": mark(512),
        "clipbros_lockup_ink.png": lockup(160, "ink"),
        "clipbros_lockup_paper.png": lockup(160, "paper"),
        "clipbros_profile_1024.png": profile_picture(1024),
    }
    paths = []
    for name, im in files.items():
        p = out_dir / name
        im.save(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in export(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "brand"):
        print(p)
