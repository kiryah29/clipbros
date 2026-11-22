from clipfarm import brand
from clipfarm.craft.captions import build_ass, fake_words
from clipfarm.craft.overlay import draw_base


def test_palette_matches_brand_kit():
    assert (brand.INK, brand.NIGHT, brand.PAPER) == ("#131119", "#1D1A26", "#F3F0E8")
    assert (brand.LIME, brand.VIOLET, brand.FOG) == ("#C8F03C", "#7C5CFF", "#A49FB2")
    assert brand.ass_colour(brand.LIME) == "&H003CF0C8"   # ASS is &HAABBGGRR


def test_logo_rules():
    big = brand.lockup(44)
    assert big.width >= brand.LOCKUP_MIN_W and big.height > 44
    small = brand.lockup(10)                      # too small for the lockup: mark alone
    assert small.size[0] == small.size[1] >= brand.MARK_MIN
    # primary lockup: Lime play button inside the mark
    px = brand.mark(200).getpixel((110, 100))
    assert px[:3] == brand.rgb(brand.LIME)


def test_captions_use_brand_colours(tmp_path):
    cfg = {"width": 1080, "height": 1920, "caption_style": "clipbros", "cta_text": "",
           "caption_words": 4}
    ass = build_ass(fake_words("bro turned it off", 0), 3, cfg, 1150, tmp_path / "c.ass").read_text()
    paper, lime, ink = (brand.ass_colour(c) for c in (brand.PAPER, brand.LIME, brand.INK))
    assert f"{paper},{lime},{ink}" in ass
    assert "BRO" in ass and "OFF" in ass.splitlines()[-1]   # all four words on one line


def test_logo_keeps_clear_space(tmp_path):
    cfg = {"width": 1080, "height": 1920, "safe_zone": {"left": 62, "top": 112, "right": 124, "bottom": 320}}
    from PIL import Image
    box = Image.open(draw_base(cfg, None, tmp_path / "b.png")).getchannel("A").point(lambda a: 255 if a > 200 else 0).getbbox()
    pad = brand.clear_space(44)
    assert box[0] >= 62 + pad - 1 and box[1] >= 112 + pad - 1
