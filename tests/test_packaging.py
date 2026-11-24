"""Clip timing anchored on the audio peak, and the gameplay band crop."""
from clipfarm.audio import anchor_on_peak
from clipfarm.craft.reel import gameplay_crop

C = {"min_length_s": 15, "max_length_s": 30,
     "anchor": {"pre_s": 4, "post_s": 7, "fade_ratio": 0.5, "fade_gap_s": 2}}


def levels(seconds: float, loud: dict[float, float]) -> list[float]:
    """0.05 background at 4 windows/s, with {second: level} for each loud 250 ms."""
    out = [0.05] * int(seconds * 4)
    for t, v in loud.items():
        out[int(t * 4)] = v
    return out


def test_clip_starts_4s_before_the_loudest_moment():
    # chat peaked ~25 s into the old window; the scream was at 18 s
    start, end, peak = anchor_on_peak(levels(45, {18: 0.9, 30: 0.3}), 0.25, 0, C)
    assert (start, peak) == (14.0, 18.0)
    assert end - start == 15                       # short reaction: padded to min length


def test_clip_ends_7s_after_the_reaction_fades():
    lv = levels(60, {10 + i / 4: 0.6 for i in range(36)})   # 9 s of yelling from 10 s
    lv[int(12 * 4)] = 0.9
    start, end, peak = anchor_on_peak(lv, 0.25, 0, C)
    assert (start, peak) == (8.0, 12.0)
    assert end == 19.0 + 7                         # yelling ends at 19 s


def test_long_reactions_are_capped_at_max_length():
    lv = levels(80, {5 + i / 4: 0.6 for i in range(240)})
    start, end, _ = anchor_on_peak(lv, 0.25, 0, C)
    assert end - start == 30


def test_search_starts_at_lo_and_short_files_give_up():
    start, _, peak = anchor_on_peak(levels(40, {2: 1.0, 20: 0.5}), 0.25, 5, C)
    assert peak == 20.0
    assert anchor_on_peak([0.1], 0.25, 0, C) is None


def test_gameplay_crop_slides_off_a_left_facecam():
    crop, mask = gameplay_crop({"x": 0.0, "y": 0.355, "w": 0.225, "h": 0.28}, 1080, 752, 1.25)
    x = float(crop.split(":")[2].split("*")[1])
    assert x >= 0.225 and mask is None


def test_gameplay_crop_masks_a_facecam_it_cannot_avoid():
    centre = {"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2}
    assert gameplay_crop(centre, 1080, 752, 1.25)[1] == [0.4, 0.4, 0.2, 0.2]
