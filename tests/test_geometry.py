"""The proportion math. These are the numbers every rule leans on."""

from __future__ import annotations

import pytest

from silhouette.geometry import compute
from silhouette.models import Outfit


def feat(cv, **kw):
    return compute(Outfit(**kw), cv.frame, cv.height_cm)


def test_true_leg_ratio_is_anatomy_only(cv, W):
    """Shoes and clothes must never change the anatomical reference."""
    bare = feat(cv, top=None, bottom=None)
    clothes = dict(top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"])
    flat = feat(cv, **clothes)
    heeled = feat(cv, **clothes, shoe=W["shoe_heel_blk_block"])
    assert W["shoe_heel_blk_block"].heel_cm > 0
    assert bare.true_leg == flat.true_leg == heeled.true_leg == pytest.approx(0.5)
    # the heel must still move what the eye reads, or this test proves nothing
    assert heeled.lift != pytest.approx(flat.lift)


def test_outfit_label_handles_empty_and_missing_top_when_tucked(W):
    """tucked=True must annotate the top only, never crash or label the bottom."""
    assert Outfit(tucked=True).label() == ""
    no_top = Outfit(bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"], tucked=True)
    assert "(tucked)" not in no_top.label()
    with_top = Outfit(
        top=W["top_blouse_struct_cream"],
        bottom=W["bot_wide_blk"],
        shoe=W["shoe_sneaker_white"],
        tucked=True,
    )
    assert with_top.label().startswith(f"{W['top_blouse_struct_cream'].name} (tucked)")


def test_rise_orders_break_ratio_monotonically(cv, W):
    """Higher rise -> higher break -> more lift. The core claim, as an ordering."""
    rises = ["bot_jean_low_wash", "bot_jean_bootcut_mid", "bot_jean_straight_indigo"]
    lifts = [
        feat(cv, top=W["top_tee_blk"], bottom=W[b], shoe=W["shoe_sneaker_white"]).lift
        for b in rises
    ]
    assert lifts == sorted(lifts), f"lift should increase with rise, got {lifts}"
    assert lifts[0] < lifts[-1]


def test_super_high_beats_high(cv, W):
    hi = feat(cv, top=W["top_crop_rib_ivory"], bottom=W["bot_wide_cream"], shoe=W["shoe_sneaker_white"])
    sup = feat(cv, top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    assert sup.lift > hi.lift


def test_untucked_hip_length_top_reads_shorter_than_anatomy(cv, W):
    """The finding that makes the metric worth having: an untucked hip-length
    top over a high rise does not merely waste the rise, it reads shorter than
    the body actually is."""
    f = feat(cv, top=W["top_blouse_struct_cream"], bottom=W["bot_wide_cream"], shoe=W["shoe_sneaker_white"])
    assert f.break_source == "top_hem"
    assert f.hem_drop > 0.0
    assert f.lift < 0.0


def test_tucking_recovers_the_rise(cv, W):
    """Same two garments, one styling decision, measurably different."""
    kw = dict(top=W["top_blouse_struct_cream"], bottom=W["bot_wide_cream"], shoe=W["shoe_sneaker_white"])
    loose = feat(cv, **kw)
    tucked = feat(cv, tucked=True, **kw)
    assert tucked.break_source == "waistband"
    assert tucked.hem_drop == 0.0
    assert tucked.lift > loose.lift
    assert tucked.lift - loose.lift > 0.10


def test_tucking_a_non_tuckable_top_does_nothing(cv, W):
    kw = dict(top=W["top_hoodie_oversized_grey"], bottom=W["bot_wide_cream"], shoe=W["shoe_sneaker_white"])
    assert feat(cv, **kw).lift == feat(cv, tucked=True, **kw).lift


def test_heel_increases_lift_without_moving_the_break_point(cv, W):
    flat = feat(cv, top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"], shoe=W["shoe_sandal_tan_flat"])
    heel = feat(cv, top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"], shoe=W["shoe_heel_blk_block"])
    assert heel.lift > flat.lift
    # the waistband has not moved on the body; it is a smaller fraction of a
    # taller total, so break_ratio goes DOWN while lift goes UP
    assert heel.break_ratio < flat.break_ratio
    assert heel.height_total > flat.height_total


def test_draped_garment_has_no_break(cv, W):
    f = feat(cv, dress=W["dress_saree_teal"], shoe=W["shoe_heel_blk_block"])
    assert f.is_column
    assert f.break_ratio is None
    assert f.break_source == "none"
    assert f.lift == 0.0


def test_empire_seam_gives_the_highest_break_available(cv, W):
    f = feat(cv, dress=W["dress_anarkali_maroon"], shoe=W["shoe_sandal_tan_flat"])
    assert f.break_source == "dress_seam"
    assert f.break_ratio < 0.35
    assert f.lift > 0.15


def test_dress_uses_its_own_two_volume_zones(cv, W):
    f = feat(cv, dress=W["dress_anarkali_maroon"], shoe=W["shoe_sandal_tan_flat"])
    assert f.upper_volume == pytest.approx(0.95)
    assert f.lower_volume == pytest.approx(1.90)
    assert f.asymmetry < 0


def test_outer_layer_counts_toward_upper_volume(cv, W):
    bare = feat(cv, top=W["top_tee_blk"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    layered = feat(cv, top=W["top_tee_blk"], bottom=W["bot_wide_blk"],
                   outer=W["out_jacket_denim"], shoe=W["shoe_sneaker_white"])
    assert layered.upper_volume > bare.upper_volume
    assert layered.upper_structure > bare.upper_structure


def test_contrast_is_measured_between_the_two_garments_at_the_break(cv, W):
    mono = feat(cv, top=W["top_tee_blk"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    split = feat(cv, top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    assert mono.contrast_at_break < 2.0
    assert split.contrast_at_break > 70.0


def test_totals_are_summed_over_worn_items(cv, W):
    f = feat(cv, top=W["top_tee_blk"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    assert f.total_clo == pytest.approx(0.12 + 0.30 + 0.05)
    assert f.min_formality == pytest.approx(0.30)
    assert f.max_formality == pytest.approx(0.60)
    assert f.pattern_count == 0
