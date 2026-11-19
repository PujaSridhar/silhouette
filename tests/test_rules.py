"""One fixture per rule.

Rules are the product. They are data, so they need tests exactly as much as
code does - arguably more, since the person editing them may not write Python.

Every soft rule gets a fixture that must fire it, and most get one that must
not. The coverage test at the bottom fails if a rule exists that no test
exercises, which is what stops the rule set rotting as it grows.
"""

from __future__ import annotations

import dataclasses

import pytest

from silhouette.models import Outfit

# --------------------------------------------------------------- proportion


def test_high_break_elongates_fires_on_a_tucked_high_rise(fired_ids, W):
    o = Outfit(top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    assert "high_break_elongates" in fired_ids(o, "weekend")


def test_high_break_elongates_does_not_fire_on_a_low_rise(fired_ids, W):
    o = Outfit(top=W["top_tee_blk"], bottom=W["bot_jean_low_wash"], shoe=W["shoe_sneaker_white"])
    assert "high_break_elongates" not in fired_ids(o, "weekend")


def test_untucked_hem_cancels_the_rise(fired_ids, verdict_for, W):
    o = Outfit(top=W["top_blouse_struct_cream"], bottom=W["bot_wide_cream"], shoe=W["shoe_sneaker_white"])
    ids = fired_ids(o, "campus")
    assert "untucked_hem_cancels_the_rise" in ids
    assert "high_break_elongates" not in ids, "cannot both waste and exploit the rise"
    _f, v = verdict_for(o, "campus")
    assert any(fr.contribution < 0 for fr in v.fired if fr.rule.id == "untucked_hem_cancels_the_rise")


def test_tucking_flips_the_penalty_into_a_bonus(fired_ids, W):
    kw = dict(top=W["top_blouse_struct_cream"], bottom=W["bot_wide_cream"], shoe=W["shoe_sneaker_white"])
    assert "untucked_hem_cancels_the_rise" in fired_ids(Outfit(**kw), "campus")
    assert "high_break_elongates" in fired_ids(Outfit(tucked=True, **kw), "campus")


def test_heel_rule_needs_a_real_heel(fired_ids, W):
    kw = dict(top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"])
    assert "heel_extends_the_leg_line" in fired_ids(Outfit(shoe=W["shoe_heel_blk_block"], **kw), "weekend")
    assert "heel_extends_the_leg_line" not in fired_ids(Outfit(shoe=W["shoe_sandal_tan_flat"], **kw), "weekend")


def test_long_outer_adds_vertical_line(fired_ids, W):
    kw = dict(top=W["top_tee_blk"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    assert "long_outer_adds_vertical_line" in fired_ids(Outfit(outer=W["out_duster_camel"], **kw), "weekend")
    assert "long_outer_adds_vertical_line" not in fired_ids(Outfit(outer=W["out_blazer_blk"], **kw), "weekend")


def test_drape_reads_as_one_column(fired_ids, W):
    assert "drape_reads_as_one_column" in fired_ids(
        Outfit(dress=W["dress_saree_teal"], shoe=W["shoe_heel_blk_block"]), "wedding_function"
    )
    # an empire seam IS a break, so the column rule must stay quiet
    assert "drape_reads_as_one_column" not in fired_ids(
        Outfit(dress=W["dress_anarkali_maroon"], shoe=W["shoe_heel_blk_block"]), "wedding_function"
    )


# ------------------------------------------------------------------ balance


def test_fitted_top_counterweights_volume(fired_ids, W):
    o = Outfit(top=W["top_tee_blk"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    assert "fitted_top_counterweights_volume" in fired_ids(o, "weekend")


def test_counterweight_rule_needs_volume_below(fired_ids, W):
    o = Outfit(top=W["top_tee_blk"], bottom=W["bot_skirt_pencil_blk"], shoe=W["shoe_sneaker_white"])
    assert "fitted_top_counterweights_volume" not in fired_ids(o, "office")


def test_volume_on_volume_loses_the_line(fired_ids, W):
    o = Outfit(top=W["top_hoodie_oversized_grey"], bottom=W["bot_palazzo_floral"], shoe=W["shoe_sneaker_white"])
    ids = fired_ids(o, "weekend")
    assert "volume_on_volume_loses_the_line" in ids
    assert "fitted_top_counterweights_volume" not in ids


def test_structure_above_balances_volume_below(fired_ids, W):
    """Puja's subtle rule: structure, not fabric, is the counterweight."""
    o = Outfit(top=W["top_blouse_puff_rust"], bottom=W["bot_wide_blk"], shoe=W["shoe_heel_blk_block"])
    ids = fired_ids(o, "office")
    assert "structure_above_balances_volume_below" in ids
    assert "flat_upper_under_high_volume" not in ids


def test_flat_upper_under_high_volume_is_the_mirror(fired_ids, verdict_for, W):
    o = Outfit(top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    ids = fired_ids(o, "weekend")
    assert "flat_upper_under_high_volume" in ids
    assert "structure_above_balances_volume_below" not in ids
    # deliberately mild - advice, not a veto
    _f, v = verdict_for(o, "weekend")
    pen = next(fr for fr in v.fired if fr.rule.id == "flat_upper_under_high_volume")
    assert -0.60 < pen.contribution < 0.0
    assert v.total > 1.0, "the outfit must still read as good overall"


# ------------------------------------------------------------------- colour


def test_monochrome_column_rescues_central_break(fired_ids, W):
    o = Outfit(top=W["top_blouse_struct_cream"], bottom=W["bot_wide_cream"], shoe=W["shoe_sneaker_white"])
    ids = fired_ids(o, "campus")
    assert "monochrome_column_rescues_central_break" in ids
    assert "high_contrast_at_central_break" not in ids


def test_high_contrast_at_central_break(fired_ids, W):
    o = Outfit(top=W["top_tee_blk"], bottom=W["bot_jean_bootcut_mid"], shoe=W["shoe_sneaker_white"])
    ids = fired_ids(o, "weekend")
    assert "high_contrast_at_central_break" in ids
    assert "monochrome_column_rescues_central_break" not in ids


def test_contrast_sharpens_a_high_break(fired_ids, W):
    """The interaction: the same contrast is good high and bad in the middle."""
    o = Outfit(top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    ids = fired_ids(o, "weekend")
    assert "contrast_sharpens_a_high_break" in ids
    assert "high_contrast_at_central_break" not in ids


def test_pattern_competes_with_pattern(fired_ids, W):
    """Needs a synthetic fixture - see test_wardrobe_cannot_trigger_every_rule."""
    printed_top = dataclasses.replace(W["top_tee_blk"], id="synthetic_printed_tee", pattern="floral")
    o = Outfit(top=printed_top, bottom=W["bot_palazzo_floral"], shoe=W["shoe_sneaker_white"])
    assert "pattern_competes_with_pattern" in fired_ids(o, "weekend")


def test_one_pattern_is_not_overload(fired_ids, W):
    o = Outfit(top=W["top_tee_blk"], bottom=W["bot_palazzo_floral"], shoe=W["shoe_sneaker_white"])
    assert "pattern_competes_with_pattern" not in fired_ids(o, "weekend")


def test_overdressed_is_soft_not_fatal(fired_ids, verdict_for, W):
    """The phase 0 correction, locked in as a test: a formality ceiling must
    never make an outfit infeasible."""
    o = Outfit(top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    _f, v = verdict_for(o, "weekend")
    assert "overdressed_for_context" in {fr.rule.id for fr in v.fired}
    assert v.feasible, "overdressed must stay wearable"


# ----------------------------------------------------------- hard constraints


def test_formality_floor_rejects_underdressed(violated_ids, W):
    o = Outfit(top=W["top_tee_blk"], bottom=W["bot_jean_bootcut_mid"], shoe=W["shoe_sandal_tan_flat"])
    assert "meets_context_formality_floor" in violated_ids(o, "dinner")


def test_formality_floor_passes_when_dressed_enough(violated_ids, W):
    o = Outfit(top=W["top_blouse_struct_cream"], bottom=W["bot_pleat_charcoal"], shoe=W["shoe_heel_blk_block"])
    assert "meets_context_formality_floor" not in violated_ids(o, "office")


def test_unavailable_garment_is_rejected(violated_ids, W):
    assert W["top_shirt_linen_white"].availability != "clean"
    o = Outfit(top=W["top_shirt_linen_white"], bottom=W["bot_wide_cream"], shoe=W["shoe_sneaker_white"])
    assert "everything_must_be_available" in violated_ids(o, "campus")


def test_warmth_band_rejects_too_much_insulation(violated_ids, W):
    o = Outfit(top=W["top_hoodie_oversized_grey"], bottom=W["bot_jean_straight_indigo"],
               outer=W["out_duster_camel"], shoe=W["shoe_sneaker_white"])
    assert "within_context_warmth_band" in violated_ids(o, "weekend")


def test_goal_gating_silences_irrelevant_rules(cv, rules, W):
    """Someone who has not asked to look taller is not lectured about leg line."""
    from silhouette.geometry import compute
    from silhouette.rules import evaluate

    o = Outfit(top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    f = compute(o, cv.frame, cv.height_cm)
    no_goals = dataclasses.replace(cv, goals=[])
    v = evaluate(o, f, no_goals, cv.context("weekend"), rules)
    assert not any(fr.rule.goal for fr in v.fired)
    assert "elongate_lower_body" in v.skipped_goals


# ------------------------------------------------------------------ coverage


def test_every_rule_is_exercised_by_some_test(rules, request):
    """Fails when a rule is added without a fixture."""
    source = "".join(
        (request.path.parent / name).read_text()
        for name in ("test_rules.py",)
    )
    missing = [r.id for r in rules if r.id not in source]
    assert not missing, f"rules with no test fixture: {missing}"


def test_wardrobe_cannot_trigger_pattern_overload(wardrobe, cv):
    """A real coverage gap, recorded rather than hidden.

    `pattern_competes_with_pattern` needs two patterned garments in one
    assembly. The wardrobe holds two patterned pieces - a floral palazzo and a
    floral wrap dress - but they are a bottom and a dress, which are mutually
    exclusive slots. So no valid assembly can contain both, and the rule is
    unreachable from this wardrobe in practice.

    The rule is still correct and still unit-tested above with a synthetic
    fixture. This test asserts the gap directly, over every candidate the
    solver can actually build, rather than guessing from a garment count. It
    will fail the day a patterned top or outer layer is added, which is the
    moment the note above stops being true.
    """
    from silhouette.geometry import compute
    from silhouette.solver import candidates

    worst = max(
        compute(o, cv.frame, cv.height_cm).pattern_count
        for o in candidates(wardrobe, cv.frame)
    )
    assert worst < 2, (
        "an assembly with two patterns is now reachable, so the rule is no "
        "longer a coverage gap - delete this test and add a real fixture"
    )
