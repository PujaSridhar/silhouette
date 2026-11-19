"""Enumeration, feasibility and suggestion diversity."""

from __future__ import annotations

import pytest

from silhouette.solver import best, by_category, candidates, score


def test_every_candidate_is_structurally_valid(wardrobe, cv):
    for o in candidates(wardrobe, cv.frame):
        assert o.shoe is not None, "an outfit without shoes is not an outfit"
        if o.dress is not None:
            assert o.top is None and o.bottom is None
        else:
            assert o.top is not None and o.bottom is not None


def test_enumeration_is_complete(wardrobe, cv):
    """The count is predictable from the wardrobe, which is what makes
    exhaustive search defensible rather than hand-wavy."""
    cat = by_category(wardrobe)
    n_shoe, n_outer = len(cat["shoe"]), len(cat["outer"]) + 1
    pairs = len(cat["top"]) * len(cat["bottom"])
    tucks = sum(
        1
        for t in cat["top"]
        for b in cat["bottom"]
        if t.tuckable
        and (lambda wb, hem: wb is not None and hem is not None and hem > wb + 0.010)(
            __import__("silhouette.geometry", fromlist=["x"]).waistband_y(b, cv.frame),
            __import__("silhouette.geometry", fromlist=["x"]).hem_y(t, cv.frame),
        )
    )
    expected = n_shoe * n_outer * (pairs + tucks + len(cat["dress"]))
    assert len(list(candidates(wardrobe, cv.frame))) == expected


def test_no_duplicate_candidates(wardrobe, cv):
    keys = [o.key for o in candidates(wardrobe, cv.frame)]
    assert len(keys) == len(set(keys))


def test_tuck_variants_only_where_tucking_changes_the_break(wardrobe, cv):
    from silhouette.geometry import compute

    for o in candidates(wardrobe, cv.frame):
        if not o.tucked:
            continue
        loose = compute(
            type(o)(top=o.top, bottom=o.bottom, outer=o.outer, shoe=o.shoe), cv.frame, cv.height_cm
        )
        tight = compute(o, cv.frame, cv.height_cm)
        assert tight.break_ratio != loose.break_ratio


def test_best_returns_only_feasible_outfits(wardrobe, cv, rules):
    for name in cv.contexts:
        for s in best(wardrobe, cv, cv.context(name), rules, top_n=5):
            assert s.feasible
            assert not s.verdict.violations


def test_best_is_sorted_descending(wardrobe, cv, rules):
    picks = best(wardrobe, cv, cv.context("office"), rules, top_n=5)
    assert [p.total for p in picks] == sorted((p.total for p in picks), reverse=True)


def test_diversity_cap_prevents_one_garment_dominating(wardrobe, cv, rules):
    """Three variations on one pair of trousers is one suggestion, not three."""
    ctx = cv.context("office")
    raw = best(wardrobe, cv, ctx, rules, top_n=3, max_per_anchor=None)
    diverse = best(wardrobe, cv, ctx, rules, top_n=3, max_per_anchor=1)
    anchors = [s.outfit.bottom.id if s.outfit.bottom else s.outfit.dress.id for s in diverse]
    assert len(set(anchors)) == len(anchors)
    # diversity costs score - that is the trade, and it should be small
    assert diverse[0].total == raw[0].total
    assert sum(s.total for s in diverse) <= sum(s.total for s in raw)


def test_every_context_has_at_least_one_wearable_outfit(wardrobe, cv, rules):
    for name in cv.contexts:
        assert best(wardrobe, cv, cv.context(name), rules, top_n=1), f"nothing for {name}"


def test_her_signature_outfit_scores_well(wardrobe, cv, rules, W):
    """The acceptance test that matters: the wearer's own go-to must rank.

    This is the phase 0 gate in a single assertion. If the engine cannot agree
    with the stylist about the outfit she already wears, nothing downstream is
    worth building.
    """
    from silhouette.models import Outfit

    sig = Outfit(top=W["top_crop_rib_ivory"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"])
    s = score(sig, cv, cv.context("weekend"), rules)
    assert s.feasible
    assert s.total > 1.5
    fired = {fr.rule.id for fr in s.verdict.fired}
    assert "high_break_elongates" in fired
    assert "fitted_top_counterweights_volume" in fired
    assert s.features.lift > 0.11
