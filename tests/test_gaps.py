"""Gap analysis: marginal outfit yield and unlocking uncovered occasions."""

from __future__ import annotations

import dataclasses
import math

from silhouette.gaps import analyse


def test_analyse_returns_finite_deltas_in_rank_order(wardrobe, catalog, cv, rules):
    gaps = analyse(wardrobe, catalog, cv, rules)
    assert len(gaps) == len(catalog)
    assert all(math.isfinite(g.best_delta) for g in gaps)
    assert [g.rank_key for g in gaps] == sorted(g.rank_key for g in gaps)


def test_unlocking_an_uncovered_occasion_ranks_first_and_stays_finite(wardrobe, catalog, cv, rules):
    """Removing the only dressy shoe leaves office and dinner with zero feasible
    outfits. A candidate shoe that unlocks them must rank ahead of items that
    only nudge already-covered occasions, without producing inf deltas."""
    no_heel = [g for g in wardrobe if g.id != "shoe_heel_blk_block"]
    gaps = analyse(no_heel, catalog, cv, rules)
    top = gaps[0]
    assert top.item.id == "cand_heel_nude_block"
    assert top.contexts_unlocked == 2
    assert set(top.unlocked) == {"office", "dinner"}
    assert math.isfinite(top.best_delta)
    assert all(math.isfinite(v) for v in top.per_context.values())


def test_two_unlockers_tiebreak_by_unlocked_score(wardrobe, catalog, cv, rules, W):
    """When two candidates both unlock the same occasion, the higher-scoring
    one ranks first."""
    no_heel = [g for g in wardrobe if g.id != "shoe_heel_blk_block"]
    flat_dressy = dataclasses.replace(
        W["shoe_heel_blk_block"], id="cand_flat_dressy", name="Dressy flat", heel_cm=1.0
    )
    high_heel = dataclasses.replace(
        W["shoe_heel_blk_block"], id="cand_high_heel", name="Dressy heel", heel_cm=8.0
    )
    gaps = analyse(no_heel, [flat_dressy, high_heel], cv, rules, context_names=["wedding_function"])
    assert [g.item.id for g in gaps] == ["cand_high_heel", "cand_flat_dressy"]
    assert all(g.contexts_unlocked == 1 for g in gaps)
    assert gaps[0].best_unlocked > gaps[1].best_unlocked
