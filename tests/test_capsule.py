"""Week planning. The one place a real solver is doing real work."""

from __future__ import annotations

import pytest

from silhouette.capsule import HAVE_ORTOOLS, plan_week

WEEK = ["office", "office", "campus", "dinner", "weekend"]


def test_greedy_always_available(wardrobe, cv, rules):
    plan = plan_week(wardrobe, cv, WEEK, rules, max_wears=2, prefer="greedy")
    assert plan.backend == "greedy"
    assert len(plan.chosen) == len(WEEK)
    assert not plan.optimal


def test_one_outfit_per_day_and_all_feasible(wardrobe, cv, rules):
    plan = plan_week(wardrobe, cv, WEEK, rules, max_wears=2, prefer="greedy")
    assert all(s.feasible for s in plan.chosen)
    assert plan.total == pytest.approx(sum(s.total for s in plan.chosen))


def test_unknown_context_is_rejected(wardrobe, cv, rules):
    with pytest.raises(KeyError):
        plan_week(wardrobe, cv, ["brunch_on_mars"], rules)


@pytest.mark.skipif(not HAVE_ORTOOLS, reason="ortools not installed")
def test_cpsat_beats_or_matches_greedy(wardrobe, cv, rules):
    """The optimality claim, as an assertion rather than a comment.

    Greedy is a valid heuristic for the same problem, so CP-SAT must never
    score worse. If this ever fails, the objective and the constraints have
    drifted apart.
    """
    opt = plan_week(wardrobe, cv, WEEK, rules, max_wears=2, prefer="cpsat")
    greedy = plan_week(wardrobe, cv, WEEK, rules, max_wears=2, prefer="greedy")

    # Compare on the objective actually being optimised: minimise broken wear
    # limits first, then maximise styling score. Comparing raw totals alone is
    # a category error - a plan that ignores the limit can always score higher.
    def objective(plan):
        return (sum(plan.overages.values()), -plan.total)

    assert objective(opt) <= objective(greedy)
    assert opt.optimal


@pytest.mark.skipif(not HAVE_ORTOOLS, reason="ortools not installed")
def test_cpsat_wins_on_score_when_both_stay_within_limits(wardrobe, cv, rules):
    """With slack out of the picture the comparison is purely on styling score,
    and the exact solver must not lose to the heuristic."""
    opt = plan_week(wardrobe, cv, WEEK, rules, max_wears=3, prefer="cpsat")
    greedy = plan_week(wardrobe, cv, WEEK, rules, max_wears=3, prefer="greedy")
    assert opt.within_limits and greedy.within_limits
    assert opt.total >= greedy.total - 1e-9


@pytest.mark.skipif(not HAVE_ORTOOLS, reason="ortools not installed")
def test_wear_limit_is_respected_when_it_can_be(wardrobe, cv, rules):
    plan = plan_week(wardrobe, cv, ["weekend", "campus"], rules, max_wears=1, prefer="cpsat")
    assert plan.within_limits
    assert all(c <= 1 for c in plan.wear_counts.values())


@pytest.mark.skipif(not HAVE_ORTOOLS, reason="ortools not installed")
def test_overage_is_diagnosed_not_raised(wardrobe, cv, rules):
    """A wardrobe that cannot cover the week must produce advice, not an
    exception. This records the real bottleneck the spike found: exactly one
    shoe clears the 0.45 formality floor, so any week with three dressy days
    over-wears it."""
    plan = plan_week(wardrobe, cv, WEEK, rules, max_wears=2, prefer="cpsat")
    assert len(plan.chosen) == len(WEEK)
    assert not plan.within_limits
    assert "shoe_heel_blk_block" in plan.overages


@pytest.mark.skipif(not HAVE_ORTOOLS, reason="ortools not installed")
def test_solver_prefers_style_over_spending_slack(wardrobe, cv, rules):
    """Overage is penalised heavily enough that the solver only ever uses it
    when there is genuinely no alternative."""
    plan = plan_week(wardrobe, cv, WEEK, rules, max_wears=2, prefer="cpsat")
    assert sum(plan.overages.values()) == 1, "should spend the minimum slack"


@pytest.mark.skipif(not HAVE_ORTOOLS, reason="ortools not installed")
def test_loosening_the_limit_removes_the_bottleneck(wardrobe, cv, rules):
    plan = plan_week(wardrobe, cv, WEEK, rules, max_wears=3, prefer="cpsat")
    assert plan.within_limits
    assert not plan.overages
