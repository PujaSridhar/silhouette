"""The eval harness, including an end-to-end test of the learning loop."""

from __future__ import annotations

import numpy as np
import pytest

from silhouette.evals.fitting import design_matrix, fit_weights
from silhouette.evals.metrics import (
    agreement, quadratic_weighted_kappa, rank, spearman,
)
from silhouette.evals.report import CONCLUSIVE_FLOOR, evaluate_labels
from silhouette.evals.sampling import next_batch, outfit_from_label
from silhouette.evals.schema import DIMENSIONS, Label, Ratings
from silhouette.evals.store import LabelStore
from silhouette.evals.synthetic import SYNTHETIC_OVERRIDES, make_labels


# ------------------------------------------------------------------- schema


def test_ratings_reject_out_of_range():
    with pytest.raises(ValueError):
        Ratings(proportion=6, balance=3, formality=3, colour=3, cohesion=3)
    with pytest.raises(ValueError):
        Ratings(proportion=0, balance=3, formality=3, colour=3, cohesion=3)


def _label(**kw):
    base = dict(
        id="x1",
        outfit={"top": "a", "bottom": "b", "shoe": "c", "tucked": False},
        context="office",
        ratings=Ratings(3, 3, 3, 3, 3),
        overall=3,
        would_wear=True,
        labeller="t",
    )
    base.update(kw)
    return Label(**base)


def test_overall_must_be_in_range():
    with pytest.raises(ValueError):
        _label(overall=9)


def test_source_is_restricted():
    with pytest.raises(ValueError):
        _label(source="vibes")


def test_outfit_key_distinguishes_tuck_and_context():
    a = _label()
    b = _label(outfit={"top": "a", "bottom": "b", "shoe": "c", "tucked": True})
    c = _label(context="weekend")
    assert len({a.outfit_key, b.outfit_key, c.outfit_key}) == 3


# -------------------------------------------------------------------- store


def test_store_roundtrip(tmp_path):
    st = LabelStore(tmp_path / "l.jsonl")
    assert st.all() == []
    st.append(_label(id="a"))
    st.append(_label(id="b", context="weekend"))
    assert len(st) == 2
    assert {l.id for l in st.all()} == {"a", "b"}


def test_store_keeps_repeats_for_self_agreement(tmp_path):
    """Overwriting a re-label would destroy the only measure of the ceiling."""
    st = LabelStore(tmp_path / "l.jsonl")
    st.append(_label(id="a", overall=4, created_at="2026-01-01T00:00:00+00:00"))
    st.append(_label(id="b", overall=5, created_at="2026-02-01T00:00:00+00:00"))
    assert len(st.all()) == 2
    assert len(st.latest_per_outfit()) == 1
    pairs = st.repeats()
    assert len(pairs) == 1
    assert (pairs[0][0].overall, pairs[0][1].overall) == (4, 5)


def test_store_separates_expert_from_bootstrap(tmp_path):
    st = LabelStore(tmp_path / "l.jsonl")
    st.append(_label(id="a", source="expert"))
    st.append(_label(id="b", source="bootstrap", context="weekend"))
    assert len(st.expert()) == 1 and len(st.bootstrap()) == 1


def test_corrupt_line_names_the_file_and_line(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text('{"not": "a label"}\n')
    with pytest.raises(ValueError, match="l.jsonl:1"):
        LabelStore(p).all()


# ------------------------------------------------------------------ metrics


def test_rank_averages_ties():
    assert rank([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_is_rank_based_not_linear():
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_qwk_bounds():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert quadratic_weighted_kappa([1, 1, 5, 5], [5, 5, 1, 1]) == pytest.approx(-1.0)


def test_qwk_penalises_larger_disagreement_more():
    """A systematic 2-point offset must score worse than a 1-point offset.

    Both raters below are perfectly rank-correlated with the reference, so
    Spearman cannot tell them apart. Quadratic-weighted kappa can, which is
    why the rubric uses it alongside rho rather than instead of it.
    """
    ref = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    off_by_one = [2, 3, 4, 2, 3, 4, 2, 3, 4]
    off_by_two = [3, 4, 5, 3, 4, 5, 3, 4, 5]
    assert spearman(ref, off_by_one) == pytest.approx(1.0)
    assert spearman(ref, off_by_two) == pytest.approx(1.0)
    assert quadratic_weighted_kappa(ref, off_by_two) < quadratic_weighted_kappa(ref, off_by_one)


def test_qwk_is_zero_for_a_constant_rater():
    """Chance correction means a rater who always says the same thing scores
    exactly 0, however close that constant happens to be. Worth pinning: it
    looks like a bug the first time you see it, and it is the reason the
    mean-baseline judge exists as an explicit comparison."""
    assert quadratic_weighted_kappa([3, 3, 3, 3], [2, 4, 2, 4]) == pytest.approx(0.0)
    assert quadratic_weighted_kappa([3, 3, 3, 3], [1, 5, 1, 5]) == pytest.approx(0.0)


def test_agreement_handles_constant_prediction():
    """A constant predictor has undefined rank correlation, not zero."""
    rep = agreement([3.0, 3.0, 3.0, 3.0], [1, 2, 4, 5])
    assert np.isnan(rep.spearman)
    assert "n/a" in rep.summary()


# ----------------------------------------------------------------- sampling


def test_sampler_skips_already_labelled(tmp_path, wardrobe, cv, rules):
    st = LabelStore(tmp_path / "l.jsonl")
    first = next_batch(wardrobe, cv, rules, st, n=3, contexts=["office"])
    assert len(first) == 3
    for t in first:
        o = t.outfit
        st.append(Label(
            id=f"x{t.key}", context=t.context,
            outfit={
                "top": o.top.id if o.top else None,
                "bottom": o.bottom.id if o.bottom else None,
                "dress": o.dress.id if o.dress else None,
                "outer": o.outer.id if o.outer else None,
                "shoe": o.shoe.id if o.shoe else None,
                "tucked": o.tucked,
            },
            ratings=Ratings(3, 3, 3, 3, 3), overall=3, would_wear=True, labeller="t",
        ))
    again = next_batch(wardrobe, cv, rules, st, n=3, contexts=["office"])
    assert not ({t.key for t in first} & {t.key for t in again})


def test_sampler_spreads_across_rules(tmp_path, wardrobe, cv, rules):
    """Uniform sampling leaves rules unobserved; this one should not."""
    st = LabelStore(tmp_path / "l.jsonl")
    tasks = next_batch(wardrobe, cv, rules, st, n=40)
    fired = {fr.rule.id for t in tasks for fr in t.scored.verdict.fired}
    soft = {r.id for r in rules if r.kind == "soft"}
    # the unreachable pattern rule cannot appear; everything else should
    assert len(fired) >= len(soft) - 1


def test_outfit_from_label_roundtrips(wardrobe, cv, rules, W):
    from silhouette.models import Outfit
    o = Outfit(top=W["top_tee_blk"], bottom=W["bot_wide_blk"], shoe=W["shoe_sneaker_white"], tucked=True)
    lab = _label(outfit={
        "top": "top_tee_blk", "bottom": "bot_wide_blk", "dress": None,
        "outer": None, "shoe": "shoe_sneaker_white", "tucked": True,
    })
    back = outfit_from_label(lab, {g.id: g for g in wardrobe})
    assert back.key == o.key


# ------------------------------------------------------------------- report


def test_report_refuses_conclusions_below_the_floor(tmp_path, wardrobe, cv, rules):
    st = LabelStore(tmp_path / "l.jsonl")
    tasks = next_batch(wardrobe, cv, rules, st, n=5, contexts=["office"])
    labels = make_labels(tasks, rules)
    rep = evaluate_labels(labels, wardrobe, cv, rules, store=st, source_used="bootstrap")
    assert not rep.conclusive
    assert "BOOTSTRAP" in rep.caveat()


def test_expert_labels_below_floor_are_marked_indicative(tmp_path, wardrobe, cv, rules):
    st = LabelStore(tmp_path / "l.jsonl")
    tasks = next_batch(wardrobe, cv, rules, st, n=4, contexts=["office"])
    labels = [
        Label(
            id=f"e{i}", context=t.context,
            outfit={
                "top": t.outfit.top.id if t.outfit.top else None,
                "bottom": t.outfit.bottom.id if t.outfit.bottom else None,
                "dress": t.outfit.dress.id if t.outfit.dress else None,
                "outer": t.outfit.outer.id if t.outfit.outer else None,
                "shoe": t.outfit.shoe.id if t.outfit.shoe else None,
                "tucked": t.outfit.tucked,
            },
            ratings=Ratings(3, 3, 3, 3, 3), overall=3, would_wear=True,
            labeller="t", source="expert",
        )
        for i, t in enumerate(tasks)
    ]
    rep = evaluate_labels(labels, wardrobe, cv, rules, store=st, source_used="expert")
    assert not rep.conclusive
    assert "INDICATIVE" in rep.caveat()
    assert CONCLUSIVE_FLOOR > len(labels)


# ------------------------------------------------- the learning loop, end to end


def test_design_matrix_reproduces_the_engine_total(tmp_path, wardrobe, cv, rules):
    """Fitting w on these features IS fitting the rule weights - so w . x must
    equal the engine's own total, exactly."""
    from silhouette.solver import score
    st = LabelStore(tmp_path / "l.jsonl")
    tasks = next_batch(wardrobe, cv, rules, st, n=12)
    labels = make_labels(tasks, rules)
    X, _y, totals, rule_ids = design_matrix(labels, wardrobe, cv, rules)
    by_id = {r.id: r for r in rules}
    w = np.asarray([by_id[rid].weight for rid in rule_ids])
    assert np.allclose(X @ w, totals, atol=1e-9)


def test_fitting_recovers_the_synthetic_rater(tmp_path, wardrobe, cv, rules):
    """The whole harness, validated against known ground truth.

    The synthetic rater uses the engine's rule scores with exactly two weights
    changed (see evals/synthetic.py). A working harness must recover both from
    the labels alone - that is sampling, design matrix, projected-gradient fit
    and cross-validation all being correct at once.

    Absolute scale is not recoverable and is not asserted: the rater maps its
    total onto 1-5 by a linear rescaling, so every weight is identified only up
    to a common factor. What must hold is the *relative* movement of the two
    overridden rules against their own priors.
    """
    st = LabelStore(tmp_path / "l.jsonl")
    tasks = next_batch(wardrobe, cv, rules, st, n=180)
    labels = make_labels(tasks, rules, noise=0.35, seed=7)

    res = fit_weights(labels, wardrobe, cv, rules, folds=5)
    assert res.n == len(labels)
    assert res.improved, "fit must beat the prior on held-out folds"

    idx = {rid: i for i, rid in enumerate(res.rule_ids)}
    prior = {rid: res.prior[i] for rid, i in idx.items()}
    fitted = {rid: res.fitted[i] for rid, i in idx.items()}

    # the rater was told a heel is not a virtue: weight must collapse
    heel = "heel_extends_the_leg_line"
    assert SYNTHETIC_OVERRIDES[heel] < prior[heel]
    assert fitted[heel] < 0.5 * prior[heel]

    # ...and that effort has a cost: weight must rise sharply
    over = "overdressed_for_context"
    assert SYNTHETIC_OVERRIDES[over] > prior[over]
    assert fitted[over] > 2.0 * prior[over]


def test_fit_never_produces_a_negative_weight(tmp_path, wardrobe, cv, rules):
    """Direction lives in each rule's score expression. A negative weight would
    silently invert a rule while its explanation kept claiming the opposite."""
    st = LabelStore(tmp_path / "l.jsonl")
    labels = make_labels(next_batch(wardrobe, cv, rules, st, n=120), rules, seed=3)
    res = fit_weights(labels, wardrobe, cv, rules, folds=5)
    assert (res.fitted >= 0).all()


def test_fit_leaves_unexercised_rules_at_the_prior(tmp_path, wardrobe, cv, rules):
    st = LabelStore(tmp_path / "l.jsonl")
    labels = make_labels(next_batch(wardrobe, cv, rules, st, n=60), rules, seed=1)
    res = fit_weights(labels, wardrobe, cv, rules, folds=5)
    idx = {rid: i for i, rid in enumerate(res.rule_ids)}
    for rid in res.unsupported():
        assert res.fitted[idx[rid]] == pytest.approx(res.prior[idx[rid]], abs=1e-6)


def test_too_few_labels_claims_no_improvement(tmp_path, wardrobe, cv, rules):
    st = LabelStore(tmp_path / "l.jsonl")
    labels = make_labels(next_batch(wardrobe, cv, rules, st, n=6), rules)
    res = fit_weights(labels, wardrobe, cv, rules, folds=5)
    assert res.folds == 0
    assert not res.improved


def test_source_filtering_isolates_expert_from_bootstrap(tmp_path, wardrobe, cv, rules):
    """Bootstrap labels in the same store must never pollute expert sampling,
    coverage, or self-agreement."""
    from silhouette.evals.sampling import coverage_report

    st = LabelStore(tmp_path / "l.jsonl")
    boot_tasks = next_batch(wardrobe, cv, rules, st, n=5, contexts=["office"], source="bootstrap")
    boot_labels = make_labels(boot_tasks, rules, seed=1)
    for l in boot_labels:
        st.append(l)
    # Add repeated bootstrap labels so store.repeats() is non-empty for bootstrap.
    for i, l in enumerate(boot_labels[:2]):
        st.append(Label(
            id=f"boot_rep_{i}", outfit=l.outfit, context=l.context,
            ratings=l.ratings, overall=l.overall, would_wear=l.would_wear,
            labeller=l.labeller, source="bootstrap",
            created_at="2026-12-01T00:00:00+00:00",
        ))

    # Expert sampling must not skip outfits that only have bootstrap labels.
    assert st.labelled_keys(source="expert") == set()
    exp_tasks = next_batch(wardrobe, cv, rules, st, n=5, contexts=["office"], source="expert")
    assert [t.key for t in exp_tasks] == [t.key for t in boot_tasks]

    # Expert coverage must report zero across all rules.
    exp_cov = coverage_report(st, wardrobe, cv, rules, source="expert")
    assert all(n == 0 for n in exp_cov.values())
    boot_cov = coverage_report(st, wardrobe, cv, rules, source="bootstrap")
    assert sum(boot_cov.values()) > 0

    # Expert evaluation must ignore bootstrap repeats and bootstrap coverage.
    rep = evaluate_labels(st.expert(), wardrobe, cv, rules, store=st, source_used="expert")
    assert rep.n_repeats == 0
    assert rep.self_agreement_qwk is None
    assert all(n == 0 for n in rep.coverage.values())
