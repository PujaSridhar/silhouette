"""The drape slot: a dupatta or stole as an optional layer over an outfit.

Uses the sample wardrobe plus data/wardrobe_drapes.csv. The sample wardrobe
itself has no drape, so every number measured on it stays where it was; the
first tests here check exactly that.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from silhouette.evals.sampling import Task, outfit_from_label
from silhouette.evals.schema import Label
from silhouette.geometry import compute
from silhouette.models import Outfit, load_wardrobe, outfit_slot_key
from silhouette.solver import best, by_category, candidates, score

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def drapes():
    return load_wardrobe(ROOT / "data" / "wardrobe_drapes.csv")


@pytest.fixture(scope="session")
def D(drapes):
    return {g.id: g for g in drapes}


@pytest.fixture(scope="session")
def with_drapes(wardrobe, drapes):
    return wardrobe + drapes


# ------------------------------------------------------------ enumeration


def test_a_wardrobe_without_drapes_enumerates_exactly_as_before(wardrobe, cv):
    outs = list(candidates(wardrobe, cv.frame))
    assert all(o.drape is None for o in outs)
    assert len(outs) == 1548  # the number README and the W1 notes quote


def test_each_drape_adds_one_variant_of_every_outfit(wardrobe, with_drapes, drapes, cv):
    plain = list(candidates(wardrobe, cv.frame))
    draped = list(candidates(with_drapes, cv.frame))
    assert len(draped) == len(plain) * (1 + len(drapes))
    assert [o for o in draped if o.drape is None] == plain
    assert len({o.key for o in draped}) == len(draped)


def test_drape_is_its_own_slot_not_a_top_or_layer(with_drapes, cv):
    cat = by_category(with_drapes)
    assert {g.id for g in cat["drape"]} == {
        "drape_dupatta_mustard", "drape_dupatta_bandhani_red", "drape_stole_grey"}
    for o in candidates(with_drapes, cv.frame):
        if o.drape is not None:
            assert o.drape.category == "drape"
            assert o.drape not in (o.top, o.outer)


def test_her_real_dupatta_is_enumerated(cv):
    """Structural only: her ratings are still being filled in, so nothing here
    reads volume, structure or formality from her file."""
    real = load_wardrobe(ROOT / "data" / "wardrobe_real.csv", require_volume=False)
    dupatta = next(g for g in real if g.id == "drape_dupatta_teal")
    assert dupatta.category == "drape" and dupatta.subcategory == "dupatta"
    kurtas = [g for g in real if g.subcategory == "kurta"]
    assert kurtas
    small = ([g for g in real if g.category == "bottom"][:2]
             + [g for g in real if g.category == "shoe"][:2] + kurtas + [dupatta])
    assert any(o.drape is dupatta and o.top in kurtas for o in candidates(small, cv.frame))


# --------------------------------------------------------------- geometry


def test_a_drape_leaves_the_silhouette_alone(with_drapes, cv):
    """No break, lift, volume, structure or contrast change: a drape's effect
    is carried by its own rules, not by the geometry."""
    same = ("break_ratio", "apparent_leg", "lift", "hem_drop", "break_source", "is_column",
            "upper_volume", "lower_volume", "asymmetry", "upper_structure",
            "contrast_at_break", "heel_cm", "height_total")
    for o in candidates(with_drapes, cv.frame):
        if o.drape is None:
            continue
        a = compute(o, cv.frame, cv.height_cm)
        b = compute(dataclasses.replace(o, drape=None), cv.frame, cv.height_cm)
        for k in same:
            assert getattr(a, k) == getattr(b, k), k


def test_a_drape_counts_toward_formality_warmth_and_pattern(cv, W, D):
    base = Outfit(top=W["top_kurta_indigo"], bottom=W["bot_salwar_white"],
                  shoe=W["shoe_sandal_tan_flat"])
    bare = compute(base, cv.frame, cv.height_cm)
    red = compute(dataclasses.replace(base, drape=D["drape_dupatta_bandhani_red"]),
                  cv.frame, cv.height_cm)
    assert red.total_clo == pytest.approx(bare.total_clo + 0.08)
    assert red.max_formality == max(bare.max_formality, 0.62)
    assert red.pattern_count == bare.pattern_count + 1


# ------------------------------------------------------------------ rules


def test_dupatta_completes_the_set(fired_ids, verdict_for, W, D):
    dup = D["drape_dupatta_mustard"]
    kurta = Outfit(top=W["top_kurta_indigo"], bottom=W["bot_salwar_white"],
                   shoe=W["shoe_sandal_tan_flat"], drape=dup)
    assert "dupatta_completes_the_set" in fired_ids(kurta, "campus")
    anarkali = Outfit(dress=W["dress_anarkali_maroon"], shoe=W["shoe_heel_blk_block"], drape=dup)
    assert "dupatta_completes_the_set" in fired_ids(anarkali, "wedding_function")
    # a kurta over jeans is still a kurta
    fusion = Outfit(top=W["top_kurta_indigo"], bottom=W["bot_jean_straight_indigo"],
                    shoe=W["shoe_sneaker_white"], drape=dup)
    assert "dupatta_completes_the_set" in fired_ids(fusion, "campus")


def test_the_set_counts_for_more_when_the_occasion_is_dressier(verdict_for, W, D):
    o = Outfit(dress=W["dress_anarkali_maroon"], shoe=W["shoe_heel_blk_block"],
               drape=D["drape_dupatta_mustard"])

    def credit(ctx):
        _f, v = verdict_for(o, ctx)
        return next(fr.contribution for fr in v.fired if fr.rule.id == "dupatta_completes_the_set")

    assert credit("weekend") == pytest.approx(0.40 * 0.4)
    assert credit("weekend") == credit("campus")
    assert credit("weekend") < credit("dinner") < credit("office") < credit("wedding_function")
    assert credit("wedding_function") == pytest.approx(0.40)


def test_dupatta_without_its_set(fired_ids, W, D):
    o = Outfit(top=W["top_tee_blk"], bottom=W["bot_jean_straight_indigo"],
               shoe=W["shoe_sneaker_white"], drape=D["drape_dupatta_mustard"])
    ids = fired_ids(o, "campus")
    assert "dupatta_without_its_set" in ids
    assert "dupatta_completes_the_set" not in ids
    # a salwar is half a salwar kameez; without the kurta it is not the set
    cami = Outfit(top=W["top_cami_silk_blk"], bottom=W["bot_salwar_white"],
                  shoe=W["shoe_heel_blk_block"], drape=D["drape_dupatta_mustard"])
    assert "dupatta_without_its_set" in fired_ids(cami, "dinner")


def test_the_two_dupatta_rules_split_every_dupatta_outfit(with_drapes, cv, rules):
    """They share one condition, negated. Exactly one fires on every outfit
    carrying a dupatta, and neither fires on a stole or on no drape at all."""
    ctx = cv.context("dinner")
    pair = {"dupatta_completes_the_set", "dupatta_without_its_set"}
    for o in candidates(with_drapes, cv.frame):
        if o.dress is not None and o.dress.subcategory == "saree":
            continue  # rejected outright, see below; the soft rules still run
        fired = {fr.rule.id for fr in score(o, cv, ctx, rules).verdict.fired} & pair
        if o.drape is not None and o.drape.subcategory == "dupatta":
            assert len(fired) == 1, o.label()
        else:
            assert not fired, o.label()


def test_a_stole_is_neutral(with_drapes, wardrobe, cv, rules, D):
    """Nothing rewards or penalises a plain stole for itself, so it never
    pushes an outfit ahead of the same outfit without it."""
    for name in cv.contexts:
        ctx = cv.context(name)
        for o in list(candidates(wardrobe, cv.frame))[::25]:
            bare = score(o, cv, ctx, rules)
            stole = score(dataclasses.replace(o, drape=D["drape_stole_grey"]), cv, ctx, rules)
            assert stole.total <= bare.total + 1e-12


def test_one_drape_at_a_time(violated_ids, W, D):
    saree = Outfit(dress=W["dress_saree_teal"], shoe=W["shoe_heel_blk_block"],
                   drape=D["drape_dupatta_mustard"])
    assert "one_drape_at_a_time" in violated_ids(saree, "wedding_function")
    stole = dataclasses.replace(saree, drape=D["drape_stole_grey"])
    assert "one_drape_at_a_time" in violated_ids(stole, "wedding_function")
    anarkali = Outfit(dress=W["dress_anarkali_maroon"], shoe=W["shoe_heel_blk_block"],
                      drape=D["drape_dupatta_mustard"])
    assert "one_drape_at_a_time" not in violated_ids(anarkali, "wedding_function")
    assert "one_drape_at_a_time" not in violated_ids(
        dataclasses.replace(saree, drape=None), "wedding_function")


def test_drape_rule_messages_name_the_drape(verdict_for, W, D):
    o = Outfit(dress=W["dress_anarkali_maroon"], shoe=W["shoe_heel_blk_block"],
               drape=D["drape_dupatta_mustard"])
    _f, v = verdict_for(o, "wedding_function")
    msg = next(fr.message for fr in v.fired if fr.rule.id == "dupatta_completes_the_set")
    assert "Mustard chanderi dupatta" in msg


# ----------------------------------------------------------------- solver


def test_suggestions_without_drapes_are_unchanged_by_the_slot(wardrobe, with_drapes, cv, rules):
    """Adding drapes may add a dupatta where it earns its place, but every
    other suggestion is the very same outfit with the very same score."""
    for name in cv.contexts:
        ctx = cv.context(name)
        before = best(wardrobe, cv, ctx, rules, top_n=3)
        after = best(with_drapes, cv, ctx, rules, top_n=3)
        for s in after:
            if s.outfit.drape is None:
                match = [b for b in before if b.outfit.key == s.outfit.key]
                assert match and match[0].total == s.total


def test_dupatta_tops_the_wedding_function_with_the_anarkali(with_drapes, cv, rules):
    top = best(with_drapes, cv, cv.context("wedding_function"), rules, top_n=1)[0]
    assert top.outfit.dress is not None and top.outfit.dress.subcategory == "anarkali"
    assert top.outfit.drape is not None and top.outfit.drape.subcategory == "dupatta"


def test_no_dupatta_over_western_basics_in_casual_suggestions(with_drapes, cv, rules):
    for name in ("weekend", "campus"):
        for s in best(with_drapes, cv, cv.context(name), rules, top_n=5):
            if s.outfit.drape is not None and s.outfit.drape.subcategory == "dupatta":
                assert "dupatta_without_its_set" not in {fr.rule.id for fr in s.verdict.fired}


# ----------------------------------------------------------- labels / keys


def test_keys_without_a_drape_keep_their_old_shape():
    ids = {"top": "a", "bottom": "b", "shoe": "c"}
    assert outfit_slot_key(ids.get) == "a|b|-|-|c"
    ids["drape"] = "d"
    assert outfit_slot_key(ids.get) == "a|b|-|-|c|drape=d"


def test_a_saved_rating_carries_the_drape(with_drapes, cv, rules, W, D):
    o = Outfit(top=W["top_kurta_indigo"], bottom=W["bot_salwar_white"],
               shoe=W["shoe_sandal_tan_flat"], drape=D["drape_dupatta_mustard"])
    t = Task(o, "campus", score(o, cv, cv.context("campus"), rules))
    lab = Label(
        id="l1", context="campus", overall=4, would_wear=True, labeller="t",
        outfit={"top": "top_kurta_indigo", "bottom": "bot_salwar_white", "dress": None,
                "outer": None, "shoe": "shoe_sandal_tan_flat",
                "drape": "drape_dupatta_mustard", "tucked": False},
    )
    assert lab.outfit_key == t.key
    back = outfit_from_label(lab, {g.id: g for g in with_drapes})
    assert back.key == o.key and back.drape is D["drape_dupatta_mustard"]
    bare = Task(dataclasses.replace(o, drape=None), "campus", t.scored)
    assert bare.key != t.key
