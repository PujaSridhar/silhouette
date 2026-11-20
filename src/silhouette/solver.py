"""Outfit assembly.

Phase 0 enumerates exhaustively, which at wardrobe scale is not a compromise -
it is exact. Thirty garments give roughly 1,500 valid assemblies including
tuck variants, and scoring all of them takes milliseconds. An approximate
search here would be strictly worse for no gain.

The place a real solver earns its keep is the WEEK, not the outfit: choosing
seven assemblies that cover seven contexts while no garment is worn more than
twice is a genuine combinatorial problem that enumeration cannot touch. That
lives in capsule.py, and that is where CP-SAT goes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .geometry import Features, compute, hem_y, waistband_y
from .models import Context, Frame, Garment, Outfit, StyleCV
from .rules import Rule, Verdict, evaluate

__all__ = ["ScoredOutfit", "candidates", "score", "best", "by_category"]

_TUCK_TOLERANCE = 0.010


@dataclass(frozen=True)
class ScoredOutfit:
    outfit: Outfit
    features: Features
    verdict: Verdict

    @property
    def total(self) -> float:
        return self.verdict.total

    @property
    def feasible(self) -> bool:
        return self.verdict.feasible


def by_category(wardrobe: list[Garment]) -> dict[str, list[Garment]]:
    out: dict[str, list[Garment]] = {}
    for g in wardrobe:
        out.setdefault(g.category, []).append(g)
    return out


def _tuck_changes_anything(top: Garment, bottom: Garment, frame: Frame) -> bool:
    """Only generate a tucked variant when tucking actually moves the break."""
    if not top.tuckable:
        return False
    wb = waistband_y(bottom, frame)
    hem = hem_y(top, frame)
    return wb is not None and hem is not None and hem > wb + _TUCK_TOLERANCE


def candidates(wardrobe: list[Garment], frame: Frame) -> Iterator[Outfit]:
    """Every structurally valid assembly: (top + bottom) or dress, plus shoe,
    plus an optional outer layer, plus an optional drape.

    Structure is all this decides. Whether a dupatta belongs with a given
    base is a styling call, so it lives in the rules (rules/core.yaml), where a
    stylist can read and change it. A wardrobe with no drape enumerates
    exactly as it did before the slot existed, in the same order.
    """
    cat = by_category(wardrobe)
    shoes = cat.get("shoe", [])
    outers: list[Garment | None] = [None, *cat.get("outer", [])]
    drapes: list[Garment | None] = [None, *cat.get("drape", [])]

    for shoe in shoes:
        for outer in outers:
            for top in cat.get("top", []):
                for bottom in cat.get("bottom", []):
                    tucks = [False, True] if _tuck_changes_anything(top, bottom, frame) else [False]
                    for tucked in tucks:
                        for drape in drapes:
                            yield Outfit(top=top, bottom=bottom, outer=outer, shoe=shoe,
                                         tucked=tucked, drape=drape)
            for dress in cat.get("dress", []):
                for drape in drapes:
                    yield Outfit(dress=dress, outer=outer, shoe=shoe, drape=drape)


def score(outfit: Outfit, cv: StyleCV, ctx: Context, rules: list[Rule]) -> ScoredOutfit:
    f = compute(outfit, cv.frame, cv.height_cm)
    return ScoredOutfit(outfit, f, evaluate(outfit, f, cv, ctx, rules))


def _anchor(s: ScoredOutfit) -> str:
    """The garment a suggestion is really *about* - the bottom, or the dress."""
    o = s.outfit
    if o.dress is not None:
        return o.dress.id
    return o.bottom.id if o.bottom is not None else "-"


def best(
    wardrobe: list[Garment],
    cv: StyleCV,
    ctx: Context,
    rules: list[Rule],
    top_n: int = 3,
    feasible_only: bool = True,
    max_per_anchor: int | None = 1,
) -> list[ScoredOutfit]:
    """Top-scoring feasible assemblies.

    `max_per_anchor` caps how many suggestions may share the same bottom or
    dress. Without it the ranking fills up with minor variations on whichever
    trousers score best, which is one suggestion presented three times. Set it
    to None to see the raw ranking.
    """
    scored = (score(o, cv, ctx, rules) for o in candidates(wardrobe, cv.frame))
    pool = [s for s in scored if s.feasible] if feasible_only else list(scored)
    pool.sort(key=lambda s: -s.total)

    if max_per_anchor is None:
        return pool[:top_n]

    out: list[ScoredOutfit] = []
    seen: dict[str, int] = {}
    for s in pool:
        a = _anchor(s)
        if seen.get(a, 0) >= max_per_anchor:
            continue
        seen[a] = seen.get(a, 0) + 1
        out.append(s)
        if len(out) >= top_n:
            break
    return out
