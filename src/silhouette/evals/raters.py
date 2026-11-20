"""Agreement between raters, and the outfits they could not judge.

Self-agreement (one rater, the same outfit twice) is the ceiling on what any
model can reach for THAT rater. Agreement between raters is a different
question: do two stylists see the same outfit the same way? If they do not,
fitting weights to their pooled labels averages two tastes into a third that
neither holds. The shared set (sampling.shared_set) exists so this number can
be measured at all: without it, stylists rate disjoint outfits and there is no
overlap to compare.

The headline is Krippendorff's alpha on `overall`, interval metric. Alpha
rather than an average of pairwise kappas because it takes any number of
raters, tolerates a rater who has not finished the set, and is the standard
reliability coefficient for this design. Interval because the rubric's 1-5 is
treated as a scale everywhere else in the harness (Spearman, MAE). Pairwise
quadratic-weighted kappa is reported beside it, so one stylist who reads the
rubric differently is visible rather than averaged away.

Each rater counts once per outfit: their FIRST rating. A repeat is a second
look by the same person, which is self-agreement's business, and letting it
in here would count one rater twice.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from .metrics import quadratic_weighted_kappa
from .schema import CantJudge, Label

__all__ = [
    "krippendorff_alpha_interval", "PairAgreement", "InterRater",
    "inter_rater_agreement", "CantJudgeSummary", "cant_judge_summary",
]


def krippendorff_alpha_interval(units: list[list[float]]) -> float:
    """Krippendorff's alpha, interval metric.

    `units` is one list of values per rated item, one value per rater who
    rated it (missing raters simply absent). Items with fewer than two values
    are not pairable and drop out, as the coefficient defines. NaN when fewer
    than two values are pairable or every value is identical (no variation to
    agree about)."""
    pairable = [[float(v) for v in u] for u in units if len(u) >= 2]
    n = sum(len(u) for u in pairable)
    if n < 2:
        return float("nan")
    d_o = 0.0
    for u in pairable:
        m = len(u)
        s = sum((a - b) ** 2 for i, a in enumerate(u) for j, b in enumerate(u) if i != j)
        d_o += s / (m - 1)
    d_o /= n
    values = [v for u in pairable for v in u]
    s = sum((a - b) ** 2 for i, a in enumerate(values) for j, b in enumerate(values) if i != j)
    d_e = s / (n * (n - 1))
    if d_e == 0.0:
        return float("nan")
    return 1.0 - d_o / d_e


@dataclass(frozen=True)
class PairAgreement:
    a: str
    b: str
    n: int            # outfits both rated
    qwk: float
    exact: float      # share of those with the same overall


@dataclass
class InterRater:
    raters: list[str]
    n_units: int                  # outfits rated by two or more raters
    alpha: float
    pairs: list[PairAgreement] = field(default_factory=list)
    would_wear_agree: float = float("nan")   # share of rater pairs, over units

    def summary(self) -> str:
        if len(self.raters) < 2:
            return "one rater so far - agreement between stylists needs a second"
        if self.n_units == 0:
            return (f"{len(self.raters)} raters, no outfit rated by more than one yet "
                    "- the shared set is how they overlap")
        a = "n/a" if math.isnan(self.alpha) else f"{self.alpha:+.3f}"
        return (f"alpha = {a}  over {self.n_units} shared outfit"
                f"{'' if self.n_units == 1 else 's'}, {len(self.raters)} raters")


def _first_per_rater(labels: list[Label]) -> dict[str, dict[str, Label]]:
    """outfit_key -> labeller -> that rater's earliest label of it."""
    out: dict[str, dict[str, Label]] = defaultdict(dict)
    for l in sorted(labels, key=lambda l: l.created_at):
        out[l.outfit_key].setdefault(l.labeller, l)
    return out


def inter_rater_agreement(labels: list[Label]) -> InterRater:
    raters = sorted({l.labeller for l in labels})
    by_unit = _first_per_rater(labels)
    shared = {k: v for k, v in by_unit.items() if len(v) >= 2}

    alpha = krippendorff_alpha_interval(
        [[float(l.overall) for l in v.values()] for v in shared.values()]
    )

    pairs: list[PairAgreement] = []
    for a, b in combinations(raters, 2):
        xa, xb = [], []
        for v in shared.values():
            if a in v and b in v:
                xa.append(v[a].overall)
                xb.append(v[b].overall)
        if xa:
            pairs.append(PairAgreement(
                a, b, len(xa), quadratic_weighted_kappa(xa, xb),
                sum(int(x == y) for x, y in zip(xa, xb)) / len(xa)))

    agree = total = 0
    for v in shared.values():
        for la, lb in combinations(v.values(), 2):
            total += 1
            agree += int(la.would_wear == lb.would_wear)
    ww = agree / total if total else float("nan")

    return InterRater(raters=raters, n_units=len(shared), alpha=alpha,
                      pairs=pairs, would_wear_agree=ww)


@dataclass
class CantJudgeSummary:
    n: int
    by_rater: dict[str, int]
    # outfit_key -> distinct raters who could not judge it, most first
    by_outfit: list[tuple[str, int]]
    reasons: dict[str, list[str]]   # outfit_key -> the reasons given

    @property
    def unclear(self) -> list[tuple[str, int]]:
        """Outfits two or more raters could not judge: the card is missing
        something, not the rater."""
        return [(k, c) for k, c in self.by_outfit if c >= 2]


def cant_judge_summary(marks: list[CantJudge]) -> CantJudgeSummary:
    by_rater = Counter(m.labeller for m in marks)
    who: dict[str, set[str]] = defaultdict(set)
    reasons: dict[str, list[str]] = defaultdict(list)
    for m in marks:
        who[m.outfit_key].add(m.labeller)
        if m.reason.strip():
            reasons[m.outfit_key].append(m.reason.strip())
    by_outfit = sorted(((k, len(v)) for k, v in who.items()), key=lambda kv: (-kv[1], kv[0]))
    return CantJudgeSummary(n=len(marks), by_rater=dict(by_rater),
                            by_outfit=by_outfit, reasons=dict(reasons))
