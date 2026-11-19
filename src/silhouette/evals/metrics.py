"""Agreement metrics, implemented directly so the definitions are inspectable.

All of these are small and standard. Writing them out rather than importing
scipy keeps the dependency surface at numpy and, more usefully, means the exact
variant in use is visible - "kappa" in particular means three different things
depending on the weighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "rank", "spearman", "pearson", "quadratic_weighted_kappa",
    "mae", "exact_agreement", "within_one", "AgreementReport", "agreement",
]


def rank(xs: list[float]) -> list[float]:
    """Ranks with ties averaged, which is what Spearman requires."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    if len(a) < 2:
        return float("nan")
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    xs, ys = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((xs**2).sum()) * float((ys**2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((xs * ys).sum() / denom)


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation. The headline metric: does the engine ORDER outfits the
    way the stylist does? Ordering is what matters for a top-3 suggestion;
    absolute score calibration does not."""
    return pearson(rank(list(a)), rank(list(b)))


def quadratic_weighted_kappa(a: list[int], b: list[int], lo: int = 1, hi: int = 5) -> float:
    """Cohen's kappa with quadratic weights.

    Quadratic rather than linear or unweighted because being two points apart on
    a 1-5 rubric is much worse than being one apart, and unweighted kappa treats
    those identically. This is the standard choice for ordinal rubric scoring.
    """
    if len(a) != len(b):
        raise ValueError("length mismatch")
    n_cat = hi - lo + 1
    if len(a) < 2:
        return float("nan")

    O = np.zeros((n_cat, n_cat), dtype=float)
    for x, y in zip(a, b):
        O[int(x) - lo, int(y) - lo] += 1.0

    hist_a = O.sum(axis=1)
    hist_b = O.sum(axis=0)
    E = np.outer(hist_a, hist_b) / max(1.0, O.sum())

    idx = np.arange(n_cat, dtype=float)
    W = (idx[:, None] - idx[None, :]) ** 2 / (n_cat - 1) ** 2

    num = float((W * O).sum())
    den = float((W * E).sum())
    if den == 0.0:
        return float("nan")
    return 1.0 - num / den


def mae(a: list[float], b: list[float]) -> float:
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean())


def exact_agreement(a: list[int], b: list[int]) -> float:
    return float(np.mean([int(x) == int(y) for x, y in zip(a, b)]))


def within_one(a: list[int], b: list[int]) -> float:
    return float(np.mean([abs(int(x) - int(y)) <= 1 for x, y in zip(a, b)]))


@dataclass(frozen=True)
class AgreementReport:
    n: int
    spearman: float
    qwk: float
    mae: float
    exact: float
    within_one: float

    def summary(self) -> str:
        if self.n < 2:
            return f"n={self.n} - not enough labels to compute agreement"
        # A constant predictor has zero variance, so rank correlation is
        # undefined, not zero. Say so rather than printing nan.
        rho = "  n/a  " if math.isnan(self.spearman) else f"{self.spearman:+.3f}"
        return (
            f"n={self.n}  rho={rho}  qwk={self.qwk:+.3f}  "
            f"mae={self.mae:.2f}  exact={self.exact:.0%}  within1={self.within_one:.0%}"
        )


def agreement(predicted: list[float], actual: list[int]) -> AgreementReport:
    """Agreement between a continuous prediction and integer rubric ratings.

    Spearman and MAE use the raw prediction. The categorical metrics need
    integers, so the prediction is rounded and clamped to the 1-5 scale.
    """
    if not predicted:
        return AgreementReport(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    rounded = [int(min(5, max(1, round(p)))) for p in predicted]
    return AgreementReport(
        n=len(actual),
        spearman=spearman(predicted, [float(a) for a in actual]),
        qwk=quadratic_weighted_kappa(rounded, list(actual)),
        mae=mae(predicted, [float(a) for a in actual]),
        exact=exact_agreement(rounded, list(actual)),
        within_one=within_one(rounded, list(actual)),
    )
