"""Fitting rule weights to expert labels.

This is what phase 0's finding 10 asked for. The rule *logic* was sound and the
*weights* were guesses: two bonuses totalling 0.95 pushed every suggestion
toward heels and blazers because nothing in the rule set knew that effort has a
cost on a Saturday. Guessing a second time would not have been better. Fitting
them to labels is.

Three design choices make the fit usable rather than merely optimal:

1. **Anchored to the prior.** The objective is
   ``||Xw + b - y||^2 + lambda*||w - w0||^2`` where ``w0`` are the hand-set
   weights. With few labels the fit stays near the stylist's judgement and only
   moves where the data insists. Plain ridge toward zero would quietly delete
   rules that happen to be rare in the sample.

2. **Non-negative.** Weights are magnitudes; direction lives in each rule's
   ``score`` expression, where a penalty returns a negative number. Allowing a
   negative weight would silently invert a rule's meaning while its explanation
   text went on claiming the opposite. Enforced by projected gradient, not by
   clipping afterwards - clipping a converged unconstrained solution does not
   give the constrained optimum.

3. **Cross-validated against the prior.** The fit is only reported as an
   improvement if held-out rank correlation beats the hand-set weights on the
   same folds. A fit that wins on training data and loses out of sample is
   reported as a loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import Garment, StyleCV
from ..rules import Rule
from ..solver import score
from .metrics import spearman
from .sampling import outfit_from_label
from .schema import Label

__all__ = ["FitResult", "design_matrix", "fit_weights"]


def design_matrix(
    labels: list[Label],
    wardrobe: list[Garment],
    cv: StyleCV,
    rules: list[Rule],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Rows are labelled outfits, columns are per-rule raw scores.

    The engine's total is exactly ``w . x`` for these features, so fitting w
    here is fitting the rule weights themselves - not a surrogate model that
    then has to be translated back.
    """
    index = {g.id: g for g in wardrobe}
    rule_ids = [r.id for r in rules if r.kind == "soft"]
    col = {rid: i for i, rid in enumerate(rule_ids)}

    rows: list[np.ndarray] = []
    ys: list[float] = []
    totals: list[float] = []

    for label in labels:
        outfit = outfit_from_label(label, index)
        s = score(outfit, cv, cv.context(label.context), rules)
        x = np.zeros(len(rule_ids), dtype=float)
        for fr in s.verdict.fired:
            if fr.rule.id in col:
                x[col[fr.rule.id]] = fr.score
        rows.append(x)
        ys.append(float(label.overall))
        totals.append(s.total)

    X = np.vstack(rows) if rows else np.zeros((0, len(rule_ids)))
    return X, np.asarray(ys), np.asarray(totals), rule_ids


def _fit_nonneg_anchored(
    X: np.ndarray, y: np.ndarray, w0: np.ndarray, lam: float, iters: int = 4000
) -> tuple[np.ndarray, float]:
    """Projected gradient descent on the anchored ridge objective, w >= 0.

    Step size is 1/L with L the Lipschitz constant of the gradient, which for
    this objective is 2*sigma_max(Xa)^2 + 2*lambda where Xa carries the
    intercept column. Fixed-step projected gradient converges for a convex
    objective on a convex set, and the problem is small enough that 4,000
    iterations is instant and comfortably past convergence.
    """
    n, k = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])          # intercept is not penalised
    theta = np.concatenate([w0, [float(np.mean(y)) if n else 0.0]])

    anchor = np.concatenate([w0, [0.0]])
    mask = np.concatenate([np.ones(k), [0.0]])    # no anchoring on intercept

    smax = float(np.linalg.norm(Xa, 2)) if n else 1.0
    L = 2.0 * smax**2 + 2.0 * lam + 1e-9
    step = 1.0 / L

    for _ in range(iters):
        resid = Xa @ theta - y
        grad = 2.0 * (Xa.T @ resid) + 2.0 * lam * mask * (theta - anchor)
        theta = theta - step * grad
        theta[:k] = np.maximum(theta[:k], 0.0)    # project onto w >= 0

    return theta[:k], float(theta[k])


@dataclass
class FitResult:
    rule_ids: list[str]
    prior: np.ndarray
    fitted: np.ndarray
    intercept: float
    lam: float
    n: int
    support: dict[str, int]
    cv_rho_prior: float
    cv_rho_fitted: float
    folds: int

    @property
    def improved(self) -> bool:
        """Only an improvement if it holds up out of sample."""
        if np.isnan(self.cv_rho_fitted) or np.isnan(self.cv_rho_prior):
            return False
        return self.cv_rho_fitted > self.cv_rho_prior

    def changes(self, min_delta: float = 0.02) -> list[tuple[str, float, float, int]]:
        out = []
        for i, rid in enumerate(self.rule_ids):
            d = float(self.fitted[i] - self.prior[i])
            if abs(d) >= min_delta:
                out.append((rid, float(self.prior[i]), float(self.fitted[i]), self.support.get(rid, 0)))
        out.sort(key=lambda t: -abs(t[2] - t[1]))
        return out

    def unsupported(self) -> list[str]:
        """Rules no label exercised. Their weights are unfalsifiable and the
        fit leaves them at the prior - which must be stated, not hidden."""
        return [rid for rid in self.rule_ids if self.support.get(rid, 0) == 0]


def fit_weights(
    labels: list[Label],
    wardrobe: list[Garment],
    cv: StyleCV,
    rules: list[Rule],
    lambdas: tuple[float, ...] = (0.5, 2.0, 8.0, 32.0),
    folds: int = 5,
    seed: int = 0,
) -> FitResult:
    X, y, _totals, rule_ids = design_matrix(labels, wardrobe, cv, rules)
    by_id = {r.id: r for r in rules}
    w0 = np.asarray([by_id[rid].weight for rid in rule_ids], dtype=float)
    support = {rid: int((X[:, i] != 0).sum()) for i, rid in enumerate(rule_ids)}

    n = X.shape[0]
    if n < 2 * folds:
        # Not enough labels to cross-validate honestly. Fit, but report no win.
        w, b = (_fit_nonneg_anchored(X, y, w0, lambdas[0]) if n else (w0.copy(), 0.0))
        return FitResult(rule_ids, w0, w, b, lambdas[0], n, support,
                         float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    parts = np.array_split(order, folds)

    best = None
    for lam in lambdas:
        rho_fit: list[float] = []
        rho_pri: list[float] = []
        for i in range(folds):
            test = parts[i]
            train = np.concatenate([parts[j] for j in range(folds) if j != i])
            w, b = _fit_nonneg_anchored(X[train], y[train], w0, lam)
            pred_fit = X[test] @ w + b
            pred_pri = X[test] @ w0
            if len(test) >= 2:
                rho_fit.append(spearman(list(pred_fit), list(y[test])))
                rho_pri.append(spearman(list(pred_pri), list(y[test])))
        mf = float(np.nanmean(rho_fit)) if rho_fit else float("nan")
        mp = float(np.nanmean(rho_pri)) if rho_pri else float("nan")
        if best is None or (not np.isnan(mf) and mf > best[1]):
            best = (lam, mf, mp)

    lam, cv_fit, cv_pri = best  # type: ignore[misc]
    w, b = _fit_nonneg_anchored(X, y, w0, lam)
    return FitResult(rule_ids, w0, w, b, lam, n, support, cv_pri, cv_fit, folds)
