"""A synthetic rater, for testing the learning loop - never for evidence.

The problem: the harness cannot be validated without labels, and real expert
labels do not exist yet. Two tempting shortcuts are both dishonest.

  Random ratings            make the engine look broken (rho about 0).
  Ratings from the engine   make the engine look perfect (rho about 1).

Neither tells you whether the *harness* works. So instead: a synthetic rater
with **deliberately different, explicitly stated preferences** from the engine.
It scores outfits using the engine's own rule scores but with two weights
changed, in the exact direction phase 0's finding 10 hypothesised:

    heel_extends_the_leg_line     0.35  ->  0.05   (a heel is not a virtue)
    overdressed_for_context       0.50  ->  1.60   (effort has a cost)

That gives a known ground truth. A working harness must *recover* those two
changes from the labels alone. That is a real, falsifiable test of the whole
loop - sampling, design matrix, fitting, cross-validation - and it is what
`tests/test_evals.py::test_fitting_recovers_the_synthetic_rater` asserts.

These labels are written with ``source="bootstrap"`` and the scoreboard refuses
to report quality conclusions from them. They prove the pipeline runs. They say
nothing about whether the engine is any good, and this module must never be
used to suggest otherwise.
"""

from __future__ import annotations

import numpy as np

from ..models import Garment, StyleCV
from ..rules import Rule
from ..solver import ScoredOutfit
from .sampling import Task
from .schema import Label, Ratings

__all__ = ["SYNTHETIC_OVERRIDES", "synthetic_weights", "rate", "make_labels"]

# The known ground truth the harness must recover.
SYNTHETIC_OVERRIDES: dict[str, float] = {
    "heel_extends_the_leg_line": 0.05,
    "overdressed_for_context": 1.60,
}


def synthetic_weights(rules: list[Rule]) -> dict[str, float]:
    return {r.id: SYNTHETIC_OVERRIDES.get(r.id, r.weight) for r in rules if r.kind == "soft"}


def _synthetic_total(scored: ScoredOutfit, weights: dict[str, float]) -> float:
    return sum(weights.get(fr.rule.id, 0.0) * fr.score for fr in scored.verdict.fired)


def rate(
    tasks: list[Task],
    rules: list[Rule],
    noise: float = 0.35,
    seed: int = 0,
) -> list[tuple[Task, int, bool]]:
    """Map synthetic totals onto the 1-5 rubric scale, with rater noise."""
    w = synthetic_weights(rules)
    totals = np.asarray([_synthetic_total(t.scored, w) for t in tasks], dtype=float)
    if len(totals) == 0:
        return []

    lo, hi = float(totals.min()), float(totals.max())
    rng = np.random.default_rng(seed)
    out: list[tuple[Task, int, bool]] = []
    for t, raw in zip(tasks, totals):
        frac = 0.5 if hi <= lo else (raw - lo) / (hi - lo)
        val = 1.0 + 4.0 * frac + rng.normal(scale=noise)
        overall = int(min(5, max(1, round(val))))
        out.append((t, overall, overall >= 4))
    return out


def make_labels(
    tasks: list[Task],
    rules: list[Rule],
    labeller: str = "synthetic-rater",
    noise: float = 0.35,
    seed: int = 0,
) -> list[Label]:
    labels: list[Label] = []
    for i, (task, overall, would_wear) in enumerate(rate(tasks, rules, noise, seed), 1):
        o = task.outfit
        labels.append(
            Label(
                id=f"boot_{i:04d}",
                outfit={
                    "top": o.top.id if o.top else None,
                    "bottom": o.bottom.id if o.bottom else None,
                    "dress": o.dress.id if o.dress else None,
                    "outer": o.outer.id if o.outer else None,
                    "shoe": o.shoe.id if o.shoe else None,
                    "tucked": o.tucked,
                },
                context=task.context,
                # dimension ratings are not modelled; only `overall` carries
                # the synthetic signal, so these are set to the neutral centre
                # rather than invented
                ratings=Ratings(proportion=3, balance=3, formality=3, colour=3, cohesion=3),
                overall=overall,
                would_wear=would_wear,
                labeller=labeller,
                source="bootstrap",
                note="synthetic rater - pipeline validation only, not evidence",
            )
        )
    return labels
