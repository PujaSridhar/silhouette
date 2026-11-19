"""Judges: things that predict a human rating.

A judge is only interesting if it is validated against expert labels, and only
honest if it is compared to a baseline that requires no intelligence at all.
Hence MeanJudge: it predicts the average rating for everything. Any judge that
cannot beat it is adding nothing, and reporting a "rho of 0.4" without that
comparison hides the question.

Note the direction of validation: a judge is scored against EXPERT LABELS,
never against the engine's own total. A judge that agrees with the engine has
told you only that two of your own components agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ..solver import ScoredOutfit

__all__ = ["Judge", "MeanJudge", "ScoreJudge", "LLMJudge", "JUDGE_PROMPT"]


class Judge(Protocol):
    name: str

    def predict(self, scored: ScoredOutfit) -> float:
        """Predicted overall rating, 1-5."""
        ...


@dataclass
class MeanJudge:
    """The baseline that requires no intelligence: predict the mean, always."""

    mean: float = 3.0
    name: str = "mean-baseline"

    def predict(self, scored: ScoredOutfit) -> float:
        return self.mean

    @classmethod
    def fit(cls, ratings: list[int]) -> "MeanJudge":
        return cls(mean=float(np.mean(ratings)) if ratings else 3.0)


@dataclass
class ScoreJudge:
    """Affine map from the engine's own total to a 1-5 rating.

    This is the judge that answers the question the project actually cares
    about: *does the engine's score predict what a stylist thinks?* If this
    beats MeanJudge by a clear margin, the rule set is capturing something
    real. If it does not, no amount of downstream modelling will save it.
    """

    a: float = 1.0
    b: float = 2.5
    name: str = "engine-score"

    def predict(self, scored: ScoredOutfit) -> float:
        return float(min(5.0, max(1.0, self.a * scored.total + self.b)))

    @classmethod
    def fit(cls, totals: list[float], ratings: list[int]) -> "ScoreJudge":
        if len(totals) < 2:
            return cls()
        A = np.vstack([np.asarray(totals, dtype=float), np.ones(len(totals))]).T
        (a, b), *_ = np.linalg.lstsq(A, np.asarray(ratings, dtype=float), rcond=None)
        return cls(a=float(a), b=float(b))


JUDGE_PROMPT = """\
You are grading an outfit against a styling rubric. You are given the wearer's
Style CV, the occasion, and the garments. You are NOT given the engine's score
or its explanations - your judgement must be independent of them.

Rate 1-5 on each dimension, then an independent overall 1-5, then would_wear.
Return JSON only: {{"proportion":n,"balance":n,"formality":n,"colour":n,
"cohesion":n,"overall":n,"would_wear":true|false,"note":"..."}}

Describe garments and geometry. Never describe the wearer's body as a problem.

Rubric: {rubric}

Style CV: {style_cv}
Occasion: {context}
Outfit: {outfit}
"""


@dataclass
class LLMJudge:
    """Wraps a callable that takes a prompt and returns the rubric JSON.

    Deliberately not wired to a provider. The harness must run, and the
    baselines must be computable, with no API key and no network - otherwise
    the eval suite stops being something you run on every change.

    Validation requirement before this judge is used for anything: report its
    quadratic-weighted kappa against expert labels, alongside its own
    self-agreement at temperature 0 on a repeated subset. An unvalidated judge
    is decoration.
    """

    complete: Callable[[str], str]
    rubric: str = ""
    name: str = "llm-judge"

    def predict(self, scored: ScoredOutfit) -> float:
        raise NotImplementedError(
            "LLMJudge needs a provider wired in, plus a validation run against "
            "expert labels. See docs/adr/0004-eval-harness.md."
        )
