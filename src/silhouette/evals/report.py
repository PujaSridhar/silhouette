"""The scoreboard.

Two things this refuses to do, both deliberate:

1. **Draw conclusions from bootstrap labels.** They validate the pipeline.
   They are not evidence about the engine, and a scoreboard that blurred the
   two would be worse than no scoreboard.
2. **Report a metric as conclusive below the sample floor.** Below 60 expert
   labels, numbers are printed with their sample size and explicitly marked
   indicative. A rank correlation on 12 labels has a confidence interval wide
   enough to contain both "excellent" and "useless".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Garment, StyleCV
from ..rules import Rule
from ..solver import score
from .judge import Judge, MeanJudge, ScoreJudge
from .metrics import AgreementReport, agreement, quadratic_weighted_kappa, spearman
from .sampling import coverage_report, outfit_from_label
from .schema import Label
from .store import LabelStore

__all__ = ["EvalReport", "evaluate_labels", "CONCLUSIVE_FLOOR"]

CONCLUSIVE_FLOOR = 60


@dataclass
class EvalReport:
    n_expert: int
    n_bootstrap: int
    engine_rho: float
    judges: dict[str, AgreementReport] = field(default_factory=dict)
    self_agreement_qwk: float | None = None
    n_repeats: int = 0
    acceptance: tuple[int, int] | None = None     # (yes, labelled) among top-3
    coverage: dict[str, int] = field(default_factory=dict)
    source_used: str = "expert"

    @property
    def conclusive(self) -> bool:
        return self.source_used == "expert" and self.n_expert >= CONCLUSIVE_FLOOR

    @property
    def unexercised_rules(self) -> list[str]:
        return sorted([r for r, n in self.coverage.items() if n == 0])

    def caveat(self) -> str:
        if self.source_used == "bootstrap":
            return (
                "BOOTSTRAP LABELS - pipeline validation only. These numbers say "
                "nothing about whether the engine is any good."
            )
        if self.n_expert == 0:
            return "No expert labels yet. Run `silhouette label` to start."
        if not self.conclusive:
            return (
                f"INDICATIVE ONLY - {self.n_expert} expert labels, "
                f"floor for a conclusion is {CONCLUSIVE_FLOOR}."
            )
        return f"Conclusive: {self.n_expert} expert labels."


def evaluate_labels(
    labels: list[Label],
    wardrobe: list[Garment],
    cv: StyleCV,
    rules: list[Rule],
    store: LabelStore | None = None,
    source_used: str = "expert",
) -> EvalReport:
    index = {g.id: g for g in wardrobe}
    totals: list[float] = []
    overalls: list[int] = []
    scoreds = []

    for label in labels:
        s = score(outfit_from_label(label, index), cv, cv.context(label.context), rules)
        scoreds.append(s)
        totals.append(s.total)
        overalls.append(label.overall)

    engine_rho = spearman(totals, [float(o) for o in overalls]) if len(labels) >= 2 else float("nan")

    judges: dict[str, AgreementReport] = {}
    if len(labels) >= 2:
        for j in (MeanJudge.fit(overalls), ScoreJudge.fit(totals, overalls)):
            preds = [j.predict(s) for s in scoreds]
            judges[j.name] = agreement(preds, overalls)

    self_qwk: float | None = None
    n_rep = 0
    if store is not None:
        pairs = store.repeats()
        n_rep = len(pairs)
        if n_rep >= 2:
            self_qwk = quadratic_weighted_kappa(
                [a.overall for a, _ in pairs], [b.overall for _, b in pairs]
            )

    acceptance: tuple[int, int] | None = None
    if labels:
        by_key = {l.outfit_key: l for l in labels}
        yes = seen = 0
        from ..solver import best as _best
        for name in cv.contexts:
            for s in _best(wardrobe, cv, cv.context(name), rules, top_n=3):
                o = s.outfit
                ids = [
                    (o.top.id if o.top else "-"), (o.bottom.id if o.bottom else "-"),
                    (o.dress.id if o.dress else "-"), (o.outer.id if o.outer else "-"),
                    (o.shoe.id if o.shoe else "-"),
                ]
                key = "|".join(ids) + f"|tucked={o.tucked}|{name}"
                if key in by_key:
                    seen += 1
                    yes += int(by_key[key].would_wear)
        acceptance = (yes, seen) if seen else None

    cov = coverage_report(store, wardrobe, cv, rules) if store is not None else {}

    return EvalReport(
        n_expert=sum(1 for l in labels if l.source == "expert"),
        n_bootstrap=sum(1 for l in labels if l.source == "bootstrap"),
        engine_rho=engine_rho,
        judges=judges,
        self_agreement_qwk=self_qwk,
        n_repeats=n_rep,
        acceptance=acceptance,
        coverage=cov,
        source_used=source_used,
    )
