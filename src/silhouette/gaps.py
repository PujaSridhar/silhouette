"""Gap analysis: which single purchase buys the most.

Marginal outfit yield. For each candidate item, measure what it adds to a
wardrobe that already exists:

  best_delta    how much it raises the best achievable score in any one
                context. This is the "it solves a problem" signal.
  contexts_won  how many contexts it improves at all. This is the
                "it works everywhere" signal.
  strong_new    how many feasible assemblies containing it clear a quality
                bar. This is the "it combines with what I own" signal.

The three disagree often, and that is the useful part. A black blazer tends to
win on contexts; a single well-cut trouser tends to win on delta. Ranking by
`best_delta` then `contexts_won` favours the item that unblocks something,
which is usually the right advice for a wardrobe with a hole in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Context, Garment, StyleCV
from .rules import Rule
from .solver import candidates, score

__all__ = ["Gap", "analyse"]


@dataclass
class Gap:
    item: Garment
    best_delta: float = 0.0
    contexts_won: int = 0
    contexts_unlocked: int = 0
    best_unlocked: float = float("-inf")
    strong_new: int = 0
    per_context: dict[str, float] = field(default_factory=dict)
    unlocked: list[str] = field(default_factory=list)

    @property
    def rank_key(self) -> tuple:
        unlocked_tie = -round(self.best_unlocked, 4) if self.contexts_unlocked else 0.0
        return (
            -self.contexts_unlocked,
            unlocked_tie,
            -round(self.best_delta, 4),
            -self.contexts_won,
            -self.strong_new,
        )


def _baseline(
    wardrobe: list[Garment], cv: StyleCV, contexts: list[Context], rules: list[Rule]
) -> dict[str, float]:
    outfits = list(candidates(wardrobe, cv.frame))
    best: dict[str, float] = {}
    for ctx in contexts:
        feas = [s.total for s in (score(o, cv, ctx, rules) for o in outfits) if s.feasible]
        best[ctx.name] = max(feas) if feas else float("-inf")
    return best


def analyse(
    wardrobe: list[Garment],
    catalog: list[Garment],
    cv: StyleCV,
    rules: list[Rule],
    context_names: list[str] | None = None,
    strong_threshold: float = 1.50,
) -> list[Gap]:
    contexts = [cv.context(n) for n in (context_names or list(cv.contexts))]
    base = _baseline(wardrobe, cv, contexts, rules)

    gaps: list[Gap] = []
    for cand in catalog:
        gap = Gap(item=cand)
        extended = [*wardrobe, cand]
        # Only assemblies that actually use the candidate can change anything.
        with_cand = [o for o in candidates(extended, cv.frame)
                     if any(g.id == cand.id for g in o.items)]
        for ctx in contexts:
            scored = [score(o, cv, ctx, rules) for o in with_cand]
            feas = [s for s in scored if s.feasible]
            gap.strong_new += sum(1 for s in feas if s.total >= strong_threshold)
            if not feas:
                gap.per_context[ctx.name] = 0.0
                continue
            top = max(s.total for s in feas)
            if base[ctx.name] == float("-inf"):
                gap.per_context[ctx.name] = 0.0
                gap.contexts_unlocked += 1
                gap.contexts_won += 1
                gap.unlocked.append(ctx.name)
                gap.best_unlocked = max(gap.best_unlocked, top)
                continue
            delta = max(0.0, top - base[ctx.name])
            gap.per_context[ctx.name] = delta
            if delta > 1e-9:
                gap.contexts_won += 1
            gap.best_delta = max(gap.best_delta, delta)
        gaps.append(gap)

    gaps.sort(key=lambda g: g.rank_key)
    return gaps
