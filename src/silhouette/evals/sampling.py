"""Which outfits to label next.

Labelling is the expensive resource in this project - a human eye, fifty
outfits at a sitting. Sampling uniformly at random wastes it: most random
assemblies are mediocre in the same way, and whole rules go unobserved.

This sampler greedily maximises two things at once:

  rule coverage     prefer outfits that fire rules with few labels so far.
                    A rule with no labels has an unfalsifiable weight.
  score spread      prefer outfits from under-filled score quintiles. Labels
                    clustered at the top measure nothing about the bottom,
                    and the ranking metric needs both ends.

Both matter because the labels have two jobs: validating the ranking
(needs spread) and fitting the per-rule weights (needs per-rule coverage).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Garment, Outfit, StyleCV
from ..rules import Rule
from ..solver import ScoredOutfit, candidates, score
from .store import LabelStore

__all__ = ["Task", "outfit_from_label", "next_batch", "coverage_report"]

N_STRATA = 5


@dataclass(frozen=True)
class Task:
    outfit: Outfit
    context: str
    scored: ScoredOutfit

    @property
    def key(self) -> str:
        slots = (
            self.outfit.top, self.outfit.bottom, self.outfit.dress,
            self.outfit.outer, self.outfit.shoe,
        )
        ids = [g.id if g else "-" for g in slots]
        return "|".join(ids) + f"|tucked={self.outfit.tucked}|{self.context}"


def outfit_from_label(label, index: dict[str, Garment]) -> Outfit:
    """Rebuild an Outfit from a stored label so it can be re-scored."""
    def g(slot: str) -> Garment | None:
        gid = label.outfit.get(slot)
        return index.get(str(gid)) if gid else None

    return Outfit(
        top=g("top"), bottom=g("bottom"), dress=g("dress"),
        outer=g("outer"), shoe=g("shoe"),
        tucked=bool(label.outfit.get("tucked")),
    )


def _stratum(total: float, lo: float, hi: float) -> int:
    if hi <= lo:
        return 0
    frac = (total - lo) / (hi - lo)
    return min(N_STRATA - 1, max(0, int(frac * N_STRATA)))


def _labelled_rule_counts(
    store: LabelStore, wardrobe: list[Garment], cv: StyleCV, rules: list[Rule]
) -> dict[str, int]:
    """How many labels exist per rule, by re-scoring each labelled outfit."""
    index = {g.id: g for g in wardrobe}
    counts: dict[str, int] = {r.id: 0 for r in rules}
    for label in store.all():
        try:
            o = outfit_from_label(label, index)
            s = score(o, cv, cv.context(label.context), rules)
        except Exception:
            continue
        for fr in s.verdict.fired:
            counts[fr.rule.id] = counts.get(fr.rule.id, 0) + 1
    return counts


def next_batch(
    wardrobe: list[Garment],
    cv: StyleCV,
    rules: list[Rule],
    store: LabelStore,
    n: int = 20,
    contexts: list[str] | None = None,
    feasible_only: bool = True,
) -> list[Task]:
    names = contexts or list(cv.contexts)
    done = store.labelled_keys()

    pool: list[Task] = []
    all_outfits = list(candidates(wardrobe, cv.frame))
    for name in names:
        ctx = cv.context(name)
        for o in all_outfits:
            s = score(o, cv, ctx, rules)
            if feasible_only and not s.feasible:
                continue
            t = Task(o, name, s)
            if t.key not in done:
                pool.append(t)
    if not pool:
        return []

    totals = [t.scored.total for t in pool]
    lo, hi = min(totals), max(totals)

    rule_counts = _labelled_rule_counts(store, wardrobe, cv, rules)
    stratum_counts: dict[int, int] = {i: 0 for i in range(N_STRATA)}

    chosen: list[Task] = []
    remaining = list(pool)
    for _ in range(min(n, len(pool))):
        best_t, best_u = None, float("-inf")
        for t in remaining:
            fired = [fr.rule.id for fr in t.scored.verdict.fired]
            # 1/(1+count) so the first few labels for a rule are worth most
            u = sum(1.0 / (1.0 + rule_counts.get(rid, 0)) for rid in fired)
            st = _stratum(t.scored.total, lo, hi)
            u += 2.0 / (1.0 + stratum_counts[st])
            if u > best_u:
                best_t, best_u = t, u
        assert best_t is not None
        chosen.append(best_t)
        remaining.remove(best_t)
        for fr in best_t.scored.verdict.fired:
            rule_counts[fr.rule.id] = rule_counts.get(fr.rule.id, 0) + 1
        stratum_counts[_stratum(best_t.scored.total, lo, hi)] += 1

    return chosen


def coverage_report(
    store: LabelStore, wardrobe: list[Garment], cv: StyleCV, rules: list[Rule]
) -> dict[str, int]:
    """Labels per rule. A rule at zero has a weight nothing can falsify."""
    return _labelled_rule_counts(store, wardrobe, cv, rules)
