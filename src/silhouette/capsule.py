"""Weekly capsule selection - the part that genuinely needs a solver.

Pick one outfit for each day of a planned week, maximising total styling score,
subject to no garment being worn more than `max_wears` times. Enumeration
cannot do this: the days are coupled through the shared wardrobe, so the best
outfit for Monday may be the wrong choice once Thursday is accounted for.

Formulated as integer programming and handed to CP-SAT:

    maximise   sum over (day d, outfit o) of  score[d][o] * x[d][o]
    subject to sum over o of x[d][o] == 1                for each day d
               sum over (d, o) containing g of x[d][o] <= max_wears   per garment g
               x[d][o] in {0, 1}

Scores are floats, so they are scaled to integers (CP-SAT is exact over
integers). A greedy fallback runs when ortools is unavailable; it respects the
same constraints but makes no optimality claim, and `CapsulePlan.backend` says
which one produced the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Context, Garment, StyleCV
from .rules import Rule
from .solver import ScoredOutfit, candidates, score

__all__ = ["CapsulePlan", "plan_week", "HAVE_ORTOOLS"]

try:  # pragma: no cover - environment dependent
    from ortools.sat.python import cp_model

    HAVE_ORTOOLS = True
except ImportError:  # pragma: no cover
    cp_model = None  # type: ignore[assignment]
    HAVE_ORTOOLS = False

_SCALE = 1000


@dataclass(frozen=True)
class CapsulePlan:
    days: list[str]
    chosen: list[ScoredOutfit]
    total: float
    backend: str
    optimal: bool
    wear_counts: dict[str, int]
    overages: dict[str, int]      # garment id -> wears above the limit

    @property
    def within_limits(self) -> bool:
        return not self.overages


def _pools(
    wardrobe: list[Garment],
    cv: StyleCV,
    contexts: list[Context],
    rules: list[Rule],
) -> list[list[ScoredOutfit]]:
    """Feasible, scored assemblies per day. Candidates are enumerated once."""
    all_outfits = list(candidates(wardrobe, cv.frame))
    pools: list[list[ScoredOutfit]] = []
    for ctx in contexts:
        day = [score(o, cv, ctx, rules) for o in all_outfits]
        pools.append([s for s in day if s.feasible])
    return pools


def plan_week(
    wardrobe: list[Garment],
    cv: StyleCV,
    context_names: list[str],
    rules: list[Rule],
    max_wears: int = 2,
    prefer: str = "auto",
) -> CapsulePlan:
    contexts = [cv.context(n) for n in context_names]
    pools = _pools(wardrobe, cv, contexts, rules)

    for name, pool in zip(context_names, pools):
        if not pool:
            raise ValueError(f"no feasible outfit at all for context {name!r}")

    use_cpsat = HAVE_ORTOOLS if prefer == "auto" else (prefer == "cpsat")
    if use_cpsat and not HAVE_ORTOOLS:
        raise RuntimeError("CP-SAT requested but ortools is not installed")

    if use_cpsat:
        return _solve_cpsat(pools, context_names, max_wears)
    return _solve_greedy(pools, context_names, max_wears)


def _solve_cpsat(
    pools: list[list[ScoredOutfit]], days: list[str], max_wears: int
) -> CapsulePlan:
    """Exact week selection, with slack so infeasibility becomes a diagnosis.

    A hard `sum(uses) <= max_wears` makes an over-subscribed wardrobe simply
    unsolvable, and "no feasible week" is useless advice. Instead each garment
    gets a slack variable and exceeding the limit is penalised heavily enough
    to dominate any styling gain. The solver then always returns a week, and
    `overages` names exactly which garments the wardrobe is short of - which is
    the actionable form of the same fact.
    """
    model = cp_model.CpModel()
    x: list[list] = []
    for d, pool in enumerate(pools):
        x.append([model.NewBoolVar(f"x_{d}_{i}") for i in range(len(pool))])
        model.AddExactlyOne(x[d])

    # Couple the days through the shared wardrobe.
    uses: dict[str, list] = {}
    for d, pool in enumerate(pools):
        for i, s in enumerate(pool):
            for g in s.outfit.items:
                uses.setdefault(g.id, []).append(x[d][i])

    n_days = len(pools)
    over: dict[str, object] = {}
    for gid, vars_ in uses.items():
        over[gid] = model.NewIntVar(0, n_days, f"over_{gid}")
        model.Add(sum(vars_) <= max_wears + over[gid])

    # One unit of overage must cost more than any achievable styling gain, so
    # the solver only ever spends it when there is no alternative.
    penalty = 10 * _SCALE * max(1, n_days)
    model.Maximize(
        sum(
            int(round(s.total * _SCALE)) * x[d][i]
            for d, pool in enumerate(pools)
            for i, s in enumerate(pool)
        )
        - penalty * sum(over.values())
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT found no assignment at all; every day pool was empty")

    chosen = [
        pool[i]
        for d, pool in enumerate(pools)
        for i in range(len(pool))
        if solver.Value(x[d][i])
    ]
    counts: dict[str, int] = {}
    for s in chosen:
        for g in s.outfit.items:
            counts[g.id] = counts.get(g.id, 0) + 1

    return CapsulePlan(
        days=days,
        chosen=chosen,
        total=sum(s.total for s in chosen),
        backend="cp-sat",
        optimal=status == cp_model.OPTIMAL,
        wear_counts=counts,
        overages={g: int(solver.Value(v)) for g, v in over.items() if solver.Value(v) > 0},
    )


def _solve_greedy(
    pools: list[list[ScoredOutfit]], days: list[str], max_wears: int
) -> CapsulePlan:
    """Hardest-day-first greedy. Respects the limit where it can, exceeds it
    where it must, and claims no optimality either way."""
    order = sorted(range(len(pools)), key=lambda d: len(pools[d]))
    counts: dict[str, int] = {}
    picked: dict[int, ScoredOutfit] = {}

    for d in order:
        pool = sorted(pools[d], key=lambda s: -s.total)
        # Same objective as CP-SAT, lexicographically: break the fewest wear
        # limits, then score highest. Taking simply the best-scoring outfit
        # here would optimise a different problem and make the two backends
        # incomparable.
        choice = min(
            pool,
            key=lambda s: (
                sum(1 for g in s.outfit.items if counts.get(g.id, 0) >= max_wears),
                -s.total,
            ),
        )
        picked[d] = choice
        for g in choice.outfit.items:
            counts[g.id] = counts.get(g.id, 0) + 1

    chosen = [picked[d] for d in range(len(pools))]
    return CapsulePlan(
        days=days,
        chosen=chosen,
        total=sum(s.total for s in chosen),
        backend="greedy",
        optimal=False,
        wear_counts=counts,
        overages={g: c - max_wears for g, c in counts.items() if c > max_wears},
    )
