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

import math
from dataclasses import dataclass

from ..models import Garment, Outfit, StyleCV, outfit_slot_key
from ..rules import Rule
from ..solver import ScoredOutfit, candidates, score
from .store import LabelStore

__all__ = ["Task", "outfit_from_label", "next_batch", "coverage_report",
           "SHARED_N", "SHARED_SEED", "fnv1a32", "shared_hash_key", "shared_set"]

N_STRATA = 5

# The shared set: the same outfits for every rater, so agreement between them
# can be measured. 12 is enough for alpha to mean something with three to five
# stylists (12 units x 3 raters = 36 pairable values) and small enough that a
# stylist finishes it in one sitting, mixed in with their usual outfits.
SHARED_N = 12
# Changing the seed changes the set. Ratings already made against the old set
# still count (agreement is computed on whatever overlaps), but new raters
# would stop overlapping with old ones, so bump it only on purpose.
SHARED_SEED = "silhouette.shared.v1"


@dataclass(frozen=True)
class Task:
    outfit: Outfit
    context: str
    scored: ScoredOutfit

    @property
    def key(self) -> str:
        def gid(slot: str) -> str | None:
            g = getattr(self.outfit, slot)
            return g.id if g else None

        return outfit_slot_key(gid) + f"|tucked={self.outfit.tucked}|{self.context}"


def outfit_from_label(label, index: dict[str, Garment]) -> Outfit:
    """Rebuild an Outfit from a stored label so it can be re-scored."""
    def g(slot: str) -> Garment | None:
        gid = label.outfit.get(slot)
        return index.get(str(gid)) if gid else None

    return Outfit(
        top=g("top"), bottom=g("bottom"), dress=g("dress"),
        outer=g("outer"), shoe=g("shoe"), drape=g("drape"),
        tucked=bool(label.outfit.get("tucked")),
    )


def _stratum(total: float, lo: float, hi: float) -> int:
    if hi <= lo:
        return 0
    frac = (total - lo) / (hi - lo)
    return min(N_STRATA - 1, max(0, int(frac * N_STRATA)))


def _labelled_rule_counts(
    store: LabelStore,
    wardrobe: list[Garment],
    cv: StyleCV,
    rules: list[Rule],
    source: str | None = "expert",
) -> dict[str, int]:
    """How many labels exist per rule, by re-scoring each labelled outfit.

    Counts expert labels by default: bootstrap labels come from the synthetic
    rater and must not make a rule look covered. Pass source=None for all."""
    index = {g.id: g for g in wardrobe}
    counts: dict[str, int] = {r.id: 0 for r in rules}
    pool = store.all() if source is None else [l for l in store.all() if l.source == source]
    for label in pool:
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
    source: str | None = "expert",
) -> list[Task]:
    names = contexts or list(cv.contexts)
    done = store.labelled_keys(source=source)

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

    rule_counts = _labelled_rule_counts(store, wardrobe, cv, rules, source=source)
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
    store: LabelStore,
    wardrobe: list[Garment],
    cv: StyleCV,
    rules: list[Rule],
    source: str | None = "expert",
) -> dict[str, int]:
    """Labels per rule. A rule at zero has a weight nothing can falsify.

    Expert labels only by default; source=None counts every label."""
    return _labelled_rule_counts(store, wardrobe, cv, rules, source=source)


def fnv1a32(text: str) -> int:
    """32-bit FNV-1a over the string's code units. Tiny, and trivially the
    same in the app's JavaScript, which is the point: Python's hash() is
    salted per process and random.Random is not portable."""
    h = 0x811C9DC5
    for ch in text:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def shared_hash_key(outfit: Outfit, context: str) -> str:
    """The string the shared set hashes. Matches the app's outfitKeyOf
    exactly (`|t=0|`, not the labels' `|tucked=False|`), so both sides pick
    the same outfits."""
    def gid(slot: str) -> str | None:
        g = getattr(outfit, slot)
        return g.id if g else None

    return outfit_slot_key(gid) + f"|t={int(bool(outfit.tucked))}|{context}"


def shared_set(
    wardrobe: list[Garment],
    cv: StyleCV,
    rules: list[Rule],
    n: int = SHARED_N,
    contexts: list[str] | None = None,
    seed: str = SHARED_SEED,
) -> list[Task]:
    """Outfits every rater rates, so their judgements overlap.

    Deterministic: a function of the wardrobe (its garment ids), the person's
    frame and goals, and the seed - never of what anyone has already rated.
    Two stylists working on the same wardrobe therefore get the same set,
    whatever order they rate in. Spread across the five score quintiles in
    turn, because agreement measured only on mediocre outfits is inflated by
    a restricted range: everyone agrees a so-so outfit is a 3.

    Within a quintile the order is a seeded hash of the outfit key, so the
    pick is arbitrary but fixed. The app's sharedSet() is the same algorithm;
    the parity fixture checks they choose the same keys."""
    names = contexts or list(cv.contexts)
    all_outfits = list(candidates(wardrobe, cv.frame))
    pool: list[tuple[Task, str]] = []
    for name in names:
        ctx = cv.context(name)
        for o in all_outfits:
            s = score(o, cv, ctx, rules)
            if s.feasible:
                pool.append((Task(o, name, s), shared_hash_key(o, name)))
    if not pool or n <= 0:
        return []

    # Rounded so a last-bit float difference between Python and JS cannot
    # move an outfit across a quintile boundary. floor(x + 0.5) is what JS's
    # Math.round does; Python's round() rounds half to even.
    tot = [math.floor(t.scored.total * 1e6 + 0.5) / 1e6 for t, _ in pool]
    lo, hi = min(tot), max(tot)
    strata: list[list[tuple[int, str, Task]]] = [[] for _ in range(N_STRATA)]
    for (t, k), v in zip(pool, tot):
        strata[_stratum(v, lo, hi)].append((fnv1a32(seed + "|" + k), k, t))
    for st in strata:
        st.sort(key=lambda x: (x[0], x[1]))

    chosen: list[Task] = []
    depth = 0
    while len(chosen) < n and any(depth < len(st) for st in strata):
        for st in strata:
            if depth < len(st) and len(chosen) < n:
                chosen.append(st[depth][2])
        depth += 1
    return chosen
