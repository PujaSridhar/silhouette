"""How precise does volume need to be?

Perturb every garment's volume and count the outfits whose set of fired rules
changes. The engine is a set of thresholds, so a volume that sits close to one
flips a rule on a small error. This answers what accuracy a measured volume
has to reach before it is worth substituting for the judged one.

Two noise models, because "perturb by ±0.05" is ambiguous:

    sign     each garment moves by exactly +delta or -delta, sign at random
    uniform  each garment moves by a uniform draw in [-delta, +delta]

`sign` is the worst case at that size; `uniform` is closer to an estimator
whose errors are bounded by delta. Both are reported.

Nothing here changes a rule, a weight or the wardrobe file.
"""

from __future__ import annotations

import dataclasses
import random
import statistics
from collections.abc import Mapping

from ..geometry import compute
from ..models import Garment, Outfit, StyleCV
from ..rules import Rule, evaluate
from ..solver import candidates

__all__ = ["VOLUME_THRESHOLDS", "near_threshold", "fired_sets", "flip_rate",
           "perturbation_study", "replace_volumes"]

# Volume thresholds as they appear in rules/core.yaml, and which side of the
# outfit each one reads. A test checks every one is still present there, so
# this list cannot silently drift from the rules.
VOLUME_THRESHOLDS: dict[float, tuple[str, str]] = {
    1.00: ("upper", "fitted_top_counterweights_volume: upper_volume <= 1.00"),
    1.15: ("upper", "volume_on_volume_loses_the_line: upper_volume > 1.15"),
    1.30: ("lower", "two balance rules: lower_volume >= 1.30"),
}

_SLOTS = ("top", "bottom", "dress", "outer", "shoe")


def _side_values(g: Garment, side: str | None) -> list[float]:
    """The volumes of `g` that feed upper_volume or lower_volume (None: all)."""
    if side is None:
        return [g.volume] + ([g.volume_lower] if g.volume_lower is not None else [])
    if side == "upper":
        return [g.volume] if g.category in ("top", "outer", "dress") else []
    return ([g.volume] if g.category == "bottom"
            else [g.lower_vol] if g.category == "dress" else [])


def near_threshold(wardrobe: list[Garment], threshold: float, within: float = 0.10,
                   side: str | None = None, inclusive: bool = True) -> list[str]:
    """Garments whose judged volume is within `within` of a threshold.

    `side` restricts to the garments whose volume that threshold actually reads
    ("upper": tops, outers, a dress's bodice; "lower": bottoms, a dress's
    skirt). Without it, shoes count too, and no rule reads a shoe's volume.

    `inclusive` counts a garment exactly `within` away (0.90 or 1.10 against
    1.00). Without the tolerance, float arithmetic decides those cases
    arbitrarily (1.10 - 1.00 is 0.10000000000000009), so say which you mean."""
    out = []
    for g in wardrobe:
        if any((abs(v - threshold) <= within + 1e-9) if inclusive
               else (abs(v - threshold) < within - 1e-9) for v in _side_values(g, side)):
            out.append(g.id)
    return out


def _swap(o: Outfit, by_id: Mapping[str, Garment]) -> Outfit:
    return dataclasses.replace(
        o, **{k: by_id[getattr(o, k).id] for k in _SLOTS if getattr(o, k) is not None})


def fired_sets(outfits: list[Outfit], by_id: Mapping[str, Garment], cv: StyleCV,
               rules: list[Rule], context: str) -> list[frozenset[str]]:
    ctx = cv.context(context)
    out = []
    for o in outfits:
        o2 = _swap(o, by_id)
        f = compute(o2, cv.frame, cv.height_cm)
        out.append(frozenset(fr.rule.id for fr in evaluate(o2, f, cv, ctx, rules).fired))
    return out


def flip_rate(base: list[frozenset[str]], other: list[frozenset[str]]) -> float:
    return sum(a != b for a, b in zip(base, other)) / len(base)


def replace_volumes(wardrobe: list[Garment], volumes: Mapping[str, Mapping[str, float]]
                    ) -> dict[str, Garment]:
    """Garments by id, with `volume` / `volume_lower` replaced where given."""
    out = {}
    for g in wardrobe:
        v = volumes.get(g.id, {})
        out[g.id] = dataclasses.replace(
            g,
            volume=v.get("volume", g.volume),
            volume_lower=v.get("volume_lower", g.volume_lower),
        )
    return out


def perturbation_study(wardrobe: list[Garment], cv: StyleCV, rules: list[Rule],
                       deltas=(0.05, 0.10), modes=("sign", "uniform"), seeds: int = 200,
                       context: str = "weekend") -> dict:
    """Flip rates over every structural candidate outfit.

    Context only gates the hard formality and warmth rules and the soft
    overdressed rule, none of which reads volume, so one context is enough.
    """
    outfits = list(candidates(wardrobe, cv.frame))
    by_id = {g.id: g for g in wardrobe}
    base = fired_sets(outfits, by_id, cv, rules, context)
    res: dict = {"outfits": len(outfits), "seeds": seeds, "runs": {}}
    for d in deltas:
        for mode in modes:
            rates = []
            for seed in range(seeds):
                rng = random.Random(seed)

                def e() -> float:
                    return d * rng.choice((-1.0, 1.0)) if mode == "sign" else rng.uniform(-d, d)

                vols = {g.id: {"volume": g.volume + e(),
                               **({"volume_lower": g.volume_lower + e()}
                                  if g.volume_lower is not None else {})}
                        for g in wardrobe}
                rates.append(flip_rate(base, fired_sets(outfits, replace_volumes(wardrobe, vols),
                                                        cv, rules, context)))
            qs = statistics.quantiles(rates, n=20) if len(rates) >= 2 else rates * 19
            res["runs"][(d, mode)] = {
                "mean": statistics.fmean(rates), "p5": qs[0], "p95": qs[-1],
                "min": min(rates), "max": max(rates),
            }
    return res
