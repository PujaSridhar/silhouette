"""Export the engine's data and a PARITY FIXTURE for the browser demo.

The demo re-implements geometry and the rule conditions in JavaScript so the
frame sliders can recompute live. A second implementation is a second chance to
be wrong, and there is no JS runtime here to diff them offline.

So this exporter also writes a fixture: a spread of outfit/context pairs with
the Python engine's exact answers. The page replays them through its own JS
engine on load and reports any mismatch on the page itself. If the port drifts,
the demo says so out loud rather than quietly showing wrong numbers.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from silhouette.geometry import compute                      # noqa: E402
from silhouette.models import Outfit, load_style_cv, load_wardrobe  # noqa: E402
from silhouette.rules import evaluate, load_rules            # noqa: E402
from silhouette.solver import candidates, score              # noqa: E402

# Python format specs -> compact directives the page's tiny formatter handles.
SPEC = {".0%": "pct0", ".0f": "f0", ".2f": "f2", ".3f": "f3", "+.3f": "s3"}


def convert_template(t: str) -> str:
    """'{lift:.0%}' -> '{lift|pct0}', so the JS side needs no format parser."""
    def sub(m: re.Match) -> str:
        name, spec = m.group(1), m.group(2) or ""
        if not spec:
            return "{" + name + "}"
        key = SPEC.get(spec.lstrip(":"))
        if key is None:
            raise SystemExit(f"unhandled format spec {spec!r} in: {t}")
        return "{" + name + "|" + key + "}"

    return re.sub(r"\{(\w+)(:[^}]+)?\}", sub, " ".join(t.split()))


def drape_parity_picks(wardrobe, drapes, cv, rules):
    """(scored outfit, context name) pairs that exercise the drape slot.

    The sample wardrobe has no drape, and adding one to it would change every
    number measured on it. So drape cases come from the sample wardrobe plus
    data/wardrobe_drapes.csv, kept apart: for every rule, one draped outfit it
    fires or rejects on in each context where it can (so a score that depends
    on the context, like dupatta_completes_the_set, is checked across the
    range), then the two best draped outfits per context so the arithmetic is
    checked where drape rules stack with the others.
    """
    draped = [o for o in candidates(wardrobe + drapes, cv.frame) if o.drape is not None]
    picks: list = []
    seen: set[str] = set()

    def take(s, name: str) -> None:
        key = f"{s.outfit.key}|{name}"
        if key in seen:
            return
        seen.add(key)
        picks.append((s, name))

    # Scored lazily and memoised: scoring every draped outfit in every
    # context would multiply the build time several times over, and the rule
    # scan below usually finds its hit early.
    memo: dict[tuple[str, int], object] = {}

    def scored(name: str, i: int):
        if (name, i) not in memo:
            memo[(name, i)] = score(draped[i], cv, cv.context(name), rules)
        return memo[(name, i)]

    def hit_ids(s) -> set[str]:
        return {fr.rule.id for fr in s.verdict.fired} | {x.id for x, _ in s.verdict.violations}

    # Rules that can only fire on a draped outfit, plus every rule a drape
    # can change the outcome of through formality, warmth or pattern.
    for r in rules:
        for name in cv.contexts:
            for i in range(len(draped)):
                s = scored(name, i)
                if r.id in hit_ids(s):
                    take(s, name)
                    break
    # High scorers, from a fixed stride through the pool rather than all of it.
    for name in cv.contexts:
        sample = [scored(name, i) for i in range(0, len(draped), 7)]
        for s in sorted(sample, key=lambda s: -s.total)[:2]:
            take(s, name)
    return picks


def main() -> int:
    wardrobe = load_wardrobe(ROOT / "data" / "wardrobe.csv")
    drapes = load_wardrobe(ROOT / "data" / "wardrobe_drapes.csv")
    catalog = load_wardrobe(ROOT / "data" / "catalog.csv")
    cv = load_style_cv(ROOT / "data" / "style_cv.yaml")
    rules = load_rules(ROOT / "rules" / "core.yaml")

    bundle = {
        "styleCV": {
            "owner": cv.owner,
            "heightCm": cv.height_cm,
            "landmarks": cv.frame.landmarks,
            "widths": cv.frame.widths,
            "goals": cv.goals,
            "contexts": [asdict(c) for c in cv.contexts.values()],
        },
        "garments": [asdict(g) for g in wardrobe],
        # Only the parity check reads these. They are not part of the demo's
        # wardrobe, so the demo's own suggestions stay as they were.
        "parityGarments": [asdict(g) for g in drapes],
        "catalog": [asdict(g) for g in catalog],
        "rules": [
            {
                "id": r.id,
                "kind": r.kind,
                "weight": r.weight,
                "goal": r.goal,
                "explain": convert_template(r.explain) if r.explain else None,
                "reject": convert_template(r.reject) if r.reject else None,
                "note": " ".join(r.note.split()) if r.note else None,
            }
            for r in rules
        ],
        "parity": [],
    }

    # A fixture that spreads across every rule and every context, so a mistake
    # in any single translated condition shows up.
    store_keys: set[str] = set()
    fixture: list[dict] = []
    all_outfits = list(candidates(wardrobe, cv.frame))
    # Memoised so a rule that never fires (a drape rule, on this wardrobe)
    # costs one pass over the outfits, not one per context per rule.
    _memo: dict = {}

    def scored_at(name: str, i: int):
        if (name, i) not in _memo:
            _memo[(name, i)] = score(all_outfits[i], cv, cv.context(name), rules)
        return _memo[(name, i)]

    rule_seen: dict[str, int] = {r.id: 0 for r in rules}

    for name in cv.contexts:
        ctx = cv.context(name)
        scored = [scored_at(name, i) for i in range(len(all_outfits))]
        scored.sort(key=lambda s: -s.total)
        # top, bottom and middle of the ranking, plus infeasible examples
        picks = scored[:6] + scored[len(scored) // 2 : len(scored) // 2 + 4] + scored[-4:]
        for s in picks:
            o = s.outfit
            key = f"{o.key}|{name}"
            if key in store_keys:
                continue
            store_keys.add(key)
            f = s.features
            fixture.append({
                "outfit": {
                    "top": o.top.id if o.top else None,
                    "bottom": o.bottom.id if o.bottom else None,
                    "dress": o.dress.id if o.dress else None,
                    "outer": o.outer.id if o.outer else None,
                    "shoe": o.shoe.id if o.shoe else None,
                    "drape": o.drape.id if o.drape else None,
                    "tucked": o.tucked,
                },
                "context": name,
                "breakRatio": f.break_ratio,
                "lift": f.lift,
                "hemDrop": f.hem_drop,
                "contrast": f.contrast_at_break,
                "upperVolume": f.upper_volume,
                "lowerVolume": f.lower_volume,
                "total": s.total,
                "feasible": s.feasible,
                "fired": sorted(fr.rule.id for fr in s.verdict.fired),
                "violated": sorted(r.id for r, _ in s.verdict.violations),
            })
            for fr in s.verdict.fired:
                rule_seen[fr.rule.id] += 1
            for r, _ in s.verdict.violations:
                rule_seen[r.id] += 1

    # Second pass: make sure EVERY reachable rule is in the fixture. The
    # ranking slices above happen to miss a few, and an unexercised rule is an
    # unverified translation - exactly the thing the fixture exists to catch.
    def add(scored_outfit, ctx_name: str) -> None:
        o = scored_outfit.outfit
        key = f"{o.key}|{ctx_name}"
        if key in store_keys:
            return
        store_keys.add(key)
        f = scored_outfit.features
        fixture.append({
            "outfit": {
                "top": o.top.id if o.top else None,
                "bottom": o.bottom.id if o.bottom else None,
                "dress": o.dress.id if o.dress else None,
                "outer": o.outer.id if o.outer else None,
                "shoe": o.shoe.id if o.shoe else None,
                "drape": o.drape.id if o.drape else None,
                "tucked": o.tucked,
            },
            "context": ctx_name,
            "breakRatio": f.break_ratio,
            "lift": f.lift,
            "hemDrop": f.hem_drop,
            "contrast": f.contrast_at_break,
            "upperVolume": f.upper_volume,
            "lowerVolume": f.lower_volume,
            "total": scored_outfit.total,
            "feasible": scored_outfit.feasible,
            "fired": sorted(fr.rule.id for fr in scored_outfit.verdict.fired),
            "violated": sorted(r.id for r, _ in scored_outfit.verdict.violations),
        })
        for fr in scored_outfit.verdict.fired:
            rule_seen[fr.rule.id] += 1
        for r, _ in scored_outfit.verdict.violations:
            rule_seen[r.id] += 1

    for target in [r.id for r in rules]:
        if rule_seen[target] >= 2:
            continue
        found = 0
        for name in cv.contexts:
            if found >= 2:
                break
            for i in range(len(all_outfits)):
                s2 = scored_at(name, i)
                hits = {fr.rule.id for fr in s2.verdict.fired} | {
                    r.id for r, _ in s2.verdict.violations
                }
                if target in hits:
                    add(s2, name)
                    found += 1
                    break

    n_plain = len(fixture)
    for s3, name in drape_parity_picks(wardrobe, drapes, cv, rules):
        add(s3, name)
    n_drape = len(fixture) - n_plain

    bundle["parity"] = fixture

    out = ROOT / "demo" / "engine-data.json"
    out.write_text(json.dumps(bundle, separators=(",", ":"), sort_keys=True))

    uncovered = [r for r, n in rule_seen.items() if n == 0]
    print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size // 1024} KB)")
    print(f"  garments {len(wardrobe)}  catalog {len(catalog)}  rules {len(rules)}")
    print(f"  parity fixture: {len(fixture)} outfit/context pairs ({n_drape} with a drape)")
    print(f"  rules exercised by the fixture: {len(rule_seen) - len(uncovered)}/{len(rule_seen)}")
    if uncovered:
        print(f"  NOT exercised: {', '.join(sorted(uncovered))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
