"""Export the compact bundle the PRODUCT needs.

Smaller than the showcase bundle: the app runs on the user's own wardrobe, so
it needs the rule set and a parity fixture, not a seed wardrobe. The fixture
carries its own garments and its own context definitions so it stays valid even
though the product renames occasions for a general audience.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from export_demo_data import convert_template            # noqa: E402
from silhouette.models import load_style_cv, load_wardrobe  # noqa: E402
from silhouette.rules import load_rules                  # noqa: E402
from silhouette.solver import candidates, score          # noqa: E402


def main() -> int:
    wardrobe = load_wardrobe(ROOT / "data" / "wardrobe.csv")
    cv = load_style_cv(ROOT / "data" / "style_cv.yaml")
    rules = load_rules(ROOT / "rules" / "core.yaml")

    rule_seen = {r.id: 0 for r in rules}
    fixture: list[dict] = []
    used_ids: set[str] = set()
    seen_keys: set[str] = set()
    all_outfits = list(candidates(wardrobe, cv.frame))

    def add(s, ctx_name: str) -> None:
        o = s.outfit
        key = f"{o.key}|{ctx_name}"
        if key in seen_keys:
            return
        seen_keys.add(key)
        f = s.features
        for g in o.items:
            used_ids.add(g.id)
        fixture.append({
            "o": {"top": o.top.id if o.top else None,
                  "bottom": o.bottom.id if o.bottom else None,
                  "dress": o.dress.id if o.dress else None,
                  "outer": o.outer.id if o.outer else None,
                  "shoe": o.shoe.id if o.shoe else None,
                  "tucked": o.tucked},
            "c": ctx_name,
            "br": f.break_ratio, "lf": f.lift, "hd": f.hem_drop,
            "ct": f.contrast_at_break, "uv": f.upper_volume, "lv": f.lower_volume,
            "tt": s.total, "fe": s.feasible,
            "fi": sorted(fr.rule.id for fr in s.verdict.fired),
            "vi": sorted(r.id for r, _ in s.verdict.violations),
        })
        for fr in s.verdict.fired:
            rule_seen[fr.rule.id] += 1
        for r, _ in s.verdict.violations:
            rule_seen[r.id] += 1

    # two cases per rule, which is enough to catch a mistranslated predicate
    for target in [r.id for r in rules]:
        found = 0
        for name in cv.contexts:
            if found >= 2:
                break
            ctx = cv.context(name)
            for o in all_outfits:
                s = score(o, cv, ctx, rules)
                hits = {fr.rule.id for fr in s.verdict.fired} | {r.id for r, _ in s.verdict.violations}
                if target in hits:
                    add(s, name)
                    found += 1
                    break
    # a few high scorers, so the arithmetic is checked where many rules stack
    for name in list(cv.contexts)[:3]:
        ctx = cv.context(name)
        top = sorted((score(o, cv, ctx, rules) for o in all_outfits),
                     key=lambda s: -s.total)[:2]
        for s in top:
            add(s, name)

    bundle = {
        "rules": [
            {"id": r.id, "kind": r.kind, "weight": r.weight, "goal": r.goal,
             "explain": convert_template(r.explain) if r.explain else None,
             "reject": convert_template(r.reject) if r.reject else None}
            for r in rules
        ],
        "parity": {
            "frame": {"landmarks": cv.frame.landmarks, "widths": cv.frame.widths,
                      "heightCm": cv.height_cm},
            "goals": cv.goals,
            "contexts": {n: asdict(c) for n, c in cv.contexts.items()},
            "garments": [asdict(g) for g in wardrobe if g.id in used_ids],
            "cases": fixture,
        },
    }

    out = ROOT / "demo" / "app-data.json"
    out.write_text(json.dumps(bundle, separators=(",", ":"), sort_keys=True))
    uncovered = [r for r, n in rule_seen.items() if n == 0]
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")
    print(f"  rules {len(rules)}  parity cases {len(fixture)}  "
          f"garments referenced {len(used_ids)}")
    print(f"  rules exercised: {len(rule_seen) - len(uncovered)}/{len(rule_seen)}"
          + (f"  (not: {', '.join(uncovered)})" if uncovered else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
