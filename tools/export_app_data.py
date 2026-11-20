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

from export_demo_data import convert_template, drape_parity_picks  # noqa: E402
from silhouette.evals.raters import inter_rater_agreement  # noqa: E402
from silhouette.evals.sampling import (  # noqa: E402
    SHARED_N, SHARED_SEED, shared_hash_key, shared_set)
from silhouette.evals.schema import label_from_dict  # noqa: E402
from silhouette.models import load_style_cv, load_wardrobe  # noqa: E402
from silhouette.rules import load_rules                  # noqa: E402
from silhouette.solver import candidates, score          # noqa: E402


def main() -> int:
    wardrobe = load_wardrobe(ROOT / "data" / "wardrobe.csv")
    drapes = load_wardrobe(ROOT / "data" / "wardrobe_drapes.csv")
    cv = load_style_cv(ROOT / "data" / "style_cv.yaml")
    rules = load_rules(ROOT / "rules" / "core.yaml")

    rule_seen = {r.id: 0 for r in rules}
    fixture: list[dict] = []
    used_ids: set[str] = set()
    seen_keys: set[str] = set()
    all_outfits = list(candidates(wardrobe, cv.frame))

    # Memoised so a rule that never fires (a drape rule, on this wardrobe)
    # costs one pass over the outfits, not one per context per rule.
    _memo: dict = {}

    def scored_at(name: str, i: int):
        if (name, i) not in _memo:
            _memo[(name, i)] = score(all_outfits[i], cv, cv.context(name), rules)
        return _memo[(name, i)]


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
                  "drape": o.drape.id if o.drape else None,
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
            for i in range(len(all_outfits)):
                s = scored_at(name, i)
                hits = {fr.rule.id for fr in s.verdict.fired} | {r.id for r, _ in s.verdict.violations}
                if target in hits:
                    add(s, name)
                    found += 1
                    break
    # a few high scorers, so the arithmetic is checked where many rules stack
    for name in list(cv.contexts)[:3]:
        top = sorted((scored_at(name, i) for i in range(len(all_outfits))),
                     key=lambda s: -s.total)[:2]
        for s in top:
            add(s, name)

    # The drape slot, on the sample wardrobe plus the sample drapes.
    n_plain = len(fixture)
    for s, name in drape_parity_picks(wardrobe, drapes, cv, rules):
        add(s, name)
    n_drape = len(fixture) - n_plain

    # The shared set: the same outfits for every rater. The app must pick the
    # same keys as sampling.shared_set or stylists rating in the app and the
    # eval would disagree about which outfits are shared. Checked on the whole
    # sample wardrobe, and on a small draped one so the drape part of the key
    # is covered too (all of it with drapes would multiply the build time).
    drape_kit = {"top_kurta_indigo", "top_cami_silk_blk", "top_tee_blk", "bot_salwar_white",
                 "bot_palazzo_floral", "bot_wide_blk", "dress_saree_teal",
                 "dress_anarkali_maroon", "out_blazer_blk", "shoe_heel_blk_block",
                 "shoe_sandal_tan_flat"}
    small = [g for g in wardrobe if g.id in drape_kit] + drapes
    shared_cases = []
    for gs in (wardrobe, small):
        shared_cases.append({
            "garments": [g.id for g in gs],
            "keys": [shared_hash_key(t.outfit, t.context) for t in shared_set(gs, cv, rules)],
        })
    shared_garments = {g.id: asdict(g) for g in wardrobe + drapes}

    # Agreement between raters, on a fixed handful of ratings: three raters,
    # a repeat (which must not count twice) and an outfit only one rated
    # (which must drop out).
    def lab(i, who, key, overall, wear, when):
        top, bot, shoe, ctx = key
        return {"id": f"ag{i}", "outfit": {"top": top, "bottom": bot, "dress": None,
                "outer": None, "shoe": shoe, "drape": None, "tucked": False},
                "context": ctx, "overall": overall, "ratings": None, "would_wear": wear,
                "labeller": who, "source": "expert", "note": "", "engine_version": 1,
                "created_at": f"2026-11-20T10:{when:02d}:00+00:00"}
    u = [("top_tee_blk", "bot_wide_blk", "shoe_sneaker_white", "weekend"),
         ("top_kurta_indigo", "bot_salwar_white", "shoe_sandal_tan_flat", "dinner"),
         ("top_cami_silk_blk", "bot_skirt_pencil_blk", "shoe_heel_blk_block", "office"),
         ("top_hoodie_oversized_grey", "bot_jean_low_wash", "shoe_sneaker_white", "campus"),
         ("top_shirt_linen_white", "bot_wide_cream", "shoe_heel_blk_block", "office")]
    ag_labels = [
        lab(1, "asha", u[0], 4, True, 1), lab(2, "bea", u[0], 5, True, 2),
        lab(3, "chen", u[0], 4, True, 3), lab(4, "asha", u[1], 2, False, 4),
        lab(5, "bea", u[1], 3, False, 5), lab(6, "asha", u[2], 3, True, 6),
        lab(7, "chen", u[2], 1, False, 7), lab(8, "bea", u[3], 2, False, 8),
        lab(9, "chen", u[3], 2, True, 9), lab(10, "asha", u[4], 5, True, 10),
        lab(11, "asha", u[0], 1, False, 30),   # asha's repeat: ignored here
    ]
    ir = inter_rater_agreement([label_from_dict(d) for d in ag_labels])

    bundle = {
        "shared": {"n": SHARED_N, "seed": SHARED_SEED},
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
            "garments": [asdict(g) for g in wardrobe + drapes if g.id in used_ids],
            "cases": fixture,
            "shared": {"garments": shared_garments, "cases": shared_cases},
            "agreement": {"labels": ag_labels, "alpha": ir.alpha, "n_units": ir.n_units,
                          "would_wear": ir.would_wear_agree},
        },
    }

    out = ROOT / "demo" / "app-data.json"
    out.write_text(json.dumps(bundle, separators=(",", ":"), sort_keys=True))
    uncovered = [r for r, n in rule_seen.items() if n == 0]
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")
    print(f"  rules {len(rules)}  parity cases {len(fixture)} ({n_drape} with a drape)  "
          f"garments referenced {len(used_ids)}")
    print(f"  shared set {SHARED_N}: cases {len(shared_cases)}  agreement alpha {ir.alpha:+.3f} "
          f"on {ir.n_units} outfits")
    print(f"  rules exercised: {len(rule_seen) - len(uncovered)}/{len(rule_seen)}"
          + (f"  (not: {', '.join(uncovered)})" if uncovered else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
