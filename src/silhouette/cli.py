"""Command line entry point.

    silhouette outfit   --context office --top 3
    silhouette critique --items top_shirt_linen_white,bot_wide_cream,shoe_sneaker_white
    silhouette gaps     --top 5
    silhouette week     --days office,office,campus,dinner,weekend
    silhouette rules
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .capsule import HAVE_ORTOOLS, plan_week
from .evals.fitting import fit_weights
from .evals.report import CONCLUSIVE_FLOOR, evaluate_labels
from .evals.sampling import coverage_report, next_batch
from .evals.schema import DIMENSIONS, Label, Ratings
from .evals.store import LabelStore
from .evals.synthetic import make_labels
from .explain import render
from .gaps import analyse
from .geometry import compute
from .models import Garment, Outfit, load_style_cv, load_wardrobe
from .rules import evaluate, load_rules
from .solver import best, score

ROOT = Path(__file__).resolve().parents[2]
D_WARDROBE = ROOT / "data" / "wardrobe.csv"
D_CATALOG = ROOT / "data" / "catalog.csv"
D_CV = ROOT / "data" / "style_cv.yaml"
D_RULES = ROOT / "rules" / "core.yaml"
D_LABELS = ROOT / "evals" / "labels" / "labels.jsonl"
D_RUBRIC = ROOT / "evals" / "rubric.md"

RULE = "-" * 78


def _load(args):
    return (
        load_wardrobe(args.wardrobe),
        load_style_cv(args.style_cv),
        load_rules(args.rules_file),
    )


def _assemble(items: list[Garment]) -> Outfit:
    """Build an Outfit from a flat list of garments, inferring slots."""
    slots: dict[str, Garment] = {}
    for g in items:
        if g.category in slots:
            raise SystemExit(f"two items in the same slot: {g.category}")
        slots[g.category] = g
    if "dress" in slots and ("top" in slots or "bottom" in slots):
        raise SystemExit("a dress cannot be combined with a top or bottom")
    return Outfit(
        top=slots.get("top"),
        bottom=slots.get("bottom"),
        dress=slots.get("dress"),
        outer=slots.get("outer"),
        shoe=slots.get("shoe"),
    )


def cmd_outfit(args) -> int:
    wardrobe, cv, rules = _load(args)
    ctx = cv.context(args.context)
    picks = best(wardrobe, cv, ctx, rules, top_n=args.top)
    print(f"\n{cv.owner} / {ctx.name} / goals: {', '.join(cv.goals)}")
    print(f"formality {ctx.formality_floor:.2f}-{ctx.formality_ceiling:.2f}   "
          f"insulation {ctx.clo_min:.2f}-{ctx.clo_max:.2f} clo")
    print(RULE)
    if not picks:
        print("No feasible outfit for this context. Try `critique` on a near miss.")
        return 1
    for i, s in enumerate(picks, 1):
        print(render(s, rank=i))
        print()
    return 0


def cmd_critique(args) -> int:
    wardrobe, cv, rules = _load(args)
    index = {g.id: g for g in wardrobe}
    missing = [i for i in args.items if i not in index]
    if missing:
        raise SystemExit(f"unknown garment ids: {missing}")
    outfit = _assemble([index[i] for i in args.items])
    if args.tucked:
        outfit = Outfit(
            top=outfit.top, bottom=outfit.bottom, dress=outfit.dress,
            outer=outfit.outer, shoe=outfit.shoe, tucked=True,
        )
    ctx = cv.context(args.context)
    s = score(outfit, cv, ctx, rules)
    print(f"\n{outfit.label()}\nfor: {ctx.name}")
    print(RULE)
    print(render(s))
    if not s.feasible:
        print("\n  -> not wearable for this context as assembled.")
    print()
    return 0


def cmd_gaps(args) -> int:
    wardrobe, cv, rules = _load(args)
    catalog = load_wardrobe(args.catalog)
    gaps = analyse(wardrobe, catalog, cv, rules)
    print(f"\nGap analysis - {len(catalog)} candidate purchases against "
          f"{len(wardrobe)} owned garments")
    print("best_delta = how much it raises the best achievable score in any one context")
    print(RULE)
    print(f"{'item':<32}{'delta':>8}{'ctx':>6}{'strong':>8}   best in")
    for g in gaps[: args.top]:
        if g.unlocked:
            where = "unlocks " + ", ".join(g.unlocked)
            delta_s = f"{g.best_delta:>+8.2f}" if g.best_delta > 0 else f"{'unlocks':>8}"
        else:
            where = max(g.per_context, key=lambda k: g.per_context[k]) if g.per_context else "-"
            delta_s = f"{g.best_delta:>+8.2f}"
        print(f"{g.item.name:<32}{delta_s}{g.contexts_won:>6}"
              f"{g.strong_new:>8}   {where}")
    print()
    return 0


def cmd_week(args) -> int:
    wardrobe, cv, rules = _load(args)
    days = [d.strip() for d in args.days.split(",") if d.strip()]
    plan = plan_week(wardrobe, cv, days, rules, max_wears=args.max_wears,
                     prefer=args.backend)
    print(f"\nWeek plan - {plan.backend}"
          f"{' (proven optimal)' if plan.optimal else ''}, "
          f"max {args.max_wears} wears per garment")
    print(f"total score {plan.total:+.2f} across {len(days)} days")
    print(RULE)
    for day, s in zip(plan.days, plan.chosen):
        print(f"{day:<18}{s.total:+.2f}  {s.outfit.label()}")
    print(f"\nmost-worn: ", end="")
    hot = sorted(plan.wear_counts.items(), key=lambda kv: -kv[1])[:5]
    print(", ".join(f"{k} x{v}" for k, v in hot))

    if not plan.within_limits:
        print(f"\n  bottleneck - the wardrobe cannot cover this week at "
              f"{args.max_wears} wears per garment.")
        index = {g.id: g for g in wardrobe}
        for gid, extra in sorted(plan.overages.items(), key=lambda kv: -kv[1]):
            name = index[gid].name if gid in index else gid
            print(f"    {name} is worn {args.max_wears + extra}x "
                  f"({extra} over). A second comparable piece unblocks it.")
    print()
    return 0


def cmd_rules(args) -> int:
    rules = load_rules(args.rules_file)
    soft = [r for r in rules if r.kind == "soft"]
    hard = [r for r in rules if r.kind == "hard"]
    print(f"\n{len(rules)} rules - {len(soft)} soft, {len(hard)} hard")
    print(RULE)
    for r in rules:
        goal = r.goal or "(any)"
        print(f"{r.kind:<5} w={r.weight:<5} {goal:<22} {r.id}")
        if args.verbose and r.note:
            print(f"        {' '.join(r.note.split())}")
    print()
    return 0


# ----------------------------------------------------------------- evaluation


def _outfit_dict(o) -> dict:
    return {
        "top": o.top.id if o.top else None,
        "bottom": o.bottom.id if o.bottom else None,
        "dress": o.dress.id if o.dress else None,
        "outer": o.outer.id if o.outer else None,
        "shoe": o.shoe.id if o.shoe else None,
        "tucked": o.tucked,
    }


def _ask_int(prompt: str) -> int | None:
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("q", "quit"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("    enter 1-5, or q to stop")


def cmd_label(args) -> int:
    """Interactive labelling. The engine's score and explanations are hidden
    on purpose - seeing them anchors the rating and destroys the independence
    the whole harness depends on (evals/rubric.md, protocol step 2)."""
    wardrobe, cv, rules = _load(args)
    store = LabelStore(args.labels)
    tasks = next_batch(wardrobe, cv, rules, store, n=args.n,
                       contexts=args.contexts.split(",") if args.contexts else None,
                       source="expert")
    if not tasks:
        print("Nothing left to label for these contexts.")
        return 0

    print(f"\n{len(tasks)} outfits to rate. Rubric: {args.rubric}")
    print("Scores are hidden deliberately. 1-5 per dimension, q to stop.\n")
    written = 0
    for i, task in enumerate(tasks, 1):
        print(RULE)
        print(f"[{i}/{len(tasks)}]  for: {task.context}")
        print(f"  {task.outfit.label()}")
        ratings: dict[str, int] = {}
        for dim in DIMENSIONS:
            v = _ask_int(f"    {dim:<12} ")
            if v is None:
                print(f"\nStopped. {written} labels written to {args.labels}")
                return 0
            ratings[dim] = v
        overall = _ask_int("    OVERALL      ")
        if overall is None:
            print(f"\nStopped. {written} labels written to {args.labels}")
            return 0
        wear = input("    would wear?  [y/N] ").strip().lower().startswith("y")
        note = input("    note (optional) ").strip()
        store.append(Label(
            id=f"exp_{len(store) + 1:04d}",
            outfit=_outfit_dict(task.outfit),
            context=task.context,
            ratings=Ratings(**ratings),
            overall=overall,
            would_wear=wear,
            labeller=args.labeller,
            source="expert",
            note=note,
        ))
        written += 1
    print(f"\n{written} labels written to {args.labels}")
    return 0


def cmd_bootstrap(args) -> int:
    """Generate synthetic labels to validate the pipeline. Never evidence."""
    wardrobe, cv, rules = _load(args)
    store = LabelStore(args.labels)
    tasks = next_batch(wardrobe, cv, rules, store, n=args.n,
                       contexts=args.contexts.split(",") if args.contexts else None,
                       source="bootstrap")
    labels = make_labels(tasks, rules, noise=args.noise, seed=args.seed)
    for l in labels:
        store.append(l)
    print(f"\n{len(labels)} BOOTSTRAP labels written to {args.labels}")
    print("These exercise the harness. They are not evidence about the engine.")
    return 0


def cmd_eval(args) -> int:
    wardrobe, cv, rules = _load(args)
    store = LabelStore(args.labels)
    pool = store.expert() if args.source == "expert" else store.bootstrap()
    rep = evaluate_labels(pool, wardrobe, cv, rules, store=store, source_used=args.source)

    print(f"\nEvaluation - {args.source} labels")
    print(RULE)
    print(f"  {rep.caveat()}")
    print(f"  labels: {rep.n_expert} expert, {rep.n_bootstrap} bootstrap\n")
    if len(pool) >= 2:
        print(f"  engine score vs label overall      rho = {rep.engine_rho:+.3f}")
        for name, ar in rep.judges.items():
            print(f"  judge {name:<22} {ar.summary()}")
        if rep.self_agreement_qwk is not None:
            print(f"\n  self-agreement (the ceiling)       qwk = {rep.self_agreement_qwk:+.3f}"
                  f"  over {rep.n_repeats} repeats")
        else:
            print(f"\n  self-agreement: not computable - {rep.n_repeats} repeated outfits "
                  "(re-label 10% in a later session)")
        if rep.acceptance:
            yes, seen = rep.acceptance
            print(f"  top-3 acceptance                   {yes}/{seen} labelled suggestions")
        else:
            print("  top-3 acceptance: none of the engine's top suggestions are labelled yet")
    unex = rep.unexercised_rules
    if unex:
        print(f"\n  {len(unex)} rules no label exercises - weights unfalsifiable:")
        for r in unex:
            print(f"      {r}")
    print()
    return 0


def cmd_coverage(args) -> int:
    wardrobe, cv, rules = _load(args)
    store = LabelStore(args.labels)
    src = None if args.source == "all" else args.source
    cov = coverage_report(store, wardrobe, cv, rules, source=src)
    n_labels = len(store.all()) if src is None else len(
        store.expert() if src == "expert" else store.bootstrap()
    )
    print(f"\nLabel coverage per rule - {n_labels} {args.source} labels")
    print(RULE)
    for rid, n in sorted(cov.items(), key=lambda kv: (kv[1], kv[0])):
        bar = "#" * min(40, n)
        flag = "  <- unexercised" if n == 0 else ""
        print(f"{rid:<42}{n:>5}  {bar}{flag}")
    print()
    return 0


def cmd_fit(args) -> int:
    wardrobe, cv, rules = _load(args)
    store = LabelStore(args.labels)
    pool = store.expert() if args.source == "expert" else store.bootstrap()
    if len(pool) < 4:
        print(f"\nOnly {len(pool)} {args.source} labels. Need at least 4 to fit.")
        return 1

    res = fit_weights(pool, wardrobe, cv, rules, folds=args.folds)
    print(f"\nWeight fit - {res.n} {args.source} labels, lambda={res.lam}, "
          f"{res.folds}-fold CV")
    print(RULE)
    if args.source == "bootstrap":
        print("  BOOTSTRAP - this validates the fitting loop, it does not tune "
              "the product.\n")
    if res.folds:
        print(f"  held-out rank correlation   prior {res.cv_rho_prior:+.3f}"
              f"   ->   fitted {res.cv_rho_fitted:+.3f}")
        print(f"  verdict: {'IMPROVEMENT' if res.improved else 'no improvement out of sample'}\n")
    else:
        print("  too few labels to cross-validate; no improvement claimed\n")

    changes = res.changes()
    if not changes:
        print("  no weight moved by more than 0.02 - the priors already fit")
    else:
        print(f"  {'rule':<42}{'prior':>8}{'fitted':>8}{'labels':>8}")
        for rid, pri, fit, sup in changes:
            print(f"  {rid:<42}{pri:>8.2f}{fit:>8.2f}{sup:>8}")
    unsup = res.unsupported()
    if unsup:
        print(f"\n  left at prior (no labels exercise them): {', '.join(unsup)}")

    if args.write:
        if not res.improved:
            print("\n  refusing to write: the fit did not improve out of sample.")
            return 1
        import yaml as _yaml
        raw = _yaml.safe_load(open(args.rules_file))
        new = {rid: float(res.fitted[i]) for i, rid in enumerate(res.rule_ids)}
        for r in raw["rules"]:
            if r["id"] in new:
                r["weight"] = round(new[r["id"]], 3)
        with open(args.rules_file, "w") as fh:
            _yaml.safe_dump(raw, fh, sort_keys=False, width=100)
        print(f"\n  written to {args.rules_file} - re-run the tests")
    print()
    return 0


def cmd_import(args) -> int:
    """Ingest a labelling export from the app.

    The app is where labels actually get made - on a phone, in five-minute
    sittings - and the fitter lives here. This is the join: it writes a
    wardrobe, a Style CV and a labels file that every other command in this
    CLI already understands, so `eval` and `fit` work on real labels with no
    further translation.
    """
    import csv as _csv
    import json as _json
    import yaml as _yaml

    raw = _json.loads(Path(args.file).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["id", "name", "category", "subcategory", "rise", "length", "fit", "volume",
            "volume_lower", "structure", "tuckable", "neckline", "sleeve", "pattern",
            "fabric_weight", "drape", "formality", "clo", "heel_cm", "waist_seam",
            "color_hex", "availability"]
    gpath = out / "wardrobe.csv"
    with open(gpath, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for g in raw.get("garments", []):
            row = {c: g.get(c, "") for c in cols}
            row["tuckable"] = "yes" if g.get("tuckable") else ""
            for k in ("volume_lower", "rise", "length", "waist_seam"):
                if row[k] is None:
                    row[k] = ""
            w.writerow(row)

    cv = raw["styleCV"]
    cvpath = out / "style_cv.yaml"
    with open(cvpath, "w") as fh:
        _yaml.safe_dump({
            "owner": cv.get("name", "imported"),
            "height_cm": cv["heightCm"],
            "version": 1,
            "frame": {"landmarks": cv["landmarks"], "widths": cv["widths"],
                      "confidence": 0.8, "source": "app_entered"},
            "goals": cv.get("goals", []),
            "palette": {},
            "contexts": cv["contexts"],
        }, fh, sort_keys=False)

    lpath = out / "labels.jsonl"
    n = 0
    with open(lpath, "w") as fh:
        for l in raw.get("labels", []):
            fh.write(_json.dumps(l, sort_keys=True) + "\n")
            n += 1

    # Prove it round-trips before claiming success, rather than writing three
    # files and hoping.
    wardrobe = load_wardrobe(gpath)
    cvobj = load_style_cv(cvpath)
    store = LabelStore(lpath)
    labels = store.all()
    bad = []
    index = {g.id: g for g in wardrobe}
    for l in labels:
        for slot in ("top", "bottom", "dress", "outer", "shoe"):
            gid = l.outfit.get(slot)
            if gid and gid not in index:
                bad.append((l.id, gid))
        if l.context not in cvobj.contexts:
            bad.append((l.id, "context:" + l.context))

    print(f"\nImported into {out}")
    print(RULE)
    print(f"  wardrobe   {len(wardrobe)} garments  -> {gpath.name}")
    print(f"  style cv   {cvobj.owner}, {len(cvobj.contexts)} contexts -> {cvpath.name}")
    print(f"  labels     {n} ({sum(1 for l in labels if l.has_dimensions)} with dimensions)"
          f" -> {lpath.name}")
    if bad:
        print(f"\n  {len(bad)} dangling reference(s) - these labels cannot be scored:")
        for lid, ref in bad[:8]:
            print(f"      {lid} -> {ref}")
        return 1
    print("\n  every label resolves against the wardrobe and the Style CV.")
    print(f"\n  next:  silhouette --wardrobe {gpath} --style-cv {cvpath} \\")
    print(f"             --labels {lpath} eval --source expert")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="silhouette", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wardrobe", default=str(D_WARDROBE))
    p.add_argument("--catalog", default=str(D_CATALOG))
    p.add_argument("--style-cv", default=str(D_CV))
    p.add_argument("--rules-file", default=str(D_RULES))
    p.add_argument("--labels", default=str(D_LABELS))
    p.add_argument("--rubric", default=str(D_RUBRIC))
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("outfit", help="best outfits for a context")
    a.add_argument("--context", default="office")
    a.add_argument("--top", type=int, default=3)
    a.set_defaults(fn=cmd_outfit)

    b = sub.add_parser("critique", help="score an outfit you assembled yourself")
    b.add_argument("--items", required=True,
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    b.add_argument("--context", default="weekend")
    b.add_argument("--tucked", action="store_true")
    b.set_defaults(fn=cmd_critique)

    c = sub.add_parser("gaps", help="which single purchase buys the most")
    c.add_argument("--top", type=int, default=8)
    c.set_defaults(fn=cmd_gaps)

    d = sub.add_parser("week", help="plan a week with wear limits (CP-SAT)")
    d.add_argument("--days", default="office,office,campus,dinner,weekend")
    d.add_argument("--max-wears", type=int, default=2)
    d.add_argument("--backend", choices=["auto", "cpsat", "greedy"], default="auto")
    d.set_defaults(fn=cmd_week)

    e = sub.add_parser("rules", help="list the loaded rule set")
    e.add_argument("-v", "--verbose", action="store_true")
    e.set_defaults(fn=cmd_rules)

    f = sub.add_parser("label", help="rate outfits against the rubric")
    f.add_argument("--n", type=int, default=20)
    f.add_argument("--contexts", default="")
    f.add_argument("--labeller", default="puja")
    f.set_defaults(fn=cmd_label)

    g = sub.add_parser("bootstrap", help="synthetic labels to validate the harness")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--contexts", default="")
    g.add_argument("--noise", type=float, default=0.35)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_bootstrap)

    h = sub.add_parser("eval", help="scoreboard against labels")
    h.add_argument("--source", choices=["expert", "bootstrap"], default="expert")
    h.set_defaults(fn=cmd_eval)

    i = sub.add_parser("coverage", help="labels per rule")
    i.add_argument("--source", choices=["expert", "bootstrap", "all"], default="expert")
    i.set_defaults(fn=cmd_coverage)

    j = sub.add_parser("fit", help="fit rule weights to labels")
    j.add_argument("--source", choices=["expert", "bootstrap"], default="expert")
    j.add_argument("--folds", type=int, default=5)
    j.add_argument("--write", action="store_true", help="apply if it improves")
    j.set_defaults(fn=cmd_fit)

    k = sub.add_parser("import-app", help="ingest a labelling export from the app")
    k.add_argument("file", help="the JSON the app's Rate tab exports")
    k.add_argument("--out", default="evals/imported", help="directory to write into")
    k.set_defaults(fn=cmd_import)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
