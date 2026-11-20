"""`silhouette-vision`: W1 commands. Kept apart from `silhouette` so the core
CLI never imports anything heavier than numpy.

    silhouette-vision sensitivity [--seeds 200] [--volumes measured.csv]
    silhouette-vision fashionpedia ANNOTATIONS [--licence commercial]
    silhouette-vision prep-photos [--photos photos]
    silhouette-vision measure --photos photos [--masks masks] [--out evals/volume/rows.csv]
    silhouette-vision compare-masks --photos photos
    silhouette-vision annotate IMAGE (--garment ID | --sleeves | --person) --box x0,y0,x1,y1 ...
    silhouette-vision predict --model runs/w1/model --photos photos
    silhouette-vision rejudge-sheet / rejudge [SHEET]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _engine(args):
    from ..models import load_style_cv, load_wardrobe
    from ..rules import load_rules
    return (load_wardrobe(args.wardrobe), load_style_cv(args.style_cv), load_rules(args.rules_file))


def cmd_sensitivity(args) -> int:
    from .sensitivity import (VOLUME_THRESHOLDS, fired_sets, flip_rate, near_threshold,
                              perturbation_study, replace_volumes)
    from ..solver import candidates
    wardrobe, cv, rules = _engine(args)
    print(f"{len(wardrobe)} garments")
    print("garments within 0.10 of a volume threshold "
          "(any garment / only those the rule reads):")
    for th, (side, what) in VOLUME_THRESHOLDS.items():
        n = len(near_threshold(wardrobe, th, 0.10))
        read = len(near_threshold(wardrobe, th, 0.10, side=side))
        print(f"  {th:.2f}: {n:>2} / {read:>2}   {what}")
    if args.volumes:
        vols: dict[str, dict[str, float]] = {}
        with open(args.volumes, newline="") as fh:
            for r in csv.DictReader(fh):
                vols[r["garment_id"]] = {k: float(r[k]) for k in ("volume", "volume_lower")
                                         if r.get(k)}
        outfits = list(candidates(wardrobe, cv.frame))
        base = fired_sets(outfits, {g.id: g for g in wardrobe}, cv, rules, args.context)
        new = fired_sets(outfits, replace_volumes(wardrobe, vols), cv, rules, args.context)
        print(f"\nreplacing {len(vols)} garments' volumes from {args.volumes}: "
              f"{flip_rate(base, new):.1%} of {len(outfits)} outfits change which rules fire")
        return 0
    res = perturbation_study(wardrobe, cv, rules, seeds=args.seeds, context=args.context)
    print(f"\n{res['outfits']} candidate outfits, {res['seeds']} random draws each")
    for (d, mode), r in res["runs"].items():
        print(f"  ±{d:.2f} {mode:<8} {r['mean']:.1%} change which rules fire "
              f"(90% of draws {r['p5']:.1%}-{r['p95']:.1%})")
    return 0


def cmd_fashionpedia(args) -> int:
    from .fashionpedia import class_counts, load
    s = load(args.annotations, licence=args.licence)
    print(f"{len(s)} images with a W1 class (licence group: {args.licence})")
    for k, v in class_counts(s).items():
        print(f"  {v:>7}  {k}")
    return 0


def cmd_measure(args) -> int:
    from ..models import load_style_cv, load_wardrobe
    from .volume_eval import measure_photos, report, summarise
    rows = measure_photos(args.photos, load_wardrobe(args.wardrobe), load_style_cv(args.style_cv),
                          masks=args.masks)
    if not rows:
        print("no measurements: check photos/manifest.csv and the mask folder")
        return 1
    s = summarise(rows)
    print(report(s))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["photo", "garment_id", "field", "definition", "judged", "measured",
                        "error", "warnings"])
            for r in rows:
                w.writerow([r.photo, r.garment_id, r.field, r.definition, r.judged,
                            "" if r.measured is None else f"{r.measured:.4f}",
                            "" if r.error is None else f"{r.error:+.4f}", " | ".join(r.warnings)])
        (out.with_suffix(".json")).write_text(json.dumps(s, indent=2))
        print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


def cmd_compare_masks(args) -> int:
    from ..models import load_wardrobe
    from .volume_eval import mask_agreement
    res = mask_agreement(args.photos, load_wardrobe(args.wardrobe), args.a, args.b)
    if not res:
        print(f"no garment has a mask in both {args.a}/ and {args.b}/")
        return 1
    for c, r in res.items():
        print(f"  {c:<7} n={r['n']:<3} mean IoU {r['mean_iou']:.3f}  worst {r['min_iou']:.3f}")
    return 0


def cmd_prep_photos(args) -> int:
    from .photos import prep_photos
    r = prep_photos(args.photos)
    print(f"prepared {Path(args.photos) / 'images'}")
    if r["zips_removed"]:
        print(f"  unzipped       {r['unzipped']} image(s) from {r['zips_removed']} zip(s)"
              f" (ignored {r['skipped_zip_entries']} non-image entries)")
    if r["converted_heic"]:
        print(f"  converted      {r['converted_heic']} HEIC file(s) -> JPEG")
    print(f"  manifest       +{r['added_to_manifest']} new row(s), "
          f"{r['total_photos']} photo(s) total in {Path(args.photos) / 'manifest.csv'}")
    return 0


def cmd_annotate(args) -> int:
    from PIL import Image

    from .photos import open_rgb
    from .sam import Segmenter, overlay, parse_box, parse_point, save_mask, union_into
    img_path = Path(args.image)
    img = open_rgb(img_path)
    seg = Segmenter(args.checkpoint)
    seg.set_image(img, str(img_path))
    mask = seg.segment([parse_box(b) for b in args.box], [parse_point(p) for p in args.point])
    mdir = Path(args.photos) / "masks" / img_path.stem
    if args.person:
        from .volume import person_extent_from_mask
        crown, sole = person_extent_from_mask(mask)
        print(f"crown_y={crown:.0f} sole_y={sole:.0f}  (copy into photos/manifest.csv)")
        target = mdir / "person.png"
    elif args.sleeves:
        target = mdir / "sleeves.png"
        mask = union_into(target, mask)
    else:
        target = mdir / f"{args.garment}.png"
    save_mask(target, mask)
    Image.fromarray(overlay(img, mask)).save(target.with_suffix(".preview.jpg"))
    print(f"wrote {target} ({int(mask.sum())} px) and a preview next to it - check the preview")
    return 0


def cmd_rejudge_sheet(args) -> int:
    from ..models import load_wardrobe
    from .rejudge import make_sheet
    out = make_sheet(load_wardrobe(args.wardrobe), args.out, seed=args.seed)
    print(f"wrote {out}: fill session 1 now, cover it, session 2 in two to three weeks")
    return 0


def cmd_rejudge(args) -> int:
    from ..models import load_wardrobe
    from .rejudge import analyse
    r = analyse(args.sheet, load_wardrobe(args.wardrobe))
    if not r["garments"]:
        print("no garment has both sessions filled in yet")
        return 1
    print(f"{r['garments']} garments judged twice")
    print(f"  one judgement's SD (volume)      {r['volume_sd_one_judgement']:.3f}")
    print(f"  session 2 minus session 1, mean  {r['volume_mean_shift']:+.3f}")
    print(f"  largest disagreement             {r['volume_max_abs_diff']:.3f}")
    print(f"  pairs within ±0.05               {r['share_within_05']:.0%}")
    if r["vs_wardrobe_csv_sd"] is not None:
        print(f"  session 1 vs wardrobe.csv SD     {r['vs_wardrobe_csv_sd']:.3f}")
    if r["structure_sd_one_judgement"] is not None:
        print(f"  one judgement's SD (structure)   {r['structure_sd_one_judgement']:.3f}")
    print("\nCompare the volume SD with the measured repeat SD from `measure`.")
    return 0


def cmd_predict(args) -> int:
    from transformers import Mask2FormerForUniversalSegmentation

    from ..models import load_wardrobe
    from .train import predict_photos
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model)
    for n in predict_photos(model, args.photos, load_wardrobe(args.wardrobe), args.size,
                            args.threshold):
        print(n)
    print(f"wrote masks to {Path(args.photos) / 'masks_model'}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="silhouette-vision", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wardrobe", default=str(ROOT / "data" / "wardrobe.csv"))
    p.add_argument("--style-cv", default=str(ROOT / "data" / "style_cv.yaml"))
    p.add_argument("--rules-file", default=str(ROOT / "rules" / "core.yaml"))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sensitivity", help="how precise volume needs to be")
    s.add_argument("--seeds", type=int, default=200)
    s.add_argument("--context", default="weekend")
    s.add_argument("--volumes", help="CSV garment_id,volume[,volume_lower] to substitute")
    s.set_defaults(fn=cmd_sensitivity)

    f = sub.add_parser("fashionpedia", help="class counts in an annotation file")
    f.add_argument("annotations")
    f.add_argument("--licence", default="all", choices=["all", "commercial"])
    f.set_defaults(fn=cmd_fashionpedia)

    pp = sub.add_parser("prep-photos", help="unzip, convert HEIC -> JPEG, fill manifest.csv")
    pp.add_argument("--photos", default=str(ROOT / "photos"))
    pp.set_defaults(fn=cmd_prep_photos)

    m = sub.add_parser("measure", help="measured vs hand-judged volume on photos")
    m.add_argument("--photos", default=str(ROOT / "photos"))
    m.add_argument("--masks", default="masks", help="masks (hand) or masks_model")
    m.add_argument("--out", default=str(ROOT / "evals" / "volume" / "rows.csv"))
    m.set_defaults(fn=cmd_measure)

    cm = sub.add_parser("compare-masks", help="IoU of model masks against hand masks")
    cm.add_argument("--photos", default=str(ROOT / "photos"))
    cm.add_argument("--a", default="masks")
    cm.add_argument("--b", default="masks_model")
    cm.set_defaults(fn=cmd_compare_masks)

    a = sub.add_parser("annotate", help="SAM 2 mask from a box or clicks")
    a.add_argument("image")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--garment", help="wardrobe id of the garment you are pointing at")
    g.add_argument("--sleeves", action="store_true", help="add to this photo's sleeve mask")
    g.add_argument("--person", action="store_true", help="whole person, prints crown/sole")
    a.add_argument("--box", action="append", default=[], help="x0,y0,x1,y1 (repeatable)")
    a.add_argument("--point", action="append", default=[], help="x,y or x,y,- (repeatable)")
    a.add_argument("--photos", default=str(ROOT / "photos"))
    a.add_argument("--checkpoint", default="facebook/sam2.1-hiera-small")
    a.set_defaults(fn=cmd_annotate)

    rs = sub.add_parser("rejudge-sheet", help="blank blind re-judging sheet")
    rs.add_argument("--out", default=str(ROOT / "evals" / "rejudge" / "sheet.csv"))
    rs.add_argument("--seed", type=int, default=None)
    rs.set_defaults(fn=cmd_rejudge_sheet)

    rj = sub.add_parser("rejudge", help="noise floor from a filled sheet")
    rj.add_argument("sheet", nargs="?", default=str(ROOT / "evals" / "rejudge" / "sheet.csv"))
    rj.set_defaults(fn=cmd_rejudge)

    r = sub.add_parser("predict", help="run the fine-tuned model on photos/")
    r.add_argument("--model", required=True)
    r.add_argument("--photos", default=str(ROOT / "photos"))
    r.add_argument("--size", type=int, default=640)
    r.add_argument("--threshold", type=float, default=0.3)
    r.set_defaults(fn=cmd_predict)

    args = p.parse_args(argv)
    if args.cmd == "annotate" and not (args.box or args.point):
        p.error("annotate needs at least one --box or --point")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
