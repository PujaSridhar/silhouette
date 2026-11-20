"""Measured volume against hand-judged volume, on her own photos.

Inputs, all under one photos directory (gitignored - these are photos of a
person and stay on her machine):

    photos/manifest.csv     one row per photo:
                              photo        file name in photos/images/
                              garments     worn garment ids, ';'-separated
                              crown_y      pixel row of the top of the head
                              sole_y       pixel row of the soles
                              heel_cm      optional; default is the listed shoe's heel_cm
    photos/images/          the photos
    photos/<masks>/<stem>/<garment_id>.png   one mask per worn garment
    photos/<masks>/<stem>/sleeves.png        optional, all sleeves in the photo

`<masks>` is `masks` for hand masks (SAM 2 helper) and `masks_model` for the
fine-tuned model's, so the same harness scores both.

What it reports, per zone definition:

  raw         measured minus judged. A systematic offset here is expected and
              is not error: the judged column runs below 1.0 for fitted tops
              (0.80-0.85), and a mask cannot be narrower than the body under
              it. The two scales differ by definition.
  calibrated  leave-one-garment-out linear map from measured onto the judged
              scale (one line for upper-body volumes, one for lower-body, as
              they are read through different zones), then the disagreement. This is the number to hold to
              +/-0.05, because the engine's thresholds live on the judged scale.
  repeat      the same garment measured in different photos. Its spread is the
              measurement's own noise floor, to set against the noise floor of
              her eye from the re-judging protocol.

Everything is a distribution: mean, median, p90 and max of |error|, and the
share inside +/-0.05 and +/-0.10, overall and per engine category.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import Garment, StyleCV
from .volume import DEFINITIONS, BodyScale, measure_garment

__all__ = ["Row", "read_manifest", "measure_photos", "summarise", "loo_calibrate", "report",
           "mask_agreement"]

TARGET = 0.05


@dataclass(frozen=True)
class Row:
    photo: str
    garment_id: str
    category: str
    field: str            # volume | volume_lower
    definition: str
    judged: float
    measured: float | None
    warnings: tuple[str, ...]

    @property
    def error(self) -> float | None:
        return None if self.measured is None else self.measured - self.judged


def _read_mask(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("L")) > 127


def read_manifest(photos: Path) -> list[dict]:
    with open(photos / "manifest.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def _extent(rec: dict) -> tuple[float | None, float | None]:
    """crown_y and sole_y, or None where the manifest leaves one blank."""
    out = []
    for key in ("crown_y", "sole_y"):
        raw = (rec.get(key) or "").strip()
        try:
            out.append(float(raw) if raw else None)
        except ValueError:
            out.append(None)
    return out[0], out[1]


def measure_photos(photos: str | Path, wardrobe: list[Garment], cv: StyleCV,
                   masks: str = "masks", definitions: list[str] | None = None) -> list[Row]:
    photos = Path(photos)
    by_id = {g.id: g for g in wardrobe}
    defs = definitions or list(DEFINITIONS)
    out: list[Row] = []
    for rec in read_manifest(photos):
        stem = Path(rec["photo"]).stem
        ids = [i.strip() for i in rec["garments"].split(";") if i.strip()]
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise ValueError(f"{rec['photo']}: unknown garment ids {unknown}")
        shoe = next((by_id[i] for i in ids if by_id[i].category == "shoe"), None)
        heel = float(rec["heel_cm"]) if rec.get("heel_cm") else (shoe.heel_cm if shoe else 0.0)
        # Every measurement is a fraction of her standing height, so a photo
        # with no crown or no sole has no scale and nothing here is measurable.
        # Feet cropped out of frame is the ordinary case, not a broken manifest,
        # so skip the photo rather than failing the whole run.
        crown, sole = _extent(rec)
        if crown is None or sole is None:
            continue
        scale = BodyScale(crown, sole, cv.height_cm, heel)
        mdir = photos / masks / stem
        sleeves = _read_mask(mdir / "sleeves.png") if (mdir / "sleeves.png").exists() else None
        for gid in ids:
            g = by_id[gid]
            if g.category == "shoe":
                continue
            mpath = mdir / f"{gid}.png"
            if not mpath.exists():
                continue
            mask = _read_mask(mpath)
            for d in defs:
                for fld, m in measure_garment(g.category, mask, scale, cv.frame, d, sleeves).items():
                    judged = g.volume if fld == "volume" else g.volume_lower
                    if judged is None:
                        continue
                    out.append(Row(rec["photo"], gid, g.category, fld, d, judged, m.volume, m.warnings))
    return out


def _dist(errs: list[float]) -> dict:
    if not errs:
        return {"n": 0}
    a = np.abs(np.array(errs))
    return {
        "n": len(errs),
        "bias": float(np.mean(errs)),
        "mean_abs": float(a.mean()),
        "median_abs": float(np.median(a)),
        "p90_abs": float(np.percentile(a, 90)),
        "max_abs": float(a.max()),
        "within_05": float((a <= 0.05 + 1e-9).mean()),
        "within_10": float((a <= 0.10 + 1e-9).mean()),
    }


def _per_garment(rows: list[Row]) -> dict[tuple[str, str], tuple[float, float]]:
    """(garment, field) -> (mean measured over photos, judged)."""
    acc: dict[tuple[str, str], list[float]] = {}
    judged: dict[tuple[str, str], float] = {}
    for r in rows:
        if r.measured is not None:
            acc.setdefault((r.garment_id, r.field), []).append(r.measured)
            judged[(r.garment_id, r.field)] = r.judged
    return {k: (statistics.fmean(v), judged[k]) for k, v in acc.items()}


def loo_calibrate(pairs: dict) -> dict:
    """Leave-one-garment-out least squares judged ~ a + b * measured.

    Returns garment -> calibrated prediction. Needs at least 3 garments; with
    fewer there is nothing to hold out against."""
    keys = list(pairs)
    if len(keys) < 3:
        return {}
    x = np.array([pairs[k][0] for k in keys])
    y = np.array([pairs[k][1] for k in keys])
    out = {}
    for i, k in enumerate(keys):
        m = np.ones(len(keys), dtype=bool)
        m[i] = False
        if np.ptp(x[m]) == 0:
            continue
        b, a = np.polyfit(x[m], y[m], 1)
        out[k] = float(a + b * x[i])
    return out


def summarise(rows: list[Row]) -> dict:
    res: dict = {}
    for d in sorted({r.definition for r in rows}):
        rs = [r for r in rows if r.definition == d]
        failed = [r for r in rs if r.measured is None]
        pairs = _per_garment(rs)
        raw = [m - j for m, j in pairs.values()]
        # Upper-body and lower-body volumes are read through different zones,
        # so each side gets its own calibration line.
        side = {(r.garment_id, r.field): ("lower" if (r.category == "bottom" or r.field ==
                                                     "volume_lower") else "upper") for r in rs}
        cal = {}
        for sd in ("upper", "lower"):
            cal.update(loo_calibrate({k: v for k, v in pairs.items() if side[k] == sd}))
        cal_err = [cal[k] - pairs[k][1] for k in cal]
        by_cat: dict[str, list[float]] = {}
        cat_of = {(r.garment_id, r.field): (r.category if r.field == "volume" else "dress_lower")
                  for r in rs}
        for k in cal:
            by_cat.setdefault(cat_of[k], []).append(cal[k] - pairs[k][1])
        # repeat consistency: same garment, several photos
        multi: dict[tuple[str, str], list[float]] = {}
        for r in rs:
            if r.measured is not None:
                multi.setdefault((r.garment_id, r.field), []).append(r.measured)
        sds = [statistics.stdev(v) for v in multi.values() if len(v) >= 2]
        res[d] = {
            "measurements": len(rs),
            "failed": len(failed),
            "garments": len(pairs),
            "raw": _dist(raw),
            "calibrated": _dist(cal_err),
            "calibrated_by_category": {c: _dist(e) for c, e in sorted(by_cat.items())},
            "repeat_garments": len(sds),
            "repeat_sd_pooled": (float(np.sqrt(np.mean(np.square(sds)))) if sds else None),
            "per_garment": {f"{k[0]}:{k[1]}": {"measured": m, "judged": j,
                                               "calibrated": cal.get(k)}
                            for k, (m, j) in sorted(pairs.items())},
        }
    return res


def _fmt(dist: dict) -> str:
    if not dist.get("n"):
        return "n=0"
    return (f"n={dist['n']:<3} bias {dist['bias']:+.3f}  |err| median {dist['median_abs']:.3f}"
            f"  p90 {dist['p90_abs']:.3f}  max {dist['max_abs']:.3f}"
            f"  within ±0.05 {dist['within_05']:.0%}  ±0.10 {dist['within_10']:.0%}")


def report(summary: dict) -> str:
    lines = []
    for d, s in summary.items():
        lines.append(f"definition: {d}   ({s['garments']} garments, "
                     f"{s['measurements']} measurements, {s['failed']} failed)")
        lines.append(f"  raw         {_fmt(s['raw'])}")
        lines.append(f"  calibrated  {_fmt(s['calibrated'])}")
        for c, dist in s["calibrated_by_category"].items():
            lines.append(f"    {c:<11} {_fmt(dist)}")
        if s["repeat_sd_pooled"] is not None:
            lines.append(f"  repeat      {s['repeat_garments']} garments in 2+ photos, "
                         f"pooled SD {s['repeat_sd_pooled']:.3f}")
        else:
            lines.append("  repeat      no garment appears in 2+ photos yet")
        cal = s["calibrated"]
        if cal.get("n"):
            verdict = ("MEETS" if cal["p90_abs"] <= TARGET else "MISSES")
            lines.append(f"  target      p90 |calibrated error| <= {TARGET}: {verdict}")
        lines.append("")
    lines.append("Raw bias is expected (the judged scale runs below 1.0; a mask cannot).")
    lines.append("Compare repeat SD with the SD from the blind re-judging protocol before")
    lines.append("deciding which input is noisier. With under ~20 garments every number")
    lines.append("here is indicative only.")
    return "\n".join(lines)


def mask_agreement(photos: str | Path, wardrobe: list[Garment], a: str = "masks",
                   b: str = "masks_model") -> dict[str, dict]:
    """IoU between two mask sets on the same photos (hand vs model), per
    engine category. This is the phone-photo check of the model's masks:
    Fashionpedia val is catalogue and street photography, her photos are not."""
    photos = Path(photos)
    by_id = {g.id: g for g in wardrobe}
    per_cat: dict[str, list[float]] = {}
    for rec in read_manifest(photos):
        stem = Path(rec["photo"]).stem
        for gid in [i.strip() for i in rec["garments"].split(";") if i.strip()]:
            pa, pb = photos / a / stem / f"{gid}.png", photos / b / stem / f"{gid}.png"
            if not (pa.exists() and pb.exists()):
                continue
            ma, mb = _read_mask(pa), _read_mask(pb)
            u = (ma | mb).sum()
            per_cat.setdefault(by_id[gid].category, []).append(float((ma & mb).sum() / u) if u else 1.0)
    return {c: {"n": len(v), "mean_iou": statistics.fmean(v), "min_iou": min(v)}
            for c, v in sorted(per_cat.items())}
