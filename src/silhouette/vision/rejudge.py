"""The noise floor of a hand-judged volume: the same eye, twice, weeks apart.

Measured volume can only be compared with judged volume up to the judge's own
consistency. If her eye wanders by ±0.08 between sittings, no mask can agree
with the CSV to ±0.05 however good it is, and the right comparison is the
mask's photo-to-photo spread against hers. This produces the sheet and reads
it back. The protocol is docs/volume-rejudging-protocol.md.

    silhouette-vision rejudge-sheet --out evals/rejudge/sheet.csv
    silhouette-vision rejudge evals/rejudge/sheet.csv
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path

from ..models import Garment

__all__ = ["DEFAULT_GARMENTS", "make_sheet", "analyse"]

# Ten garments chosen to sit on or near the 1.00 / 1.15 / 1.30 thresholds,
# with two far from any (0.85 and 1.95) as anchors.
DEFAULT_GARMENTS = (
    "top_tee_blk", "top_blouse_puff_rust", "top_blouse_struct_cream", "out_blazer_blk",
    "bot_jean_straight_indigo", "bot_jean_bootcut_mid", "top_kurta_indigo",
    "top_shirt_linen_white", "out_jacket_denim", "bot_wide_blk",
)

FIELDS = ["order", "garment_id", "name", "session_1_date", "session_1_volume",
          "session_1_structure", "session_2_date", "session_2_volume", "session_2_structure"]


def make_sheet(wardrobe: list[Garment], out: str | Path, ids=DEFAULT_GARMENTS, seed: int | None = None):
    """A blank sheet in shuffled order, with names but no current values."""
    by_id = {g.id: g for g in wardrobe}
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for k, gid in enumerate(ids, 1):
            w.writerow({"order": k, "garment_id": gid, "name": by_id[gid].name})
    return out


def _sd_of_one_judgement(a: list[float], b: list[float]) -> float | None:
    """Two sittings of the same items: SD(a-b)/sqrt(2) is one sitting's SD."""
    d = [x - y for x, y in zip(a, b)]
    if len(d) < 2:
        return None
    return math.sqrt(sum(x * x for x in d) / len(d) / 2)


def analyse(sheet: str | Path, wardrobe: list[Garment]) -> dict:
    by_id = {g.id: g for g in wardrobe}
    with open(sheet, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("session_1_volume") and r.get("session_2_volume")]
    s1 = [float(r["session_1_volume"]) for r in rows]
    s2 = [float(r["session_2_volume"]) for r in rows]
    csv_v = [by_id[r["garment_id"]].volume for r in rows]
    out = {
        "garments": len(rows),
        "volume_sd_one_judgement": _sd_of_one_judgement(s1, s2),
        "volume_mean_shift": statistics.fmean([b - a for a, b in zip(s1, s2)]) if rows else None,
        "volume_max_abs_diff": max((abs(b - a) for a, b in zip(s1, s2)), default=None),
        "share_within_05": (sum(abs(b - a) <= 0.05 + 1e-9 for a, b in zip(s1, s2)) / len(rows))
        if rows else None,
        "vs_wardrobe_csv_sd": _sd_of_one_judgement(s1, csv_v),
    }
    st = [r for r in rows if r.get("session_1_structure") and r.get("session_2_structure")]
    out["structure_sd_one_judgement"] = _sd_of_one_judgement(
        [float(r["session_1_structure"]) for r in st], [float(r["session_2_structure"]) for r in st])
    return out
