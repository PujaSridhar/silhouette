"""W1: loader, measured volume, metrics, sensitivity and the volume harness.

None of this needs a GPU. The Mask2Former smoke test runs only when torch,
transformers and scipy are installed (the [vision] extra), and is skipped
otherwise.
"""

from __future__ import annotations

import csv
import importlib.util
import json

import numpy as np
import pytest

from silhouette.vision import categories as C
from silhouette.vision.metrics import IoUAccumulator
from silhouette.vision.sensitivity import (
    VOLUME_THRESHOLDS, fired_sets, flip_rate, near_threshold, perturbation_study,
    replace_volumes,
)
from silhouette.vision.volume import (
    DEFINITIONS, BodyScale, body_width_frac, measure, measure_garment,
    person_extent_from_mask, row_widths,
)

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PIL = pytest.importorskip("PIL")


# ------------------------------------------------------------- categories

def test_train_space_is_14_garments_plus_sleeve():
    assert len(C.TRAIN_CLASSES) == 15
    assert C.TRAIN_CLASSES[-1] == "sleeve"
    assert {C.train_to_engine(i) for i in range(14)} == {"top", "outer", "bottom", "dress", "shoe"}
    assert C.train_to_engine(14) is None
    assert C.source_to_train(13) is None          # glasses are not trained on


def test_commercial_licence_group_excludes_noncommercial_and_unknown():
    com = C.licence_group("commercial")
    assert not com & {3, 4, 5, 11}                # BY-NC-*, all-rights-reserved
    with pytest.raises(KeyError):
        C.licence_group("nope")


# ----------------------------------------------------------- fashionpedia

def _rle_encode(mask: np.ndarray) -> dict:
    """COCO compressed RLE, a port of rleToString, to round-trip the decoder."""
    flat = mask.T.flatten()
    counts, cur, n = [], False, 0
    for v in flat:
        if bool(v) == cur:
            n += 1
        else:
            counts.append(n)
            cur, n = bool(v), 1
    counts.append(n)
    s = []
    for i, x in enumerate(counts):
        if i > 2:
            x -= counts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            s.append(chr(c + 48))
    return {"size": list(mask.shape), "counts": "".join(s)}


def _fp_file(tmp_path, licence=0):
    m = np.zeros((40, 30), dtype=bool)
    m[5:25, 8:20] = True
    m[30:35, 2:4] = True
    ann = {
        "images": [{"id": 1, "width": 30, "height": 40, "file_name": "a.jpg", "license": licence},
                   {"id": 2, "width": 30, "height": 40, "file_name": "b.jpg", "license": 4}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 6, "bbox": [8, 5, 12, 20], "area": 250,
             "iscrowd": 0, "segmentation": _rle_encode(m), "attribute_ids": [132]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100,
             "iscrowd": 0, "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]},
            {"id": 3, "image_id": 1, "category_id": 13, "bbox": [0, 0, 2, 2], "area": 4,
             "iscrowd": 0, "segmentation": [[0, 0, 2, 0, 2, 2]]},
            {"id": 4, "image_id": 2, "category_id": 10, "bbox": [0, 0, 5, 5], "area": 25,
             "iscrowd": 0, "segmentation": [[0, 0, 5, 0, 5, 5, 0, 5]]},
        ],
        "categories": [{"id": k, "name": v} for k, v in C.FASHIONPEDIA.items()],
        "licenses": [],
    }
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(ann))
    return p, m


def test_loader_maps_classes_decodes_rle_and_polygons(tmp_path):
    from silhouette.vision.fashionpedia import class_counts, load
    p, m = _fp_file(tmp_path)
    s = load(p)
    assert [x.image_id for x in s] == [1, 2]
    inst = s[0].instances
    assert [i.name for i in inst] == ["pants", "top, t-shirt, sweatshirt"]   # glasses dropped
    assert inst[0].engine_category == "bottom"
    assert np.array_equal(inst[0].mask(40, 30), m)
    poly = inst[1].mask(40, 30)
    assert poly[2:9, 2:9].all() and not poly[20:, 20:].any()
    assert class_counts(s)["dress"] == 1


def test_loader_filters_by_licence(tmp_path):
    from silhouette.vision.fashionpedia import load
    p, _ = _fp_file(tmp_path)
    assert [x.image_id for x in load(p, licence="commercial")] == [1]


def test_loader_refuses_a_file_with_a_different_ontology(tmp_path):
    from silhouette.vision.fashionpedia import load
    p, _ = _fp_file(tmp_path)
    raw = json.loads(p.read_text())
    raw["categories"][6]["name"] = "trousers"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="class mapping"):
        load(p)


# ------------------------------------------------------------------ volume

H_PX = 1000.0
SCALE = BodyScale(crown_y=50, sole_y=50 + H_PX, height_cm=160.0)


def _paint(shape, scale, frame, y0_frac, y1_frac, width_px_of_row, cx=600):
    m = np.zeros(shape, dtype=bool)
    for r in range(int(round(scale.row(y0_frac))), int(round(scale.row(y1_frac))) + 1):
        w = int(round(width_px_of_row(r)))
        m[r, cx - w // 2: cx - w // 2 + w] = True
    return m


def test_row_widths_measure_outer_extent_not_pixel_count():
    m = np.zeros((3, 20), dtype=bool)
    m[1, 2:5] = True
    m[1, 12:15] = True          # two trouser legs with a gap
    assert row_widths(m).tolist() == [0, 13, 0]


def test_body_profile_hits_the_frame_at_landmarks(cv):
    f = cv.frame
    for k in ("shoulder", "bust", "waist", "hip"):
        assert body_width_frac(f, f.y(k)) == pytest.approx(f.widths[k])
    mid = (f.y("bust") + f.y("waist")) / 2
    assert body_width_frac(f, mid) == pytest.approx((f.widths["bust"] + f.widths["waist"]) / 2)


def test_heel_shrinks_the_barefoot_scale():
    s = BodyScale(0, 1070, 160.0, heel_cm=7.0)
    assert s.barefoot_px == pytest.approx(1070 * 160 / 167)


def test_measured_top_matches_a_known_ratio(cv):
    f = cv.frame
    m = _paint((1200, 1200), SCALE, f, f.y("bust"), f.y("waist"),
               lambda r: 1.20 * body_width_frac(f, SCALE.frac(r)) * SCALE.barefoot_px)
    got = measure_garment("top", m, SCALE, f, sleeves=np.zeros_like(m))["volume"]
    assert got.volume == pytest.approx(1.20, abs=0.01)
    assert not got.warnings


def test_measured_bottom_reads_a_wide_leg_against_the_hip(cv):
    f = cv.frame
    w = 1.95 * f.widths["hip"] * SCALE.barefoot_px
    m = _paint((1200, 1200), SCALE, f, f.y("hip"), f.y("ankle"), lambda r: w)
    assert measure_garment("bottom", m, SCALE, f)["volume"].volume == pytest.approx(1.95, abs=0.01)


def test_sleeves_are_subtracted_from_a_top(cv):
    f = cv.frame
    body = lambda r: body_width_frac(f, SCALE.frac(r)) * SCALE.barefoot_px  # noqa: E731
    torso = _paint((1200, 1200), SCALE, f, f.y("bust"), f.y("waist"), body)
    arms = _paint((1200, 1200), SCALE, f, f.y("bust"), f.y("waist"), lambda r: body(r) + 160)
    sleeves = arms & ~torso
    with_arms = measure_garment("top", arms, SCALE, f)["volume"]
    assert with_arms.volume > 1.4 and any("sleeve" in w for w in with_arms.warnings)
    fixed = measure_garment("top", arms, SCALE, f, sleeves=sleeves)["volume"]
    assert fixed.volume == pytest.approx(1.0, abs=0.01)


def test_dress_gets_both_volumes_and_shoes_none(cv):
    f = cv.frame
    m = _paint((1200, 1200), SCALE, f, f.y("bust"), f.y("ankle"),
               lambda r: f.widths["hip"] * SCALE.barefoot_px * 1.3)
    out = measure_garment("dress", m, SCALE, f)
    assert set(out) == {"volume", "volume_lower"}
    assert out["volume_lower"].volume == pytest.approx(1.3, abs=0.01)
    assert measure_garment("shoe", m, SCALE, f) == {}


def test_occluded_or_cut_off_masks_warn_instead_of_guessing(cv):
    f = cv.frame
    m = np.zeros((1200, 1200), dtype=bool)
    r0 = int(SCALE.row(f.y("bust")))
    m[r0:r0 + 20, 0:300] = True                    # a sliver, touching the left edge
    got = measure(m, SCALE, f, DEFINITIONS["default"]["top"], sleeves=np.zeros_like(m))
    assert got.volume is None
    assert any("edge" in w for w in got.warnings)
    assert any("covers only" in w for w in got.warnings)


def test_person_extent():
    p = np.zeros((10, 4), dtype=bool)
    p[2:8, 1] = True
    assert person_extent_from_mask(p) == (2.0, 8.0)


# ----------------------------------------------------------------- metrics

def test_iou_per_class_and_grouped():
    acc = IoUAccumulator(["a", "b", "c"])
    t = np.zeros((4, 4), dtype=bool)
    t[:2] = True
    p = np.zeros((4, 4), dtype=bool)
    p[:1] = True
    acc.add({0: t, 1: t}, {0: p, 1: t}, (4, 4))
    pc = acc.per_class()
    assert pc["a"] == pytest.approx(0.5) and pc["b"] == 1.0 and pc["c"] is None
    assert acc.mean() == pytest.approx(0.75)
    assert acc.grouped({0: "x", 1: "x", 2: None})["x"] == pytest.approx(12 / 16)


# ------------------------------------------------------------- sensitivity

def test_every_volume_threshold_is_still_in_the_rules():
    text = (ROOT / "rules" / "core.yaml").read_text()
    for th, (_side, what) in VOLUME_THRESHOLDS.items():
        op_and_value = what.split(": ", 1)[1].split(" ", 1)[1]
        assert op_and_value in text, f"{op_and_value!r} no longer in core.yaml"
        assert f"{th:.2f}" in op_and_value


def test_near_threshold_counts_only_garments_the_rule_reads(wardrobe):
    any_ = near_threshold(wardrobe, 1.00)
    read = near_threshold(wardrobe, 1.00, side="upper")
    assert "shoe_heel_blk_block" in any_ and "shoe_heel_blk_block" not in read
    assert set(read) <= set(any_)


def test_replacing_nothing_flips_nothing(wardrobe, cv, rules):
    from silhouette.solver import candidates
    outfits = list(candidates(wardrobe, cv.frame))[:200]
    base = fired_sets(outfits, {g.id: g for g in wardrobe}, cv, rules, "weekend")
    same = fired_sets(outfits, replace_volumes(wardrobe, {}), cv, rules, "weekend")
    assert flip_rate(base, same) == 0.0


def test_perturbation_study_shape(wardrobe, cv, rules):
    res = perturbation_study(wardrobe, cv, rules, deltas=(0.0, 0.10), modes=("sign",), seeds=2)
    assert res["runs"][(0.0, "sign")]["mean"] == 0.0
    assert 0.0 < res["runs"][(0.10, "sign")]["mean"] < 1.0


# ------------------------------------------------------- the volume harness

def _write_photo_dir(tmp_path, cv, wardrobe, scale_judged=1.0, offset=0.0):
    """One synthetic photo per garment pair, masks painted at a known volume."""
    from silhouette.vision.sam import save_mask
    f = cv.frame
    W = {g.id: g for g in wardrobe}
    photos = tmp_path / "photos"
    rows = []
    pairs = [("top_tee_blk", "bot_wide_blk"), ("top_blouse_struct_cream", "bot_jean_straight_indigo"),
             ("top_hoodie_oversized_grey", "bot_palazzo_floral"), ("top_tee_blk", "bot_skirt_pencil_blk")]
    for k, (t, b) in enumerate(pairs):
        stem = f"p{k}"
        top = _paint((1200, 1200), SCALE, f, f.y("bust"), f.y("waist"),
                     lambda r, v=W[t].volume: (scale_judged * v + offset)
                     * body_width_frac(f, SCALE.frac(r)) * SCALE.barefoot_px)
        bot = _paint((1200, 1200), SCALE, f, f.y("hip"), f.y("ankle"),
                     lambda r, v=W[b].volume: (scale_judged * v + offset)
                     * f.widths["hip"] * SCALE.barefoot_px)
        save_mask(photos / "masks" / stem / f"{t}.png", top)
        save_mask(photos / "masks" / stem / f"{b}.png", bot)
        save_mask(photos / "masks" / stem / "sleeves.png", np.zeros_like(top))
        rows.append({"photo": f"{stem}.jpg", "garments": f"{t};{b};shoe_sneaker_white",
                     "crown_y": SCALE.crown_y,
                     "sole_y": SCALE.crown_y + H_PX * (160 + 2.0) / 160, "heel_cm": ""})
    with open(photos / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return photos


def test_harness_recovers_judged_volumes_from_exact_masks(tmp_path, cv, wardrobe):
    from silhouette.vision.volume_eval import measure_photos, report, summarise
    rows = measure_photos(_write_photo_dir(tmp_path, cv, wardrobe), wardrobe, cv,
                          definitions=["default"])
    s = summarise(rows)["default"]
    assert s["failed"] == 0
    assert s["raw"]["max_abs"] < 0.015                  # the shoe's heel was honoured
    assert s["repeat_garments"] == 1                    # the black tee is in two photos
    assert "definition: default" in report(summarise(rows))


def test_calibration_removes_a_systematic_scale_difference(tmp_path, cv, wardrobe):
    """A mask that reads every garment 10% wider plus 0.15 is badly biased raw
    and nearly exact after leave-one-out calibration: a scale difference is not
    a measurement error."""
    from silhouette.vision.volume_eval import measure_photos, summarise
    rows = measure_photos(_write_photo_dir(tmp_path, cv, wardrobe, 1.10, 0.15), wardrobe, cv,
                          definitions=["default"])
    s = summarise(rows)["default"]
    assert s["raw"]["bias"] > 0.2
    assert s["calibrated"]["max_abs"] < 0.03


# ----------------------------------------------------------- the segmenter

_HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "scipy"))


@pytest.mark.skipif(not _HAVE_TORCH, reason="needs the [vision] extra")
def test_mask2former_trains_on_a_tiny_synthetic_set(tmp_path):
    from PIL import Image

    from silhouette.vision.fashionpedia import load
    from silhouette.vision.train import TrainConfig, evaluate, fit
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    images, anns = [], []
    for i in range(2):
        a = np.full((96, 64, 3), 200, np.uint8)
        a[10:50, 16:48] = (30, 30, 30)
        a[50:90, 12:52] = (20, 40, 120)
        Image.fromarray(a).save(img_dir / f"{i}.jpg")
        images.append({"id": i, "width": 64, "height": 96, "file_name": f"{i}.jpg", "license": 0})
        for j, (cid, (x0, y0, x1, y1)) in enumerate(((1, (16, 10, 48, 50)), (6, (12, 50, 52, 90)))):
            anns.append({"id": 10 * i + j, "image_id": i, "category_id": cid,
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                         "iscrowd": 0, "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]]})
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"images": images, "annotations": anns, "licenses": [],
                             "categories": [{"id": k, "name": v} for k, v in C.FASHIONPEDIA.items()]}))
    s = load(p)
    cfg = TrainConfig(images=str(img_dir), out=str(tmp_path / "run"), tiny=True, size=64,
                      batch_size=2, max_steps=3, warmup_steps=1, num_workers=0, fp16=False)
    r = fit(cfg, samples=s, log=lambda *a: None)
    assert r["steps"] == 3 and np.isfinite(r["loss_last"])
    assert (tmp_path / "run" / "last.pt").exists()
    from transformers import Mask2FormerForUniversalSegmentation
    m = Mask2FormerForUniversalSegmentation.from_pretrained(tmp_path / "run" / "model")
    res = evaluate(m, s, str(img_dir), size=64, threshold=0.0)
    assert set(res["by_engine_category"]) >= {"top", "bottom"}


# ---------------------------------------------------------------- rejudging

def test_rejudge_sheet_is_blind_and_the_analysis_reads_it(tmp_path, wardrobe):
    from silhouette.vision.rejudge import DEFAULT_GARMENTS, analyse, make_sheet
    p = make_sheet(wardrobe, tmp_path / "s.csv", seed=1)
    rows = list(csv.DictReader(open(p)))
    assert {r["garment_id"] for r in rows} == set(DEFAULT_GARMENTS)
    assert all(not r["session_1_volume"] for r in rows)         # no current values shown
    W = {g.id: g for g in wardrobe}
    for k, r in enumerate(rows):
        r["session_1_volume"] = W[r["garment_id"]].volume
        r["session_2_volume"] = W[r["garment_id"]].volume + (0.1 if k % 2 else -0.1)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a = analyse(p, wardrobe)
    assert a["garments"] == 10
    assert a["volume_sd_one_judgement"] == pytest.approx(0.1 / 2 ** 0.5)
    assert a["share_within_05"] == 0.0


def test_mask_agreement_is_one_for_identical_mask_sets(tmp_path, cv, wardrobe):
    import shutil

    from silhouette.vision.volume_eval import mask_agreement
    photos = _write_photo_dir(tmp_path, cv, wardrobe)
    shutil.copytree(photos / "masks", photos / "masks_model")
    res = mask_agreement(photos, wardrobe)
    assert set(res) == {"top", "bottom"}
    assert all(r["mean_iou"] == 1.0 for r in res.values())
