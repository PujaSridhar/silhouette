# W1 — garment segmentation, built to measure `volume`

**Status:** code complete, not yet trained. No GPU run, no photos, no measured
volumes exist yet. Every result below that needs either is marked as pending.

## Why W1, and what decides whether it worked

`volume` drives four of the engine's balance rules. It is defined as garment
width over body width at the garment's own zone (`models.py`), and in phase 0
it is a judgement: one person's eye, typed into `data/wardrobe.csv`.

The engine is sensitive to it. Re-checked with
`silhouette-vision sensitivity` over all 1,548 candidate outfits, 200 random
draws each:

| perturbation of every garment's volume | outfits whose fired rules change |
|---|---|
| exactly ±0.05, random sign | **10.8%** (90% of draws: 7.4–14.3%) |
| uniform in [−0.05, +0.05] | 7.5% (3.5–11.2%) |
| exactly ±0.10, random sign | **22.4%** (17.3–28.5%) |
| uniform in [−0.10, +0.10] | 13.3% (8.5–18.6%) |

The ±0.05 row reproduces the figure in the review that set this task
(10.8%); the ±0.10 row lands at 22.4% against its 21.4%, inside the seed-to-seed
spread. "Within 0.10 of the 1.00 threshold" is 12 garments counting the two
that sit exactly 0.10 away, but three of those are shoes and two are bottoms,
and the 1.00 threshold only reads tops, outer layers and dress bodices. The
garments that threshold actually reads: 7.

So W1's success criterion is not mask mIoU. It is: **does a mask-measured
volume agree with the engine's scale to ±0.05, more consistently than the
human eye does?** mIoU is reported alongside, because a mask can score well on
IoU and still be a few pixels too wide at the hip.

Note that ±0.05 is a target, not a safe zone: an error that size still changes
the fired rules on roughly 1 outfit in 10, because some garments sit exactly on
a threshold (the cream blouse is 1.00 against `upper_volume <= 1.00`). That is
a property of hard thresholds in the rules, not of any estimator.

## The finding that shapes the eval: two scales

The judged column runs **below 1.0** for close-fitting pieces: the fitted tee
is 0.85, the crop 0.80, the pencil skirt 0.80. Taken literally, a garment
narrower than the body it covers is impossible, and a mask will measure those
pieces at about 1.00–1.05.

So measured and judged volume are on different scales by construction. Her
eye scores *how close* a fit reads, not a literal width ratio. Two
consequences:

1. A measured volume cannot be dropped into `wardrobe.csv` as-is. The rule
   thresholds (1.00, 1.15, 1.30) were set on the judged scale.
2. The harness therefore reports two things: the **raw** difference (where a
   bias is expected and is not error), and the difference after a
   **leave-one-garment-out linear calibration** from measured onto judged
   (one line for upper-body volumes, one for lower-body).
   The calibrated number is the one held to ±0.05.

The other fix — re-setting the thresholds on the measured scale — is a rules
change and out of scope here. It is the better long-term answer, and the one
to take once expert labels exist to set thresholds against.

## How volume is measured

`src/silhouette/vision/volume.py`:

```
volume = reducer over zone rows of  garment_mask_width(row) / body_width(row)
```

- **Garment width** is the mask's outer extent in each row (the gap between
  two trouser legs is part of how wide the garment reads). For tops, outer
  layers and bodices, predicted **sleeve** masks are subtracted first, so an
  arm at the side does not read as torso.
- **Body width does not come from the photo.** The garment covers the body at
  exactly the rows being measured. It comes from the Style CV, which already
  stores body widths as fractions of barefoot height. The photo supplies only
  the wearer's height in pixels (crown and sole rows), corrected for the heel
  of the shoe she is wearing.
- **Zones** (default definition): tops and outer layers from bust to waist,
  against the interpolated shoulder–bust–waist–hip profile, median over rows.
  Bottoms and dress skirts from hip to hem (capped at the ankle), against the
  hip width, 90th percentile over rows, so a wide leg is read at its width. A
  second definition, `bottom_median`, uses the median instead. Which one
  tracks her eye is an empirical question; the harness reports both.

### Worn photos only

This needs a person in the frame: a full-length photo, standing, facing the
camera. A garment on a bed has no crown or sole and no body to compare with.

A flat-lay would need three things this code does not have: a scale object in
frame (an A4 sheet, a ruler) to turn pixels into centimetres; her body widths
in centimetres (height × the frame widths — easy); and a model of how a flat
width becomes a worn width, which depends on fabric, drape and cut. The last
is the hard one, and the reason flat-lays are not attempted.

## The model

- **Data:** Fashionpedia (Jia et al., 2020): 45,623 train and 1,158 val images
  with instance masks. Loader: `vision/fashionpedia.py`. It checks the file's
  category names against the mapping and refuses a file that differs.
- **Classes:** the 14 whole garments the engine has a slot for, plus `sleeve`
  (`vision/categories.py`). Fashionpedia's classes are kept as the training
  labels and mapped to the engine's five categories afterwards:
  shirt/blouse, top/t-shirt/sweatshirt, sweater → top; cardigan, jacket, vest,
  coat, cape → outer; pants, shorts, skirt → bottom; dress, jumpsuit → dress;
  shoe → shoe. Vest is ambiguous (worn as a top or over one); it is filed as
  outer.
- **Architecture:** Mask2Former, Swin-S, from the COCO instance checkpoint
  (`facebook/mask2former-swin-small-coco-instance`), re-headed to 15 classes.
  Instance rather than semantic segmentation, because a top and the jacket
  over it are two garments to measure separately, and sleeves overlap their
  garments by construction. SegFormer is semantic-only and would merge both.
- **Training:** `vision/train.py`, driven by
  `notebooks/w1_mask2former_fashionpedia.ipynb`. Checkpoints and resumes.

Verified here without a GPU: the loader reproduces the annotation areas and
boxes exactly on the real val file (polygons and compressed RLE); all 1,158
val images match their annotated sizes; the training loop runs on real
Fashionpedia images with a tiny random model, and overfits a synthetic set to
0.94 mIoU. The pretrained checkpoint could not be downloaded in this
environment, so loading it is first exercised in the notebook.

### Licence

Fashionpedia's annotations are released under **CC BY 4.0**. The images are
not one licence: each carries its own, listed in the annotation file. In the
val split 69% are Creative Commons NonCommercial variants (BY-NC-ND alone is
55%). `--licence commercial` trains only on images whose licence allows
commercial use: 31% of val. For a portfolio or research model, `all` is fine.
For a model inside a product, use `commercial`, or get a view on whether a
model trained on NC images is a derivative of them, which is a legal question.

## SAM 2

Annotation accelerator only (`vision/sam.py`, `silhouette-vision annotate`).
It turns a box or a few clicks into a mask; it does not classify, and the
garment id is whatever the person clicking says it is. Its masks are the
reference for her own photos: what the model's masks and the measured volumes
are checked against. Install it with the `[sam]` extra, which pulls Meta's own
repository. The `sam2` package on PyPI is a third-party upload and is not used.

## What she needs to take

Photos stay on her machine; `photos/` is gitignored.

1. **About 15 outfits, each garment in at least two of them**, so every
   garment's measurement can be repeated. All 27 non-shoe garments should
   appear.
2. **Each photo:** full length with a margin above the head and below the
   feet; facing the camera; standing straight; arms held a hand's width away
   from the body; phone propped at about hip height, 2.5–3 m away, portrait,
   level rather than tilted down; timer rather than a mirror if possible. A
   mirror selfie works only if the phone arm does not cross the torso. Plain
   background, even light.
3. **Shoes on** as listed in the manifest (their heel height is read from the
   wardrobe).
4. **Manifest** `photos/manifest.csv`:

   ```
   photo,garments,crown_y,sole_y,heel_cm
   IMG_0412.jpg,top_tee_blk;bot_wide_blk;shoe_sneaker_white,38,1912,
   ```

   `crown_y`/`sole_y` come from `silhouette-vision annotate IMG --person --box ...`.
5. **Hand masks:** one `annotate --garment ID` per worn garment, plus
   `annotate --sleeves` for the sleeves. Check each preview image it writes.

## Running it

```bash
make sensitivity                                   # the table above (~8 min)
silhouette-vision measure --masks masks            # hand masks vs judged
silhouette-vision predict --model runs/w1/model    # after the GPU run
silhouette-vision compare-masks                    # model vs hand masks
silhouette-vision measure --masks masks_model      # model masks vs judged
silhouette-vision sensitivity --volumes evals/volume/model_calibrated.csv
```

## How to read the result

Agreement with the CSV is bounded by the CSV's own noise. If her eye wanders
by ±0.08 between sittings, no mask can agree with it to ±0.05. So the decision
uses three numbers:

- calibrated |measured − judged|, p90 (target ≤ 0.05);
- the measurement's photo-to-photo SD (the `repeat` line);
- her own sitting-to-sitting SD, from the blind re-judging protocol
  ([volume-rejudging-protocol.md](volume-rejudging-protocol.md)).

If the measurement's repeat SD is clearly below hers, measured volume is the
more consistent input even where it disagrees with the CSV, and the
disagreement says more about the CSV. With 30 garments every number is
indicative; the harness says so.

## Not in W1

Attribute extraction (rise, length, fit, structure) is W2. Crown and sole
from a pose model is W3; for now they are tapped or read from a SAM person
mask. Nothing here changes a rule, a weight or `wardrobe.csv`.
