# Labels

Deliberately empty in version control.

Two different things produce files here and only one of them is evidence:

- **Expert labels** — a human rating outfits against `../rubric.md`, with the
  engine's score hidden. These are the only labels that may inform a claim
  about whether the engine works. They are also somebody's judgements about
  their own body and wardrobe, so publishing them is a decision to make on
  purpose rather than by default.
- **Bootstrap labels** — produced by `silhouette bootstrap` from the synthetic
  rater in `src/silhouette/evals/synthetic.py`. They exist to prove the
  harness runs end to end. They say nothing about engine quality, and the
  scoreboard refuses to draw conclusions from them.

A `labels.jsonl` committed here would be indistinguishable from the first kind
at a glance, so neither is tracked.

## Getting labels in

```bash
# in the app: Rate tab -> label -> "Export for the fitter"
silhouette import-app export.json --out evals/imported

silhouette --wardrobe evals/imported/wardrobe.csv \
           --style-cv evals/imported/style_cv.yaml \
           --labels   evals/imported/labels.jsonl  eval --source expert
```

## Several stylists

Each stylist types their name on the Rate tab, so every label says who made it.

- **The shared set.** Every rater is served the same 12 outfits
  (`evals/sampling.shared_set`), one in every three, mixed in with their usual
  ones. The set depends only on the wardrobe's garment ids, the frame, the
  goals and a fixed seed, never on what anyone has rated, so it is the same for
  everyone rating the same wardrobe. It is spread across the score range,
  because agreement measured only on so-so outfits looks better than it is.
- **Agreement between stylists.** `eval` reports Krippendorff's alpha on
  `overall` (interval metric) over outfits two or more raters rated, each
  rater's first rating only, plus quadratic-weighted kappa for every pair and
  how far each rater is through the shared set. The app shows the same alpha.
- **Can't judge.** With a name entered, the Can't judge button saves a mark
  (rater, outfit, and the note as the reason) instead of a score. Marks are
  exported as `cantJudge`, imported to `cant_judge.jsonl` beside the labels,
  and never read as ratings. `eval` lists outfits two or more raters could not
  judge: the card is missing something, not the rater.

Overlap needs the same garment ids, so stylists must be rating the same
wardrobe record. The app cannot yet load a wardrobe from another account, so
today that means rating on the wearer's own device or account, each under
their own name. When stylists do have the same wardrobe in their own browsers,
each exports one file, and `import-app` merges them, refusing files whose
garments disagree:

```bash
silhouette import-app asha.json bea.json chen.json --out evals/imported
```

## Regenerating bootstrap labels

```bash
make bootstrap    # synthetic, for validating the pipeline only
make evals SOURCE=bootstrap
```
