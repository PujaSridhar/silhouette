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

## Regenerating bootstrap labels

```bash
make bootstrap    # synthetic, for validating the pipeline only
make evals SOURCE=bootstrap
```
