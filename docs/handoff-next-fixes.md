# Handoff: the next set of fixes

**For whoever picks this up (human or agent). Read all of this before editing.**

Seven tasks, verified against the code on 2026-11-19. Each has a file, a
problem, a fix and an acceptance test. Do these and nothing else — the
"Explicitly out of scope" section exists because a reviewer proposed a much
longer list and most of it optimises things that do not bind yet.

---

## What this project is, in five lines

A styling engine that is a **constraint solver over explicit rules**, not a
recommender. Rules live in `rules/core.yaml` as data. Every suggestion is
explained by rendering the rule that actually fired, with the numbers that made
it fire. The headline metric is `lift` — apparent leg length minus actual, in
fractions of the wearer's height. The project's real bottleneck is **expert
labels**, of which there are currently zero.

---

## Rules of engagement

**The two engines must stay in sync.** The rule logic exists twice: in Python
(`src/silhouette/`) and re-implemented in JavaScript inside
`app/index.template.html` and `demo/index.template.html`. Any change to rule
conditions, geometry or scoring **must be mirrored in all three**. The safety
net is the parity fixture: `make build` re-exports cases computed by Python,
and each page replays them on load and shows a badge. If you change Python and
not the JS, that badge goes red — so check it.

**Do not rewrite `rules/core.yaml` wholesale.** Its comments and `note:` fields
are documentation, including the reasoning behind specific weights. Edit in
place. Never run it through a YAML dumper.

**Do not commit generated or private files.** `app/index.html`,
`demo/index.html`, `demo/*-data.json`, `evals/labels/*.jsonl` and
`evals/imported/` are gitignored deliberately. Label files in particular would
be mistaken for evidence; see `evals/labels/README.md`.

**Do not change a number in `README.md` or `docs/` unless you re-derived it.**
Several documented figures (lift values, rule counts, feasible-outfit counts,
the +0.930 parity result) are claims that were measured. If your change makes
one wrong, say so in the PR; do not guess a replacement.

**Do not touch the published artifact.** Editing the template is fine;
publishing happens elsewhere.

---

## T1 — `numpy` is missing from dependencies  *(do this first)*

`pyproject.toml:19` declares only `pyyaml>=6.0`, but `src/silhouette/cli.py:17`
imports `evals.fitting`, which imports numpy. `numpy` currently arrives only as
a transitive dependency of `ortools`, which is in the optional `[solver]` extra.

So `pip install -e .` followed by `silhouette --help` crashes.

**Fix:** add `numpy>=1.24` to `dependencies`.

**Accept:** in a fresh virtualenv, `pip install -e .` (no extras) then
`silhouette --help` exits 0.

---

## T2 — Bootstrap labels leak into expert evaluation

Bootstrap labels come from the synthetic rater and are explicitly not evidence.
Three places read every label regardless of source:

- `src/silhouette/evals/report.py:98` — `store.repeats()` with no source, so
  self-agreement can be computed from synthetic repeats.
- `src/silhouette/evals/report.py:124` — `coverage_report(store, ...)` reads
  `store.all()`, so bootstrap labels make rules look covered.
- `src/silhouette/evals/sampling.py:96` — `store.labelled_keys()` spans all
  sources, so `silhouette label` skips outfits only a bootstrap label has seen.

**Fix:** give `LabelStore.labelled_keys()`, `_labelled_rule_counts()`,
`coverage_report()` and `next_batch()` a `source` parameter. Thread
`source_used` through `evaluate_labels`. Default to `"expert"` where a default
is needed.

**Accept:** a new test builds a store with 2 expert labels and 50 bootstrap
labels and asserts that (a) coverage counts reflect only the expert labels and
(b) `next_batch` will still serve an outfit that only a bootstrap label covers.

---

## T3 — Gap analysis breaks on occasions you cannot currently dress for

Unlocking an occasion with zero feasible outfits is the most valuable thing a
purchase can do, and both implementations get it wrong:

- `src/silhouette/gaps.py:51` sets the baseline to `float("-inf")`; line 80
  then produces `+inf`, which prints as `inf` and makes two unlocking items
  incomparable.
- `app/index.template.html`, in `computeGaps()`, has
  `top - (base === -Infinity ? top : base)` — which is `0`, so an unlocking
  garment scores **zero**.

**Fix:** track `contexts_unlocked: int` as a separate field. Never produce
`inf`: when the baseline is unreachable, record the unlock and leave `delta` at
0 for that occasion. Sort by `(-contexts_unlocked, -best_delta, -contexts_won)`.
Mirror in the app and add an "unlocks" column to its table.

**Accept:** a test in `tests/test_gaps.py` builds a wardrobe with nothing
feasible for one occasion and asserts a candidate that unlocks it ranks above a
candidate that merely improves an already-covered occasion, and that no
returned value is infinite.

---

## T4 — Empire-seam dresses get no proportion credit  *(highest styling impact)*

Measured on the reference wardrobe:

```
anarkali (empire seam)   lift +0.191   break 0.31   proportion rules fired: NONE
saree (no break)         lift +0.000   break none   drape_reads_as_one_column (+0.55)
```

The anarkali has the **highest lift in the entire wardrobe** — higher than the
wearer's signature crop + wide-leg at +0.120 — and earns nothing, because
`high_break_elongates` requires `bottom.rise in ('high','super_high')` and a
dress has no bottom. The saree, with zero lift, collects a bonus.

This disproportionately affects South Asian garments, which is the area the
project treats as its differentiator.

**Fix:** add a **new** soft rule, do not broaden the existing one — widening
`high_break_elongates` would change scores for every top-and-bottom outfit and
invalidate documented numbers. Suggested:

```yaml
- id: high_seam_elongates
  kind: soft
  goal: elongate_lower_body
  weight: 0.85
  note: >
    The dress counterpart to high_break_elongates. A seam above the natural
    waist does the same work as a high rise, and the engine was blind to it.
  when: "f.break_source == 'dress_seam' and f.break_ratio is not None and 0.22 <= f.break_ratio <= 0.43"
  score: "norm(f.lift, 0.0, 0.20)"
  explain: >
    The seam sits at {break_ratio:.0%} of your height, well above the natural
    waist, so the eye reads about {lift:.0%} more length below it.
```

Note the wider `norm` ceiling (0.20, not 0.12): empire seams reach higher lift
than any waistband can.

**Mirror it in the `COND` table of both HTML templates**, then `make build`.

**Accept:** the anarkali fires a proportion rule; **all existing tests pass
unchanged** (if any existing test changes value, you broadened something you
should not have); a new fixture asserts the rule fires on an empire seam and
does not fire on the saree or on any top-and-bottom outfit; the in-browser
parity badge is still green.

---

## T5 — Two small real bugs

**`tests/test_geometry.py:15` is vacuous.** `test_true_leg_ratio_is_anatomy_only`
builds `bare` and `heeled` identically — neither has a shoe — so it cannot
fail. Give `heeled` an actual heeled shoe and assert `true_leg` is unchanged
while `lift` differs.

**`src/silhouette/models.py:132` crashes.** `Outfit.label()` does `bits[0] +=
" (tucked)"` with no guard: `Outfit(tucked=True)` with no items raises
`IndexError`, and when `top` is None it tags the bottom instead. Guard the
empty case and tag the top specifically.

**Accept:** both covered by tests.

---

## T6 — Two documentation claims that are false

**`README.md:166` says "66 tests".** There are 90. Make it match, and prefer
wording that will not drift.

**`README.md:286` claims the language policy is "asserted in tests".** It is
not — no test references the banned vocabulary. Rather than weaken the claim,
make it true: add `tests/test_language_policy.py` that loads every `explain`
and `reject` template from `rules/core.yaml` and asserts none contains a banned
term from `docs/language-policy.md`:

> flattering, slimming, hides, disguises, camouflages, minimise/minimize,
> problem area, figure flaw, cheats, fakes, tricks the eye, dresses for your
> shape

Match case-insensitively on word boundaries. This is the cheapest high-value
test in the project: the policy is a product constraint, and right now nothing
enforces it.

---

## T7 — Dead configuration

`data/style_cv.yaml:40-48` defines `palette` and `constraints.hard`, neither of
which the engine reads. `constraints.hard` still names
`within_context_formality_band`, a rule that was split into a hard floor and a
soft ceiling in phase 0 and no longer exists.

**Fix:** delete both blocks, or wire them up. Deleting is correct unless you
have a reason. Check `models.py` still parses the file afterwards.

---

## How to verify everything

```bash
python -m venv .venv && .venv/bin/pip install -e '.[solver,dev]'
.venv/bin/python -m pytest tests/ -q          # expect >90 passing, 0 failing
make build                                     # regenerates pages + parity fixtures
python -m http.server 8788 --directory .       # open /app/ and /demo/
```

On both pages, check the parity line. It must read **N/N**. A red badge means
the JS and Python engines disagree and the change is not finished.

---

## Explicitly out of scope

Do not do these, even though a reviewer suggested them. They are either
premature or would obscure the diff:

- Deduplicating the JS engine into a shared `engine.js`, or running node in CI.
  Parity already runs in-browser on every page load.
- Pre-compiling rule expressions, replacing `eval` with an AST evaluator, or
  caching CIELAB for speed. A 30-second test suite is not a problem, and
  `docs/adr/0002-rules-as-data.md` explains why `eval` is acceptable here.
- New rules for other body types (`broaden_shoulder_line`,
  `soften_waist_definition`). Needed for a second person, not the first.
- Changing how outer-layer volume is combined (mean vs max), or adding
  style-family compatibility metadata. These are modelling judgements that
  should be settled by expert labels, not by a code review.
- Any change to the eval harness's statistics (group k-fold by `outfit_key`,
  out-of-fold judge scoring, weight-scale normalisation, nested CV for lambda).
  All real concerns, all only bite once real labels exist. They are the *next*
  batch, not this one.

---

## Definition of done

1. All tests pass; the count in `README.md` matches reality.
2. `make build` succeeds and both parity badges are green in a browser.
3. A fresh clone + `pip install -e .` gives a working `silhouette --help`.
4. No generated or label files committed.
5. The PR description lists any documented number your change invalidated.
