# Silhouette

**A styling reasoning engine.** Not a recommender — a solver that assembles
outfits against explicit styling rules and tells you, with numbers, why each one
works on your frame.

> Most wardrobe apps are a catalog plus visual similarity search. They answer
> *what looks like what*. They cannot answer *why this works on you*, because
> they have no model of proportion, balance or intent.

```
                     Solver for structure
                     Model for taste
                     Language model for explanation only
```

The language model never picks the outfit. It describes what the solver chose.
That is the whole trust model.

---

## The idea in one number

Styling is mostly **measurable illusion work**. A high-rise trouser does not
change your legs; it moves the visual break from the middle of your body into
the upper third, and the eye reads the lower section as longer.

So measure that:

```
lift = apparent_leg_ratio − true_leg_ratio(barefoot)
```

`true_leg_ratio` is anatomy and fixed. `apparent_leg_ratio` is what the eye
reads: the visual break to the floor, over total height including heel. The
difference is the illusion, in fractions of your own height.

On the reference frame (hip at 0.50, so a true leg ratio of 0.50):

| outfit | break at | lift |
|---|---|---|
| super-high wide-leg + crop + 7 cm heel | 37% | **+0.131** |
| super-high wide-leg + crop, flat shoe | 38% | +0.120 |
| high-rise wide-leg + tucked blouse | 40% | +0.105 |
| mid-rise bootcut | 44% | +0.056 |
| low-rise jean | 48% | +0.016 |
| high rise + **untucked** hip-length top | 52% | **−0.019** |

That last row is the point. An untucked hip-length top over a high rise does not
merely waste the rise — it reads *shorter* than you actually are. A plain
goodness score could never tell you that.

---

## What it says

Real output, unedited:

```
$ silhouette critique --context weekend \
    --items top_crop_rib_ivory,bot_wide_blk,shoe_sneaker_white

Ivory ribbed crop + Black super-high wide-leg + White leather sneaker
    score  +1.74   break 38%   lift +0.120   apparent lower body 62%
                   dE 84   volume 0.80/1.95   heel 2cm

    + [+0.90] Volume below (x1.95 body width) against a close top (x0.80)
              keeps the silhouette legible instead of widening the whole column.
    + [+0.85] The break sits at 38% of your height instead of near the middle,
              so the eye reads about 12% more length below it.
    + [+0.35] The break is already high at 38%, so the colour change there
              (dE 84) works for you - it marks the high waist rather than
              cutting you in half.
    - [-0.24] This works, and it has room to go further - the top is soft
              (structure 0.10) against x1.95 volume below. A shoulder seam, a
              sleeve detail or an open jacket would add weight up top without
              adding width.
    - [-0.11] weekend tops out around 0.52 and this reaches 0.60 - dressier
              than the occasion asks, which may be exactly what you want.
```

Every line is rendered from a rule that actually fired, with the numbers that
made it fire. None of it is generated prose.

### One decision, measured

The same blouse, the same trousers, the same shoes — tucked or not:

```
untucked   score +0.83   break 50%   lift -0.003
tucked     score +1.98   break 38%   lift +0.117
```

No shopping involved.

---

## Architecture

```
  annotations ──▶ geometry ──▶ features ──┐
  (CSV, hand-        │                    │
   labelled in    break_ratio             ▼
   phase 0)       lift, volume       rules engine ──▶ hard: feasible?
                  asymmetry,         (YAML, 17 rules)  soft: weighted score
                  ΔE2000 contrast          │                  │
                                           ▼                  ▼
                                      explanations      CP-SAT week planner
                                      (per fired rule)  (days coupled through
                                                         a shared wardrobe)
```

| module | does |
|---|---|
| `color.py` | sRGB → CIELAB, CIEDE2000. Perceived contrast, not RGB distance. |
| `models.py` | Garment, Frame, StyleCV, Context, Outfit. Every field explicit. |
| `geometry.py` | `break_ratio`, `lift`, volume asymmetry, contrast at the break. |
| `rules.py` | DSL loader and evaluator. Hard constraints and weighted soft rules. |
| `solver.py` | Exhaustive enumeration — exact at wardrobe scale, 158 µs per candidate. |
| `capsule.py` | CP-SAT week planning with slack-based bottleneck diagnosis. |
| `gaps.py` | Marginal outfit yield: which single purchase buys the most. |
| `explain.py` | Renders the fired rules. No generated text. |

**Where the solver actually earns its keep** is the *week*, not the outfit.
Scoring one outfit is enumeration. Choosing seven outfits that cover seven
occasions while no garment is worn more than twice couples the days through a
shared wardrobe — that is integer programming, and that is where CP-SAT goes.

---

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e '.[solver,dev]'
```

```bash
make demo
```

```bash
make test
```

Individually:

```bash
silhouette outfit   --context office --top 3
silhouette critique --items top_blouse_struct_cream,bot_wide_cream,shoe_heel_blk_block --tucked
silhouette gaps
silhouette week     --days office,office,campus,dinner,weekend --max-wears 2
silhouette rules -v
```

`ortools` is optional. Without it the week planner falls back to a greedy
heuristic that answers the same question and says so.

---

## Status: phase 0 and the eval harness complete

Phase 0 exists to answer one question before any GPU money is spent: **does the
engine produce outfits a stylist would actually use?**

Thirty hand-annotated garments, 17 rules, an exact solver, **no machine learning
anywhere**. 66 tests, including a must-fire fixture for every rule and a
coverage test that fails when a rule is added without one.

**Result: pass on reasoning.** The engine states the stylist's two rules back to
her — including a nuance she mentioned only in passing ("balance it, but adding
more fabric on top makes it worse"), which falls out of a volume-asymmetry
threshold rather than having been written in as prose. Her own go-to outfit
scores +1.74 and ranks 75th of 499 feasible weekend assemblies, the top 15%.

It does **not** rank first, and that turned out to be the most useful result of
the whole spike — see the last bullet.

Phase 0 found ten design problems. The most valuable:

- **A formality ceiling cannot be a hard constraint.** As a hard rule it
  rejected the stylist's own signature outfit for a weekend. Underdressed is a
  failure; overdressed is a preference. No training run would have caught this.
- **A shoe bottleneck, found twice independently.** Gap analysis ranked a second
  dressy heel top on occasion coverage; the week planner came back
  over-subscribed and named the same shoe. Exactly one shoe in thirty garments
  clears a 0.45 formality floor.
- **Gap analysis picked an item nobody planted.** Top recommendation by score
  delta is a fitted bodysuit — because a bodysuit *structurally cannot untuck*,
  permanently removing the failure mode in the `lift` table above. First
  evidence the approach generalises past its hand-written rules.
- **Occasion coverage is wildly uneven:** 729 feasible assemblies for campus,
  **12** for a wedding, out of 1,548 candidates. A ranker trained on uniform
  samples would barely learn the 0.8% case. A solver finds all twelve every
  time.
- **The engine over-rewards dressiness, and found it out on itself.** Its top
  weekend pick keeps the stylist's top and her whole strategy but adds a heel
  and a blazer — exactly what its own rules advised. Those bonuses total 0.95
  in weight while nothing in the rule set knows that effort has a cost on a
  Saturday. A real calibration defect, left unfixed on purpose: it is a fitting
  problem for the phase 4 eval set, not a number to guess at twice.

Full write-up with all ten: [`docs/phase-0-findings.md`](docs/phase-0-findings.md).

### Phase 4 — the eval harness

Finding 10 said the weights needed fitting, not guessing. That now works:
a rubric, an active sampler that optimises for rule coverage rather than sampling
uniformly, Spearman and quadratic-weighted kappa, a judge with an explicit
mean-baseline comparison, and prior-anchored non-negative weight fitting that
refuses to write unless it beats the hand-set weights out of sample.

It is validated against a **synthetic rater with two deliberately different,
documented preferences** — because random labels would make the engine look broken
and engine-derived labels would make it look perfect, and neither tests the harness.
The fitter recovers both changes from labels alone:

| rule | prior | fitted | synthetic ground truth |
|---|---|---|---|
| `overdressed_for_context` | 0.50 | **1.51** | 1.60 |
| `heel_extends_the_leg_line` | 0.35 | **0.00** | 0.05 |

Held-out rank correlation rises from +0.882 to +0.915. The scoreboard still refuses
to draw quality conclusions from those labels, and says so on every run — see
[ADR 0004](docs/adr/0004-eval-harness.md).

**Expert labels so far: zero.** That is the honest state, and the next action.

### Deliberately not here yet

No model, no photos, no embeddings, no language model. Every number traces to a
hand-annotated CSV cell and a rule in a YAML file — which is exactly what makes
this engine the thing that will judge whether phase 1's vision models are any
good.

---

## Try it

**[Interactive demo](https://claude.ai/artifact/UDuBtZmHQY11P5shRLu7GR)** — assemble an
outfit, move the frame landmarks, watch the proportion math and the rules respond.

The demo re-implements the geometry and rule conditions in JavaScript so the sliders
work live. A second implementation is a second chance to be wrong, so the page replays
74 outfit/context cases whose answers the Python engine computed and compares every
number and every fired rule. The badge in the corner reports the result — currently
**74/74**. If the port ever drifts, the page says so itself.

## The labelling loop

Labels are the binding constraint on this project: weight fitting needs them, and
nobody but the stylist can make them. So labelling lives in the app, where it will
actually happen, and the fitter lives here, where the maths is.

```bash
# in the app: Rate tab -> label -> "Export for the fitter"
silhouette import-app export.json --out evals/imported
silhouette --wardrobe evals/imported/wardrobe.csv \
           --style-cv evals/imported/style_cv.yaml \
           --labels   evals/imported/labels.jsonl  eval --source expert
```

The Rate tab hides the engine's score and its explanations while you rate — seeing
them anchors the judgement and the whole point is independence. Every eighth label
re-serves something already rated, because two independent ratings of one outfit give
**self-agreement**, which is the ceiling any model is measured against.

It also separates two things that look alike: rules **not yet labelled** (reachable,
the sampler is aiming at them) from rules that **cannot fire at all** for this frame
and wardrobe. The second is information, not a gap — no amount of labelling reaches
them.

Verified end to end: 16 labels made in the app, exported, imported, and re-scored
here produced **the same rank correlation on both sides (+0.930)**, so the browser
engine and the Python engine agree on the labelled outfits. At 16 labels the
scoreboard still reports INDICATIVE ONLY and the fitter still refuses to claim an
improvement, which is the behaviour we want.

## Design decisions

- [ADR 0001 — a solver, not a recommender](docs/adr/0001-solver-not-recommender.md)
- [ADR 0002 — rules as data, and the `eval` boundary](docs/adr/0002-rules-as-data.md)
- [ADR 0003 — why `lift` is defined against a barefoot reference](docs/adr/0003-lift-metric.md)
- [ADR 0004 — the eval harness, and what it refuses to do](docs/adr/0004-eval-harness.md)
- [The pitch](docs/pitch.md) — the stylist-first wedge, the moat, and the risks
- [Language policy](docs/language-policy.md) — describe garments and geometry,
  never a body as a problem. Enforced in all 17 message templates and asserted in
  tests, not left to a prompt.

## Honest limits

- `volume` and `structure` are hand-judged by one person on a 0–1 scale. No
  inter-annotator agreement yet, so the balance rules are calibrated to one eye.
- Landmarks are hand-measured from one frame. Untested on a second body.
- Rule weights were set by judgement. They are not fit to anything — that needs
  the labelled eval set in phase 4.
- Monocular proportion estimation from clothed photos (phase 1) is genuinely
  hard. The design reports confidence and falls back to manual entry rather
  than pretending otherwise.

## Roadmap

Phase 0 ✅ · W1 garment parsing · W2 attribute extraction (discriminative vs.
LoRA VLM, head to head) · W3 frame extraction · W4 compatibility · W5 bandit
personalisation · W6 a South Asian garment dataset supplement — kurta, saree,
lehenga, salwar, where the drape itself creates the break and the existing
datasets have close to nothing.

Full plan in the project spec.
