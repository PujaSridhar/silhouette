# Phase 0 findings

Phase 0 exists to answer one question before any GPU money is spent:

> **Does the engine produce outfits a stylist would actually put someone in?**

Thirty hand-annotated garments, seventeen rules, an exact solver, no machine
learning anywhere. Here is what it produced.

---

## The gate: pass on reasoning, with one calibration defect

The wearer's own go-to outfit - ivory ribbed crop, black super-high wide-leg,
sneakers - scores **+1.74** and ranks **75th of 499** feasible weekend
assemblies. That is the top 15%, not the top of the list, and the reason it is
not first turned out to be worth more than a pass would have been: see
finding 10.

What matters for the gate is that the engine states her two rules back to her
without having been told them as prose:

```
+ [+0.90] Volume below (x1.95 body width) against a close top (x0.80) keeps
          the silhouette legible instead of widening the whole column.
+ [+0.85] The break sits at 38% of your height instead of near the middle,
          so the eye reads about 12% more length below it.
+ [+0.35] The break is already high at 38%, so the colour change there
          (dE 84) works for you - it marks the high waist rather than
          cutting you in half.
- [-0.24] This works, and it has room to go further - the top is soft
          (structure 0.10) against x1.95 volume below. A shoulder seam, a
          sleeve detail or an open jacket would add weight up top without
          adding width.
```

That last line is the one worth staring at. Nobody wrote "a crop top with wide
trousers has headroom for a blazer". It falls out of the volume asymmetry
crossing a threshold, and it is the stylist's own nuance - *balance it, but do
not add fabric* - arrived at from geometry.

**It is therefore worth building phase 1** - the reasoning is sound. The
*weights* are not yet, which is exactly what finding 10 is about and exactly
what the phase 4 eval set exists to fix.

---

## Findings, in order of how much they changed the design

### 1. A formality ceiling cannot be a hard constraint

The spike rejected the wearer's signature outfit for a weekend, because a black
wide-leg trouser scores 0.60 formality against a weekend ceiling of 0.52.

The modelling error: being **underdressed is a failure, being overdressed is a
preference.** Formality is not a band with two hard walls. The floor is hard;
the ceiling is a soft nudge at most.

Fixed by splitting one hard rule into `meets_context_formality_floor` (hard) and
`overdressed_for_context` (soft, weight 0.50). Locked in by
`test_overdressed_is_soft_not_fatal`.

This is exactly the class of error that no amount of model training would have
caught, and that cost one afternoon to find at phase 0 rather than month four.

### 2. The wardrobe has a shoe bottleneck, found twice independently

Two different analyses converged on the same conclusion without being designed
to agree:

- **Gap analysis** ranked a second dressy heel as winning the most contexts
  (4 of 5) and unlocking the most assemblies (269).
- **Week planning** came back infeasible at two wears per garment, and the
  slack diagnosis named the reason: the black block heel is needed three times.

The cause: exactly one shoe in thirty garments clears a 0.45 formality floor.
Office, dinner and wedding all collapse onto it.

Two independent methods agreeing is the strongest signal phase 0 produced.

### 3. Infeasibility must be a diagnosis, not an exception

The first week planner raised `no feasible week under max_wears=2`. Technically
correct, completely useless. Rewritten with per-garment slack variables and a
penalty large enough to dominate any styling gain, so the solver always returns
a week *and* names which garments the wardrobe is short of.

A constraint solver that answers "no" is a bug in the modelling, not an answer.

### 4. Top-N collapsed onto a single garment

The first three office suggestions were three different tops on the same floral
palazzo. That is one suggestion presented three times.

Fixed with `max_per_anchor` in `best()`, capping how many suggestions share a
bottom or dress. The cost is small - the top pick is unchanged, the ranking
below it loses a few hundredths - and the output becomes actually useful.

### 5. The two solver backends were optimising different problems

CP-SAT scored **+12.30** against greedy's **+12.62** and looked worse. It was
not: greedy had broken the wear limit four times to get there, CP-SAT once.

Greedy was taking the best-scoring outfit when it could not respect the limit;
CP-SAT minimises broken limits first, then score. Comparing their raw totals was
a category error. Greedy now shares the lexicographic objective, and the test
compares `(overage, -score)` rather than score alone.

Current standing, same week, same wardrobe:

| limit | CP-SAT | greedy | overage |
|---|---|---|---|
| 2 wears | **+12.30** | +12.20 | 1 each |
| 3 wears | **+12.66** | +12.62 | 0 each |

The margins are small because thirty garments is a small problem. The point is
the ordering is now correct and asserted.

### 6. Negative lift is real, and it is the most useful number here

An untucked hip-length top over a high rise does not merely waste the rise. It
produces **lift = -0.019**: the outfit reads *shorter* than the wearer actually
is. The high rise sits at 40% of height, the hem covers it at 52%, and the eye
takes the lower number.

The same two garments, tucked, score **+1.98** against **+0.83** untucked. One
decision, no shopping, a 1.15-point swing. This is the single best demo the
project has.

### 7. Gap analysis picked an item for a reason nobody planted

The top recommendation by score delta is a **fitted bodysuit** - not the blazer,
not the trousers. The engine's reasoning, recovered from which rules changed:
a bodysuit *structurally cannot untuck*, so it permanently removes the
`hem_drop` failure mode that finding 6 is about.

That is a genuine piece of styling insight produced by combinatorics rather
than taste, and it is the first evidence the approach generalises past the rules
that were hand-written into it.

### 8. A rule exists that this wardrobe can never trigger

`pattern_competes_with_pattern` needs two patterned garments in one assembly.
The wardrobe has two patterned pieces, but they are a bottom and a dress, which
are mutually exclusive slots. The rule is correct, unit-tested with a synthetic
fixture, and unreachable in practice.

Recorded by `test_wardrobe_cannot_trigger_pattern_overload`, which enumerates
every candidate the solver can build and asserts the gap directly. It will fail
the day a patterned top is added - which is the moment the note stops being
true.

Worth knowing before trusting a green test suite as evidence the rules work.

### 9. Occasion coverage is wildly uneven

Feasible assemblies out of 1,548 candidates:

| context | feasible |
|---|---|
| campus | 729 |
| weekend | 499 |
| office | 128 |
| dinner | 122 |
| wedding_function | **12** |

Wedding dressing is 0.8% of the space. A recommender trained on uniform samples
would almost never learn it. A solver finds all twelve every time - which is a
concrete argument for the architecture, not just a preference.

### 10. The engine over-rewards dressiness in casual contexts

The signature outfit ranks 75th rather than 1st for a weekend. Its top pick:

```
1. Ivory ribbed crop + Floral printed palazzo + Black structured blazer
   + Black block heel                                            +2.49
   ...
   Ivory ribbed crop + Black super-high wide-leg + Black block heel   +1.85
   Ivory ribbed crop + Black super-high wide-leg + White sneaker      +1.74  <- hers
```

Note what the engine kept and what it changed. It kept her crop top and her
entire strategy - high break, close top, volume below. It swapped the sneaker
for a block heel and added a structured blazer, which are precisely the two
things its own rules advised (`heel_extends_the_leg_line`, and
`structure_above_balances_volume_below` answering the `flat_upper_under_high_volume`
nudge).

So the reasoning is coherent. The *calibration* is not. Those two bonuses carry
weights 0.35 and 0.60, totalling 0.95, while the only thing pushing back on
dressiness for a weekend is `overdressed_for_context` at weight 0.50 - and that
one measures garment formality, not whether a heel is a sensible choice for a
Saturday. The engine will therefore always prefer a heel and a blazer, in every
context, because nothing in the rule set knows that effort has a cost.

Two candidate fixes, neither applied yet because this is precisely the kind of
weight question that should be settled by the labelled eval set rather than by
me guessing twice:

1. Make `heel_extends_the_leg_line` and the structure bonus context-sensitive,
   scaling with the context's formality floor.
2. Add an `effort` dimension to garments, separate from formality, and penalise
   effort above what the occasion asks.

Recording it as a known defect is the honest move at phase 0. It is also a
concrete argument for building the eval harness before adding more rules: with
500 labelled outfits this is a fitting problem, and without them it is an
argument.

---

## What phase 0 deliberately did not do

No model, no photos, no embeddings, no language model. Every number traces to a
hand-annotated CSV cell and a rule in a YAML file. That is the point: when the
vision models arrive in phase 1, this engine is the thing that tells us whether
their predictions are any good.

## Performance

1,548 candidates enumerated and fully scored in **245 ms** (158 us each),
single-threaded, pure Python. Exhaustive search is not a compromise at this
scale - it is exact and it is fast. The solver earns its keep on the *week*,
where the days are coupled through a shared wardrobe and enumeration is
impossible.

## Honest limits

- `volume` and `structure` are hand-judged on a 0-1 scale by one person. No
  inter-annotator agreement exists yet, so treat the balance rules as
  calibrated to one eye.
- Landmark positions are hand-measured from one frame. Nothing here has been
  tested against a second body.
- Rule weights were set by judgement, then adjusted once when finding 1
  surfaced. They are not fit to anything.
- Thirty garments is enough to prove the engine reasons. It is not enough to
  prove the rule *weights* are right. That needs the labelled eval set in
  phase 4.
