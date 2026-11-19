# Evaluation rubric

The engine produces a score. This rubric produces the ground truth that score
is measured against. Without it, "the engine works" is an opinion.

Every outfit is rated on five dimensions, 1-5, plus one holistic judgement and
one binary. The holistic rating is **not** derived from the five - it is rated
independently, because the whole point is to find out whether the dimensions
the engine reasons over actually add up to the judgement a stylist makes.

## Dimensions

### 1. Proportion (1-5)
Does the visual break sit where it should, given the goals in the Style CV?

| | |
|---|---|
| 5 | The break is working hard. You would point at this to explain the principle. |
| 4 | Break is in the right region; nothing wasted. |
| 3 | Neutral. Neither helps nor hurts. |
| 2 | The break works against the stated goal. |
| 1 | Actively shortens or cuts the line. You would change it before leaving. |

### 2. Balance (1-5)
Does the distribution of volume read as deliberate?

| | |
|---|---|
| 5 | Volume and close fit are placed so the silhouette is unmistakable. |
| 4 | Clearly intentional, minor slack. |
| 3 | Not wrong, not saying anything. |
| 2 | Reads accidental - the eye cannot find an edge. |
| 1 | Both halves fighting. Shape is lost. |

### 3. Formality (1-5)
Right register for the occasion?

| | |
|---|---|
| 5 | Exactly the register. |
| 4 | Slightly over or under, nobody notices. |
| 3 | Noticeably off but survivable. |
| 2 | Wrong register; you would feel it all evening. |
| 1 | Would not be let in, or would be the only one dressed like this. |

### 4. Colour (1-5)
Does the palette work on this person, and does contrast land where it helps?

### 5. Cohesion (1-5)
Does it read as one outfit rather than three garments? Pattern, texture,
period, and formality consistency.

## Holistic

### Overall (1-5)
Rated independently, before or after the dimensions but **not** computed from
them. "How good is this outfit on this person for this occasion?"

### Would wear (yes/no)
Would you actually put someone in this and send them out? This is the headline
metric - **top-3 acceptance rate** is the fraction of the engine's top three
suggestions that get a yes.

## Labelling protocol

1. Rate in batches of no more than 50. Fatigue shows up as regression to 3.
2. The Style CV and the context are visible; the engine's score and its
   explanations are **hidden**. Seeing the score anchors the rating and
   destroys the independence the whole harness depends on.
3. Re-label 10% of outfits in a later session without seeing the earlier
   rating. That gives **self-agreement**, which is the ceiling on how well any
   model can possibly do. A model scoring above your own self-agreement is
   overfitting, not succeeding.
4. Record a one-line note on anything rated 1, 2 or 5. Those notes are where
   new rules come from.

## What gets reported

| metric | question it answers |
|---|---|
| Spearman rho (engine total vs. overall) | does the engine's score rank outfits the way the stylist does? |
| Quadratic-weighted kappa (judge vs. overall) | does the judge agree beyond chance? |
| MAE | how far off, in rating points? |
| Top-3 acceptance | would you wear what it suggests? |
| Self-agreement | the ceiling. Nothing may be reported as beating it. |
| Per-rule residual | which rules are mis-weighted. This is what feeds weight fitting. |

## Honesty rules for this harness

- A metric computed on fewer than 60 labels is reported with its sample size
  and no conclusion.
- Bootstrap labels (generated, not rated by a human) may exercise the pipeline
  and may never be reported as quality evidence. The scoreboard refuses to draw
  conclusions from them.
- Judge agreement is reported *against expert labels*, never against the
  engine's own score. A judge that agrees with the engine tells you nothing.
