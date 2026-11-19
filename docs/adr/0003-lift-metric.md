# ADR 0003 - `lift` is the headline metric

**Status:** accepted

## Context

The project needs one number that means something to a wearer, is computable
from a photo in later phases, and is comparable across outfits.

## Decision

```
lift = apparent_leg_ratio - true_leg_ratio(barefoot)

  true_leg_ratio  = (sole - hip) / barefoot_height          anatomy, fixed
  apparent_leg    = (total - break_y) / total               what the eye reads
  total           = barefoot_height + heel
```

`lift` is the illusion, measured in fractions of the wearer's own height.

## Why this definition specifically

The reference frame for `true_leg_ratio` is **barefoot**, while
`apparent_leg_ratio` includes the heel. That asymmetry is deliberate.

If both included the heel, adding a heel would raise the numerator and the
denominator together and `lift` would barely move - which is wrong, because a
heel visibly buys leg. With a barefoot reference, both levers a stylist actually
controls show up in one number:

- raising the break (higher rise, tucking, a shorter top) increases `lift`
- adding heel height increases `lift`
- nothing else does

Measured on the phase 0 reference frame (hip at 0.50, so `true_leg` = 0.50):

| outfit | break | lift |
|---|---|---|
| super-high wide-leg, crop, 7 cm heel | 0.369 | **+0.131** |
| super-high wide-leg, crop, flat | 0.380 | +0.120 |
| high-rise wide-leg, tucked blouse | 0.395 | +0.105 |
| mid-rise bootcut | 0.444 | +0.056 |
| low-rise jean | 0.484 | +0.016 |
| high rise, untucked hip-length top | 0.519 | **-0.019** |

## Why it matters that it goes negative

The last row is the most useful output the metric produces. An untucked
hip-length top over a high rise does not merely waste the rise - it reads
*shorter* than the wearer actually is. A purely positive "goodness score" could
never say that, and it is precisely the correction a stylist gives with their
hands.

## Note on `break_ratio` moving the other way

Adding a heel *lowers* `break_ratio` while *raising* `lift`. Both are correct:
the waistband has not moved on the body, it is simply a smaller fraction of a
taller total. Asserted in
`test_heel_increases_lift_without_moving_the_break_point`, because it reads like
a bug the first time you see it.

## Consequences

- `lift` requires a frame. Without landmarks there is no `true_leg_ratio` and no
  metric, which is why frame extraction (W3) is on the critical path.
- It is sensitive to hip landmark error. A 0.01 error in hip position is a 0.01
  error in `lift`, which is 8% of the full useful range. Phase 1 must report
  frame confidence alongside it rather than presenting it as exact.
- For draped and column garments there is no break, so `lift` is undefined.
  Those are scored by a separate rule (`drape_reads_as_one_column`), not by
  forcing a number that has no meaning.
