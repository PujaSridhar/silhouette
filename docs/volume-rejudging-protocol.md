# Protocol: the noise floor of a judged volume

Measured volume (W1) can only be compared with judged volume up to the
judge's own consistency. This protocol measures that consistency. It costs
two short sittings and no GPU.

## Blind re-judging, ten garments, weeks apart

1. **Make the sheet:** `silhouette-vision rejudge-sheet`. It writes
   `evals/rejudge/sheet.csv` (gitignored) with ten garments in shuffled order,
   names only, no current values. The ten sit on or near the 1.00, 1.15 and
   1.30 thresholds, plus two far from any as anchors.
2. **Session 1.** Wear or hang each garment as you would when judging it,
   same light, same mirror. Write `session_1_volume` to the nearest 0.05 and,
   if you like, `session_1_structure` (also hand-judged, also uncalibrated).
   Do not open `wardrobe.csv` first, and fill in the date.
3. **Cover it.** Hide the session 1 columns (move them to another file if that
   is easier) so session 2 cannot see them.
4. **Session 2, two to three weeks later**, same conditions, same order.
5. **Read it:** `silhouette-vision rejudge`. It reports one judgement's SD
   (the SD of the session differences over √2), the mean shift between
   sessions, the largest disagreement, the share of pairs within ±0.05, and
   how session 1 compares with the original CSV values.

Rules that keep it honest: no peeking at earlier numbers, ever; do not change
`wardrobe.csv` because of what this shows; record what you got even if it is
embarrassing. Ten garments is small: the SD estimate is good to roughly ±25%
of itself, which is enough to tell 0.03 from 0.10.

## What the number is for

Set it beside the measurement's `repeat` SD from `silhouette-vision measure`:

- measured SD well below yours: the mask is the steadier input;
- similar: the mask buys convenience, not precision;
- measured SD above yours: the photos or the masks need work before the
  measurement replaces anything.

If one judgement's SD is above 0.05 itself, the ±0.05 target is tighter than
the reference it is measured against. Say so when reporting W1.

## Timing note: wardrobe entry with the first stylists

The case for photo capture assumes manual entry is slow. Nobody but Puja has
entered a wardrobe, so that is untested. With the first three to five
stylists recruited for rating:

- time a manual entry of their own wardrobe, start to finish, and per
  garment (median), and count the garments;
- note where they stop or ask a question, and which fields they leave blank
  or guess;
- if photo capture exists by then, time the same garments by photo, including
  the time spent fixing what the model got wrong.

Record minutes, garment counts and the fields that caused trouble; no names
needed. If manual entry turns out to take minutes rather than an hour, photo
capture is justified by consistency (above), not by speed.
