# Language policy

An application that comments on bodies can cause real harm. This is a
constraint on the product, enforced in code and tests, not an aspiration.

## The rule

**Describe garments, geometry and colour. Never describe a body as a problem.**

Every message the engine emits is rendered from a rule that fired, so the policy
is enforceable by reviewing fourteen `explain` templates and three rejection
messages, rather than by hoping a language model behaves.

## Specifically

**Never used:** flattering, slimming, hides, disguises, camouflages, minimises,
problem area, figure flaw, balances out your…, cheats, fakes, tricks the eye
into thinking you are…, dresses for your shape.

**Used instead:** the break sits at X%, the eye reads N% more length, volume
below against a close top, adds visual weight without adding width, reads as one
continuous column.

The difference is not politeness. "This hides your hips" states that the hips
are a defect being concealed. "Volume below against a close top keeps the
silhouette legible" states what the garments do. The second is also more
*useful*, because it tells the wearer the mechanism and lets them apply it
again.

## Goals belong to the wearer

A goal such as `elongate_lower_body` appears in the Style CV only because the
wearer put it there. Rules carrying a `goal:` are skipped entirely for anyone
who has not asked for that outcome - asserted by
`test_goal_gating_silences_irrelevant_rules`.

Nobody gets told their legs could look longer unless they asked how to make
their legs look longer.

## Never inferred, at any phase

Weight, BMI, clothing size, gender, age, pregnancy, health status. Not as a
feature, not as an internal signal, not as a training target. The frame model
stores landmark positions and widths as fractions of height, which is what the
geometry needs and nothing more.

## Phase 1 and beyond

When a language model is introduced to smooth these messages into conversation,
it rewrites sentences that already exist. It does not generate reasons. A
classifier runs over all generated text against this policy, and the red-team
set lives in `evals/`.
