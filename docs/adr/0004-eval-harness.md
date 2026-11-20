# ADR 0004 — The eval harness, and what it refuses to do

**Status:** accepted, built in phase 4

## Context

Phase 0 found that the rule *logic* was sound and the *weights* were guesses
(finding 10: two bonuses totalling 0.95 pushed every suggestion toward heels and
blazers because nothing knew that effort has a cost on a Saturday). Guessing a
second time would not be better. The fix needs labels, and labels need a harness
that cannot lie to us.

## Decisions

### 1. The holistic rating is not derived from the dimensions

A labeller rates proportion, balance, formality, colour and cohesion — and then
rates **overall** independently. If overall were computed from the five, the
harness could never discover that the dimensions we reason over fail to add up
to the judgement a stylist actually makes. That discovery is the most valuable
thing it could tell us, so the design has to leave room for it.

### 2. Labelling hides the engine's score

The rubric protocol hides the score and the explanations during labelling.
Seeing them anchors the rating, and an anchored label measures the labeller's
agreement with themselves-having-seen-the-answer. The whole harness depends on
that independence.

### 3. Re-labels are kept, never overwritten

A second rating of the same outfit in a later session gives **self-agreement**,
which is the ceiling on how well any model can do. A model scoring above the
labeller's own self-agreement is overfitting, not succeeding. Tidying duplicates
away would delete the only number that makes the others interpretable.

### 4. Bootstrap labels may validate the pipeline and may never be evidence

Two tempting shortcuts are both dishonest: random labels make the engine look
broken, labels derived from the engine make it look perfect. Neither says
whether the *harness* works.

So the synthetic rater uses the engine's own rule scores with **two weights
deliberately changed**, in the direction finding 10 hypothesised, and that is
stated in the module and the docs. A working harness must recover both from the
labels alone. It moves both the right way — `heel_extends_the_leg_line` 0.35 →
0.09 against a ground truth of 0.05, and `overdressed_for_context` 0.50 → 1.17
against 1.60, with held-out rank correlation rising from +0.896 to +0.908
(180 bootstrap labels, seed 7, 18 rules).

That is a falsifiable test of sampling, design matrix, fitting and
cross-validation at once. The scoreboard still refuses to draw quality
conclusions from these labels, and `EvalReport.caveat()` says so on every run.

### 5. A metric below the sample floor is reported, not concluded

Under 60 expert labels the scoreboard prints numbers with their sample size and
marks them indicative. A rank correlation on 12 labels has a confidence interval
wide enough to contain both "excellent" and "useless".

### 6. Weight fitting is anchored, non-negative and cross-validated

- **Anchored** to the hand-set weights rather than to zero, so with few labels
  the fit stays near the stylist's judgement and only moves where the data
  insists. Plain ridge would quietly delete rules that happen to be rare.
- **Non-negative**, enforced by projected gradient rather than by clipping a
  converged unconstrained solution. Direction lives in each rule's `score`
  expression; a negative weight would silently invert a rule while its
  explanation went on claiming the opposite.
- **Cross-validated against the prior.** `silhouette fit --write` refuses to
  write unless held-out rank correlation beats the hand-set weights on the same
  folds. A fit that wins on training data and loses out of sample is reported
  as a loss.

### 7. A judge is validated against expert labels, never against the engine

A judge that agrees with the engine has told you only that two of your own
components agree. Every judge is also reported beside `MeanJudge`, which
predicts the average for everything — any judge that cannot beat it is adding
nothing, and a rho quoted without that comparison hides the question.

`LLMJudge` is deliberately left unwired: the eval suite must run with no API key
and no network, or it stops being something you run on every change.

## Consequences

- The expensive resource is a human eye, so the sampler optimises for rule
  coverage and score spread rather than sampling uniformly. A rule with no
  labels has an unfalsifiable weight, and `silhouette coverage` names those.
- Hard constraints never appear in fitting: they are not weighted, and labels
  are sampled from feasible outfits only.
- The harness currently reports zero expert labels, which is the honest state.
  The next action is 500 labels, and it is the one thing only the stylist can do.
