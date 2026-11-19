# ADR 0001 - A solver, not a recommender

**Status:** accepted, validated by phase 0

## Context

The obvious architecture for a wardrobe app is a learned ranker: embed the
garments, embed the outfits, train on preference data, serve top-k. It is what
the field does and what the datasets are shaped for.

## Decision

**Solver for structure. Model for taste. Language model for explanation only.**

A constraint solver assembles outfits against explicit rules. A learned model
re-ranks the feasible set on taste. A language model describes what the solver
chose, and never chooses.

## Why

1. **Explainability is the product, not a feature.** The thing users cannot get
   anywhere else is *why this works on you*. A ranker can produce a good outfit
   but cannot say which rule it honoured. Here every suggestion carries the
   rules that fired and the numbers that fired them.

2. **The sparse occasions are the valuable ones.** Phase 0 measured 12 feasible
   wedding assemblies out of 1,548 - 0.8% of the space. A model trained on
   uniform preference data would barely learn that region. A solver enumerates
   it exactly, every time.

3. **Hard constraints are genuinely hard.** "Is clean" and "clears the formality
   floor" are not preferences to be weighted. A ranker expresses them as large
   penalties and will occasionally violate them. A solver cannot.

4. **No training data exists on day one.** A rule set works from the first
   garment. A ranker needs hundreds of labelled outfits that do not yet exist -
   and producing them well requires the engine this ADR describes.

## Consequences

- The styling knowledge becomes an authored asset (`rules/*.yaml`) that must be
  maintained, reviewed and tested. See ADR 0002.
- Taste genuinely does need a learned model; rules handle structure, not charm.
  That is workstream W4, re-ranking a feasible set rather than generating.
- The rule weights are hand-set and currently fit to nothing. Phase 4's eval
  set is what turns them from judgement into calibration.

## Validation

Phase 0 ran the architecture end to end with no ML at all. It reproduced the
stylist's own reasoning on her own outfit, and found nine design problems - one
of which (ADR 0001's own premise about formality bands) was a modelling error
that no training run would have surfaced. See `docs/phase-0-findings.md`.
