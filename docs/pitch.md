# Silhouette — the pitch

> Working draft for YC. Everything here is either built and verifiable, or
> marked **[NEEDS DATA]** where a real number has to replace a claim. Nothing in
> this document invents a market size, a growth rate or a user count. An
> application that cites a figure you cannot source is worse than one that says
> "we have not measured this yet".

---

## The 50-character line

> **Styling software that can explain and prove itself.**

## What we're making

Silhouette turns styling from taste into measurement. A stylist photographs a
client's wardrobe; the system builds a **Style CV** — a structured model of that
person's proportions, goals and constraints — then assembles outfits against
explicit styling rules and explains each one in the stylist's own vocabulary.

The engine is a constraint solver, not a recommender. It does not say "people
who liked this also liked that". It says:

> The break sits at 38% of your height instead of near the middle, so the eye
> reads about 12% more length below it.

Live demo (JS port, verified against the Python engine on every case):
see the artifact link in the README.

## Why this idea, and why me

I style myself and the people around me, and I have been doing it long enough
to know the advice is rule-shaped. I choose high-waisted wide-leg trousers with
a crop because it moves the visual break from roughly half my height to about a
third, and the eye reads the lower section as longer. I pair volume below with a
close top because my lower body carries more visual width — and past a point I
add *structure* above, not more fabric, or the silhouette reads bottom-heavy.

Those are two real rules with measurable inputs. I wrote them into the engine
and it reproduces them from geometry. It also produced the nuance I only
mentioned in passing, from a volume-asymmetry threshold rather than from being
told:

> This works, and it has room to go further — the top is soft (structure 0.10)
> against x1.95 volume below. A shoulder seam, a sleeve detail or an open jacket
> would add weight up top without adding width.

**How I know people need it:** the people I style cannot do it for themselves,
and what they lack is not clothes or taste — it is a *reason*. They can follow a
rule once they understand the mechanism. That is the product.

**[NEEDS DATA]** Before applying: talk to 20 working stylists. Record how they
currently deliver a lookbook, what they charge, how long a client session takes,
and what they'd pay to halve it. Quote three of them by name in the application.

## What's new

Three things, in order of defensibility.

**1. The architecture.** Solver for structure, model for taste, language model
for explanation only. The LLM never picks the outfit; it describes what the
solver chose. Explanations are rendered from the rule that actually fired, with
the numbers that made it fire, so they cannot hallucinate. Nobody shipping in
this space does this — the field is embedding-and-rank.

**2. A metric nobody else reports.** `lift` = apparent leg ratio minus actual,
in fractions of the wearer's own height. It is exactly what someone is buying
when they choose a high rise, it is computable from a photo, and it goes
negative — an untucked hip-length top over a high rise reads *shorter* than the
body actually is. A goodness score can never tell you that.

**3. An eval harness, which is the real moat.** We measure whether our advice is
right, against expert-labelled outfits, with a validated judge and a published
self-agreement ceiling. Fashion-AI companies ship recommendations nobody can
grade. We can grade ours, and the labels compound.

## What we understand that others don't

**Styling is mostly deterministic.** The industry treats it as taste, so it
builds taste-shaped software — feeds, similarity, influencers. Most of the
actual work is proportion, balance and line, and those are geometry. Taste is
the last 20%, and it is the only part that needs a learned model.

**The explanation *is* the product.** A recommendation you cannot interrogate
gets followed once. A mechanism you understand gets reused for life. Retention
in consumer wardrobe apps is brutal precisely because they deliver outfits
instead of understanding.

**Cold start is a labour problem, not a tech problem.** Every wardrobe app dies
on the same step: somebody has to photograph 80 garments before the product does
anything. Consumers won't. **Stylists already do this, and get paid for it.**
That single observation is why we sell to stylists first.

## The wedge: stylists, not consumers

| | consumer-first | stylist-first (ours) |
|---|---|---|
| who digitises the wardrobe | the user, for free, reluctantly | a professional, already paid to |
| value per wardrobe | one person | every client they advise |
| willingness to pay | very low | it is a work tool |
| labelled data produced | none | every session is graded ground truth |

A working stylist hands clients a lookbook assembled by hand. The Style CV *is*
that deliverable, generated, with the reasoning attached. We are selling them
back their own hours.

Then the data flywheel: each session produces expert-labelled outfits, which fit
the rule weights, which make the engine better, which makes the tool worth more
to the next stylist. The labels are the asset and only we have them.

Consumer comes later, seeded by the clients who already received a Style CV.

## Why now

- **Garment attribute extraction became a feature, not a company.** Open VLMs
  now read rise, length, drape and fit from a photo well enough to fine-tune
  cheaply. Three years ago that was the whole startup. The hard part has moved
  up the stack to reasoning and evaluation, which is where we are.
- **Evaluation is the differentiator nobody built yet.** The industry is
  shipping ungraded generative recommendations. Buyers are starting to ask how
  anyone knows they work.
- **Personal styling went independent.** Stylists increasingly work for
  themselves, on Instagram and WhatsApp, with no software built for them.
  **[NEEDS DATA]** size this properly before applying.

## Competition

- **Consumer wardrobe apps** (Whering, Acloset, Stylebook, Indyx). Real users,
  real products. They optimise cataloguing and outfit logging. None of them
  reasons about proportion or explains a choice, and all of them carry the
  consumer cold-start problem.
- **Retail styling and try-on** (Stitch Fix, Doji, virtual try-on vendors).
  Aimed at selling inventory. Their incentive is a purchase; ours is making what
  you already own work, which is a different product and a different trust
  relationship.
- **The one I actually fear:** a large model provider shipping "upload your
  wardrobe, get outfits" as a generic multimodal feature. The defence is not the
  rules — those are reproducible. It is the expert-labelled eval set, the fitted
  weights, and the garment coverage nobody else has bothered with (below). We
  should assume the generic version arrives and be the one with evidence.

## The dataset nobody built

Fashionpedia, DeepFashion2 and Polyvore are overwhelmingly Western ready-to-wear.
They have close to nothing for saree, kurta, lehenga, salwar or drape-based
garments generally — where **the garment itself creates the proportion break**,
so rise-and-hem features do not even apply.

We are annotating that gap under an extended schema and will report accuracy on
Western versus South Asian garments before and after. It is a real research
contribution, it is directly commercial in India, and it is the piece a generic
competitor is least likely to replicate.

## Business model

- **Stylist seats.** Monthly per-seat SaaS. **[NEEDS DATA]** price from the 20
  interviews, not from a spreadsheet.
- **Client tier.** A client who received a Style CV keeps access to their own
  wardrobe. Low price, high margin, zero acquisition cost.
- **Shopping, carefully.** Gap analysis already computes which single purchase
  unlocks the most outfits. That is genuinely useful and genuinely corruptible.
  If we ever take affiliate revenue it must not influence ranking, and the
  ranking must stay auditable — which our architecture uniquely allows us to
  prove. **[NEEDS DATA]** model this only after seat revenue exists.

**[NEEDS DATA]** "How much could you make" needs a real bottom-up estimate:
addressable stylists x realistic penetration x price. Build it from the
interviews.

## How far along

Built and verifiable today:

| | |
|---|---|
| rules engine | 17 rules as versioned data, each with a must-fire test |
| wardrobe | 30 hand-annotated garments, 1,548 valid assemblies |
| solver | exhaustive scoring at 158 µs/candidate; CP-SAT week planning with slack-based bottleneck diagnosis |
| eval harness | rubric, active sampler, Spearman / quadratic-weighted kappa, judge + baseline, prior-anchored non-negative weight fitting with cross-validation |
| tests | 97, including an end-to-end test that the fitter recovers a synthetic rater's known preferences from labels alone |
| demo | interactive, JS port verified 76/76 against the Python engine in the browser |

**Not built, and we say so:** no vision model yet (garments are hand-annotated),
no app, no users, no revenue, and zero expert labels — the harness is validated
against a synthetic rater, deliberately, because inventing labels would be
inventing evidence.

**[NEEDS DATA]** runway, hours/week, and whether this is full-time.

## Traction plan to first users

1. **Label 500 outfits myself.** Turns the harness from plumbing into evidence
   and fits the rule weights. This is the one thing only I can do.
2. **Style 10 people with it**, free, in person. Record where the engine was
   wrong. Those become rules.
3. **Three working stylists**, free, in exchange for labels. If they stop using
   it, that is the answer.
4. Only then build the photo pipeline.

## The honest risks

- **Solo, technical, no distribution.** Stylists are reached through Instagram
  and word of mouth, which I have not done.
- **Consumer fashion is a graveyard.** Mitigated by selling to stylists, but the
  consumer tier could still be a mirage.
- **The rules may be mine, not universal.** Everything is calibrated to one eye
  and one body so far. If a second stylist disagrees with the weights, that is a
  finding, and the harness is already built to measure it.
- **Proportion estimation from clothed photos is genuinely hard.** We report
  confidence and fall back to manual entry rather than pretending otherwise.

## The one-line version

> Every fashion AI company ships recommendations nobody can check.
> We built the thing that checks them, and the styling engine is what we
> check it with.
