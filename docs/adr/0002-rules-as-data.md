# ADR 0002 - Rules are data, and expressions are evaluated

**Status:** accepted, with a stated boundary

## Context

The styling knowledge is the asset. It has to be editable by someone who can
style but may not write Python, reviewable in a diff, and testable.

## Decision

Rules live in `rules/*.yaml`. Conditions and scores are Python expressions
evaluated at scoring time with `eval`, against a restricted namespace: the
outfit slots, the computed features, the frame, the context, and two numeric
helpers. Builtins are replaced with a nine-function numeric whitelist.

Each rule carries exactly one message. If a condition deserves two different
explanations, it is two rules.

## Why `eval` rather than a parsed expression tree

The alternative is a small expression language with its own parser. That is
roughly 300 lines to write and maintain, and it would be used to express things
like `0.26 <= f.break_ratio <= 0.43`, which Python already says perfectly.

**The security boundary that makes this acceptable:** rule files are
first-party content in the repository, at the same trust level as the code
beside them. Nothing in the product path lets a user supply a rule file. A user
supplies garment annotations and photos, which are data, never expressions.

**If that ever changes** - user-authored rules, rule sharing between wearers, a
rules marketplace, rules fetched from a server - this decision is void and the
expression language gets written before the feature ships. There is no safe
sandbox for `eval` on untrusted input and this ADR is not claiming otherwise.

## Why one message per rule

Explanations are rendered from the rule that actually fired, with the numbers
that made it fire. A rule covering both a bonus and a penalty would need its
message to branch, and the branch would drift from the condition. Splitting
them keeps every message provably about its own condition.

This is what makes the explanations trustworthy rather than plausible, and it
is the property a language model must preserve in phase 1: it rewrites these
sentences, it does not invent reasons.

## Consequences

- Rules need tests exactly as much as code does. `tests/test_rules.py` carries a
  must-fire fixture per rule, plus a coverage test that fails when a rule is
  added without one.
- A rule authoring mistake surfaces as a `RuntimeError` naming the rule id and
  the failing expression, not a silent wrong score.
- Message templates render a `None` field as `n/a` rather than raising, so an
  undefined feature on an unusual outfit cannot abort a scoring run.
