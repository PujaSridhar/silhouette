"""The rules DSL: loader, evaluator, verdict.

Rules are DATA, in rules/*.yaml, not code. That is the central bet of this
project: the styling knowledge is the asset, so it has to be reviewable,
diffable and testable by someone who can style but does not write Python.

Each rule is one claim with one message. If a condition deserves two different
explanations (a bonus and a penalty), it is two rules. This is why explanations
are trustworthy: they are rendered from the rule that actually fired, with the
numbers that made it fire, not written by a language model guessing at reasons.

  kind: hard   `forbid` is a condition that must NOT hold. If it does, the
               outfit is infeasible and `reject` says why.
  kind: soft   `when` gates the rule; `score` returns roughly -1..1; the
               contribution is weight * score.

A rule carrying `goal:` only applies if that goal is in the wearer's Style CV.
Someone who has not asked to look taller does not get lectured about leg line.

On evaluation: expressions are `eval`'d with no builtins beyond a small
numeric whitelist. Rule files are first-party content living in the repo
alongside the code, at the same trust level as the code - they are not user
input, and nothing in the product path lets a user supply one. If that ever
changes, this becomes a parsed expression tree instead. See docs/adr/0002.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .geometry import Features
from .models import Context, Frame, Outfit, StyleCV

__all__ = ["Rule", "FiredRule", "Verdict", "load_rules", "evaluate"]


class _Null:
    """Stands in for an empty slot so rules need no `is not None` boilerplate.

    Attribute access yields None, so `bottom.rise in ('high',)` is False rather
    than an AttributeError when there is no bottom at all.
    """

    __slots__ = ()

    def __getattr__(self, _name: str) -> None:
        return None

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<empty slot>"


NULL = _Null()


def _norm(x: float | None, lo: float, hi: float) -> float:
    """Map x from [lo, hi] onto [0, 1], clamped. None -> 0."""
    if x is None:
        return 0.0
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (float(x) - lo) / (hi - lo)))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


_SAFE_BUILTINS: dict[str, Any] = {
    "min": min, "max": max, "abs": abs, "len": len, "round": round,
    "any": any, "all": all, "sum": sum, "sorted": sorted,
    "True": True, "False": False, "None": None,
}


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str                      # soft | hard
    weight: float = 1.0
    goal: str | None = None
    when: str | None = None
    forbid: str | None = None
    score: str | None = None
    explain: str | None = None
    reject: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("soft", "hard"):
            raise ValueError(f"{self.id}: kind must be soft|hard, got {self.kind!r}")
        if self.kind == "hard":
            if not self.forbid:
                raise ValueError(f"{self.id}: hard rule needs `forbid`")
            if not self.reject:
                raise ValueError(f"{self.id}: hard rule needs `reject`")
        else:
            if not self.when:
                raise ValueError(f"{self.id}: soft rule needs `when`")
            if not self.explain:
                raise ValueError(f"{self.id}: soft rule needs `explain`")


@dataclass(frozen=True)
class FiredRule:
    rule: Rule
    score: float
    contribution: float
    message: str


@dataclass(frozen=True)
class Verdict:
    feasible: bool
    total: float
    fired: list[FiredRule]
    violations: list[tuple[Rule, str]]
    skipped_goals: list[str]

    @property
    def positives(self) -> list[FiredRule]:
        return sorted([f for f in self.fired if f.contribution > 0],
                      key=lambda f: -f.contribution)

    @property
    def negatives(self) -> list[FiredRule]:
        return sorted([f for f in self.fired if f.contribution < 0],
                      key=lambda f: f.contribution)


def load_rules(path: str | Path) -> list[Rule]:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    rules = [Rule(**r) for r in raw["rules"]]
    ids = [r.id for r in rules]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate rule ids: {sorted(dupes)}")
    return rules


def _namespace(outfit: Outfit, f: Features, frame: Frame, ctx: Context) -> dict[str, Any]:
    return {
        "top": outfit.top or NULL,
        "bottom": outfit.bottom or NULL,
        "dress": outfit.dress or NULL,
        "outer": outfit.outer or NULL,
        "shoe": outfit.shoe or NULL,
        "drape": outfit.drape or NULL,
        "tucked": outfit.tucked,
        "items": outfit.items,
        "f": f,
        "frame": frame,
        "ctx": ctx,
        "norm": _norm,
        "clamp": _clamp,
    }


def _eval(expr: str, ns: dict[str, Any], rule_id: str, field: str) -> Any:
    try:
        return eval(expr, {"__builtins__": _SAFE_BUILTINS}, ns)  # noqa: S307
    except Exception as exc:  # pragma: no cover - surfaces authoring mistakes
        raise RuntimeError(f"rule {rule_id!r}, {field} expression failed: {expr!r} -> {exc}") from exc


def _facts(outfit: Outfit, f: Features, ctx: Context) -> dict[str, Any]:
    d = dict(f.facts())
    d.update(
        ctx_name=ctx.name,
        formality_floor=ctx.formality_floor,
        formality_ceiling=ctx.formality_ceiling,
        clo_min=ctx.clo_min,
        clo_max=ctx.clo_max,
        top_name=outfit.top.name if outfit.top else "",
        bottom_name=outfit.bottom.name if outfit.bottom else "",
        dress_name=outfit.dress.name if outfit.dress else "",
        outer_name=outfit.outer.name if outfit.outer else "",
        shoe_name=outfit.shoe.name if outfit.shoe else "",
        drape_name=outfit.drape.name if outfit.drape else "",
        dirty=", ".join(g.name for g in outfit.items if g.availability != "clean"),
    )
    return d


class _SafeFormatter(string.Formatter):
    """Renders a None field as "n/a" instead of raising.

    A rule can legitimately fire on an outfit where some unrelated feature is
    undefined - `apparent_leg` is None for a draped column garment, for
    instance. A message template referencing it should degrade, not abort a
    scoring run over 1,500 candidates.
    """

    def format_field(self, value: Any, format_spec: str) -> str:
        if value is None:
            return "n/a"
        return super().format_field(value, format_spec)


_FORMATTER = _SafeFormatter()


def _render(template: str, facts: dict[str, Any], rule_id: str) -> str:
    try:
        return _FORMATTER.vformat(template, (), facts)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"rule {rule_id!r}: could not render message: {exc}") from exc


def evaluate(
    outfit: Outfit,
    f: Features,
    cv: StyleCV,
    ctx: Context,
    rules: list[Rule],
) -> Verdict:
    ns = _namespace(outfit, f, cv.frame, ctx)
    facts = _facts(outfit, f, ctx)

    fired: list[FiredRule] = []
    violations: list[tuple[Rule, str]] = []
    skipped: list[str] = []
    total = 0.0

    for r in rules:
        if r.goal and r.goal not in cv.goals:
            if r.goal not in skipped:
                skipped.append(r.goal)
            continue

        if r.kind == "hard":
            if _eval(r.forbid, ns, r.id, "forbid"):
                violations.append((r, _render(r.reject, facts, r.id)))
            continue

        if not _eval(r.when, ns, r.id, "when"):
            continue

        raw = 1.0 if r.score is None else float(_eval(r.score, ns, r.id, "score"))
        contribution = r.weight * raw
        if contribution == 0.0:
            continue
        total += contribution
        fired.append(FiredRule(r, raw, contribution, _render(r.explain, facts, r.id)))

    return Verdict(
        feasible=not violations,
        total=total,
        fired=fired,
        violations=violations,
        skipped_goals=skipped,
    )
