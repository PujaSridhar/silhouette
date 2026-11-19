"""Render a scored outfit for a human.

Every line here is produced from a rule that actually fired, with the numbers
that made it fire. Nothing is generated prose. In later phases a language model
smooths these into conversation - but it will be rewriting these sentences, not
inventing reasons, and that distinction is the whole trust model.
"""

from __future__ import annotations

from .solver import ScoredOutfit

__all__ = ["headline", "detail", "render"]


def headline(s: ScoredOutfit) -> str:
    f = s.features
    bits = []
    if f.break_ratio is not None:
        bits.append(f"break {f.break_ratio:.0%}")
        bits.append(f"lift {f.lift:+.3f}")
        bits.append(f"apparent lower body {f.apparent_leg:.0%}")
    else:
        bits.append("no break (column)")
    bits.append(f"dE {f.contrast_at_break:.0f}")
    bits.append(f"volume {f.upper_volume:.2f}/{f.lower_volume:.2f}")
    if f.heel_cm:
        bits.append(f"heel {f.heel_cm:.0f}cm")
    return "   ".join(bits)


def detail(s: ScoredOutfit, indent: str = "    ") -> list[str]:
    lines: list[str] = []
    for fr in s.verdict.positives:
        lines.append(f"{indent}+ [{fr.contribution:+.2f}] {' '.join(fr.message.split())}")
    for fr in s.verdict.negatives:
        lines.append(f"{indent}- [{fr.contribution:+.2f}] {' '.join(fr.message.split())}")
    for rule, msg in s.verdict.violations:
        lines.append(f"{indent}x REJECTED by {rule.id}: {' '.join(msg.split())}")
    return lines


def render(s: ScoredOutfit, rank: int | None = None, indent: str = "    ") -> str:
    tag = f"#{rank}  " if rank is not None else ""
    head = f"{tag}{s.outfit.label()}"
    out = [f"{head}\n{indent}{'score':<6} {s.total:+.2f}   {headline(s)}"]
    out.extend(detail(s, indent))
    return "\n".join(out)
