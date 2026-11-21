"""Label records.

One record is one human judgement of one outfit in one context. Stored as
JSONL so labels append cleanly, diff readably, and never need a migration.

`source` matters more than it looks:
  expert     rated by a human against evals/rubric.md. The only kind that
             counts as evidence.
  bootstrap  generated to exercise the pipeline. Never evidence. The
             scoreboard refuses to draw conclusions from these.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from ..models import outfit_slot_key

__all__ = ["Ratings", "Label", "label_from_dict", "CantJudge", "cant_judge_from_dict",
           "DIMENSIONS"]

DIMENSIONS = ("proportion", "balance", "formality", "colour", "cohesion")


@dataclass(frozen=True)
class Ratings:
    proportion: int
    balance: int
    formality: int
    colour: int
    cohesion: int

    def __post_init__(self) -> None:
        for d in DIMENSIONS:
            v = getattr(self, d)
            if not (1 <= int(v) <= 5):
                raise ValueError(f"{d} must be 1-5, got {v!r}")

    def as_vector(self) -> list[int]:
        return [int(getattr(self, d)) for d in DIMENSIONS]


@dataclass(frozen=True)
class Label:
    """One human judgement of one outfit in one context.

    `ratings` is optional on purpose. Only `overall` feeds weight fitting; the
    five dimensions exist to diagnose WHICH part the engine gets wrong. Making
    them mandatory would have meant seven taps per label instead of three, and
    the binding constraint on this project is how many labels a person will
    actually sit through. A quick label is worth more than a complete one that
    never gets made - but a label with invented dimension scores would be worth
    less than nothing, so absent means absent.
    """

    id: str
    outfit: dict[str, object]      # slot -> garment id, plus "tucked": bool
    context: str
    overall: int                   # rated independently, NOT derived
    ratings: Ratings | None = None
    would_wear: bool = True
    labeller: str = "unknown"
    source: str = "expert"         # expert | bootstrap
    note: str = ""
    engine_version: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        if not (1 <= int(self.overall) <= 5):
            raise ValueError(f"overall must be 1-5, got {self.overall!r}")
        if self.source not in ("expert", "bootstrap"):
            raise ValueError(f"source must be expert|bootstrap, got {self.source!r}")

    @property
    def outfit_key(self) -> str:
        """Stable identity for an outfit, for detecting re-labels."""
        return (outfit_slot_key(self.outfit.get)
                + f"|tucked={bool(self.outfit.get('tucked'))}|{self.context}")

    @property
    def has_dimensions(self) -> bool:
        return self.ratings is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ratings"] = asdict(self.ratings) if self.ratings else None
        return d


def label_from_dict(d: dict) -> Label:
    r = d.get("ratings")
    return Label(
        id=d["id"],
        outfit=d["outfit"],
        context=d["context"],
        ratings=Ratings(**{k: int(r[k]) for k in DIMENSIONS}) if r else None,
        overall=int(d["overall"]),
        would_wear=bool(d["would_wear"]),
        labeller=d.get("labeller", "unknown"),
        source=d.get("source", "expert"),
        note=d.get("note", ""),
        engine_version=int(d.get("engine_version", 1)),
        created_at=d.get("created_at", ""),
    )


@dataclass(frozen=True)
class CantJudge:
    """A rater looked at an outfit and could not judge it.

    Kept apart from `Label` on purpose, in its own file. It carries no score,
    so nothing that reads labels (the scoreboard, the fitter, agreement) can
    mistake it for one. It is still worth keeping: an outfit several stylists
    could not judge says the card is missing something (a photo, a fabric, a
    drape), and a rater who skips everything of one kind says the same about
    the rubric. `reason` is whatever they typed in the note, often nothing.
    """

    id: str
    outfit: dict[str, object]      # slot -> garment id, plus "tucked": bool
    context: str
    labeller: str
    reason: str = ""
    engine_version: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def outfit_key(self) -> str:
        """Same identity as `Label.outfit_key`, so the two can be joined."""
        return (outfit_slot_key(self.outfit.get)
                + f"|tucked={bool(self.outfit.get('tucked'))}|{self.context}")

    def to_dict(self) -> dict:
        return asdict(self)


def cant_judge_from_dict(d: dict) -> CantJudge:
    return CantJudge(
        id=d["id"],
        outfit=d["outfit"],
        context=d["context"],
        labeller=d.get("labeller") or "unknown",
        reason=d.get("reason") or "",
        engine_version=int(d.get("engine_version", 1)),
        created_at=d.get("created_at", ""),
    )
