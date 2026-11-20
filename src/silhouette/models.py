"""Garments, frames, Style CVs, contexts and outfits.

Everything the rules engine reads is an explicit, named field. No magic, no
embeddings, nothing inferred. That is deliberate: in phase 0 the whole point is
that a human can read a row of the CSV and predict what the engine will say.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = [
    "Garment", "Frame", "Context", "StyleCV", "Outfit",
    "SLOTS", "outfit_slot_key", "load_wardrobe", "load_style_cv",
]


def _f(v: str | float | None, default: float | None = None) -> float | None:
    if v is None or v == "":
        return default
    return float(v)


def _s(v: str | None, default: str | None = None) -> str | None:
    if v is None or v == "":
        return default
    return str(v)


@dataclass(frozen=True)
class Garment:
    id: str
    name: str
    category: str                 # top | bottom | dress | outer | shoe | drape
    subcategory: str = ""
    rise: str | None = None       # low | mid | high | super_high   (bottoms)
    length: str | None = None     # crop | waist | hip | tunic | knee | midi | maxi | ankle | full
    fit: str | None = None
    volume: float = 1.0           # garment width / body width at its own zone
    volume_lower: float | None = None   # dresses only: the skirt zone
    structure: float = 0.0        # 0 = limp knit, 1 = built shoulder
    tuckable: bool = False
    neckline: str | None = None
    sleeve: str | None = None
    pattern: str = "solid"
    fabric_weight: str | None = None
    drape: str | None = None
    formality: float = 0.5        # 0 loungewear .. 1 formal/festive
    clo: float = 0.0              # thermal insulation, roughly
    heel_cm: float = 0.0          # shoes only
    waist_seam: str | None = None # dresses: empire | natural | drop | none
    color_hex: str = "#808080"
    availability: str = "clean"

    @property
    def lower_vol(self) -> float:
        return self.volume_lower if self.volume_lower is not None else self.volume


@dataclass(frozen=True)
class Frame:
    landmarks: dict[str, float]   # fractions of barefoot height, from crown
    widths: dict[str, float]      # fractions of barefoot height
    confidence: float = 1.0
    source: str = "hand_measured"

    def y(self, name: str) -> float:
        return self.landmarks[name]

    @property
    def true_leg_ratio(self) -> float:
        """Barefoot, anatomical: hip to sole as a fraction of height."""
        return 1.0 - self.y("hip")

    @property
    def hip_to_shoulder(self) -> float:
        return self.widths["hip"] / self.widths["shoulder"]

    @property
    def shoulder_to_hip(self) -> float:
        return self.widths["shoulder"] / self.widths["hip"]


@dataclass(frozen=True)
class Context:
    name: str
    formality_floor: float
    formality_ceiling: float
    clo_min: float
    clo_max: float


@dataclass(frozen=True)
class StyleCV:
    owner: str
    height_cm: float
    frame: Frame
    goals: list[str]
    contexts: dict[str, Context]
    version: int = 1
    palette: dict = field(default_factory=dict)

    def context(self, name: str) -> Context:
        if name not in self.contexts:
            raise KeyError(f"unknown context {name!r}; have {sorted(self.contexts)}")
        return self.contexts[name]


# Every garment slot, in the order items, labels and keys list them. `drape`
# (a dupatta, a stole) comes last so an outfit without one reads, lists and
# keys exactly as it did before the slot existed.
SLOTS = ("top", "bottom", "dress", "outer", "shoe", "drape")


def outfit_slot_key(get) -> str:
    """The garment part of an outfit's stable identity, from `get(slot) -> id`.

    Labels, the sampler and the report all key outfits this way, so it lives
    in one place. The drape is appended only when there is one, which keeps
    every key written before drapes existed valid.
    """
    ids = [str(get(s) or "-") for s in SLOTS[:5]]
    drape = get("drape")
    if drape:
        ids.append(f"drape={drape}")
    return "|".join(ids)


@dataclass(frozen=True)
class Outfit:
    """A concrete assembly. `tucked` is a styling choice, not a garment property.

    `drape` is an optional loose layer worn over everything else, a dupatta or
    a stole. It adds to formality, warmth and pattern like any garment, but not
    to the silhouette: it does not move the break and its volume is not read,
    because how it is worn (open, pinned, over one shoulder) changes its width
    more than the cloth does. Its effect is carried by its own rules.
    """

    top: Garment | None = None
    bottom: Garment | None = None
    dress: Garment | None = None
    outer: Garment | None = None
    shoe: Garment | None = None
    tucked: bool = False
    drape: Garment | None = None

    @property
    def items(self) -> list[Garment]:
        return [g for g in (self.top, self.bottom, self.dress, self.outer, self.shoe,
                            self.drape) if g]

    @property
    def key(self) -> tuple:
        return tuple(g.id for g in self.items) + (self.tucked,)

    def label(self) -> str:
        bits = [
            f"{g.name} (tucked)" if (self.tucked and g is self.top) else g.name
            for g in self.items
        ]
        return " + ".join(bits)


def load_wardrobe(path: str | Path, *, require_volume: bool = True) -> list[Garment]:
    """Read a wardrobe CSV.

    An empty `volume` used to become 1.0 here, which is not "unjudged" but a
    real width the engine scores against: a half-rated file would quietly
    produce confident, wrong answers. So a blank volume is an error by
    default. `require_volume=False` is for the tools whose job is to fill the
    column in, and for tests that only care about the other fields.
    """
    out: list[Garment] = []
    unrated: list[str] = []
    with open(path, newline="") as fh:
        for lineno, row in enumerate(csv.DictReader(fh), start=2):
            if require_volume and not str(row.get("volume") or "").strip():
                unrated.append(f"line {lineno}: {row.get('id') or '<no id>'}")
            out.append(
                Garment(
                    id=row["id"],
                    name=row["name"],
                    category=row["category"],
                    subcategory=_s(row.get("subcategory"), "") or "",
                    rise=_s(row.get("rise")),
                    length=_s(row.get("length")),
                    fit=_s(row.get("fit")),
                    volume=_f(row.get("volume"), 1.0),
                    volume_lower=_f(row.get("volume_lower")),
                    structure=_f(row.get("structure"), 0.0),
                    tuckable=str(row.get("tuckable", "")).strip().lower() in ("yes", "true", "1"),
                    neckline=_s(row.get("neckline")),
                    sleeve=_s(row.get("sleeve")),
                    pattern=_s(row.get("pattern"), "solid") or "solid",
                    fabric_weight=_s(row.get("fabric_weight")),
                    drape=_s(row.get("drape")),
                    formality=_f(row.get("formality"), 0.5),
                    clo=_f(row.get("clo"), 0.0),
                    heel_cm=_f(row.get("heel_cm"), 0.0),
                    waist_seam=_s(row.get("waist_seam")),
                    color_hex=_s(row.get("color_hex"), "#808080") or "#808080",
                    availability=_s(row.get("availability"), "clean") or "clean",
                )
            )
    if unrated:
        shown = "\n  ".join(unrated[:5])
        more = f"\n  ... and {len(unrated) - 5} more" if len(unrated) > 5 else ""
        raise ValueError(
            f"{path}: {len(unrated)} garment(s) have no volume, so this wardrobe "
            f"cannot be scored:\n  {shown}{more}\n"
            "Rate them with `python tools/rate_wardrobe.py`, or pass "
            "require_volume=False if you only need the other columns."
        )
    ids = [g.id for g in out]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate garment ids: {sorted(dupes)}")
    return out


def load_style_cv(path: str | Path) -> StyleCV:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    fr = raw["frame"]
    frame = Frame(
        landmarks={k: float(v) for k, v in fr["landmarks"].items()},
        widths={k: float(v) for k, v in fr["widths"].items()},
        confidence=float(fr.get("confidence", 1.0)),
        source=str(fr.get("source", "hand_measured")),
    )
    contexts = {
        c["name"]: Context(
            name=c["name"],
            formality_floor=float(c["formality_floor"]),
            formality_ceiling=float(c["formality_ceiling"]),
            clo_min=float(c["clo_min"]),
            clo_max=float(c["clo_max"]),
        )
        for c in raw["contexts"]
    }
    return StyleCV(
        owner=raw["owner"],
        height_cm=float(raw["height_cm"]),
        frame=frame,
        goals=list(raw.get("goals", [])),
        contexts=contexts,
        version=int(raw.get("version", 1)),
        palette=raw.get("palette", {}) or {},
    )
