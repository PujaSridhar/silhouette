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
    "load_wardrobe", "load_style_cv",
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
    category: str                 # top | bottom | dress | outer | shoe
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


@dataclass(frozen=True)
class Outfit:
    """A concrete assembly. `tucked` is a styling choice, not a garment property."""

    top: Garment | None = None
    bottom: Garment | None = None
    dress: Garment | None = None
    outer: Garment | None = None
    shoe: Garment | None = None
    tucked: bool = False

    @property
    def items(self) -> list[Garment]:
        return [g for g in (self.top, self.bottom, self.dress, self.outer, self.shoe) if g]

    @property
    def key(self) -> tuple:
        return tuple(g.id for g in self.items) + (self.tucked,)

    def label(self) -> str:
        bits = [
            f"{g.name} (tucked)" if (self.tucked and g is self.top) else g.name
            for g in self.items
        ]
        return " + ".join(bits)


def load_wardrobe(path: str | Path) -> list[Garment]:
    out: list[Garment] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
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
