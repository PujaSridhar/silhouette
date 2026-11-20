"""Volume as a measurement: garment width over body width, read off a mask.

`models.Garment.volume` is defined as garment width / body width at the
garment's own zone. Phase 0 judged that number by eye. Here it is measured:

    volume = median over zone rows of  garment_mask_width(row) / body_width(row)

The garment width comes from a segmentation mask. The body width does NOT come
from the photo - the garment covers the body at exactly the rows we measure, so
the body is not visible there. It comes from the Style CV instead, which
already stores body widths as fractions of barefoot height. All a photo has to
supply is the wearer's height in pixels, read from crown and sole:

    barefoot_px  = (sole_y - crown_y) * height_cm / (height_cm + heel_cm)
    row(frac)    = crown_y + frac * barefoot_px
    body_px(row) = width_frac(row) * barefoot_px

So this works on **worn photos only**: the wearer standing, full length, facing
the camera. A garment on a bed has no crown or sole and no body to be wider
than. A flat-lay needs three things this module does not have: a scale object
in frame (an A4 sheet, a ruler) to turn pixels into centimetres, the wearer's
body widths in centimetres (height_cm times the frame widths gives those), and
a model of how a flat width becomes a worn width, which differs by fabric and
drape. The last one is the hard part and the reason flat-lays are not done.

Every measurement carries warnings rather than failing silently: a mask that
touches the image edge is probably cut off, a zone with too few mask rows is
probably occluded, and a short body in pixels means a low-resolution read.

Zone definitions are named, because which rows and which reducer match the
judged numbers is an empirical question. The eval runs every definition and
reports how each tracks the hand-judged column; see docs/w1-segmentation.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import Frame

__all__ = [
    "BodyScale", "ZoneSpec", "DEFINITIONS", "Measurement",
    "body_width_frac", "row_widths", "measure", "measure_garment",
    "person_extent_from_mask",
]

# Body widths are only given at these landmarks. Below the hip the frame has
# per-leg widths (thigh, calf) with no landmark height and an unclear
# both-legs-or-one meaning, so lower-body zones use the hip width throughout:
# a bottom's volume is read against the hip, which is what the balance rules
# compare it with.
_UPPER = ("shoulder", "bust", "waist", "hip")


def body_width_frac(frame: Frame, y_frac: float) -> float:
    """Front body width at a height (fraction of barefoot height from crown),
    interpolated between the shoulder, bust, waist and hip landmarks and held
    constant outside them."""
    ys = [frame.y(k) for k in _UPPER]
    ws = [frame.widths[k] for k in _UPPER]
    return float(np.interp(y_frac, ys, ws))


@dataclass(frozen=True)
class BodyScale:
    """Where the wearer is in the image, in pixels."""

    crown_y: float
    sole_y: float
    height_cm: float
    heel_cm: float = 0.0

    @property
    def barefoot_px(self) -> float:
        return (self.sole_y - self.crown_y) * self.height_cm / (self.height_cm + self.heel_cm)

    def row(self, y_frac: float) -> float:
        return self.crown_y + y_frac * self.barefoot_px

    def frac(self, row: float) -> float:
        return (row - self.crown_y) / self.barefoot_px


@dataclass(frozen=True)
class ZoneSpec:
    """Which rows of a garment count, what they are compared against, and how
    the per-row ratios become one number."""

    start: str                 # landmark name
    end: str                   # landmark name, or "hem" for the mask's lowest row
    reference: str = "profile" # "profile" interpolates; a landmark name holds one width
    reducer: str = "median"    # median | p90 | max
    subtract_sleeves: bool = False
    cap: str | None = None     # landmark the end may not pass (e.g. ankle)


def _top(end: str = "waist") -> ZoneSpec:
    return ZoneSpec("bust", end, "profile", "median", subtract_sleeves=True)


# The default reads tops through the bust-to-waist band (where a fitted top
# and a boxy one differ) and bottoms from the hip to the hem at their 90th
# percentile width (so a wide leg is read at its width, not averaged with the
# narrow top of the thigh). `bottom_median` is the obvious alternative.
DEFINITIONS: dict[str, dict[str, ZoneSpec]] = {
    "default": {
        "top": _top(),
        "outer": _top(),
        "bottom": ZoneSpec("hip", "hem", "hip", "p90", cap="ankle"),
        "dress": _top(),
        "dress_lower": ZoneSpec("hip", "hem", "hip", "p90", cap="ankle"),
    },
    "bottom_median": {
        "top": _top(),
        "outer": _top(),
        "bottom": ZoneSpec("hip", "hem", "hip", "median", cap="ankle"),
        "dress": _top(),
        "dress_lower": ZoneSpec("hip", "hem", "hip", "median", cap="ankle"),
    },
}

_MIN_ROW_COVERAGE = 0.6
_MIN_BODY_PX = 400


def row_widths(mask: np.ndarray) -> np.ndarray:
    """Outer extent of the mask in each row, in pixels (0 for empty rows).

    Extent, not pixel count: the gap between two trouser legs is part of how
    wide the garment reads."""
    any_ = mask.any(axis=1)
    w = np.zeros(mask.shape[0], dtype=float)
    if not any_.any():
        return w
    left = np.argmax(mask, axis=1)
    right = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    w[any_] = (right - left + 1)[any_]
    return w


@dataclass(frozen=True)
class Measurement:
    volume: float | None
    rows_used: int
    rows_in_zone: int
    ratios: np.ndarray = field(repr=False)
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.volume is not None


def _reduce(xs: np.ndarray, how: str) -> float:
    if how == "median":
        return float(np.median(xs))
    if how == "p90":
        return float(np.percentile(xs, 90))
    if how == "max":
        return float(xs.max())
    raise ValueError(f"unknown reducer {how!r}")


def measure(
    mask: np.ndarray,
    scale: BodyScale,
    frame: Frame,
    zone: ZoneSpec,
    sleeves: np.ndarray | None = None,
) -> Measurement:
    """Measure one garment's volume from its mask."""
    mask = np.asarray(mask, dtype=bool)
    warn: list[str] = []
    if zone.subtract_sleeves:
        if sleeves is None:
            warn.append("no sleeve mask: sleeves at the sides would read as width")
        else:
            mask = mask & ~np.asarray(sleeves, dtype=bool)

    h, w = mask.shape
    if mask[:, 0].any() or mask[:, -1].any() or mask[0].any() or mask[-1].any():
        warn.append("mask touches the image edge: the garment may be cut off")
    if scale.barefoot_px < _MIN_BODY_PX:
        warn.append(f"wearer is only {scale.barefoot_px:.0f}px tall: low-resolution read")

    widths = row_widths(mask)
    r0 = int(round(scale.row(frame.y(zone.start))))
    if zone.end == "hem":
        rows = np.nonzero(widths)[0]
        r1 = int(rows.max()) if rows.size else r0
    else:
        r1 = int(round(scale.row(frame.y(zone.end))))
    if zone.cap:
        r1 = min(r1, int(round(scale.row(frame.y(zone.cap)))))
    r0, r1 = max(r0, 0), min(r1, h - 1)

    if r1 <= r0:
        return Measurement(None, 0, 0, np.array([]),
                           tuple(warn + ["the garment has no rows inside its zone"]))

    ys = np.arange(r0, r1 + 1)
    gw = widths[ys]
    if zone.reference == "profile":
        body = np.array([body_width_frac(frame, scale.frac(y)) for y in ys]) * scale.barefoot_px
    else:
        body = np.full(ys.shape, frame.widths[zone.reference] * scale.barefoot_px)

    present = gw > 0
    ratios = gw[present] / body[present]
    coverage = present.mean()
    if coverage < _MIN_ROW_COVERAGE:
        warn.append(f"mask covers only {coverage:.0%} of its zone: occluded or mis-scaled")
        return Measurement(None, int(present.sum()), len(ys), ratios, tuple(warn))
    return Measurement(_reduce(ratios, zone.reducer), int(present.sum()), len(ys),
                       ratios, tuple(warn))


def measure_garment(
    category: str,
    mask: np.ndarray,
    scale: BodyScale,
    frame: Frame,
    definition: str = "default",
    sleeves: np.ndarray | None = None,
) -> dict[str, Measurement]:
    """All the volumes the engine stores for a garment of this category:
    `volume`, plus `volume_lower` for a dress. Shoes have none."""
    spec = DEFINITIONS[definition]
    if category == "shoe":
        return {}
    if category == "dress":
        return {
            "volume": measure(mask, scale, frame, spec["dress"], sleeves),
            "volume_lower": measure(mask, scale, frame, spec["dress_lower"]),
        }
    if category not in spec:
        raise ValueError(f"no zone for category {category!r}")
    return {"volume": measure(mask, scale, frame, spec[category], sleeves)}


def person_extent_from_mask(person: np.ndarray) -> tuple[float, float]:
    """Crown and sole rows from a whole-person mask (hair and shoes included).

    The sole row includes the shoe, which is why BodyScale takes heel_cm. The
    crown row includes hair, which reads a little tall; for a volume that is a
    small, consistent bias, but tap the crown by hand if hair is up."""
    rows = np.nonzero(np.asarray(person, dtype=bool).any(axis=1))[0]
    if rows.size == 0:
        raise ValueError("empty person mask")
    return float(rows.min()), float(rows.max() + 1)
