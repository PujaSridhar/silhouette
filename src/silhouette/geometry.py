"""Turn garment annotations plus a frame into the numbers the rules reason over.

The one definition that matters most, because every other number leans on it:

    lift = apparent_leg_ratio - true_leg_ratio(barefoot)

`true_leg_ratio` is anatomical and fixed: hip to sole over barefoot height.
`apparent_leg_ratio` is what the eye reads: the visual break to the floor, over
total height *including heel*. So both levers a stylist actually has — moving
the break up, and adding height under the heel — increase `lift`, and nothing
else does. That is the quantity being bought when someone chooses a high rise.

A worked example on the reference frame (hip at 0.50, so true_leg = 0.50):

    barefoot, high rise, waistband at 0.40   -> apparent 0.60, lift +0.10
    the same trousers, 7 cm heel             -> apparent 0.617, lift +0.117
    mid rise, waistband at 0.45              -> apparent 0.55, lift +0.05
    high rise but an untucked hip-length tee -> break drops to 0.525,
                                                apparent 0.475, lift -0.025

That last line is the useful one: an untucked hip-length top over a high rise
does not merely waste the rise, it reads *shorter* than the anatomy. The engine
says so, with a number, which is exactly what a stylist says with their hands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .color import contrast_hex
from .models import Frame, Garment, Outfit

__all__ = ["Features", "waistband_y", "hem_y", "dress_break_y", "compute"]

# Where a waistband sits, as a function of the wearer's own landmarks.
# Expressed relative to the frame so it personalises automatically.
_RISE = {
    "super_high": lambda f: f.y("waist") - 0.015,
    "high":       lambda f: f.y("waist"),
    "mid":        lambda f: f.y("waist") + 0.50 * (f.y("hip") - f.y("waist")),
    "low":        lambda f: f.y("hip") - 0.010,
}

# Where a hem falls, likewise frame-relative.
_LENGTH = {
    "crop":  lambda f: f.y("waist") - 0.055,
    "waist": lambda f: f.y("waist") + 0.005,
    "hip":   lambda f: f.y("hip") + 0.025,
    "tunic": lambda f: f.y("hip") + 0.45 * (f.y("knee") - f.y("hip")),
    "knee":  lambda f: f.y("knee"),
    "midi":  lambda f: f.y("knee") + 0.55 * (f.y("ankle") - f.y("knee")),
    "maxi":  lambda f: f.y("ankle") - 0.010,
    "ankle": lambda f: f.y("ankle"),
    "full":  lambda f: f.y("ankle"),
}

# Where a dress's own horizontal division sits. `none` means a draped or
# column garment with no break at all - a saree, a shift.
_SEAM = {
    "empire":  lambda f: f.y("bust") + 0.40 * (f.y("waist") - f.y("bust")),
    "natural": lambda f: f.y("waist"),
    "drop":    lambda f: f.y("hip"),
    "none":    lambda f: None,
}

_TUCK_TOLERANCE = 0.010   # within this, a hem reads as "at" the waistband


def waistband_y(g: Garment, frame: Frame) -> float | None:
    fn = _RISE.get(g.rise or "")
    return fn(frame) if fn else None


def hem_y(g: Garment, frame: Frame) -> float | None:
    fn = _LENGTH.get(g.length or "")
    return fn(frame) if fn else None


def dress_break_y(g: Garment, frame: Frame) -> float | None:
    fn = _SEAM.get(g.waist_seam or "none")
    return fn(frame) if fn else None


@dataclass(frozen=True)
class Features:
    # --- the proportion story ---
    break_ratio: float | None       # visual break, as a fraction of total height
    apparent_leg: float | None      # break -> floor, over total height
    true_leg: float                 # hip -> sole, barefoot. anatomy.
    lift: float                     # apparent_leg - true_leg. the illusion.
    hem_drop: float                 # how far a hem falls below the waistband
    break_source: str               # waistband | top_hem | dress_seam | none
    is_column: bool                 # no horizontal break at all

    # --- the balance story ---
    upper_volume: float
    lower_volume: float
    asymmetry: float                # upper - lower. negative = more below.
    upper_structure: float          # best structure among the upper layers

    # --- the rest ---
    contrast_at_break: float
    min_formality: float
    max_formality: float
    total_clo: float
    pattern_count: int
    heel_cm: float
    height_total: float             # 1.0 + heel, in height-fractions

    def facts(self) -> dict:
        """Flat dict for rendering explanation templates."""
        return asdict(self)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compute(outfit: Outfit, frame: Frame, height_cm: float) -> Features:
    heel = outfit.shoe.heel_cm if outfit.shoe else 0.0
    total = 1.0 + (heel / height_cm)

    break_y: float | None
    hem_drop = 0.0
    source = "none"

    if outfit.dress is not None:
        break_y = dress_break_y(outfit.dress, frame)
        source = "dress_seam" if break_y is not None else "none"
        upper_items = [outfit.dress] + ([outfit.outer] if outfit.outer else [])
        upper_vol = _mean([outfit.dress.volume] + ([outfit.outer.volume] if outfit.outer else []))
        lower_vol = outfit.dress.lower_vol
    else:
        wb = waistband_y(outfit.bottom, frame) if outfit.bottom else None
        raw_hem = hem_y(outfit.top, frame) if outfit.top else None

        # Tucking is a choice the solver makes, not a garment property.
        if outfit.tucked and outfit.top is not None and outfit.top.tuckable and wb is not None:
            hem = wb
        else:
            hem = raw_hem

        if wb is None:
            break_y, source = (hem, "top_hem") if hem is not None else (None, "none")
        elif hem is None:
            break_y, source = wb, "waistband"
        elif hem <= wb + _TUCK_TOLERANCE:
            # the top ends at or above the waistband, so the waistband is the division
            break_y, source = wb, "waistband"
        else:
            # the top covers the waistband; its own hem becomes the division
            break_y, source = hem, "top_hem"
            hem_drop = hem - wb

        upper_items = ([outfit.top] if outfit.top else []) + ([outfit.outer] if outfit.outer else [])
        upper_vol = _mean([g.volume for g in upper_items])
        lower_vol = outfit.bottom.volume if outfit.bottom else 1.0

    if break_y is None:
        break_ratio = None
        apparent = None
        lift = 0.0
    else:
        break_ratio = break_y / total
        apparent = (total - break_y) / total
        lift = apparent - frame.true_leg_ratio

    # Perceived contrast across the break. For a dress there is one fabric
    # meeting itself, so there is no contrast to score.
    if outfit.dress is not None or not (outfit.top and outfit.bottom):
        contrast = 0.0
    else:
        contrast = contrast_hex(outfit.top.color_hex, outfit.bottom.color_hex)

    items = outfit.items
    return Features(
        break_ratio=break_ratio,
        apparent_leg=apparent,
        true_leg=frame.true_leg_ratio,
        lift=lift,
        hem_drop=hem_drop,
        break_source=source,
        is_column=break_y is None,
        upper_volume=upper_vol,
        lower_volume=lower_vol,
        asymmetry=upper_vol - lower_vol,
        upper_structure=max([g.structure for g in upper_items], default=0.0),
        contrast_at_break=contrast,
        min_formality=min((g.formality for g in items), default=0.5),
        max_formality=max((g.formality for g in items), default=0.5),
        total_clo=sum(g.clo for g in items),
        pattern_count=sum(1 for g in items if g.pattern != "solid"),
        heel_cm=heel,
        height_total=total,
    )
