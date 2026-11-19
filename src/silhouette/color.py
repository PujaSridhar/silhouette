"""sRGB -> CIELAB conversion and the CIEDE2000 colour-difference metric.

Why bother with CIEDE2000 instead of plain RGB distance: the whole point of
`contrast_at_break` is *perceived* contrast. RGB distance says navy and black
are far apart and that mustard and olive are close; both are wrong to the eye.
CIEDE2000 is the current CIE recommendation for small-to-moderate perceptual
differences and is what the styling rules are calibrated against.

Rough reading of the returned dE values:
    0-2    indistinguishable to most people
    2-10   clearly different, still reads as one colour family
    10-25  obvious contrast
    25+    hard break
"""

from __future__ import annotations

import math

__all__ = ["hex_to_rgb", "srgb_to_lab", "hex_to_lab", "delta_e_2000", "contrast_hex"]


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    """'#1C1C1C' -> (0.109, 0.109, 0.109) in 0..1 sRGB."""
    s = value.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"not a hex colour: {value!r}")
    return tuple(int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _linearise(c: float) -> float:
    """Undo the sRGB transfer function."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# sRGB (D65) -> XYZ, and the D65 white point.
_M = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_WHITE = (0.95047, 1.00000, 1.08883)


def srgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = (_linearise(c) for c in rgb)
    xyz = tuple(m[0] * r + m[1] * g + m[2] * b for m in _M)

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29

    fx, fy, fz = (f(v / w) for v, w in zip(xyz, _WHITE))
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def hex_to_lab(value: str) -> tuple[float, float, float]:
    return srgb_to_lab(hex_to_rgb(value))


def delta_e_2000(
    lab1: tuple[float, float, float],
    lab2: tuple[float, float, float],
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> float:
    """CIEDE2000 difference between two CIELAB colours.

    Implementation follows Sharma, Wu & Dalal (2005), the reference formulation
    with the corrected hue-rotation term.
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cbar = (C1 + C2) / 2.0

    # a-axis correction for the blue region
    Cbar7 = Cbar**7
    G = 0.5 * (1 - math.sqrt(Cbar7 / (Cbar7 + 25.0**7))) if Cbar > 0 else 0.0
    a1p, a2p = (1 + G) * a1, (1 + G) * a2

    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0 if (a1p or b1) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0 if (a2p or b2) else 0.0

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        hbarp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbarp = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        hbarp = (h1p + h2p + 360) / 2.0
    else:
        hbarp = (h1p + h2p - 360) / 2.0

    T = (
        1
        - 0.17 * math.cos(math.radians(hbarp - 30))
        + 0.24 * math.cos(math.radians(2 * hbarp))
        + 0.32 * math.cos(math.radians(3 * hbarp + 6))
        - 0.20 * math.cos(math.radians(4 * hbarp - 63))
    )

    dTheta = 30 * math.exp(-(((hbarp - 275) / 25) ** 2))
    Cbarp7 = Cbarp**7
    Rc = 2 * math.sqrt(Cbarp7 / (Cbarp7 + 25.0**7)) if Cbarp > 0 else 0.0
    Rt = -Rc * math.sin(math.radians(2 * dTheta))

    sq = (Lbarp - 50) ** 2
    Sl = 1 + (0.015 * sq) / math.sqrt(20 + sq)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T

    return math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )


def contrast_hex(a: str, b: str) -> float:
    """Perceived contrast between two hex colours, as a CIEDE2000 dE."""
    return delta_e_2000(hex_to_lab(a), hex_to_lab(b))
