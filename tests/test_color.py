"""Colour conversion and CIEDE2000, checked against known values."""

from __future__ import annotations

import pytest

from silhouette.color import contrast_hex, delta_e_2000, hex_to_lab, hex_to_rgb


def test_hex_parsing_short_and_long_form():
    assert hex_to_rgb("#fff") == pytest.approx(hex_to_rgb("#ffffff"))
    with pytest.raises(ValueError):
        hex_to_rgb("#12345")


@pytest.mark.parametrize(
    "hexval,expected_L",
    [("#000000", 0.0), ("#FFFFFF", 100.0), ("#808080", 53.59)],
)
def test_lightness_anchors(hexval, expected_L):
    """L* has fixed, checkable values at black, white and mid grey."""
    assert hex_to_lab(hexval)[0] == pytest.approx(expected_L, abs=0.01)


def test_neutral_greys_have_no_chroma():
    """A grey must carry no colour cast.

    Tolerance is 1e-4, not 0: the D65 white point is specified to five decimal
    places, so a perfectly neutral sRGB grey cannot round-trip to exactly
    a*=b*=0. The residual is around 1e-5, which is roughly six orders of
    magnitude below a just-noticeable difference.
    """
    _L, a, b = hex_to_lab("#808080")
    assert a == pytest.approx(0.0, abs=1e-4)
    assert b == pytest.approx(0.0, abs=1e-4)


def test_black_to_white_is_a_hundred():
    assert contrast_hex("#000000", "#FFFFFF") == pytest.approx(100.0, abs=0.01)


def test_identical_colours_are_zero():
    assert contrast_hex("#9C4A2E", "#9C4A2E") == 0.0


def test_symmetry():
    a, b = "#1C1C1C", "#7A3B5C"
    assert contrast_hex(a, b) == pytest.approx(contrast_hex(b, a), abs=1e-9)


def test_near_blacks_are_perceptually_close():
    """The whole reason for CIEDE2000: #1C1C1C and #1A1A1A are the same colour
    to the eye, and the metric must say so."""
    assert contrast_hex("#1C1C1C", "#1A1A1A") < 2.0


def test_zero_chroma_pair_does_not_divide_by_zero():
    assert delta_e_2000((50.0, 0.0, 0.0), (50.0, 0.0, 0.0)) == 0.0
