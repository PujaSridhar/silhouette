"""The rating tool, and the loader rule that makes it necessary.

An empty volume used to load as 1.0. That is not a missing value, it is a
real width the engine scores against, so a half-rated wardrobe produced
confident wrong answers with nothing on screen to say so. These tests pin
both halves: the loader refuses, and the tool fills the column in.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from silhouette.models import load_wardrobe

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["id", "name", "category", "subcategory", "rise", "length", "fit", "volume",
           "volume_lower", "structure", "tuckable", "neckline", "sleeve", "pattern",
           "fabric_weight", "drape", "formality", "clo", "heel_cm", "waist_seam",
           "color_hex", "availability"]


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location(
        "rate_wardrobe", ROOT / "tools" / "rate_wardrobe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def _row(gid, **kw):
    base = dict(id=gid, name=gid, category="top", subcategory="tee",
                color_hex="#111111", availability="clean")
    base.update(kw)
    return base


# --- the loader -------------------------------------------------------------

def test_an_empty_volume_is_an_error_not_a_silent_1(tmp_path):
    p = tmp_path / "w.csv"
    _write(p, [_row("top_rated", volume="1.2"), _row("top_unrated")])
    with pytest.raises(ValueError) as e:
        load_wardrobe(p)
    assert "top_unrated" in str(e.value)
    assert "top_rated" not in str(e.value)
    assert "rate_wardrobe" in str(e.value)


def test_the_error_names_the_csv_line(tmp_path):
    p = tmp_path / "w.csv"
    _write(p, [_row("a", volume="1.0"), _row("b", volume="1.0"), _row("c")])
    with pytest.raises(ValueError, match="line 4"):
        load_wardrobe(p)


def test_require_volume_false_still_loads_and_does_not_invent_a_width(tmp_path):
    p = tmp_path / "w.csv"
    _write(p, [_row("x")])
    got = load_wardrobe(p, require_volume=False)
    assert len(got) == 1
    assert got[0].volume == 1.0, "the old default still applies when opted out of"


def test_the_shipped_reference_wardrobe_is_fully_rated():
    assert len(load_wardrobe(ROOT / "data" / "wardrobe.csv")) == 30


# --- the tool ---------------------------------------------------------------

def test_saving_writes_the_answer_and_leaves_every_other_column_alone(tmp_path, tool):
    p = tmp_path / "w.csv"
    _write(p, [_row("a", fit="fitted", color_hex="#ABCDEF"), _row("b")])
    tool.REAL = p
    out = tool.save("a", {"volume": "0.85", "structure": "0.1", "formality": "0.3"})
    assert out == {"ok": True, "id": "a", "rated": 1, "total": 2}

    rows = {r["id"]: r for r in csv.DictReader(open(p))}
    assert rows["a"]["volume"] == "0.85"
    assert rows["a"]["fit"] == "fitted" and rows["a"]["color_hex"] == "#ABCDEF"
    assert rows["b"]["volume"] == "", "a skipped piece stays in the queue"


def test_a_skipped_piece_can_be_rated_later(tmp_path, tool):
    p = tmp_path / "w.csv"
    _write(p, [_row("a"), _row("b")])
    tool.REAL = p
    tool.save("b", {"volume": "1.4"})            # out of order, a still blank
    tool.save("a", {"volume": "0.9"})
    rows = {r["id"]: r["volume"] for r in csv.DictReader(open(p))}
    assert rows == {"a": "0.9", "b": "1.4"}


def test_saving_refuses_a_column_that_is_not_hers_to_set(tmp_path, tool):
    p = tmp_path / "w.csv"
    _write(p, [_row("a")])
    tool.REAL = p
    with pytest.raises(ValueError, match="not a ratable column"):
        tool.save("a", {"volume": "1.0", "color_hex": "#000000"})
    assert next(csv.DictReader(open(p)))["color_hex"] == "#111111"


def test_saving_refuses_a_value_that_is_not_a_number(tmp_path, tool):
    p = tmp_path / "w.csv"
    _write(p, [_row("a")])
    tool.REAL = p
    with pytest.raises(ValueError, match="must be a number"):
        tool.save("a", {"volume": "quite wide"})
    assert next(csv.DictReader(open(p)))["volume"] == ""


def test_saving_an_unknown_id_is_an_error(tmp_path, tool):
    p = tmp_path / "w.csv"
    _write(p, [_row("a")])
    tool.REAL = p
    with pytest.raises(KeyError):
        tool.save("nope", {"volume": "1.0"})


def test_anchors_bracket_the_scale_rather_than_taking_the_first_three(tool):
    got = tool.anchors("bottom")
    assert len(got) == 3
    vols = [float(a["volume"]) for a in got]
    assert vols == sorted(vols)
    allv = [float(r["volume"]) for r in csv.DictReader(open(tool.REFERENCE))
            if r["category"] == "bottom" and r["volume"]]
    assert vols[0] == min(allv) and vols[-1] == max(allv)


def test_a_category_with_nothing_judged_yields_no_anchors(tool):
    assert tool.anchors("drape") == []


def test_every_unrated_piece_is_offered(tool):
    ids = {i["id"] for i in tool.items()}
    real = {r["id"] for r in csv.DictReader(open(ROOT / "data" / "wardrobe_real.csv"))}
    assert ids == real


def test_only_draped_pieces_ask_for_a_lower_volume(tool):
    for it in tool.items():
        assert it["two_part"] == (it["category"] == "dress"), it["id"]


# --- the word scales --------------------------------------------------------
#
# The numbers are hers and the words are the interface to them, so both are
# pinned here: a silent edit to either changes what every rating means, and
# the volumes already recorded would no longer be comparable with later ones.

def test_the_default_volume_scale(tool):
    assert tool.VOLUME_SCALES["default"] == [
        ("Fitted", 0.85), ("Close", 1.0), ("Relaxed", 1.2),
        ("Loose", 1.4), ("Oversized", 1.65)]


def test_bottoms_get_their_own_volume_scale(tool):
    assert tool.VOLUME_SCALES["bottom"] == [
        ("Skinny", 0.85), ("Straight", 1.1), ("Relaxed", 1.4),
        ("Wide", 1.85), ("Very wide", 2.1)]


def test_the_skirt_structure_and_formality_scales(tool):
    assert tool.VOLUME_LOWER_SCALE == [("Narrow", 1.1), ("A-line", 1.4), ("Full", 1.9)]
    assert tool.STRUCTURE_SCALE == [("Soft", 0.15), ("Medium", 0.4), ("Stiff", 0.75)]
    assert tool.FORMALITY_SCALE == [("Casual", 0.25), ("Smart", 0.55), ("Dressy", 0.85)]


@pytest.mark.parametrize("category", ["top", "outer", "dress", "shoe", "drape"])
def test_everything_that_is_not_a_bottom_uses_the_default_scale(tool, category):
    assert tool.volume_scale(category) is tool.VOLUME_SCALES["default"]


def test_a_bottom_uses_the_bottom_scale(tool):
    assert tool.volume_scale("bottom") is tool.VOLUME_SCALES["bottom"]


def test_every_piece_is_offered_a_full_set_of_buttons(tool):
    for it in tool.items():
        assert len(it["volume_steps"]) == 5, it["id"]
        assert len(it["structure_steps"]) == 3, it["id"]
        assert len(it["formality_steps"]) == 3, it["id"]
        assert len(it["volume_lower_steps"]) == 3, it["id"]
        expect = "Skinny" if it["category"] == "bottom" else "Fitted"
        assert it["volume_steps"][0]["label"] == expect, it["id"]


def test_a_button_value_round_trips_into_the_csv(tmp_path, tool):
    p = tmp_path / "w.csv"
    _write(p, [_row("a", category="bottom")])
    tool.REAL = p
    wide = dict(tool.VOLUME_SCALES["bottom"])["Wide"]
    tool.save("a", {"volume": str(wide), "structure": "0.4", "formality": "0.25"})
    row = next(csv.DictReader(open(p)))
    assert row["volume"] == "1.85" and row["structure"] == "0.4"


# --- order ------------------------------------------------------------------

def test_pieces_seen_in_two_or_more_photos_come_first(tool):
    got = tool.items()
    counts = [len(i["photos"]) for i in got]
    first_single = next((n for n, c in enumerate(counts) if c < 2), len(counts))
    assert all(c >= 2 for c in counts[:first_single])
    assert all(c < 2 for c in counts[first_single:]), "a multi-photo piece is stranded"
    assert first_single >= 1, "nothing has two photos; the ordering is untested"


def test_ordering_loses_nobody(tool):
    ids = [i["id"] for i in tool.items()]
    real = [r["id"] for r in csv.DictReader(open(ROOT / "data" / "wardrobe_real.csv"))]
    assert sorted(ids) == sorted(real)
    assert len(ids) == len(set(ids))
