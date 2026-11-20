"""Several stylists: the shared set, agreement between them, and the outfits
they could not judge."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from silhouette.cli import main
from silhouette.evals.raters import (
    cant_judge_summary, inter_rater_agreement, krippendorff_alpha_interval,
)
from silhouette.evals.report import evaluate_labels
from silhouette.evals.sampling import SHARED_N, fnv1a32, shared_hash_key, shared_set
from silhouette.evals.schema import CantJudge, Label, cant_judge_from_dict
from silhouette.evals.store import CantJudgeStore, LabelStore, cant_judge_path

ROOT = Path(__file__).resolve().parents[1]


def _label(id, who, outfit, overall, when, context="office", wear=True):
    return Label(id=id, outfit=outfit, context=context, overall=overall,
                 would_wear=wear, labeller=who, created_at=f"2026-11-20T10:{when:02d}:00+00:00")


O1 = {"top": "a", "bottom": "b", "shoe": "c", "tucked": False}
O2 = {"top": "d", "bottom": "b", "shoe": "c", "tucked": False}
O3 = {"top": "e", "bottom": "f", "shoe": "c", "tucked": True}


# ------------------------------------------------------------------ alpha


def test_alpha_matches_krippendorffs_worked_example():
    """Krippendorff (2011), 'Computing Krippendorff's alpha-reliability': four
    observers, twelve units, some missing. Interval alpha is 0.849."""
    A = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
    B = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]
    C = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
    D = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]
    units = [[x for x in col if x is not None] for col in zip(A, B, C, D)]
    assert krippendorff_alpha_interval(units) == pytest.approx(0.849, abs=5e-4)


def test_alpha_is_one_for_perfect_agreement_and_nan_without_overlap():
    assert krippendorff_alpha_interval([[1, 1], [3, 3], [5, 5, 5]]) == pytest.approx(1.0)
    assert math.isnan(krippendorff_alpha_interval([[3], [4]]))      # nothing pairable
    assert math.isnan(krippendorff_alpha_interval([[3, 3], [3, 3]]))  # no variation


def test_alpha_goes_negative_when_raters_disagree_systematically():
    assert krippendorff_alpha_interval([[1, 5], [5, 1], [2, 4], [4, 2]]) < 0


# ----------------------------------------------------------- inter-rater


def test_agreement_uses_each_raters_first_rating_only():
    """A repeat is self-agreement's business; here it would count asha twice."""
    ls = [
        _label("a1", "asha", O1, 4, 1), _label("b1", "bea", O1, 4, 2),
        _label("a2", "asha", O2, 2, 3), _label("b2", "bea", O2, 2, 4),
        _label("a3", "asha", O1, 1, 9),          # asha's later repeat
    ]
    ir = inter_rater_agreement(ls)
    assert ir.n_units == 2
    assert ir.alpha == pytest.approx(1.0)
    assert ir.pairs[0].n == 2 and ir.pairs[0].exact == 1.0


def test_agreement_ignores_outfits_only_one_rater_saw():
    ls = [_label("a1", "asha", O1, 4, 1), _label("b1", "bea", O1, 3, 2),
          _label("a2", "asha", O3, 1, 3)]
    ir = inter_rater_agreement(ls)
    assert ir.n_units == 1 and ir.raters == ["asha", "bea"]
    assert "1 shared outfit" in ir.summary()


def test_agreement_with_one_rater_says_so():
    ir = inter_rater_agreement([_label("a1", "asha", O1, 4, 1)])
    assert "second" in ir.summary()


def test_would_wear_agreement_counts_rater_pairs():
    ls = [_label("a", "asha", O1, 4, 1, wear=True), _label("b", "bea", O1, 4, 2, wear=True),
          _label("c", "chen", O1, 4, 3, wear=False)]
    assert inter_rater_agreement(ls).would_wear_agree == pytest.approx(1 / 3)


# ---------------------------------------------------------- can't judge


def test_cant_judge_round_trips_and_keys_like_a_label(tmp_path):
    st = CantJudgeStore(tmp_path / "cant_judge.jsonl")
    m = CantJudge(id="cj1", outfit=O1, context="office", labeller="asha", reason="no photo")
    st.append(m)
    back = st.all()
    assert back == [m]
    assert m.outfit_key == _label("x", "asha", O1, 3, 1).outfit_key


def test_cant_judge_is_not_a_label():
    """No score field, so nothing that reads labels can take it for one."""
    d = CantJudge(id="cj1", outfit=O1, context="office", labeller="asha").to_dict()
    assert "overall" not in d and "would_wear" not in d
    with pytest.raises(KeyError):
        from silhouette.evals.schema import label_from_dict
        label_from_dict(d)


def test_cant_judge_lives_beside_the_labels_not_in_them(tmp_path):
    assert cant_judge_path(tmp_path / "labels.jsonl") == tmp_path / "cant_judge.jsonl"


def test_cant_judge_summary_finds_outfits_several_raters_found_unclear():
    marks = [
        cant_judge_from_dict({"id": "1", "outfit": O1, "context": "office", "labeller": "asha",
                              "reason": "can't see the fabric"}),
        cant_judge_from_dict({"id": "2", "outfit": O1, "context": "office", "labeller": "bea"}),
        cant_judge_from_dict({"id": "3", "outfit": O2, "context": "office", "labeller": "bea"}),
        cant_judge_from_dict({"id": "4", "outfit": O1, "context": "office", "labeller": "asha"}),
    ]
    cj = cant_judge_summary(marks)
    assert cj.n == 4 and cj.by_rater == {"asha": 2, "bea": 2}
    assert len(cj.unclear) == 1
    key, k = cj.unclear[0]
    assert k == 2 and cj.reasons[key] == ["can't see the fabric"]


# ------------------------------------------------------------ shared set


def test_fnv1a32_known_values():
    """The published FNV-1a test vectors, so the JS port has a fixed target."""
    assert fnv1a32("") == 0x811C9DC5
    assert fnv1a32("a") == 0xE40C292C
    assert fnv1a32("foobar") == 0xBF9CF968


@pytest.fixture(scope="module")
def shared(wardrobe, cv, rules):
    return shared_set(wardrobe, cv, rules)


def test_shared_set_size_and_feasibility(shared):
    assert len(shared) == SHARED_N
    assert all(t.scored.feasible for t in shared)
    keys = [shared_hash_key(t.outfit, t.context) for t in shared]
    assert len(set(keys)) == SHARED_N


def test_shared_set_is_deterministic_and_order_independent(wardrobe, cv, rules, shared):
    """Same wardrobe, same set: whatever order the garments come in."""
    again = shared_set(list(reversed(wardrobe)), cv, rules)
    key = lambda ts: [shared_hash_key(t.outfit, t.context) for t in ts]
    assert key(again) == key(shared)


def test_shared_set_ignores_what_anyone_has_rated():
    """It takes no label store at all, which is the point: a stylist who starts
    late gets the same set as one who finished."""
    import inspect
    assert "store" not in inspect.signature(shared_set).parameters


def test_shared_set_spans_the_score_range(wardrobe, cv, rules, shared):
    """Agreement on only so-so outfits is flattered by the narrow range, so
    the set reaches into the bottom and top fifths of what can be served."""
    from silhouette.solver import candidates, score
    outfits = list(candidates(wardrobe, cv.frame))
    pool = sorted(s.total for n in cv.contexts for s in
                  (score(o, cv, cv.context(n), rules) for o in outfits) if s.feasible)
    lo, hi = pool[0], pool[-1]
    fifth = (hi - lo) / 5
    totals = [t.scored.total for t in shared]
    assert min(totals) < lo + fifth
    assert max(totals) > hi - fifth


def test_shared_set_changes_with_the_seed(wardrobe, cv, rules, shared):
    other = shared_set(wardrobe, cv, rules, seed="something else")
    assert {shared_hash_key(t.outfit, t.context) for t in other} != {
        shared_hash_key(t.outfit, t.context) for t in shared}


# -------------------------------------------------------- import and eval


def _export(garments, labels, marks, cv_name="Wearer"):
    from silhouette.models import load_style_cv
    cv = load_style_cv(ROOT / "data" / "style_cv.yaml")
    from dataclasses import asdict
    return {
        "styleCV": {"name": cv_name, "heightCm": cv.height_cm, "landmarks": cv.frame.landmarks,
                    "widths": cv.frame.widths, "goals": cv.goals,
                    "contexts": [dict(asdict(c), name=n) for n, c in cv.contexts.items()]},
        "garments": garments, "labels": labels, "cantJudge": marks,
    }


def _garments(wardrobe):
    from dataclasses import asdict
    return [asdict(g) for g in wardrobe]


def test_import_merges_stylists_and_keeps_marks_apart(tmp_path, wardrobe, cv, rules, shared,
                                                       capsys):
    gs = _garments(wardrobe)

    def lab(i, who, t, overall):
        o = t.outfit
        return {"id": f"{who}{i}", "context": t.context, "overall": overall, "ratings": None,
                "would_wear": overall >= 3, "labeller": who, "source": "expert", "note": "",
                "engine_version": 1, "created_at": f"2026-11-20T10:{i:02d}:00+00:00",
                "outfit": {s: (getattr(o, s).id if getattr(o, s) else None)
                           for s in ("top", "bottom", "dress", "outer", "shoe", "drape")}
                | {"tucked": o.tucked}}

    a_labels = [lab(i, "asha", t, 1 + i % 5) for i, t in enumerate(shared)]
    b_labels = [lab(i, "bea", t, 1 + (i + 1) % 5) for i, t in enumerate(shared[:6])]
    mark = {"id": "cj1", "outfit": a_labels[0]["outfit"], "context": shared[0].context,
            "labeller": "bea", "reason": "no photo", "engine_version": 1,
            "created_at": "2026-11-20T11:00:00+00:00"}
    fa, fb = tmp_path / "asha.json", tmp_path / "bea.json"
    fa.write_text(json.dumps(_export(gs, a_labels, [])))
    fb.write_text(json.dumps(_export(gs, b_labels, [mark])))
    out = tmp_path / "imp"

    assert main(["import-app", str(fa), str(fb), "--out", str(out)]) == 0
    txt = capsys.readouterr().out
    assert "couldn't judge 1" in txt
    labels = LabelStore(out / "labels.jsonl").all()
    assert len(labels) == 18
    assert len(CantJudgeStore(out / "cant_judge.jsonl").all()) == 1
    assert all("reason" not in json.loads(l) for l in (out / "labels.jsonl").read_text().splitlines())

    # The eval reports agreement on the overlap, shared-set progress and the mark.
    rc = main(["--wardrobe", str(out / "wardrobe.csv"), "--style-cv", str(out / "style_cv.yaml"),
               "--labels", str(out / "labels.jsonl"), "eval"])
    assert rc == 0
    txt = capsys.readouterr().out
    assert "between raters" in txt and "over 6 shared outfits, 2 raters" in txt
    assert "asha vs bea" in txt
    assert f"asha {SHARED_N}/{SHARED_N}, bea 6/{SHARED_N}" in txt
    assert "couldn't judge" in txt and "not ratings" in txt


def test_import_refuses_two_different_wardrobes(tmp_path, wardrobe):
    gs = _garments(wardrobe)
    other = [dict(g) for g in gs]
    other[0]["name"] = "something else entirely"
    fa, fb = tmp_path / "a.json", tmp_path / "b.json"
    fa.write_text(json.dumps(_export(gs, [], [])))
    fb.write_text(json.dumps(_export(other, [], [])))
    with pytest.raises(SystemExit):
        main(["import-app", str(fa), str(fb), "--out", str(tmp_path / "imp")])


def test_old_exports_without_marks_still_import(tmp_path, wardrobe, capsys):
    raw = _export(_garments(wardrobe), [], [])
    del raw["cantJudge"]
    f = tmp_path / "old.json"
    f.write_text(json.dumps(raw))
    assert main(["import-app", str(f), "--out", str(tmp_path / "imp")]) == 0
    assert "couldn't judge 0" in capsys.readouterr().out


def test_marks_never_reach_the_scoreboard(tmp_path, wardrobe, cv, rules, shared):
    """evaluate_labels reads labels only; the marks file beside them is invisible to it."""
    st = LabelStore(tmp_path / "labels.jsonl")
    t = shared[0]
    o = {s: (getattr(t.outfit, s).id if getattr(t.outfit, s) else None)
         for s in ("top", "bottom", "dress", "outer", "shoe", "drape")} | {"tucked": t.outfit.tucked}
    st.append(Label(id="l1", outfit=o, context=t.context, overall=4, labeller="asha"))
    st.append(Label(id="l2", outfit=o, context=t.context, overall=2, labeller="bea"))
    CantJudgeStore(cant_judge_path(st.path)).append(
        CantJudge(id="m", outfit=o, context=t.context, labeller="chen"))
    rep = evaluate_labels(st.expert(), wardrobe, cv, rules, store=st)
    assert rep.n_expert == 2


# ------------------------------------------------------------- the app


def test_app_keeps_marks_out_of_the_labels_collection():
    tpl = (ROOT / "app" / "index.template.html").read_text()
    assert 'collection("cantjudge")' in tpl
    assert "Store.saveCantJudge(c.id,mark)" in tpl
    assert "cantJudge:MARKS()" in tpl


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_rate_flow_in_the_built_app():
    """The built page's own click handlers, driven in Node: two raters meet the
    same shared outfits, and Can't judge saves a mark, not a label. Skips when
    the page has not been built since the template changed."""
    page, tpl = ROOT / "app" / "index.html", ROOT / "app" / "index.template.html"
    if not page.exists() or page.stat().st_mtime < tpl.stat().st_mtime:
        pytest.skip("app/index.html not built from the current template (make build)")
    r = subprocess.run(["node", str(ROOT / "tools" / "check_rate_flow.js"), str(page)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
