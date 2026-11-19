from __future__ import annotations

from pathlib import Path

import pytest

from silhouette.geometry import compute
from silhouette.models import Outfit, load_style_cv, load_wardrobe
from silhouette.rules import evaluate, load_rules

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def wardrobe():
    return load_wardrobe(ROOT / "data" / "wardrobe.csv")


@pytest.fixture(scope="session")
def catalog():
    return load_wardrobe(ROOT / "data" / "catalog.csv")


@pytest.fixture(scope="session")
def cv():
    return load_style_cv(ROOT / "data" / "style_cv.yaml")


@pytest.fixture(scope="session")
def rules():
    return load_rules(ROOT / "rules" / "core.yaml")


@pytest.fixture(scope="session")
def W(wardrobe):
    """Garments by id, for readable fixtures."""
    return {g.id: g for g in wardrobe}


@pytest.fixture
def verdict_for(cv, rules):
    """Score an outfit in a context and hand back (features, verdict)."""

    def _go(outfit: Outfit, context: str):
        f = compute(outfit, cv.frame, cv.height_cm)
        return f, evaluate(outfit, f, cv, cv.context(context), rules)

    return _go


@pytest.fixture
def fired_ids(verdict_for):
    def _go(outfit: Outfit, context: str) -> set[str]:
        _f, v = verdict_for(outfit, context)
        return {fr.rule.id for fr in v.fired}

    return _go


@pytest.fixture
def violated_ids(verdict_for):
    def _go(outfit: Outfit, context: str) -> set[str]:
        _f, v = verdict_for(outfit, context)
        return {r.id for r, _msg in v.violations}

    return _go
