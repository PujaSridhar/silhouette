"""Fashionpedia's ontology, and the subset W1 trains on.

Fashionpedia (Jia et al., 2020) labels 46 categories: 27 whole garments and
accessories, and 19 garment parts and decorations. Category ids below are the
dataset's own, checked against `instances_attributes_val2020.json`.

W1 trains on the 14 whole garments the engine can use, plus `sleeve`. Sleeve is
a part, not a garment, and it is here for one reason: a top's mask includes its
sleeves, and a sleeve at the wearer's side would be read as torso width. The
volume measurement subtracts predicted sleeves before measuring a top.

Engine categories are the ones in `models.Garment.category`. Fashionpedia's
classes are finer than the engine's five, so the model keeps them and the
mapping is applied afterwards. The fine class is a free subcategory hint for W2.
"""

from __future__ import annotations

__all__ = [
    "FASHIONPEDIA", "GARMENT_TO_ENGINE", "SLEEVE_ID", "TRAIN_SOURCE_IDS",
    "TRAIN_CLASSES", "source_to_train", "train_to_engine",
    "LICENCE_GROUPS", "licence_group",
]

# id -> name, exactly as the dataset spells them.
FASHIONPEDIA: dict[int, str] = {
    0: "shirt, blouse", 1: "top, t-shirt, sweatshirt", 2: "sweater",
    3: "cardigan", 4: "jacket", 5: "vest", 6: "pants", 7: "shorts", 8: "skirt",
    9: "coat", 10: "dress", 11: "jumpsuit", 12: "cape", 13: "glasses", 14: "hat",
    15: "headband, head covering, hair accessory", 16: "tie", 17: "glove",
    18: "watch", 19: "belt", 20: "leg warmer", 21: "tights, stockings",
    22: "sock", 23: "shoe", 24: "bag, wallet", 25: "scarf", 26: "umbrella",
    27: "hood", 28: "collar", 29: "lapel", 30: "epaulette", 31: "sleeve",
    32: "pocket", 33: "neckline", 34: "buckle", 35: "zipper", 36: "applique",
    37: "bead", 38: "bow", 39: "flower", 40: "fringe", 41: "ribbon",
    42: "rivet", 43: "ruffle", 44: "sequin", 45: "tassel",
}

# Whole garments the engine has a slot for. Choices worth knowing about:
#   cardigan, jacket, coat, cape -> outer (worn over a top, never as the top)
#   vest -> outer, because over a shirt is the common case; it is ambiguous
#   jumpsuit -> dress, because the engine's dress slot means "one piece"
GARMENT_TO_ENGINE: dict[int, str] = {
    0: "top", 1: "top", 2: "top",
    3: "outer", 4: "outer", 5: "outer", 9: "outer", 12: "outer",
    6: "bottom", 7: "bottom", 8: "bottom",
    10: "dress", 11: "dress",
    23: "shoe",
}

SLEEVE_ID = 31

# The training label space: source ids in a fixed order, so class index i
# always means the same thing in a checkpoint.
TRAIN_SOURCE_IDS: tuple[int, ...] = tuple(sorted(GARMENT_TO_ENGINE)) + (SLEEVE_ID,)
TRAIN_CLASSES: tuple[str, ...] = tuple(FASHIONPEDIA[i] for i in TRAIN_SOURCE_IDS)

_SRC_TO_TRAIN = {s: i for i, s in enumerate(TRAIN_SOURCE_IDS)}


def source_to_train(source_id: int) -> int | None:
    """Fashionpedia category id -> training class index, or None if unused."""
    return _SRC_TO_TRAIN.get(source_id)


def train_to_engine(train_id: int) -> str | None:
    """Training class index -> engine category. `sleeve` maps to None."""
    return GARMENT_TO_ENGINE.get(TRAIN_SOURCE_IDS[train_id])


# Fashionpedia gives every image its own licence id (the `licenses` block of
# the annotation file). The annotations are CC BY 4.0; the images are not one
# licence. In the val split about 70% are NonCommercial variants. `commercial`
# keeps only licences that allow commercial use, for a model that might ship.
LICENCE_GROUPS: dict[str, frozenset[int]] = {
    "all": frozenset(range(12)),
    # BY, BY-SA, BY-ND, public domain, Unsplash, Pexels, Freestocks, Burst.
    # BY-ND allows commercial use; a trained model is arguably not a
    # derivative of an image, but that is a legal question, not ours.
    "commercial": frozenset({0, 1, 2, 6, 7, 8, 9, 10}),
}


def licence_group(name: str) -> frozenset[int]:
    if name not in LICENCE_GROUPS:
        raise KeyError(f"unknown licence group {name!r}; have {sorted(LICENCE_GROUPS)}")
    return LICENCE_GROUPS[name]
