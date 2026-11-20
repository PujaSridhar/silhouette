"""Load Fashionpedia's COCO-format annotations into the W1 label space.

Nothing here touches images or a GPU. It reads the annotation JSON, keeps the
classes in `categories.TRAIN_SOURCE_IDS`, filters by licence, and decodes masks
on demand. Masks come as polygons (about 97% in val) or compressed RLE; both
are decoded here without pycocotools, so the loader works in a bare install.

    samples = load("instances_attributes_train2020.json", licence="commercial")
    s = samples[0]
    m = s.instances[0].mask(s.height, s.width)   # bool array, H x W

Image files are not part of the annotations. The paths come from `file_name`
joined to whatever directory the images were unzipped into.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .categories import (
    FASHIONPEDIA, SLEEVE_ID, TRAIN_CLASSES, licence_group, source_to_train,
    train_to_engine,
)

__all__ = ["Instance", "Sample", "load", "class_counts", "decode_rle", "rle_counts_from_string"]


def rle_counts_from_string(s: str) -> list[int]:
    """COCO's compressed RLE string -> run lengths. A port of maskApi.c."""
    counts: list[int] = []
    p = 0
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    """COCO RLE (compressed string or plain list) -> bool mask, H x W."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = rle_counts_from_string(counts)
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    val = False
    for n in counts:
        if val:
            flat[pos:pos + n] = True
        pos += n
        val = not val
    # COCO stores masks column-major.
    return flat.reshape((w, h)).T


def _polygons_to_mask(polys: list[list[float]], h: int, w: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for poly in polys:
        if len(poly) >= 6:
            draw.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=1)
    return np.asarray(img, dtype=bool)


@dataclass(frozen=True)
class Instance:
    train_id: int
    source_id: int
    bbox: tuple[float, float, float, float]   # x, y, w, h
    area: float
    segmentation: Any
    attribute_ids: tuple[int, ...] = ()
    iscrowd: bool = False

    @property
    def name(self) -> str:
        return FASHIONPEDIA[self.source_id]

    @property
    def engine_category(self) -> str | None:
        return train_to_engine(self.train_id)

    @property
    def is_sleeve(self) -> bool:
        return self.source_id == SLEEVE_ID

    def mask(self, height: int, width: int) -> np.ndarray:
        seg = self.segmentation
        if isinstance(seg, dict):
            m = decode_rle(seg)
            if m.shape != (height, width):
                raise ValueError(f"RLE size {m.shape} does not match image {(height, width)}")
            return m
        return _polygons_to_mask(seg, height, width)


@dataclass(frozen=True)
class Sample:
    image_id: int
    file_name: str
    width: int
    height: int
    licence: int
    instances: list[Instance] = field(default_factory=list)

    def path(self, image_dir: str | Path) -> Path:
        return Path(image_dir) / self.file_name


def load(
    annotation_path: str | Path,
    licence: str = "all",
    min_area: float = 0.0,
    keep_empty: bool = False,
) -> list[Sample]:
    """Read an annotation file and return samples in the W1 label space.

    `licence` is a group from `categories.LICENCE_GROUPS`. `min_area` drops
    tiny instances (in pixels). Images with no kept instance are dropped
    unless `keep_empty`.
    """
    with open(annotation_path) as fh:
        raw = json.load(fh)

    names = {c["id"]: c["name"] for c in raw["categories"]}
    for cid, name in FASHIONPEDIA.items():
        if names.get(cid) != name:
            raise ValueError(
                f"category {cid} is {names.get(cid)!r} in this file, expected {name!r}; "
                "the class mapping in categories.py would be wrong for it"
            )

    allowed = licence_group(licence)
    by_image: dict[int, list[Instance]] = {}
    for a in raw["annotations"]:
        tid = source_to_train(a["category_id"])
        if tid is None or a.get("area", 0) < min_area:
            continue
        by_image.setdefault(a["image_id"], []).append(
            Instance(
                train_id=tid,
                source_id=a["category_id"],
                bbox=tuple(a["bbox"]),
                area=float(a.get("area", 0)),
                segmentation=a["segmentation"],
                attribute_ids=tuple(a.get("attribute_ids", ())),
                iscrowd=bool(a.get("iscrowd", 0)),
            )
        )

    out: list[Sample] = []
    for im in raw["images"]:
        if int(im.get("license", 11)) not in allowed:
            continue
        inst = by_image.get(im["id"], [])
        if not inst and not keep_empty:
            continue
        out.append(Sample(im["id"], im["file_name"], im["width"], im["height"],
                          int(im.get("license", 11)), inst))
    return out


def class_counts(samples: list[Sample]) -> dict[str, int]:
    """Instances per training class, in training-class order."""
    c = Counter(i.train_id for s in samples for i in s.instances)
    return {TRAIN_CLASSES[t]: c.get(t, 0) for t in range(len(TRAIN_CLASSES))}
