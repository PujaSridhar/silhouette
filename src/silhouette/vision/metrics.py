"""Mask accuracy per class.

Fashionpedia's parts overlap their garments (a sleeve sits inside its shirt's
mask), so a single label map cannot hold the ground truth. IoU is computed per
class on binary masks instead: the union of every instance of that class in an
image, predicted against true, accumulated over the whole split.

This is the standard to report, but it is not the number that decides W1. A
mask can score a high IoU and still be a few pixels too wide at the hip, and
volume is read from exactly those edge pixels. See volume_eval for the metric
that matters.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

__all__ = ["IoUAccumulator"]


class IoUAccumulator:
    def __init__(self, class_names: Iterable[str]):
        self.names = list(class_names)
        self.inter = np.zeros(len(self.names), dtype=np.int64)
        self.union = np.zeros(len(self.names), dtype=np.int64)
        self.images_with = np.zeros(len(self.names), dtype=np.int64)

    def add(self, true: Mapping[int, np.ndarray], pred: Mapping[int, np.ndarray],
            shape: tuple[int, int]) -> None:
        """`true` and `pred` map class index -> bool mask (union of instances).
        A class missing from one side counts as an empty mask there."""
        empty = np.zeros(shape, dtype=bool)
        for c in set(true) | set(pred):
            t = np.asarray(true.get(c, empty), dtype=bool)
            p = np.asarray(pred.get(c, empty), dtype=bool)
            self.inter[c] += int((t & p).sum())
            self.union[c] += int((t | p).sum())
            if t.any():
                self.images_with[c] += 1

    def per_class(self) -> dict[str, float | None]:
        return {n: (float(i / u) if u else None)
                for n, i, u in zip(self.names, self.inter, self.union)}

    def mean(self) -> float | None:
        vals = [v for v in self.per_class().values() if v is not None]
        return float(np.mean(vals)) if vals else None

    def grouped(self, group_of: Mapping[int, str | None]) -> dict[str, float | None]:
        """IoU pooled over groups of classes, e.g. engine categories."""
        out: dict[str, list[int]] = {}
        for c, g in group_of.items():
            if g is not None:
                out.setdefault(g, []).append(c)
        res: dict[str, float | None] = {}
        for g, cs in out.items():
            u = int(self.union[cs].sum())
            res[g] = float(self.inter[cs].sum() / u) if u else None
        return res
