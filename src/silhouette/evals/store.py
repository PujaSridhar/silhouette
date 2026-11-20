"""Append-only JSONL label store.

Re-labels are kept, not overwritten. A second rating of the same outfit by the
same person in a later session is the only way to measure self-agreement, and
self-agreement is the ceiling on what any model can achieve. Throwing the
duplicate away to keep the file tidy would destroy the most important number in
the harness.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .schema import Label, label_from_dict

__all__ = ["LabelStore"]


class LabelStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, label: Label) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(label.to_dict(), sort_keys=True) + "\n")

    def all(self) -> list[Label]:
        if not self.path.exists():
            return []
        out: list[Label] = []
        for n, line in enumerate(self.path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(label_from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{self.path}:{n} is not a valid label: {exc}") from exc
        return out

    def expert(self) -> list[Label]:
        return [l for l in self.all() if l.source == "expert"]

    def bootstrap(self) -> list[Label]:
        return [l for l in self.all() if l.source == "bootstrap"]

    def latest_per_outfit(self, source: str | None = None) -> list[Label]:
        """One label per outfit - the most recent. For metrics that need
        independent observations rather than repeats."""
        pool = self.all() if source is None else [l for l in self.all() if l.source == source]
        by_key: dict[str, Label] = {}
        for l in sorted(pool, key=lambda l: l.created_at):
            by_key[l.outfit_key] = l
        return list(by_key.values())

    def repeats(self, source: str | None = None) -> list[tuple[Label, Label]]:
        """Pairs of independent ratings of the same outfit, for self-agreement."""
        pool = self.all() if source is None else [l for l in self.all() if l.source == source]
        groups: dict[str, list[Label]] = defaultdict(list)
        for l in pool:
            groups[l.outfit_key].append(l)
        pairs: list[tuple[Label, Label]] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda l: l.created_at)
            pairs.append((group[0], group[-1]))
        return pairs

    def labelled_keys(self, source: str | None = "expert") -> set[str]:
        """Outfits already rated. Expert labels by default, so the labelling
        queue never skips an outfit that only the synthetic rater has seen.
        Pass source=None for every source."""
        pool = self.all() if source is None else [l for l in self.all() if l.source == source]
        return {l.outfit_key for l in pool}

    def __len__(self) -> int:
        return len(self.all())
