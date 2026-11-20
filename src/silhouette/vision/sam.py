"""SAM 2 as an annotation accelerator, and nothing more.

SAM 2 segments whatever you point at. It does not know a shirt from a skirt,
and it is not trained here. Its job is to turn a box or a couple of clicks
into a clean mask quickly, so hand masks for her own photos take seconds each
instead of minutes of polygon tracing. The person doing the clicking says
which garment it is; that id is the label, not anything SAM outputs.

Those hand masks are the reference the fine-tuned model is scored against on
phone photos, and the input the volume harness measures first.

    silhouette-vision annotate photos/images/IMG_0412.jpg \
        --garment bot_wide_blk --box 140,610,560,1480
    silhouette-vision annotate photos/images/IMG_0412.jpg --sleeves \
        --box 90,300,190,820 --box 520,300,620,820
    silhouette-vision annotate photos/images/IMG_0412.jpg --person \
        --box 60,20,660,1560          # prints crown_y and sole_y for the manifest

Needs the [sam] extra and downloads the checkpoint on first use. Runs on CPU,
slowly; a GPU or Apple MPS is much faster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["parse_box", "parse_point", "union_into", "save_mask", "overlay", "Segmenter"]

DEFAULT_CHECKPOINT = "facebook/sam2.1-hiera-small"


def parse_box(s: str) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in s.split(","))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"box {s!r} must be x0,y0,x1,y1 with x1>x0 and y1>y0")
    return x0, y0, x1, y1


def parse_point(s: str) -> tuple[float, float, int]:
    """'x,y' or 'x,y,-' for a point that is NOT part of the garment."""
    parts = s.split(",")
    label = 0 if len(parts) == 3 and parts[2].strip() == "-" else 1
    return float(parts[0]), float(parts[1]), label


def union_into(path: Path, mask: np.ndarray) -> np.ndarray:
    """OR a mask into an existing PNG mask (for sleeves, drawn one at a time)."""
    mask = np.asarray(mask, dtype=bool)
    if path.exists():
        from PIL import Image
        old = np.asarray(Image.open(path).convert("L")) > 127
        if old.shape != mask.shape:
            raise ValueError(f"{path} is {old.shape}, new mask is {mask.shape}")
        mask = mask | old
    return mask


def save_mask(path: Path, mask: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def overlay(image: np.ndarray, mask: np.ndarray, color=(255, 64, 64), alpha=0.45) -> np.ndarray:
    out = image.astype(float).copy()
    m = np.asarray(mask, dtype=bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, dtype=float)
    return out.astype(np.uint8)


class Segmenter:
    """Thin wrapper over SAM2ImagePredictor."""

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None):
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise SystemExit(
                "SAM 2 is not installed. pip install -e '.[sam]', which installs it "
                "from github.com/facebookresearch/sam2. Do not pip install sam2 from "
                "PyPI: that name is a third-party upload, not Meta's."
            ) from e
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.predictor = SAM2ImagePredictor.from_pretrained(checkpoint, device=device)
        self._image_key = None

    def set_image(self, image: np.ndarray, key: str) -> None:
        if key != self._image_key:
            self.predictor.set_image(image)
            self._image_key = key

    def segment(self, boxes=(), points=()) -> np.ndarray:
        """One mask from one box plus any points; several boxes are OR-ed."""
        pts = np.array([[x, y] for x, y, _ in points], dtype=float) if points else None
        lbl = np.array([lab for _x, _y, lab in points], dtype=int) if points else None
        prompts = list(boxes) or [None]
        out = None
        for b in prompts:
            masks, _scores, _ = self.predictor.predict(
                point_coords=pts, point_labels=lbl,
                box=None if b is None else np.array(b, dtype=float),
                multimask_output=False,
            )
            m = np.asarray(masks[0]) > 0.5
            out = m if out is None else (out | m)
        return out
