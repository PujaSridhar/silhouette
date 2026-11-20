"""Fine-tune Mask2Former on Fashionpedia, evaluate it, and run it on her photos.

Needs the [vision] extra (torch, transformers, Pillow). Training is a GPU job;
the notebook in notebooks/ drives this module and is the intended way in.

Model: Mask2Former with a Swin backbone, starting from the COCO instance
checkpoint, re-headed to the 15 W1 classes (14 garments + sleeve). Instance
segmentation rather than semantic, because a top and the jacket over it are
two garments with overlapping extents, and sleeves overlap their garments by
construction.

Why not SegFormer: it is semantic-only, so it would give one "upper body"
region where the engine needs a top and an outer layer measured separately,
and it cannot represent a sleeve inside its shirt. It would be the right
choice only if every photo were a single garment.

    python -m silhouette.vision.train fit --annotations .../instances_attributes_train2020.json \
        --images .../train --val-annotations .../instances_attributes_val2020.json \
        --val-images .../test --out runs/w1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .categories import TRAIN_CLASSES, train_to_engine
from .fashionpedia import Sample, load
from .metrics import IoUAccumulator

__all__ = ["TrainConfig", "build_model", "fit", "evaluate", "predict_instances", "predict_photos"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class TrainConfig:
    annotations: str = ""
    images: str = ""
    val_annotations: str = ""
    val_images: str = ""
    out: str = "runs/w1"
    checkpoint: str = "facebook/mask2former-swin-small-coco-instance"
    licence: str = "all"            # "commercial" for a model that might ship
    size: int = 640                 # longer side, pixels
    batch_size: int = 8
    max_steps: int = 30000
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.05
    warmup_steps: int = 500
    clip_grad: float = 1.0
    min_area: float = 64.0
    fp16: bool = True
    num_workers: int = 4
    save_every: int = 2000
    eval_every: int = 5000
    eval_limit: int = 0             # 0 = whole val split
    seed: int = 0
    tiny: bool = False              # random tiny model, for CPU smoke tests only
    subset: int = 0                 # 0 = all training images
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------- data

def _resize_pad(img: np.ndarray, size: int):
    """Longer side to `size`, pad bottom/right to size x size. Returns the
    padded float image (3,S,S), the scale, and the valid (h, w)."""
    from PIL import Image
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    im = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
    im = (im - MEAN) / STD
    out = np.zeros((size, size, 3), dtype=np.float32)
    out[:nh, :nw] = im
    return out.transpose(2, 0, 1), s, (nh, nw)


def _resize_mask(m: np.ndarray, nh: int, nw: int, size: int) -> np.ndarray:
    from PIL import Image
    r = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)) > 127
    out = np.zeros((size, size), dtype=bool)
    out[:nh, :nw] = r
    return out


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def make_dataset(samples: list[Sample], image_dir: str, size: int, train: bool):
    import torch
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            s = samples[i]
            img = _load_rgb(s.path(image_dir))
            if img.shape[:2] != (s.height, s.width):
                raise ValueError(f"{s.file_name}: image is {img.shape[:2]}, annotation says "
                                 f"{(s.height, s.width)}")
            inst = [x for x in s.instances if not x.iscrowd]
            masks = [x.mask(s.height, s.width) for x in inst]
            if train and random.random() < 0.5:
                img = img[:, ::-1].copy()
                masks = [m[:, ::-1].copy() for m in masks]
            px, _scale, (nh, nw) = _resize_pad(img, size)
            ml = [_resize_mask(m, nh, nw, size) for m in masks]
            keep = [k for k, m in enumerate(ml) if m.any()]
            return {
                "pixel_values": torch.from_numpy(px),
                "mask_labels": torch.from_numpy(np.stack([ml[k] for k in keep]).astype(np.float32))
                if keep else torch.zeros((0, size, size)),
                "class_labels": torch.tensor([inst[k].train_id for k in keep], dtype=torch.long),
            }

    return _DS()


def collate(batch):
    import torch
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask_labels": [b["mask_labels"] for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
    }


# ---------------------------------------------------------------- model

def build_model(cfg: TrainConfig):
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    id2label = dict(enumerate(TRAIN_CLASSES))
    label2id = {v: k for k, v in id2label.items()}
    if cfg.tiny:
        from transformers import SwinConfig
        backbone = SwinConfig(embed_dim=16, depths=[1, 1, 1, 1], num_heads=[1, 1, 2, 2],
                              window_size=4, out_features=["stage1", "stage2", "stage3", "stage4"])
        config = Mask2FormerConfig(
            backbone_config=backbone, num_labels=len(TRAIN_CLASSES), id2label=id2label,
            label2id=label2id, hidden_dim=32, mask_feature_size=32, feature_size=32,
            encoder_feedforward_dim=64, dim_feedforward=64, encoder_layers=1,
            decoder_layers=2, num_attention_heads=2, num_queries=8, train_num_points=256,
            use_pretrained_backbone=False,
        )
        return Mask2FormerForUniversalSegmentation(config)
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg.checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)


def _optimizer(model, cfg: TrainConfig):
    import torch
    back, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (back if "pixel_level_module.encoder" in n else rest).append(p)
    return torch.optim.AdamW(
        [{"params": back, "lr": cfg.lr * cfg.backbone_lr_mult},
         {"params": rest, "lr": cfg.lr}],
        weight_decay=cfg.weight_decay)


def _lr_factor(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    return (1 - (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)) ** 0.9


# ------------------------------------------------------------- training

def fit(cfg: TrainConfig, samples: list[Sample] | None = None,
        val_samples: list[Sample] | None = None, log=print) -> dict:
    """Train, checkpointing to cfg.out, resuming from cfg.out/last if present."""
    import torch
    from torch.utils.data import DataLoader

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    if samples is None:
        samples = load(cfg.annotations, licence=cfg.licence, min_area=cfg.min_area)
    if cfg.subset:
        samples = samples[: cfg.subset]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    opt = _optimizer(model, cfg)
    use_amp = cfg.fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        log(f"resumed at step {step}")

    dl = DataLoader(make_dataset(samples, cfg.images, cfg.size, train=True),
                    batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    collate_fn=collate, drop_last=len(samples) > cfg.batch_size)
    history = []
    model.train()
    t0 = time.time()
    while step < cfg.max_steps:
        for batch in dl:
            if step >= cfg.max_steps:
                break
            for g, base in zip(opt.param_groups, (cfg.lr * cfg.backbone_lr_mult, cfg.lr)):
                g["lr"] = base * _lr_factor(step, cfg)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outp = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[m.to(device) for m in batch["mask_labels"]],
                    class_labels=[c.to(device) for c in batch["class_labels"]],
                )
            loss = outp.loss
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"loss is {loss.item()} at step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(opt)
            scaler.update()
            step += 1
            history.append(float(loss.item()))
            if step % 50 == 0 or step == 1:
                log(f"step {step:>6}  loss {loss.item():.3f}  {time.time() - t0:.0f}s")
            if step % cfg.save_every == 0 or step == cfg.max_steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, last)
            if val_samples is not None and cfg.eval_every and step % cfg.eval_every == 0:
                res = evaluate(model, val_samples, cfg.val_images, cfg.size, cfg.eval_limit)
                log(f"step {step}: mIoU {res['mIoU']}")
                (out / f"eval_{step}.json").write_text(json.dumps(res, indent=2))
                model.train()
    model.save_pretrained(out / "model")
    return {"steps": step, "loss_first": history[0] if history else None,
            "loss_last": history[-1] if history else None}


# ------------------------------------------------------------ inference

def _instances_from_outputs(cls_logits, mask_logits, valid_hw, orig_hw, threshold):
    """One image's queries -> [(train_id, score, bool mask at original size)]."""
    import torch
    import torch.nn.functional as F

    probs = cls_logits.softmax(-1)[:, :-1]          # drop the no-object class
    scores, labels = probs.max(-1)
    ml = F.interpolate(mask_logits[None], scale_factor=4, mode="bilinear",
                       align_corners=False)[0]
    nh, nw = valid_hw
    ml = ml[:, :nh, :nw]
    ml = F.interpolate(ml[None], size=orig_hw, mode="bilinear", align_corners=False)[0]
    sig = ml.sigmoid()
    binm = sig > 0.5
    # Mask2Former's own scoring: class confidence times mean mask confidence.
    mscore = (sig * binm).flatten(1).sum(1) / binm.flatten(1).sum(1).clamp(min=1)
    final = scores * mscore
    out = []
    for q in torch.nonzero(final > threshold).flatten().tolist():
        if binm[q].any():
            out.append((int(labels[q]), float(final[q]), binm[q].cpu().numpy()))
    return sorted(out, key=lambda t: -t[1])


def predict_instances(model, image: np.ndarray, size: int = 640, threshold: float = 0.3):
    import torch
    device = next(model.parameters()).device
    px, _s, valid = _resize_pad(image, size)
    model.eval()
    with torch.no_grad():
        o = model(pixel_values=torch.from_numpy(px)[None].to(device))
    return _instances_from_outputs(o.class_queries_logits[0], o.masks_queries_logits[0],
                                   valid, image.shape[:2], threshold)


def evaluate(model, samples: list[Sample], image_dir: str, size: int = 640,
             limit: int = 0, threshold: float = 0.3) -> dict:
    """Per-class mask IoU on a split, plus IoU pooled by engine category."""
    acc = IoUAccumulator(TRAIN_CLASSES)
    for s in (samples[:limit] if limit else samples):
        img = _load_rgb(s.path(image_dir))
        true: dict[int, np.ndarray] = {}
        for x in s.instances:
            m = x.mask(s.height, s.width)
            true[x.train_id] = true.get(x.train_id, np.zeros_like(m)) | m
        pred: dict[int, np.ndarray] = {}
        for tid, _score, m in predict_instances(model, img, size, threshold):
            pred[tid] = pred.get(tid, np.zeros_like(m)) | m
        acc.add(true, pred, (s.height, s.width))
    group = {i: (train_to_engine(i) or "sleeve") for i in range(len(TRAIN_CLASSES))}
    return {"mIoU": acc.mean(), "per_class": acc.per_class(),
            "by_engine_category": acc.grouped(group),
            "images": len(samples[:limit] if limit else samples)}


def predict_photos(model, photos: str | Path, wardrobe, size: int = 640,
                   threshold: float = 0.3, out_name: str = "masks_model") -> list[str]:
    """Run the model on every photo in photos/manifest.csv and write one mask
    per worn garment, matched by engine category, plus sleeves.png.

    The model does not know garment ids. It knows "this is a bottom"; the
    manifest says which bottom she was wearing. If two worn garments share a
    category the photo is skipped, because the match would be a guess."""
    from .sam import save_mask
    from .volume_eval import read_manifest

    photos = Path(photos)
    by_id = {g.id: g for g in wardrobe}
    notes = []
    for rec in read_manifest(photos):
        stem = Path(rec["photo"]).stem
        ids = [i.strip() for i in rec["garments"].split(";") if i.strip()]
        cats = [by_id[i].category for i in ids]
        if len(set(cats)) != len(cats):
            notes.append(f"{rec['photo']}: two worn garments share a category, skipped")
            continue
        img = _load_rgb(photos / "images" / rec["photo"])
        inst = predict_instances(model, img, size, threshold)
        odir = photos / out_name / stem
        sleeves = np.zeros(img.shape[:2], dtype=bool)
        for tid, _sc, m in inst:
            if train_to_engine(tid) is None:
                sleeves |= m
        save_mask(odir / "sleeves.png", sleeves)
        for gid, cat in zip(ids, cats):
            if cat == "shoe":
                continue
            best = next((m for tid, _sc, m in inst if train_to_engine(tid) == cat), None)
            if best is None:
                notes.append(f"{rec['photo']}: no {cat} predicted for {gid}")
                continue
            save_mask(odir / f"{gid}.png", best)
    return notes


def main(argv=None):  # pragma: no cover - thin CLI
    p = argparse.ArgumentParser(prog="python -m silhouette.vision.train")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    for k, v in asdict(TrainConfig()).items():
        if k == "extra":
            continue
        if isinstance(v, bool):
            f.add_argument(f"--{k.replace('_', '-')}", action="store_true", default=v)
        else:
            f.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = p.parse_args(argv)
    cfg = TrainConfig(**{k: v for k, v in vars(a).items() if k != "cmd"})
    val = load(cfg.val_annotations, licence="all") if cfg.val_annotations else None
    print(fit(cfg, val_samples=val))


if __name__ == "__main__":  # pragma: no cover
    main()
