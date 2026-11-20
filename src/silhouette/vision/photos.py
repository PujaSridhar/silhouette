"""Photo preparation and EXIF-safe image loading for W1.

Phone photos come with two practical traps:

1. **EXIF orientation.** An iPhone portrait shot is often stored sideways in the
   raw pixel buffer with EXIF tag 0x0112 (Orientation=6). Preview honours the
   tag; plain ``PIL.Image.open`` does not, so boxes read off Preview land on
   the wrong axes and ``crown_y`` / ``sole_y`` come out sideways. ``open_rgb``
   always runs ``ImageOps.exif_transpose`` before returning pixels.
2. **HEIC and Google Photos Live Photo zips.** Google Photos exports Live
   Photos as ``.zip`` archives containing both a ``.heic``/``.jpg`` still and a
   ``.mov`` companion video, and multiple exports can share filenames like
   ``FullSizeRender.heic`` / ``FullSizeRender.jpg``. ``prep_photos`` unpacks
   only the image stills without overwriting collisions, converts HEIC to JPEG,
   and appends new photos to ``photos/manifest.csv`` without touching rows
   already filled in.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

__all__ = ["IMAGE_EXTS", "HEIC_EXTS", "MANIFEST_COLUMNS", "open_rgb", "convert_heic", "prep_photos"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
HEIC_EXTS = {".heic", ".heif"}
ALL_IMAGE_EXTS = IMAGE_EXTS | HEIC_EXTS
MANIFEST_COLUMNS = ["photo", "garments", "crown_y", "sole_y", "heel_cm"]

_HEIF_REGISTERED = False


def _register_heif() -> bool:
    """Register pillow-heif with Pillow if installed."""
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return True
    try:
        import pillow_heif  # type: ignore[import-not-found]
    except ImportError:
        return False
    pillow_heif.register_heif_opener()
    _HEIF_REGISTERED = True
    return True


def convert_heic(src: str | Path, dst: str | Path) -> None:
    """Convert a HEIC/HEIF image to JPEG.

    Uses ``pillow-heif`` + Pillow when installed (baking EXIF orientation into
    pixels), otherwise falls back to macOS ``sips`` (which preserves the EXIF
    orientation tag for ``open_rgb`` and Preview).
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _register_heif():
        try:
            from PIL import Image, ImageOps
            with Image.open(src) as im:
                ImageOps.exif_transpose(im).convert("RGB").save(dst, format="JPEG", quality=95)
            return
        except ImportError:
            pass
    sips = shutil.which("sips")
    if sips:
        subprocess.run(
            [sips, "-s", "format", "jpeg", "-s", "formatOptions", "92", str(src), "--out", str(dst)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return
    raise RuntimeError(
        f"cannot convert {src.name}: install pillow-heif (pip install pillow-heif) "
        "or run on macOS with sips"
    )


def open_rgb(path: str | Path) -> np.ndarray:
    """Open an image as an upright uint8 RGB array, honouring EXIF rotation and HEIC."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    p = Path(path)
    _register_heif()
    try:
        with Image.open(p) as im:
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except (UnidentifiedImageError, OSError):
        if p.suffix.lower() not in HEIC_EXTS:
            raise
        if not shutil.which("sips"):
            raise SystemExit(
                f"Cannot open HEIC file {p.name}. Install pillow-heif "
                "(pip install -e '.[sam]') or run `silhouette-vision prep-photos` first."
            )
        with tempfile.TemporaryDirectory() as tmp:
            jpg = Path(tmp) / f"{p.stem}.jpg"
            convert_heic(p, jpg)
            with Image.open(jpg) as im:
                return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def _unique_path(directory: Path, stem: str, suffix: str, used_stems: set[str]) -> Path:
    """Pick a path in *directory* whose stem does not collide (case-insensitively).

    Masks live under ``photos/masks/<stem>/``, so two photos sharing a stem
    (like ``FullSizeRender.jpg`` and ``FullSizeRender.heic``) would collide on
    masks even before HEIC conversion. Disambiguating by stem protects both.
    """
    cand_stem = stem
    n = 2
    while cand_stem.lower() in used_stems or (directory / f"{cand_stem}{suffix}").exists():
        cand_stem = f"{stem}_{n}"
        n += 1
    used_stems.add(cand_stem.lower())
    return directory / f"{cand_stem}{suffix}"


def prep_photos(photos: str | Path = "photos") -> dict:
    """Unzip Google Photos archives, convert HEIC -> JPEG, and populate manifest.csv.

    Safe to re-run: existing manifest rows (and any garment ids / crown_y /
    sole_y already entered) are preserved untouched, and filename/stem
    collisions are disambiguated instead of overwritten.
    """
    root = Path(photos)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)

    # Track stems already present among non-HEIC images so nothing is clobbered.
    used_stems: set[str] = {
        p.stem.lower()
        for p in images.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }

    unzipped = 0
    skipped_zip_entries = 0
    zips_removed = 0

    for zpath in sorted(images.glob("*.zip")):
        if not zpath.is_file():
            continue
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename).parts
                if any(part == "__MACOSX" or part.startswith("._") for part in parts):
                    skipped_zip_entries += 1
                    continue
                name = Path(info.filename).name
                ext = Path(name).suffix.lower()
                if ext not in ALL_IMAGE_EXTS:
                    skipped_zip_entries += 1
                    continue
                stem = Path(name).stem
                if ext in HEIC_EXTS:
                    # Keep a distinct stem from both regular images and other HEICs.
                    taken = used_stems | {
                        p.stem.lower()
                        for p in images.iterdir()
                        if p.is_file() and p.suffix.lower() in HEIC_EXTS
                    }
                    dst = _unique_path(images, stem, ext, taken)
                else:
                    dst = _unique_path(images, stem, ext, used_stems)
                with zf.open(info) as src_f, open(dst, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                unzipped += 1
        zpath.unlink()
        zips_removed += 1

    # Convert every HEIC/HEIF file to JPEG without overwriting existing stems.
    converted_heic = 0
    heics = sorted(
        p for p in images.iterdir()
        if p.is_file() and p.suffix.lower() in HEIC_EXTS
    )
    for src in heics:
        dst = _unique_path(images, src.stem, ".jpg", used_stems)
        convert_heic(src, dst)
        if not (dst.exists() and dst.stat().st_size > 0):
            raise RuntimeError(f"conversion produced empty output for {src.name}")
        src.unlink()
        converted_heic += 1

    # Sync manifest.csv: preserve existing rows, append any new image files.
    manifest = root / "manifest.csv"
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    if manifest.exists():
        with open(manifest, newline="") as fh:
            for rec in csv.DictReader(fh):
                if not rec.get("photo"):
                    continue
                row = {col: rec.get(col, "") for col in MANIFEST_COLUMNS}
                rows.append(row)
                seen.add(row["photo"])

    added = 0
    image_files = sorted(
        (p for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name.lower(),
    )
    for img in image_files:
        if img.name in seen:
            continue
        rows.append({col: (img.name if col == "photo" else "") for col in MANIFEST_COLUMNS})
        seen.add(img.name)
        added += 1

    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    return {
        "zips_removed": zips_removed,
        "unzipped": unzipped,
        "skipped_zip_entries": skipped_zip_entries,
        "converted_heic": converted_heic,
        "added_to_manifest": added,
        "total_photos": len(image_files),
    }
