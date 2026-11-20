#!/usr/bin/env python3
"""A local rating page for the empty volume/structure/formality columns.

    python tools/rate_wardrobe.py          # then open the URL it prints

Binds 127.0.0.1 only and has no network calls in it or in the page: the
photos are hers, and nothing here uploads them anywhere. Each answer is
written into data/wardrobe_real.csv the moment it is saved, so the session
can be stopped at any point and picked up later. Skipping a piece leaves its
row empty, which is how it stays in the queue.

Crops come from the person masks in photos/, which is gitignored and stays
that way.
"""

from __future__ import annotations

import csv
import io
import json
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REAL = ROOT / "data" / "wardrobe_real.csv"
REFERENCE = ROOT / "data" / "wardrobe.csv"
PHOTOS = ROOT / "photos"
PAGE = Path(__file__).with_name("rate_wardrobe.html")

#: The scales, as words. Judging "is this Relaxed or Loose" is a question a
#: stylist can answer off a photo; judging "is this 1.2 or 1.4" is the same
#: question wearing a disguise, and invites false precision. The number is
#: what gets stored; the word is what gets asked.
VOLUME_SCALES = {
    "default": [("Fitted", 0.85), ("Close", 1.0), ("Relaxed", 1.2),
                ("Loose", 1.4), ("Oversized", 1.65)],
    "bottom": [("Skinny", 0.85), ("Straight", 1.1), ("Relaxed", 1.4),
               ("Wide", 1.85), ("Very wide", 2.1)],
}
#: Only the draped pieces are asked for a second, lower width.
VOLUME_LOWER_SCALE = [("Narrow", 1.1), ("A-line", 1.4), ("Full", 1.9)]
STRUCTURE_SCALE = [("Soft", 0.15), ("Medium", 0.4), ("Stiff", 0.75)]
FORMALITY_SCALE = [("Casual", 0.25), ("Smart", 0.55), ("Dressy", 0.85)]


def volume_scale(category: str) -> list[tuple[str, float]]:
    """Bottoms get their own words; everything else uses the default set.

    Shoes and the dupatta fall through to the default scale, which was
    written for tops, outerwear and dresses. Its top end ("Oversized", 1.65)
    does not mean much for a shoe, so those six pieces are the ones to look
    at again if their numbers come out odd.
    """
    return VOLUME_SCALES["bottom" if category == "bottom" else "default"]


def _steps(scale) -> list[dict]:
    return [{"label": w, "value": v} for w, v in scale]


#: Fractions of crown->sole to crop for each category, so the zone the volume
#: question is about is what she actually sees.
ZONES = {
    "top": (0.14, 0.58), "outer": (0.12, 0.62), "drape": (0.12, 0.75),
    "bottom": (0.42, 1.02), "dress": (0.10, 1.02), "shoe": (0.86, 1.04),
}
RANK = {"high": 0, "medium": 1, "low": 2}

_lock = threading.Lock()


# --- data -------------------------------------------------------------------

def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _fieldnames(path: Path) -> list[str]:
    with open(path, newline="") as fh:
        return next(csv.reader(fh))


def best_photos() -> dict[str, list[str]]:
    """garment id -> photos showing it, best first."""
    out: dict[str, list[str]] = {}
    draft = PHOTOS / "garments_draft.csv"
    if not draft.exists():
        return out
    rows = sorted(_rows(draft), key=lambda r: RANK.get(r.get("confidence", ""), 3))
    for r in rows:
        for gid in (r.get("garments") or "").split(";"):
            if gid.strip():
                out.setdefault(gid.strip(), []).append(r["photo"])
    return out


def anchors(category: str) -> list[dict]:
    """Three judged pieces from the reference wardrobe, low/middle/high.

    Spanning the range rather than taking the first three is the point: they
    are there to pin the scale, so they have to bracket it.
    """
    same = [r for r in _rows(REFERENCE)
            if r["category"] == category and (r.get("volume") or "").strip()]
    if not same:
        return []
    same.sort(key=lambda r: float(r["volume"]))
    picks = [same[0], same[len(same) // 2], same[-1]] if len(same) >= 3 else same
    seen, out = set(), []
    for r in picks:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append({"id": r["id"], "name": r["name"], "volume": r["volume"],
                    "volume_lower": r.get("volume_lower", ""),
                    "structure": r.get("structure", ""),
                    "formality": r.get("formality", ""),
                    "color_hex": r.get("color_hex", "#808080")})
    return out


def items() -> list[dict]:
    photos = best_photos()
    out = []
    for r in _rows(REAL):
        shots = photos.get(r["id"], [])
        out.append({
            "volume_steps": _steps(volume_scale(r["category"])),
            "volume_lower_steps": _steps(VOLUME_LOWER_SCALE),
            "structure_steps": _steps(STRUCTURE_SCALE),
            "formality_steps": _steps(FORMALITY_SCALE),
            "id": r["id"], "name": r["name"], "category": r["category"],
            "subcategory": r.get("subcategory", ""),
            "fit": r.get("fit", ""), "length": r.get("length", ""),
            "fabric_weight": r.get("fabric_weight", ""), "drape": r.get("drape", ""),
            "color_hex": r.get("color_hex", "#808080"),
            # volume_lower is only meaningful where one piece covers both
            # zones at different widths, which here means the draped pieces.
            "two_part": r["category"] == "dress",
            "photos": shots,
            "volume": (r.get("volume") or "").strip(),
            "volume_lower": (r.get("volume_lower") or "").strip(),
            "structure": (r.get("structure") or "").strip(),
            "formality": (r.get("formality") or "").strip(),
            "anchors": anchors(r["category"]),
        })
    # A piece photographed more than once is the one worth judging first: a
    # second angle is the only check available on whether the first was
    # misleading, and those answers then anchor the single-photo ones.
    out.sort(key=lambda i: (len(i["photos"]) < 2))
    return out


def save(gid: str, values: dict) -> dict:
    allowed = {"volume", "volume_lower", "structure", "formality"}
    bad = set(values) - allowed
    if bad:
        raise ValueError(f"not a ratable column: {sorted(bad)}")
    for k, v in values.items():
        if v == "":
            continue
        try:
            float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{k} must be a number, got {v!r}") from None
    with _lock:
        fields = _fieldnames(REAL)
        rows = _rows(REAL)
        hit = next((r for r in rows if r["id"] == gid), None)
        if hit is None:
            raise KeyError(gid)
        hit.update({k: str(v) for k, v in values.items()})
        tmp = REAL.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(REAL)          # atomic, so an interrupted save cannot truncate
        done = sum(1 for r in rows if (r.get("volume") or "").strip())
        return {"ok": True, "id": gid, "rated": done, "total": len(rows)}


# --- crops ------------------------------------------------------------------

def crop(gid: str, category: str, photo: str) -> bytes:
    import numpy as np
    from PIL import Image

    from silhouette.vision.photos import open_rgb

    stem = Path(photo).stem
    mask_path = PHOTOS / "masks" / stem / "person.png"
    img = open_rgb(PHOTOS / "images" / photo)
    H, W = img.shape[:2]
    lo, hi = ZONES.get(category, (0.0, 1.0))
    if mask_path.exists():
        m = np.asarray(Image.open(mask_path).convert("L")) > 127
        ys, _ = np.where(m)
        crown, sole = ys.min(), ys.max()
        span = sole - crown
        y0, y1 = int(crown + lo * span), int(crown + hi * span)
        y0, y1 = max(0, y0), min(H, max(y0 + 8, y1))
        band = m[y0:y1]
        xs = np.where(band.any(0))[0]
        if len(xs):
            pad = max(8, int(0.16 * (xs.max() - xs.min())))
            x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
        else:
            x0, x1 = 0, W
    else:
        y0, y1, x0, x1 = 0, H, 0, W
    im = Image.fromarray(img[y0:y1, x0:x1])
    im.thumbnail((900, 900))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


# --- server -----------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # the page is self-contained; no third party is allowed to load anything
        self.send_header("Content-Security-Policy",
                         "default-src 'self' 'unsafe-inline'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/items":
                return self._json({"items": items()})
            if path.startswith("/crop/"):
                gid = path[len("/crop/"):].removesuffix(".jpg")
                it = next((i for i in items() if i["id"] == gid), None)
                if not it or not it["photos"]:
                    return self._json({"error": "no photo for this piece"}, 404)
                which = urlparse(self.path).query
                idx = 0
                if which.startswith("n="):
                    idx = max(0, min(len(it["photos"]) - 1, int(which[2:] or 0)))
                return self._send(200, crop(gid, it["category"], it["photos"][idx]),
                                  "image/jpeg")
            self._json({"error": "not found"}, 404)
        except Exception as e:  # a broken photo must not take the page down
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        if unquote(urlparse(self.path).path) != "/api/save":
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            gid = body.pop("id", "")
            self._json(save(gid, body))
        except (KeyError, ValueError) as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def _free_port(preferred: int = 8765) -> int:
    for port in (preferred, 0):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            return s.getsockname()[1]
        except OSError:
            continue
        finally:
            s.close()
    raise SystemExit("no free port on 127.0.0.1")


def main() -> int:
    if not REAL.exists():
        raise SystemExit(f"{REAL} is missing")
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    rows = _rows(REAL)
    todo = sum(1 for r in rows if not (r.get("volume") or "").strip())
    print(f"{todo} of {len(rows)} pieces still need a volume")
    print(f"rating page: {url}   (127.0.0.1 only, nothing leaves this machine)")
    print("Ctrl-C when you are done; every answer is already saved.")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
