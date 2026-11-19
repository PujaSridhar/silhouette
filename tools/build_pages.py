"""Build the published pages from their templates.

Both pages are a template plus a JSON bundle exported from the Python engine.
Keeping the built HTML out of git and regenerating it here means the pages can
never silently drift from the rules and data they claim to implement - the
parity fixture each page checks itself against is exported in the same run.

    python tools/export_demo_data.py          # demo bundle + parity fixture
    PYTHONPATH=tools python tools/export_app_data.py   # app bundle
    python tools/build_pages.py               # inject, write index.html
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

PAGES = [
    ("demo/index.template.html", "demo/engine-data.json", "__ENGINE_DATA__", "demo/index.html"),
    ("app/index.template.html", "demo/app-data.json", "__APP_DATA__", "app/index.html"),
]


def build_one(tpl_rel: str, data_rel: str, token: str, out_rel: str) -> int:
    tpl_p, data_p, out_p = ROOT / tpl_rel, ROOT / data_rel, ROOT / out_rel
    if not data_p.exists():
        raise SystemExit(
            f"{data_rel} is missing - run the exporters first (see this file's docstring)"
        )
    tpl, data = tpl_p.read_text(), data_p.read_text()
    if token not in tpl:
        raise SystemExit(f"{tpl_rel} has no {token} placeholder")
    # A closing tag inside the JSON would break out of the <script> block.
    if "</script" in data.lower():
        raise SystemExit(f"{data_rel} contains a closing script tag")
    out = tpl.replace(token, data)
    out_p.write_text(out)
    return len(out)


def main() -> int:
    for tpl, data, token, out in PAGES:
        size = build_one(tpl, data, token, out)
        print(f"  {out:<22} {size // 1024:>4} KB")
    return 0


if __name__ == "__main__":
    print("built:")
    sys.exit(main())
