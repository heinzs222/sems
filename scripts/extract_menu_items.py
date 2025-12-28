#!/usr/bin/env python3
"""
Extract menu items + prices from `menu.html`.

This reads the Uber Eats-style HTML export and writes a clean JSON file that is
easy to inspect and safe to feed into the agent.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.menu import load_menu_from_html, resolve_menu_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", default="menu.html", help="Path to menu.html")
    parser.add_argument(
        "--out",
        dest="output_path",
        default="menu_items.json",
        help="Output JSON path (default: menu_items.json)",
    )
    args = parser.parse_args()

    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    menu_path = resolve_menu_path(args.input_path)
    if not menu_path.exists():
        print(f"[ERR] Menu file not found: {menu_path}")
        return 1

    catalog = load_menu_from_html(menu_path)

    try:
        source = str(menu_path.relative_to(project_root))
    except ValueError:
        source = str(menu_path)

    payload = {
        "source": source,
        "num_sections": len(catalog.sections),
        "num_items": len(catalog.items),
        "sections": [
            {
                "name": section,
                "items": [
                    {
                        "name": item.name,
                        "price": item.price_text,
                        "price_cents": item.price_cents,
                    }
                    for item in catalog.sections[section]
                ],
            }
            for section in sorted(catalog.sections.keys())
        ],
    }

    out_path = Path(args.output_path)
    if not out_path.is_absolute():
        out_path = project_root / out_path

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {out_path} ({payload['num_items']} items, {payload['num_sections']} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
