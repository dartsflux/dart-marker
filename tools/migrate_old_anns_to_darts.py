#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def migrate_one(p: Path, dry_run: bool) -> bool:
    try:
        ann: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False

    if isinstance(ann.get("darts"), list):
        return False  # already new

    # Old format expected: bbox + tip at top-level (tail not present)
    bbox = ann.get("bbox", None)
    tip = ann.get("tip", None)

    # If we can’t recognize it, skip (don’t wreck unknown formats)
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return False

    dart = {
        "bbox": [int(x) for x in bbox],
        "tip": [int(tip[0]), int(tip[1])] if (isinstance(tip, list) and len(tip) == 2) else None,
        "tail": None,
    }

    ann.pop("bbox", None)
    ann.pop("tip", None)

    ann["darts"] = [dart]
    # keep your newer convention: include empty only if you want; audit tolerates missing
    ann["empty"] = (len(ann["darts"]) == 0)

    if not dry_run:
        p.write_text(json.dumps(ann, indent=2), encoding="utf-8")

    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="annotations/data", help="Root folder to scan")
    ap.add_argument("--dry-run", action="store_true", help="Show what would change, but don’t write")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERR] root not found: {root}")
        return 2

    candidates = sorted(root.rglob("ann/*.json"))
    changed = 0
    for p in candidates:
        if migrate_one(p, args.dry_run):
            changed += 1
            print(f"[OK] migrated: {p}")

    print(f"\nDone. Migrated {changed} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())