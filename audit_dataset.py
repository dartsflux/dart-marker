#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class Issue:
    kind: str
    path: Path
    msg: str


def sha256_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pick_existing_image(img_dir: Path, stem: str) -> Optional[Path]:
    # prefer common ext, but accept any
    for ext in (".jpg", ".jpeg", ".png"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _is_xy(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)


def _is_bbox(v) -> bool:
    return isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v)


def _validate_xy(issues: List[Issue], ann_path: Path, label: str, xy, w: int, h: int) -> None:
    if xy is None:
        return
    if not _is_xy(xy):
        issues.append(Issue("schema", ann_path, f"{label} must be list[2] or null"))
        return
    x, y = int(xy[0]), int(xy[1])
    if not (0 <= x < w and 0 <= y < h):
        issues.append(Issue("range", ann_path, f"{label} out of bounds {label}=({x},{y}) wh=({w},{h})"))


def validate_ann(ann_path: Path, ann: dict) -> List[Issue]:
    issues: List[Issue] = []

    # Required top-level keys (empty is optional now)
    required = ["image", "frame_idx", "w", "h", "darts"]
    for k in required:
        if k not in ann:
            issues.append(Issue("schema", ann_path, f"Missing key '{k}'"))

    # Basic type checks (must establish darts before empty default)
    try:
        w = int(ann.get("w", -1))
        h = int(ann.get("h", -1))
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid w/h: w={w} h={h}")

        darts = ann.get("darts", None)
        if not isinstance(darts, list):
            raise ValueError("darts must be list")

        empty = ann.get("empty", None)
        if empty is None:
            # default based on darts content
            empty = (len(darts) == 0)
        elif not isinstance(empty, bool):
            raise ValueError("empty must be bool")
    except Exception as e:
        issues.append(Issue("schema", ann_path, f"Invalid top-level types: {e}"))
        return issues

    # empty consistency
    if empty and len(darts) != 0:
        issues.append(Issue("schema", ann_path, f"empty=true but darts has {len(darts)} items"))
    if (not empty) and len(darts) == 0:
        issues.append(Issue("schema", ann_path, "empty=false but darts is empty"))

    # Validate each dart
    for i, d in enumerate(darts):
        if not isinstance(d, dict):
            issues.append(Issue("schema", ann_path, f"darts[{i}] must be object"))
            continue

        if "bbox" not in d:
            issues.append(Issue("schema", ann_path, f"darts[{i}] missing 'bbox'"))
            continue

        bbox = d.get("bbox")
        if not _is_bbox(bbox):
            issues.append(Issue("schema", ann_path, f"darts[{i}].bbox must be list[4]"))
            continue

        x1, y1, x2, y2 = map(int, bbox)

        if x2 < x1 or y2 < y1:
            issues.append(Issue("bbox", ann_path, f"darts[{i}].bbox not normalized: {bbox}"))

        if not (0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h):
            issues.append(Issue("range", ann_path, f"darts[{i}].bbox out of bounds: {bbox} wh=({w},{h})"))

        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            issues.append(Issue("bbox_small", ann_path, f"darts[{i}].bbox too small: {bbox}"))

        tip = d.get("tip", None)
        tail = d.get("tail", None)

        _validate_xy(issues, ann_path, f"darts[{i}].tip", tip, w, h)
        _validate_xy(issues, ann_path, f"darts[{i}].tail", tail, w, h)

    return issues


def audit_pair_folder(pair_root: Path, check_dupes: bool) -> Tuple[Dict[str, int], List[Issue], Dict[str, List[Path]]]:
    """
    pair_root is a folder that contains:
      - images/
      - ann/

    Returns:
      counts, issues, dupes
    """
    issues: List[Issue] = []
    dupes: Dict[str, List[Path]] = {}

    img_dir = pair_root / "images"
    ann_dir = pair_root / "ann"

    if not img_dir.exists() or not ann_dir.exists():
        issues.append(Issue("missing_dir", pair_root, "Expected 'images/' and 'ann/'"))
        return {"images": 0, "anns": 0, "pairs": 0}, issues, dupes

    images = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    anns = sorted([p for p in ann_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])

    img_stems = {p.stem for p in images}
    ann_stems = {p.stem for p in anns}

    only_imgs = sorted(img_stems - ann_stems)
    only_anns = sorted(ann_stems - img_stems)

    for stem in only_imgs[:50]:
        issues.append(Issue("missing_ann", img_dir / (stem + ".jpg"), f"Image '{stem}' has no .json"))
    if len(only_imgs) > 50:
        issues.append(Issue("missing_ann_more", pair_root, f"... and {len(only_imgs)-50} more images missing anns"))

    for stem in only_anns[:50]:
        issues.append(Issue("missing_img", ann_dir / (stem + ".json"), f"Ann '{stem}' has no image"))
    if len(only_anns) > 50:
        issues.append(Issue("missing_img_more", pair_root, f"... and {len(only_anns)-50} more anns missing images"))

    pairs = sorted(img_stems & ann_stems)

    for stem in pairs:
        img_path = pick_existing_image(img_dir, stem)
        if img_path is None:
            issues.append(Issue("missing_img", pair_root, f"Stem {stem} has ann but no supported image ext"))
            continue

        ann_path = ann_dir / f"{stem}.json"
        try:
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception as e:
            issues.append(Issue("bad_json", ann_path, str(e)))
            continue

        issues.extend(validate_ann(ann_path, ann))

        if check_dupes:
            try:
                digest = sha256_file(img_path)
                dupes.setdefault(digest, []).append(img_path)
            except Exception as e:
                issues.append(Issue("hash_fail", img_path, str(e)))

    counts = {"images": len(images), "anns": len(anns), "pairs": len(pairs)}
    return counts, issues, dupes


def find_pair_folders(root: Path) -> List[Path]:
    """
    Find any directory under root that contains both 'images' and 'ann' subfolders.
    Returns the parent directories (pair_root).
    """
    pair_roots: List[Path] = []
    if not root.exists():
        return pair_roots

    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        if (d / "images").is_dir() and (d / "ann").is_dir():
            pair_roots.append(d)

    return sorted(set(pair_roots))


def format_group_name(pair_root: Path, base_root: Path) -> str:
    rel = pair_root.relative_to(base_root)
    return str(rel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="annotations/data",
        help="Root to scan (default: annotations/data). Scans recursively for folders containing images/ and ann/.",
    )
    ap.add_argument("--dupes", action="store_true", help="Hash images to find duplicates (slower)")
    ap.add_argument(
        "--include-rejected",
        action="store_true",
        help="Also scan annotations/rejected (same recursive logic)",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERR] root does not exist: {root}")
        return 2

    pair_roots = find_pair_folders(root)
    extra_pair_roots: List[Path] = []

    rejected_root = root.parent / "rejected"  # annotations/rejected
    if args.include_rejected and rejected_root.exists():
        extra_pair_roots = find_pair_folders(rejected_root)

    all_pair_roots = sorted(pair_roots + extra_pair_roots)

    print(f"[INFO] Root: {root.resolve()}")
    print(f"[INFO] Found {len(pair_roots)} datasets with images/ + ann/ under data/")
    if args.include_rejected:
        print(f"[INFO] Found {len(extra_pair_roots)} datasets with images/ + ann/ under rejected/")
    print()

    if not all_pair_roots:
        print("[WARN] No folders containing both images/ and ann/ were found.")
        return 1

    total_images = total_anns = total_pairs = 0
    all_issues: List[Issue] = []
    all_dupes: Dict[str, List[Path]] = {}

    for pr in all_pair_roots:
        base = rejected_root if (args.include_rejected and rejected_root in pr.parents) else root
        label = format_group_name(pr, base)

        counts, issues, dupes = audit_pair_folder(pr, args.dupes)
        total_images += counts["images"]
        total_anns += counts["anns"]
        total_pairs += counts["pairs"]
        all_issues.extend(issues)

        if args.dupes:
            for k, v in dupes.items():
                all_dupes.setdefault(k, []).extend(v)

        print(f"{label}: images={counts['images']} anns={counts['anns']} pairs={counts['pairs']} issues={len(issues)}")

    print("\n=== TOTAL ===")
    print(f"images: {total_images}")
    print(f"anns:   {total_anns}")
    print(f"pairs:  {total_pairs}")

    if all_issues:
        print("\n=== ISSUES (first 50) ===")
        for it in all_issues[:50]:
            print(f"- [{it.kind}] {it.path}: {it.msg}")
        if len(all_issues) > 50:
            print(f"... and {len(all_issues) - 50} more")
        return 1

    if args.dupes:
        dup_groups = [(k, v) for k, v in all_dupes.items() if len(v) > 1]
        if dup_groups:
            print("\n=== DUPLICATE IMAGES (sha256) ===")
            dup_groups.sort(key=lambda kv: len(kv[1]), reverse=True)
            for digest, paths in dup_groups[:20]:
                print(f"{digest[:12]}... count={len(paths)}")
                for p in paths[:10]:
                    print(f"  - {p}")
                if len(paths) > 10:
                    print(f"  ... and {len(paths) - 10} more")
            print(f"\n[WARN] Found {len(dup_groups)} duplicate groups.")
        else:
            print("\n[OK] No duplicate images detected (by sha256).")

    print("\n[OK] Dataset looks consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())