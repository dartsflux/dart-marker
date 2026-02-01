from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
RE_FRAME = re.compile(r"(\d+)$")  # captures trailing digits in stem, e.g. frame_000123 -> 000123


@dataclass(frozen=True)
class Pair:
    img: Path
    ann: Path
    session: str
    cam: str
    frame_num: int
    stem: str


@dataclass(frozen=True)
class Group:
    session: str
    cam: str
    items: List[Pair]  # length == group_size (usually)


def parse_frame_num(stem: str) -> Optional[int]:
    """
    Extract trailing integer from stem.
    Works for: frame_000123, img_12, 000456, etc.
    """
    m = RE_FRAME.search(stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def scan_pairs(src_root: Path) -> List[Pair]:
    """
    Expects:
      annotations/data/<session>/<cam>/images/*.jpg
      annotations/data/<session>/<cam>/ann/*.json

    Returns pairs sorted by (session, cam, frame_num, filename) for stable ordering.
    """
    pairs: List[Pair] = []
    if not src_root.exists():
        return pairs

    for img_path in src_root.rglob("images/*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        if "rejected" in [p.name for p in img_path.parents]:
            continue

        # derive session/cam from .../<session>/<cam>/images/<file>
        try:
            rel = img_path.relative_to(src_root)
            # rel: <session>/<cam>/images/<file>
            if len(rel.parts) < 4:
                continue
            session = rel.parts[0]
            cam = rel.parts[1]
        except Exception:
            continue

        cam_dir = img_path.parent.parent  # .../<session>/<cam>
        ann_path = cam_dir / "ann" / f"{img_path.stem}.json"

        frame_num = parse_frame_num(img_path.stem)
        if frame_num is None:
            # Still include, but push to end within cam; keep stable by name
            frame_num = 10**12

        pairs.append(
            Pair(
                img=img_path,
                ann=ann_path,
                session=session,
                cam=cam,
                frame_num=frame_num,
                stem=img_path.stem,
            )
        )

    pairs.sort(key=lambda p: (p.session, p.cam, p.frame_num, p.img.name))
    return pairs


def make_groups(pairs: List[Pair], group_size: int) -> List[Group]:
    """
    Group sequential frames per (session, cam) into fixed-size groups.
    IMPORTANT: does NOT mix across sessions/cams.

    Assumption: your "same dart position" frames are contiguous in frame numbering.
    We group in strict order: first N -> group1, next N -> group2 ...
    """
    by_key: Dict[Tuple[str, str], List[Pair]] = {}
    for p in pairs:
        by_key.setdefault((p.session, p.cam), []).append(p)

    groups: List[Group] = []
    for (session, cam) in sorted(by_key.keys()):
        lst = by_key[(session, cam)]
        i = 0
        while i < len(lst):
            chunk = lst[i : i + group_size]
            # keep remainder as smaller last group (still kept together)
            groups.append(Group(session=session, cam=cam, items=chunk))
            i += group_size

    return groups


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, do_copy: bool) -> None:
    ensure_dir(dst.parent)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="annotations/data", help="Source root (v1)")
    ap.add_argument("--dst", default="annotations/buckets_v1", help="Destination buckets root")
    ap.add_argument("--bucket-images", type=int, default=300, help="Approx images per bucket (will not split groups)")
    ap.add_argument("--group-size", type=int, default=3, help="Images per 'position' group (default 3)")
    ap.add_argument("--copy", action="store_true", help="Copy instead of move")
    ap.add_argument("--dry-run", action="store_true", help="No writes, just print plan")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    group_size = max(1, int(args.group_size))

    pairs = scan_pairs(src_root)
    if not pairs:
        print(f"[ERR] No images found under: {src_root}")
        return 1

    groups = make_groups(pairs, group_size=group_size)

    # Bucket capacity in *groups* so we never split a group.
    bucket_images = max(group_size, int(args.bucket_images))
    cap_groups = max(1, bucket_images // group_size)
    # If bucket_images not divisible by group_size, we effectively round down.
    eff_bucket_images = cap_groups * group_size

    print(f"[OK] Found pairs: {len(pairs)}")
    print(f"[OK] Group size: {group_size} -> groups: {len(groups)}")
    print(f"[OK] Bucket target: ~{bucket_images} images => {cap_groups} groups (~{eff_bucket_images} images) per bucket")
    print(f"[OK] Mode: {'COPY' if args.copy else 'MOVE'} | Dry-run: {args.dry_run}")
    print(f"[OK] SRC: {src_root}")
    print(f"[OK] DST: {dst_root}")

    b = 1
    g_in_bucket = 0

    def bucket_name(bi: int) -> str:
        return f"bucket{bi:03d}"

    for gi, grp in enumerate(groups):
        if g_in_bucket >= cap_groups:
            b += 1
            g_in_bucket = 0

        bname = bucket_name(b)

        # Write into: buckets_v1/bucketXYZ/<session>/<cam>/images + ann
        for p in grp.items:
            out_img = dst_root / bname / p.session / p.cam / "images" / p.img.name
            out_ann = dst_root / bname / p.session / p.cam / "ann" / f"{p.stem}.json"

            if args.dry_run:
                print(f"{bname}: {p.session}/{p.cam} frame={p.frame_num}  {p.img.name}")
                continue

            copy_or_move(p.img, out_img, do_copy=args.copy)
            if p.ann.exists():
                copy_or_move(p.ann, out_ann, do_copy=args.copy)

        g_in_bucket += 1

    print("[DONE] Bucketing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
