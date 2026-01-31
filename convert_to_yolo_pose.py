from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def place_file(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        try:
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def safe_name(s: str) -> str:
    # keep it filesystem friendly
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in s)


def derive_session_cam(ann_path: Path, src_root: Path) -> Tuple[str, str]:
    """
    ann_path: <src_root>/<session>/<cam>/ann/<file>.json
    """
    rel = ann_path.relative_to(src_root)
    # rel parts: [session, cam, 'ann', filename.json]
    if len(rel.parts) >= 4:
        session = rel.parts[0]
        cam = rel.parts[1]
        return session, cam
    # fallback (shouldn't happen)
    return "unknown_session", "unknown_cam"


def write_yolo_label_pose2(
    ann_path: Path,
    label_dst: Path,
) -> None:
    """
    Writes YOLO-Pose labels for 2 keypoints: tip + tail.

    - empty frames -> write empty file (0 lines)
    - missing kp -> (0,0,0)
    - requires bbox to exist for each object line
    """
    try:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception:
        # broken JSON -> treat as empty (still write empty label file)
        label_dst.write_text("", encoding="utf-8")
        return

    w = float(ann.get("w", 0) or 0)
    h = float(ann.get("h", 0) or 0)
    if w <= 0 or h <= 0:
        # can't normalize; write empty label to avoid crashes
        label_dst.write_text("", encoding="utf-8")
        return

    darts = ann.get("darts", [])
    # empty frames must produce an empty label file
    if not darts:
        label_dst.write_text("", encoding="utf-8")
        return

    lines: List[str] = []

    for d in darts:
        bbox = d.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(float, bbox)

        # bbox -> normalized cx,cy,w,h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h

        # tip kp
        tip = d.get("tip")
        if tip and len(tip) == 2:
            tx, ty = map(float, tip)
            tipx = tx / w
            tipy = ty / h
            tipv = 2
        else:
            tipx, tipy, tipv = 0.0, 0.0, 0

        # tail kp
        tail = d.get("tail")
        if tail and len(tail) == 2:
            sx, sy = map(float, tail)
            tailx = sx / w
            taily = sy / h
            tailv = 2
        else:
            tailx, taily, tailv = 0.0, 0.0, 0

        # class 0 = dart
        lines.append(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
            f"{tipx:.6f} {tipy:.6f} {tipv} "
            f"{tailx:.6f} {taily:.6f} {tailv}"
        )

    # If no valid bboxes, still write empty label (negative sample)
    label_dst.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="annotations/data", help="Input root (default: annotations/data)")
    ap.add_argument("--dst", default="datasets/yolo_dart_pose_tip_tail", help="Output YOLO dataset folder")
    ap.add_argument("--val", type=float, default=0.1, help="Validation split fraction (default 0.1)")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    ap.add_argument("--copy", action="store_true", help="Copy images (default is hardlink when possible)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"[ERR] Source not found: {src}")
        return 1

    # Collect (img_path, ann_path)
    pairs: List[Tuple[Path, Path]] = []

    for ann_path in src.rglob("ann/*.json"):
        # IMPORTANT: rejected lives elsewhere: annotations/rejected
        # If user passes a broader src in the future, guard anyway:
        if "rejected" in ann_path.parts:
            continue

        cam_dir = ann_path.parent.parent  # <session>/<cam>
        img_dir = cam_dir / "images"
        img_path = find_image(img_dir, ann_path.stem)
        if img_path is None:
            continue

        pairs.append((img_path, ann_path))

    if not pairs:
        print("[ERR] No image/ann pairs found.")
        return 2

    random.seed(args.seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_val = int(round(n_total * args.val))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    def convert_split(split: str, split_pairs: List[Tuple[Path, Path]]):
        for img_path, ann_path in split_pairs:
            session, cam = derive_session_cam(ann_path, src)
            uniq = f"{safe_name(session)}__{safe_name(cam)}__{ann_path.stem}"

            out_img = dst / "images" / split / f"{uniq}{img_path.suffix.lower()}"
            out_lbl = dst / "labels" / split / f"{uniq}.txt"

            place_file(img_path, out_img, copy=args.copy)
            write_yolo_label_pose2(ann_path, out_lbl)

    convert_split("train", train_pairs)
    convert_split("val", val_pairs)

    # dataset.yaml
    yaml = (
        f"path: {dst.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
        f"  0: dart\n\n"
        f"kpt_shape: [2, 3]\n"
    )
    (dst / "dataset.yaml").write_text(yaml, encoding="utf-8")

    print("[OK] Converted to YOLO pose (tip+tail) with empty frames included.")
    print(f"Total: {n_total}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"YAML:  {dst / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())