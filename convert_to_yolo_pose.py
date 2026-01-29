from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple


IMG_EXTS = [".jpg", ".jpeg", ".png"]


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to datasets/dart_tip_kpt")
    ap.add_argument("--dst", default="datasets/yolo_dart_tip_pose", help="Output dataset folder")
    ap.add_argument("--val", type=float, default=0.1, help="Validation split fraction (default 0.1)")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    ap.add_argument("--copy", action="store_true", help="Copy images (default). If false, will hardlink when possible.")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    pairs: List[Tuple[Path, Path]] = []  # (img_path, ann_path)

    sessions = sorted([p for p in src.iterdir() if p.is_dir()])
    for s in sessions:
        img_dir = s / "images"
        ann_dir = s / "ann"
        if not img_dir.exists() or not ann_dir.exists():
            continue
        for ann_path in ann_dir.glob("*.json"):
            stem = ann_path.stem
            img_path = find_image(img_dir, stem)
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

    def prepare_dirs(split: str):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    prepare_dirs("train")
    prepare_dirs("val")

    def place_file(src_path: Path, dst_path: Path):
        if dst_path.exists():
            return
        if not args.copy:
            try:
                dst_path.hardlink_to(src_path)
                return
            except Exception:
                pass
        shutil.copy2(src_path, dst_path)

    def write_label(img_dst: Path, ann_path: Path, label_dst: Path):
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        w = float(ann["w"])
        h = float(ann["h"])
        x1, y1, x2, y2 = map(float, ann["bbox"])
        tx, ty = map(float, ann["tip"])

        # bbox -> normalized cx,cy,w,h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h

        # tip -> normalized x,y + visibility=2
        kx = tx / w
        ky = ty / h
        v = 2

        # class 0 = dart
        line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kx:.6f} {ky:.6f} {v}\n"
        label_dst.write_text(line, encoding="utf-8")

    def convert_split(split: str, split_pairs: List[Tuple[Path, Path]]):
        for img_path, ann_path in split_pairs:
            out_img = dst / "images" / split / img_path.name
            out_lbl = dst / "labels" / split / (ann_path.stem + ".txt")

            place_file(img_path, out_img)
            write_label(out_img, ann_path, out_lbl)

    convert_split("train", train_pairs)
    convert_split("val", val_pairs)

    # dataset.yaml
    yaml = (
        f"path: {dst.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: dart\n"
        f"kpt_shape: [1, 3]\n"
    )
    (dst / "dataset.yaml").write_text(yaml, encoding="utf-8")

    print("[OK] Converted.")
    print(f"Total: {n_total}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"YAML:  {dst / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
