from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# 20 unlabeled points in order p00..p19
KPT_COUNT = 20


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
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in s)


def derive_session_cam(ann_path: Path, src_root: Path) -> Tuple[str, str]:
    """
    ann_path: <src_root>/<session>/<cam>/ann/<file>.json
    """
    rel = ann_path.relative_to(src_root)
    if len(rel.parts) >= 4:
        return rel.parts[0], rel.parts[1]
    return "unknown_session", "unknown_cam"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def norm_bbox_xyxy(bbox: List[float], w: float, h: float) -> Tuple[float, float, float, float] | None:
    """
    bbox [x1,y1,x2,y2] in pixels -> (cx, cy, bw, bh) normalized.
    """
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = map(float, bbox)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    x1 = clamp(x1, 0.0, w - 1.0)
    x2 = clamp(x2, 0.0, w - 1.0)
    y1 = clamp(y1, 0.0, h - 1.0)
    y2 = clamp(y2, 0.0, h - 1.0)

    if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
        return None

    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    return cx, cy, bw, bh


def _parse_points_list(ann: dict) -> List[List[float] | None]:
    """
    Expect: ann["points"] is a list of length 20: [[x,y], null, ...]
    We tolerate:
      - shorter/longer list (will pad/truncate)
      - dict-style points (old format) -> not supported; will treat as missing
    """
    pts = ann.get("points", None)

    if isinstance(pts, list):
        out: List[List[float] | None] = []
        for p in pts[:KPT_COUNT]:
            if p and isinstance(p, list) and len(p) == 2:
                out.append([float(p[0]), float(p[1])])
            else:
                out.append(None)
        while len(out) < KPT_COUNT:
            out.append(None)
        return out

    # fallback: missing
    return [None] * KPT_COUNT


def write_yolo_pose20_label(ann_path: Path, label_dst: Path) -> None:
    """
    Writes YOLO-Pose labels for ONE object (board) with 20 keypoints.

    YOLO pose line:
      cls cx cy bw bh kx ky kv ... (20x)

    - If board_bbox missing/invalid -> empty file (negative sample)
    - Missing keypoint -> 0 0 0
    """
    try:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception:
        label_dst.write_text("", encoding="utf-8")
        return

    w = float(ann.get("w", 0) or 0)
    h = float(ann.get("h", 0) or 0)
    if w <= 0 or h <= 0:
        label_dst.write_text("", encoding="utf-8")
        return

    board_bbox = ann.get("board_bbox")
    nb = norm_bbox_xyxy(board_bbox, w, h) if board_bbox is not None else None
    if nb is None:
        label_dst.write_text("", encoding="utf-8")
        return

    cx, cy, bw, bh = nb

    pts = _parse_points_list(ann)

    kp_parts: List[str] = []
    for p in pts:
        if p is not None:
            x, y = p
            kx = x / w
            ky = y / h
            kv = 2  # visible / labeled
        else:
            kx, ky, kv = 0.0, 0.0, 0
        kp_parts.append(f"{kx:.6f} {ky:.6f} {kv}")

    line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp_parts)
    label_dst.write_text(line + "\n", encoding="utf-8")


def write_dataset_yaml(dst: Path) -> None:
    # Horizontal flip for circular ordered points typically reverses index:
    # p00..p19 -> p00 stays opposite? In practice, safest generic mapping is reverse:
    # i -> (K-1-i). This assumes your p00..p19 order is clockwise.
    flip_idx = list(reversed(range(KPT_COUNT)))

    yaml = (
        f"path: {dst.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
        f"  0: board\n\n"
        f"kpt_shape: [{KPT_COUNT}, 3]\n"
        f"flip_idx: {flip_idx}\n"
    )
    (dst / "dataset.yaml").write_text(yaml, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="annotations/calibration", help="Input root (default: annotations/calibration)")
    ap.add_argument("--dst", default="datasets/yolo_board_pose20", help="Output YOLO dataset folder")
    ap.add_argument("--val", type=float, default=0.1, help="Validation split fraction (default 0.1)")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    ap.add_argument("--copy", action="store_true", help="Copy images (default hardlink when possible)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"[ERR] Source not found: {src}")
        return 1

    pairs: List[Tuple[Path, Path]] = []

    for ann_path in src.rglob("ann/*.json"):
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
            write_yolo_pose20_label(ann_path, out_lbl)

    convert_split("train", train_pairs)
    convert_split("val", val_pairs)

    write_dataset_yaml(dst)

    print("[OK] Converted calibration to YOLO pose (board bbox + 20 keypoints).")
    print(f"Total: {n_total}")
    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"YAML:  {dst / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())