from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# Data model
# -----------------------------
@dataclass
class DartAnn:
    bbox: Optional[Tuple[int, int, int, int]] = None
    tip: Optional[Tuple[int, int]] = None
    tail: Optional[Tuple[int, int]] = None


@dataclass
class Item:
    img_path: Path
    ann_path: Path


@dataclass
class State:
    paused: bool = True  # reviewer is always paused; you manually advance
    idx: int = 0
    items: List[Item] = field(default_factory=list)

    frame: Optional[np.ndarray] = None
    w: int = 0
    h: int = 0

    darts: List[DartAnn] = field(default_factory=list)
    current: int = 0

    # mouse & drawing
    mouse_xy: Optional[Tuple[int, int]] = None
    drawing: bool = False
    drag_start: Optional[Tuple[int, int]] = None
    drag_end: Optional[Tuple[int, int]] = None

    # modes
    tip_mode: bool = False
    tail_mode: bool = False

    # dirty tracking / confirmation
    dirty: bool = False
    pending_confirm_key: Optional[int] = None  # e.g. ord(' ') or ord('q')
    message: str = ""
    msg_ttl: int = 0  # frames


# -----------------------------
# Helpers
# -----------------------------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def normalize_bbox(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int
) -> Optional[Tuple[int, int, int, int]]:
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 0, h - 1)

    if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
        return None
    return x1, y1, x2, y2


def point_in_bbox(x: int, y: int, b: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = b
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def ensure_current_dart(st: State) -> None:
    if not st.darts:
        st.darts.append(DartAnn())
        st.current = 0
    st.current = clamp(st.current, 0, len(st.darts) - 1)


def add_new_dart(st: State) -> None:
    st.darts.append(DartAnn())
    st.current = len(st.darts) - 1
    st.tip_mode = False
    st.tail_mode = False
    st.drawing = False
    st.drag_start = None
    st.drag_end = None
    st.dirty = True


def delete_current_dart(st: State) -> None:
    if not st.darts:
        return
    st.darts.pop(st.current)
    if st.current >= len(st.darts):
        st.current = max(0, len(st.darts) - 1)
    st.tip_mode = False
    st.tail_mode = False
    st.drawing = False
    st.drag_start = None
    st.drag_end = None
    st.dirty = True


def reset_labels(st: State) -> None:
    st.darts = []
    st.current = 0
    st.tip_mode = False
    st.tail_mode = False
    st.drawing = False
    st.drag_start = None
    st.drag_end = None
    st.dirty = True


def load_ann(path: Path) -> List[DartAnn]:
    """
    Supports both formats:

    NEW:
      { ..., "darts": [ { "bbox": [...], "tip": [...], "tail": [...] }, ... ] }

    OLD (v1):
      { ..., "bbox": [...], "tip": [...]}   # single dart, no darts[]
    """
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    # New format
    if "darts" in data and isinstance(data.get("darts"), list):
        darts: List[DartAnn] = []
        for d in data.get("darts", []) or []:
            bbox = d.get("bbox")
            tip = d.get("tip")
            tail = d.get("tail")
            darts.append(
                DartAnn(
                    bbox=tuple(bbox) if bbox else None,
                    tip=tuple(tip) if tip else None,
                    tail=tuple(tail) if tail else None,
                )
            )
        return darts

    # Old format (single dart at top-level)
    bbox = data.get("bbox")
    tip = data.get("tip")
    tail = data.get("tail")
    if bbox is None and tip is None and tail is None:
        return []

    return [
        DartAnn(
            bbox=tuple(bbox) if bbox else None,
            tip=tuple(tip) if tip else None,
            tail=tuple(tail) if tail else None,
        )
    ]


def save_ann(path: Path, img_path: Path, frame_idx: int, w: int, h: int, darts: List[DartAnn]) -> None:
    darts_out = []
    for d in darts:
        if d.bbox is None:
            continue
        x1, y1, x2, y2 = d.bbox
        darts_out.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "tip": [int(d.tip[0]), int(d.tip[1])] if d.tip else None,
                "tail": [int(d.tail[0]), int(d.tail[1])] if d.tail else None,
            }
        )

    ann = {
        "image": img_path.name,
        "frame_idx": int(frame_idx),
        "w": int(w),
        "h": int(h),
        "darts": darts_out,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ann, indent=2), encoding="utf-8")


def set_msg(st: State, msg: str, ttl: int = 90) -> None:
    st.message = msg
    st.msg_ttl = ttl


# -----------------------------
# Drawing
# -----------------------------
def _draw_kp(out: np.ndarray, x: int, y: int, label: str, color: Tuple[int, int, int], scale: float = 0.55) -> None:
    h, w = out.shape[:2]
    x = clamp(x, 0, w - 1)
    y = clamp(y, 0, h - 1)

    KP_BOX = 10
    x1 = clamp(x - KP_BOX, 0, w - 1)
    y1 = clamp(y - KP_BOX, 0, h - 1)
    x2 = clamp(x + KP_BOX, 0, w - 1)
    y2 = clamp(y + KP_BOX, 0, h - 1)

    cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
    cv2.circle(out, (x, y), 2, color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    pad = 2
    bx1 = clamp(x2 + 4, 0, w - 1)
    by1 = clamp(y1, 0, h - 1)
    bx2 = clamp(bx1 + tw + 2 * pad, 0, w - 1)
    by2 = clamp(by1 + th + 2 * pad, 0, h - 1)
    cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
    cv2.putText(out, label, (bx1 + pad, by1 + th + pad - 1), font, scale, color, thick, cv2.LINE_AA)


def _draw_panel_bottom_left(out: np.ndarray, lines: List[str]) -> None:
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2
    line_h = 20
    pad = 10

    max_tw = 0
    for s in lines:
        (tw, _), _ = cv2.getTextSize(s, font, scale, thick)
        max_tw = max(max_tw, tw)

    panel_w = max_tw + pad * 2
    panel_h = len(lines) * line_h + pad * 2

    x1 = 10
    y2 = h - 10
    y1 = max(0, y2 - panel_h)
    x2 = min(w - 10, x1 + panel_w)

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    y = y1 + pad + line_h - 6
    for s in lines:
        cv2.putText(out, s, (x1 + pad, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h


def _draw_toast(out: np.ndarray, msg: str) -> None:
    if not msg:
        return
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thick = 2
    (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
    pad = 10
    x1 = 10
    y1 = 10
    x2 = min(w - 10, x1 + tw + 2 * pad)
    y2 = y1 + th + 2 * pad

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    cv2.putText(out, msg, (x1 + pad, y1 + th + pad), font, scale, (0, 255, 255), thick, cv2.LINE_AA)


def draw_overlay(img: np.ndarray, st: State) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    ensure_current_dart(st)

    if st.drawing and st.drag_start and st.drag_end:
        nb = normalize_bbox(st.drag_start[0], st.drag_start[1], st.drag_end[0], st.drag_end[1], w, h)
        if nb is not None:
            x1, y1, x2, y2 = nb
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    for i, d in enumerate(st.darts):
        if d.bbox is not None:
            x1, y1, x2, y2 = d.bbox
            is_cur = (i == st.current)
            col = (0, 255, 0) if not is_cur else (0, 200, 255)
            th = 2 if not is_cur else 3
            cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
            cv2.putText(out, f"dart {i+1}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        if d.tip is not None:
            _draw_kp(out, d.tip[0], d.tip[1], f"{i+1}", (55, 55, 255), scale=0.3)
        if d.tail is not None:
            _draw_kp(out, d.tail[0], d.tail[1], f"{i+1}", (55, 255, 55), scale=0.3)

    item = st.items[st.idx]
    hud = [
        f"{st.idx+1}/{len(st.items)}  {item.img_path.name}  {'*' if st.dirty else ''}",
        f"ann: {item.ann_path}",
        f"darts: {len(st.darts)}  current: {st.current+1 if st.darts else 0}",
        f"mode: {'TIP' if st.tip_mode else ('TAIL' if st.tail_mode else 'BBOX')}",
        "SPACE next | b prev | s save | z save-zero+next | x reject+next | q quit",
        "a add | d del | [ ] switch | click bbox select",
        "t tip-mode (LMB/RMB sets tip) | e tail-mode (LMB sets tail)",
        "drag LMB draws bbox (when not in tip/tail mode)",
    ]
    _draw_panel_bottom_left(out, hud)

    if st.msg_ttl > 0:
        _draw_toast(out, st.message)
        st.msg_ttl -= 1

    return out


# -----------------------------
# Dataset scan
# -----------------------------
def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def scan_dataset_roots(data_root: Path) -> List[Item]:
    """
    Preferred layout:
      annotations/data/<session>/<cam>/images/*.jpg
      annotations/data/<session>/<cam>/ann/*.json

    Also supports flat layout:
      annotations/data/<session>/<cam>/*.jpg + *.json
    """
    items: List[Item] = []
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}

    if not data_root.exists():
        return items

    # Prefer scanning for ".../images/*.jpg"
    for img_path in sorted(data_root.rglob("images/*")):
        if img_path.suffix.lower() not in img_exts:
            continue
        # Skip anything already rejected (paranoia)
        if "rejected" in [p.name for p in img_path.parents]:
            continue

        images_dir = img_path.parent              # .../images
        cam_dir = images_dir.parent               # .../<cam>
        ann_dir = cam_dir / "ann"                 # .../<cam>/ann
        ann_path = ann_dir / f"{img_path.stem}.json"
        items.append(Item(img_path=img_path, ann_path=ann_path))

    if items:
        return items

    # Fallback: flat dirs that contain images directly
    for cam_dir in sorted(p for p in data_root.rglob("*") if p.is_dir()):
        if "rejected" in [p.name for p in cam_dir.parents] or cam_dir.name == "rejected":
            continue
        imgs = sorted(p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts)
        if not imgs:
            continue
        for img_path in imgs:
            ann_path = cam_dir / f"{img_path.stem}.json"
            items.append(Item(img_path=img_path, ann_path=ann_path))

    return items


def compute_rejected_paths(img_path: Path, ann_path: Path, data_root: Path, rejected_root: Path) -> Tuple[Path, Path]:
    """
    Map:
      annotations/data/<session>/<cam>/images/<file>.jpg
      annotations/data/<session>/<cam>/ann/<file>.json
    -> annotations/rejected/<session>/<cam>/images/<file>.jpg
       annotations/rejected/<session>/<cam>/ann/<file>.json
    """
    # img leaf: .../data/<session>/<cam>/images/file.jpg  => rel = <session>/<cam>/images/file.jpg
    try:
        rel_img = img_path.resolve().relative_to(data_root.resolve())
    except Exception:
        # Fallback: keep last 4 parts if something odd
        rel_img = Path(*img_path.parts[-4:])

    # Replace "images" or flat with proper destination layout
    # We want base = <session>/<cam>
    parts = list(rel_img.parts)
    if len(parts) >= 3 and parts[-2] == "images":
        base_rel = Path(*parts[:-2])  # <session>/<cam>
    else:
        # flat: <session>/<cam>/<file>.jpg
        base_rel = Path(*parts[:-1])

    dst_img_dir = rejected_root / base_rel / "images"
    dst_ann_dir = rejected_root / base_rel / "ann"
    dst_img = dst_img_dir / img_path.name
    dst_ann = dst_ann_dir / ann_path.name
    return dst_img, dst_ann


# -----------------------------
# Main loop
# -----------------------------
def main() -> int:
    DATA_ROOT = Path("annotations/data")
    REJECTED_ROOT = Path("annotations/rejected")

    items = scan_dataset_roots(DATA_ROOT)
    if not items:
        print(f"[ERR] No images found under: {DATA_ROOT}")
        return 1

    st = State(items=items, idx=0)

    win = "review_labels"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    def load_current() -> bool:
        item = st.items[st.idx]
        img = cv2.imread(str(item.img_path))
        if img is None:
            print(f"[WARN] Could not read image: {item.img_path}")
            return False

        st.frame = img
        st.h, st.w = img.shape[:2]

        st.darts = load_ann(item.ann_path)
        st.current = 0
        st.tip_mode = False
        st.tail_mode = False
        st.drawing = False
        st.drag_start = None
        st.drag_end = None
        st.mouse_xy = None
        st.dirty = False
        st.pending_confirm_key = None
        st.message = ""
        st.msg_ttl = 0

        if item.ann_path.exists() and not st.darts:
            set_msg(st, "Loaded ann but 0 darts", ttl=60)
        if not item.ann_path.exists():
            set_msg(st, "No ann file (new frame?)", ttl=60)

        return True

    def select_dart_by_click(x: int, y: int) -> None:
        for i in range(len(st.darts) - 1, -1, -1):
            d = st.darts[i]
            if d.bbox and point_in_bbox(x, y, d.bbox):
                st.current = i
                return

    def mark_dirty():
        st.dirty = True
        st.pending_confirm_key = None

    def require_confirm(keycode: int, action_desc: str) -> bool:
        if not st.dirty:
            return True
        if st.pending_confirm_key == keycode:
            st.pending_confirm_key = None
            return True
        st.pending_confirm_key = keycode
        set_msg(st, f"Unsaved changes — press again to {action_desc}", ttl=120)
        return False

    def jump(delta: int) -> None:
        st.idx = clamp(st.idx + delta, 0, len(st.items) - 1)
        load_current()

    def save_current(darts: List[DartAnn]) -> None:
        item = st.items[st.idx]
        frame_idx = st.idx  # keep simple
        save_ann(item.ann_path, item.img_path, frame_idx, st.w, st.h, darts)
        st.dirty = False
        st.pending_confirm_key = None
        set_msg(st, f"Saved {item.ann_path.name}", ttl=60)
        print(f"[OK] Saved {item.ann_path}")

    def reject_current_and_next() -> None:
        item = st.items[st.idx]

        dst_img, dst_ann = compute_rejected_paths(item.img_path, item.ann_path, DATA_ROOT, REJECTED_ROOT)
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_ann.parent.mkdir(parents=True, exist_ok=True)

        # move image
        try:
            shutil.move(str(item.img_path), str(dst_img))
        except Exception as e:
            print(f"[ERR] Failed moving image: {e}")

        # move or create ann
        if item.ann_path.exists():
            try:
                shutil.move(str(item.ann_path), str(dst_ann))
            except Exception as e:
                print(f"[ERR] Failed moving ann: {e}")
        else:
            # create empty ann in rejected to keep pairs
            save_ann(dst_ann, dst_img, st.idx, st.w, st.h, [])

        print(f"[REJECT] {item.img_path.name} -> {dst_img}")
        set_msg(st, f"Rejected -> {dst_img}", ttl=80)

        # remove from list and stay at same index (now points to next item)
        st.items.pop(st.idx)
        if not st.items:
            print("[DONE] No items left.")
            return
        st.idx = clamp(st.idx, 0, len(st.items) - 1)
        load_current()

    def on_mouse(event, x, y, flags, userdata):
        if st.frame is None:
            return

        st.mouse_xy = (clamp(x, 0, st.w - 1), clamp(y, 0, st.h - 1))
        ensure_current_dart(st)
        cur = st.darts[st.current]

        if event == cv2.EVENT_LBUTTONDOWN and not st.drawing:
            select_dart_by_click(x, y)

        # RMB sets tip always
        if event == cv2.EVENT_RBUTTONDOWN:
            cur.tip = (clamp(x, 0, st.w - 1), clamp(y, 0, st.h - 1))
            mark_dirty()
            return

        # Tail mode: LMB sets tail
        if st.tail_mode and event == cv2.EVENT_LBUTTONDOWN:
            cur.tail = (clamp(x, 0, st.w - 1), clamp(y, 0, st.h - 1))
            mark_dirty()
            return

        # Tip mode: LMB sets tip
        if st.tip_mode and event == cv2.EVENT_LBUTTONDOWN:
            cur.tip = (clamp(x, 0, st.w - 1), clamp(y, 0, st.h - 1))
            mark_dirty()
            return

        # bbox draw
        if (not st.tip_mode) and (not st.tail_mode):
            if event == cv2.EVENT_LBUTTONDOWN:
                st.drawing = True
                st.drag_start = (x, y)
                st.drag_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and st.drawing:
                st.drag_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and st.drawing:
                st.drawing = False
                st.drag_end = (x, y)
                if st.drag_start and st.drag_end:
                    nb = normalize_bbox(
                        st.drag_start[0], st.drag_start[1],
                        st.drag_end[0], st.drag_end[1],
                        st.w, st.h
                    )
                    if nb is not None:
                        cur.bbox = nb
                        mark_dirty()
                st.drag_start = None
                st.drag_end = None

    cv2.setMouseCallback(win, on_mouse)

    if not load_current():
        return 1

    while True:
        assert st.frame is not None
        view = draw_overlay(st.frame, st)
        cv2.imshow(win, view)

        key = cv2.waitKey(10) & 0xFF
        if key in (0, 255):
            continue

        # quit
        if key in (ord("q"), 27):
            if not require_confirm(key, "quit"):
                continue
            break

        # next/prev
        if key == ord(" "):
            if not require_confirm(key, "go next"):
                continue
            jump(+1)
            continue

        if key == ord("b"):
            if not require_confirm(key, "go previous"):
                continue
            jump(-1)
            continue

        # modes
        if key == ord("t"):
            st.tip_mode = not st.tip_mode
            if st.tip_mode:
                st.tail_mode = False
            set_msg(st, f"Mode: {'TIP' if st.tip_mode else 'BBOX'}", ttl=40)
            continue

        if key == ord("e"):
            st.tail_mode = not st.tail_mode
            if st.tail_mode:
                st.tip_mode = False
            set_msg(st, f"Mode: {'TAIL' if st.tail_mode else 'BBOX'}", ttl=40)
            continue

        # edit
        if key == ord("a"):
            add_new_dart(st)
            set_msg(st, "Added dart", ttl=40)
            continue

        if key == ord("d"):
            delete_current_dart(st)
            set_msg(st, "Deleted dart", ttl=40)
            continue

        if key == ord("["):
            if st.darts:
                st.current = (st.current - 1) % len(st.darts)
            continue

        if key == ord("]"):
            if st.darts:
                st.current = (st.current + 1) % len(st.darts)
            continue

        if key == ord("r"):
            reset_labels(st)
            set_msg(st, "Reset labels (not saved)", ttl=60)
            continue

        # save
        if key == ord("s"):
            save_current(st.darts)
            continue

        # save-zero + next
        if key == ord("z"):
            save_current([])  # darts: []
            jump(+1)
            continue

        # reject + next
        if key == ord("x"):
            if not require_confirm(key, "reject this frame"):
                continue
            reject_current_and_next()
            continue

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())