# relabel_datav2.py
from __future__ import annotations

import json
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import shutil


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
    img_path: Path          # source image
    ann_path_src: Path
    img_path_dst: Path      # datav2 image (marker)
    ann_path_dst: Path


@dataclass
class State:
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

    # view
    hide_others: bool = False  # H toggles: show only current dart

    # message / toast
    message: str = ""
    msg_ttl: int = 0

    # copy-from-previous (stored on successful save)
    prev_darts: List[DartAnn] = field(default_factory=list)


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
    """
    For this tool we do NOT auto-create a dart on load (we load tips-only from src).
    But once user starts interacting, ensure there's a current dart if list is empty.
    """
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


def reset_labels(st: State) -> None:
    st.darts = []
    st.current = 0
    st.tip_mode = False
    st.tail_mode = False
    st.drawing = False
    st.drag_start = None
    st.drag_end = None


def set_msg(st: State, msg: str, ttl: int = 90) -> None:
    st.message = msg
    st.msg_ttl = ttl


def validate_for_save(darts: List[DartAnn]) -> Tuple[bool, str]:
    """
    Save gate:
      - Every dart must have BOTH bbox and tip.
      - Also blocks saving if darts list is empty (prevents accidental SPACE).
    """
    # if not darts:
    #     return False, "No darts to save. Press A to add a dart (or N to skip)."

    for i, d in enumerate(darts):
        idx = i + 1
        if d.bbox is None and d.tip is None and d.tail is None:
            return False, f"Dart {idx} is empty. Delete it (D) or label bbox+tip."
        if d.bbox is None:
            return False, f"Dart {idx}: missing BBOX. Draw bbox before saving."
        if d.tip is None:
            return False, f"Dart {idx}: missing TIP. Set tip before saving."
    return True, ""


# -----------------------------
# I/O: load src ann (any format) -> keep ONLY TIP per dart
# -----------------------------
def load_ann_any_format_keep_only_tip(path: Path) -> List[DartAnn]:
    """
    Supports:
      NEW:
        { ..., "darts": [ { "bbox": [...], "tip": [...], "tail": [...] }, ... ] }
      OLD (v1):
        { ..., "bbox": [...], "tip": [...] }  # single dart (maybe no darts[])

    IMPORTANT (your requested behavior):
      - The number of darts loaded must equal the number of TIPS present.
      - If an entry has no tip, it is skipped (no placeholder empty DartAnn).
    """
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    darts_out: List[DartAnn] = []

    if isinstance(data, dict) and isinstance(data.get("darts"), list):
        for d in data.get("darts", []) or []:
            tip = d.get("tip")
            if tip and isinstance(tip, (list, tuple)) and len(tip) == 2:
                darts_out.append(DartAnn(bbox=None, tip=(int(tip[0]), int(tip[1])), tail=None))
        return darts_out

    tip = data.get("tip") if isinstance(data, dict) else None
    if tip and isinstance(tip, (list, tuple)) and len(tip) == 2:
        return [DartAnn(bbox=None, tip=(int(tip[0]), int(tip[1])), tail=None)]

    return []


def save_ann_v2(path: Path, img_path: Path, frame_idx: int, w: int, h: int, darts: List[DartAnn]) -> None:
    darts_out = []
    for d in darts:
        darts_out.append(
            {
                "bbox": [int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3])] if d.bbox else None,
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
        "format": "datav2",
        "note": "relabel tool: src tips kept, bbox/tail re-labeled",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ann, indent=2), encoding="utf-8")


def mark_done(done_path: Path) -> None:
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text("ok\n", encoding="utf-8")


# -----------------------------
# Drawing helpers
# -----------------------------
def _draw_kp(out: np.ndarray, x: int, y: int, label: str, color: Tuple[int, int, int], scale: float = 0.5) -> None:
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


def _draw_loupe(out: np.ndarray, src: np.ndarray, center_xy: Tuple[int, int], title: str) -> None:
    """
    Zoomed loupe that FOLLOWS the mouse cursor.
    - Samples a patch around center_xy from src
    - Draws the zoomed patch near the cursor (clamped inside the frame)
    """
    H, W = out.shape[:2]
    cx, cy = center_xy

    half = 28
    zoom = 4

    sx1 = clamp(cx - half, 0, W - 1)
    sy1 = clamp(cy - half, 0, H - 1)
    sx2 = clamp(cx + half, 0, W - 1)
    sy2 = clamp(cy + half, 0, H - 1)

    patch = src[sy1 : sy2 + 1, sx1 : sx2 + 1]
    if patch.size == 0:
        return

    patch_big = cv2.resize(
        patch,
        (patch.shape[1] * zoom, patch.shape[0] * zoom),
        interpolation=cv2.INTER_NEAREST,
    )
    ph, pw = patch_big.shape[:2]

    margin = 10
    offset_x = 20
    offset_y = 20

    ox1 = cx + offset_x
    oy1 = cy + offset_y

    if ox1 + pw + margin > W:
        ox1 = cx - offset_x - pw
    if oy1 + ph + margin > H:
        oy1 = cy - offset_y - ph

    ox1 = clamp(ox1, margin, max(margin, W - pw - margin))
    oy1 = clamp(oy1, margin + 28, max(margin + 28, H - ph - margin))
    ox2 = ox1 + pw
    oy2 = oy1 + ph

    overlay = out.copy()
    cv2.rectangle(overlay, (ox1 - 4, oy1 - 28), (ox2 + 4, oy2 + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    out[oy1:oy2, ox1:ox2] = patch_big

    cv2.rectangle(out, (ox1 - 1, oy1 - 1), (ox2, oy2), (255, 255, 255), 1)

    mx = ox1 + pw // 2
    my = oy1 + ph // 2
    cv2.line(out, (mx, oy1), (mx, oy2 - 1), (255, 255, 255), 1)
    cv2.line(out, (ox1, my), (ox2 - 1, my), (255, 255, 255), 1)
    cv2.circle(out, (mx, my), 4, (0, 255, 255), 1)

    cv2.putText(
        out,
        title,
        (ox1, oy1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_overlay(img: np.ndarray, st: State) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    # show bbox preview while dragging
    if st.drawing and st.drag_start and st.drag_end:
        nb = normalize_bbox(st.drag_start[0], st.drag_start[1], st.drag_end[0], st.drag_end[1], w, h)
        if nb is not None:
            x1, y1, x2, y2 = nb
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    idxs = [st.current] if (st.hide_others and st.darts) else list(range(len(st.darts)))

    for i in idxs:
        d = st.darts[i]

        if d.bbox is not None:
            x1, y1, x2, y2 = d.bbox
            is_cur = (i == st.current)
            col = (0, 255, 0) if not is_cur else (0, 200, 255)
            th = 2 if not is_cur else 3
            cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
            cv2.putText(out, f"dart {i+1}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        if d.tip is not None:
            _draw_kp(out, d.tip[0], d.tip[1], f"T{i+1}", (55, 55, 255), scale=0.45)
        if d.tail is not None:
            _draw_kp(out, d.tail[0], d.tail[1], f"S{i+1}", (55, 255, 55), scale=0.45)

    item = st.items[st.idx]
    hud = [
        f"{st.idx+1}/{len(st.items)}  {item.img_path.name}",
        f"SRC: {item.ann_path_src}",
        f"DST: {item.ann_path_dst}",
        f"darts: {len(st.darts)}  current: {st.current+1 if st.darts else 0}  hide_others(H): {st.hide_others}",
        f"mode: {'TIP' if st.tip_mode else ('TAIL' if st.tail_mode else 'BBOX')}",
        "SPACE save->datav2 + next | b prev | n next(no save) | q quit",
        "a add | d del | [ ] switch | click bbox select | C copy prev | H toggle show-only-current",
        "t tip-mode (LMB/RMB sets tip) | e tail-mode (LMB sets tail)",
        "drag LMB draws bbox (when not in tip/tail mode)",
        "SAVE RULE: ALL darts must have bbox + tip",
    ]
    _draw_panel_bottom_left(out, hud)

    if st.mouse_xy is not None:
        if st.tip_mode:
            _draw_loupe(out, img, st.mouse_xy, "LOUPE: TIP (LMB/RMB)")
        elif st.tail_mode:
            _draw_loupe(out, img, st.mouse_xy, "LOUPE: TAIL (LMB)")

    if st.msg_ttl > 0:
        _draw_toast(out, st.message)
        st.msg_ttl -= 1

    return out


# -----------------------------
# Dataset scan & mapping
# -----------------------------
def scan_items_datav2(src_root: Path, dst_root: Path) -> List[Item]:
    items: List[Item] = []
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}

    if not src_root.exists():
        return items

    for img_path in sorted(src_root.rglob("images/*")):
        if img_path.suffix.lower() not in img_exts:
            continue
        if "rejected" in [p.name for p in img_path.parents]:
            continue

        cam_dir = img_path.parent.parent
        ann_src = cam_dir / "ann" / f"{img_path.stem}.json"

        rel = img_path.resolve().relative_to(src_root.resolve())
        parts = list(rel.parts)
        base_rel = Path(*parts[:-2])

        img_dst = dst_root / base_rel / "images" / img_path.name
        ann_dst = dst_root / base_rel / "ann" / f"{img_path.stem}.json"

        if img_dst.exists() or ann_dst.exists():
            continue

        items.append(
            Item(
                img_path=img_path,
                ann_path_src=ann_src,
                img_path_dst=img_dst,
                ann_path_dst=ann_dst,
            )
        )

    if items:
        return items

    for cam_dir in sorted(p for p in src_root.rglob("*") if p.is_dir()):
        if "rejected" in [p.name for p in cam_dir.parents] or cam_dir.name == "rejected":
            continue
        imgs = sorted(p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts)
        if not imgs:
            continue

        for img_path in imgs:
            ann_src = cam_dir / f"{img_path.stem}.json"
            rel = img_path.resolve().relative_to(src_root.resolve())
            base_rel = Path(*rel.parts[:-1])

            img_dst = dst_root / base_rel / "images" / img_path.name
            ann_dst = dst_root / base_rel / "ann" / f"{img_path.stem}.json"

            if img_dst.exists() or ann_dst.exists():
                continue

            items.append(
                Item(
                    img_path=img_path,
                    ann_path_src=ann_src,
                    img_path_dst=img_dst,
                    ann_path_dst=ann_dst,
                )
            )

    return items


# -----------------------------
# Main loop
# -----------------------------
def main() -> int:
    SRC_ROOT = Path("annotations/data")
    DST_ROOT = Path("annotations/datav2")

    items = scan_items_datav2(SRC_ROOT, DST_ROOT)
    if not items:
        print(f"[OK] No pending items. (Either none found under {SRC_ROOT}, or everything already processed into {DST_ROOT})")
        return 0

    st = State(items=items, idx=0)

    win = "relabel_datav2"
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

        # Load SRC annotations but KEEP ONLY TIPS (clear bbox/tail)
        # Number of darts = number of tips
        st.darts = load_ann_any_format_keep_only_tip(item.ann_path_src)

        st.current = 0

        # start in BBOX mode
        st.tip_mode = False
        st.tail_mode = False

        st.drawing = False
        st.drag_start = None
        st.drag_end = None
        st.mouse_xy = None

        if st.darts:
            set_msg(st, f"Loaded {len(st.darts)} tips from src. Start: BBOX mode. SPACE=save+next", ttl=80)
        else:
            set_msg(st, "Loaded 0 tips from src. Press A to add dart if needed. SPACE will not save.", ttl=110)

        return True

    def select_dart_by_click(x: int, y: int) -> None:
        if st.hide_others:
            return
        for i in range(len(st.darts) - 1, -1, -1):
            d = st.darts[i]
            if d.bbox and point_in_bbox(x, y, d.bbox):
                st.current = i
                return

    def jump(delta: int) -> None:
        st.idx = clamp(st.idx + delta, 0, len(st.items) - 1)
        load_current()

    def save_and_mark_done_then_next() -> None:
        ok, why = validate_for_save(st.darts)
        if not ok:
            set_msg(st, why, ttl=140)
            return

        item = st.items[st.idx]

        save_ann_v2(
            item.ann_path_dst,
            item.img_path,
            st.idx,
            st.w,
            st.h,
            st.darts,
        )

        item.img_path_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.img_path, item.img_path_dst)

        st.prev_darts = copy.deepcopy(st.darts)
        set_msg(st, "SAVED -> datav2 (image + ann)", ttl=55)

        # ADVANCE
        st.items.pop(st.idx)
        if not st.items:
            return
        st.idx = clamp(st.idx, 0, len(st.items) - 1)
        load_current()

    def copy_prev_to_current() -> None:
        if not st.prev_darts:
            set_msg(st, "No previous labels to copy yet", ttl=60)
            return
        st.darts = copy.deepcopy(st.prev_darts)
        st.current = 0
        st.drawing = False
        st.drag_start = None
        st.drag_end = None
        set_msg(st, "Copied labels from previous (C)", ttl=60)

    def toggle_hide_others() -> None:
        st.hide_others = not st.hide_others
        if st.hide_others:
            set_msg(st, "H: show ONLY current dart (others hidden & unclickable)", ttl=90)
        else:
            set_msg(st, "H: show ALL darts", ttl=60)

    def on_mouse(event, x, y, flags, userdata):
        if st.frame is None:
            return

        st.mouse_xy = (clamp(x, 0, st.w - 1), clamp(y, 0, st.h - 1))
        ensure_current_dart(st)
        cur = st.darts[st.current]

        if event == cv2.EVENT_LBUTTONDOWN and not st.drawing:
            select_dart_by_click(x, y)

        # RMB sets tip always (fast)
        if event == cv2.EVENT_RBUTTONDOWN:
            cur.tip = (clamp(x, 0, st.w - 1), clamp(y, 0, st.h - 1))
            return

        if st.tail_mode and event == cv2.EVENT_LBUTTONDOWN:
            cur.tail = (clamp(x, 0, st.w - 1), clamp(y, 0, st.h - 1))
            return

        if st.tip_mode and event == cv2.EVENT_LBUTTONDOWN:
            cur.tip = (clamp(x, 0, st.w - 1), clamp(y, 0, st.h - 1))
            return

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

        if key in (ord("q"), 27):
            break

        # SPACE = save to datav2 + next (guarded)
        if key == ord(" "):
            save_and_mark_done_then_next()
            if not st.items:
                break
            continue

        if key == ord("b"):
            jump(-1)
            continue
        if key == ord("n"):
            jump(+1)
            continue

        if key in (ord("c"), ord("C")):
            copy_prev_to_current()
            continue

        if key in (ord("h"), ord("H")):
            toggle_hide_others()
            continue

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

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
