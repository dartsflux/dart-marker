from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# Data model
# -----------------------------
Point = Optional[Tuple[int, int]]


@dataclass
class CalibAnn:
    # bbox around board (x1,y1,x2,y2) optional but recommended
    board_bbox: Optional[Tuple[int, int, int, int]] = None

    # 20 outer wire intersection points (order is your click order / selection order)
    points: List[Point] = field(default_factory=lambda: [None] * 20)


@dataclass
class Toast:
    text: str = ""
    until_ms: int = 0


@dataclass
class LabelState:
    mouse_xy: Optional[Tuple[int, int]] = None
    paused: bool = False
    frame_idx: int = 0
    frame: Optional[np.ndarray] = None

    ann: CalibAnn = field(default_factory=CalibAnn)

    # drag interaction (for bbox modes)
    drawing: bool = False
    drag_start: Optional[Tuple[int, int]] = None
    drag_end: Optional[Tuple[int, int]] = None

    # current edit mode:
    # 0 = point placement
    # 1 = draw board bbox
    mode: int = 0

    # point selection
    point_sel: int = 0  # 0..19

    toast: Toast = field(default_factory=Toast)


# Display-only names for HUD / overlay
POINT_NAMES = [f"p{i:02d}" for i in range(20)]
REQUIRED_COUNT = 20

MODE_NAMES = ["POINTS", "BOARD_BBOX"]


# -----------------------------
# Time / Toast helpers
# -----------------------------
def _now_ms() -> int:
    return int((cv2.getTickCount() / cv2.getTickFrequency()) * 1000.0)


def set_toast(st: LabelState, text: str, duration_ms: int = 900) -> None:
    st.toast.text = text
    st.toast.until_ms = _now_ms() + int(duration_ms)


def toast_active(st: LabelState) -> bool:
    return bool(st.toast.text) and (_now_ms() <= st.toast.until_ms)


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


def ensure_dirs(base_dir: Path) -> Tuple[Path, Path]:
    img_dir = base_dir / "images"
    ann_dir = base_dir / "ann"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, ann_dir


def derive_session_cam(video_path: Path) -> Tuple[str, str]:
    session = video_path.parent.name
    cam = video_path.stem
    return session, cam


def compute_output_dir(video_path: Path, out_data_root: Path) -> Path:
    session, cam = derive_session_cam(video_path)
    return out_data_root / session / cam


def set_point(ann: CalibAnn, idx: int, xy: Tuple[int, int]) -> None:
    ann.points[idx] = xy


def count_points_set(ann: CalibAnn) -> int:
    return sum(1 for p in ann.points if p is not None)


def all_required_points_set(ann: CalibAnn) -> bool:
    return count_points_set(ann) >= REQUIRED_COUNT


def reset_all(st: LabelState) -> None:
    st.ann = CalibAnn()
    st.drawing = False
    st.drag_start = None
    st.drag_end = None
    st.mode = 0
    st.point_sel = 0


# -----------------------------
# Drawing
# -----------------------------
def _draw_hud_bottom_left(out: np.ndarray, lines: List[str]) -> None:
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
    x2 = x1 + panel_w

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    y = y1 + pad + line_h - 6
    for s in lines:
        cv2.putText(out, s, (x1 + pad, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h


def _draw_toast_top(out: np.ndarray, text: str) -> None:
    if not text:
        return
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thick = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad_x = 18
    pad_y = 10
    box_w = tw + 2 * pad_x
    box_h = th + 2 * pad_y

    x1 = (w - box_w) // 2
    y1 = 18
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    tx = x1 + pad_x
    ty = y1 + pad_y + th
    cv2.putText(out, text, (tx, ty), font, scale, (0, 255, 255), thick, cv2.LINE_AA)


def _safe_crop(img: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    h, w = img.shape[:2]
    cx = clamp(cx, 0, w - 1)
    cy = clamp(cy, 0, h - 1)

    x1 = clamp(cx - r, 0, w - 1)
    x2 = clamp(cx + r, 0, w - 1)
    y1 = clamp(cy - r, 0, h - 1)
    y2 = clamp(cy + r, 0, h - 1)

    crop = img[y1 : y2 + 1, x1 : x2 + 1]

    target = 2 * r + 1
    pad_top = max(0, r - (cy - y1))
    pad_left = max(0, r - (cx - x1))
    pad_bottom = max(0, target - crop.shape[0] - pad_top)
    pad_right = max(0, target - crop.shape[1] - pad_left)

    if pad_top or pad_left or pad_bottom or pad_right:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE
        )

    return crop


def _draw_lens(out: np.ndarray, cx: int, cy: int) -> None:
    h, w = out.shape[:2]

    R = 16
    ZOOM = 4
    BORDER = 2
    PAD = 8
    SHOW_SIZE = (2 * R + 1) * ZOOM

    crop = _safe_crop(out, cx, cy, R)
    zoomed = cv2.resize(crop, (SHOW_SIZE, SHOW_SIZE), interpolation=cv2.INTER_NEAREST)

    mid = SHOW_SIZE // 2
    cv2.line(zoomed, (mid, 0), (mid, SHOW_SIZE - 1), (255, 255, 255), 1)
    cv2.line(zoomed, (0, mid), (SHOW_SIZE - 1, mid), (255, 255, 255), 1)
    cv2.rectangle(
        zoomed,
        (mid - ZOOM // 2, mid - ZOOM // 2),
        (mid + ZOOM // 2, mid + ZOOM // 2),
        (0, 255, 255),
        1,
    )

    lx = cx + 24
    ly = cy - SHOW_SIZE - 24
    if lx + SHOW_SIZE + 2 * BORDER + PAD > w:
        lx = cx - (SHOW_SIZE + 24)
    if ly < PAD:
        ly = cy + 24

    lx = clamp(lx, PAD, max(PAD, w - SHOW_SIZE - 2 * BORDER - PAD))
    ly = clamp(ly, PAD, max(PAD, h - SHOW_SIZE - 2 * BORDER - PAD))

    x1, y1 = lx, ly
    x2, y2 = lx + SHOW_SIZE + 2 * BORDER, ly + SHOW_SIZE + 2 * BORDER

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    out[y1 + BORDER : y1 + BORDER + SHOW_SIZE, x1 + BORDER : x1 + BORDER + SHOW_SIZE] = zoomed

    txt = f"({cx},{cy})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2
    (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
    tx1 = x1
    ty1 = y2 + 6
    tx2 = clamp(tx1 + tw + 10, 0, w - 1)
    ty2 = clamp(ty1 + th + 10, 0, h - 1)
    if ty2 > y2 + 4 and ty2 < h:
        cv2.rectangle(out, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
        cv2.putText(out, txt, (tx1 + 5, ty1 + th + 5), font, scale, (0, 255, 255), thick, cv2.LINE_AA)


def _draw_point(out: np.ndarray, xy: Tuple[int, int], label: str, is_sel: bool) -> None:
    x, y = xy
    col = (0, 200, 255) if is_sel else (0, 255, 0)
    cv2.circle(out, (x, y), 4 if is_sel else 3, col, -1)
    cv2.rectangle(out, (x - 10, y - 10), (x + 10, y + 10), col, 1 if not is_sel else 2)
    cv2.putText(
        out,
        label,
        (x + 12, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        col,
        2,
        cv2.LINE_AA,
    )


def _draw_bbox(out: np.ndarray, bbox: Tuple[int, int, int, int], label: str, color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_overlay(canvas: np.ndarray, st: LabelState, out_dir: Path) -> np.ndarray:
    out = canvas.copy()
    h, w = out.shape[:2]

    # preview bbox while dragging
    if st.mode == 1 and st.drawing and st.drag_start and st.drag_end:
        x1, y1 = st.drag_start
        x2, y2 = st.drag_end
        nb = normalize_bbox(x1, y1, x2, y2, w, h)
        if nb is not None:
            _draw_bbox(out, nb, "board bbox (preview)", (0, 255, 255))

    # draw saved bbox
    if st.ann.board_bbox is not None:
        _draw_bbox(out, st.ann.board_bbox, "board bbox", (255, 255, 0))

    # draw points
    for i, name in enumerate(POINT_NAMES):
        p = st.ann.points[i]
        if p is not None:
            _draw_point(out, p, name, is_sel=(st.mode == 0 and i == st.point_sel))

    # status
    n_set = count_points_set(st.ann)
    ready = (n_set >= REQUIRED_COUNT)

    sel_name = POINT_NAMES[st.point_sel]
    hud_lines = [
        f"Out: {out_dir}",
        f"Frame: {st.frame_idx}",
        f"State: {'PAUSED' if st.paused else 'PLAYING'}",
        f"Mode: {MODE_NAMES[st.mode]}   (p=points, v=board bbox)",
        f"Selected: {st.point_sel+1}/{REQUIRED_COUNT}:{sel_name}  (c cycles)",
        f"Points set: {n_set}/{REQUIRED_COUNT}",
        f"BBox: {'OK' if st.ann.board_bbox is not None else '—'} (optional)",
        "SPACE pause/resume | n next | b back | r reset | s save(10 frames) | q quit",
        ("STATUS: READY TO SAVE" if ready else "STATUS: NEED ALL 20 POINTS"),
        "TIP: Click 20 outer wire intersections in a consistent clockwise order.",
        "SAVE: writes 10 frames: F, F+3, ... F+27 using same annotation.",
    ]
    _draw_hud_bottom_left(out, hud_lines)

    if toast_active(st):
        _draw_toast_top(out, st.toast.text)

    if st.paused and st.mode == 0 and st.mouse_xy is not None:
        mx, my = st.mouse_xy
        _draw_lens(out, mx, my)

    return out


# -----------------------------
# Save
# -----------------------------
def save_calib_one(
    img_dir: Path,
    ann_dir: Path,
    frame: np.ndarray,
    frame_idx: int,
    ann: CalibAnn,
    prefix: str,
) -> Tuple[Path, Path]:
    h, w = frame.shape[:2]
    name = f"{prefix}{frame_idx:06d}"
    img_path = img_dir / f"{name}.jpg"
    ann_path = ann_dir / f"{name}.json"

    cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

    data = {
        "image": img_path.name,
        "frame_idx": int(frame_idx),
        "w": int(w),
        "h": int(h),
        "board_bbox": list(ann.board_bbox) if ann.board_bbox is not None else None,
        "points": [list(p) if p is not None else None for p in ann.points],
        "point_order": POINT_NAMES,
        "notes": "board_bbox optional. points are 20 outer wire intersections in the order you clicked/selected.",
    }
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[OK] Saved {img_path.name} (+ calib ann)")
    return img_path, ann_path


def save_calib_sequence(
    cap: cv2.VideoCapture,
    img_dir: Path,
    ann_dir: Path,
    ann: CalibAnn,
    prefix: str,
    start_idx: int,
    *,
    count: int = 10,
    step: int = 3,
) -> int:
    """
    Save `count` frames starting at `start_idx`, stepping by `step` frames.
    Uses the SAME annotation data for each saved frame.
    Returns number of frames successfully saved.
    """
    saved = 0
    # remember current position so we can restore
    cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    for k in range(count):
        idx = start_idx + k * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        save_calib_one(img_dir, ann_dir, fr, idx, ann, prefix)
        saved += 1

    # restore position (best effort)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)
    return saved


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to mp4")
    ap.add_argument("--out", default="annotations/calibration", help="Output root folder (default: annotations/calibration)")
    ap.add_argument("--start-frame", type=int, default=0, help="Start from this frame index")
    ap.add_argument("--prefix", default="calib_", help="Filename prefix (default: calib_)")

    ap.add_argument("--delay-ms", type=int, default=25, help="Playback delay in ms while playing (default 25)")
    ap.add_argument("--paused-delay-ms", type=int, default=10, help="UI delay in ms while paused (default 10)")

    args = ap.parse_args()

    video_path = Path(args.video).expanduser()
    out_data_root = Path(args.out)
    out_dir = compute_output_dir(video_path, out_data_root)
    img_dir, ann_dir = ensure_dirs(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    st = LabelState()
    win = "label_calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        st.frame_idx = args.start_frame - 1

    def read_next_frame() -> bool:
        ok, fr = cap.read()
        if not ok:
            return False
        st.frame = fr
        st.frame_idx += 1
        return True

    def seek_to(frame_idx: int) -> bool:
        frame_idx = max(0, int(frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        st.frame_idx = frame_idx - 1
        return read_next_frame()

    def on_mouse(event, x, y, flags, userdata):
        if st.frame is None or not st.paused:
            return

        hh, ww = st.frame.shape[:2]
        x = clamp(x, 0, ww - 1)
        y = clamp(y, 0, hh - 1)
        st.mouse_xy = (x, y)

        # MODE 0: POINTS
        if st.mode == 0:
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                set_point(st.ann, st.point_sel, (x, y))
                set_toast(st, f"SET {POINT_NAMES[st.point_sel].upper()} = ({x},{y})", 700)
            return

        # MODE 1: BOARD BBOX DRAW
        if st.mode == 1:
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
                    x1, y1 = st.drag_start
                    x2, y2 = st.drag_end
                    nb = normalize_bbox(x1, y1, x2, y2, ww, hh)
                    if nb is not None:
                        st.ann.board_bbox = nb
                        set_toast(st, "SET BOARD BBOX", 700)
                st.drag_start = None
                st.drag_end = None

    cv2.setMouseCallback(win, on_mouse)

    if not read_next_frame():
        print("Could not read first frame.")
        return 1

    while True:
        assert st.frame is not None
        canvas = draw_overlay(st.frame, st, out_dir)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(args.paused_delay_ms if st.paused else args.delay_ms) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord(" "):
            st.paused = not st.paused
            st.drawing = False
            st.drag_start = None
            st.drag_end = None
            continue

        if st.paused:
            # modes
            if key in (ord("p"), ord("P")):
                st.mode = 0
                st.drawing = False
                set_toast(st, "MODE: POINTS", 800)
                continue

            if key in (ord("v"), ord("V")):
                st.mode = 1
                st.drawing = False
                set_toast(st, "MODE: BOARD BBOX (drag)", 900)
                continue

            # cycle points
            if key == ord("c"):
                st.point_sel = (st.point_sel + 1) % len(POINT_NAMES)
                set_toast(st, f"SELECT {st.point_sel+1}/{REQUIRED_COUNT}:{POINT_NAMES[st.point_sel].upper()}", 700)
                continue

            if key == ord("r"):
                reset_all(st)
                set_toast(st, "RESET", 800)
                continue

            if key == ord("n"):
                if not read_next_frame():
                    print("End of video.")
                    break
                continue

            if key == ord("b"):
                back_to = max(0, st.frame_idx - 1)
                if not seek_to(back_to):
                    print("Cannot seek back.")
                    break
                continue

            if key == ord("s"):
                if not all_required_points_set(st.ann):
                    set_toast(st, "NEED ALL 20 POINTS", 1200)
                    continue

                # Save 10 frames: F, F+3, ... F+27
                saved = save_calib_sequence(
                    cap=cap,
                    img_dir=img_dir,
                    ann_dir=ann_dir,
                    ann=st.ann,
                    prefix=args.prefix,
                    start_idx=st.frame_idx,
                    count=10,
                    step=3,
                )

                # Restore current frame in UI (best effort)
                seek_to(st.frame_idx)

                set_toast(st, f"SAVED {saved}/10 (step=3)", 1200)
                continue

        # playing
        if not st.paused:
            if not read_next_frame():
                print("End of video.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Calibration saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())