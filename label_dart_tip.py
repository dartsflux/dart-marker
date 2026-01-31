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
@dataclass
class DartAnn:
    # bbox stored as (x1, y1, x2, y2) in pixels, normalized so x1<x2,y1<y2
    bbox: Optional[Tuple[int, int, int, int]] = None
    # keypoints stored as (x, y) in pixels
    tip: Optional[Tuple[int, int]] = None
    tail: Optional[Tuple[int, int]] = None


@dataclass
class Toast:
    text: str = ""
    until_ms: int = 0  # absolute ms timestamp (monotonic-ish via tickcount)


@dataclass
class LabelState:
    mouse_xy: Optional[Tuple[int, int]] = None  # tracked while paused for lens
    paused: bool = False
    frame_idx: int = 0
    frame: Optional[np.ndarray] = None

    darts: List[DartAnn] = field(default_factory=list)
    current: int = 0  # index into darts

    # drawing interaction
    drawing: bool = False
    drag_start: Optional[Tuple[int, int]] = None
    drag_end: Optional[Tuple[int, int]] = None

    # modes (when paused)
    tip_mode: bool = False   # if True, next LMB sets tip
    tail_mode: bool = False  # if True, next LMB sets tail

    # debug / hud
    last_motion_score: float = 0.0

    # UI toggles
    hide_others: bool = False  # hide non-current darts (H toggles)

    # toast
    toast: Toast = field(default_factory=Toast)


# Keep last saved annotations to allow copy-forward
last_saved_darts: list[DartAnn] = []


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


def clone_darts(darts: list[DartAnn]) -> list[DartAnn]:
    out: list[DartAnn] = []
    for d in darts:
        out.append(DartAnn(bbox=d.bbox, tip=d.tip, tail=d.tail))
    return out


def normalize_bbox(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int
) -> Optional[Tuple[int, int, int, int]]:
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 0, h - 1)

    # Reject zero-area / tiny boxes (usually accidental clicks)
    if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
        return None

    return x1, y1, x2, y2


def point_in_bbox(x: int, y: int, b: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = b
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def parse_roi(s: str) -> Optional[Tuple[int, int, int, int]]:
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = [int(v) for v in parts]
    return (x1, y1, x2, y2)


def apply_roi(gray: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if roi is None:
        return gray
    x1, y1, x2, y2 = roi

    x1 = clamp(x1, 0, gray.shape[1] - 1)
    x2 = clamp(x2, 1, gray.shape[1])
    y1 = clamp(y1, 0, gray.shape[0] - 1)
    y2 = clamp(y2, 1, gray.shape[0])

    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return gray
    return gray[y1:y2, x1:x2]


def ensure_dirs(base_dir: Path) -> Tuple[Path, Path]:
    """
    Creates:
      base_dir/images
      base_dir/ann
    """
    img_dir = base_dir / "images"
    ann_dir = base_dir / "ann"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, ann_dir


def derive_session_cam(video_path: Path) -> Tuple[str, str]:
    """
    Typical input:
      .../videos/20260128_024058/cam1.mp4
    => session='20260128_024058', cam='cam1'

    We use:
      session = parent folder name
      cam = file stem
    """
    session = video_path.parent.name
    cam = video_path.stem
    return session, cam


def compute_output_dir(video_path: Path, out_data_root: Path) -> Path:
    """
    New layout:
      annotations/data/{session}/{cam}/
    """
    session, cam = derive_session_cam(video_path)
    return out_data_root / session / cam


def ensure_current_dart(st: LabelState) -> None:
    """Ensure there is at least one dart and current index is valid."""
    if not st.darts:
        st.darts.append(DartAnn())
        st.current = 0
    st.current = clamp(st.current, 0, len(st.darts) - 1)


def add_new_dart(st: LabelState) -> None:
    st.darts.append(DartAnn())
    st.current = len(st.darts) - 1
    st.tip_mode = False
    st.tail_mode = False
    st.drawing = False
    st.drag_start = None
    st.drag_end = None


def delete_current_dart(st: LabelState) -> None:
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


def reset_frame_labels(st: LabelState) -> None:
    st.darts = []
    st.current = 0
    st.tip_mode = False
    st.tail_mode = False
    st.drawing = False
    st.drag_start = None
    st.drag_end = None


def ready_to_save(st: LabelState, allow_empty: bool) -> Tuple[bool, str]:
    """
    Return (ok, message).
    If allow_empty=True, you can save frames with 0 darts (negative samples).
    Otherwise require at least one bbox; tip/tail optional.
    """
    if not st.darts:
        return (True, "READY (EMPTY)") if allow_empty else (False, "NEED: at least one dart bbox")

    bad = [i for i, d in enumerate(st.darts) if d.bbox is None]
    if bad:
        return False, f"NEED: bbox for dart(s) {', '.join(str(i+1) for i in bad)}"

    return True, "READY TO SAVE"


def save_annot(
    img_dir: Path,
    ann_dir: Path,
    frame: np.ndarray,
    frame_idx: int,
    darts: List[DartAnn],
    prefix: str,
) -> Tuple[Path, Path, bool, int]:
    """
    Returns (img_path, ann_path, empty, darts_count)
    """
    h, w = frame.shape[:2]

    name = f"{prefix}{frame_idx:06d}"
    img_path = img_dir / f"{name}.jpg"
    ann_path = ann_dir / f"{name}.json"

    cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

    darts_out = []
    for d in darts:
        if d.bbox is None:
            continue
        x1, y1, x2, y2 = d.bbox
        obj = {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "tip": [int(d.tip[0]), int(d.tip[1])] if d.tip is not None else None,
            "tail": [int(d.tail[0]), int(d.tail[1])] if d.tail is not None else None,
        }
        darts_out.append(obj)

    empty = (len(darts_out) == 0)

    ann = {
        "image": img_path.name,
        "frame_idx": int(frame_idx),
        "w": int(w),
        "h": int(h),
        "empty": empty,
        "darts": darts_out,
    }
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_path.write_text(json.dumps(ann, indent=2), encoding="utf-8")
    print(f"[OK] Saved {img_path.name} (+ ann, darts={len(darts_out)}, empty={empty})")
    return img_path, ann_path, empty, len(darts_out)


# -----------------------------
# Drawing / HUD
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
    thick = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    pad = 3
    bx1 = clamp(x2 + 4, 0, w - 1)
    by1 = clamp(y1, 0, h - 1)
    bx2 = clamp(bx1 + tw + 2 * pad, 0, w - 1)
    by2 = clamp(by1 + th + 2 * pad, 0, h - 1)
    cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
    cv2.putText(out, label, (bx1 + pad, by1 + th + pad - 1), font, scale, color, thick, cv2.LINE_AA)


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


def draw_overlay(
    canvas: np.ndarray,
    st: LabelState,
    out_dir: Path,
    roi: Optional[Tuple[int, int, int, int]],
    allow_empty: bool,
) -> np.ndarray:
    out = canvas.copy()
    h, w = out.shape[:2]

    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        rx1 = clamp(rx1, 0, w - 1)
        rx2 = clamp(rx2, 0, w - 1)
        ry1 = clamp(ry1, 0, h - 1)
        ry2 = clamp(ry2, 0, h - 1)
        if abs(rx2 - rx1) >= 2 and abs(ry2 - ry1) >= 2:
            cv2.rectangle(out, (min(rx1, rx2), min(ry1, ry2)), (max(rx1, rx2), max(ry1, ry2)), (255, 255, 0), 2)
            cv2.putText(
                out,
                "ROI (motion)",
                (min(rx1, rx2), max(0, min(ry1, ry2) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

    if st.darts or st.tip_mode or st.tail_mode or st.drawing:
        ensure_current_dart(st)

    if st.drawing and st.drag_start and st.drag_end:
        x1, y1 = st.drag_start
        x2, y2 = st.drag_end
        nb = normalize_bbox(x1, y1, x2, y2, w, h)
        if nb is not None:
            x1, y1, x2, y2 = nb
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    for i, d in enumerate(st.darts):
        if st.hide_others and i != st.current:
            continue

        if d.bbox is not None:
            x1, y1, x2, y2 = d.bbox
            is_cur = (i == st.current)
            col = (0, 255, 0) if not is_cur else (0, 200, 255)
            th = 2 if not is_cur else 3
            cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
            cv2.putText(out, f"dart {i+1}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        if d.tip is not None:
            tx, ty = d.tip
            _draw_kp(out, tx, ty, f"{i+1}", (55, 255, 55), 0.4)

        if d.tail is not None:
            sx, sy = d.tail
            _draw_kp(out, sx, sy, f"{i+1}", (255, 55, 55), 0.4)

    ok, status = ready_to_save(st, allow_empty=allow_empty)

    hud_lines = [
        f"Out: {out_dir}",
        f"Frame: {st.frame_idx}",
        f"State: {'PAUSED' if st.paused else 'PLAYING'}",
        f"Darts: {len(st.darts)} | Current: {st.current+1 if st.darts else 0}",
        f"Hide others: {'ON' if st.hide_others else 'OFF'} (H toggles)",
        f"Tip mode: {'ON' if st.tip_mode else 'OFF'} (t toggles; LMB sets) | RMB sets tip",
        f"Tail mode: {'ON' if st.tail_mode else 'OFF'} (e toggles; LMB sets)",
        f"Motion: {st.last_motion_score:.2f}%" if st.last_motion_score > 0 else "Motion: -",
        "SPACE pause/resume | n next | b back | a add | d del | k copy | h hide",
        "[ / ] prev/next | click bbox selects | r reset | s save(+extras) | x save empty | q quit",
        f"STATUS: {status}",
    ]
    _draw_hud_bottom_left(out, hud_lines)

    if toast_active(st):
        _draw_toast_top(out, st.toast.text)

    if st.paused and (st.tip_mode or st.tail_mode) and st.mouse_xy is not None:
        mx, my = st.mouse_xy
        _draw_lens(out, mx, my)

    return out


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    global last_saved_darts

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to mp4")
    ap.add_argument("--out", default="annotations/data", help="Output root folder (default: annotations/data)")
    ap.add_argument("--start-frame", type=int, default=0, help="Start from this frame index")
    ap.add_argument("--prefix", default="frame_", help="Filename prefix (default: frame_)")

    ap.add_argument("--after-count", type=int, default=2, help="How many extra frames to save after current (default 2)")
    ap.add_argument("--after-step", type=int, default=4, help="Frame step between extras (default 4)")

    ap.add_argument("--delay-ms", type=int, default=25, help="Playback delay in ms while playing (default 25)")
    ap.add_argument("--paused-delay-ms", type=int, default=10, help="UI delay in ms while paused (default 10)")

    ap.add_argument("--auto-pause", action="store_true", help="Auto-pause when motion settles (dart likely stuck)")
    ap.add_argument("--stable-frames", type=int, default=10, help="Consecutive stable frames before pausing (default 10)")

    ap.add_argument("--downscale", type=float, default=0.33, help="Downscale ROI for motion metric (default 0.33)")
    ap.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel (odd). 0 disables. (default 5)")
    ap.add_argument("--diff-px", type=int, default=12, help="Per-pixel diff threshold for 'changed' (default 12)")
    ap.add_argument("--motion-frac", type=float, default=0.008, help="Start event if changed fraction >= this (default 0.008)")
    ap.add_argument("--stable-frac", type=float, default=0.002, help="Stable if changed fraction <= this (default 0.002)")

    ap.add_argument("--roi", type=str, default="", help="ROI x1,y1,x2,y2 for motion (recommended)")
    ap.add_argument("--cooldown-frames", type=int, default=20, help="Cooldown after autopause (default 20)")
    ap.add_argument("--motion-debug", action="store_true", help="Print per-frame motion debug logs")

    ap.add_argument("--allow-empty", action="store_true", help="(kept for compatibility) allow saving empty frames")
    ap.add_argument("--save-empty-key", default="x", help="Key to save an empty frame quickly (default: x)")

    args = ap.parse_args()
    roi = parse_roi(args.roi)

    # You asked for empty saving support; make it ON by default.
    allow_empty = True

    video_path = Path(args.video).expanduser()
    out_data_root = Path(args.out)
    out_dir = compute_output_dir(video_path, out_data_root)
    img_dir, ann_dir = ensure_dirs(out_dir)

    event_active = False
    stable_count = 0
    cooldown = 0
    prev_gray_roi: Optional[np.ndarray] = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    st = LabelState()
    win = "label_darts_pose"
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

    def advance_frames(n: int) -> bool:
        for _ in range(n):
            ok = cap.grab()
            if not ok:
                return False
        ok, fr = cap.retrieve()
        if not ok:
            return False
        st.frame = fr
        st.frame_idx += n
        return True

    def select_dart_by_click(x: int, y: int) -> None:
        if st.hide_others:
            return
        for i in range(len(st.darts) - 1, -1, -1):
            d = st.darts[i]
            if d.bbox and point_in_bbox(x, y, d.bbox):
                st.current = i
                return

    def on_mouse(event, x, y, flags, userdata):
        if st.frame is None or not st.paused:
            return

        h, w = st.frame.shape[:2]
        st.mouse_xy = (clamp(x, 0, w - 1), clamp(y, 0, h - 1))

        if (st.tip_mode or st.tail_mode or st.drawing) and not st.darts:
            ensure_current_dart(st)

        if not st.darts:
            return
        cur = st.darts[st.current]

        if event == cv2.EVENT_LBUTTONDOWN and not st.drawing:
            select_dart_by_click(x, y)

        if event == cv2.EVENT_RBUTTONDOWN:
            cur.tip = (clamp(x, 0, w - 1), clamp(y, 0, h - 1))
            return

        if st.tail_mode and event == cv2.EVENT_LBUTTONDOWN:
            cur.tail = (clamp(x, 0, w - 1), clamp(y, 0, h - 1))
            return

        if st.tip_mode and event == cv2.EVENT_LBUTTONDOWN:
            cur.tip = (clamp(x, 0, w - 1), clamp(y, 0, h - 1))
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
                    x1, y1 = st.drag_start
                    x2, y2 = st.drag_end
                    nb = normalize_bbox(x1, y1, x2, y2, w, h)
                    if nb is not None:
                        cur.bbox = nb
                st.drag_start = None
                st.drag_end = None

    cv2.setMouseCallback(win, on_mouse)

    if not read_next_frame():
        print("Could not read first frame.")
        return 1

    while True:
        assert st.frame is not None
        canvas = draw_overlay(st.frame, st, out_dir, roi, allow_empty=allow_empty)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(args.paused_delay_ms if st.paused else args.delay_ms) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord(" "):
            st.paused = not st.paused
            st.tip_mode = False
            st.tail_mode = False
            st.drawing = False
            st.drag_start = None
            st.drag_end = None
            continue

        if st.paused:
            if key in (ord("h"), ord("H")):
                st.hide_others = not st.hide_others
                set_toast(st, f"HIDE OTHERS: {'ON' if st.hide_others else 'OFF'}", 900)
                continue

            if key == ord("a"):
                add_new_dart(st)
                continue

            if key == ord("["):
                if st.darts:
                    st.current = (st.current - 1) % len(st.darts)
                continue

            if key == ord("]"):
                if st.darts:
                    st.current = (st.current + 1) % len(st.darts)
                continue

            if key == ord("t"):
                st.tip_mode = not st.tip_mode
                if st.tip_mode:
                    st.tail_mode = False
                st.drawing = False
                st.drag_start = None
                st.drag_end = None
                continue

            if key == ord("e"):
                st.tail_mode = not st.tail_mode
                if st.tail_mode:
                    st.tip_mode = False
                st.drawing = False
                st.drag_start = None
                st.drag_end = None
                continue

            if key == ord("d"):
                delete_current_dart(st)
                continue

            if key == ord("r"):
                reset_frame_labels(st)
                continue

            if key == ord("k"):
                if last_saved_darts:
                    st.darts = clone_darts(last_saved_darts)
                    st.current = max(0, len(st.darts) - 1)
                    print(f"[COPY] Copied {len(st.darts)} dart(s) from last save")
                else:
                    print("[COPY] No previous annotations to copy")
                continue

            if key == ord("s"):
                ok, msg = ready_to_save(st, allow_empty=allow_empty)
                if not ok:
                    print(f"[WARN] {msg}")
                    continue

                save_annot(img_dir, ann_dir, st.frame, st.frame_idx, st.darts, args.prefix)
                last_saved_darts = clone_darts(st.darts)
                set_toast(st, "SAVED", 900)

                for _ in range(args.after_count):
                    if not advance_frames(args.after_step):
                        print("[INFO] Reached end of video while saving extras.")
                        break
                    save_annot(img_dir, ann_dir, st.frame, st.frame_idx, st.darts, args.prefix)
                continue

            if key == ord(args.save_empty_key.lower()):
                st.darts = []
                st.current = 0
                st.tip_mode = False
                st.tail_mode = False
                st.drawing = False
                st.drag_start = None
                st.drag_end = None

                ok, msg = ready_to_save(st, allow_empty=allow_empty)
                if not ok:
                    print(f"[WARN] {msg}")
                    continue

                save_annot(img_dir, ann_dir, st.frame, st.frame_idx, st.darts, args.prefix)
                last_saved_darts = []
                set_toast(st, "SAVED (EMPTY)", 900)
                continue

            if key == ord("n"):
                if not read_next_frame():
                    print("End of video.")
                    break
                reset_frame_labels(st)
                continue

            if key == ord("b"):
                back_to = max(0, st.frame_idx - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, back_to)
                st.frame_idx = back_to - 1
                prev_gray_roi = None
                if not read_next_frame():
                    print("Cannot seek back.")
                    break
                reset_frame_labels(st)
                continue

        if not st.paused:
            if not read_next_frame():
                print("End of video.")
                break

            if args.auto_pause:
                if cooldown > 0:
                    cooldown -= 1

                gray = cv2.cvtColor(st.frame, cv2.COLOR_BGR2GRAY)
                gray_roi = apply_roi(gray, roi)

                if prev_gray_roi is None:
                    prev_gray_roi = gray_roi.copy()
                    st.last_motion_score = 0.0
                    continue

                g = gray_roi
                p = prev_gray_roi

                if args.blur and args.blur >= 3:
                    k = args.blur + 1 if (args.blur % 2 == 0) else args.blur
                    g = cv2.GaussianBlur(g, (k, k), 0)
                    p = cv2.GaussianBlur(p, (k, k), 0)

                if args.downscale and args.downscale != 1.0:
                    g = cv2.resize(g, (0, 0), fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)
                    p = cv2.resize(p, (0, 0), fx=args.downscale, fy=args.downscale, interpolation=cv2.INTER_AREA)

                diff = cv2.absdiff(g, p)
                changed = (diff >= args.diff_px).astype(np.uint8)
                frac = float(changed.mean())

                prev_gray_roi = gray_roi.copy()
                st.last_motion_score = frac * 100.0

                if args.motion_debug:
                    print(
                        f"[motion] frame={st.frame_idx:06d} "
                        f"changed={frac*100:.2f}% "
                        f"event={event_active} stable={stable_count} cooldown={cooldown}"
                    )

                if cooldown == 0:
                    if not event_active:
                        if frac >= args.motion_frac:
                            event_active = True
                            stable_count = 0
                    else:
                        if frac <= args.stable_frac:
                            stable_count += 1
                        else:
                            stable_count = 0

                        if stable_count >= args.stable_frames:
                            st.paused = True
                            st.tip_mode = False
                            st.tail_mode = False
                            st.drawing = False
                            st.drag_start = None
                            st.drag_end = None

                            event_active = False
                            stable_count = 0
                            cooldown = args.cooldown_frames

                            if last_saved_darts:
                                st.darts = clone_darts(last_saved_darts)
                                st.current = max(0, len(st.darts) - 1)
                            else:
                                reset_frame_labels(st)

                            print(f"[AUTO] paused at frame {st.frame_idx} (changed={frac*100:.2f}%)")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Session saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())