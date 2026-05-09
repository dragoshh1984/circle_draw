"""Step-by-step circle-fitting video renderer.

Produces two MP4s in the debug directory:
  fitting_slow.mp4  — 5 fps, readable text
  fitting_fast.mp4  — 30 fps, quick overview

The viewport is fitted to the bounding box of all eligible contours (with
padding) rather than the full source-image canvas, so the action is always
centred and zoomed in.
"""

import os
from typing import Callable, Optional

import cv2
import numpy as np

# ── Layout ────────────────────────────────────────────────────────────────────

VIDEO_W    = 1280
VIDEO_H    = 720
STATUS_H   = 88          # reserved at bottom for status text
DRAW_H     = VIDEO_H - STATUS_H
MAX_DRAW_R = 1800        # skip circle outlines whose screen radius exceeds this
VIEW_PAD   = 0.18        # fractional padding added around the contour bbox

# ── Colours (BGR) ─────────────────────────────────────────────────────────────

C_BG         = (18,  18,  18)
C_STATUS_BG  = ( 8,   8,   8)
C_DIVIDER    = (50,  50,  50)
C_TEXT       = (210, 210, 210)
C_TEXT_DIM   = ( 90,  90,  90)
C_ALL_PT     = (45,  45,  45)   # dim background — all contour points
C_INPUT_PT   = (200, 200,  60)  # bright — overview frame: chosen contours
C_CURR_PT    = (85,  85,  85)   # other points of the contour being processed
C_ACTIVE_PT  = (40, 195, 255)   # points already in the active segment
C_CIRCLE     = (30, 210,  40)   # current fitted circle
C_DONE_SEG   = (70, 130,  70)   # finished segments in the current contour
C_CAND_OK    = (40, 220,  40)   # accepted candidate point
C_CAND_STOP  = (40,  40, 220)   # rejected candidate point

_ACTION_COLOR = {
    "init":      (160, 220,  50),
    "added":     ( 40, 220,  40),
    "stopped":   ( 40,  40, 220),
    "backward":  (200, 100, 220),
    "finalized": ( 40, 210, 210),
}
_ACTION_LABEL = {
    "init":      ">>> INIT  initial window set",
    "added":     "+   ADDED point",
    "stopped":   "X   STOPPED",
    "backward":  "<   BACKWARD EXTENSION",
    "finalized": "v   SEGMENT FINALIZED",
}

# How many video frames to hold the input-contours overview (slow fps = 5)
_OVERVIEW_HOLD = 25   # 5 s at 5 fps


class VideoCollector:
    """Accumulates per-step frames from every contour and writes two MP4 videos."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        debug_dir: str,
        max_frames: int = 3000,
    ) -> None:
        # Fallback scale from full image — overridden by set_all_contours()
        w, h = image_shape[0], image_shape[1]
        self._scale = min(VIDEO_W / w, DRAW_H / h)
        self._ox    = int((VIDEO_W - w * self._scale) / 2)
        self._oy    = int((DRAW_H  - h * self._scale) / 2)

        self.debug_dir  = debug_dir
        self.max_frames = max_frames
        self.frames: list[np.ndarray] = []

        self._add_count = 0
        self._subsample = 1

        self._bg_base: Optional[np.ndarray] = None
        self._done_pts: list[np.ndarray] = []
        self._curr_pts: Optional[np.ndarray] = None
        self._contour_idx    = 0
        self._total_contours = 0

    # ── Setup API called by pipeline ─────────────────────────────────────────

    def set_all_contours(self, contours: list[np.ndarray]) -> None:
        """Compute viewport from contour bbox, render background, prepend overview."""
        if contours:
            all_pts = np.concatenate(contours)
            mn, mx  = all_pts.min(axis=0), all_pts.max(axis=0)
            bw, bh  = mx[0] - mn[0], mx[1] - mn[1]
            pad_x   = bw * VIEW_PAD
            pad_y   = bh * VIEW_PAD
            x0, y0  = mn[0] - pad_x, mn[1] - pad_y
            x1, y1  = mx[0] + pad_x, mx[1] + pad_y
            fw, fh  = x1 - x0, y1 - y0
            if fw > 0 and fh > 0:
                self._scale = min(VIDEO_W / fw, DRAW_H / fh)
                self._ox    = int((VIDEO_W - fw * self._scale) / 2 - x0 * self._scale)
                self._oy    = int((DRAW_H  - fh * self._scale) / 2 - y0 * self._scale)

        # Static background: all contour points as dim dots
        bg = np.full((VIDEO_H, VIDEO_W, 3), C_BG, dtype=np.uint8)
        bg[DRAW_H:, :] = C_STATUS_BG
        for pts in contours:
            for p in pts[::4]:
                x, y = self._tx(p[0]), self._ty(p[1])
                if 0 <= x < VIDEO_W and 0 <= y < DRAW_H:
                    bg[y, x] = C_ALL_PT
        self._bg_base = bg

        # Overview frames: highlight chosen contours + label
        self.frames.extend(
            self._make_overview_frame(contours) for _ in range(_OVERVIEW_HOLD)
        )

    def _make_overview_frame(self, contours: list[np.ndarray]) -> np.ndarray:
        img = np.full((VIDEO_H, VIDEO_W, 3), C_BG, dtype=np.uint8)
        img[DRAW_H:, :] = C_STATUS_BG
        cv2.line(img, (0, DRAW_H), (VIDEO_W, DRAW_H), C_DIVIDER, 1)

        for pts in contours:
            for p in pts[::2]:
                cv2.circle(img, self._pt(p[0], p[1]), 1, C_INPUT_PT, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"INPUT  —  {len(contours)} contours selected"
        cv2.putText(img, label, (10, DRAW_H + 22), font, 0.55, C_INPUT_PT, 1, cv2.LINE_AA)
        cv2.putText(img, "circle fitting begins next ...",
                    (10, DRAW_H + 50), font, 0.47, C_TEXT_DIM, 1, cv2.LINE_AA)
        return img

    def set_contour(self, pts: np.ndarray, idx: int, total: int) -> None:
        self._curr_pts       = pts
        self._contour_idx    = idx
        self._total_contours = total
        self._add_count      = 0

    def on_contour_done(self, pts: np.ndarray) -> None:
        self._done_pts.append(pts[::4])

    def make_callback(self) -> Callable[[dict], None]:
        return self._on_frame

    # ── Frame collection ─────────────────────────────────────────────────────

    def _on_frame(self, state: dict) -> None:
        action = state["action"]
        if action == "added":
            self._add_count += 1
            if self._add_count % self._subsample != 0:
                return

        self.frames.append(self._render(state))

        n = len(self.frames)
        if n > self.max_frames * 0.70:
            self._subsample = max(self._subsample, 4)
        if n > self.max_frames * 0.88:
            self._subsample = max(self._subsample, 10)

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _tx(self, x: float) -> int:
        return int(x * self._scale + self._ox)

    def _ty(self, y: float) -> int:
        return int(y * self._scale + self._oy)

    def _tr(self, r: float) -> int:
        return max(1, int(r * self._scale))

    def _pt(self, x: float, y: float) -> tuple[int, int]:
        return (self._tx(x), self._ty(y))

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self, state: dict) -> np.ndarray:
        img = (self._bg_base.copy() if self._bg_base is not None
               else np.full((VIDEO_H, VIDEO_W, 3), C_BG, dtype=np.uint8))
        cv2.line(img, (0, DRAW_H), (VIDEO_W, DRAW_H), C_DIVIDER, 1)

        for pts in self._done_pts:
            for p in pts:
                x, y = self._tx(p[0]), self._ty(p[1])
                if 0 <= x < VIDEO_W and 0 <= y < DRAW_H:
                    img[y, x] = C_CURR_PT

        if self._curr_pts is not None:
            for p in self._curr_pts[::2]:
                cv2.circle(img, self._pt(p[0], p[1]), 1, C_CURR_PT, -1)

        for seg_pts, seg_circ in state["completed_segs"]:
            cx, cy, r = float(seg_circ[0]), float(seg_circ[1]), float(seg_circ[2])
            sr = self._tr(r)
            if sr < MAX_DRAW_R:
                cv2.circle(img, self._pt(cx, cy), sr, C_DONE_SEG, 1, cv2.LINE_AA)
            for p in seg_pts[::2]:
                cv2.circle(img, self._pt(p[0], p[1]), 2, C_DONE_SEG, -1)

        for p in state["active_pts"]:
            cv2.circle(img, self._pt(float(p[0]), float(p[1])), 2, C_ACTIVE_PT, -1)

        cx, cy, r = (float(state["current_circle"][0]),
                     float(state["current_circle"][1]),
                     float(state["current_circle"][2]))
        sr = self._tr(r)
        if sr < MAX_DRAW_R:
            cv2.circle(img, self._pt(cx, cy), sr, C_CIRCLE, 1, cv2.LINE_AA)
        cv2.circle(img, self._pt(cx, cy), 4, C_CIRCLE, -1)

        if state["candidate_pt"] is not None:
            px, py = float(state["candidate_pt"][0]), float(state["candidate_pt"][1])
            c = C_CAND_OK if state["action"] == "added" else C_CAND_STOP
            cv2.circle(img, self._pt(px, py), 5, c, -1)
            cv2.circle(img, self._pt(px, py), 9, c, 1, cv2.LINE_AA)

        self._draw_status(img, state)
        return img

    def _draw_status(self, img: np.ndarray, state: dict) -> None:
        font   = cv2.FONT_HERSHEY_SIMPLEX
        action = state["action"]
        q      = state["quality"]
        pq     = state["prev_quality"]
        thr    = state["quality_threshold"]
        r      = float(state["current_circle"][2])
        n_pts  = len(state["active_pts"])

        def qfmt(v: Optional[float]) -> str:
            if v is None:
                return "—"
            return "∞" if v > 1e9 else f"{v:.4f}"

        rel_dev = (n_pts / q) if (q and q > 0) else 0.0

        row1 = (
            f"contour {self._contour_idx + 1}/{self._total_contours}  "
            f"seg {len(state['completed_segs']) + 1}  "
            f"pts {n_pts}  r = {r:.1f} px"
        )
        row2 = (
            f"quality {qfmt(q)}  prev {qfmt(pq)}  "
            f"threshold x{thr:.2f}  "
            f"rel_dev {rel_dev:.5f}  avg_dev {rel_dev * r:.2f} px"
        )
        reason = state.get("stop_reason", "")
        label  = _ACTION_LABEL.get(action, action)
        row3   = f"{label}   {reason}" if reason else label

        y0 = DRAW_H + 20
        cv2.putText(img, row1, (10, y0),      font, 0.47, C_TEXT,       1, cv2.LINE_AA)
        cv2.putText(img, row2, (10, y0 + 23), font, 0.47, C_TEXT,       1, cv2.LINE_AA)
        cv2.putText(img, row3, (10, y0 + 48), font, 0.50,
                    _ACTION_COLOR.get(action, C_TEXT), 1, cv2.LINE_AA)
        cv2.putText(img, f"#{len(self.frames) + 1}",
                    (VIDEO_W - 72, y0), font, 0.44, C_TEXT_DIM, 1, cv2.LINE_AA)

    # ── Video output ──────────────────────────────────────────────────────────

    def save_videos(self) -> None:
        if not self.frames:
            print("Debug video: no frames collected, skipping.")
            return
        os.makedirs(self.debug_dir, exist_ok=True)
        n    = len(self.frames)
        slow = os.path.join(self.debug_dir, "fitting_slow.mp4")
        fast = os.path.join(self.debug_dir, "fitting_fast.mp4")
        self._write_mp4(slow, fps=5)
        self._write_mp4(fast, fps=30)
        print(f"Video slow → {slow}  ({n} frames, {n/5:.0f}s @ 5 fps)")
        print(f"Video fast → {fast}  ({n} frames, {n/30:.0f}s @ 30 fps)")

    def _write_mp4(self, path: str, fps: int) -> None:
        h, w = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in self.frames:
            writer.write(frame)
        writer.release()
