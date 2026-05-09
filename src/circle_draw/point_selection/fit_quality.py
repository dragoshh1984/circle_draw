"""Segment a contour into circular arcs by greedily growing segments while fit quality holds."""

from dataclasses import dataclass
import math
from typing import Callable, Optional, Sequence

import numpy as np

from circle_draw.circle_fitting.least_squares import fit, init_from_endpoints

Point  = tuple[float, float]
Circle = np.ndarray
MIN_SEGMENT_POINTS = 3


@dataclass
class SegmentShapeMetadata:
    """Per-point shape descriptors aligned with one accepted segment."""

    contour_indices: np.ndarray
    turn_angles: np.ndarray
    is_sharp: np.ndarray


def _required_quality_ratio(quality_threshold: float) -> float:
    return max(1.0, quality_threshold)


def _distance_between(p1: Point, p2: Point) -> float:
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def _average_relative_deviation(circle: Circle, points: Sequence[Point]) -> float:
    cx, cy, r = circle
    radius = abs(float(r))
    deviations = [
        abs(_distance_between((cx, cy), p) - radius)
        for p in points
    ]
    if radius > np.finfo(float).eps:
        deviations = [dev / radius for dev in deviations]
    return sum(deviations) / len(deviations)


def _fit_quality(circle: Circle, points: Sequence[Point]) -> float:
    relative_dev = _average_relative_deviation(circle, points)
    if relative_dev <= np.finfo(float).eps:
        return float(2**31 - 1)
    return len(points) / relative_dev


def _turn_angle_at_index(points: np.ndarray, index: int, circle: Circle) -> float:
    """Return the residual turn beyond what the fitted circle predicts (radians).

    For any smooth arc — large or small — the heading change between neighbours
    equals the arc's central angle, so the residual is near zero.  A genuine
    kink produces a residual independent of the circle's size.
    """
    n = len(points)
    if n < 3:
        return 0.0

    prev_point = points[(index - 1) % n]
    curr_point = points[index % n]
    next_point = points[(index + 1) % n]

    v_in  = curr_point - prev_point
    v_out = next_point - curr_point
    norm_in  = float(np.linalg.norm(v_in))
    norm_out = float(np.linalg.norm(v_out))
    if norm_in <= np.finfo(float).eps or norm_out <= np.finfo(float).eps:
        return 0.0

    raw_dot = float(np.clip(np.dot(v_in / norm_in, v_out / norm_out), -1.0, 1.0))
    raw_turn = float(math.acos(raw_dot))

    # Central angle subtended by prev→next at the fitted circle's centre —
    # this is the turn that is *expected* for a point on this arc.
    cx, cy = float(circle[0]), float(circle[1])
    rp = prev_point - np.array([cx, cy])
    rn = next_point - np.array([cx, cy])
    norm_rp = float(np.linalg.norm(rp))
    norm_rn = float(np.linalg.norm(rn))
    if norm_rp <= np.finfo(float).eps or norm_rn <= np.finfo(float).eps:
        return raw_turn

    arc_dot = float(np.clip(np.dot(rp / norm_rp, rn / norm_rn), -1.0, 1.0))
    arc_angle = float(math.acos(arc_dot))

    return max(0.0, raw_turn - arc_angle)


def _batch_has_sharp_turn(
    contour_points: np.ndarray,
    contour_indices: np.ndarray,
    circle: Circle,
    sharp_turn_threshold: float,
) -> bool:
    if not np.isfinite(sharp_turn_threshold):
        return False
    for idx in contour_indices:
        if _turn_angle_at_index(contour_points, int(idx), circle) > sharp_turn_threshold:
            return True
    return False


def _build_segment_metadata(
    contour_points: np.ndarray,
    contour_indices: np.ndarray,
    circle: Circle,
    sharp_turn_threshold: float,
) -> SegmentShapeMetadata:
    turn_angles = np.array([
        _turn_angle_at_index(contour_points, int(idx), circle)
        for idx in contour_indices
    ], dtype=float)
    if np.isfinite(sharp_turn_threshold):
        is_sharp = turn_angles > sharp_turn_threshold
    else:
        is_sharp = np.zeros_like(turn_angles, dtype=bool)
    return SegmentShapeMetadata(
        contour_indices=np.array(contour_indices, dtype=int),
        turn_angles=turn_angles,
        is_sharp=is_sharp,
    )


def _extend_backward(
    circle: Circle,
    pts: np.ndarray,
    pts_indices: np.ndarray,
    preceding: np.ndarray,
    preceding_indices: np.ndarray,
    contour_points: np.ndarray,
    quality_threshold: float,
    k: int,
    sharp_turn_threshold: float,
    min_preceding_points: int = MIN_SEGMENT_POINTS,
) -> tuple[Circle, np.ndarray, np.ndarray, int]:
    """Try to prepend up to k trailing points from preceding into pts.

    Uses the same quality-ratio logic as the forward pass: stop as soon as
    prepending the next candidate would degrade quality beyond the threshold.
    Returns the updated (circle, pts, n_absorbed).
    """
    candidate_start = max(min_preceding_points, len(preceding) - k)
    candidates = preceding[candidate_start:]
    candidate_indices = preceding_indices[candidate_start:]
    current_quality = _fit_quality(circle, pts)
    absorbed = 0

    for i in range(len(candidates) - 1, -1, -1):
        candidate_index = np.array([int(candidate_indices[i])], dtype=int)
        extended = np.concatenate(([candidates[i]], pts))
        extended_indices = np.concatenate((candidate_index, pts_indices))
        new_circle = fit(circle, extended)
        if _batch_has_sharp_turn(contour_points, candidate_index, new_circle, sharp_turn_threshold):
            break

        new_quality = _fit_quality(new_circle, extended)
        if new_quality >= _required_quality_ratio(quality_threshold) * current_quality:
            pts = extended
            pts_indices = extended_indices
            circle = new_circle
            current_quality = new_quality
            absorbed += 1
        else:
            break

    return circle, pts, pts_indices, absorbed


def segment(
    points: np.ndarray,
    quality_threshold: float = 1.0,
    forward_k: int = 10,
    backward_k: int = 10,
    sharp_turn_threshold: float = float("inf"),
    frame_callback: Optional[Callable[[dict], None]] = None,
) -> tuple[np.ndarray, list[Circle], list[np.ndarray], list[SegmentShapeMetadata]]:
    """Split a contour into circular arc segments using fit quality.

    Starts with an initial window, then greedily extends each segment forward
    by trying to add up to `forward_k` points at once. If the largest batch
    fails the refit quality check, smaller batches are tried until either one
    is accepted or zero points can be added.
    After all segments are built, tries to extend the first segment backwards
    by up to `backward_k` points from the final segment's tail.
    frame_callback: optional callable receiving a state dict on every step.
    Returns end_points: shape (N, 2) — last point of each segment,
            circles:    list of N fitted circles [cx, cy, r],
            segment point arrays in contour order,
            per-segment point-shape metadata aligned with segment arrays.
    """
    n = len(points)
    initial_window = min(20, max(10, n // 10))

    current_circle: list = []
    current_point = 0
    circles: list[Circle] = []
    seg_point_arrays: list[np.ndarray] = []
    seg_index_arrays: list[np.ndarray] = []

    def _emit(
        action: str,
        active_pts: np.ndarray,
        circle: np.ndarray,
        candidate_pt: Optional[tuple] = None,
        quality: float = 0.0,
        prev_quality: Optional[float] = None,
        stop_reason: str = "",
        rejection_reason: str = "",
        attempted_batch_size: int = 0,
    ) -> None:
        if frame_callback is None:
            return
        c = np.array(circle) if not isinstance(circle, np.ndarray) else circle
        frame_callback({
            "action": action,
            "active_pts": active_pts.copy(),
            "current_circle": c.copy(),
            "candidate_pt": candidate_pt,
            "quality": quality,
            "prev_quality": prev_quality,
            "quality_threshold": _required_quality_ratio(quality_threshold),
            "configured_quality_threshold": quality_threshold,
            "sharp_turn_threshold": sharp_turn_threshold,
            "stop_reason": stop_reason,
            "rejection_reason": rejection_reason,
            "attempted_batch_size": attempted_batch_size,
            "completed_segs": [
                (p.copy(), np.array(cc).copy())
                for p, cc in zip(seg_point_arrays, circles)
            ],
        })

    while current_point < n:
        if len(current_circle) == 0:
            # ── Start a new segment ───────────────────────────────────────────
            if current_point + initial_window > n:
                # Tail: grab whatever remains
                current_pts = points[n - initial_window : n]
                current_indices = np.arange(n - initial_window, n, dtype=int)
                current_point = n
                init = init_from_endpoints(tuple(current_pts[0]), tuple(current_pts[-1]))
                current_circle = fit(init, current_pts)
                q = _fit_quality(current_circle, current_pts)
                _emit("init", current_pts, current_circle, quality=q)
                seg_point_arrays.append(current_pts)
                seg_index_arrays.append(current_indices)
                circles.append(current_circle)
                _emit("finalized", current_pts, current_circle, quality=q)
                current_circle = []
                continue

            current_pts = points[current_point : current_point + initial_window]
            current_indices = np.arange(current_point, current_point + initial_window, dtype=int)
            current_point += initial_window
            init = init_from_endpoints(tuple(current_pts[0]), tuple(current_pts[-1]))
            current_circle = fit(init, current_pts)
            current_quality = _fit_quality(current_circle, current_pts)
            _emit("init", current_pts, current_circle, quality=current_quality)

        else:
            # ── Try to extend the current segment by the largest passing batch ─
            prev_quality = current_quality
            required_ratio = _required_quality_ratio(quality_threshold)
            max_batch = min(max(0, forward_k), n - current_point)
            accepted: Optional[tuple[int, np.ndarray, Circle, float]] = None
            accepted_indices: Optional[np.ndarray] = None
            last_rejected: Optional[tuple[int, float, str]] = None

            for batch_size in range(max_batch, 0, -1):
                batch = points[current_point : current_point + batch_size]
                batch_indices = np.arange(current_point, current_point + batch_size, dtype=int)
                extended = np.concatenate((current_pts, batch))
                new_circle = fit(current_circle, extended)
                new_quality = _fit_quality(new_circle, extended)
                if _batch_has_sharp_turn(points, batch_indices, new_circle, sharp_turn_threshold):
                    last_rejected = (batch_size, current_quality, "sharp_turn")
                    continue

                if new_quality >= required_ratio * current_quality:
                    accepted = (batch_size, extended, new_circle, new_quality)
                    accepted_indices = batch_indices
                    break
                last_rejected = (batch_size, new_quality, "quality")

            if accepted is not None:
                # Accept
                accepted_count, extended, new_circle, new_quality = accepted
                current_pts = extended
                if accepted_indices is not None:
                    current_indices = np.concatenate((current_indices, accepted_indices))
                current_circle = new_circle
                current_quality = new_quality
                current_point += accepted_count
                _emit(
                    "added", current_pts, current_circle,
                    candidate_pt=tuple(current_pts[-1]),
                    quality=current_quality, prev_quality=prev_quality,
                    stop_reason=f"accepted {accepted_count} pts",
                    attempted_batch_size=accepted_count,
                )

            else:
                # Reject — finalise segment
                rejected_count, new_quality, rejection_reason = last_rejected or (0, 0.0, "none")
                quality_ratio = new_quality / prev_quality if prev_quality > 0 else 0.0
                if rejection_reason == "sharp_turn":
                    stop_reason = (
                        f"best batch {rejected_count} rejected due to sharp turn over "
                        f"threshold {sharp_turn_threshold:.3f} rad"
                    )
                else:
                    stop_reason = (
                        f"best batch {rejected_count} quality ratio {quality_ratio:.3f} below {required_ratio:.3f}  "
                        f"({prev_quality:.4f} -> {new_quality:.4f})"
                    )
                _emit(
                    "stopped", current_pts, current_circle,
                    candidate_pt=tuple(points[current_point]) if current_point < n else None,
                    quality=prev_quality, prev_quality=prev_quality,
                    stop_reason=stop_reason,
                    rejection_reason=rejection_reason,
                    attempted_batch_size=rejected_count,
                )

                seg_point_arrays.append(current_pts)
                seg_index_arrays.append(current_indices)
                circles.append(current_circle)
                _emit(
                    "finalized", current_pts, current_circle,
                    quality=_fit_quality(current_circle, current_pts),
                )
                current_circle = []

    if len(current_circle) != 0:
        seg_point_arrays.append(current_pts)
        seg_index_arrays.append(current_indices)
        circles.append(current_circle)
        _emit(
            "finalized", current_pts, current_circle,
            quality=_fit_quality(current_circle, current_pts),
        )

    # Backward extension is only for the first segment, using the closed
    # contour's final segment as its predecessor.
    if len(seg_point_arrays) >= 2 and backward_k > 0:
        first_circle = circles[0]
        first_pts = seg_point_arrays[0]
        first_indices = seg_index_arrays[0]
        first_circle, first_pts, first_indices, absorbed = _extend_backward(
            first_circle,
            first_pts,
            first_indices,
            seg_point_arrays[-1],
            seg_index_arrays[-1],
            points,
            quality_threshold,
            backward_k,
            sharp_turn_threshold,
        )
        if absorbed:
            seg_point_arrays[-1] = seg_point_arrays[-1][: len(seg_point_arrays[-1]) - absorbed]
            seg_index_arrays[-1] = seg_index_arrays[-1][: len(seg_index_arrays[-1]) - absorbed]
            if len(seg_point_arrays[-1]) >= MIN_SEGMENT_POINTS:
                circles[-1] = fit(circles[-1], seg_point_arrays[-1])
            seg_point_arrays[0] = first_pts
            seg_index_arrays[0] = first_indices
            circles[0] = first_circle
            _emit(
                "backward", first_pts, first_circle,
                quality=_fit_quality(first_circle, first_pts),
                stop_reason=f"first segment absorbed {absorbed} pts from final segment",
            )

    end_points_arr = np.array([seg[-1].tolist() for seg in seg_point_arrays])
    segment_metadata = [
        _build_segment_metadata(points, seg_indices, circle, sharp_turn_threshold)
        for seg_indices, circle in zip(seg_index_arrays, circles)
    ]
    return end_points_arr, circles, seg_point_arrays, segment_metadata
