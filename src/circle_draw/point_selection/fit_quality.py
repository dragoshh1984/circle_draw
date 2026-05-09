"""Segment a contour into circular arcs by greedily growing segments while fit quality holds."""

from typing import Callable, Optional, Sequence

import numpy as np

from circle_draw.circle_fitting.least_squares import fit, init_from_endpoints

Point  = tuple[float, float]
Circle = np.ndarray
MIN_SEGMENT_POINTS = 3


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


def _extend_backward(
    circle: Circle,
    pts: np.ndarray,
    preceding: np.ndarray,
    quality_threshold: float,
    k: int,
    min_preceding_points: int = MIN_SEGMENT_POINTS,
) -> tuple[Circle, np.ndarray, int]:
    """Try to prepend up to k trailing points from preceding into pts.

    Uses the same quality-ratio logic as the forward pass: stop as soon as
    prepending the next candidate would degrade quality beyond the threshold.
    Returns the updated (circle, pts, n_absorbed).
    """
    candidate_start = max(min_preceding_points, len(preceding) - k)
    candidates = preceding[candidate_start:]
    current_quality = _fit_quality(circle, pts)
    absorbed = 0

    for i in range(len(candidates) - 1, -1, -1):
        extended = np.concatenate(([candidates[i]], pts))
        new_circle = fit(circle, extended)
        new_quality = _fit_quality(new_circle, extended)
        if new_quality >= _required_quality_ratio(quality_threshold) * current_quality:
            pts = extended
            circle = new_circle
            current_quality = new_quality
            absorbed += 1
        else:
            break

    return circle, pts, absorbed


def segment(
    points: np.ndarray,
    quality_threshold: float = 1.0,
    forward_k: int = 10,
    backward_k: int = 10,
    frame_callback: Optional[Callable[[dict], None]] = None,
) -> tuple[np.ndarray, list[Circle], list[np.ndarray]]:
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
            segment point arrays in contour order.
    """
    n = len(points)
    initial_window = min(20, max(10, n // 10))

    current_circle: list = []
    current_point = 0
    circles: list[Circle] = []
    seg_point_arrays: list[np.ndarray] = []

    def _emit(
        action: str,
        active_pts: np.ndarray,
        circle: np.ndarray,
        candidate_pt: Optional[tuple] = None,
        quality: float = 0.0,
        prev_quality: Optional[float] = None,
        stop_reason: str = "",
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
            "stop_reason": stop_reason,
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
                current_point = n
                init = init_from_endpoints(tuple(current_pts[0]), tuple(current_pts[-1]))
                current_circle = fit(init, current_pts)
                q = _fit_quality(current_circle, current_pts)
                _emit("init", current_pts, current_circle, quality=q)
                seg_point_arrays.append(current_pts)
                circles.append(current_circle)
                _emit("finalized", current_pts, current_circle, quality=q)
                current_circle = []
                continue

            current_pts = points[current_point : current_point + initial_window]
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
            last_rejected: Optional[tuple[int, float]] = None

            for batch_size in range(max_batch, 0, -1):
                batch = points[current_point : current_point + batch_size]
                extended = np.concatenate((current_pts, batch))
                new_circle = fit(current_circle, extended)
                new_quality = _fit_quality(new_circle, extended)
                if new_quality >= required_ratio * current_quality:
                    accepted = (batch_size, extended, new_circle, new_quality)
                    break
                last_rejected = (batch_size, new_quality)

            if accepted is not None:
                # Accept
                accepted_count, extended, new_circle, new_quality = accepted
                current_pts = extended
                current_circle = new_circle
                current_quality = new_quality
                current_point += accepted_count
                _emit(
                    "added", current_pts, current_circle,
                    candidate_pt=tuple(current_pts[-1]),
                    quality=current_quality, prev_quality=prev_quality,
                    stop_reason=f"accepted {accepted_count} pts",
                )

            else:
                # Reject — finalise segment
                rejected_count, new_quality = last_rejected or (0, 0.0)
                quality_ratio = new_quality / prev_quality if prev_quality > 0 else 0.0
                stop_reason = (
                    f"best batch {rejected_count} quality ratio {quality_ratio:.3f} below {required_ratio:.3f}  "
                    f"({prev_quality:.4f} -> {new_quality:.4f})"
                )
                _emit(
                    "stopped", current_pts, current_circle,
                    candidate_pt=tuple(points[current_point]) if current_point < n else None,
                    quality=prev_quality, prev_quality=prev_quality,
                    stop_reason=stop_reason,
                )

                seg_point_arrays.append(current_pts)
                circles.append(current_circle)
                _emit(
                    "finalized", current_pts, current_circle,
                    quality=_fit_quality(current_circle, current_pts),
                )
                current_circle = []

    if len(current_circle) != 0:
        seg_point_arrays.append(current_pts)
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
        first_circle, first_pts, absorbed = _extend_backward(
            first_circle, first_pts, seg_point_arrays[-1], quality_threshold, backward_k,
        )
        if absorbed:
            seg_point_arrays[-1] = seg_point_arrays[-1][: len(seg_point_arrays[-1]) - absorbed]
            if len(seg_point_arrays[-1]) >= MIN_SEGMENT_POINTS:
                circles[-1] = fit(circles[-1], seg_point_arrays[-1])
            seg_point_arrays[0] = first_pts
            circles[0] = first_circle
            _emit(
                "backward", first_pts, first_circle,
                quality=_fit_quality(first_circle, first_pts),
                stop_reason=f"first segment absorbed {absorbed} pts from final segment",
            )

    end_points_arr = np.array([seg[-1].tolist() for seg in seg_point_arrays])
    return end_points_arr, circles, seg_point_arrays
