"""Segment a contour into circular arcs by greedily growing segments while fit quality holds."""

import numpy as np
from typing import Sequence

from circle_draw.circle_fitting.least_squares import fit, init_from_endpoints

Point = tuple[float, float]
Circle = np.ndarray


def _distance_between(p1: Point, p2: Point) -> float:
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def _average_deviation(circle: Circle, points: Sequence[Point]) -> float:
    cx, cy, r = circle
    deviations = [_distance_between((cx, cy), p) - r for p in points]
    return sum(deviations) / len(deviations)


def _fit_quality(circle: Circle, points: Sequence[Point]) -> float:
    avg_dev = _average_deviation(circle, points)
    if avg_dev == 0:
        return float(2**31 - 1)
    return len(points) / avg_dev


def _extend_backward(
    circle: Circle,
    pts: np.ndarray,
    preceding: np.ndarray,
    quality_threshold: float,
    k: int,
) -> tuple[Circle, np.ndarray, int]:
    """Try to prepend up to k trailing points from preceding into pts.

    Uses the same quality-ratio logic as the forward pass: stop as soon as
    prepending the next candidate would degrade quality beyond the threshold.
    Returns the updated (circle, pts, n_absorbed).
    """
    candidates = preceding[-k:]  # last k points of the preceding segment
    current_quality = _fit_quality(circle, pts)
    absorbed = 0

    for i in range(len(candidates) - 1, -1, -1):
        extended = np.concatenate(([candidates[i]], pts))
        new_circle = fit(circle, extended)
        new_quality = _fit_quality(new_circle, extended)
        if new_quality >= quality_threshold * current_quality:
            pts = extended
            circle = new_circle
            current_quality = new_quality
            absorbed += 1
        else:
            break

    return circle, pts, absorbed


def segment(
    points: np.ndarray,
    quality_threshold: float = 0.8,
    backward_k: int = 10,
) -> tuple[np.ndarray, list[Circle]]:
    """Split a contour into circular arc segments using fit quality.

    Starts with an initial window, then greedily extends each segment forward
    until adding a new point degrades fit quality beyond `quality_threshold`.
    After each segment is built, tries to extend it backwards by up to
    `backward_k` points from the preceding segment's tail — points that may
    fit the same circle despite falling outside the initial window.

    Returns end_points: shape (N, 2) — last point of each segment,
            circles:    list of N fitted circles [cx, cy, r].
    """
    n = len(points)
    initial_window = min(20, max(10, n // 10))

    current_circle: list = []
    current_point = 0
    circles: list[Circle] = []
    seg_point_arrays: list[np.ndarray] = []

    while current_point < n:
        if len(current_circle) == 0:
            if current_point + initial_window > n:
                current_pts = points[n - initial_window : n]
                current_point = n

                init = init_from_endpoints(tuple(current_pts[0]), tuple(current_pts[-1]))
                current_circle = fit(init, current_pts)

                seg_point_arrays.append(current_pts)
                circles.append(current_circle)
                current_circle = []
                continue
            else:
                current_pts = points[current_point : current_point + initial_window]
                current_point += initial_window

            init = init_from_endpoints(tuple(current_pts[0]), tuple(current_pts[-1]))
            current_circle = fit(init, current_pts)
            current_quality = _fit_quality(current_circle, current_pts)
        else:
            current_pts = np.concatenate((current_pts, [points[current_point]]))
            new_quality = _fit_quality(current_circle, current_pts)
            if new_quality <= quality_threshold * current_quality:
                if len(current_pts) % initial_window == 0:
                    current_circle = fit(current_circle, current_pts)
                current_quality = _fit_quality(current_circle, current_pts)
                current_point += 1
            else:
                current_circle = fit(current_circle, current_pts[: len(current_pts) - 1])

                # --- backward extension ---
                if seg_point_arrays and backward_k > 0:
                    current_circle, current_pts, absorbed = _extend_backward(
                        current_circle,
                        current_pts[: len(current_pts) - 1],
                        seg_point_arrays[-1],
                        quality_threshold,
                        backward_k,
                    )
                    if absorbed:
                        seg_point_arrays[-1] = seg_point_arrays[-1][: len(seg_point_arrays[-1]) - absorbed]
                        circles[-1] = fit(circles[-1], seg_point_arrays[-1]) if len(seg_point_arrays[-1]) >= 3 else circles[-1]
                else:
                    current_pts = current_pts[: len(current_pts) - 1]

                seg_point_arrays.append(current_pts)
                circles.append(current_circle)
                current_circle = []

    # backward extension for the final segment
    if len(seg_point_arrays) >= 2 and backward_k > 0 and len(current_circle) == 0:
        last_circle = circles[-1]
        last_pts = seg_point_arrays[-1]
        last_circle, last_pts, absorbed = _extend_backward(
            last_circle, last_pts, seg_point_arrays[-2], quality_threshold, backward_k
        )
        if absorbed:
            seg_point_arrays[-2] = seg_point_arrays[-2][: len(seg_point_arrays[-2]) - absorbed]
            circles[-2] = fit(circles[-2], seg_point_arrays[-2]) if len(seg_point_arrays[-2]) >= 3 else circles[-2]
            seg_point_arrays[-1] = last_pts
            circles[-1] = last_circle

    end_points_arr = np.array([seg[-1].tolist() for seg in seg_point_arrays])
    return end_points_arr, circles
