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


def segment(
    points: np.ndarray,
    quality_threshold: float = 0.8,
) -> tuple[np.ndarray, list[Circle]]:
    """Split a contour into circular arc segments using fit quality.

    Starts with an initial window, then greedily extends each segment until
    adding a new point degrades fit quality beyond `quality_threshold` of the
    current quality.

    Returns end_points: shape (N, 2) — last point of each segment,
            circles:    list of N fitted circles [cx, cy, r].
    """
    n = len(points)
    initial_window = min(20, max(10, n // 10))

    current_circle: list = []
    current_point = 0
    circles: list[Circle] = []
    segments: list[np.ndarray] = []
    end_pts: list[Point] = []

    while current_point < n:
        if len(current_circle) == 0:
            if current_point + initial_window > n:
                current_pts = points[n - initial_window : n]
                current_point = n

                init = init_from_endpoints(tuple(current_pts[0]), tuple(current_pts[-1]))
                current_circle = fit(init, current_pts)

                segments.append(current_pts)
                circles.append(current_circle)
            else:
                current_pts = points[current_point : current_point + initial_window]
                current_point += initial_window

            init = init_from_endpoints(tuple(current_pts[0]), tuple(current_pts[-1]))
            current_circle = fit(init, current_pts)
            current_quality = _fit_quality(current_circle, current_pts)
            end_pts.append(current_pts[-1])
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

                segments.append(current_pts)
                circles.append(current_circle)
                current_circle = []

    seg_endpoints = [[seg[0], seg[-1]] for seg in segments]
    end_points_arr = np.array([ep[1].tolist() for ep in seg_endpoints])
    return end_points_arr, circles
