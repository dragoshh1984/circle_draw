"""Geometric primitives for circle-circle and line-circle intersections."""

import numpy as np
from typing import Optional

Point = tuple[float, float]
Circle = np.ndarray  # [cx, cy, r]


def distance_between(p1: Point, p2: Point) -> float:
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def _point_valid(p: Point) -> bool:
    return not (np.isnan(p[0]) or np.isnan(p[1]))


def circle_circle_intersection(c1: Circle, c2: Circle) -> Optional[list[Point]]:
    """Return the two intersection points of c1 and c2, or None if they don't meet."""
    cx1, cy1, r1 = c1[0], c1[1], c1[2]
    cx2, cy2, r2 = c2[0], c2[1], c2[2]

    if distance_between((cx1, cy1), (cx2, cy2)) > r1 + r2:
        return None

    d = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    l = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0.0, r1**2 - l**2))

    x1 = (l / d) * (cx2 - cx1) + (h / d) * (cy2 - cy1) + cx1
    x2 = (l / d) * (cx2 - cx1) - (h / d) * (cy2 - cy1) + cx1
    y1 = (l / d) * (cy2 - cy1) - (h / d) * (cx2 - cx1) + cy1
    y2 = (l / d) * (cy2 - cy1) + (h / d) * (cx2 - cx1) + cy1

    return [(x1, y1), (x2, y2)]


def _nearest_intersection(candidates: list[Point], reference: Point) -> Point:
    if distance_between(candidates[0], reference) < distance_between(candidates[1], reference):
        return candidates[0]
    return candidates[1]


def line_circle_intersection(p1: Point, p2: Point, circle: Circle) -> tuple[Point, Point]:
    """Return both intersection points of the line through p1,p2 with the circle."""
    cx, cy, r = circle[0], circle[1], circle[2]

    x1 = p1[0] - cx
    x2 = p2[0] - cx
    y1 = p1[1] - cy
    y2 = p2[1] - cy

    dx = x2 - x1
    dy = y2 - y1
    dr = np.sqrt(dx**2 + dy**2)
    D = x1 * y2 - x2 * y1

    disc = max(0.0, r**2 * dr**2 - D**2)
    sqrt_disc = np.sqrt(disc)
    sign_dy = np.sign(dy) if dy != 0 else 1.0

    ix1 = (D * dy + sign_dy * dx * sqrt_disc) / dr**2 + cx
    iy1 = (-D * dx + abs(dy) * sqrt_disc) / dr**2 + cy
    ix2 = (D * dy - sign_dy * dx * sqrt_disc) / dr**2 + cx
    iy2 = (-D * dx - abs(dy) * sqrt_disc) / dr**2 + cy

    return (ix1, iy1), (ix2, iy2)


def closest_boundary_points(c1: Circle, c2: Circle) -> tuple[Point, Point]:
    """Find the pair of boundary points — one on each circle — that are closest.

    When circles intersect, uses the center-to-center line to find boundary points
    and picks the closest pair. When they don't intersect, returns the facing points.
    """
    cx1, cy1, r1 = c1[0], c1[1], c1[2]
    cx2, cy2, r2 = c2[0], c2[1], c2[2]

    if distance_between((cx1, cy1), (cx2, cy2)) <= r1 + r2:
        a1, a2 = line_circle_intersection((cx1, cy1), (cx2, cy2), c1)
        b1, b2 = line_circle_intersection((cx1, cy1), (cx2, cy2), c2)

        pairs = [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]
        dists = [distance_between(pa, pb) for pa, pb in pairs]
        best = int(np.argmin(dists))
        return pairs[best]
    else:
        v1x = cx2 - cx1
        v1y = cy2 - cy1
        v2x = -v1x
        v2y = -v1y
        norm1 = np.sqrt(v1x**2 + v1y**2)
        norm2 = np.sqrt(v2x**2 + v2y**2)
        p1 = (cx1 + v1x / norm1 * r1, cy1 + v1y / norm1 * r1)
        p2 = (cx2 + v2x / norm2 * r2, cy2 + v2y / norm2 * r2)
        return p1, p2


def build_arc_boundaries(
    circles: list[Circle], end_points: np.ndarray
) -> list[list[Point]]:
    """For each circle compute (arc_start, arc_end) using actual circle intersections.

    end_points[i] is the expected boundary between circles[i] and circles[i+1],
    used to disambiguate which of the two intersection points to pick.
    """
    n = len(circles)
    boundaries: list[list[Point]] = [[(np.nan, np.nan), (np.nan, np.nan)] for _ in range(n)]

    for i in range(n - 1):
        candidates = circle_circle_intersection(circles[i], circles[i + 1])
        if candidates is not None:
            pt = _nearest_intersection(candidates, tuple(end_points[i]))
            if _point_valid(pt):
                boundaries[i][1] = pt
                boundaries[i + 1][0] = pt
                continue
        p1, p2 = closest_boundary_points(circles[i], circles[i + 1])
        boundaries[i][1] = p1
        boundaries[i + 1][0] = p2

    # wrap-around: connect last circle back to first
    candidates = circle_circle_intersection(circles[0], circles[-1])
    if candidates is not None:
        pt = _nearest_intersection(candidates, tuple(end_points[-1]))
        if _point_valid(pt):
            boundaries[0][0] = pt
            boundaries[-1][1] = pt
            return boundaries

    p1, p2 = closest_boundary_points(circles[0], circles[-1])
    boundaries[0][0] = p1
    boundaries[-1][1] = p2
    return boundaries
