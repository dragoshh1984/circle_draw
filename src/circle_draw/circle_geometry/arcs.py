"""Explicit circular arc descriptions derived from fitted contour segments."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from circle_draw.circle_geometry.intersections import Circle, Point

TAU = 2 * math.pi
ANGLE_EPS = 1e-9
MAX_INTERSECTION_REBIND_DISTANCE = 30.0
MAX_INTERSECTION_REBIND_GAP_MULTIPLIER = 3.0


def _angle(circle: Circle, point: Point) -> float:
    return math.atan2(point[1] - circle[1], point[0] - circle[0])


def _project_to_circle(circle: Circle, point: Point) -> Point:
    cx, cy, r = circle
    vx = point[0] - cx
    vy = point[1] - cy
    norm = math.hypot(vx, vy)
    if norm <= ANGLE_EPS:
        return (cx + r, cy)
    return (cx + vx / norm * r, cy + vy / norm * r)


def _angle_near(angle: float, target: float) -> float:
    return angle + round((target - angle) / TAU) * TAU


def _distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _circle_intersections(c1: Circle, c2: Circle) -> list[Point]:
    """Return actual circle-circle intersections, or [] when there are none."""
    cx1, cy1, r1 = float(c1[0]), float(c1[1]), abs(float(c1[2]))
    cx2, cy2, r2 = float(c2[0]), float(c2[1]), abs(float(c2[2]))
    dx = cx2 - cx1
    dy = cy2 - cy1
    d = math.hypot(dx, dy)

    if d <= ANGLE_EPS:
        return []
    if d > r1 + r2 + ANGLE_EPS:
        return []
    if d < abs(r1 - r2) - ANGLE_EPS:
        return []
    if r1 <= ANGLE_EPS or r2 <= ANGLE_EPS:
        return []

    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h_sq = r1 * r1 - a * a
    if h_sq < -ANGLE_EPS:
        return []

    h = math.sqrt(max(0.0, h_sq))
    xm = cx1 + a * dx / d
    ym = cy1 + a * dy / d
    rx = -dy * h / d
    ry = dx * h / d

    p1 = (xm + rx, ym + ry)
    p2 = (xm - rx, ym - ry)
    if _distance(p1, p2) <= ANGLE_EPS:
        return [p1]
    return [p1, p2]


def _nearest_boundary_intersection(
    candidates: Sequence[Point],
    previous_end: Point,
    next_start: Point,
) -> Point:
    return min(
        candidates,
        key=lambda point: _distance(point, previous_end) + _distance(point, next_start),
    )


def _intersection_is_near_boundary(
    point: Point,
    previous_end: Point,
    next_start: Point,
) -> bool:
    old_gap = _distance(previous_end, next_start)
    max_shift = max(_distance(point, previous_end), _distance(point, next_start))
    allowed_shift = max(
        MAX_INTERSECTION_REBIND_DISTANCE,
        MAX_INTERSECTION_REBIND_GAP_MULTIPLIER * old_gap,
    )
    return max_shift <= allowed_shift


def _equivalent_angles_near_span(
    circle: Circle,
    start: Point,
    end: Point,
    old_start: float,
    old_end: float,
) -> tuple[float, float]:
    """Choose angle equivalents whose signed span stays closest to the old arc."""
    target_span = old_end - old_start
    start_base = _angle_near(_angle(circle, start), old_start)
    end_base = _angle_near(_angle(circle, end), old_end)
    candidates: list[tuple[float, float, float, float]] = []

    for start_turn in (-1, 0, 1):
        angle_start = start_base + start_turn * TAU
        for end_turn in (-1, 0, 1):
            angle_end = end_base + end_turn * TAU
            span = angle_end - angle_start
            score = (
                abs(span - target_span),
                abs(angle_start - old_start) + abs(angle_end - old_end),
            )
            candidates.append((*score, angle_start, angle_end))

    _, _, angle_start, angle_end = min(candidates)
    return float(angle_start), float(angle_end)


def _segment_arc_geometry(
    circle: Circle,
    points: np.ndarray,
) -> tuple[Point, Point, float, float]:
    if len(points) == 0:
        start = (circle[0] + circle[2], circle[1])
        return start, start, 0.0, 0.0

    point_angles = np.unwrap([_angle(circle, tuple(point)) for point in points])
    start_angle = float(point_angles[0])
    end_angle = float(point_angles[-1])
    start = _project_to_circle(circle, tuple(points[0]))
    end = _project_to_circle(circle, tuple(points[-1]))
    return start, end, start_angle, end_angle


@dataclass
class CircleArc:
    """A fitted circle arc with an explicit draw direction."""

    circle: Circle
    start: Point
    end: Point
    angle_start: float
    angle_end: float

    @property
    def clockwise(self) -> bool:
        return self.angle_end >= self.angle_start

    @classmethod
    def from_segment_points(
        cls,
        circle: Circle,
        points: np.ndarray,
    ) -> "CircleArc":
        start, end, angle_start, angle_end = _segment_arc_geometry(circle, points)
        return cls(
            circle=circle,
            start=start,
            end=end,
            angle_start=angle_start,
            angle_end=angle_end,
        )

    @classmethod
    def from_boundaries(
        cls,
        circle: Circle,
        start: Point,
        end: Point,
        clockwise: bool = True,
    ) -> "CircleArc":
        angle_start = _angle(circle, start)
        angle_end = _angle_near(_angle(circle, end), angle_start)
        if clockwise:
            while angle_end < angle_start:
                angle_end += TAU
        else:
            while angle_end > angle_start:
                angle_end -= TAU
        return cls(
            circle=circle,
            start=start,
            end=end,
            angle_start=angle_start,
            angle_end=angle_end,
        )

    def cairo_angles(self) -> tuple[float, float]:
        return self.angle_start, self.angle_end

    def rebind_boundaries(
        self,
        start: Point | None = None,
        end: Point | None = None,
    ) -> None:
        """Move boundaries and keep their angles close to the previous arc."""
        old_start = self.angle_start
        old_end = self.angle_end

        if start is not None:
            self.start = start
        if end is not None:
            self.end = end

        self.angle_start, self.angle_end = _equivalent_angles_near_span(
            self.circle,
            self.start,
            self.end,
            old_start,
            old_end,
        )

    def close_full_circle(self) -> None:
        direction = 1.0 if self.angle_end >= self.angle_start else -1.0
        self.end = self.start
        self.angle_end = self.angle_start + direction * TAU

    def transform_boundaries(self, scale: float, dx: float, dy: float) -> None:
        self.start = (self.start[0] * scale + dx, self.start[1] * scale + dy)
        self.end = (self.end[0] * scale + dx, self.end[1] * scale + dy)


def _rebind_adjacent_intersections(arcs: list[CircleArc]) -> None:
    if len(arcs) < 2:
        return

    starts: list[Point | None] = [None] * len(arcs)
    ends: list[Point | None] = [None] * len(arcs)
    old_starts = [arc.start for arc in arcs]
    old_ends = [arc.end for arc in arcs]

    for idx, arc in enumerate(arcs):
        next_idx = (idx + 1) % len(arcs)
        next_arc = arcs[next_idx]
        candidates = _circle_intersections(arc.circle, next_arc.circle)
        if not candidates:
            continue

        boundary = _nearest_boundary_intersection(
            candidates,
            old_ends[idx],
            old_starts[next_idx],
        )
        if not _intersection_is_near_boundary(boundary, old_ends[idx], old_starts[next_idx]):
            continue

        ends[idx] = boundary
        starts[next_idx] = boundary

    for idx, arc in enumerate(arcs):
        arc.rebind_boundaries(start=starts[idx], end=ends[idx])


def build_circle_arcs(
    circles: Sequence[Circle],
    segment_points: Sequence[np.ndarray],
) -> list[CircleArc]:
    if len(circles) != len(segment_points):
        raise ValueError("circles and segment_points must have the same length")
    arcs = [
        CircleArc.from_segment_points(circle, points)
        for circle, points in zip(circles, segment_points)
    ]
    if len(arcs) == 1:
        arcs[0].close_full_circle()
        return arcs

    _rebind_adjacent_intersections(arcs)
    return arcs
