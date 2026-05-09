"""Explicit circular arc descriptions derived from fitted contour segments."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from circle_draw.circle_geometry.intersections import Circle, Point

TAU = 2 * math.pi
ANGLE_EPS = 1e-9


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

    def transform_boundaries(self, scale: float, dx: float, dy: float) -> None:
        self.start = (self.start[0] * scale + dx, self.start[1] * scale + dy)
        self.end = (self.end[0] * scale + dx, self.end[1] * scale + dy)


def build_circle_arcs(
    circles: Sequence[Circle],
    segment_points: Sequence[np.ndarray],
) -> list[CircleArc]:
    if len(circles) != len(segment_points):
        raise ValueError("circles and segment_points must have the same length")
    return [
        CircleArc.from_segment_points(circle, points)
        for circle, points in zip(circles, segment_points)
    ]
