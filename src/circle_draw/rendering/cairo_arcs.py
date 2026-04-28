"""Cairo-based renderer: draws each circle arc opaque where the contour was, faint elsewhere."""

import math
import numpy as np
import cairo

from circle_draw.circle_geometry.intersections import Circle, Point

# ARGB channel order when using cairo.FORMAT_ARGB32 with numpy buffers.
# image_shape convention: (width, height, 4) — width=cairo_width, height=cairo_height.


def _quadrant(point: Point, circle: Circle) -> int:
    """Return screen quadrant of point relative to circle center (image y-axis points down)."""
    cx, cy = circle[0], circle[1]
    if point[0] >= cx and point[1] <= cy:
        return 1
    elif point[0] <= cx and point[1] <= cy:
        return 2
    elif point[0] <= cx and point[1] >= cy:
        return 3
    return 4


def arc_boundary_angles(point1: Point, point2: Point, circle: Circle) -> tuple[float, float]:
    """Return (angle_start, angle_stop) for the arc from point1 to point2 on the circle.

    Angles are in radians, measured clockwise from the positive-x axis (Cairo convention).
    """
    r = circle[2]
    zero_pt = (circle[0] + r, circle[1])

    def chord_to_angle(pt: Point) -> float:
        dx = zero_pt[0] - pt[0]
        dy = zero_pt[1] - pt[1]
        chord = math.sqrt(dx**2 + dy**2)
        arg = max(-1.0, min(1.0, 1 - chord**2 / (2 * r**2)))
        a = np.arccos(arg)
        q = _quadrant(pt, circle)
        if q == 1:
            a = math.pi / 2 - a + 3 / 2 * math.pi
        elif q == 2:
            a = 2 * math.pi - a
        return a

    a1 = chord_to_angle(point1)
    a2 = chord_to_angle(point2)

    q1 = _quadrant(point1, circle)
    q2 = _quadrant(point2, circle)
    if (q1 == 1 and q2 == 4) or (q2 == 1 and q1 == 4):
        return max(a1, a2), min(a1, a2)
    return min(a1, a2), max(a1, a2)


def render(
    contours: list[np.ndarray],
    circles_per_contour: list[list[Circle]],
    boundaries_per_contour: list[list[list[Point]]],
    image_shape: tuple[int, int, int],
    output_path: str,
    arc_color: tuple[float, float, float] = (1.0, 0.2, 0.2),
    ghost_alpha: float = 0.1,
    arc_line_width: float = 3.0,
) -> None:
    """Render all arcs onto a transparent canvas and write to output_path.

    image_shape: (width, height, channels=4)
    """
    width, height, channels = image_shape
    assert channels == 4, "Cairo requires 4-channel ARGB image"

    background = np.zeros(image_shape, dtype=np.uint8)
    surface = cairo.ImageSurface.create_for_data(background, cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(surface)
    cr.set_source_rgba(1.0, 1.0, 1.0, 0.0)
    cr.paint()

    r, g, b = arc_color

    for circles, boundaries in zip(circles_per_contour, boundaries_per_contour):
        for i, circle in enumerate(circles):
            fp = boundaries[i][0]
            sp = boundaries[i][1]

            # draw full circle faintly
            cr.set_source_rgba(r, g, b, ghost_alpha)
            cr.arc(circle[0], circle[1], circle[2], 0, 2 * math.pi)
            cr.set_line_width(arc_line_width)
            cr.stroke()

            # draw the active arc fully opaque
            a_start, a_stop = arc_boundary_angles(sp, fp, circle)
            cr.set_source_rgba(r, g, b, 1.0)
            cr.arc(circle[0], circle[1], circle[2], a_start, a_stop)
            cr.set_line_width(arc_line_width)
            cr.stroke()

    surface.write_to_png(output_path)
