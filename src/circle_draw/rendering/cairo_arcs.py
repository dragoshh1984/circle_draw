"""Cairo-based renderer: draws each selected circle arc opaque."""

import math
import numpy as np
import cairo
import imageio.v3 as iio

from circle_draw.circle_geometry.arcs import CircleArc
from circle_draw.circle_geometry.intersections import Circle, Point

# ARGB channel order when using cairo.FORMAT_ARGB32 with numpy buffers.
# image_shape convention: (width, height, 4) — width=cairo_width, height=cairo_height.


def _to_rgba(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0) * 255 if image.dtype.kind == "f" else np.clip(image, 0, 255)
        image = image.astype(np.uint8)

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)

    if image.shape[2] == 3:
        alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
        image = np.concatenate((image, alpha), axis=2)
    elif image.shape[2] != 4:
        raise ValueError("background image must have 1, 3, or 4 channels")

    return image


def resolve_render_shape(
    image_shape: tuple[int, int, int],
    background_path: str | None,
    background_scale: float = 1.0,
) -> tuple[int, int, int]:
    if not background_path:
        return image_shape

    width, height, channels = image_shape
    bg_shape = iio.improps(background_path).shape
    bg_height, bg_width = bg_shape[:2]
    scale = max(0.01, float(background_scale))
    scaled_w = max(1, int(round(bg_width * scale)))
    scaled_h = max(1, int(round(bg_height * scale)))
    if bg_width >= width and bg_height >= height:
        if scaled_w >= width and scaled_h >= height:
            return (scaled_w, scaled_h, channels)
        return image_shape
    return image_shape


def _fit_background_to_canvas(image: np.ndarray, width: int, height: int) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    if image_width == width and image_height == height:
        return image

    # If larger than canvas, center-crop. If smaller, center-pad with transparency.
    x0_src = max(0, (image_width - width) // 2)
    y0_src = max(0, (image_height - height) // 2)
    x1_src = min(image_width, x0_src + width)
    y1_src = min(image_height, y0_src + height)
    crop = image[y0_src:y1_src, x0_src:x1_src]

    out = np.zeros((height, width, 4), dtype=np.uint8)
    x0_dst = max(0, (width - crop.shape[1]) // 2)
    y0_dst = max(0, (height - crop.shape[0]) // 2)
    out[y0_dst:y0_dst + crop.shape[0], x0_dst:x0_dst + crop.shape[1]] = crop
    return out


def _load_background(background_path: str, width: int, height: int, background_scale: float = 1.0) -> np.ndarray:
    image = _to_rgba(iio.imread(background_path))
    scale = max(0.01, float(background_scale))
    if abs(scale - 1.0) > 1e-9:
        import cv2
        image_height, image_width = image.shape[:2]
        scaled_width = max(1, int(round(image_width * scale)))
        scaled_height = max(1, int(round(image_height * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        image = cv2.resize(image, (scaled_width, scaled_height), interpolation=interpolation)
    return _fit_background_to_canvas(image, width, height)


def _composite_over_background(layer: np.ndarray, background: np.ndarray) -> np.ndarray:
    src = layer.astype(np.float32) / 255.0
    bg = background.astype(np.float32) / 255.0

    src_alpha = src[:, :, 3:4]
    bg_alpha = bg[:, :, 3:4]
    src_rgb_premul = src[:, :, [2, 1, 0]]
    bg_rgb_premul = bg[:, :, :3] * bg_alpha

    out_alpha = src_alpha + bg_alpha * (1.0 - src_alpha)
    out_rgb_premul = src_rgb_premul + bg_rgb_premul * (1.0 - src_alpha)
    out_rgb = np.divide(
        out_rgb_premul,
        out_alpha,
        out=np.zeros_like(out_rgb_premul),
        where=out_alpha > 0,
    )

    out = np.concatenate((out_rgb, out_alpha), axis=2)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def arc_boundary_angles(
    point1: Point,
    point2: Point,
    circle: Circle,
    clockwise: bool = True,
) -> tuple[float, float]:
    """Return (angle_start, angle_stop) for the arc from point1 to point2 on the circle.

    Angles are in radians, measured clockwise from the positive-x axis (Cairo convention).
    """
    return CircleArc.from_boundaries(
        circle=circle,
        start=point1,
        end=point2,
        clockwise=clockwise,
    ).cairo_angles()


def render(
    arcs_per_contour: list[list[CircleArc]],
    image_shape: tuple[int, int, int],
    output_path: str,
    background_path: str | None = None,
    background_scale: float = 1.0,
    arc_color: tuple[float, float, float] = (1.0, 0.2, 0.2),
    ghost_alpha: float = 0.1,
    arc_line_width: float = 3.0,
) -> None:
    """Render all arcs onto a transparent canvas and write to output_path.

    image_shape: (width, height, channels=4)
    """
    width, height, channels = image_shape
    assert channels == 4, "Cairo requires 4-channel ARGB image"

    layer = np.zeros((height, width, channels), dtype=np.uint8)
    surface = cairo.ImageSurface.create_for_data(layer, cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(surface)
    cr.set_source_rgba(1.0, 1.0, 1.0, 0.0)
    cr.paint()

    r, g, b = arc_color

    for arcs in arcs_per_contour:
        for selected_arc in arcs:
            circle = selected_arc.circle

            if ghost_alpha > 0:
                cr.set_source_rgba(r, g, b, ghost_alpha)
                cr.arc(circle[0], circle[1], circle[2], 0, 2 * math.pi)
                cr.set_line_width(arc_line_width)
                cr.stroke()

            # draw the active arc fully opaque
            a_start, a_stop = selected_arc.cairo_angles()
            cr.set_source_rgba(r, g, b, 1.0)
            if selected_arc.clockwise:
                cr.arc(circle[0], circle[1], circle[2], a_start, a_stop)
            else:
                cr.arc_negative(circle[0], circle[1], circle[2], a_start, a_stop)
            cr.set_line_width(arc_line_width)
            cr.stroke()

    surface.flush()
    if background_path:
        background = _load_background(background_path, width, height, background_scale=background_scale)
        iio.imwrite(output_path, _composite_over_background(layer, background))
    else:
        surface.write_to_png(output_path)
