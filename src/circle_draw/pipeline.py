"""End-to-end pipeline: image → contours → circle arcs → rendered PNG."""

import os
from typing import Optional

import numpy as np

from circle_draw.circle_geometry.arcs import CircleArc, build_circle_arcs
from circle_draw.point_selection.fit_quality import segment
from circle_draw.rendering.cairo_arcs import render, resolve_render_shape


def _centering_transform(
    contours: list[np.ndarray],
    image_shape: tuple[int, int, int],
    padding_fraction: float = 0.08,
    offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, float]:
    """Return scale, dx, dy that fit kept contours and center them on the canvas."""
    if not contours:
        return 1.0, 0.0, 0.0

    points = np.concatenate(contours)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        return 1.0, 0.0, 0.0

    width, height = image_shape[0], image_shape[1]
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    bbox_w, bbox_h = max_xy - min_xy

    available_w = width * max(0.0, 1.0 - 2 * padding_fraction)
    available_h = height * max(0.0, 1.0 - 2 * padding_fraction)

    if bbox_w == 0 and bbox_h == 0:
        scale = 1.0
    elif bbox_w == 0:
        scale = available_h / bbox_h
    elif bbox_h == 0:
        scale = available_w / bbox_w
    else:
        scale = min(available_w / bbox_w, available_h / bbox_h)

    object_center = (min_xy + max_xy) / 2
    canvas_center = np.array([width / 2, height / 2])
    dx, dy = canvas_center - object_center * scale
    dx += offset[0]
    dy += offset[1]
    return float(scale), float(dx), float(dy)


def _centered_canvas_offset(
    canvas_shape: tuple[int, int, int],
    render_shape: tuple[int, int, int],
) -> tuple[float, float]:
    canvas_w, canvas_h = canvas_shape[0], canvas_shape[1]
    render_w, render_h = render_shape[0], render_shape[1]
    return ((render_w - canvas_w) / 2, (render_h - canvas_h) / 2)


def _apply_transform(
    contours_all: list[np.ndarray],
    circles_per_contour: list[list[np.ndarray]],
    arcs_per_contour: list[list[CircleArc]],
    scale: float,
    dx: float,
    dy: float,
) -> None:
    """Apply an affine scale/translate to contours, circles, and arc boundaries."""
    for contour in contours_all:
        contour[:, 0] = contour[:, 0] * scale + dx
        contour[:, 1] = contour[:, 1] * scale + dy

    for circles in circles_per_contour:
        for circle in circles:
            circle[0] = circle[0] * scale + dx
            circle[1] = circle[1] * scale + dy
            circle[2] = circle[2] * scale

    for arcs in arcs_per_contour:
        for arc in arcs:
            arc.transform_boundaries(scale, dx, dy)


def run(
    image_path: str,
    output_path: str,
    image_shape: tuple[int, int, int] = (1920, 1080, 4),
    threshold: float = 199.0,
    contour_step: int = 5,
    min_contour_points: int = 40,
    min_circles: int = 1,
    quality_threshold: float = 1.0,
    forward_k: int = 10,
    backward_k: int = 10,
    use_cv2_preprocessing: bool = True,
    center_on_canvas: bool = True,
    output_padding_fraction: float = 0.08,
    background_path: Optional[str] = None,
    debug_dir: Optional[str] = None,
) -> None:
    """Convert image to a circle-arc drawing and save as transparent PNG.

    image_shape: (width, height, 4) — matches Cairo's width/height convention.
    threshold:   marching-squares contour level.
    contour_step: subsample every Nth contour point.
    min_contour_points: skip contours shorter than this.
    min_circles: skip contours that produce fewer fitted circles than this.
    forward_k: max number of new points to try accepting in one refit batch.
    use_cv2_preprocessing: if True, equalise + denoise before extracting contours.
    center_on_canvas: if True, fit and center kept contours on the output canvas.
    output_padding_fraction: canvas fraction reserved as padding on each side.
    background_path: optional image to draw behind the final render.
    debug_dir:   if set, save step-by-step plots here.
    """
    render_shape = resolve_render_shape(image_shape, background_path)

    if use_cv2_preprocessing:
        from circle_draw.contour_extraction.cv2_enhanced import load_and_extract
    else:
        from circle_draw.contour_extraction.skimage_marching import load_and_extract

    contours_all = load_and_extract(image_path, threshold=threshold, step=contour_step)

    # Set up video collector if debug is requested
    collector = None
    if debug_dir:
        from circle_draw.debug_video import VideoCollector
        os.makedirs(debug_dir, exist_ok=True)
        collector = VideoCollector(image_shape, debug_dir)
        eligible = [p for p in contours_all if len(p) >= min_contour_points]
        collector.set_all_contours(eligible)
        _video_idx   = 0
        _video_total = len(eligible)

    all_contours: list[np.ndarray] = []
    all_circles: list[list] = []
    all_arcs: list[list[CircleArc]] = []

    for points in contours_all:
        if len(points) < min_contour_points:
            continue

        frame_cb = None
        if collector is not None:
            collector.set_contour(points, _video_idx, _video_total)
            frame_cb = collector.make_callback()

        _, circles, segment_points = segment(
            points,
            quality_threshold=quality_threshold,
            forward_k=forward_k,
            backward_k=backward_k,
            frame_callback=frame_cb,
        )

        if collector is not None:
            collector.on_contour_done(points)
            _video_idx += 1

        if len(circles) < min_circles:
            continue

        arcs = build_circle_arcs(circles, segment_points)

        all_contours.append(points)
        all_circles.append(circles)
        all_arcs.append(arcs)

    if center_on_canvas:
        scale, dx, dy = _centering_transform(
            all_contours,
            image_shape,
            output_padding_fraction,
            offset=_centered_canvas_offset(image_shape, render_shape),
        )
        _apply_transform(contours_all, all_circles, all_arcs, scale, dx, dy)

    print(f"Rendering {len(all_contours)} contours ({sum(len(c) for c in all_circles)} circles total)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    render(all_arcs, render_shape, output_path, background_path=background_path)
    print(f"Saved → {output_path}")

    if collector is not None:
        collector.save_videos()
