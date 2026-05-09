"""End-to-end pipeline: image → contours → circle arcs → rendered PNG."""

import os
from typing import Optional

import numpy as np

from circle_draw.circle_geometry.intersections import build_arc_boundaries
from circle_draw.point_selection.fit_quality import segment
from circle_draw.rendering.cairo_arcs import render


def _centering_transform(
    contours: list[np.ndarray],
    image_shape: tuple[int, int, int],
    padding_fraction: float = 0.08,
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
    return float(scale), float(dx), float(dy)


def _apply_transform(
    contours_all: list[np.ndarray],
    circles_per_contour: list[list[np.ndarray]],
    boundaries_per_contour: list[list[list[tuple[float, float]]]],
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

    for boundaries in boundaries_per_contour:
        for boundary_pair in boundaries:
            for i, point in enumerate(boundary_pair):
                if np.isnan(point[0]) or np.isnan(point[1]):
                    continue
                boundary_pair[i] = (point[0] * scale + dx, point[1] * scale + dy)


def run(
    image_path: str,
    output_path: str,
    image_shape: tuple[int, int, int] = (1920, 1080, 4),
    threshold: float = 199.0,
    contour_step: int = 5,
    min_contour_points: int = 40,
    min_circles: int = 3,
    quality_threshold: float = 0.8,
    backward_k: int = 10,
    use_cv2_preprocessing: bool = True,
    center_on_canvas: bool = True,
    output_padding_fraction: float = 0.08,
    debug_dir: Optional[str] = None,
) -> None:
    """Convert image to a circle-arc drawing and save as transparent PNG.

    image_shape: (width, height, 4) — matches Cairo's width/height convention.
    threshold:   marching-squares contour level.
    contour_step: subsample every Nth contour point.
    min_contour_points: skip contours shorter than this.
    min_circles: skip contours that produce fewer circles than this.
    use_cv2_preprocessing: if True, equalise + denoise before extracting contours.
    center_on_canvas: if True, fit and center kept contours on the output canvas.
    output_padding_fraction: canvas fraction reserved as padding on each side.
    debug_dir:   if set, save step-by-step plots here.
    """
    if use_cv2_preprocessing:
        from circle_draw.contour_extraction.cv2_enhanced import load_and_extract
    else:
        from circle_draw.contour_extraction.skimage_marching import load_and_extract

    if debug_dir:
        from circle_draw import debug_plots

    contours_all = load_and_extract(image_path, threshold=threshold, step=contour_step)

    all_contours: list[np.ndarray] = []
    all_circles: list[list] = []
    all_boundaries: list[list] = []

    for points in contours_all:
        if len(points) < min_contour_points:
            continue

        end_pts, circles = segment(points, quality_threshold=quality_threshold, backward_k=backward_k)

        if len(circles) < min_circles:
            continue

        boundaries = build_arc_boundaries(circles, end_pts)

        all_contours.append(points)
        all_circles.append(circles)
        all_boundaries.append(boundaries)

    if center_on_canvas:
        scale, dx, dy = _centering_transform(all_contours, image_shape, output_padding_fraction)
        _apply_transform(contours_all, all_circles, all_boundaries, scale, dx, dy)

    print(f"Rendering {len(all_contours)} contours ({sum(len(c) for c in all_circles)} circles total)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    render(all_contours, all_circles, all_boundaries, image_shape, output_path)
    print(f"Saved → {output_path}")

    if debug_dir:
        print(f"Saving debug plots → {debug_dir}/")
        debug_plots.save_contours_overview(contours_all, all_contours, image_shape, debug_dir)
        # limit per-contour plots to avoid flooding disk on images with many contours
        max_contour_plots = 50
        for i, (points, circles, boundaries) in enumerate(zip(all_contours, all_circles, all_boundaries)):
            if i >= max_contour_plots:
                print(f"  (skipping debug plots for remaining {len(all_contours) - max_contour_plots} contours)")
                break
            debug_plots.save_arc_segmentation(points, circles, boundaries, i, debug_dir)
            debug_plots.save_arc_boundaries(points, circles, boundaries, i, debug_dir)
        print(f"Debug plots saved ({min(len(all_contours), max_contour_plots)} contours)")
