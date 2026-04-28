"""End-to-end pipeline: image → contours → circle arcs → rendered PNG."""

import os
from typing import Optional

import numpy as np

from circle_draw.circle_geometry.intersections import build_arc_boundaries
from circle_draw.point_selection.fit_quality import segment
from circle_draw.rendering.cairo_arcs import render


def run(
    image_path: str,
    output_path: str,
    image_shape: tuple[int, int, int] = (1600, 2100, 4),
    threshold: float = 199.0,
    contour_step: int = 5,
    min_contour_points: int = 40,
    min_circles: int = 3,
    quality_threshold: float = 0.8,
    use_cv2_preprocessing: bool = True,
    debug_dir: Optional[str] = None,
) -> None:
    """Convert image to a circle-arc drawing and save as transparent PNG.

    image_shape: (width, height, 4) — matches Cairo's width/height convention.
    threshold:   marching-squares contour level.
    contour_step: subsample every Nth contour point.
    min_contour_points: skip contours shorter than this.
    min_circles: skip contours that produce fewer circles than this.
    use_cv2_preprocessing: if True, equalise + denoise before extracting contours.
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

        end_pts, circles = segment(points, quality_threshold=quality_threshold)

        if len(circles) < min_circles:
            continue

        boundaries = build_arc_boundaries(circles, end_pts)

        all_contours.append(points)
        all_circles.append(circles)
        all_boundaries.append(boundaries)

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
