"""End-to-end pipeline: image → contours → circle arcs → rendered PNG."""

import os
from typing import Optional

import numpy as np

from circle_draw.circle_geometry.arcs import CircleArc, build_circle_arcs
from circle_draw.point_selection.fit_quality import segment
from circle_draw.rendering.skia_arcs import render, resolve_render_shape


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
    contour_leveling_scale: float = 0.35,
    contour_turn_preserve_threshold: float = 0.75,
    contour_turn_preserve_radius: int = 1,
    min_contour_points: int = 40,
    min_circles: int = 1,
    quality_threshold: float = 1.0,
    forward_k: int = 10,
    backward_k: int = 10,
    sharp_turn_threshold: float = float("inf"),
    sharp_turn_min_count: int = 1,
    sharp_turn_method: str = "circle",
    poly_window_radius: int = 7,
    poly_high_degree: int = 5,
    poly_improvement_ratio: float = 2.0,
    max_circle_radius: float = 500.0,
    use_cv2_preprocessing: bool = True,
    cv2_equalize_hist: bool = True,
    cv2_denoise_h: float = 30.0,
    cv2_denoise_template_window: int = 3,
    cv2_denoise_search_window: int = 21,
    center_on_canvas: bool = True,
    output_padding_fraction: float = 0.08,
    background_path: Optional[str] = None,
    background_scale: float = 1.0,
    debug_dir: Optional[str] = None,
) -> None:
    """Convert image to a circle-arc drawing and save as transparent PNG.

    image_shape: (width, height, 4) — matches Cairo's width/height convention.
    threshold:   marching-squares contour level.
    contour_step: subsample every Nth contour point.
    contour_leveling_scale: contour down/up resampling scale used to level noise.
    contour_turn_preserve_threshold: preserve points whose local turn exceeds this radians threshold.
    contour_turn_preserve_radius: keep this many neighbours around preserved turn points.
    min_contour_points: skip contours shorter than this.
    min_circles: skip contours that produce fewer fitted circles than this.
    forward_k: max number of new points to try accepting in one refit batch.
    sharp_turn_threshold: reject candidate batches containing any local turn above this radians threshold.
    sharp_turn_min_count: reject only when at least this many points in a candidate batch are sharp.
    sharp_turn_method: sharp-turn detector to use: circle or poly.
    poly_window_radius: half-window (in points) for local polynomial detector.
    poly_high_degree: high polynomial degree used against quadratic baseline.
    poly_improvement_ratio: reject when high-degree fit improves over quadratic by at least this ratio.
    max_circle_radius: maximum allowed fitted circle radius; circles above this are rejected.
    use_cv2_preprocessing: if True, equalise + denoise before extracting contours.
    cv2_equalize_hist: apply cv2 histogram equalization before contour extraction.
    cv2_denoise_h: denoising strength for cv2 fastNlMeansDenoising; 0 disables denoising.
    cv2_denoise_template_window: denoising template window size.
    cv2_denoise_search_window: denoising search window size.
    center_on_canvas: if True, fit and center kept contours on the output canvas.
    output_padding_fraction: canvas fraction reserved as padding on each side.
    background_path: optional image to draw behind the final render.
    background_scale: resize factor applied to background image before compositing.
    debug_dir:   if set, save step-by-step plots here.
    """
    render_shape = resolve_render_shape(image_shape, background_path, background_scale=background_scale)

    if use_cv2_preprocessing:
        from circle_draw.contour_extraction.cv2_enhanced import load_and_extract
    else:
        from circle_draw.contour_extraction.skimage_marching import load_and_extract

    if use_cv2_preprocessing:
        contours_all = load_and_extract(
            image_path,
            threshold=threshold,
            step=contour_step,
            equalize_hist=cv2_equalize_hist,
            denoise_h=cv2_denoise_h,
            denoise_template_window=cv2_denoise_template_window,
            denoise_search_window=cv2_denoise_search_window,
        )
    else:
        contours_all = load_and_extract(image_path, threshold=threshold, step=contour_step)

    from circle_draw.contour_extraction.leveling import level_contours
    contours_all = level_contours(
        contours_all,
        downsample_scale=contour_leveling_scale,
        preserve_turn_threshold=contour_turn_preserve_threshold,
        preserve_radius=contour_turn_preserve_radius,
    )

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

        _, circles, segment_points, segment_metadata = segment(
            points,
            quality_threshold=quality_threshold,
            forward_k=forward_k,
            backward_k=backward_k,
            sharp_turn_threshold=sharp_turn_threshold,
            sharp_turn_min_count=sharp_turn_min_count,
            sharp_turn_method=sharp_turn_method,
            poly_window_radius=poly_window_radius,
            poly_high_degree=poly_high_degree,
            poly_improvement_ratio=poly_improvement_ratio,
            max_circle_radius=max_circle_radius,
            frame_callback=frame_cb,
        )

        if collector is not None:
            collector.on_contour_done(points)
            _video_idx += 1

        if len(circles) < min_circles:
            continue

        if np.isfinite(max_circle_radius):
            kept_pairs = [
                (c, s, m) for c, s, m in zip(circles, segment_points, segment_metadata)
                if abs(float(c[2])) <= float(max_circle_radius)
            ]
            if not kept_pairs:
                continue
            circles = [c for c, _, _ in kept_pairs]
            segment_points = [s for _, s, _ in kept_pairs]
            segment_metadata = [m for _, _, m in kept_pairs]

        if debug_dir:
            from circle_draw.debug_plots import save_turn_sharpness_map
            save_turn_sharpness_map(
                points=points,
                segment_points=segment_points,
                segment_metadata=segment_metadata,
                contour_idx=_video_idx - 1 if collector is not None else len(all_contours),
                sharp_turn_threshold=sharp_turn_threshold,
                debug_dir=debug_dir,
            )

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
    render(
        all_arcs,
        render_shape,
        output_path,
        background_path=background_path,
        background_scale=background_scale,
    )
    print(f"Saved → {output_path}")

    if collector is not None:
        collector.save_videos()
