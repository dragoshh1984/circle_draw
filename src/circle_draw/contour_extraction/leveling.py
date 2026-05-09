"""Contour leveling to reduce noise while preserving true turns.

The operation mimics image downscale/upscale on a closed polyline:
1) resample contour to fewer points,
2) resample back to original count,
3) restore points around strong turn anchors from the original contour.
"""

import math

import numpy as np


def _resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    n = len(points)
    if n == 0 or count <= 0:
        return points.copy()
    if n == 1:
        return np.repeat(points, count, axis=0)

    t_old = np.linspace(0.0, 1.0, n, endpoint=False)
    t_new = np.linspace(0.0, 1.0, count, endpoint=False)

    # Close the loop for periodic interpolation.
    t_old_ext = np.concatenate((t_old, np.array([1.0], dtype=float)))
    x_ext = np.concatenate((points[:, 0], np.array([points[0, 0]], dtype=float)))
    y_ext = np.concatenate((points[:, 1], np.array([points[0, 1]], dtype=float)))

    x_new = np.interp(t_new, t_old_ext, x_ext)
    y_new = np.interp(t_new, t_old_ext, y_ext)
    return np.stack((x_new, y_new), axis=1)


def _turn_angle(points: np.ndarray, index: int) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    prev_point = points[(index - 1) % n]
    curr_point = points[index]
    next_point = points[(index + 1) % n]

    v1 = curr_point - prev_point
    v2 = next_point - curr_point
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 <= np.finfo(float).eps or norm2 <= np.finfo(float).eps:
        return 0.0

    dot = float(np.dot(v1, v2)) / (norm1 * norm2)
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(math.acos(dot))


def level_contour(
    points: np.ndarray,
    downsample_scale: float,
    preserve_turn_threshold: float,
    preserve_radius: int,
) -> np.ndarray:
    """Return leveled contour of same shape, preserving strong turns.

    downsample_scale: fraction of points to keep during coarse pass (0, 1].
    preserve_turn_threshold: raw local heading change threshold in radians.
    preserve_radius: number of neighbouring indices to preserve around anchors.
    """
    n = len(points)
    if n < 6 or downsample_scale >= 0.999:
        return points.copy()

    scale = max(0.05, min(1.0, float(downsample_scale)))
    coarse_count = max(6, int(round(n * scale)))
    coarse_count = min(coarse_count, n)

    coarse = _resample_closed(points, coarse_count)
    leveled = _resample_closed(coarse, n)

    if not np.isfinite(preserve_turn_threshold):
        return leveled

    radius = max(0, int(preserve_radius))
    for i in range(n):
        if _turn_angle(points, i) > preserve_turn_threshold:
            for offset in range(-radius, radius + 1):
                idx = (i + offset) % n
                leveled[idx] = points[idx]

    return leveled


def level_contours(
    contours: list[np.ndarray],
    downsample_scale: float,
    preserve_turn_threshold: float,
    preserve_radius: int,
) -> list[np.ndarray]:
    """Apply contour leveling to every contour array."""
    return [
        level_contour(c, downsample_scale, preserve_turn_threshold, preserve_radius)
        for c in contours
    ]
