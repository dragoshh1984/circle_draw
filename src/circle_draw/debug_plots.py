"""Debug visualisations saved to disk — one directory per run."""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle as MplCircle

from circle_draw.circle_geometry.intersections import Circle, Point


def _ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ── Step 1: contour extraction ────────────────────────────────────────────────

def save_contours_overview(
    contours_all: list[np.ndarray],
    contours_kept: list[np.ndarray],
    image_shape: tuple[int, int, int],
    debug_dir: str,
) -> None:
    """All extracted contours (grey) with kept ones highlighted."""
    _ensure(debug_dir)
    width, height = image_shape[0], image_shape[1]

    fig, ax = plt.subplots(figsize=(10, 10 * height / width))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_title(f"Contour extraction — {len(contours_kept)}/{len(contours_all)} kept")
    ax.axis("off")

    kept_set = {id(c) for c in contours_kept}
    for c in contours_all:
        color = "steelblue" if id(c) in kept_set else "#cccccc"
        alpha = 0.8 if id(c) in kept_set else 0.3
        ax.plot(c[:, 0], c[:, 1], color=color, linewidth=0.8, alpha=alpha)

    fig.tight_layout()
    fig.savefig(os.path.join(debug_dir, "01_contours.png"), dpi=120)
    plt.close(fig)


# ── Step 2: arc segmentation ──────────────────────────────────────────────────

def save_arc_segmentation(
    points: np.ndarray,
    circles: list[Circle],
    boundaries: list[list[Point]],
    contour_idx: int,
    debug_dir: str,
) -> None:
    """Contour points coloured by segment + fitted circles."""
    _ensure(debug_dir)

    all_x = np.concatenate([points[:, 0], [c[0] for c in circles]])
    all_y = np.concatenate([points[:, 1], [c[1] for c in circles]])
    pad = max((all_x.max() - all_x.min()), (all_y.max() - all_y.min())) * 0.1 + 10
    xmin, xmax = all_x.min() - pad, all_x.max() + pad
    ymin, ymax = all_y.min() - pad, all_y.max() + pad

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # y-axis flipped for image coordinates
    ax.set_aspect("equal")
    ax.set_title(f"Contour {contour_idx:03d} — {len(circles)} segments")
    ax.axis("off")

    cmap = plt.get_cmap("tab20")
    n = len(circles)

    ax.scatter(points[:, 0], points[:, 1], s=2, color="#aaaaaa", zorder=1)

    for i, circle in enumerate(circles):
        color = cmap(i % 20)
        fp = boundaries[i][0]
        sp = boundaries[i][1]

        circ = MplCircle(
            (circle[0], circle[1]), circle[2],
            fill=False, edgecolor=color, linewidth=1.2, alpha=0.6, zorder=2,
        )
        ax.add_patch(circ)

        for pt in (fp, sp):
            if not (np.isnan(pt[0]) or np.isnan(pt[1])):
                ax.plot(pt[0], pt[1], "o", color=color, markersize=5, zorder=3)

    fig.tight_layout()
    fig.savefig(os.path.join(debug_dir, f"02_segmentation_{contour_idx:03d}.png"), dpi=120)
    plt.close(fig)


# ── Step 3: arc boundaries ────────────────────────────────────────────────────

def save_arc_boundaries(
    points: np.ndarray,
    circles: list[Circle],
    boundaries: list[list[Point]],
    contour_idx: int,
    debug_dir: str,
) -> None:
    """Each circle drawn faint; active arc drawn bold between its boundary points."""
    _ensure(debug_dir)

    all_x = np.concatenate([points[:, 0], [c[0] for c in circles]])
    all_y = np.concatenate([points[:, 1], [c[1] for c in circles]])
    pad = max((all_x.max() - all_x.min()), (all_y.max() - all_y.min())) * 0.1 + 10
    xmin, xmax = all_x.min() - pad, all_x.max() + pad
    ymin, ymax = all_y.min() - pad, all_y.max() + pad

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.set_title(f"Contour {contour_idx:03d} — arc boundaries")
    ax.axis("off")

    cmap = plt.get_cmap("tab20")

    ax.scatter(points[:, 0], points[:, 1], s=2, color="#dddddd", zorder=1)

    for i, circle in enumerate(circles):
        color = cmap(i % 20)
        cx, cy, r = circle[0], circle[1], circle[2]
        fp = boundaries[i][0]
        sp = boundaries[i][1]

        # ghost circle
        ghost = MplCircle((cx, cy), r, fill=False, edgecolor=color, linewidth=0.7, alpha=0.25, zorder=2)
        ax.add_patch(ghost)

        # active arc — compute angle range
        if not (np.isnan(fp[0]) or np.isnan(sp[0])):
            try:
                from circle_draw.rendering.skia_arcs import arc_boundary_angles
                a_start, a_stop = arc_boundary_angles(sp, fp, circle)
                # matplotlib Arc uses degrees, counter-clockwise from east
                # arc angles here are clockwise in screen coordinates (y-down)
                # converting: matplotlib angle = -arc_angle (degrees)
                deg_start = -math.degrees(a_start)
                deg_stop = -math.degrees(a_stop)
                arc = Arc(
                    (cx, cy), 2 * r, 2 * r,
                    angle=0, theta1=min(deg_start, deg_stop), theta2=max(deg_start, deg_stop),
                    color=color, linewidth=2.0, zorder=3,
                )
                ax.add_patch(arc)
            except Exception:
                pass  # skip degenerate arcs

        for pt in (fp, sp):
            if not (np.isnan(pt[0]) or np.isnan(pt[1])):
                ax.plot(pt[0], pt[1], "o", color=color, markersize=6, zorder=4)

    fig.tight_layout()
    fig.savefig(os.path.join(debug_dir, f"03_boundaries_{contour_idx:03d}.png"), dpi=120)
    plt.close(fig)


def save_turn_sharpness_map(
    points: np.ndarray,
    segment_points: list[np.ndarray],
    segment_metadata: list,
    contour_idx: int,
    sharp_turn_threshold: float,
    debug_dir: str,
) -> None:
    """Plot local turn angles per point and highlight sharp points."""
    _ensure(debug_dir)

    all_x = points[:, 0]
    all_y = points[:, 1]
    pad = max((all_x.max() - all_x.min()), (all_y.max() - all_y.min())) * 0.1 + 10
    xmin, xmax = all_x.min() - pad, all_x.max() + pad
    ymin, ymax = all_y.min() - pad, all_y.max() + pad

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.axis("off")

    all_turn = np.concatenate([m.turn_angles for m in segment_metadata]) if segment_metadata else np.array([0.0])
    vmax = max(float(np.max(all_turn)), max(1e-6, sharp_turn_threshold if np.isfinite(sharp_turn_threshold) else 0.0))

    sharp_total = 0
    for seg_pts, seg_meta in zip(segment_points, segment_metadata):
        turn_angles = np.array(seg_meta.turn_angles, dtype=float)
        is_sharp = np.array(seg_meta.is_sharp, dtype=bool)
        sharp_total += int(np.sum(is_sharp))

        scatter = ax.scatter(
            seg_pts[:, 0],
            seg_pts[:, 1],
            c=turn_angles,
            cmap="plasma",
            vmin=0.0,
            vmax=vmax,
            s=14,
            zorder=2,
        )

        if np.any(is_sharp):
            sharp_pts = seg_pts[is_sharp]
            ax.scatter(
                sharp_pts[:, 0],
                sharp_pts[:, 1],
                facecolors="none",
                edgecolors="#00ffff",
                linewidths=0.9,
                s=42,
                zorder=3,
            )

    title_threshold = "inf" if not np.isfinite(sharp_turn_threshold) else f"{sharp_turn_threshold:.3f}"
    ax.set_title(
        f"Contour {contour_idx:03d} — turn-angle sharpness (sharp {sharp_total}, thr={title_threshold} rad)"
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("local turn angle [rad]")
    if np.isfinite(sharp_turn_threshold):
        cbar.ax.axhline(sharp_turn_threshold, color="cyan", linewidth=1.0)

    fig.tight_layout()
    fig.savefig(os.path.join(debug_dir, f"04_turn_sharpness_{contour_idx:03d}.png"), dpi=120)
    plt.close(fig)
