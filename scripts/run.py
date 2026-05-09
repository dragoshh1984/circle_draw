"""CLI entry point for the circle-draw pipeline."""

import argparse
import json
import os
from datetime import datetime

DEFAULT_BACKGROUND = "input/background_sample.png"


def _make_run_dir(output_dir: str, image_path: str) -> tuple[str, str]:
    """Create a timestamped run directory and return (run_dir, output_png_path)."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{timestamp}_{stem}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, os.path.join(run_dir, "result.png")


def _save_config(args: argparse.Namespace, run_dir: str, output_path: str) -> None:
    params = {k: v for k, v in vars(args).items()}

    flags: list[str] = ["uv run python scripts/run.py", args.image, args.output_dir]
    defaults = {
        "width": 1920, "height": 1080, "threshold": 199.0, "step": 5,
        "contour_leveling_scale": 0.35,
        "contour_turn_preserve_threshold": 0.75,
        "contour_turn_preserve_radius": 1,
        "min_points": 40, "min_circles": 1, "quality_threshold": 1.0,
        "forward_k": 10, "backward_k": 10,
        "sharp_turn_threshold": float("inf"), "sharp_turn_min_count": 1,
        "sharp_turn_method": "circle", "poly_window_radius": 7,
        "poly_high_degree": 5, "poly_improvement_ratio": 2.0,
        "max_circle_radius": 500.0,
        "background": DEFAULT_BACKGROUND,
        "no_cv2": False, "no_center": False, "padding": 0.08, "debug": False,
    }
    flag_map = {
        "width": "--width", "height": "--height", "threshold": "--threshold",
        "step": "--step", "min_points": "--min-points", "min_circles": "--min-circles",
        "contour_leveling_scale": "--contour-leveling-scale",
        "contour_turn_preserve_threshold": "--contour-turn-preserve-threshold",
        "contour_turn_preserve_radius": "--contour-turn-preserve-radius",
        "quality_threshold": "--quality-threshold", "forward_k": "--forward-k", "backward_k": "--backward-k",
        "sharp_turn_threshold": "--sharp-turn-threshold", "sharp_turn_min_count": "--sharp-turn-min-count",
        "sharp_turn_method": "--sharp-turn-method",
        "poly_window_radius": "--poly-window-radius",
        "poly_high_degree": "--poly-high-degree",
        "poly_improvement_ratio": "--poly-improvement-ratio",
        "max_circle_radius": "--max-circle-radius",
        "background": "--background", "no_cv2": "--no-cv2", "no_center": "--no-center",
        "padding": "--padding", "debug": "--debug",
    }
    bool_flags = {"no_cv2", "no_center", "debug"}
    for key, flag in flag_map.items():
        val = params.get(key)
        default = defaults.get(key)
        if key in bool_flags:
            if val:
                flags.append(flag)
        elif val != default and val is not None:
            flags.append(f"{flag} {val}")

    config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(flags),
        "output": output_path,
        "parameters": params,
    }

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config  → {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct image contours as circle arcs and render a transparent PNG."
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("output_dir", help="Directory where a timestamped run folder will be created")
    parser.add_argument("--width", type=int, default=1920, help="Output canvas width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Output canvas height (default: 1080)")
    parser.add_argument(
        "--threshold", type=float, default=199.0,
        help="Marching-squares contour level (default: 199)"
    )
    parser.add_argument(
        "--step", type=int, default=5,
        help="Subsample contour every N points (default: 5)"
    )
    parser.add_argument(
        "--contour-leveling-scale", type=float, default=0.35,
        help="Contour leveling down/up scale in (0,1]; lower smooths more (default: 0.35)"
    )
    parser.add_argument(
        "--contour-turn-preserve-threshold", type=float, default=0.75,
        help="Preserve strong turns above this local raw angle (radians, default: 0.75)"
    )
    parser.add_argument(
        "--contour-turn-preserve-radius", type=int, default=1,
        help="Preserve this many neighbours around each strong turn (default: 1)"
    )
    parser.add_argument(
        "--min-points", type=int, default=40,
        help="Skip contours shorter than this (default: 40)"
    )
    parser.add_argument(
        "--min-circles", type=int, default=1,
        help="Skip contours with fewer fitted circles (default: 1)"
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=1.0,
        help="Minimum accepted refit quality ratio before starting a new arc segment (default: 1.0)"
    )
    parser.add_argument(
        "--backward-k", type=int, default=10,
        help="After forward pass stops, try absorbing up to K points backwards (default: 10)"
    )
    parser.add_argument(
        "--forward-k", type=int, default=10,
        help="Try accepting up to K new points per forward refit, shrinking to 0 on failure (default: 10)"
    )
    parser.add_argument(
        "--sharp-turn-threshold", type=float, default=float("inf"),
        help="Reject candidate point batches containing local turns above this radian threshold (default: inf)"
    )
    parser.add_argument(
        "--sharp-turn-min-count", type=int, default=1,
        help="Reject a candidate batch only if at least this many points are sharp (default: 1)"
    )
    parser.add_argument(
        "--sharp-turn-method", choices=["circle", "poly"], default="circle",
        help="Sharp-turn detector to use (default: circle)"
    )
    parser.add_argument(
        "--poly-window-radius", type=int, default=7,
        help="Half-window size for local polynomial turn detector (default: 7)"
    )
    parser.add_argument(
        "--poly-high-degree", type=int, default=5,
        help="High polynomial degree used against quadratic baseline (default: 5)"
    )
    parser.add_argument(
        "--poly-improvement-ratio", type=float, default=2.0,
        help="Reject when high-degree polynomial MSE beats quadratic by this ratio (default: 2.0)"
    )
    parser.add_argument(
        "--max-circle-radius", type=float, default=500.0,
        help="Maximum allowed fitted circle radius; larger circles are rejected (default: 500)"
    )
    parser.add_argument(
        "--no-cv2", action="store_true",
        help="Use plain skimage contours (no cv2 preprocessing)"
    )
    parser.add_argument(
        "--no-center", action="store_true",
        help="Render in source-image coordinates instead of fitting and centering on the canvas"
    )
    parser.add_argument(
        "--padding", type=float, default=0.08,
        help="Canvas padding fraction when centering output (default: 0.08)"
    )
    parser.add_argument(
        "--background", default=DEFAULT_BACKGROUND,
        help=f"Background image composited under the final result (default: {DEFAULT_BACKGROUND})"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save step-by-step debug plots to a 'debug/' dir inside the run folder"
    )

    args = parser.parse_args()

    run_dir, output_path = _make_run_dir(args.output_dir, args.image)
    print(f"Run dir → {run_dir}")

    debug_dir = os.path.join(run_dir, "debug") if args.debug else None

    from circle_draw.pipeline import run

    run(
        image_path=args.image,
        output_path=output_path,
        image_shape=(args.width, args.height, 4),
        threshold=args.threshold,
        contour_step=args.step,
        contour_leveling_scale=args.contour_leveling_scale,
        contour_turn_preserve_threshold=args.contour_turn_preserve_threshold,
        contour_turn_preserve_radius=args.contour_turn_preserve_radius,
        min_contour_points=args.min_points,
        min_circles=args.min_circles,
        quality_threshold=args.quality_threshold,
        forward_k=args.forward_k,
        backward_k=args.backward_k,
        sharp_turn_threshold=args.sharp_turn_threshold,
        sharp_turn_min_count=args.sharp_turn_min_count,
        sharp_turn_method=args.sharp_turn_method,
        poly_window_radius=args.poly_window_radius,
        poly_high_degree=args.poly_high_degree,
        poly_improvement_ratio=args.poly_improvement_ratio,
        max_circle_radius=args.max_circle_radius,
        use_cv2_preprocessing=not args.no_cv2,
        center_on_canvas=not args.no_center,
        output_padding_fraction=args.padding,
        background_path=args.background or None,
        debug_dir=debug_dir,
    )
    _save_config(args, run_dir, output_path)


if __name__ == "__main__":
    main()
