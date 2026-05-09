"""CLI entry point for the circle-draw pipeline."""

import argparse
import json
import os
from datetime import datetime


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
        "min_points": 40, "min_circles": 3, "quality_threshold": 0.8,
        "backward_k": 10,
        "no_cv2": False, "no_center": False, "padding": 0.08, "debug": False,
    }
    flag_map = {
        "width": "--width", "height": "--height", "threshold": "--threshold",
        "step": "--step", "min_points": "--min-points", "min_circles": "--min-circles",
        "quality_threshold": "--quality-threshold", "backward_k": "--backward-k",
        "no_cv2": "--no-cv2", "no_center": "--no-center",
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
        "--min-points", type=int, default=40,
        help="Skip contours shorter than this (default: 40)"
    )
    parser.add_argument(
        "--min-circles", type=int, default=3,
        help="Skip contours with fewer fitted circles (default: 3)"
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=0.8,
        help="Fit-quality ratio to trigger new arc segment (default: 0.8)"
    )
    parser.add_argument(
        "--backward-k", type=int, default=10,
        help="After forward pass stops, try absorbing up to K points backwards (default: 10)"
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
        min_contour_points=args.min_points,
        min_circles=args.min_circles,
        quality_threshold=args.quality_threshold,
        backward_k=args.backward_k,
        use_cv2_preprocessing=not args.no_cv2,
        center_on_canvas=not args.no_center,
        output_padding_fraction=args.padding,
        debug_dir=debug_dir,
    )
    _save_config(args, run_dir, output_path)


if __name__ == "__main__":
    main()
