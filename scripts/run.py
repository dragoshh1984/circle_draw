"""CLI entry point for the circle-draw pipeline."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct image contours as circle arcs and render a transparent PNG."
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("output", help="Path for output PNG")
    parser.add_argument("--width", type=int, default=1600, help="Output canvas width (default: 1600)")
    parser.add_argument("--height", type=int, default=2100, help="Output canvas height (default: 2100)")
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
        "--no-cv2", action="store_true",
        help="Use plain skimage contours (no cv2 preprocessing)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save step-by-step debug plots to a 'debug/' dir next to the output file"
    )

    args = parser.parse_args()

    import os
    debug_dir = None
    if args.debug:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        debug_dir = os.path.join(output_dir, "debug")

    from circle_draw.pipeline import run

    run(
        image_path=args.image,
        output_path=args.output,
        image_shape=(args.width, args.height, 4),
        threshold=args.threshold,
        contour_step=args.step,
        min_contour_points=args.min_points,
        min_circles=args.min_circles,
        quality_threshold=args.quality_threshold,
        use_cv2_preprocessing=not args.no_cv2,
        debug_dir=debug_dir,
    )


if __name__ == "__main__":
    main()
