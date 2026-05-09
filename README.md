# circle_draw

Reconstruct image contours as fitted circle arcs, then render the selected arc
segments over an optional background.

## Example

The checked-in demo run lives in `outputs/demo_kakapo_logo/` and includes its
input image, result, config, and debug fitting videos. The current VS Code
launch profile uses the same demo input with these settings:

```bash
uv run python scripts/run.py input/kakapo_logo.png outputs/ \
	--width 1920 --height 1080 --threshold 199.0 --step 5 \
	--contour-leveling-scale 0.35 --contour-turn-preserve-threshold 0.75 \
	--contour-turn-preserve-radius 1 --min-points 40 --min-circles 1 \
	--quality-threshold 0.8 --forward-k 10 --backward-k 100 --padding 0.08 \
	--sharp-turn-method poly --poly-window-radius 10 --poly-high-degree 8 \
	--poly-improvement-ratio 2.5 --max-circle-radius 500 \
	--sharp-turn-min-count 2 --debug
```

**Input**

![kakapo logo input](outputs/demo_kakapo_logo/input.png)

**Goal**

![desired circle drawing goal](desired_image.jpg)

Contours are leveled after extraction using a downsample/upscale pass. Tune with
`--contour-leveling-scale`, `--contour-turn-preserve-threshold`, and
`--contour-turn-preserve-radius` to reduce noise while keeping real corners.

**Output**

![kakapo logo result](outputs/demo_kakapo_logo/result.png)

**Debug fitting preview**

![circle fitting debug preview](outputs/demo_kakapo_logo/debug/fitting_fast.gif)

Full debug videos:

- [Fast MP4](outputs/demo_kakapo_logo/debug/fitting_fast.mp4)
- [Slow MP4](outputs/demo_kakapo_logo/debug/fitting_slow.mp4)
- [Run config](outputs/demo_kakapo_logo/config.json)

## Setup

Requires `libcairo2-dev` on the system (for pycairo).

```bash
sudo apt-get install libcairo2-dev pkg-config
uv sync
```

## Usage

```bash
uv run python scripts/run.py path/to/image.png outputs/
```

The command creates a timestamped run directory under `outputs/`. The default
canvas is 1920×1080. Kept contours are fitted and centered on the canvas
automatically; use `--no-center` to keep source-image coordinates.

Use `--sharp-turn-threshold` (radians) to reject forward/backward candidate
point batches containing local contour kinks above the threshold.

Use `--sharp-turn-min-count` to make rejection less sensitive: a K batch is
rejected only when at least that many points in the batch are sharp.

Use `--sharp-turn-method poly` to switch to a local polynomial-overfit turn
detector (after local PCA rotation). Tune with `--poly-window-radius`,
`--poly-high-degree`, and `--poly-improvement-ratio`.

Use `--max-circle-radius` to cap fitted circle size (default: 500).


Add `--debug` to write step-by-step fitting videos into the run's `debug/`
directory.
