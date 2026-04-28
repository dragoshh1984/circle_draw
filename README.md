# circle_draw

Reconstruct image contours as arcs of circles, where each arc is delimited by the intersection of adjacent fitted circles.

## Example

**Input**

![kakapo logo input](input/kakapo_logo.png)

**Output**

![kakapo logo result](outputs/logo_result.png)

## Setup

Requires `libcairo2-dev` on the system (for pycairo).

```bash
sudo apt-get install libcairo2-dev pkg-config
uv sync
```

## Usage

```bash
uv run python scripts/run.py path/to/image.jpg outputs/result.png
```

The default canvas is 1920×1080. Kept contours are fitted and centered on the
canvas automatically; use `--no-center` to keep source-image coordinates.

Add `--debug` to write step-by-step plots to `outputs/debug/`.
