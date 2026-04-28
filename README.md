# circle_draw

Reconstruct image contours as arcs of circles, where each arc is delimited by the intersection of adjacent circles.

![example output](desired_image.jpg)

## How it works

1. **Contour extraction** — extract pixel-level contours from the input image
2. **Arc segmentation** — greedily fit circles to consecutive contour points, splitting when fit quality drops
3. **Intersection chaining** — find where adjacent circle arcs intersect to define exact arc boundaries
4. **Rendering** — draw each arc fully opaque at the boundary, faint elsewhere, on a transparent canvas

## Project structure

```
src/circle_draw/
├── contour_extraction/
│   ├── skimage_marching.py   # pure skimage — for clean binary images
│   └── cv2_enhanced.py       # cv2 preprocessing + skimage — for photos
├── point_selection/
│   └── fit_quality.py        # greedy arc segmentation by fit quality
├── circle_fitting/
│   └── least_squares.py      # Gauss-Newton least-squares circle fit
├── circle_geometry/
│   └── intersections.py      # circle-circle / line-circle geometry
├── rendering/
│   └── cairo_arcs.py         # Cairo arc renderer with transparency
└── pipeline.py               # orchestrates the full run
scripts/
└── run.py                    # CLI entry point
notebooks/                    # original exploration notebooks
```

## Setup

Requires system library `libcairo2-dev` (for pycairo).

```bash
# Ubuntu / Debian
sudo apt-get install libcairo2-dev pkg-config

# then install Python deps
uv sync
```

## Usage

```bash
uv run python scripts/run.py path/to/image.jpg outputs/result.png
```

Options:

| flag | default | description |
|------|---------|-------------|
| `--width` | 1600 | output canvas width |
| `--height` | 2100 | output canvas height |
| `--threshold` | 199 | marching-squares contour level |
| `--step` | 5 | subsample contour every N points |
| `--min-points` | 40 | skip short contours |
| `--min-circles` | 3 | skip contours with few arcs |
| `--quality-threshold` | 0.8 | fit-quality ratio to split arc |
| `--no-cv2` | — | skip cv2 preprocessing |
