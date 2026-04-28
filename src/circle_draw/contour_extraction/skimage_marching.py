"""Contour extraction via skimage marching squares — works well on clean binary images."""

import numpy as np
import imageio.v3 as iio
from skimage import measure
from skimage.color import rgb2gray


def extract(image: np.ndarray, threshold: float = 0, step: int = 5) -> list[np.ndarray]:
    """Return contours as list of (N, 2) arrays in (x, y) order, subsampled by step."""
    contours = measure.find_contours(image, threshold, fully_connected="high", positive_orientation="low")
    result = []
    for c in contours:
        c = c.copy()
        # skimage returns (row, col) = (y, x); swap to (x, y)
        c[:, 0], c[:, 1] = c[:, 1].copy(), c[:, 0].copy()
        result.append(c[1::step])
    return result


def load_and_extract(image_path: str, threshold: float = 0, step: int = 5) -> list[np.ndarray]:
    img = iio.imread(image_path)
    gray = rgb2gray(img[:, :, :3]).astype(np.uint8)
    return extract(gray, threshold, step)
