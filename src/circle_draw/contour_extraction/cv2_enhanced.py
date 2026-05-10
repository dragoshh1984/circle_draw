"""Contour extraction with cv2 preprocessing — better for noisy or photographic images."""

import numpy as np
import cv2 as cv
from skimage import measure


def preprocess(
    image: np.ndarray,
    equalize_hist: bool = True,
    denoise_h: float = 30.0,
    denoise_template_window: int = 3,
    denoise_search_window: int = 21,
) -> np.ndarray:
    """Convert to grayscale, optional equalisation, and configurable denoising."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if equalize_hist:
        gray = cv.equalizeHist(gray)
    if denoise_h > 0:
        gray = cv.fastNlMeansDenoising(
            gray,
            None,
            float(denoise_h),
            int(denoise_template_window),
            int(denoise_search_window),
        )
    return gray


def extract(image: np.ndarray, threshold: float = 199, step: int = 5) -> list[np.ndarray]:
    """Return contours as list of (N, 2) arrays in (x, y) order, subsampled by step."""
    contours = measure.find_contours(image, threshold, fully_connected="high", positive_orientation="low")
    result = []
    for c in contours:
        c = c.copy()
        # skimage returns (row, col) = (y, x); swap to (x, y)
        c[:, 0], c[:, 1] = c[:, 1].copy(), c[:, 0].copy()
        result.append(c[1::step])
    return result


def load_and_extract(
    image_path: str,
    threshold: float = 199,
    step: int = 5,
    equalize_hist: bool = True,
    denoise_h: float = 30.0,
    denoise_template_window: int = 3,
    denoise_search_window: int = 21,
) -> list[np.ndarray]:
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    preprocessed = preprocess(
        img,
        equalize_hist=equalize_hist,
        denoise_h=denoise_h,
        denoise_template_window=denoise_template_window,
        denoise_search_window=denoise_search_window,
    )
    return extract(preprocessed, threshold, step)
