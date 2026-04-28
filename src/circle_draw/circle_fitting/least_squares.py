import numpy as np
from numpy.linalg import pinv
from typing import Sequence

Point = tuple[float, float]
Circle = np.ndarray  # shape (3,): [cx, cy, r]


def fit(initial: Circle, points: Sequence[Point], iterations: int = 20) -> Circle:
    """Gauss-Newton least-squares circle fitting.

    Reference: Gander, Golub, Strebel — Least-Squares Fitting of Circles and Ellipses.
    http://fourier.eng.hmc.edu/e176/lectures/NM/node36.html
    """
    x = np.array(initial, dtype=float)
    for _ in range(iterations):
        J = _jacobian(x, points)
        r = np.array([_residual(x, p) for p in points])
        JtJ = np.nan_to_num(J.T @ J)
        x = x - pinv(JtJ) @ J.T @ r
    return x


def init_from_endpoints(p_first: Point, p_last: Point) -> Circle:
    """Midpoint + half-chord initialisation for two boundary points."""
    cx = min(p_first[0], p_last[0]) + abs(p_first[0] - p_last[0]) / 2
    cy = min(p_first[1], p_last[1]) + abs(p_first[1] - p_last[1]) / 2
    r = np.sqrt((p_first[0] - p_last[0]) ** 2 + (p_first[1] - p_last[1]) ** 2) / 2
    return np.array([cx, cy, r])


def _residual(circle: Circle, point: Point) -> float:
    cx, cy, r = circle
    return float(np.sqrt((cx - point[0]) ** 2 + (cy - point[1]) ** 2) - r)


def _jacobian(circle: Circle, points: Sequence[Point]) -> np.ndarray:
    cx, cy, _ = circle
    n = len(points)
    J = np.ones((n, 3))
    for i, p in enumerate(points):
        d = np.sqrt((cx - p[0]) ** 2 + (cy - p[1]) ** 2)
        if d > 0:
            J[i, 0] = (cx - p[0]) / d
            J[i, 1] = (cy - p[1]) / d
        else:
            J[i, 0] = 0.0
            J[i, 1] = 0.0
        J[i, 2] = -1.0
    return J
