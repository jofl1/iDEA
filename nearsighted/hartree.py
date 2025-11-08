"""Hartree potential helpers."""
from __future__ import annotations

import numpy as np

from .grids import central_difference


def v_H_from_density(x: np.ndarray, n: np.ndarray, lambda_soft: float) -> np.ndarray:
    """Compute softened 1D Hartree potential via direct summation."""

    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    if x.shape != n.shape:
        raise ValueError("x and n must have the same shape")
    if x.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")
    if lambda_soft <= 0:
        raise ValueError("lambda_soft must be positive")
    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx):
        raise ValueError("Grid must be uniform for Hartree evaluation")
    diff = x[:, None] - x[None, :]
    kernel = 1.0 / np.sqrt(diff**2 + lambda_soft**2)
    return dx * kernel @ n
