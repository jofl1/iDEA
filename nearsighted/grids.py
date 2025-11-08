"""Grid utilities for the nearsightedness metric.

All distances are in Bohr.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Uniform one-dimensional grid."""

    x: np.ndarray
    dx: float


def uniform_grid(xmin: float, xmax: float, num: int) -> Grid:
    """Return a uniform grid.

    Parameters
    ----------
    xmin, xmax
        Domain boundaries in Bohr.
    num
        Number of grid points (>=2).
    """

    if num < 2:
        raise ValueError("Grid requires at least two points.")
    x = np.linspace(xmin, xmax, num, dtype=float)
    dx = float(x[1] - x[0])
    return Grid(x=x, dx=dx)


def central_difference(values: np.ndarray, dx: float) -> np.ndarray:
    """Compute first derivative using central differences.

    Uses second-order accurate central differences for interior points and
    second-order forward/backward differences at the boundaries.
    """

    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("central_difference expects a 1D array")
    if values.size < 3:
        raise ValueError("central_difference requires at least 3 points")
    derivative = np.empty_like(values)
    derivative[1:-1] = (values[2:] - values[:-2]) / (2.0 * dx)
    derivative[0] = (-3 * values[0] + 4 * values[1] - values[2]) / (2.0 * dx)
    derivative[-1] = (3 * values[-1] - 4 * values[-2] + values[-3]) / (2.0 * dx)
    return derivative


def strip_boundary_mask(x: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Return boolean mask excluding a fraction of the grid at each end."""

    if not 0.0 <= fraction < 0.5:
        raise ValueError("fraction must be in [0, 0.5)")
    x = np.asarray(x)
    n = x.size
    strip = int(np.floor(fraction * n))
    mask = np.ones(n, dtype=bool)
    if strip > 0:
        mask[:strip] = False
        mask[-strip:] = False
    return mask


def apply_mask(*arrays: Iterable[np.ndarray], mask: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Apply mask to arrays, returning tuple of masked copies."""

    masked = []
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.shape != mask.shape:
            raise ValueError("Array and mask must share shape")
        masked.append(arr[mask])
    return tuple(masked)


def remove_gauge(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Remove additive constant by subtracting weighted mean."""

    values = np.asarray(values, dtype=float)
    if weights is None:
        mean = values.mean()
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != values.shape:
            raise ValueError("weights must match values shape")
        total = float(np.sum(weights))
        if total == 0:
            mean = values.mean()
        else:
            mean = float(np.sum(weights * values) / total)
    return values - mean
