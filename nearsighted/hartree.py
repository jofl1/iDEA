"""Hartree potential helpers."""
from __future__ import annotations

import numpy as np

def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system using the Thomas algorithm."""

    n = diag.size
    c_prime = np.zeros(n, dtype=float)
    d_prime = np.zeros(n, dtype=float)
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom
    solution = np.zeros_like(rhs)
    solution[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    return solution


def v_H_from_density(x: np.ndarray, n: np.ndarray, lambda_soft: float) -> np.ndarray:
    """Solve the 1D Poisson equation for the Hartree potential."""

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
    npts = x.size
    lower = np.zeros(npts, dtype=float)
    upper = np.zeros(npts, dtype=float)
    diag = np.full(npts, -2.0 / dx**2, dtype=float)
    lower[1:] = 1.0 / dx**2
    upper[:-1] = 1.0 / dx**2
    lower[-1] = 0.0
    upper[0] = 0.0
    diag[0] = diag[-1] = 1.0
    rhs = -4.0 * np.pi * n.astype(float)
    rhs[0] = rhs[-1] = 0.0
    solution = _solve_tridiagonal(lower, diag, upper, rhs)
    return solution
